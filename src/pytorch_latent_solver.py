from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import logging as log
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from tqdm import tqdm

from config import Config
from gating_functions import make_gating_function
from neighbor_graph import NeighbourGraph

GatingFnName = Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"]


@dataclass
class ActiveSetMap:
    global_ids: np.ndarray


class TorchLatentSolver:
    """
    PyTorch latent solver with active-subgraph updates.

    Keeps the same objective used in NumPy code, but solves only on a local active
    set A (batch bins + graph neighbors) with strict block updates.
    """

    def __init__(
        self,
        cfg: Config,
        neighbour_graph: NeighbourGraph,
        embed_dim: int = 1,
        gating_fn: GatingFnName = "sigmoid",
    ) -> None:
        self.cfg = cfg
        self.ng = neighbour_graph
        self.n_bins = neighbour_graph.n_bins
        self.embed_dim = embed_dim

        if embed_dim > 1:
            self.gating = make_gating_function(
                gating_fn,
                alpha=cfg.gating_alpha,
                kappa=cfg.gating_kappa,
                epsilon=cfg.gating_epsilon,
            )

        latent_device = getattr(self.cfg, "latent_device", None)
        if latent_device is not None:
            self.device = torch.device(latent_device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # Sparse CSR matmul support is not robust on all MPS paths; keep CPU fallback.
            if str(self.cfg.device).lower() == "mps":
                log.warning("cfg.device is mps, but TorchLatentSolver uses CPU fallback for sparse CSR stability")
            self.device = torch.device("cpu")

        self.H_smooth: Optional[sparse.csr_matrix] = None
        self.I_minus_H_smooth: Optional[sparse.csr_matrix] = None
        self._I_minus_H_smooth_csc: Optional[sparse.csc_matrix] = None
        self._graph_neighbors: Optional[List[np.ndarray]] = None
        self._latent_param: Optional[torch.nn.Parameter] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._state_initialized: bool = False

    def build_V_and_H(
        self,
        X: pd.DataFrame,
        bin_index: Dict[Any, int],
        method: str = "nw",
    ) -> None:
        """
        Torch backend setup hook.

        The historical API name is kept for compatibility with Trainer, but this
        path builds graph smoothness matrices only (not observation matrix V).
        """
        del X  # not needed for local GPU solver
        del bin_index  # not needed for local GPU solver
        rows_H: List[int] = []
        cols_H: List[int] = []
        vals_H: List[float] = []

        for b in tqdm(range(self.n_bins), desc="Building H matrix", unit="bin", leave=False):
            if method == "nw":
                neigh, w = self.ng.nw_weights_for_node(b)
            else:
                neigh, w = self.ng.llr_coeffs_for_node(b)

            if len(neigh) == 0:
                rows_H.append(b)
                cols_H.append(b)
                vals_H.append(1.0)
                continue

            for j, wj in zip(neigh, w):
                rows_H.append(int(b))
                cols_H.append(int(j))
                vals_H.append(float(wj))

        H_smooth = sparse.csr_matrix(
            (
                np.asarray(vals_H, dtype=np.float64),
                (np.asarray(rows_H, dtype=np.int32), np.asarray(cols_H, dtype=np.int32)),
            ),
            shape=(self.n_bins, self.n_bins),
        )
        self.H_smooth = H_smooth
        self.I_minus_H_smooth = sparse.csr_matrix(sparse.identity(self.n_bins, format="csr") - H_smooth)
        self._I_minus_H_smooth_csc = sparse.csc_matrix(self.I_minus_H_smooth)
        self._graph_neighbors = self._build_graph_active_neighbors()
        self._reset_solver_state()

        log.info(
            "Torch latent solver: H_smooth=%s, embed_dim=%d, device=%s",
            H_smooth.shape,
            self.embed_dim,
            self.device,
        )

    def _reset_solver_state(self) -> None:
        self._latent_param = None
        self._optimizer = None
        self._state_initialized = False

    def reset_state(self) -> None:
        """Public state reset hook for new runs or explicit reinitialization."""
        self._reset_solver_state()

    def is_state_initialized(self) -> bool:
        return self._state_initialized and self._latent_param is not None and self._optimizer is not None

    def export_state(self) -> Dict[str, Any]:
        if not self.is_state_initialized() or self._latent_param is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "latent": self._latent_param.detach().cpu().numpy(),
            "optimizer": self._optimizer.state_dict() if self._optimizer is not None else None,
        }

    def load_state(self, latent: np.ndarray, optimizer_state: Optional[Dict[str, Any]] = None) -> None:
        latent_shape = (self.n_bins, self.embed_dim)
        latent_np = np.asarray(latent, dtype=np.float32).reshape(latent_shape)
        self._reset_solver_state()
        self._ensure_solver_state(latent_np)
        if self._optimizer is not None and optimizer_state is not None:
            self._optimizer.load_state_dict(optimizer_state)

    def _ensure_solver_state(self, x0: np.ndarray) -> None:
        if self.is_state_initialized():
            return

        latent_init = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        self._latent_param = torch.nn.Parameter(latent_init.clone())
        self._optimizer = torch.optim.Adam([self._latent_param], lr=float(self.cfg.latent_adam_lr))
        self._state_initialized = True

    def _csr_to_torch(self, mat: sparse.csr_matrix) -> torch.Tensor:
        crow = torch.as_tensor(mat.indptr.astype(np.int64), device=self.device)
        col = torch.as_tensor(mat.indices.astype(np.int64), device=self.device)
        vals = torch.as_tensor(mat.data.astype(np.float32), device=self.device)
        return torch.sparse_csr_tensor(
            crow,
            col,
            vals,
            size=mat.shape,
            dtype=torch.float32,
            device=self.device,
        )

    def _build_row_set(self, active_ids: np.ndarray) -> np.ndarray:
        """
        Build row closure R for smoothness so rows outside active that depend on
        active columns are included in the local quadratic.
        """
        if self._I_minus_H_smooth_csc is None:
            raise RuntimeError("I_minus_H_smooth CSC is not initialized")

        csc = cast(sparse.csc_matrix, self._I_minus_H_smooth_csc)
        dep_rows = csc[:, active_ids].indices.astype(np.int64, copy=False)
        row_ids = np.unique(np.concatenate([active_ids.astype(np.int64, copy=False), dep_rows]))
        return row_ids

    def _mask_optimizer_state(self, active_mask: torch.Tensor) -> None:
        if self._optimizer is None or self._latent_param is None:
            return
        state = self._optimizer.state.get(self._latent_param)
        if not state:
            return

        inactive_mask = ~active_mask
        for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            t = state.get(k)
            if isinstance(t, torch.Tensor):
                t[inactive_mask] = 0.0

    def solve(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        final_weights: Optional[np.ndarray] = None,
        bin_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        loss_type: Literal["cross_entropy", "logistic"] = "cross_entropy",
        x0: Optional[np.ndarray] = None,
        prox_weight: float = 0.0,
        x_anchor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.I_minus_H_smooth is None:
            raise RuntimeError("Matrices not built; call build_V_and_H first")
        if bin_ids is None:
            raise ValueError("bin_ids are required for torch latent solving")

        if self.embed_dim > 1 and final_weights is None:
            raise ValueError("final_weights are required for embed_dim > 1")

        latent_shape = (self.n_bins, self.embed_dim)
        if x0 is None:
            if not self._state_initialized:
                raise ValueError("x0 must be provided on the first solve call to initialize latent state")
            x0_full = (
                self._latent_param.detach().cpu().numpy()
                if self._latent_param is not None
                else np.zeros(latent_shape, dtype=np.float32)
            )
        else:
            x0_full = np.asarray(x0, dtype=np.float32).reshape(latent_shape)

            if self._state_initialized and self._latent_param is not None:
                current = self._latent_param.detach().cpu().numpy()
                if not np.allclose(current, x0_full, rtol=1e-5, atol=1e-7):
                    log.warning("Received x0 different from internal solver state; reinitializing latent optimizer state")
                    self._reset_solver_state()

        self._ensure_solver_state(x0_full)
        if self._latent_param is None or self._optimizer is None:
            raise RuntimeError("Solver state was not initialized")

        anchor_full = x0_full if x_anchor is None else np.asarray(x_anchor, dtype=np.float32).reshape(latent_shape)
        batch_bin_ids = np.asarray(bin_ids, dtype=np.int64).reshape(-1)
        active_map = self._build_active_set(batch_bin_ids)

        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1), device=self.device)
        intrinsic_t = torch.as_tensor(np.asarray(intrinsic_vec, dtype=np.float32), device=self.device)
        if intrinsic_t.ndim == 1:
            intrinsic_t = intrinsic_t.unsqueeze(-1)
        if self.embed_dim == 1 and intrinsic_t.shape[1] != 1:
            intrinsic_t = intrinsic_t.reshape(-1, 1)
        if intrinsic_t.shape[1] != self.embed_dim:
            raise ValueError(
                f"intrinsic_vec has shape {tuple(intrinsic_t.shape)}, expected (*, {self.embed_dim})"
            )

        sample_ids_t = None
        if sample_ids is not None:
            sample_ids_t = torch.as_tensor(np.asarray(sample_ids, dtype=np.int64).reshape(-1), dtype=torch.long, device=self.device)

        w_t: Optional[torch.Tensor] = None
        if self.embed_dim > 1:
            w_t = torch.as_tensor(np.asarray(final_weights, dtype=np.float32).reshape(-1), device=self.device)
            if w_t.shape[0] != self.embed_dim:
                raise ValueError(f"final_weights has shape {tuple(w_t.shape)}, expected ({self.embed_dim},)")

        H_anchor = anchor_full[active_map.global_ids]
        H_anchor_t = torch.as_tensor(H_anchor, dtype=torch.float32, device=self.device)

        row_ids = self._build_row_set(active_map.global_ids)
        L_rows, L_RA = self._build_row_and_active_operators(row_ids, active_map.global_ids)
        active_ids_t = torch.as_tensor(active_map.global_ids, dtype=torch.long, device=self.device)

        local_bin_ids = np.searchsorted(active_map.global_ids, batch_bin_ids)
        if not np.array_equal(active_map.global_ids[local_bin_ids], batch_bin_ids):
            raise RuntimeError("Active-set searchsorted mapping failed for one or more bin ids")
        local_bin_ids_t = torch.as_tensor(local_bin_ids, dtype=torch.long, device=self.device)

        steps = int(self.cfg.latent_adam_steps)
        if steps < 1:
            steps = 1

        # Only active-bin gradients are allowed; outside bins stay frozen for this block update.
        active_mask = torch.zeros((self.n_bins,), dtype=torch.bool, device=self.device)
        active_mask[active_ids_t] = True

        for _ in range(steps):
            self._optimizer.zero_grad(set_to_none=True)
            h_frozen_full = self._latent_param.detach()
            h_active = self._latent_param[active_ids_t]
            h_active_frozen = h_frozen_full[active_ids_t]
            delta_active = h_active - h_active_frozen

            logits = self._logits_from_latent(h_active, intrinsic_t, local_bin_ids_t, final_weights_t=w_t)

            if loss_type == "cross_entropy":
                if sample_ids_t is None:
                    raise ValueError("sample_ids are required for cross_entropy latent solving")
                data_loss, scale = self._cross_entropy_loss(y_t, logits, sample_ids_t)
            elif loss_type == "logistic":
                data_loss, scale = self._logistic_loss(y_t, logits)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            loss = data_loss

            # Regularization terms - L2 on H and smoothness via L_A
            l2_coef = float(self.cfg.latent_l2_reg)
            if l2_coef > 0:
                loss = loss + 0.5 * l2_coef * torch.sum(h_active * h_active)

            # Smoothness with row closure and frozen-boundary contribution.
            smooth_coef = float(self.cfg.latent_smooth_reg)
            if smooth_coef > 0:
                base = torch.sparse.mm(L_rows, h_frozen_full)
                corr = torch.sparse.mm(L_RA, delta_active)
                diff = base + corr
                loss = loss + 0.5 * smooth_coef * torch.sum(diff * diff)

            if prox_weight > 0:
                d_prox = h_active - H_anchor_t
                loss = loss + 0.5 * float(prox_weight) * torch.sum(d_prox * d_prox)

            loss = loss / scale
            loss.backward()
            if self._latent_param.grad is None:
                raise RuntimeError("Latent gradient is missing")
            self._latent_param.grad[~active_mask] = 0.0

            frozen_before_step = self._latent_param.detach().clone()
            self._optimizer.step()

            # Strict block update: restore inactive bins and clear their optimizer moments.
            with torch.no_grad():
                self._latent_param.data[~active_mask] = frozen_before_step[~active_mask]
            self._mask_optimizer_state(active_mask)

        out = self._latent_param.detach().cpu().numpy()
        if self.embed_dim == 1:
            return out.reshape(-1)
        return out

    def _build_row_and_active_operators(
        self,
        row_ids: np.ndarray,
        active_ids: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.I_minus_H_smooth is None:
            raise RuntimeError("I_minus_H_smooth is not initialized")
        L_rows_csr = self.I_minus_H_smooth[row_ids, :].tocsr()
        L_RA_csr = L_rows_csr[:, active_ids].tocsr()
        return self._csr_to_torch(L_rows_csr), self._csr_to_torch(L_RA_csr)

    def _build_active_set(self, batch_bin_ids: np.ndarray) -> ActiveSetMap:
        if self._graph_neighbors is None:
            raise RuntimeError("Graph neighbors are not initialized")

        mode = str(getattr(self.cfg, "latent_k_hop_mode", "threshold"))
        max_hops = max(1, int(getattr(self.cfg, "latent_k_hop_threshold", 1)))
        knn_cap = max(1, int(getattr(self.cfg, "latent_k_hop_knn", 64)))

        active = set(int(b) for b in batch_bin_ids.tolist())
        frontier = set(active)
        hops = 1 if mode == "knn" else max_hops

        for _ in range(hops):
            next_frontier: set[int] = set()
            for b in frontier:
                neigh_arr = self._graph_neighbors[int(b)]
                if mode == "knn":
                    neigh_arr = neigh_arr[:knn_cap]
                for n in neigh_arr:
                    n_int = int(n)
                    if n_int not in active:
                        active.add(n_int)
                        next_frontier.add(n_int)
            frontier = next_frontier
            if not frontier:
                break

        active_ids = np.fromiter(active, dtype=np.int64)
        active_ids.sort()
        return ActiveSetMap(global_ids=active_ids)

    def _build_graph_active_neighbors(self) -> List[np.ndarray]:
        """
        Build local expansion neighborhoods from the precomputed neighbor graph.

        This uses the same graph source that the model already relies on for smoothness,
        which may be embedding-based in the current setup. We keep the expansion simple:
        the active set is the batch bins plus their direct graph neighbors.
        """
        neighbors: List[np.ndarray] = []
        for i in range(self.n_bins):
            neigh = np.asarray(self.ng.neighbours[i], dtype=np.int64)
            neighbors.append(neigh)

        log.info("Active graph neighbors initialized from NeighbourGraph.neighbours")
        return neighbors

    def _logits_from_latent(
        self,
        H_active: torch.Tensor,
        intrinsic_t: torch.Tensor,
        local_bin_ids_t: torch.Tensor,
        final_weights_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_obs = H_active[local_bin_ids_t]
        if self.embed_dim == 1:
            return intrinsic_t.squeeze(-1) + h_obs.squeeze(-1)

        if final_weights_t is None:
            raise ValueError("final_weights_t is required for embed_dim > 1")

        gated = self.gating.gate_torch(h_obs)
        m_tilde = intrinsic_t * gated
        return torch.sum(m_tilde * final_weights_t.unsqueeze(0), dim=1)

    def _cross_entropy_loss(
        self,
        y_t: torch.Tensor,
        logits: torch.Tensor,
        sample_ids_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_s, inv = torch.unique(sample_ids_t, sorted=False, return_inverse=True)
        n_samples = max(1, int(unique_s.numel()))

        max_per = torch.full((n_samples,), -torch.inf, dtype=logits.dtype, device=logits.device)
        max_per.scatter_reduce_(0, inv, logits, reduce="amax", include_self=True)

        logits_shift = logits - max_per[inv]
        exp_logits = torch.exp(logits_shift)
        denom = torch.zeros((n_samples,), dtype=logits.dtype, device=logits.device)
        denom.index_add_(0, inv, exp_logits)
        denom = denom + 1e-12

        logsumexp = torch.log(denom) + max_per
        ce_sum = -(y_t * (logits - logsumexp[inv])).sum()
        return ce_sum, torch.tensor(float(n_samples), dtype=logits.dtype, device=logits.device)

    def _logistic_loss(
        self,
        y_t: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bce_sum = F.binary_cross_entropy_with_logits(logits, y_t, reduction="sum")
        scale = torch.tensor(float(max(1, y_t.numel())), dtype=logits.dtype, device=logits.device)
        return bce_sum, scale
