from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, cast

import logging as log
import numpy as np
import pandas as pd
import time
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


class LatentSolver:
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

        self.device = torch.device(self.cfg.device) if str(self.cfg.device.lower()) != "mps" else torch.device("cpu") # Use CPU for latent solver if MPS due to sparse CSR stability issues, otherwise use configured device

        self.H_smooth: Optional[sparse.csr_matrix] = None               # Smoothness operator matrix built from neighbor graph; H_smooth @ h gives neighbor-interpolated latent
        self.I_minus_H_smooth: Optional[sparse.csr_matrix] = None       # Precompute I - H_smooth for efficient regularization; (I - H_smooth) @ h gives difference between latent and neighbor interpolation used for smoothness regularization
        self._I_minus_H_smooth_csc: Optional[sparse.csc_matrix] = None  # CSC format of I - H_smooth for efficient column slicing when building row closure
        self._graph_neighbors: Optional[List[np.ndarray]] = None        # List of neighbor arrays for each node, used to build active set on each solve call
        self.H_interp: Dict[bool, Optional[torch.Tensor]] = {False: None, True: None}
        self._H_interp_csr: Dict[bool, Optional[sparse.csr_matrix]] = {False: None, True: None}

    def _row_normalize_csr(self, mat: sparse.csr_matrix) -> sparse.csr_matrix:
        row_sums = np.asarray(mat.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
        row_sums[row_sums == 0.0] = 1.0
        inv = sparse.diags(1.0 / row_sums, format="csr")
        return (inv @ mat).tocsr()

    def _build_interpolation_operator(self, include_self_in_interpolation: bool) -> sparse.csr_matrix:
        if self.H_smooth is None:
            raise RuntimeError("H_smooth is not initialized")

        base = self.H_smooth.copy()
        if include_self_in_interpolation:
            base = base + sparse.identity(self.n_bins, format="csr")
        return self._row_normalize_csr(base.tocsr())

    def get_interpolation_operator(self, include_self_in_interpolation: bool, device: Optional[torch.device] = None) -> torch.Tensor:
        op = self.H_interp.get(include_self_in_interpolation)
        if op is None:
            raise RuntimeError("Interpolation operators are not built; call build_interpolation_matrix first")

        target_device = self.device if device is None else device
        if target_device.type == "mps":
            target_device = torch.device("cpu")
        if op.device == target_device:
            return op
        return op.to(target_device)

    def _compute_logits_from_latent_values(
        self,
        latent_obs: torch.Tensor,
        intrinsic_t: torch.Tensor,
        final_weights_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embed_dim == 1:
            return intrinsic_t.squeeze(-1) + latent_obs.squeeze(-1)

        if final_weights_t is None:
            raise ValueError("final_weights_t is required for embed_dim > 1")

        gated = self.gating.gate_torch(latent_obs)
        m_tilde = intrinsic_t * gated
        return torch.sum(m_tilde * final_weights_t.unsqueeze(0), dim=1)

    def build_interpolation_matrix(
        self,
        method: str = "nw",
    ) -> None:
        """
        Build the H_smooth matrix from the neighbor graph using either Nadaraya-Watson or LLR weights, precompute I - H_smooth for efficient smoothness regularization, and build neighbor lists for active set construction.
        The matrix H_smooth is defined such that H_smooth @ h gives the latent interpolated from neighbors, and (I - H_smooth) @ h gives the difference from neighbors used for smoothness regularization.
        """
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

        self._H_interp_csr[False] = self._build_interpolation_operator(include_self_in_interpolation=False)
        self._H_interp_csr[True] = self._build_interpolation_operator(include_self_in_interpolation=True)
        self.H_interp[False] = self._csr_to_torch(self._H_interp_csr[False])
        self.H_interp[True] = self._csr_to_torch(self._H_interp_csr[True])

        log.info(
            "Torch latent solver: H_smooth=%s, embed_dim=%d, device=%s",
            H_smooth.shape,
            self.embed_dim,
            self.device,
        )

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
        Build the set of row indices needed for the current solve, which includes all active bins 
        plus any bins that share a smoothness constraint with the active bins (i.e., any bin that is
        a neighbor of an active bin in the graph, since smoothness regularization couples neighbors)
        """
        if self._I_minus_H_smooth_csc is None:
            raise RuntimeError("I_minus_H_smooth CSC is not initialized")

        csc = cast(sparse.csc_matrix, self._I_minus_H_smooth_csc)
        dep_rows = csc[:, active_ids].indices.astype(np.int64, copy=False)
        row_ids = np.unique(np.concatenate([active_ids.astype(np.int64, copy=False), dep_rows]))
        return row_ids

    def solve(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        final_weights: Optional[np.ndarray] = None,
        bin_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        interpolation_mask: Optional[np.ndarray] = None,
        loss_type: Literal["cross_entropy", "logistic"] = "cross_entropy",
        prox_weight: float = 0.0,
        x_anchor: Optional[np.ndarray] = None,
        latent_lr: Optional[float] = None,
        include_self_in_interpolation: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Solve for the latent variable h using PyTorch optimization on the active set of bins 
        corresponding to the current batch, with optional proximal regularization towards an 
        anchor point. The optimization is performed using Adam with gradients computed from 
        the specified loss type, and only updates the latent variables for the active bins 
        while keeping others fixed.
        
        Args:
            y (np.ndarray): Observed target values for the current batch (shape [batch_size,]).
            intrinsic_vec (np.ndarray): Intrinsic predictions obtained from the MLP for the current batch (shape [batch_size, embed_dim] or [batch_size,]).
            final_weights (Optional[np.ndarray], optional): Weights for the final linear layer (shape [embed_dim,]) used in the loss computation when embed_dim > 1. Required if embed_dim > 1. Defaults to None.
            bin_ids (Optional[np.ndarray], optional): BIN indices corresponding to each observation in the current batch (shape [batch_size,]). Required for mapping observations to latent variables. Defaults to None.
            sample_ids (Optional[np.ndarray], optional): Sample indices corresponding to each observation in the current batch (shape [batch_size,]). Used for sample-level loss modes. Defaults to None.
            loss_type (Literal[&quot;cross_entropy&quot;, &quot;logistic&quot;], optional): Type of loss function to use. Defaults to "cross_entropy".
            prox_weight (float, optional): Weight for proximal regularization towards x_anchor. If > 0, adds a term prox_weight * ||h - x_anchor||^2 to the loss. Defaults to 0.0.
            x_anchor (Optional[np.ndarray], optional): Anchor values for the latent variables, obtained from the previous iteration (shape [n_bins, embed_dim]). Defaults to None.
            latent_lr (Optional[float], optional): Learning rate to use for the freshly initialized Adam optimizer. Defaults to cfg.latent_adam_lr.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing:
                - The optimized latent variable h as a numpy array of shape [n_bins, embed_dim] (or [n_bins,] if embed_dim=1).
                - A dictionary of timing information for different parts of the optimization process.
        """
        if self.I_minus_H_smooth is None:
            raise RuntimeError("Matrices not built; call build_interpolation_matrix first")
        if bin_ids is None:
            raise ValueError("bin_ids are required for torch latent solving")

        if self.embed_dim > 1 and final_weights is None:
            raise ValueError("final_weights are required for embed_dim > 1")

        latent_shape = (self.n_bins, self.embed_dim)
        if x_anchor is None:
            raise ValueError("x_anchor is required for latent solving")

        lr = float(self.cfg.latent_adam_lr if latent_lr is None else latent_lr)
        
        solve_start = time.perf_counter()

        anchor_np = np.asarray(x_anchor, dtype=np.float32).reshape(latent_shape)
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

        interpolation_mask_t: Optional[torch.Tensor] = None
        has_interpolation = False
        if interpolation_mask is not None:
            interpolation_mask_t = torch.as_tensor(np.asarray(interpolation_mask, dtype=np.bool_).reshape(-1), dtype=torch.bool, device=self.device)
            has_interpolation = bool(torch.any(interpolation_mask_t).item())

        w_t: Optional[torch.Tensor] = None
        if self.embed_dim > 1:
            w_t = torch.as_tensor(np.asarray(final_weights, dtype=np.float32).reshape(-1), device=self.device)
            if w_t.shape[0] != self.embed_dim:
                raise ValueError(f"final_weights has shape {tuple(w_t.shape)}, expected ({self.embed_dim},)")

        anchor_t = torch.as_tensor(anchor_np, dtype=torch.float32, device=self.device)
        active_ids_t = torch.as_tensor(active_map.global_ids, dtype=torch.long, device=self.device)
        active_anchor = anchor_t[active_ids_t]
        active_latent = torch.nn.Parameter(active_anchor.clone())
        optimizer = torch.optim.Adam([active_latent], lr=lr)

        row_ids = self._build_row_set(active_map.global_ids)
        L_rows, L_RA = self._build_row_and_active_operators(row_ids, active_map.global_ids)

        # Map global bin ids to local active set indices for the current batch; these are used to index into the active latent variable during optimization and must be consistent with the active set construction
        local_bin_ids = np.searchsorted(active_map.global_ids, batch_bin_ids)
        if not np.array_equal(active_map.global_ids[local_bin_ids], batch_bin_ids):
            raise RuntimeError("Active-set searchsorted mapping failed for one or more bin ids")
        local_bin_ids_t = torch.as_tensor(local_bin_ids, dtype=torch.long, device=self.device)
        batch_bin_ids_t = torch.as_tensor(batch_bin_ids, dtype=torch.long, device=self.device)

        steps = int(self.cfg.latent_adam_steps)
        if steps < 1:
            steps = 1
        
        # Timing for the entire solve
        setup_done = time.perf_counter() - solve_start
        timings = {"setup_s": setup_done, "forward_s": [], "backward_s": [], "optim_s": [], "latent_lr": lr}

        for _ in range(steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            active_update = active_latent - active_anchor

            if has_interpolation and interpolation_mask_t is not None:
                full_latent = anchor_t.clone().index_copy(0, active_ids_t, active_latent)
                logits = self._logits_from_latent(
                    full_latent,
                    intrinsic_t,
                    batch_bin_ids_t,
                    final_weights_t=w_t,
                    interpolation_mask=interpolation_mask_t,
                    include_self_in_interpolation=include_self_in_interpolation,
                )
            else:
                logits = self._logits_from_latent(active_latent, intrinsic_t, local_bin_ids_t, final_weights_t=w_t)

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
                loss = loss + 0.5 * l2_coef * torch.sum(active_latent * active_latent)

            # Smoothness with row closure and frozen-boundary contribution.
            smooth_coef = float(self.cfg.latent_smooth_reg)
            if smooth_coef > 0:
                base = torch.sparse.mm(L_rows, anchor_t)
                corr = torch.sparse.mm(L_RA, active_update)
                diff = base + corr
                loss = loss + 0.5 * smooth_coef * torch.sum(diff * diff)

            if prox_weight > 0:
                loss = loss + 0.5 * float(prox_weight) * torch.sum(active_update * active_update)

            loss = loss / scale
            t1 = time.perf_counter()
            
            loss.backward()
            t2 = time.perf_counter()
            optimizer.step()
            t3 = time.perf_counter()
            
            timings["forward_s"].append(t1 - t0)
            timings["backward_s"].append(t2 - t1)
            timings["optim_s"].append(t3 - t2)

        out = anchor_np.copy()
        out[active_map.global_ids] = active_latent.detach().cpu().numpy()
        if self.embed_dim == 1:
            return out.reshape(-1), timings
        return out, timings

    def _build_row_and_active_operators(
        self,
        row_ids: np.ndarray,
        active_ids: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the sparse operators needed for the smoothness regularization term in the optimization 
        problem. L_rows is the operator that computes the smoothness differences for all rows in 
        row_ids (using I - H), and L_RA is the operator that computes the contribution of active 
        variables to those differences, allowing us to exclude inactive variables from the 
        optimization problem while still correctly computing the smoothness regularization.

        Args:
            row_ids (np.ndarray): Row indices corresponding to the smoothness constraints that need to be evaluated for the current active set (includes active bins and any bins that share a smoothness constraint with them).
            active_ids (np.ndarray): Indices of the active bins for the current optimization block (subset of row_ids that correspond to the bins we are optimizing over).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two sparse tensors: 
                - L_rows: Sparse operator that computes the smoothness differences for all rows in row_ids when multiplied by the full latent variable h (shape [len(row_ids), n_bins]).
                - L_RA: Sparse operator that computes the contribution of the active variables to those differences, effectively giving us the part of L_rows that corresponds to the active variables (shape [len(row_ids), len(active_ids)]).
        """
        if self.I_minus_H_smooth is None:
            raise RuntimeError("I_minus_H_smooth is not initialized")
        L_rows_csr = self.I_minus_H_smooth[row_ids, :].tocsr()
        L_RA_csr = L_rows_csr[:, active_ids].tocsr()
        return self._csr_to_torch(L_rows_csr), self._csr_to_torch(L_RA_csr)

    def _build_active_set(self, batch_bin_ids: np.ndarray) -> ActiveSetMap:
        """
        Build the active set of bins for the current optimization block based on the batch BIN ids 
        and the neighbor graph. The active set includes all bins in the current batch plus their 
        neighbors up to a certain number of hops or a KNN-based cap, depending on the configuration. 
        This function uses a breadth-first search approach to traverse the neighbor graph starting 
        from the batch bins and adding neighbors until the specified criteria are met.

        Args:
            batch_bin_ids (np.ndarray): Array of BIN indices corresponding to the current batch (shape [batch_size,]).

        Returns:
            ActiveSetMap: Data structure containing the global BIN indices of the active set for the current optimization block, which includes the batch bins and their neighbors as determined by the configured criteria.
        """
        if self._graph_neighbors is None:
            raise RuntimeError("Graph neighbors are not initialized")

        mode = str(self.cfg.latent_k_hop_mode)
        max_hops = max(1, int(self.cfg.latent_k_hop_threshold))
        knn_cap = max(1, int(self.cfg.latent_hop_knn_cap))

        active = set(int(b) for b in batch_bin_ids.tolist())
        frontier = set(active)

        if mode == "knn":
            # Global cap on added neighbors: continue to further hops until we hit it.
            target_size = len(active) + knn_cap
            while frontier and len(active) < target_size:
                next_frontier: set[int] = set()
                for b in sorted(frontier):
                    neigh_arr = self._graph_neighbors[int(b)]
                    for n in neigh_arr:
                        n_int = int(n)
                        if n_int in active:
                            continue
                        active.add(n_int)
                        next_frontier.add(n_int)
                        if len(active) >= target_size:
                            break
                    if len(active) >= target_size:
                        break
                frontier = next_frontier
        else:
            for _ in range(max_hops):
                next_frontier: set[int] = set()
                for b in frontier:
                    neigh_arr = self._graph_neighbors[int(b)]
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
        Build the list of neighbor arrays for each node from the NeighbourGraph.neighbours 
        structure, which is used to construct the active set of bins for each optimization block. 
        Each entry in the list corresponds to a node and contains an array of its neighbors 
        (defined with KNN or threshold, on taxonomic or embedding-based distances), which 
        are included in the active set when that node is part of the batch.
        """
        neighbors: List[np.ndarray] = []
        for i in range(self.n_bins):
            neigh = np.asarray(self.ng.neighbours[i], dtype=np.int64)
            neighbors.append(neigh)

        log.info("Active graph neighbors initialized from NeighbourGraph.neighbours")
        return neighbors

    def _logits_from_latent(
        self,
        latent_source: torch.Tensor,
        intrinsic_t: torch.Tensor,
        bin_ids_t: torch.Tensor,
        final_weights_t: Optional[torch.Tensor] = None,
        interpolation_mask: Optional[torch.Tensor] = None,
        include_self_in_interpolation: bool = False,
    ) -> torch.Tensor:
        latent_obs = latent_source[bin_ids_t]
        own_logits = self._compute_logits_from_latent_values(latent_obs, intrinsic_t, final_weights_t)

        if interpolation_mask is None:
            return own_logits

        mask_t = interpolation_mask.to(device=latent_source.device, dtype=torch.bool).reshape(-1)
        if not bool(torch.any(mask_t).item()):
            return own_logits

        latent_2d = latent_source.unsqueeze(-1) if latent_source.ndim == 1 else latent_source
        interp_operator = self.get_interpolation_operator(include_self_in_interpolation, device=latent_2d.device)
        interpolated_full = torch.sparse.mm(interp_operator, latent_2d)
        interpolated_obs = interpolated_full[bin_ids_t]
        interp_logits = self._compute_logits_from_latent_values(interpolated_obs, intrinsic_t, final_weights_t)

        if own_logits.ndim == 1:
            return torch.where(mask_t, interp_logits, own_logits)
        return torch.where(mask_t.unsqueeze(-1), interp_logits, own_logits)

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
