from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

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
        latent: torch.Tensor,
        intrinsic: torch.Tensor,
        final_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embed_dim == 1:
            return intrinsic.squeeze(-1) + latent.squeeze(-1)

        if final_weights is None:
            raise ValueError("final_weights is required for embed_dim > 1")

        gated = self.gating.gate_torch(latent)
        m_tilde = intrinsic * gated
        return torch.sum(m_tilde * final_weights.unsqueeze(0), dim=1)

    def build_interpolation_matrix(self) -> None:
        """
        Build sparse interpolation/smoothness operators from the neighbor graph.

        method="nw" uses kernel-normalized Nadaraya-Watson weights, while other
        values currently route to LLR coefficients from the neighbor graph.

        Semantics:
        - H_smooth @ h gives neighbor-interpolated latent values.
        - (I - H_smooth) @ h gives smoothness residuals used by regularization.
        - If a node has no neighbors, we place a 1 on the diagonal so that node keeps its own latent
        instead of introducing an undefined row.
        """
        rows_H: List[int] = []
        cols_H: List[int] = []
        vals_H: List[float] = []

        for b in tqdm(range(self.n_bins), desc="Building H matrix", unit="bin", leave=False):
            if self.cfg.interpolation_method == "nw":
                neigh, w = self.ng.nw_weights_for_node(b)
            elif self.cfg.interpolation_method == "llr":
                neigh, w = self.ng.llr_coeffs_for_node(b)
            else:
                raise ValueError(f"Unsupported interpolation method: {self.cfg.interpolation_method}")

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

    def solve(
        self,
        y: torch.Tensor,
        intrinsic: torch.Tensor,
        final_weights: Optional[torch.Tensor] = None,
        bin_ids: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        interpolation_mask: Optional[torch.Tensor] = None,
        loss_type: Literal["cross_entropy", "logistic"] = "cross_entropy",
        prox_weight: float = 0.0,
        latent: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Solve for the latent variable h using PyTorch optimization on the active set of bins 
        corresponding to the current batch, with optional proximal regularization towards an 
        anchor point. The optimization is performed using a persistent latent tensor
        and optimizer supplied by the caller, and only updates the part of the latent variable 
        corresponding to the active set of bins for the current batch.
        
        Args:
            y (torch.Tensor): Observed target values for the current batch (shape [batch_size,]).
            intrinsic (torch.Tensor): Intrinsic predictions obtained from the MLP for the current batch (shape [batch_size, embed_dim] or [batch_size,]).
            final_weights (Optional[torch.Tensor], optional): Weights for the final linear layer (shape [embed_dim,]) used in the loss computation when embed_dim > 1. Required if embed_dim > 1. Defaults to None.
            bin_ids (Optional[torch.Tensor], optional): BIN indices corresponding to each observation in the current batch (shape [batch_size,]). Required for mapping observations to latent variables. Defaults to None.
            sample_ids (Optional[torch.Tensor], optional): Sample indices corresponding to each observation in the current batch (shape [batch_size,]). Used for sample-level loss modes. Defaults to None.
            loss_type (Literal[&quot;cross_entropy&quot;, &quot;logistic&quot;], optional): Type of loss function to use. Defaults to "cross_entropy".
            prox_weight (float, optional): Weight for proximal regularization towards x_anchor. If > 0, adds a term prox_weight * ||h - x_anchor||^2 to the loss. Defaults to 0.0.
            latent (Optional[torch.Tensor], optional): Persistent latent tensor to optimize in-place. If omitted, a temporary parameter is created for this call.
            optimizer (Optional[torch.optim.Optimizer], optional): Optimizer for latent. Required when latent is supplied.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: A tuple containing:
                - The optimized latent variable h tensor of shape [n_bins, embed_dim] (or [n_bins,] if embed_dim=1).
                - A dictionary of timing information for different parts of the optimization process.
        """
        solve_start = time.perf_counter()
        
        if self.I_minus_H_smooth is None: 
            raise RuntimeError("Matrices not built; call build_interpolation_matrix first")
        if bin_ids is None: 
            raise ValueError("bin_ids are required for latent solving")
        if self.embed_dim > 1 and final_weights is None: 
            raise ValueError("final_weights are required for embed_dim > 1")
        if latent is None or optimizer is None:
            raise ValueError("latent_param and optimizer must both be provided for persistent latent solving")
        if latent.device != self.device:
            raise ValueError(f"latent_param is on {latent.device}, expected {self.device}")

        y = y.to(device=self.device, dtype=torch.float32).reshape(-1)
        intrinsic = intrinsic.to(device=self.device, dtype=torch.float32)
        if intrinsic.ndim == 1:
            intrinsic = intrinsic.unsqueeze(-1)
        if self.embed_dim == 1 and intrinsic.shape[1] != 1:
            intrinsic = intrinsic.reshape(-1, 1)
        if intrinsic.shape[1] != self.embed_dim:
            raise ValueError(
                f"intrinsic has shape {tuple(intrinsic.shape)}, expected (*, {self.embed_dim})"
            )

        w: Optional[torch.Tensor] = None
        if self.embed_dim > 1:
            w = final_weights.to(device=self.device, dtype=torch.float32).reshape(-1)
            if w.shape[0] != self.embed_dim:
                raise ValueError(f"final_weights has shape {tuple(w.shape)}, expected ({self.embed_dim},)")

        bin_ids = bin_ids.to(device=self.device, dtype=torch.long).reshape(-1)

        if sample_ids is not None:
            sample_ids = sample_ids.to(device=self.device, dtype=torch.long).reshape(-1)

        has_interpolation = False
        if interpolation_mask is not None:
            interpolation_mask = interpolation_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
            has_interpolation = bool(torch.any(interpolation_mask).item())

        anchor_latent = latent.detach().to(self.device, dtype=torch.float32).reshape(self.n_bins, self.embed_dim)
        batch_bin_ids = bin_ids.detach().cpu().numpy()
        active_map = self._build_active_set(batch_bin_ids) # batch bins + neighbors that are optimized over in this block
        active_bin_ids = torch.as_tensor(active_map.global_ids, dtype=torch.long, device=self.device)
        active_anchor_latent = anchor_latent[active_bin_ids]

        # Map global bin ids to local active set ids for the current batch; used to index into the 
        # active latent during optimization (must be consistent with active set construction)
        local_bin_ids = np.searchsorted(active_map.global_ids, batch_bin_ids)
        if not np.array_equal(active_map.global_ids[local_bin_ids], batch_bin_ids):
            raise RuntimeError("Active-set searchsorted mapping failed for one or more bin ids")

        # Build the sparse operators for smoothness reg term based on the active set for this batch
        L_rows, L_RA = self._build_row_and_active_operators(active_map.global_ids)
        frozen_smooth_contribution: Optional[torch.Tensor] = None
        if float(self.cfg.latent_smooth_reg) > 0:
            # Cache baseline smoothness term once; per-step updates only need the
            # active-set correction through L_RA @ (h_active - h_anchor_active).
            frozen_smooth_contribution = torch.sparse.mm(L_rows, anchor_latent)

        setup_done = time.perf_counter() - solve_start
        timings = {
            "setup_s": float(setup_done),
            "forward_s": 0.0,
            "backward_s": 0.0,
            "optim_s": 0.0,
            "latent_lr": float(optimizer.param_groups[0]["lr"]),
            "loss_CE": 0.0,
            "loss_l2": 0.0,
            "loss_smooth": 0.0,
            "loss_prox": 0.0,
            "loss_total": 0.0,
        }

        steps = max(1, int(self.cfg.latent_optim_steps))
        for _ in range(steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            active_current_latent = latent[active_bin_ids]
            active_latent_update = active_current_latent - active_anchor_latent

            if has_interpolation and interpolation_mask is not None:
                logits = self._logits_from_latent(
                    latent,
                    intrinsic,
                    bin_ids,
                    final_weights=w,
                    interpolation_mask=interpolation_mask
                )
            else:
                logits = self._logits_from_latent(latent, intrinsic, bin_ids, final_weights=w)

            if loss_type == "cross_entropy":
                if sample_ids is None:
                    raise ValueError("sample_ids are required for cross_entropy latent solving")
                data_term, scale = self._cross_entropy_loss(y, logits, sample_ids)
            elif loss_type == "logistic":
                data_term, scale = self._logistic_loss(y, logits)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            l2_term = torch.zeros((), dtype=data_term.dtype, device=data_term.device)
            smooth_term = torch.zeros((), dtype=data_term.dtype, device=data_term.device)
            prox_term = torch.zeros((), dtype=data_term.dtype, device=data_term.device)

            # Regularization terms - L2 on H and smoothness via L_A
            l2_coef = float(self.cfg.latent_l2_reg)
            if l2_coef > 0:
                l2_term = 0.5 * l2_coef * torch.sum(active_current_latent * active_current_latent)

            # Smoothness regularization on the active set only (for efficiency)
            smooth_coef = float(self.cfg.latent_smooth_reg)
            if smooth_coef > 0:
                if frozen_smooth_contribution is None:
                    raise RuntimeError("frozen_smooth_contribution was not initialized")
                # L_RA @ active_latent = smoothness diff of new latents with active BINs + neighbors
                active_correction = torch.sparse.mm(L_RA, active_latent_update)
                diff = frozen_smooth_contribution + active_correction
                smooth_term = 0.5 * smooth_coef * torch.sum(diff * diff)

            # Proximal term stabilizes local active-set updates relative to anchor_latent,
            # especially early in training when gradients can move sparse blocks abruptly.
            if prox_weight > 0: # FIXME: I'm not completely sure this is necessary because why would we need the latent update to be small, especially if we already have L2 and smoothness regularization? Maybe it just helps with optimization stability early on when the latent can change a lot from the anchor + given we only optimize over the active set?
                prox_term = 0.5 * float(prox_weight) * torch.sum(active_latent_update * active_latent_update)

            loss_unscaled = data_term + l2_term + smooth_term + prox_term

            # data_loss returns a scale so CE/logistic modes have comparable step sizes.
            inv_scale = 1.0 / scale
            loss = loss_unscaled * inv_scale

            timings["loss_CE"] += float((data_term * inv_scale).detach().item())
            timings["loss_l2"] += float((l2_term * inv_scale).detach().item())
            timings["loss_smooth"] += float((smooth_term * inv_scale).detach().item())
            timings["loss_prox"] += float((prox_term * inv_scale).detach().item())
            timings["loss_total"] += float(loss.detach().item())
            t1 = time.perf_counter()
            
            loss.backward()
            t2 = time.perf_counter()
            optimizer.step()
            t3 = time.perf_counter()
            
            timings["forward_s"] += float(t1 - t0)
            timings["backward_s"] += float(t2 - t1)
            timings["optim_s"] += float(t3 - t2)

        inv_steps = 1.0 / float(steps)
        timings["loss_CE"] *= inv_steps
        timings["loss_l2"] *= inv_steps
        timings["loss_smooth"] *= inv_steps
        timings["loss_prox"] *= inv_steps
        timings["loss_total"] *= inv_steps


        out = latent.detach()
        if self.embed_dim == 1:
            return out.reshape(-1), timings
        return out, timings
    
    def _build_row_and_active_operators(
        self,
        active_ids: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the sparse operators needed for the smoothness regularization term in the 
        optimization problem.

        This combines the logic of:
        - building the set of row indices needed for the current solve, and
        - constructing the sparse operators for those rows and for the active subset.

        The row set includes:
        - all active bins, and
        - any bins that share a smoothness constraint with the active bins.

        In practice, this means we take all rows of I - H_smooth that have a nonzero in any of the
        active columns. These rows correspond to:
        - the active bins themselves, because the diagonal of I - H_smooth is 1, and
        - any neighboring bins coupled through smoothness regularization.

        Args:
            active_ids (np.ndarray): Indices of the active bins for the current optimization block.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - L_rows: Sparse operator that computes the smoothness differences for all rows in
                row_ids when multiplied by the full latent variable h. Shape:
                [len(row_ids), n_bins].
                - L_RA: Sparse operator that computes the contribution of the active variables to
                those differences, i.e. the part of L_rows restricted to the active columns.
                Shape: [len(row_ids), len(active_ids)].
        """
        if self._I_minus_H_smooth_csc is None:
            raise RuntimeError("I_minus_H_smooth CSC is not initialized")
        if self.I_minus_H_smooth is None:
            raise RuntimeError("I_minus_H_smooth is not initialized")

        active_ids = active_ids.astype(np.int64, copy=False)

        # CSC is used here because column slicing is efficient for identifying dependencies
        csc = cast(sparse.csc_matrix, self._I_minus_H_smooth_csc)
        dep_rows = csc[:, active_ids].indices.astype(np.int64, copy=False)

        # Get all rows that have a nonzero in any active column, which corresponds to all bins that 
        # are either active or share a smoothness constraint with an active bin
        row_ids = np.union1d(active_ids, dep_rows)

        # Slice the I - H_smooth matrix to get the operators for the relevant rows & active columns
        # L_rows @ latent gives the smoothness differences for all rows in the active set, and
        # L_RA @ latent_active gives the contribution of the active variables to those differences.
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
        latent: torch.Tensor,
        intrinsic: torch.Tensor,
        bin_ids: torch.Tensor,
        final_weights: Optional[torch.Tensor] = None,
        interpolation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_obs = latent[bin_ids]
        own_logits = self._compute_logits_from_latent_values(latent_obs, intrinsic, final_weights)

        if interpolation_mask is None:
            return own_logits

        mask = interpolation_mask.to(device=latent.device, dtype=torch.bool).reshape(-1)
        if not bool(torch.any(mask).item()):
            return own_logits

        latent_2d = latent.unsqueeze(-1) if latent.ndim == 1 else latent
        interp_operator = self.get_interpolation_operator(self.cfg.include_self_in_interpolation, device=latent_2d.device)
        interpolated_full = torch.sparse.mm(interp_operator, latent_2d)
        interpolated_obs = interpolated_full[bin_ids]
        interp_logits = self._compute_logits_from_latent_values(interpolated_obs, intrinsic, final_weights)

        if own_logits.ndim == 1:
            return torch.where(mask, interp_logits, own_logits)
        return torch.where(mask.unsqueeze(-1), interp_logits, own_logits)

    def _cross_entropy_loss(
        self,
        y: torch.Tensor,
        logits: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expected format: flattened observations grouped contiguously by sample id.
        # We repack into a padded [n_samples, max_bins] layout and reuse dense log_softmax CE.
        _, counts = torch.unique_consecutive(sample_ids, return_counts=True)
        n_samples = max(1, int(counts.numel()))
        max_bins = int(counts.max().item())

        sample_rows = torch.repeat_interleave(
            torch.arange(n_samples, device=logits.device, dtype=torch.long),
            counts,
        )
        starts = torch.cumsum(counts, dim=0) - counts
        obs_pos = torch.arange(logits.numel(), device=logits.device, dtype=torch.long)
        sample_cols = obs_pos - torch.repeat_interleave(starts, counts)

        logits_dense = torch.full((n_samples, max_bins), -torch.inf, dtype=logits.dtype, device=logits.device)
        targets_dense = torch.zeros((n_samples, max_bins), dtype=y.dtype, device=y.device)
        valid_dense = torch.zeros((n_samples, max_bins), dtype=torch.bool, device=logits.device)
        logits_dense[sample_rows, sample_cols] = logits
        targets_dense[sample_rows, sample_cols] = y
        valid_dense[sample_rows, sample_cols] = True

        log_probs = F.log_softmax(logits_dense, dim=-1)
        log_probs = torch.where(valid_dense, log_probs, torch.zeros_like(log_probs))
        ce_sum = -(targets_dense * log_probs).sum()
        return ce_sum, torch.tensor(float(n_samples), dtype=logits.dtype, device=logits.device)

    def _logistic_loss(
        self,
        y: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bce_sum = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        scale = torch.tensor(float(max(1, y.numel())), dtype=logits.dtype, device=logits.device)
        return bce_sum, scale
