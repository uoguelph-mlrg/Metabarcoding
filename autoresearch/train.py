from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast, Set, Callable, Iterable, Type
import os

import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from scipy import sparse
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.preprocessing import normalize
import sys
from tqdm import tqdm

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from prepare import MBDataset, collate_samples, load, load_or_preprocess, Loss, TIME_BUDGET


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@dataclass
class Config:
    # Run configuration
    data_path: str = os.path.join(PROJECT_ROOT, "data", "data_merged.csv")  # Path to raw data CSV file
    results_dir: str = "../results"             # Directory where run artifacts are saved
    checkpoint_every: int = 5                   # Save periodic checkpoint every N epochs
    diag_ablation_interval: int = 20            # Compute latent ablation delta every N epochs (0 = disabled)

    # Train / val / test split
    train_frac: float = 0.8
    val_frac: float = 0.1
    
    # Basic training settings
    loss_type: Literal["cross_entropy", "logistic"] = "cross_entropy"
    device: str = (
        "mps" if torch.backends.mps.is_available() else 
        "cuda" if torch.cuda.is_available() else 
        "cpu"
    )
    batch_size_bin: int = 1024                  # Batch size (in number of observations not samples)
    batch_size_sample: int = 8                  # Batch size in number of samples
    epochs: int = 200                           # Epochs per training phase
    grad_clip: Optional[float] = 1.0            # Gradient clipping value (None to disable)

    # Neighbour graph
    use_taxonomy: bool = False                  # set to True to use taxonomic distances
    use_embedding: bool = True                  # set to True to use DNA embedding-based neighbors
    neighbor_mode: str = "knn"                  # "threshold" for distance-based, "knn" for K-nearest neighbors
    K: int = 25                                 # number of neighbors (used when neighbor_mode="knn")
    dist_thres: int = 4                         # max taxonomic distance (used when neighbor_mode="threshold")
    emb_radius: float = 1.0                     # max embedding distance (used when neighbor_mode="threshold")
    kernel_q: Optional[float] = None            # kernel q parameter (if None, computed adaptively)
    interpolation_method: Literal["nw", "llr"] = "nw"  # interpolation method for latent solver: "nw" for Nadaraya-Watson, "llr" for locally linear regression

    # DNA embedding settings (used when use_embedding=True)
    embedding_path: Optional[str] = os.path.join(PROJECT_ROOT, "data", "embeddings.npy")  # path to precomputed embeddings (.npy dict: bin_uri->vector)
    barcode_data_path: Optional[str] = None     # path to TSV with 'bin_uri' and 'seq' columns
    emb_distance_metric: str = "cosine"         # distance metric: "cosine" or "euclidean"

    # MLP - architecture & optimization settings
    mlp_hidden_dims : List[int] = field(default_factory=lambda: [128, 128, 128, 128])  # Hidden layer dimensions for MLP
    mlp_lr: float = 5e-4                        # Learning rate for MLP parameters
    weight_decay: float = 1e-5                  # Weight decay for MLP parameters
    mlp_warmup_start_factor: float = 1e-3       # Initial multiplier for MLP LR warmup
    mlp_warmup_frac: float = 0.1                # Fraction of total training steps used for MLP LR warmup
    mlp_lr_eta_min: float = 1e-6                # Minimum MLP LR reached by cosine decay
    dropout: float = 0.15                       # Dropout rate in MLP

    # Latent solver - regularization settings
    latent_smooth_reg: float = 1e-3             # Smoothness regularization (parameter λ_smooth)
    latent_present_only: bool = False           # If True, only fit latent on observations where y > 0 (useful with loss='logistic' to avoid distribution shift)
    latent_l2_reg: float = 1e-3                 # L2 norm regularization on D (parameter r)
    latent_init_prox_reg: float = 0.0           # Initial proximal regularization weight; annealed to 0 across epochs to stabilize early active-set latent updates.

    # Latent solver - optimization settings
    latent_optim_steps: int = 15                # Number of latent optimization steps per batch / solver call
    latent_lr: float = 1e-2                     # Learning rate for the latent AdamW optimizer
    latent_init_std: float = 0.0                # Standard deviation for initializing latent embeddings (0 for zeros, >0 for Gaussian noise)
    latent_warmup_start_factor: float = 1e-3    # Initial multiplier for latent LR warmup
    latent_warmup_frac: float = 0.2             # Fraction of total latent solves used for warmup
    latent_lr_eta_min: float = 1e-6             # Minimum latent LR reached by cosine decay
    latent_k_hop_mode: Literal["threshold", "knn"] = "threshold"  # Method for selecting subset of neighbors for latent optimization 
    latent_k_hop_threshold: int = 2             # Number of neighbor graph hops to select BINs from (used when latent_k_hop_mode="threshold")
    latent_hop_knn_cap: int = 64                # Max number of neighbors to include in latent optimization (used when latent_k_hop_mode="knn")

    # Training with interpolated latents settings
    interpolated_sample_fraction: float = 0.0   # Fraction of training samples using interpolated latent (set to 0 to disable interpolation during training)
    train_MLP_with_interpolation: bool = False  # Whether to train the MLP on interpolated latents too (instead of only using them in the latent solver)
    inference_with_interpolation: bool = False  # Whether to use interpolated latents during inference (if False, uses BINs own latent)
    include_self_in_interpolation: bool = False # Whether to include the BIN's own latent in the interpolation (instead of only using neighbors)

    # Sizes and combination modalities for latent and intrinsic vectors
    embed_dim: int = 10                         # Embedding dimension d for both latent and intrinsic vectors (set to 1 for scalars)
    gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid"  # Gating function for combining latent and intrinsic vectors
    gating_alpha: float = 0.5                   # Scaling factor for scaled_exp gating (in (0,1])
    gating_kappa: float = 0.5                   # Scaling factor for tanh gating
    gating_epsilon: float = 0.693               # Offset for softplus gating (log(2), so g(0)=1)
    final_linear_weight_decay: float = 1e-3     # Weight decay specifically for final linear layer w

def cpu_if_mps(device: torch.device) -> torch.device:
    """Return CPU when device is MPS.

    Sparse CSR operations (used by the latent solver and interpolation operators)
    are numerically unstable on MPS. All callers that need a sparse-safe device
    should use this helper instead of duplicating the check.
    """
    return torch.device("cpu") if device.type == "mps" else device


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GatingFunction(ABC):
    """
    Abstract base for gating functions g(h) with the property g(0) = 1.

    Used for multiplicative modulation: m̃ = m(x) ⊙ g(h[bin])
    Logit: z = w^T m̃

    Subclasses implement the forward and gradient in both numpy (for the
    L-BFGS latent solver) and PyTorch (for the MLP training forward pass).
    """

    @abstractmethod
    def gate_np(self, h: np.ndarray) -> np.ndarray:
        """Compute g(h) — numpy, used by LatentSolver."""

    @abstractmethod
    def gate_grad_np(self, h: np.ndarray) -> np.ndarray:
        """Compute g'(h) — numpy gradient, used by LatentSolver."""

    @abstractmethod
    def gate_torch(self, h: torch.Tensor) -> torch.Tensor:
        """Compute g(h) — PyTorch, used by Model.forward."""


# ---------------------------------------------------------------------------
# Concrete gating functions
# ---------------------------------------------------------------------------

class ExpGating(GatingFunction):
    """g(h) = exp(h),  g(0) = 1."""

    def gate_np(self, h):
        return np.exp(h)

    def gate_grad_np(self, h):
        return np.exp(h)

    def gate_torch(self, h):
        return torch.exp(h)


class ScaledExpGating(GatingFunction):
    """g(h) = exp(α·h),  g(0) = 1.  α ∈ (0, 1] controls curvature."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def gate_np(self, h):
        return np.exp(self.alpha * h)

    def gate_grad_np(self, h):
        return self.alpha * np.exp(self.alpha * h)

    def gate_torch(self, h):
        return torch.exp(self.alpha * h)


class AdditiveGating(GatingFunction):
    """g(h) = 1 + h,  g(0) = 1.  Linear, unbounded."""

    def gate_np(self, h):
        return 1.0 + h

    def gate_grad_np(self, h):
        return np.ones_like(h)

    def gate_torch(self, h):
        return 1.0 + h


class SoftplusGating(GatingFunction):
    """g(h) = 1 + softplus(h) − ε,  g(0) = 1 when ε = log(2) ≈ 0.693."""

    def __init__(self, epsilon: float = 0.693):
        self.epsilon = epsilon

    def gate_np(self, h):
        return 1.0 + np.log1p(np.exp(h)) - self.epsilon

    def gate_grad_np(self, h):
        return 1.0 / (1.0 + np.exp(-h))  # σ(h) = derivative of softplus

    def gate_torch(self, h):
        return 1.0 + F.softplus(h) - self.epsilon


class TanhGating(GatingFunction):
    """g(h) = 1 + κ·tanh(h),  g(0) = 1.  Bounded to [1−κ, 1+κ]."""

    def __init__(self, kappa: float = 0.5):
        self.kappa = kappa

    def gate_np(self, h):
        return 1.0 + self.kappa * np.tanh(h)

    def gate_grad_np(self, h):
        return self.kappa * (1.0 - np.tanh(h) ** 2)

    def gate_torch(self, h):
        return 1.0 + self.kappa * torch.tanh(h)


class SigmoidGating(GatingFunction):
    """g(h) = 2·σ(h),  g(0) = 1.  Bounded to (0, 2)."""

    def gate_np(self, h):
        return 2.0 / (1.0 + np.exp(-h))

    def gate_grad_np(self, h):
        s = 1.0 / (1.0 + np.exp(-h))
        return 2.0 * s * (1.0 - s)

    def gate_torch(self, h):
        return 2.0 * torch.sigmoid(h)


class DotProductGating(GatingFunction):
    """
    g(h) = h  →  m̃ = m ⊙ h,  z = w^T (m ⊙ h).

    Note: g(0) = 0, so the latent must be non-zero to produce non-trivial
    predictions.  Unlike the other gating functions, this variant passes
    through the final linear layer (w) just like all other gating functions.
    """

    def gate_np(self, h):
        return h

    def gate_grad_np(self, h):
        return np.ones_like(h)

    def gate_torch(self, h):
        return h


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GATING_MAP: dict[str, type] = {
    "exp":          ExpGating,
    "scaled_exp":   ScaledExpGating,
    "additive":     AdditiveGating,
    "softplus":     SoftplusGating,
    "tanh":         TanhGating,
    "sigmoid":      SigmoidGating,
    "dot_product":  DotProductGating,
}


def make_gating_function(
    name: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"],
    alpha: float = 0.5,    # for ScaledExpGating
    kappa: float = 0.5,    # for TanhGating
    epsilon: float = 0.693, # for SoftplusGating  (log(2) ensures g(0) = 1)
) -> GatingFunction:
    """
    Instantiate a GatingFunction by name.

    Parameters are forwarded only to the functions that use them:
        alpha   → ScaledExpGating
        kappa   → TanhGating
        epsilon → SoftplusGating

    All default values match the Config defaults and satisfy g(0) = 1.
    """
    if name not in _GATING_MAP:
        raise ValueError(f"Unknown gating function '{name}'. Choose from: {list(_GATING_MAP)}")

    if name == "scaled_exp":
        return ScaledExpGating(alpha=alpha)
    if name == "tanh":
        return TanhGating(kappa=kappa)
    if name == "softplus":
        return SoftplusGating(epsilon=epsilon)
    return _GATING_MAP[name]()


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

        self.device = cpu_if_mps(torch.device(self.cfg.device))  # sparse CSR ops are unstable on MPS

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
        return sparse.csr_matrix(inv @ mat)

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

        target_device = cpu_if_mps(self.device if device is None else device)
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

    def _csr_to_torch(self, mat: Optional[sparse.csr_matrix]) -> torch.Tensor:
        if mat is None:
            raise ValueError("mat cannot be None")
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
        if self.embed_dim > 1 and final_weights is not None:
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
            # Proximal pull is annealed to 0 across epochs (see Trainer.run). It is most
            # useful early in training: the active-set latent optimizes only a local
            # subgraph per batch, so without a proximal anchor nearby bins can drift
            # far from each other before the global picture converges. L2 and smoothness
            # regularization alone do not prevent this because they are relative terms;
            # the proximal term ties each bin to its own initialization anchor.
            if prox_weight > 0:
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
        latent_source: torch.Tensor,
        intrinsic: torch.Tensor,
        bin_ids: torch.Tensor,
        final_weights: Optional[torch.Tensor] = None,
        interpolation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_obs = latent_source[bin_ids]
        own_logits = self._compute_logits_from_latent_values(latent_obs, intrinsic, final_weights)

        if interpolation_mask is None:
            return own_logits

        mask_t = interpolation_mask.to(device=latent_source.device, dtype=torch.bool).reshape(-1)
        if not bool(torch.any(mask_t).item()):
            return own_logits

        latent_2d = latent_source.unsqueeze(-1) if latent_source.ndim == 1 else latent_source
        interp_operator = self.get_interpolation_operator(self.cfg.include_self_in_interpolation, device=latent_2d.device)
        interpolated_full = torch.sparse.mm(interp_operator, latent_2d)
        interpolated_obs = interpolated_full[bin_ids]
        interp_logits = self._compute_logits_from_latent_values(interpolated_obs, intrinsic, final_weights)

        if own_logits.ndim == 1:
            return torch.where(mask_t, interp_logits, own_logits)
        return torch.where(mask_t.unsqueeze(-1), interp_logits, own_logits)

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


class MLPModel(nn.Module):
    """
    MLP that predicts m_theta(s,b) given sample and bin features.
    Simple architecture with light regularization for small datasets.
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int] = [128, 128, 128, 128],
        output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always returns [N, output_dim]; callers squeeze when needed.
        return self.net(x)


class Model(nn.Module):
    """
    Joint model combining an MLP and a latent variable.

    Scalar mode (embed_dim == 1):
        - MLP outputs scalar m(x) ∈ R
        - Latent d_b ∈ R per BIN
        - Logit: z = m(x) + d_b  (additive)
        - Prediction: p = sigmoid(z)

    Vector mode (embed_dim > 1):
        - MLP outputs m(x) ∈ R^d
        - Latent h_b ∈ R^d provides feature-wise modulation
        - Gating: m̃ = m(x) ⊙ g(h[bin_id])
        - Logit: z = w^T m̃  (no bias)
        - Prediction: p = sigmoid(z)
    """

    def __init__(
        self,
        mlp: nn.Module,
        latent_solver: Any,
        n_bins: int,
        device: torch.device,
        embed_dim: int = 1,
        gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid",
        gating_alpha: float = 0.5,  # Only used by scaled_exp; other gating functions ignore it.
        gating_kappa: float = 0.5,
        gating_epsilon: float = 0.693,
        latent_init_std: float = 0.0,
        interpolation_enabled: bool = False,
        include_self_in_interpolation: bool = True,
    ):
        """
        Args:
            mlp: MLP model predicting intrinsic embedding m(s,b)
            latent_solver: Solver for latent variables
            n_bins: Number of BINs
            device: Compute device
            embed_dim: Embedding dimension (1 = scalar additive mode, >1 = vector gating mode)
            gating_fn: Gating function (used only when embed_dim > 1)
            gating_alpha: Scale for scaled_exp gating
            gating_kappa: Scale for tanh gating
            gating_epsilon: Offset for softplus gating
            interpolation_enabled: Whether to enable latent interpolation (requires precomputed interpolation operators in latent_solver)
            include_self_in_interpolation: Whether to include the BIN's own latent in the interpolation (instead of only using neighbors)
        """
        super().__init__()
        self.mlp = mlp
        self.latent_solver = latent_solver
        self.device = device
        self.embed_dim = embed_dim
        self.n_bins = n_bins
        self._interpolation_enabled = interpolation_enabled
        self.H_interp: Optional[torch.Tensor] = None

        if interpolation_enabled:
            interp_device = cpu_if_mps(self.device)
            self.H_interp = latent_solver.get_interpolation_operator(include_self_in_interpolation, device=interp_device)

        if embed_dim > 1:
            self.gating_fn = gating_fn
            self.gating = make_gating_function(
                gating_fn, alpha=gating_alpha, kappa=gating_kappa, epsilon=gating_epsilon
            )
            # Latent matrix H ∈ R^{n_bins × d}
            latent_init = torch.randn((n_bins, embed_dim), device=device) * latent_init_std if latent_init_std > 0 else torch.zeros((n_bins, embed_dim), device=device)
            self.latent_vec = nn.Parameter(
                latent_init,
                requires_grad=False,
            )
            # Final linear layer w: R^d → R (no bias)
            self.final_linear = nn.Linear(embed_dim, 1, bias=False, device=device)
            nn.init.xavier_uniform_(self.final_linear.weight)
        else:
            # Scalar mode: latent is a 1D vector, no gating, no final linear
            latent_init = torch.randn(n_bins, device=device) * latent_init_std if latent_init_std > 0 else torch.zeros(n_bins, device=device)
            self.latent_vec = nn.Parameter(
                latent_init,
                requires_grad=False,
            )

    def _compute_logits_from_latent_values(
        self,
        latent_obs: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        if self.embed_dim == 1:
            return intrinsic.squeeze(-1) + latent_obs.squeeze(-1)

        gated = self.gating.gate_torch(latent_obs)
        modulated = intrinsic * gated
        return self.final_linear(modulated).squeeze(-1)

    def _lookup_latent(
        self,
        bin_ids: torch.Tensor,
        interpolation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fetch latent values for observations, optionally mixing interpolated latents.

        interpolation_mask is observation-aligned (shape [N], True means use
        interpolation operator output for that observation, False means use the BIN's
        own latent value).
        """
        latent_source = self.latent_vec
        own_latent = latent_source[bin_ids]

        if interpolation_mask is None:
            return own_latent

        if not self._interpolation_enabled:
            raise RuntimeError("Interpolation was requested but interpolation operators were not initialized")

        mask_t = interpolation_mask.to(device=latent_source.device, dtype=torch.bool).reshape(-1)
        if not bool(torch.any(mask_t).item()):
            return own_latent

        latent_2d = latent_source.unsqueeze(-1) if self.embed_dim == 1 else latent_source
        interp_operator = self.H_interp
        if interp_operator is None:
            raise RuntimeError("Interpolation operator is not initialized")

        interp_device = interp_operator.device
        # Keep sparse matmul on the operator device to avoid repeated sparse transfers.
        latent_for_interp = latent_2d if interp_device == latent_2d.device else latent_2d.to(interp_device)
        if interp_device != latent_2d.device:
            interp_operator = interp_operator.to(interp_device)

        interpolated_full = torch.sparse.mm(interp_operator, latent_for_interp)
        interpolated_obs = interpolated_full[bin_ids]
        if interpolated_obs.device != own_latent.device:
            interpolated_obs = interpolated_obs.to(own_latent.device)
        if self.embed_dim == 1:
            own_latent_2d = own_latent.unsqueeze(-1)
            mixed = torch.where(mask_t.unsqueeze(-1), interpolated_obs, own_latent_2d)
            return mixed.squeeze(-1)

        mixed = torch.where(mask_t.unsqueeze(-1), interpolated_obs, own_latent)
        return mixed

    def forward(
        self,
        x: torch.Tensor,
        bin_ids: torch.Tensor,
        interpolation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [N, input_dim]
            bin_ids: BIN indices [N]

        Returns:
            Predicted logits [N] (raw logits before sigmoid/softmax)
        """
        if self.embed_dim > 1:
            intrinsic = self.mlp(x)                                          # [N, d]
        else:
            intrinsic = self.mlp(x).squeeze(-1)                              # [N]

        latent = self._lookup_latent(
            bin_ids,
            interpolation_mask=interpolation_mask,
        )

        return self._compute_logits_from_latent_values(latent, intrinsic)

    @torch.no_grad()
    def set_latent(self, latent_new: Union[torch.Tensor, np.ndarray]) -> None:
        """Update the latent variable after a Phase A solve."""
        if isinstance(latent_new, np.ndarray):
            latent_new = torch.tensor(latent_new, dtype=torch.float32)
        if self.embed_dim > 1:
            if latent_new.shape != (self.n_bins, self.embed_dim):
                raise ValueError(f"Expected shape ({self.n_bins}, {self.embed_dim}), got {latent_new.shape}")
        else:
            latent_new = latent_new.reshape(-1)
            if latent_new.shape[0] != self.n_bins:
                raise ValueError(f"Expected shape ({self.n_bins},), got {latent_new.shape}")
        self.latent_vec.copy_(latent_new.to(self.device))

    @torch.no_grad()
    def predict_MLP_only(self, data_loader, loss_mode: str = "bin") -> np.ndarray:
        """Predict intrinsic MLP outputs without latent modulation.

        In sample mode, predictions are returned only for valid (non-padded) bins in
        each sample. In bin mode, predictions are returned for every observation.

        Returns:
            numpy array of shape (N,) for scalar mode or (N, embed_dim) for vector mode.
        """
        if data_loader is None:
            return np.array([])

        self.eval()
        all_preds = []
        
        if loss_mode == "sample":
            for batch in data_loader:
                x = batch["input"].to(self.device)  # [B, max_bins, features]
                mask = batch.get("mask")  # [B, max_bins]

                B, max_bins, n_feat = x.shape
                x_flat = x.view(B * max_bins, n_feat)

                intrinsic_flat = self.mlp(x_flat)  # [B * max_bins, d] or [B * max_bins]
                mask_np = mask.cpu().numpy().astype(bool) if mask is not None else np.ones((B, max_bins), dtype=bool)

                if self.embed_dim > 1:
                    intrinsic_np = intrinsic_flat.view(B, max_bins, self.embed_dim).cpu().numpy()
                    for b in range(B):
                        all_preds.append(intrinsic_np[b, mask_np[b]])  # [n_valid, d]
                else:
                    intrinsic_np = intrinsic_flat.squeeze(-1).view(B, max_bins).cpu().numpy()
                    for b in range(B):
                        all_preds.extend(intrinsic_np[b, mask_np[b]])
            if self.embed_dim > 1:
                return np.concatenate(all_preds, axis=0)
            return np.array(all_preds)
        else:
            for batch in data_loader:
                x = batch["input"].to(self.device)
                intrinsic = self.mlp(x)  # [N, d] or [N]
                if self.embed_dim == 1:
                    intrinsic = intrinsic.squeeze(-1)
                all_preds.append(intrinsic.cpu().numpy())
            return np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    
    
    def save_model(self, path: str) -> None:
        """Save the model state to the specified path."""
        print(f"Saving model to {path}")
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load the model state from the specified path."""
        try:
            self.load_state_dict(torch.load(path, map_location=self.device))
        except RuntimeError as e:
            import logging as log
            log.warning(f"Could not load checkpoint from {path} (architecture mismatch?): {e}")


class NeighbourGraph:
    """
    Build neighbour lists and compute interpolation weights.

    Modes supported:
    - taxonomy-only: discrete taxonomic distance
    - embedding-only: continuous neighbors on embeddings
    - hybrid: taxonomy + embedding ranking

    Neighbor selection modes (cfg.neighbor_mode):
    - "threshold": select all neighbors within a distance threshold
    - "knn": select K nearest neighbors

    Provides functions to compute NW weights and LLR coefficients per node.
    """

    def __init__(self, cfg: Config, bins_df: pd.DataFrame):
        self.cfg = cfg
        self.bins = bins_df.copy().reset_index(drop=True)
        self.n_bins = len(self.bins)

        # taxonomy columns expected: species, genus, subfamily, family, order, class, phylum
        self.tax_levels = ["species", "genus", "subfamily", "family", "order", "class", "phylum", "kingdom"]

        # output structures
        self.neighbours: List[List[int]] = [[] for _ in range(self.n_bins)]
        self.distances: List[np.ndarray] = [np.array([]) for _ in range(self.n_bins)]

        # Load (or compute) embeddings when embedding mode is requested
        self.embeddings: Optional[np.ndarray] = None  # shape [n_bins, emb_dim]
        self.bins_with_embedding: np.ndarray = np.zeros(self.n_bins, dtype=bool)  # True where embedding exists
        if self.cfg.use_embedding:
            self._load_or_compute_embeddings()

    # ------------------------------------------------------------------ #
    # Embedding loading / computation                                      #
    # ------------------------------------------------------------------ #

    def _load_or_compute_embeddings(self) -> None:
        """
        Populate self.embeddings (shape [n_bins, emb_dim]) and self.bins_with_embedding.

        Priority:
            1. Load from cfg.embedding_path if the file exists.
            2. Otherwise compute via BarcodeBERT using cfg.barcode_data_path and save to 
            cfg.embedding_path (if provided) so future runs skip inference.
            3. If neither path is usable, raise a descriptive error.

        Bins with no sequence get a zero vector and bins_with_embedding[i] = False;
        those bins will fall back to taxonomy-based neighbours at build time.
        """
        embedding_path = self.cfg.embedding_path
        barcode_data_path = self.cfg.barcode_data_path

        emb_dict: Dict[str, np.ndarray] = {}  # bin_uri -> np.ndarray

        if embedding_path is not None and os.path.exists(embedding_path):
            log.info(f"Loading precomputed embeddings from {embedding_path}")
            emb_dict = np.load(embedding_path, allow_pickle=True).item()
        elif barcode_data_path is not None:
            log.info(
                f"Precomputed embeddings not found; running BarcodeBERT inference "
                f"on {barcode_data_path}"
            )
            emb_dict = self._compute_barcodebert_embeddings(barcode_data_path)
            # Cache to disk for future runs
            if embedding_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(embedding_path)), exist_ok=True)
                np.save(embedding_path, np.array(emb_dict, dtype=object), allow_pickle=True)
                log.info(f"Saved computed embeddings to {embedding_path}")
        else:
            raise ValueError(
                "use_embedding=True requires at least one of:\n"
                "  cfg.embedding_path  — path to a precomputed .npy embedding file\n"
                "  cfg.barcode_data_path — path to a TSV with 'bin_uri' and 'seq' columns"
            )

        # Determine embedding dimension from first available vector
        emb_dim = next(iter(emb_dict.values())).shape[0]
        self.embeddings = np.zeros((self.n_bins, emb_dim), dtype=np.float32)

        for row_i, row in self.bins.iterrows():
            uri = row["bin_uri"]
            row_idx = int(row_i) # pyright: ignore[reportArgumentType]
            if uri in emb_dict:
                self.embeddings[row_idx] = emb_dict[uri].astype(np.float32)
                self.bins_with_embedding[row_idx] = True

        n_missing = int((~self.bins_with_embedding).sum())
        n_present = int(self.bins_with_embedding.sum())
        log.info(
            f"Embeddings loaded: {n_present}/{self.n_bins} bins have sequences; "
            f"{n_missing} will use taxonomy fallback."
        )

    def _compute_barcodebert_embeddings(
        self,
        barcode_data_path: str,
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Run BarcodeBERT inference on sequences in barcode_data_path and return a
        dict mapping bin_uri -> mean-pooled embedding vector (numpy float32).

        The function uses mean-pooling of the last hidden state across all token
        positions (recommended by the BarcodeBERT authors).

        Args:
            barcode_data_path: Path to TSV/CSV with 'bin_uri' and 'seq' columns.
            batch_size: Number of sequences per inference batch.

        Returns:
            Dict[str, np.ndarray]: {bin_uri: np.ndarray of shape [hidden_dim]}
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError as e:
            raise ImportError(
                "BarcodeBERT inference requires the 'transformers' and 'torch' packages. "
                "Install them with: pip install transformers torch"
            ) from e

        MODEL_NAME = "bioscan-ml/BarcodeBERT"
        log.info(f"Loading BarcodeBERT from HuggingFace ({MODEL_NAME}) ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        device = torch.device(self.cfg.device)
        model = model.to(device).eval()

        # Read data: one consensus sequence per BIN
        sep = "\t" if barcode_data_path.endswith(".tsv") else ","
        df = pd.read_csv(barcode_data_path, sep=sep)
        if "bin_uri" not in df.columns or "seq" not in df.columns:
            raise ValueError(
                f"{barcode_data_path} must contain 'bin_uri' and 'seq' columns. "
                f"Found: {list(df.columns)}"
            )

        # Aggregate: take the first (consensus) sequence per BIN
        bin_seqs = df.groupby("bin_uri")["seq"].first().to_dict()
        uris = list(bin_seqs.keys())
        sequences = [bin_seqs[u] for u in uris]

        log.info(f"Running BarcodeBERT inference on {len(sequences)} BINs (batch_size={batch_size}) ...")

        emb_dict: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch_seqs = sequences[start : start + batch_size]
                batch_uris = uris[start : start + batch_size]

                # KmerTokenizer is single-sequence only: encode each one individually
                # and stack — safe because padding makes all outputs the same length
                batch_input_ids = []
                batch_attention_mask = []
                for seq in batch_seqs:
                    encoded = tokenizer(seq, padding=True)
                    batch_input_ids.append(encoded["input_ids"])
                    batch_attention_mask.append(encoded["attention_mask"])

                input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
                attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # last_hidden_state: [B, seq_len, hidden_dim]
                last_hidden = outputs.last_hidden_state
                # Mean-pool over non-padding token positions
                mask_exp = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
                sum_hidden = (last_hidden * mask_exp).sum(dim=1) # [B, hidden_dim]
                count = mask_exp.sum(dim=1).clamp(min=1e-9)      # [B, 1]
                mean_pooled = (sum_hidden / count).cpu().numpy()  # [B, hidden_dim]

                for uri, emb in zip(batch_uris, mean_pooled):
                    emb_dict[str(uri)] = emb.astype(np.float32)

                if (start // batch_size) % 10 == 0:
                    log.debug(f"  Processed {start + len(batch_seqs)}/{len(sequences)} sequences")

        log.info("BarcodeBERT inference complete.")
        return emb_dict
    
    def build_taxonomy_neighbors_knn(self, K: int) -> None:
        """
        Populate neighbours using taxonomic discrete distance with KNN.
        Selects the K nearest neighbors by taxonomy.
        
        Args:
            K: Number of nearest neighbors to select.
        """
        from tqdm import tqdm
        if "bin_uri" not in self.bins.columns:
            raise ValueError("bins dataframe must contain 'bin_uri'")
        
        log.debug(f"Building taxonomy neighbors (KNN mode) for {self.n_bins} bins with K={K}...")
        
        # Encode taxonomy levels as integers for fast comparison
        tax_groups = {}  # level -> {value -> list of bin indices}
        tax_codes = {}   # level -> array of codes for each bin
        
        for level in self.tax_levels:
            if level not in self.bins.columns:
                tax_codes[level] = np.full(self.n_bins, -1, dtype=np.int32)
                tax_groups[level] = {}
                continue
            
            codes, uniques = pd.factorize(self.bins[level], sort=False)
            tax_codes[level] = codes
            
            groups = {}
            for idx, code in enumerate(codes):
                if code >= 0:
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(idx)
            tax_groups[level] = groups
        
        # For each bin, find K nearest neighbors by taxonomy
        for i in tqdm(range(self.n_bins), desc="Building neighbors (KNN)"):
            candidates = []  # list of (distance, bin_idx)
            seen = {i}  # exclude self
            
            for d, level in enumerate(self.tax_levels, start=1):
                code = tax_codes[level][i]
                if code < 0:
                    continue
                
                same_group = tax_groups[level].get(code, [])
                for j in same_group:
                    if j not in seen:
                        candidates.append((d, j))
                        seen.add(j)
            
            # If still not enough, add remaining bins with max distance
            if len(candidates) < K:
                for j in range(self.n_bins):
                    if j not in seen:
                        candidates.append((len(self.tax_levels) + 1, j))
                        if len(candidates) >= K:
                            break
            
            # Sort by distance and take top K
            candidates.sort(key=lambda x: x[0])
            top_k = candidates[:K]
            
            self.neighbours[i] = [x[1] for x in top_k]
            self.distances[i] = np.array([x[0] for x in top_k], dtype=float)
        
        # Log statistics
        neighbor_counts = [len(n) for n in self.neighbours]
        log.debug(
            f"Neighbor count stats: min={min(neighbor_counts)}, "
            f"max={max(neighbor_counts)}, mean={np.mean(neighbor_counts):.1f}")

    def build_taxonomy_neighbors_threshold(self, dist_threshold: int) -> None:
        """
        Populate neighbours using taxonomic discrete distance with a threshold.
        Selects all neighbors within the given taxonomic distance threshold.
        
        Args:
            dist_threshold: Maximum taxonomic distance to include as neighbor.
                Distance 1 = same species, 2 = same genus, etc.
        """
        from tqdm import tqdm
        if "bin_uri" not in self.bins.columns:
            raise ValueError("bins dataframe must contain 'bin_uri'")
        
        log.debug(f"Building taxonomy neighbors for {self.n_bins} bins with threshold {dist_threshold}...")
        
        # Encode taxonomy levels as integers for fast comparison
        # Build lookup tables for each level: bins grouped by their taxonomy value
        tax_groups = {}  # level -> {value -> list of bin indices}
        tax_codes = {}   # level -> array of codes for each bin
        
        for level in self.tax_levels:
            if level not in self.bins.columns:
                # Use -1 as missing-taxonomy sentinel, matching pandas.factorize NaN behavior.
                tax_codes[level] = np.full(self.n_bins, -1, dtype=np.int32)
                tax_groups[level] = {}
                continue
            
            # Factorize the taxonomy column (converts to integer codes, NaN -> -1)
            codes, uniques = pd.factorize(self.bins[level], sort=False)
            tax_codes[level] = codes  # np array of int codes, same length as n_bins
            
            # Build reverse lookup: code -> list of bin indices
            groups = {}
            for idx, code in enumerate(codes):
                if code >= 0:  # skip NaN values
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(idx)
            tax_groups[level] = groups  # {tax_code: [list of bin indices with this tax_code]}
        
        # For each bin, find all neighbors within the distance threshold
        # Strategy: iterate through taxonomy levels up to dist_threshold,
        # collecting all bins that share taxonomy at each level
        
        for i in tqdm(range(self.n_bins), desc="Building neighbors"):
            candidates = []  # list of (distance, bin_idx)
            seen = {i}  # exclude self
            
            # Only iterate through levels up to dist_threshold
            for d, level in enumerate(self.tax_levels[:dist_threshold], start=1):
                code = tax_codes[level][i]
                if code < 0:  # NaN at this level
                    continue
                
                # Get all bins with same taxonomy value at this level
                same_group = tax_groups[level].get(code, [])
                for j in same_group:
                    if j not in seen:
                        candidates.append((d, j))
                        seen.add(j)
            
            # Sort by distance (closest first)
            candidates.sort(key=lambda x: x[0])
            
            self.neighbours[i] = [x[1] for x in candidates]
            self.distances[i] = np.array([x[0] for x in candidates], dtype=float)
        
        # Log statistics about neighbor counts
        neighbor_counts = [len(n) for n in self.neighbours]
        log.debug(
            f"Neighbor count stats: min={min(neighbor_counts)}, "
            f"max={max(neighbor_counts)}, mean={np.mean(neighbor_counts):.1f}, "
            f"median={np.median(neighbor_counts):.1f}"
        )

    def analyze_taxonomy_thresholds(self) -> pd.DataFrame:
        """
        Analyze neighbor counts for each possible taxonomic distance threshold.
        
        Returns a DataFrame with statistics (min, max, mean, std) for each threshold,
        helping you choose an appropriate dist_thres value.
        
        Returns:
            pd.DataFrame with columns: threshold, level, min, max, mean, std, median, pct_zero (percentage of bins with 0 neighbors)
        """
        from tqdm import tqdm
        
        if "bin_uri" not in self.bins.columns:
            raise ValueError("bins dataframe must contain 'bin_uri'")
        
        # Encode taxonomy levels
        tax_groups = {}
        tax_codes = {}
        
        for level in self.tax_levels:
            if level not in self.bins.columns:
                tax_codes[level] = np.full(self.n_bins, -1, dtype=np.int32)
                tax_groups[level] = {}
                continue
            
            codes, uniques = pd.factorize(self.bins[level], sort=False)
            tax_codes[level] = codes
            
            groups = {}
            for idx, code in enumerate(codes):
                if code >= 0:
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(idx)
            tax_groups[level] = groups
        
        # Compute cumulative neighbor counts for each threshold
        results = []
        
        for threshold in range(1, len(self.tax_levels) + 1):
            level_name = self.tax_levels[threshold - 1]
            neighbor_counts = []
            
            for i in tqdm(range(self.n_bins), desc=f"Analyzing threshold {threshold} ({level_name})", leave=False):
                seen = {i}
                count = 0
                
                for d, level in enumerate(self.tax_levels[:threshold], start=1):
                    code = tax_codes[level][i]
                    if code < 0:
                        continue
                    
                    same_group = tax_groups[level].get(code, [])
                    for j in same_group:
                        if j not in seen:
                            count += 1
                            seen.add(j)
                
                neighbor_counts.append(count)
            
            counts_arr = np.array(neighbor_counts)
            
            # Calculate percentage of missing values at this level
            has_taxonomy = tax_codes[level_name] >= 0
            pct_missing = round(100 * (~has_taxonomy).sum() / self.n_bins, 2)
            
            # Min excluding bins with missing taxonomy at this level
            counts_with_tax = counts_arr[has_taxonomy]
            min_val = int(counts_with_tax.min()) if len(counts_with_tax) > 0 else None
            
            results.append({
                'threshold': threshold,
                'level': level_name,
                'pct_missing': pct_missing,
                'min': min_val,
                'max': int(counts_arr.max()),
                'mean': round(counts_arr.mean(), 2),
                'std': round(counts_arr.std(), 2),
                'median': round(np.median(counts_arr), 1),
                'pct_zero': round(100 * (counts_arr == 0).sum() / len(counts_arr), 2)
            })
        
        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("TAXONOMY DISTANCE THRESHOLD ANALYSIS")
        print("=" * 80)
        print(f"\nTotal bins: {self.n_bins}")
        print(f"Taxonomy levels: {self.tax_levels}\n")
        print(df.to_string(index=False))
        print("\n" + "=" * 80)
        print("Interpretation:")
        print("  - threshold 1 (species): neighbors share the same species")
        print("  - threshold 2 (genus): neighbors share species OR genus")
        print("  - threshold 3 (subfamily): neighbors share species OR genus OR subfamily")
        print("  - etc.")
        print("  - pct_missing: percentage of bins with missing taxonomy at this level")
        print("  - pct_zero: percentage of bins with NO neighbors at this threshold")
        print("=" * 80 + "\n")
        
        return df

    def _build_taxonomy_neighbors_for_subset(self, subset_indices: np.ndarray, K: int) -> None:
        """
        Compute KNN taxonomy neighbors for a specific subset of bin indices and write
        the results into self.neighbours / self.distances only for those indices.
        Used as a fallback for bins that lack DNA sequences.

        Args:
            subset_indices: Array of global bin indices that need taxonomy neighbors.
            K: Number of nearest neighbors to select.
        """
        from tqdm import tqdm

        # Build (or reuse) taxonomy code maps
        tax_codes: dict = {}
        tax_groups: dict = {}
        for level in self.tax_levels:
            if level not in self.bins.columns:
                tax_codes[level] = np.full(self.n_bins, -1, dtype=np.int32)
                tax_groups[level] = {}
                continue
            codes, _ = pd.factorize(self.bins[level], sort=False)
            tax_codes[level] = codes
            groups: dict = {}
            for idx, code in enumerate(codes):
                if code >= 0:
                    groups.setdefault(code, []).append(idx)
            tax_groups[level] = groups

        for i in tqdm(subset_indices, desc="Taxonomy fallback for bins without sequences", leave=False):
            candidates = []
            seen = {int(i)}
            for d, level in enumerate(self.tax_levels, start=1):
                code = tax_codes[level][i]
                if code < 0:
                    continue
                for j in tax_groups[level].get(code, []):
                    if j not in seen:
                        candidates.append((d, j))
                        seen.add(j)
            # If still not enough, pad with remaining bins
            if len(candidates) < K:
                for j in range(self.n_bins):
                    if j not in seen:
                        candidates.append((len(self.tax_levels) + 1, j))
                        seen.add(j)
                        if len(candidates) >= K:
                            break
            candidates.sort(key=lambda x: x[0])
            top_k = candidates[:K]
            self.neighbours[i] = [x[1] for x in top_k]
            self.distances[i] = np.array([float(x[0]) for x in top_k], dtype=float)

    def build_embedding_neighbors_knn(self, K: int) -> None:
        """
        Build KNN based on embeddings using sklearn.NearestNeighbors.

        Cosine distance is computed by L2-normalizing embeddings before Euclidean
        nearest-neighbor search.
        Bins without embeddings fall back to taxonomy-based KNN.

        Note:
        - For bins with embeddings, exactly min(K, n_with_embeddings - 1) neighbors are
            selected after removing self from the query result.
        - For bins without embeddings, neighbors are filled by taxonomy fallback.

        Args:
            K: Number of nearest neighbors to select.
        """
        if self.embeddings is None:
            raise ValueError("embeddings not available — call _load_or_compute_embeddings() first")

        emb_indices = np.where(self.bins_with_embedding)[0]
        fallback_indices = np.where(~self.bins_with_embedding)[0]

        if len(emb_indices) == 0:
            raise ValueError(
                "No bins have embeddings. Check that bin_uri values in cfg.embedding_path "
                "match those in the training data."
            )

        # Optionally L2-normalise for cosine distance
        emb = self.embeddings.copy()
        if getattr(self.cfg, "emb_distance_metric", "cosine") == "cosine":
            emb = normalize(emb, norm="l2")

        # Fit NearestNeighbors on the subset that actually has sequences
        emb_subset = emb[emb_indices]  # [n_with_emb, dim]
        n_neighbors_query = min(K + 1, len(emb_indices))  # +1 to exclude self in the result
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_query, algorithm="auto", metric="euclidean")
        nbrs.fit(emb_subset)

        # Query from the same subset (no cross-queries; fallback bins handled separately)
        distances, local_indices = nbrs.kneighbors(emb_subset)

        for rank, global_i in enumerate(emb_indices):
            # local_indices[rank, 0] == rank (self) → skip first column
            neighbor_local = local_indices[rank, 1:]
            neighbor_global = emb_indices[neighbor_local]
            self.neighbours[global_i] = neighbor_global.tolist()
            self.distances[global_i] = distances[rank, 1:]

        log.debug(
            f"Built embedding neighbors (KNN, K={K}, metric={self.cfg.emb_distance_metric}) "
            f"for {len(emb_indices)} bins."
        )

        # Taxonomy fallback for bins with no sequence
        if len(fallback_indices) > 0:
            log.debug(f"Running taxonomy fallback for {len(fallback_indices)} bins without sequences.")
            self._build_taxonomy_neighbors_for_subset(fallback_indices, K=K)

    def build_embedding_neighbors_threshold(self, radius: float) -> None:
        """
        Build radius-based neighbors using embeddings (BallTree).

        Cosine distance is computed by L2-normalizing embeddings before a Euclidean
        radius query (d_euclidean on unit sphere = sqrt(2 - 2*cos) ≈ cosine distance).
        Bins without embeddings fall back to taxonomy-based threshold neighbors.

        Args:
            radius: Maximum embedding distance to include as neighbor.
        """
        if self.embeddings is None:
            raise ValueError("embeddings not available — call _load_or_compute_embeddings() first")

        emb_indices = np.where(self.bins_with_embedding)[0]
        fallback_indices = np.where(~self.bins_with_embedding)[0]

        if len(emb_indices) == 0:
            raise ValueError(
                "No bins have embeddings. Check that bin_uri values in cfg.embedding_path "
                "match those in the training data."
            )

        # Optionally L2-normalise for cosine distance
        emb = self.embeddings.copy()
        if getattr(self.cfg, "emb_distance_metric", "cosine") == "cosine":
            emb = normalize(emb, norm="l2")

        emb_subset = emb[emb_indices]
        tree = BallTree(emb_subset, metric="euclidean")
        local_indices_arr, distances_arr = tree.query_radius(
            emb_subset, r=radius, return_distance=True, sort_results=True
        )

        for rank, global_i in enumerate(emb_indices):
            local_nbrs = local_indices_arr[rank]
            dists = distances_arr[rank]
            # Remove self (distance 0)
            not_self = local_nbrs != rank
            self.neighbours[global_i] = emb_indices[local_nbrs[not_self]].tolist()
            self.distances[global_i] = dists[not_self]

        neighbor_counts = [len(self.neighbours[i]) for i in emb_indices]
        log.debug(
            f"Embedding threshold neighbors (radius={radius}, metric={self.cfg.emb_distance_metric}): "
            f"min={min(neighbor_counts)}, max={max(neighbor_counts)}, "
            f"mean={np.mean(neighbor_counts):.1f}, median={np.median(neighbor_counts):.1f}"
        )

        # Taxonomy fallback for bins with no sequence
        if len(fallback_indices) > 0:
            log.debug(f"Running taxonomy fallback for {len(fallback_indices)} bins without sequences.")
            self._build_taxonomy_neighbors_for_subset(
                fallback_indices, K=self.cfg.K  # use K as a reasonable default for fallback
            )

    def build_hybrid_neighbors(self, tax_threshold: int, emb_radius: float, K: int) -> None:
        """Build neighbors using both taxonomy and embedding distances."""
        raise NotImplementedError("Hybrid neighbor graph not yet implemented")

    def build(self) -> None:
        """
        Build the neighbor graph based on cfg.neighbor_mode.
        
        Modes:
        - "threshold": uses cfg.dist_thres (taxonomy) or cfg.emb_radius (embedding)
        - "knn": uses cfg.K for K-nearest neighbors
        """
        mode = getattr(self.cfg, 'neighbor_mode', 'threshold')
        
        if self.cfg.use_taxonomy and not self.cfg.use_embedding:
            if mode == "knn":
                self.build_taxonomy_neighbors_knn(self.cfg.K)
            else:
                self.build_taxonomy_neighbors_threshold(self.cfg.dist_thres)
        elif self.cfg.use_embedding and not self.cfg.use_taxonomy:
            if mode == "knn":
                self.build_embedding_neighbors_knn(self.cfg.K)
            else:
                self.build_embedding_neighbors_threshold(self.cfg.emb_radius)
        else:
            # Hybrid mode
            self.build_hybrid_neighbors(self.cfg.dist_thres, self.cfg.emb_radius, self.cfg.K)

    # --------- kernel weights (Nadaraya-Watson)
    def compute_kernel_q(self) -> float:
        """Compute an adaptive q if not provided: q = 1 / median(dist_to_Kth_neighbor^2)
        Works for continuous distances; for discrete taxonomic distances will return 1.
        """
        if self.cfg.kernel_q is not None:
            return self.cfg.kernel_q
        # use median of last neighbour distance
        last_dists = np.array([d[-1] if len(d) > 0 else 1.0 for d in self.distances])
        med = np.median(last_dists)
        if med <= 0:
            return 1.0
        # Floor med to prevent q = 1/med² from overflowing exp(-q·dist²) when
        # distances are very small (e.g. near-identical sequences).
        med = max(med, 1e-6)
        return 1.0 / (med ** 2)

    def nw_weights_for_node(self, i: int, q: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return neighbor indices and kernel-normalized weights for node i.
        Returns (indices_array, weights_array).
        """
        idx = np.array(self.neighbours[i])
        if len(idx) == 0:
            return idx, np.array([])
        dists = np.array(self.distances[i], dtype=float)
        if q is None:
            q = self.compute_kernel_q()
        w_unnorm = np.exp(-q * (dists ** 2))
        w = w_unnorm / (w_unnorm.sum() + 1e-12)
        return idx, w

    # -------- LLR coefficients per node
    def llr_coeffs_for_node(self, i: int, q: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return local-linear interpolation coefficients for node i.

        Returns (indices, coeffs) such that h_i ≈ sum_j coeffs[j] * h_j over neighbors.
        Coeffs are derived from weighted least squares on local offsets X_j - X_i and
        correspond to the intercept row e0^T (Z^T W Z)^(-1) Z^T W.

        If the local normal matrix is ill-conditioned, this falls back to NW weights.
        """
        idx = np.array(self.neighbours[i])
        if len(idx) == 0 or self.embeddings is None:
            return idx, np.array([])
        X = self.embeddings[idx] - self.embeddings[i]
        # build design Z = [1 | X]
        ones = np.ones((len(idx), 1))
        Z = np.hstack([ones, X])
        if q is None:
            q = self.compute_kernel_q()
        dists = np.linalg.norm(self.embeddings[idx] - self.embeddings[i], axis=1)
        w = np.exp(-q * (dists ** 2))
        W = np.diag(w)
        # solve for beta = (Z^T W Z)^{-1} Z^T W d  ; but we only need intercept effect on d_j
        # The interpolation matrix row that maps d -> a_i is: coeffs = e_0^T (Z^T W Z)^{-1} Z^T W
        # We'll compute coeffs row explicitly by solving (Z^T W Z)^T x = e_0 -> x obtains coefficients
        A = Z.T @ W @ Z
        try:
            invA = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # ill-conditioned -> fallback to NW weights
            idx_nw, w_nw = self.nw_weights_for_node(i, q)
            return idx_nw, w_nw
        # row vector: e0^T invA Z^T W  (shape 1 x n_idx)
        e0 = np.zeros((A.shape[0],))
        e0[0] = 1.0
        row = e0 @ invA @ Z.T @ W
        return idx, row


class Trainer:
    """
    Trains MLP and latent embedding jointly in a single forward-backward pass per epoch.
    """
    def __init__(
        self,
        cfg: Config,
        model_name: str = "default",
        run_id: Optional[str] = None,
        resume: bool = False,
        fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H-%M-%S")
        self.resume = resume
        self._validate_interpolation_config()

        if self.cfg.use_embedding and self.cfg.barcode_data_path is None and (
            self.cfg.embedding_path is None or not os.path.exists(self.cfg.embedding_path)
        ):
            self.cfg.barcode_data_path = self.cfg.data_path

        self.start_epoch = 0
        self.current_epoch = -1
        self.best_val_loss = float("inf")
        self.best_val_kl = float("inf")   # primary autoresearch objective
        self.last_val_metrics: Dict[str, float] = {}

        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

        self.base_artifact_dir = os.path.abspath(os.path.join(self.cfg.results_dir, self.model_name))
        self.checkpoint_dir = os.path.join(self.base_artifact_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        data, bins_df, bin_index, sample_index, split_indices = load_or_preprocess(
            self.cfg,
            fixed_split_indices=fixed_split_indices,
        )

        self.data = data
        self.bin_index = bin_index
        self.sample_index = sample_index
        self.split_indices = split_indices

        self.neighbour_graph = NeighbourGraph(self.cfg, bins_df)
        self.neighbour_graph.build()

        latent_solver = LatentSolver(
            self.cfg,
            self.neighbour_graph,
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
        )
        latent_solver.build_interpolation_matrix()

        self.device = torch.device(self.cfg.device)
        input_dim = data["train"]["X"].shape[1]
        mlp_model = MLPModel(
            input_dim,
            hidden_dims=self.cfg.mlp_hidden_dims,
            output_dim=self.cfg.embed_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

        self.model = Model(
            mlp_model,
            latent_solver,
            n_bins=len(bin_index),
            device=self.device,
            latent_init_std=self.cfg.latent_init_std,
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
            gating_alpha=self.cfg.gating_alpha,
            gating_kappa=self.cfg.gating_kappa,
            gating_epsilon=self.cfg.gating_epsilon,
            interpolation_enabled=bool(self.cfg.train_MLP_with_interpolation or self.cfg.inference_with_interpolation),
        )
        self.model.latent_vec.requires_grad_(False) # latent is optimized separately, not by the main optimizer

        if self.cfg.embed_dim > 1:
            optim_params = [
                {"params": self.model.mlp.parameters(), "weight_decay": self.cfg.weight_decay},
                {"params": self.model.final_linear.parameters(), "weight_decay": self.cfg.final_linear_weight_decay},
            ]
        else:
            optim_params = [
                {"params": self.model.mlp.parameters(), "weight_decay": self.cfg.weight_decay},
            ]
        self.mlp_optimizer = torch.optim.AdamW(optim_params, lr=self.cfg.mlp_lr)
        self.latent_optimizer = torch.optim.AdamW([self.model.latent_vec], lr=self.cfg.latent_lr)

        self.loss_type: Literal["cross_entropy", "logistic"] = self.cfg.loss_type
        self.loss_mode = "sample" if self.loss_type == "cross_entropy" else "bin"
        self.criterion = Loss(task=self.loss_type)

        train = MBDataset(data["train"], bin_index, sample_index, loss_mode=self.loss_mode)
        val = MBDataset(data["val"], bin_index, sample_index, loss_mode=self.loss_mode)
        test = MBDataset(data["test"], bin_index, sample_index, loss_mode=self.loss_mode)
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self._train_sample_ids = np.unique(train.sample_ids).astype(np.int64, copy=False)
        self._epoch_selected_sample_ids = np.empty(0, dtype=np.int64)
        self._epoch_selected_sample_ids_t = torch.empty(0, dtype=torch.long)

        batch_size = self.cfg.batch_size_sample if self.loss_mode == "sample" else self.cfg.batch_size_bin
        collate_fn = collate_samples if self.loss_mode == "sample" else None
        num_workers = int(getattr(self.cfg, "num_workers", 0 if sys.platform == "darwin" else 8))
        pin_memory = bool(getattr(self.cfg, "pin_memory", self.device.type == "cuda"))
        args = {
            "batch_size": batch_size, "collate_fn": collate_fn, "num_workers": num_workers, "pin_memory": pin_memory
        }

        self.train_loader = DataLoader(train, shuffle=True, drop_last=True, **args)
        self.val_loader = DataLoader(val, shuffle=False, **args)
        self.test_loader = DataLoader(test, shuffle=False, **args)

        total_steps = max(1, self.cfg.epochs * len(self.train_loader))
        warmup_steps = max(1, int(self.cfg.mlp_warmup_frac * total_steps))
        warmup_scheduler = LinearLR(self.mlp_optimizer, start_factor=self.cfg.mlp_warmup_start_factor, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.mlp_optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=self.cfg.mlp_lr_eta_min)
        self.mlp_scheduler = SequentialLR(self.mlp_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        total_steps = max(1, self.cfg.epochs * len(self.train_loader) * self.cfg.latent_optim_steps)
        warmup_steps = max(1, int(self.cfg.latent_warmup_frac * total_steps))
        warmup_scheduler = LinearLR(self.latent_optimizer, start_factor=self.cfg.latent_warmup_start_factor, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.latent_optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=self.cfg.latent_lr_eta_min)
        self.latent_scheduler = SequentialLR(self.latent_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        self.latent_diagnostics: List[Dict[str, Any]] = []

        if self.resume:
            self._resume_from_latest()

    def _validate_interpolation_config(self) -> None:
        fraction = float(self.cfg.interpolated_sample_fraction)
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("interpolated_sample_fraction must be in [0, 1]")

    def _select_epoch_interpolated_samples(self) -> np.ndarray:
        fraction = float(self.cfg.interpolated_sample_fraction)
        if fraction <= 0.0 or self._train_sample_ids.size == 0:
            return np.empty(0, dtype=np.int64)

        n_select = min(self._train_sample_ids.size, int(np.ceil(fraction * self._train_sample_ids.size)))
        if n_select <= 0:
            return np.empty(0, dtype=np.int64)

        selected = np.random.choice(self._train_sample_ids, size=n_select, replace=False)
        selected.sort()
        return selected.astype(np.int64, copy=False)

    def _refresh_epoch_interpolation_selection(self) -> None:
        self._epoch_selected_sample_ids = self._select_epoch_interpolated_samples()
        self._epoch_selected_sample_ids_t = torch.as_tensor(
            self._epoch_selected_sample_ids,
            dtype=torch.long,
            device=self.device,
        )

    def _epoch_sample_selection_mask(self, sample_idx: torch.Tensor) -> Optional[torch.Tensor]:
        """Return per-sample boolean mask for this epoch's interpolation subset.

        The returned mask has the same first dimension as sample_idx and is used to
        decide where interpolated latents are enabled during training/latent solves.
        """
        if self._epoch_selected_sample_ids_t.numel() == 0:
            return None
        sample_idx_np = sample_idx.detach().cpu().numpy().reshape(-1)
        selected_np = np.isin(sample_idx_np, self._epoch_selected_sample_ids)
        return torch.as_tensor(selected_np, dtype=torch.bool, device=sample_idx.device)

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self.checkpoint_dir, name)

    def _build_checkpoint_payload(self, epoch: int, val_loss: float, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "run_id": self.run_id,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_kl": self.best_val_kl,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "config": asdict(self.cfg),
            "model_state_dict": self.model.state_dict(),
            "mlp_optimizer_state_dict": self.mlp_optimizer.state_dict(),
            "mlp_scheduler_state_dict": self.mlp_scheduler.state_dict(),
            "latent_optimizer_state_dict": self.latent_optimizer.state_dict(),
            "latent_scheduler_state_dict": self.latent_scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "latent_diagnostics": self.latent_diagnostics,
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "saved_at": time.strftime("%Y-%m-%d_%H-%M-%S"),
        }

    def _save_checkpoint(self, epoch: int, val_loss: float, val_metrics: Dict[str, float], best: bool = False) -> None:
        payload = self._build_checkpoint_payload(epoch=epoch, val_loss=val_loss, val_metrics=val_metrics)

        latest_path = self._checkpoint_path("latest.pt")
        torch.save(payload, latest_path)

        if (epoch + 1) % self.cfg.checkpoint_every == 0:
            periodic_name = f"epoch_{epoch + 1:04d}.pt"
            torch.save(payload, self._checkpoint_path(periodic_name))

        if best:
            torch.save(payload, self._checkpoint_path("best.pt"))

    def _find_latest_checkpoint_path(self) -> Optional[str]:
        latest = self._checkpoint_path("latest.pt")
        if os.path.exists(latest):
            return latest
        if not os.path.exists(self.checkpoint_dir):
            return None
        ckpts = [
            os.path.join(self.checkpoint_dir, p)
            for p in os.listdir(self.checkpoint_dir)
            if p.endswith(".pt")
        ]
        if not ckpts:
            return None
        ckpts.sort(key=os.path.getmtime, reverse=True)
        return ckpts[0]

    def _resume_from_latest(self) -> None:
        ckpt_path = self._find_latest_checkpoint_path()
        if ckpt_path is None:
            log.warning("Resume requested but no checkpoint was found. Starting from epoch 0.")
            return

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        saved_cfg = checkpoint.get("config", {})
        for key in ["embed_dim", "gating_fn", "loss_type"]:
            if key in saved_cfg and getattr(self.cfg, key) != saved_cfg[key]:
                raise ValueError(
                    f"Checkpoint/config mismatch for '{key}': "
                    f"checkpoint={saved_cfg[key]} current={getattr(self.cfg, key)}"
                )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mlp_optimizer.load_state_dict(checkpoint["mlp_optimizer_state_dict"])
        self.mlp_scheduler.load_state_dict(checkpoint["mlp_scheduler_state_dict"])
        self.latent_optimizer.load_state_dict(checkpoint["latent_optimizer_state_dict"])
        self.latent_scheduler.load_state_dict(checkpoint["latent_scheduler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", -1))
        self.start_epoch = self.current_epoch + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.best_val_kl = float(checkpoint.get("best_val_kl", float("inf")))
        self.train_losses = list(checkpoint.get("train_losses", []))
        self.val_losses = list(checkpoint.get("val_losses", []))
        self.last_val_metrics = dict(checkpoint.get("val_metrics", {}))
        self.latent_diagnostics = list(checkpoint.get("latent_diagnostics", []))

        rng_numpy = checkpoint.get("rng_numpy")
        rng_torch = checkpoint.get("rng_torch")
        rng_cuda = checkpoint.get("rng_cuda")
        if rng_numpy is not None:
            np.random.set_state(rng_numpy)
        if rng_torch is not None:
            torch.set_rng_state(rng_torch)
        if rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_cuda)

        log.info(f"Resumed from checkpoint: {ckpt_path} (epoch {self.current_epoch})")

    @staticmethod
    def _has_interpolation_samples(sample_selection: Optional[torch.Tensor]) -> bool:
        """Return True iff sample_selection is non-None and contains at least one True entry."""
        if sample_selection is None or sample_selection.numel() == 0:
            return False
        return bool(torch.any(sample_selection).item())

    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Run one MLP optimization step and return loss plus timing breakdown.

        In sample mode, inputs are flattened to observation level, then reshaped back
        to [batch_size, max_bins] for masked cross-entropy. In bin mode, observations
        are already independent and no reshape is needed.
        """
        self.model.train()
        inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
        sample_selection = self._epoch_sample_selection_mask(sample_idx)

        t0 = time.perf_counter()
        if self.loss_mode == "sample":
            bsz, max_bins, n_feat = inputs.shape
            # Flatten keeps sample order, so bin/sample alignment is preserved.
            inputs_flat = inputs.view(bsz * max_bins, n_feat)
            bin_idx_flat = bin_idx.view(bsz * max_bins)
            interpolation_mask = None
            if self.cfg.train_MLP_with_interpolation and self._has_interpolation_samples(sample_selection):
                # Expand sample-level selection to observation-level and drop padded bins.
                interpolation_mask = sample_selection.unsqueeze(1).expand(-1, max_bins)
                if mask is not None:
                    interpolation_mask = interpolation_mask & mask.bool()
                interpolation_mask = interpolation_mask.reshape(-1)
            outputs_flat = self.model(
                inputs_flat,
                bin_idx_flat,
                interpolation_mask=interpolation_mask,
            )
            outputs = outputs_flat.view(bsz, max_bins)
            # Padded positions must be -inf so softmax contributes exactly zero mass.
            outputs = outputs.masked_fill(mask == 0, float("-inf"))
            loss = self.criterion(outputs, targets, mask)
        else:
            interpolation_mask = None
            if self.cfg.train_MLP_with_interpolation and self._has_interpolation_samples(sample_selection):
                interpolation_mask = sample_selection
            outputs = self.model(
                inputs,
                bin_idx,
                interpolation_mask=interpolation_mask,
            )
            loss = self.criterion(outputs, targets)
        t1 = time.perf_counter()

        self.mlp_optimizer.zero_grad()
        loss.backward()
        t2 = time.perf_counter()

        if self.cfg.grad_clip is not None:
            params_to_clip = list(self.model.mlp.parameters())
            if self.cfg.embed_dim > 1:
                params_to_clip += list(self.model.final_linear.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.grad_clip)

        self.mlp_optimizer.step()
        t3 = time.perf_counter()

        timing = {
            "forward_s": float(t1 - t0),
            "backward_s": float(t2 - t1),
            "optim_s": float(t3 - t2),
            "total_s": float(t3 - t0),
        }
        return float(loss.item()), timing

    @torch.no_grad()
    def validate(self, split: Literal["train", "val", "test"]) -> float:
        data_loader = (
            self.train_loader if split == "train" else
            self.val_loader if split == "val" else
            self.test_loader
        )

        self.model.eval()
        use_interpolation = bool(self.cfg.inference_with_interpolation)
        running_loss = 0.0
        n_samples = 0
        for batch in data_loader:
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
            if self.loss_mode == "sample":
                bsz, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(bsz * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(bsz * max_bins)
                interpolation_mask = mask.bool().view(-1) if use_interpolation and mask is not None else None
                outputs_flat = self.model(
                    inputs_flat,
                    bin_idx_flat,
                    interpolation_mask=interpolation_mask,
                )
                outputs = outputs_flat.view(bsz, max_bins)
                outputs = outputs.masked_fill(mask == 0, float("-inf"))
                loss = self.criterion(outputs, targets, mask)
            else:
                interpolation_mask = torch.ones_like(sample_idx, dtype=torch.bool) if use_interpolation else None
                outputs = self.model(
                    inputs,
                    bin_idx,
                    interpolation_mask=interpolation_mask,
                )
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return float(running_loss / max(1, n_samples))

    def solve_latent(
        self,
        batch: Dict[str, torch.Tensor],
        prox_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run one latent-variable solve for the current batch.

        The latent solver consumes torch tensors directly on its configured device.
        This method temporarily enables gradient tracking for latent_vec only, then restores its
        previous requires_grad state before returning.
        """
        self.model.eval()

        with torch.no_grad():
            inputs, targets, bin_ids, sample_ids, mask = self._to_device(batch)
            sample_selection = self._epoch_sample_selection_mask(sample_ids)

            if self.loss_mode == "sample":
                bsz, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(bsz * max_bins, n_feat)
                intrinsic = self.model.mlp(inputs_flat).view(bsz, max_bins, self.cfg.embed_dim)

                sample_grid = sample_ids.unsqueeze(1).expand(-1, max_bins)
                valid = mask.bool() if mask is not None else torch.ones_like(bin_ids, dtype=torch.bool)
                interpolation_mask = None
                if self._has_interpolation_samples(sample_selection):
                    interpolation_mask = sample_selection.unsqueeze(1).expand(-1, max_bins)
                    interpolation_mask = interpolation_mask & valid
                    interpolation_mask = interpolation_mask[valid]

                intrinsic = intrinsic[valid]
                y = targets[valid]
                bin_ids = bin_ids[valid]
                sample_ids = sample_grid[valid]
            else:
                intrinsic = self.model.mlp(inputs)
                interpolation_mask = None
                if self._has_interpolation_samples(sample_selection):
                    interpolation_mask = sample_selection.to(dtype=torch.bool)

        previous_requires_grad = self.model.latent_vec.requires_grad
        # Invariant: latent is optimized only in this block, not in _train_batch.
        self.model.latent_vec.requires_grad_(True)

        try:
            latent_vec, timings = self.model.latent_solver.solve(
                y=y,
                intrinsic=intrinsic,
                final_weights=self.model.final_linear.weight if self.cfg.embed_dim > 1 else None,
                bin_ids=bin_ids,
                sample_ids=sample_ids,
                interpolation_mask=interpolation_mask,
                loss_type=self.loss_type,
                prox_weight=prox_weight,
                latent=self.model.latent_vec,
                optimizer=self.latent_optimizer,
            )
            self.latent_scheduler.step()
        finally:
            self.model.latent_vec.requires_grad_(previous_requires_grad)

        return latent_vec, timings

    @torch.no_grad()
    def get_predictions(
        self,
        split: Literal["train", "val", "test"] = "test",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_loader = (
            self.train_loader if split == "train" else
            self.val_loader if split == "val" else
            self.test_loader
        )
        self.model.eval()
        use_interpolation = bool(self.cfg.inference_with_interpolation)

        sample_pred: Dict[int, List[float]] = {}
        sample_true: Dict[int, List[float]] = {}
        sample_bins: Dict[int, List[int]] = {}

        for batch in data_loader:
            if self.loss_mode == "sample":
                inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
                bsz = inputs.shape[0]
                for b in range(bsz):
                    s_idx = int(sample_idx[b].item())
                    if mask is not None:
                        valid_mask = mask[b].bool()
                    else:
                        valid_mask = torch.ones(inputs.shape[1], dtype=torch.bool, device=inputs.device)
                    inputs_flat = inputs[b][valid_mask]
                    bin_idx_flat = bin_idx[b][valid_mask]
                    interpolation_mask = valid_mask if use_interpolation else None
                    outputs = self.model(
                        inputs_flat,
                        bin_idx_flat,
                        interpolation_mask=interpolation_mask,
                    ).unsqueeze(0)
                    probs = F.softmax(outputs, dim=-1).squeeze(0).cpu().numpy()
                    y_true = targets[b][valid_mask].cpu().numpy()
                    sample_pred.setdefault(s_idx, []).extend(probs.tolist())
                    sample_true.setdefault(s_idx, []).extend(y_true.tolist())
                    sample_bins.setdefault(s_idx, []).extend(bin_idx_flat.cpu().numpy().tolist())
            else:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                bin_idx = batch["bin_idx"].to(self.device)
                sample_idx = batch["sample_idx"].to(self.device)

                if len(inputs.shape) == 1:
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0)
                    bin_idx = bin_idx.unsqueeze(0)
                    sample_idx = sample_idx.unsqueeze(0)

                for i in range(inputs.shape[0]):
                    s_idx = int(sample_idx[i].item())
                    b_idx = int(bin_idx[i].item())
                    interpolation_mask = torch.ones(1, dtype=torch.bool, device=self.device) if use_interpolation else None
                    output = self.model(
                        inputs[i].unsqueeze(0),
                        bin_idx[i].unsqueeze(0),
                        interpolation_mask=interpolation_mask,
                    )
                    prob = float(torch.sigmoid(output).cpu().numpy().item())
                    y_true = float(targets[i].cpu().numpy().item())
                    sample_pred.setdefault(s_idx, []).append(prob)
                    sample_true.setdefault(s_idx, []).append(y_true)
                    sample_bins.setdefault(s_idx, []).append(b_idx)

        if self.loss_type == "logistic":
            for s_idx, preds in sample_pred.items():
                pred_arr = np.array(preds)
                pred_sum = pred_arr.sum()
                if pred_sum > 0:
                    sample_pred[s_idx] = (pred_arr / pred_sum).tolist()

        idx_to_sample = {v: k for k, v in self.sample_index.items()}
        idx_to_bin = {v: k for k, v in self.bin_index.items()}

        preds_flat, trues_flat, sample_labels, bin_labels = [], [], [], []
        for s_idx in sorted(sample_pred.keys()):
            n = len(sample_pred[s_idx])
            preds_flat.extend(sample_pred[s_idx])
            trues_flat.extend(sample_true[s_idx])
            sample_labels.extend([idx_to_sample[s_idx]] * n)
            bin_labels.extend([idx_to_bin[int(b)] for b in sample_bins[s_idx]])

        return (
            np.array(preds_flat, dtype=np.float32),
            np.array(trues_flat, dtype=np.float32),
            np.array(sample_labels),
            np.array(bin_labels),
        )


    def compute_metrics(
        self,
        split: Literal["train", "val", "test"],
        predictions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Compute per-observation and per-sample evaluation metrics.

        Micro metrics are computed on all observations pooled together. Macro metrics
        are computed per sample first and then averaged to avoid domination by samples
        with many observed BINs.

        Args:
            split: Dataset split to evaluate.
            predictions: Optional pre-computed output of get_predictions(). When
                provided the forward pass is skipped, avoiding a redundant computation
                that would otherwise double the inference cost at each epoch.
        """
        y_pred, y_true, sample_labels, _ = predictions if predictions is not None else self.get_predictions(split=split)

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = np.clip(y_pred[valid], 0, 1)
        eps = 1e-10
        
        def _shannon_diversity(values: np.ndarray, eps: float = 1e-10) -> float:
            """Compute Shannon diversity from non-negative abundance values."""
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == 0:
                return np.nan
            # eps is added before normalization so zero-only vectors still produce a
            # finite distribution and log argument.
            probs = (arr + eps) / np.sum(arr + eps)
            return float(-np.sum(probs * np.log(probs + eps)))
        
        def _spearman_rho(x: np.ndarray, y: np.ndarray) -> Optional[float]:
            """Compute Spearman rho and return None if undefined."""
            rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
            rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
            if np.std(rank_x) == 0 or np.std(rank_y) == 0:
                return None
            rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
            return rho if np.isfinite(rho) else None
        
        def _r2_and_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
            """Compute R² plus fitted intercept for Shannon diversity."""
            if len(y_true) == 0:
                return np.nan, np.nan
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + eps))
            
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            intercept = float(intercept) if np.isfinite(intercept) else np.nan
            return r2, intercept

        rmse_macro = np.nan
        mae_macro = np.nan
        kl_divergence = np.nan
        shannon_r2 = np.nan
        shannon_intercept = np.nan
        spearman_macro = np.nan

        if sample_labels is not None:
            sample_labels_v = np.asarray(sample_labels)[valid]
            rmse_per, mae_per, kl_per = [], [], []
            shannon_true_per, shannon_pred_per = [], []
            spearman_per = []
            for sample in np.unique(sample_labels_v):
                mask = sample_labels_v == sample
                true_s = y_true[mask]
                pred_s = y_pred[mask]
                if len(true_s) == 0:
                    continue
                # Macro metrics weight each sample equally regardless of sample size.
                rmse_per.append(float(np.sqrt(np.mean((true_s - pred_s) ** 2))))
                mae_per.append(float(np.mean(np.abs(true_s - pred_s))))
                true_s_norm = (true_s + eps) / (true_s + eps).sum()
                pred_s_norm = (pred_s + eps) / (pred_s + eps).sum()
                kl_per.append(float(np.sum(true_s_norm * np.log(true_s_norm / pred_s_norm))))

                shannon_true = _shannon_diversity(true_s, eps)
                shannon_pred = _shannon_diversity(pred_s, eps)
                if np.isfinite(shannon_true) and np.isfinite(shannon_pred):
                    shannon_true_per.append(shannon_true)
                    shannon_pred_per.append(shannon_pred)

                if len(true_s) > 1:
                    rho = _spearman_rho(true_s, pred_s)
                    if rho is not None:
                        spearman_per.append(rho)

            if rmse_per:
                rmse_macro = float(np.mean(rmse_per))
                mae_macro = float(np.mean(mae_per))
                kl_divergence = float(np.mean(kl_per))

            if len(shannon_true_per) > 1:
                shannon_r2, shannon_intercept = _r2_and_intercept(np.array(shannon_true_per), np.array(shannon_pred_per))

            if spearman_per:
                spearman_macro = float(np.mean(spearman_per))

        rmse_micro = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae_micro = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + eps))

        y_true_log = np.log(y_true + 1)
        y_pred_log = np.log(y_pred + 1)
        ss_res_log = np.sum((y_true_log - y_pred_log) ** 2)
        ss_tot_log = np.sum((y_true_log - np.mean(y_true_log)) ** 2)
        r2_log = float(1 - ss_res_log / (ss_tot_log + eps))

        zero_mask = y_true == 0
        nonzero_mask = y_true > 0
        # Report sparsity-sensitive errors separately because abundance vectors can be
        # strongly zero-inflated.
        rmse_zeros = float(np.sqrt(np.mean((y_true[zero_mask] - y_pred[zero_mask]) ** 2))) if zero_mask.sum() > 0 else np.nan
        mae_zeros = float(np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask]))) if zero_mask.sum() > 0 else np.nan
        rmse_nonzeros = float(np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2))) if nonzero_mask.sum() > 0 else np.nan
        mae_nonzeros = float(np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]))) if nonzero_mask.sum() > 0 else np.nan

        corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        correlation = 0.0 if np.isnan(corr) else float(corr)

        nz = y_true != 0
        rel_error = np.zeros_like(y_true, dtype=float)
        rel_error[nz] = np.abs(y_pred[nz] - y_true[nz]) / np.abs(y_true[nz])
        absolute_relative_error = float(np.mean(rel_error[nz])) if nz.sum() > 0 else np.nan

        return {
            "RMSE (micro)": rmse_micro,
            "RMSE (macro)": rmse_macro,
            "MAE (micro)": mae_micro,
            "MAE (macro)": mae_macro,
            "Absolute Relative Error": absolute_relative_error,
            "R² (Shannon diversity)": shannon_r2,
            "Shannon intercept": shannon_intercept,
            "Spearman Rho (macro)": spearman_macro,
            "R²": r2,
            "R² (log + 1)": r2_log,
            "RMSE (zeros)": rmse_zeros,
            "MAE (zeros)": mae_zeros,
            "RMSE (non-zeros)": rmse_nonzeros,
            "MAE (non-zeros)": mae_nonzeros,
            "KL Divergence": kl_divergence,
            "Correlation": correlation,
            "n_zeros": float(zero_mask.sum()),
            "n_nonzeros": float(nonzero_mask.sum()),
        }


    def _metric_key(self, metric_name: str) -> str:
        return (
            metric_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("²", "2")
            .replace("+", "plus")
            .replace("-", "_")
            .replace("/", "_")
        )

    def _plot_training_progress(self) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        epoch_nums = [e for e, _ in self.train_losses]
        train_vals = [l for _, l in self.train_losses]
        val_vals = [l for _, l in self.val_losses]

        ax.plot(epoch_nums, train_vals, "b-", linewidth=2, alpha=0.8, label="Train Loss")
        ax.plot(epoch_nums, val_vals, "r-", linewidth=2, alpha=0.8, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Progress")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        fig_dir = os.path.join(self.base_artifact_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, f"training_progress_{self.model_name}_{self.run_id}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def run(self, use_wandb: bool = True) -> Dict[str, Any]:
        log.info(
            f"Starting training for model={self.model_name}, epochs={self.cfg.epochs}, resume={self.resume}"
        )
        training_seconds = 0.0
        total_start = time.perf_counter()

        for epoch in tqdm(range(self.start_epoch, self.cfg.epochs), desc="Epochs", leave=False):
            self.current_epoch = epoch

            # Stop if wall-clock training budget is spent (epoch 0 always runs for a baseline)
            if epoch > 0 and training_seconds >= TIME_BUDGET:
                log.info(f"Time budget of {TIME_BUDGET}s reached after epoch {epoch}. Stopping.")
                break

            epoch_start = time.perf_counter()
            if self.cfg.interpolated_sample_fraction > 0.0:
                self._refresh_epoch_interpolation_selection()
            else:
                self._epoch_selected_sample_ids = np.empty(0, dtype=np.int64)
                self._epoch_selected_sample_ids_t = torch.empty(0, dtype=torch.long)

            # Linearly fade proximal pull as training progresses: early stability,
            # late freedom for latent adaptation.
            alpha = min(1.0, epoch / self.cfg.epochs)
            prox_weight = self.cfg.latent_init_prox_reg * (1.0 - alpha)

            for batch_idx, batch in enumerate(self.train_loader):
                batch_start = time.perf_counter()

                latent_start = time.perf_counter()
                latent_vec, latent_timings = self.solve_latent(batch=batch, prox_weight=prox_weight)
                latent_s = time.perf_counter() - latent_start

                loss_value, mlp_timings = self._train_batch(batch)
                self.mlp_scheduler.step()

                total_s = time.perf_counter() - batch_start

                if use_wandb and WANDB_AVAILABLE:
                    # timings = {"setup_s": setup_done, "epoch_logits_s": [], "epoch_loss_s": [], "epoch_backward_s": [], "epoch_optimizer_s": []}
                    wandb.log({
                        "train/batch/loss": loss_value,
                        "train/batch/mlp_lr": self.mlp_optimizer.param_groups[0]["lr"],
                        "train/batch/latent_lr": latent_timings["latent_lr"],
                        "timing/batch/mlp/forward_s": mlp_timings["forward_s"],
                        "timing/batch/mlp/backward_s": mlp_timings["backward_s"],
                        "timing/batch/mlp/optim_s": mlp_timings["optim_s"],
                        "timing/batch/latent/total": latent_s,
                        "timing/batch/latent/setup_s": latent_timings["setup_s"],
                        "timing/batch/latent/forward_s": latent_timings["forward_s"],
                        "timing/batch/latent/backward_s": latent_timings["backward_s"],
                        "timing/batch/latent/optim_s": latent_timings["optim_s"],
                        "train/batch/latent/loss_CE": latent_timings["loss_CE"],
                        "train/batch/latent/loss_l2": latent_timings["loss_l2"],
                        "train/batch/latent/loss_smooth": latent_timings["loss_smooth"],
                        "train/batch/latent/loss_prox": latent_timings["loss_prox"],
                        "train/batch/latent/loss_total": latent_timings["loss_total"],
                        "timing/batch/total_s": total_s,
                        "train/epoch": epoch,
                        "train/batch_idx": batch_idx,
                        "train/interpolated_sample_count": int(self._epoch_selected_sample_ids.size),
                    })

            # Accumulate only batch-loop time (excludes eval/metrics overhead)
            training_seconds += time.perf_counter() - epoch_start

            train_eval_start = time.perf_counter()
            train_preds = self.get_predictions(split="train")
            train_loss = self.validate(split="train")
            train_eval_s = time.perf_counter() - train_eval_start

            val_eval_start = time.perf_counter()
            val_preds = self.get_predictions(split="val")
            val_loss = self.validate(split="val")
            val_eval_s = time.perf_counter() - val_eval_start

            train_metric_start = time.perf_counter()
            train_metrics = self.compute_metrics(split="train", predictions=train_preds)
            train_metric_s = time.perf_counter() - train_metric_start

            val_metric_start = time.perf_counter()
            val_metrics = self.compute_metrics(split="val", predictions=val_preds)
            val_metric_s = time.perf_counter() - val_metric_start
            self.last_val_metrics = val_metrics

            run_ablation = bool(self.cfg.diag_ablation_interval > 0 and (epoch % self.cfg.diag_ablation_interval == 0))
            diag = self._collect_diagnostics(epoch=epoch, run_abl=run_ablation)
            self.latent_diagnostics.append(diag)

            self.train_losses.append((epoch, train_loss))
            self.val_losses.append((epoch, val_loss))

            val_kl = val_metrics.get("KL Divergence", float("inf"))
            improved = np.isfinite(val_kl) and val_kl < self.best_val_kl
            if improved:
                self.best_val_kl = val_kl
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch=epoch, val_loss=val_loss, val_metrics=val_metrics, best=improved)

            epoch_total_s = time.perf_counter() - epoch_start

            epoch_log_payload = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/epoch": epoch,
                "train/interpolated_sample_count": int(self._epoch_selected_sample_ids.size),
                "timing/epoch/total_s": float(epoch_total_s),
                "timing/epoch/train_eval_s": float(train_eval_s),
                "timing/epoch/val_eval_s": float(val_eval_s),
                "timing/epoch/train_metrics_s": float(train_metric_s),
                "timing/epoch/val_metrics_s": float(val_metric_s),
            }
            for metric_name, metric_value in train_metrics.items():
                epoch_log_payload[f"train/metrics/{self._metric_key(metric_name)}"] = metric_value
            for metric_name, metric_value in val_metrics.items():
                epoch_log_payload[f"val/metrics/{self._metric_key(metric_name)}"] = metric_value
            for diag_name, diag_value in diag.items():
                epoch_log_payload[f"diag/{diag_name}"] = diag_value

            if use_wandb and WANDB_AVAILABLE:
                wandb.log(epoch_log_payload)

            log.info(
                f"Epoch {epoch + 1}/{self.cfg.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"val_kl={val_kl:.6f}, best_val_kl={self.best_val_kl:.6f}, epoch_s={epoch_total_s:.2f}"
            )

        best_ckpt = self._checkpoint_path("best.pt")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self._plot_training_progress()

        test_loss = self.validate(split="test")
        test_preds = self.get_predictions(split="test")
        test_metrics = self.compute_metrics(split="test", predictions=test_preds)

        total_seconds = time.perf_counter() - total_start
        test_kl = test_metrics.get("KL Divergence", float("nan"))
        print("---")
        print(f"val_kl_divergence:  {self.best_val_kl:.6f}")
        print(f"test_kl_divergence: {test_kl:.6f}")
        print(f"val_loss:           {self.best_val_loss:.6f}")
        print(f"test_loss:          {test_loss:.6f}")
        print(f"training_seconds:   {training_seconds:.1f}")
        print(f"total_seconds:      {total_seconds:.1f}")
        print(f"num_epochs:         {self.current_epoch + 1}")
        for k, v in test_metrics.items():
            print(f"test/{self._metric_key(k)}:  {v:.6f}")

        if use_wandb and WANDB_AVAILABLE:
            payload = {
                "test/loss": test_loss,
            }
            for metric_name, metric_value in test_metrics.items():
                payload[f"test/metrics/{self._metric_key(metric_name)}"] = metric_value
            wandb.log(payload)

        predictions, targets, sample_labels, bin_labels = test_preds
        latent_vector = self.model.latent_vec.detach().cpu().numpy()

        return {
            "model": self.model_name,
            "run_id": self.run_id,
            "best_val_kl": self.best_val_kl,
            "best_val_loss": self.best_val_loss,
            "test_loss": test_loss,
            "predictions": predictions,
            "targets": targets,
            "sample_labels": sample_labels,
            "bin_labels": bin_labels,
            "latent_vector": latent_vector,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.last_val_metrics,
            "latent_diagnostics": self.latent_diagnostics,
            "test_metrics": test_metrics,
        }

    @torch.no_grad()
    def _compute_ablation_delta(self) -> float:
        saved = self.model.latent_vec.data.clone()
        self.model.latent_vec.data.zero_()
        loss_no_latent = self.validate(split="val")
        self.model.latent_vec.data.copy_(saved)
        loss_with_latent = self.validate(split="val")
        return float(loss_no_latent - loss_with_latent)

    def _collect_diagnostics(self, epoch: int, run_abl: bool = False) -> Dict[str, Any]:
        diag: Dict[str, Any] = {
            "epoch": int(epoch),
            # Latent metrics are computed as the average per-dimension to avoid domination by large embed_dim values.
            "latent_mean": float(self.model.latent_vec.data.mean(dim=0).mean().item()),
            "latent_std": float(self.model.latent_vec.data.std(dim=0).mean().item()),
            "latent_min": float(self.model.latent_vec.data.min().item()),
            "latent_max": float(self.model.latent_vec.data.max().item()),
            "ablation_delta": self._compute_ablation_delta() if run_abl else None,
            "final_weight_mean": float(self.model.final_linear.weight.data.mean().item()) if self.cfg.embed_dim > 1 else None,
            "final_weight_std": float(self.model.final_linear.weight.data.std().item()) if self.cfg.embed_dim > 1 else None,
            "final_weight_norm": float(self.model.final_linear.weight.data.norm().item()) if self.cfg.embed_dim > 1 else None,
        }
        return diag

    def _to_device(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        bin_idx = batch["bin_idx"].to(self.device)
        sample_idx = batch["sample_idx"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        return inputs, targets, bin_idx, sample_idx, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metabarcoding training entrypoint")
    parser.add_argument("--model", type=str, default="default", help="Name of the model variant being trained")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint for this model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    set_seed()
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = Config()

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.abspath(os.path.join(cfg.results_dir, args.model))
    os.makedirs(run_dir, exist_ok=True)

    use_wandb = WANDB_AVAILABLE
    if not WANDB_AVAILABLE:
        log.warning("wandb is not installed; continuing without wandb logging")
    else:
        wandb.init(
            project="metabarcoding",
            name=f"{args.model}_{run_id}",
            config={"model": args.model, **asdict(cfg)},
            dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wandb"),
        )

    trainer = Trainer(cfg=cfg, model_name=args.model, run_id=run_id, resume=args.resume)
    results = trainer.run(use_wandb=use_wandb)

    pkl_name = f"results_{args.model}_{run_id}.pkl"
    pkl_path = os.path.join(run_dir, pkl_name)
    with open(pkl_path, "wb") as fh:
        pickle.dump(results, fh)
    log.info(f"Results saved to: {pkl_path}")

    if use_wandb:
        wandb.finish()
