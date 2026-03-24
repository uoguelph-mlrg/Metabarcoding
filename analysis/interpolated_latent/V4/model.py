from typing import Union, Optional, Literal
import numpy as np
import torch
import torch.nn as nn
from mlp import MLPModel
from latent_solver import LatentSolver
from gating_functions import make_gating_function

LatentUsageMode = Literal["normal", "interpolated", "mixed"]

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
        latent_solver: LatentSolver,
        n_bins: int,
        device: torch.device,
        embed_dim: int = 1,
        gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid",
        gating_alpha: float = 0.5, # FIXME: have generic parameters variable that can be assigned from a dict (using the correct key for each gating function) to avoid unused parameters and confusion
        gating_kappa: float = 0.5,
        gating_epsilon: float = 0.693,
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
        """
        super().__init__()
        self.mlp = mlp
        self.latent_solver = latent_solver
        self.device = device
        self.embed_dim = embed_dim
        self.n_bins = n_bins
        self._init_neighbor_smoothing()
        self.latent_mode: LatentUsageMode = "normal"
        self.register_buffer("interpolated_bin_mask", torch.zeros(n_bins, dtype=torch.bool, device=device))

        if embed_dim > 1:
            self.gating_fn = gating_fn
            self.gating = make_gating_function(
                gating_fn, alpha=gating_alpha, kappa=gating_kappa, epsilon=gating_epsilon
            )
            # Latent matrix H ∈ R^{n_bins × d}, initialized to 0 so g(0) = 1 (identity modulation)
            self.latent_vec = nn.Parameter(
                torch.zeros((n_bins, embed_dim), device=device),
                requires_grad=False,
            )
            # Final linear layer w: R^d → R (no bias)
            self.final_linear = nn.Linear(embed_dim, 1, bias=False, device=device)
            nn.init.xavier_uniform_(self.final_linear.weight)
        else:
            # Scalar mode: latent is a 1D vector, no gating, no final linear
            self.latent_vec = nn.Parameter(
                torch.zeros(n_bins, device=device),
                requires_grad=False,
            )

    def _init_neighbor_smoothing(self) -> None:
        """Cache neighbor smoothing matrix from latent solver for interpolation."""
        h_smooth = getattr(self.latent_solver, "H_smooth", None)
        if h_smooth is None:
            # Safe fallback if solver matrix was not built yet.
            self.register_buffer("H_smooth_dense", torch.eye(self.n_bins, device=self.device))
            return

        h_smooth = h_smooth.tocoo()
        indices = torch.tensor(
            np.vstack((h_smooth.row, h_smooth.col)),
            dtype=torch.long,
            device=self.device,
        )
        values = torch.tensor(h_smooth.data, dtype=torch.float32, device=self.device)
        H_sparse = torch.sparse_coo_tensor(indices, values, (self.n_bins, self.n_bins), device=self.device)
        H_sparse = H_sparse.coalesce()
        self.register_buffer("H_smooth_sparse", H_sparse)

        # Also cache self-inclusive interpolation matrix (I + H_smooth) normalized
        h_with_self = getattr(self.latent_solver, "H_with_self", None)
        if h_with_self is not None:
            h_with_self = h_with_self.tocoo()
            indices_self = torch.tensor(
                np.vstack((h_with_self.row, h_with_self.col)),
                dtype=torch.long,
                device=self.device,
            )
            values_self = torch.tensor(h_with_self.data, dtype=torch.float32, device=self.device)
            H_with_self_sparse = torch.sparse_coo_tensor(indices_self, values_self, (self.n_bins, self.n_bins), device=self.device)
            H_with_self_sparse = H_with_self_sparse.coalesce()
            self.register_buffer("H_with_self_sparse", H_with_self_sparse)
        else:
            self.register_buffer("H_with_self_sparse", H_sparse)  # fallback to H_smooth if not computed

    def _interpolate_latent_all_bins(self) -> torch.Tensor:
        """Compute neighbor-interpolated latent for all bins using self-inclusive matrix."""
        # Check for self-inclusive matrix first (used in interpolated mode)
        if hasattr(self, "H_with_self_sparse"):
            if self.embed_dim > 1:
                return torch.sparse.mm(self.H_with_self_sparse, self.latent_vec)
            return torch.sparse.mm(self.H_with_self_sparse, self.latent_vec.unsqueeze(1)).squeeze(1)

        if hasattr(self, "H_smooth_sparse"):
            if self.embed_dim > 1:
                return torch.sparse.mm(self.H_smooth_sparse, self.latent_vec)
            return torch.sparse.mm(self.H_smooth_sparse, self.latent_vec.unsqueeze(1)).squeeze(1)

        if hasattr(self, "H_smooth_dense"):
            if self.embed_dim > 1:
                return self.H_smooth_dense @ self.latent_vec
            return (self.H_smooth_dense @ self.latent_vec.unsqueeze(1)).squeeze(1)

        return self.latent_vec

    @torch.no_grad()
    def set_latent_mode(self, mode: LatentUsageMode) -> None:
        """Set how latent is used during forward pass."""
        if mode not in ("normal", "interpolated", "mixed"):
            raise ValueError(f"Unknown latent mode: {mode}")
        self.latent_mode = mode

    @torch.no_grad()
    def set_interpolated_bin_mask(self, mask: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        """Set BIN mask selecting which bins use interpolated latent in mixed mode."""
        mask_buffer = self.get_buffer("interpolated_bin_mask")
        if mask is None:
            mask_buffer.zero_()
            return

        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask.astype(bool, copy=False))
        elif isinstance(mask, torch.Tensor):
            mask_t = mask.bool().detach().cpu()
        else:
            raise TypeError("mask must be a numpy array, torch tensor, or None")

        if mask_t.numel() != self.n_bins:
            raise ValueError(f"Expected mask of length {self.n_bins}, got {mask_t.numel()}")

        mask_buffer.copy_(mask_t.reshape(-1).to(self.device))

    @torch.no_grad()
    def configure_latent_usage(
        self,
        mode: LatentUsageMode,
        interpolated_bin_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Convenience helper to set mode and optional mixed-mode BIN mask."""
        self.set_latent_mode(mode)
        self.set_interpolated_bin_mask(interpolated_bin_mask)

    def _lookup_latent(
        self,
        bin_ids: torch.Tensor,
        interpolated_obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lookup latent values according to the current latent usage mode."""
        latent_raw = self.latent_vec[bin_ids]

        # Observation-level mask takes precedence when provided.
        if interpolated_obs_mask is not None:
            obs_mask = interpolated_obs_mask.bool()
            if obs_mask.shape != bin_ids.shape:
                raise ValueError(f"Expected interpolated_obs_mask shape {bin_ids.shape}, got {obs_mask.shape}")
            latent_interp_all = self._interpolate_latent_all_bins()
            latent_interp = latent_interp_all[bin_ids]
            if self.embed_dim > 1:
                obs_mask = obs_mask.unsqueeze(-1)
            return torch.where(obs_mask, latent_interp, latent_raw)

        if self.latent_mode == "normal":
            return latent_raw

        latent_interp_all = self._interpolate_latent_all_bins()
        latent_interp = latent_interp_all[bin_ids]
        if self.latent_mode == "interpolated":
            return latent_interp

        mask = self.get_buffer("interpolated_bin_mask")[bin_ids]
        if self.embed_dim > 1:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, latent_interp, latent_raw)

    def forward(
        self,
        x: torch.Tensor,
        bin_ids: torch.Tensor,
        interpolated_obs_mask: Optional[torch.Tensor] = None,
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
            latent = self._lookup_latent(bin_ids, interpolated_obs_mask)     # [N, d]
            modulated = intrinsic * self.gating.gate_torch(latent)           # [N, d]
            return self.final_linear(modulated).squeeze(-1)                  # [N]
        else:
            intrinsic = self.mlp(x).squeeze(-1)                              # [N]
            latent = self._lookup_latent(bin_ids, interpolated_obs_mask)     # [N]
            return intrinsic + latent                                        # [N]

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
        """Predict m for all observations using the current MLP (without latent).
        
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
                    return np.concatenate(all_preds, axis=0)
                else:
                    intrinsic_np = intrinsic_flat.squeeze(-1).view(B, max_bins).cpu().numpy()
                    for b in range(B):
                        all_preds.extend(intrinsic_np[b, mask_np[b]])
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