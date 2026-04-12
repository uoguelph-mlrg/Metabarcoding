from typing import Any, Dict, Optional, Literal, Union
import numpy as np
import torch
import torch.nn as nn
from gating_functions import make_gating_function

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
        latent_input_dim: int,
        embed_dim: int = 1,
        gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid",
        gating_alpha: float = 0.5, # FIXME: have generic parameters variable that can be assigned from a dict (using the correct key for each gating function) to avoid unused parameters and confusion
        gating_kappa: float = 0.5,
        gating_epsilon: float = 0.693,
        latent_init_std: float = 0.0,
        interpolation_enabled: bool = False,
        include_self_in_interpolation: bool = True,
    ):
        """
        - Z (input embedding) is co-trained with MLP via joint gradient descent (Phase B, like V1)
        - D (output scalar) is solved analytically via CG/L-BFGS (Phase A, like baseline src)

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
        self.latent_input_dim = int(latent_input_dim)
        self.embed_dim = embed_dim
        self.n_bins = n_bins
        self._interpolation_enabled = interpolation_enabled
        self.H_interp: Optional[torch.Tensor] = None

        if interpolation_enabled:
            interp_device = self.device if self.device.type != "mps" else torch.device("cpu")
            self.H_interp = latent_solver.get_interpolation_operator(include_self_in_interpolation, device=interp_device)

        z_init = torch.randn((n_bins, self.latent_input_dim), device=device) * latent_init_std if latent_init_std > 0 else torch.zeros((n_bins, self.latent_input_dim), device=device)
        self.latent_z = nn.Parameter(
            z_init,
            requires_grad=True,
        )

        if embed_dim > 1:
            self.gating_fn = gating_fn
            self.gating = make_gating_function(
                gating_fn, alpha=gating_alpha, kappa=gating_kappa, epsilon=gating_epsilon
            )
            # Output latent matrix D ∈ R^{n_bins × d}
            d_init = torch.randn((n_bins, embed_dim), device=device) * latent_init_std if latent_init_std > 0 else torch.zeros((n_bins, embed_dim), device=device)
            self.latent_d = nn.Parameter(
                d_init,
                requires_grad=False,
            )
            # Final linear layer w: R^d → R (no bias)
            self.final_linear = nn.Linear(embed_dim, 1, bias=False, device=device)
            nn.init.xavier_uniform_(self.final_linear.weight)
        elif embed_dim == 1:
            # Scalar mode: output latent is a 1D vector, no gating, no final linear
            d_init = torch.randn(n_bins, device=device) * latent_init_std if latent_init_std > 0 else torch.zeros(n_bins, device=device)
            self.latent_d = nn.Parameter(
                d_init,
                requires_grad=False,
            )
        else:
            # embed_dim == 0: no output latent branch, only Z-augmented MLP direct logits.
            self.register_parameter("latent_d", None)

    def _compute_logits_from_latent_values(
        self,
        latent_obs: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        if self.embed_dim == 0:
            return intrinsic.squeeze(-1)
        if self.embed_dim == 1:
            return intrinsic.squeeze(-1) + latent_obs.squeeze(-1)

        gated = self.gating.gate_torch(latent_obs)
        modulated = intrinsic * gated
        return self.final_linear(modulated).squeeze(-1)

    def _lookup_output_latent(
        self,
        bin_ids: torch.Tensor,
        interpolation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embed_dim == 0:
            raise RuntimeError("Output latent lookup is unavailable when embed_dim == 0")
        """Fetch latent values for observations, optionally mixing interpolated latents.

        interpolation_mask is observation-aligned (shape [N], True means use
        interpolation operator output for that observation, False means use the BIN's
        own latent value).
        """
        latent_source = self.latent_d
        own_latent = latent_source[bin_ids]

        if interpolation_mask is None:
            return own_latent

        if not self._interpolation_enabled:
            raise RuntimeError("Interpolation was requested but interpolation operators were not initialized")

        mask = interpolation_mask.to(device=latent_source.device, dtype=torch.bool).reshape(-1)
        if not bool(torch.any(mask).item()):
            return own_latent

        latent_source = latent_source.unsqueeze(-1) if self.embed_dim == 1 else latent_source
        interp_operator = self.H_interp
        if interp_operator is None:
            raise RuntimeError("Interpolation operator is not initialized")

        interp_device = interp_operator.device
        # Keep sparse matmul on the operator device to avoid repeated sparse transfers.
        latent_for_interp = latent_source if interp_device == latent_source.device else latent_source.to(interp_device)
        if interp_device != latent_source.device:
            interp_operator = interp_operator.to(interp_device)

        interpolated_full = torch.sparse.mm(interp_operator, latent_for_interp)
        interpolated_obs = interpolated_full[bin_ids]
        if interpolated_obs.device != own_latent.device:
            interpolated_obs = interpolated_obs.to(own_latent.device)
        if self.embed_dim == 1:
            own_latent = own_latent.unsqueeze(-1)
            mixed = torch.where(mask.unsqueeze(-1), interpolated_obs, own_latent)
            return mixed.squeeze(-1)

        mixed = torch.where(mask.unsqueeze(-1), interpolated_obs, own_latent)
        return mixed

    def _lookup_input_latent(
        self,
        bin_ids: torch.Tensor,
        interpolation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fetch Z input latents for observations, optionally mixing interpolated latents."""
        latent_source = self.latent_z
        own_latent = latent_source[bin_ids]

        if interpolation_mask is None:
            return own_latent

        if not self._interpolation_enabled:
            raise RuntimeError("Interpolation was requested but interpolation operators were not initialized")

        mask = interpolation_mask.to(device=latent_source.device, dtype=torch.bool).reshape(-1)
        if not bool(torch.any(mask).item()):
            return own_latent

        interp_operator = self.H_interp
        if interp_operator is None:
            raise RuntimeError("Interpolation operator is not initialized")

        interp_device = interp_operator.device
        latent_for_interp = latent_source if interp_device == latent_source.device else latent_source.to(interp_device)
        if interp_device != latent_source.device:
            interp_operator = interp_operator.to(interp_device)

        interpolated_full = torch.sparse.mm(interp_operator, latent_for_interp)
        interpolated_obs = interpolated_full[bin_ids]
        if interpolated_obs.device != own_latent.device:
            interpolated_obs = interpolated_obs.to(own_latent.device)
        return torch.where(mask.unsqueeze(-1), interpolated_obs, own_latent)

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
        z_obs = self._lookup_input_latent(
            bin_ids,
            interpolation_mask=interpolation_mask,
        )
        x_augmented = torch.cat([x, z_obs], dim=-1)

        if self.embed_dim > 1:
            intrinsic = self.mlp(x_augmented)                                # [N, d]
        else:
            intrinsic = self.mlp(x_augmented).squeeze(-1)                    # [N]

        if self.embed_dim == 0:
            return intrinsic

        latent = self._lookup_output_latent(
            bin_ids,
            interpolation_mask=interpolation_mask,
        )

        return self._compute_logits_from_latent_values(latent, intrinsic)

    @torch.no_grad()
    def set_latent(self, latent_new: Union[torch.Tensor, np.ndarray]) -> None:
        """Update the output latent D variable after a Phase A solve."""
        if self.embed_dim == 0:
            raise RuntimeError("set_latent is unavailable when embed_dim == 0")
        if isinstance(latent_new, np.ndarray):
            latent_new = torch.tensor(latent_new, dtype=torch.float32)
        if self.embed_dim > 1:
            if latent_new.shape != (self.n_bins, self.embed_dim):
                raise ValueError(f"Expected shape ({self.n_bins}, {self.embed_dim}), got {latent_new.shape}")
        else:
            latent_new = latent_new.reshape(-1)
            if latent_new.shape[0] != self.n_bins:
                raise ValueError(f"Expected shape ({self.n_bins},), got {latent_new.shape}")
        self.latent_d.copy_(latent_new.to(self.device))

    @torch.no_grad()
    def predict_MLP_only(self, data_loader, loss_mode: str = "bin") -> np.ndarray:
        """Predict intrinsic MLP outputs without output latent modulation.

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
                bin_idx = batch["bin_idx"].to(self.device)
                mask = batch.get("mask")  # [B, max_bins]
                
                B, max_bins, n_feat = x.shape
                x_flat = x.view(B * max_bins, n_feat)
                bin_flat = bin_idx.view(B * max_bins)
                z_flat = self._lookup_input_latent(bin_flat)
                x_aug = torch.cat([x_flat, z_flat], dim=-1)
                
                intrinsic_flat = self.mlp(x_aug)  # [B * max_bins, d] or [B * max_bins]
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
                bin_idx = batch["bin_idx"].to(self.device)
                z_obs = self._lookup_input_latent(bin_idx)
                intrinsic = self.mlp(torch.cat([x, z_obs], dim=-1))  # [N, d] or [N]
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