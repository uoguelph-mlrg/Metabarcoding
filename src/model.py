from typing import Any, Union, Optional, Literal
import numpy as np
import torch
import torch.nn as nn
from mlp import MLPModel
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

    def forward(self, x: torch.Tensor, bin_ids: torch.Tensor) -> torch.Tensor:
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
            latent = self.latent_vec[bin_ids]                                # [N, d]
            modulated = intrinsic * self.gating.gate_torch(latent)           # [N, d]
            return self.final_linear(modulated).squeeze(-1)                  # [N]
        else:
            intrinsic = self.mlp(x).squeeze(-1)                              # [N]
            latent = self.latent_vec[bin_ids]                                # [N]
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