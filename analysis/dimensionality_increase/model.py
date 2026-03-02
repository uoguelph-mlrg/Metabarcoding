from typing import Union, Optional, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLPModel
from latent_solver import LatentSolver

class Model(nn.Module):
    """
    Joint model with multiplicative gating architecture:
        - MLP outputs m(x) ∈ R^d (log-feature embedding)
        - Latent h ∈ R^{n_bins × d} acts as feature-wise bias field
        - Gating: m̃ = m(x) ⊙ g(h[bin_id]) where g is positive gating function
            gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid",
        - Final prediction: z = w^T m̃ (scalar logit, no bias)
    
    Key property: if m(x) = 0 → z = 0 (latent cannot predict alone)
    """

    def __init__(
        self,
        mlp: nn.Module,
        latent_solver: LatentSolver,
        n_bins: int,
        embed_dim: int,
        device: torch.device,
        gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid",
        gating_alpha: float = 0.5,
        gating_kappa: float = 0.5,
        gating_epsilon: float = 0.693,  # log(2), so softplus(0) - epsilon ≈ 0
    ):
        """
        Args:
            mlp: MLP model predicting intrinsic embedding m(s,b) ∈ R^d
            latent_solver: Solver for latent vector h ∈ R^{n_bins × d}
            n_bins: Number of BINs
            embed_dim: Embedding dimension d
            device: Compute device
            gating_fn: Gating function type
                - "exp": exponential gating (primary) m̃ = m ⊙ exp(h)
                - "scaled_exp": scaled exponential m̃ = m ⊙ exp(α·h)
                - "additive": additive-multiplicative m̃ = m ⊙ (1 + h)
                - "softplus": non-negative additive m̃ = m ⊙ (1 + softplus(h) - ε)
                - "tanh": bounded tanh m̃ = m ⊙ (1 + tanh(h)·κ)
                - "sigmoid": bounded sigmoid m̃ = m ⊙ (2·σ(h))
                - "dot_product": direct dot product z = m(x) · h (bypasses final_linear)
            gating_alpha: Scaling factor for scaled_exp (in (0,1])
            gating_kappa: Scaling factor for tanh
            gating_epsilon: Offset for softplus (default log(2))
        """
        super().__init__()
        self.mlp = mlp
        self.latent_solver = latent_solver
        self.device = device
        self.embed_dim = embed_dim
        self.gating_fn = gating_fn
        self.gating_alpha = gating_alpha
        self.gating_kappa = gating_kappa
        self.gating_epsilon = gating_epsilon

        # Latent vector h ∈ R^{n_bins × d} (updated in Phase A, frozen during Phase B)
        # Initialize to h=0 so that g(h=0) = 1 (identity modulation)
        self.latent_vec = nn.Parameter(
            torch.zeros((n_bins, embed_dim), device=device),
            requires_grad=False,
        )
        self.n_bins = n_bins
        
        # Final linear layer w: R^d → R for scalar logit (NO BIAS per specification)
        self.final_linear = nn.Linear(embed_dim, 1, bias=False, device=device)
        # Initialize with small random values (Xavier/Glorot)
        nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, x: torch.Tensor, bin_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multiplicative gating.

        Args:
            x: Input features [N, input_dim]
            bin_ids: BIN indices [N]

        Returns:
            Predicted logits [N] (raw logits before sigmoid/softmax)
        """
        # Intrinsic embedding m(x) ∈ R^{N × d}
        intrinsic = self.mlp(x)  # [N, d]
        
        # Latent bias field h[bin] ∈ R^{N × d}
        latent = self.latent_vec[bin_ids]  # [N, d]
        
        # Apply gating function g(h) - all should give g(0) = 1
        if self.gating_fn == "exp":
            # (A) Exponential: exp(h), g(0) = 1
            gate = torch.exp(latent)
        elif self.gating_fn == "scaled_exp":
            # (B) Scaled exponential: exp(α·h), g(0) = 1
            gate = torch.exp(self.gating_alpha * latent)
        elif self.gating_fn == "additive":
            # (C) Additive-multiplicative: 1 + h, g(0) = 1
            gate = 1.0 + latent
        elif self.gating_fn == "softplus":
            # (D) Non-negative additive: 1 + softplus(h) - ε
            # Choose ε so that g(0) = 1: ε = softplus(0) = log(2) ≈ 0.693
            gate = 1.0 + F.softplus(latent) - self.gating_epsilon
        elif self.gating_fn == "tanh":
            # (E) Bounded tanh: 1 + tanh(h)·κ, g(0) = 1
            gate = 1.0 + torch.tanh(latent) * self.gating_kappa
        elif self.gating_fn == "sigmoid":
            # (E) Bounded sigmoid: 2·σ(h), g(0) = 2·0.5 = 1
            gate = 2.0 * torch.sigmoid(latent)
        elif self.gating_fn == "dot_product":
            # (F) Direct dot product: z = m(x) · h (no final_linear)
            # The logit is the inner product of the MLP embedding and the latent vector.
            # This collapses the two vectors into a scalar directly, bypassing w.
            output = (intrinsic * latent).sum(dim=-1)  # [N]
            return output
        else:
            raise ValueError(f"Unknown gating function: {self.gating_fn}")
        
        # Feature-wise multiplicative modulation: m̃ = m ⊙ g(h)
        modulated = intrinsic * gate  # [N, d]
        
        # Final scalar logit: z = w^T m̃ (no bias)
        output = self.final_linear(modulated).squeeze(-1)  # [N]
        
        return output

    @torch.no_grad()
    def set_latent(self, latent_new: Union[torch.Tensor, np.ndarray]) -> None:
        """Update the latent vector h after a Phase A solve.
        
        Args:
            latent_new: New latent values, shape (n_bins, embed_dim)
        """
        if isinstance(latent_new, np.ndarray):
            latent_new = torch.tensor(latent_new, dtype=torch.float32)
        if latent_new.shape != (self.n_bins, self.embed_dim):
            raise ValueError(f"Expected shape ({self.n_bins}, {self.embed_dim}), got {latent_new.shape}")
        self.latent_vec.copy_(latent_new.to(self.device))

    @torch.no_grad()
    def predict_MLP_only(self, data_loader, loss_mode: str = "bin") -> np.ndarray:
        """Predict m for all observations using the current MLP (without latent).
        
        For the latent solver, we need predictions aligned with the original data order.
        This handles both sample and bin mode data loaders.
        
        Args:
            data_loader: DataLoader in either sample or bin mode
            loss_mode: "sample" or "bin" to indicate loader mode
        
        Returns:
            numpy array of MLP predictions, shape (n_obs, embed_dim)
        """
        if data_loader is None:
            return np.array([])

        self.eval()
        
        if loss_mode == "sample":
            # Sample mode: collect predictions maintaining original order
            # We need to unflatten and handle masks
            all_preds = []
            
            for batch in data_loader:
                x = batch["input"].to(self.device)  # [B, max_bins, features]
                mask = batch.get("mask")  # [B, max_bins]
                
                B, max_bins, n_feat = x.shape
                x_flat = x.view(B * max_bins, n_feat)
                
                intrinsic_flat = self.mlp(x_flat)  # [B * max_bins, d]
                intrinsic = intrinsic_flat.view(B, max_bins, self.embed_dim)  # [B, max_bins, d]
                
                # Extract valid predictions (non-padded)
                intrinsic_np = intrinsic.cpu().numpy()
                if mask is not None:
                    mask_np = mask.cpu().numpy().astype(bool)
                else:
                    mask_np = np.ones((B, max_bins), dtype=bool)
                
                for b in range(B):
                    valid_mask = mask_np[b]
                    all_preds.append(intrinsic_np[b, valid_mask])  # [n_valid, d]
            
            return np.concatenate(all_preds, axis=0)  # [n_obs, d]
        else:
            # Bin mode: straightforward
            preds = []
            for batch in data_loader:
                x = batch["input"].to(self.device)
                intrinsic = self.mlp(x)  # [N, d]
                preds.append(intrinsic.cpu().numpy())
            return np.concatenate(preds, axis=0)  # [n_obs, d]
    
    
    def save_model(self, path: str) -> None:
        """Save the model state to the specified path."""
        print(f"Saving model to {path}")
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load the model state from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))