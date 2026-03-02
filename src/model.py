from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn
from mlp import MLPModel
from latent_solver import LatentSolver

class Model(nn.Module):
    """
    Joint model combining:
        - an MLPModel for m_theta(s, b) in logit space
        - a latent vector D (one scalar per BIN) in logit space
    
    The model predicts: output = intrinsic + D (in logit space)
    The predicted probability is: p = sigmoid(output)
    
    Loss is computed using BCEWithLogitsLoss: BCE(output, target_probability)
    """

    def __init__(
        self,
        mlp: nn.Module,
        latent_solver: LatentSolver,
        n_bins: int,
        device: torch.device,
    ):
        """
        Args:
            mlp: MLP model predicting intrinsic logit m(s,b)
            latent_solver: Solver for latent vector D
            n_bins: Number of BINs (size of latent vector)
            device: Compute device
        """
        super().__init__()
        self.mlp = mlp
        self.latent_solver = latent_solver
        self.device = device

        # Latent vector D (updated in Phase A, frozen during Phase B)
        # Latent vector D (initialized to zeros, not random noise)
        self.latent_vec = nn.Parameter(
            torch.zeros(n_bins, device=device),
            requires_grad=False,
        )
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor, bin_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining MLP prediction and latent effect.

        Args:
            x: Input features [N, input_dim]
            bin_ids: BIN indices [N]

        Returns:
            Predicted logits [N] (use sigmoid to get probabilities)
        """
        intrinsic = self.mlp(x).squeeze(-1)
        latent = self.latent_vec[bin_ids]
        return intrinsic + latent

    @torch.no_grad()
    def set_latent(self, latent_new: Union[torch.Tensor, np.ndarray]) -> None:
        """Update the latent vector D after a Phase A solve."""
        if isinstance(latent_new, np.ndarray):
            latent_new = torch.tensor(latent_new, dtype=torch.float32)
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
            numpy vector of MLP predictions, one per observation
        """
        if data_loader is None:
            return np.array([])

        self.eval()
        
        if loss_mode == "sample":
            # Sample mode: collect predictions maintaining original order
            # We need to unflatten and handle masks
            all_preds = []
            all_indices = []
            
            for batch in data_loader:
                x = batch["input"].to(self.device)  # [B, max_bins, features]
                mask = batch.get("mask")  # [B, max_bins]
                bin_idx = batch["bin_idx"]  # [B, max_bins] - these are the original bin indices
                sample_idx = batch["sample_idx"]  # [B]
                
                B, max_bins, n_feat = x.shape
                x_flat = x.view(B * max_bins, n_feat)
                
                intrinsic_flat = self.mlp(x_flat).squeeze(-1)  # [B * max_bins]
                intrinsic = intrinsic_flat.view(B, max_bins)  # [B, max_bins]
                
                # Extract valid predictions (non-padded)
                intrinsic_np = intrinsic.cpu().numpy()
                if mask is not None:
                    mask_np = mask.cpu().numpy().astype(bool)
                else:
                    mask_np = np.ones((B, max_bins), dtype=bool)
                
                for b in range(B):
                    valid_mask = mask_np[b]
                    all_preds.extend(intrinsic_np[b, valid_mask])
            
            return np.array(all_preds)
        else:
            # Bin mode: straightforward
            preds = []
            for batch in data_loader:
                x = batch["input"].to(self.device)
                intrinsic = self.mlp(x).squeeze(-1)
                preds.append(intrinsic.cpu().numpy())
            return np.concatenate(preds, axis=0)
    
    
    def save_model(self, path: str) -> None:
        """Save the model state to the specified path."""
        print(f"Saving model to {path}")
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load the model state from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))