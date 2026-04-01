from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn

# Import from src folder (reusing existing infrastructure)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import from local folder (modified for latent-as-input)
from latent_solver import LatentSolver

class Model(nn.Module):
    """
    Latent-as-input-V2: blend of baseline src and latent_as_input (V1).

    Architecture:
        output = MLP([x, z_b]) + D[b]

    Training protocol (EM-style):
        Phase A: D (output scalar latent) is solved analytically using CG/L-BFGS,
                 given fixed MLP and Z.  D has requires_grad=False.
        Phase B: MLP and Z (input embedding) are trained jointly via gradient descent,
                 given fixed D.  Z has requires_grad=True.

    The only architectural difference from V1 (latent_as_input) is the +D[b] term.
    The only training difference from baseline (src) is that Z is gradient-trained
    alongside the MLP in Phase B instead of D being the sole latent.
    """

    def __init__(
        self,
        mlp: nn.Module,
        latent_solver: LatentSolver,
        n_bins: int,
        latent_dim: int,
        latent_init_std: float,
        device: torch.device,
    ):
        """
        Args:
            mlp: MLP model that takes concatenated features + latent as input
            latent_solver: Solver for latent embeddings Z
            n_bins: Number of BINs (size of latent embedding table)
            latent_dim: Dimension of latent embedding per BIN
            latent_init_std: Std for Gaussian initialization of latent
            device: Compute device
        """
        super().__init__()
        self.mlp = mlp
        self.latent_solver = latent_solver
        self.device = device
        self.n_bins = n_bins
        self.latent_dim = latent_dim

        # Latent embedding Z: (n_bins, latent_dim) in Euclidean space
        # Co-trained with MLP and D via joint optimizer
        self.latent_embedding = nn.Embedding(n_bins, latent_dim)
        # Initialize with zeros to match baseline (Issue #5)
        # Note: latent_init_std kept in signature for compatibility but not used if 0.0
        if latent_init_std > 0.0:
            nn.init.normal_(self.latent_embedding.weight, mean=0.0, std=latent_init_std)
        else:
            nn.init.zeros_(self.latent_embedding.weight)
        # Gradient enabled: Z is co-trained with MLP and D via a joint optimizer

        # Scalar latent vector D (one value per BIN) added to MLP output, as in src/model.py
        # Solved analytically in Phase A (EM style) — NOT trained by gradient.
        self.latent_vec = nn.Parameter(
            torch.zeros(n_bins, device=device),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, bin_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with latent concatenated to input features and added to MLP output.

        Args:
            x: Input features [N, input_dim]
            bin_ids: BIN indices [N]

        Returns:
            Predicted logits [N] - one scalar per observation
            For cross-entropy loss, these are reshaped to [B, n_bins_per_sample]
            and softmax is applied across the bin dimension within each sample.
        """
        # Get latent embeddings for the given BINs: [N, latent_dim]
        latent = self.latent_embedding(bin_ids)
        
        # Concatenate features with latent: [N, input_dim + latent_dim]
        x_augmented = torch.cat([x, latent], dim=-1)
        
        # MLP output in logit space: [N]
        output = self.mlp(x_augmented)
        
        # Add scalar latent D to output (same as src/model.py: intrinsic + D)
        scalar_latent = self.latent_vec[bin_ids]
        output = output + scalar_latent
        
        return output

    @torch.no_grad()
    def set_latent(self, latent_new: Union[torch.Tensor, np.ndarray]) -> None:
        """Update the latent embedding Z after a Phase A solve."""
        if isinstance(latent_new, np.ndarray):
            latent_new = torch.tensor(latent_new, dtype=torch.float32)
        self.latent_embedding.weight.copy_(latent_new.to(self.device))

    @torch.no_grad()
    def set_latent_vec(self, latent_vec_new: Union[torch.Tensor, np.ndarray]) -> None:
        """Update the scalar latent vector D after a Phase A solve."""
        if isinstance(latent_vec_new, np.ndarray):
            latent_vec_new = torch.tensor(latent_vec_new, dtype=torch.float32)
        self.latent_vec.copy_(latent_vec_new.to(self.device))

    @torch.no_grad()
    def get_latent(self) -> np.ndarray:
        """Get current latent embeddings as numpy array [n_bins, latent_dim]."""
        return self.latent_embedding.weight.detach().cpu().numpy()

    @torch.no_grad()
    def predict_MLP_only(self, data_loader, loss_mode: str = "bin") -> np.ndarray:
        """Predict using MLP with current latent embeddings (not excluding latent).
        
        Note: In latent-as-input variant, MLP predictions inherently include latent effect.
        This method returns predictions for all observations using current frozen latents.
        
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
            all_preds = []
            
            for batch in data_loader:
                x = batch["input"].to(self.device)  # [B, max_bins, features]
                mask = batch.get("mask")  # [B, max_bins]
                bin_idx = batch["bin_idx"].to(self.device)  # [B, max_bins]
                
                B, max_bins, n_feat = x.shape
                x_flat = x.view(B * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(B * max_bins)
                
                # Get latent and concatenate
                latent_flat = self.latent_embedding(bin_idx_flat)
                x_augmented = torch.cat([x_flat, latent_flat], dim=-1)
                
                # MLP prediction
                output_flat = self.mlp(x_augmented)  # [B * max_bins, n_classes] or [B * max_bins]
                if output_flat.dim() == 1:
                    output = output_flat.view(B, max_bins)
                else:
                    output = output_flat.view(B, max_bins, -1)
                
                # Extract valid predictions (non-padded)
                output_np = output.cpu().numpy()
                if mask is not None:
                    mask_np = mask.cpu().numpy().astype(bool)
                else:
                    mask_np = np.ones((B, max_bins), dtype=bool)
                
                for b in range(B):
                    valid_mask = mask_np[b]
                    if output_np.ndim == 3:
                        all_preds.extend(output_np[b, valid_mask, :])
                    else:
                        all_preds.extend(output_np[b, valid_mask].tolist())
            
            return np.array(all_preds)
        else:
            # Bin mode: straightforward
            preds = []
            for batch in data_loader:
                x = batch["input"].to(self.device)
                bin_idx = batch["bin_idx"].to(self.device)
                
                # Get latent and concatenate
                latent = self.latent_embedding(bin_idx)
                x_augmented = torch.cat([x, latent], dim=-1)
                
                # MLP prediction
                output = self.mlp(x_augmented)
                preds.append(output.cpu().numpy())
            
            all_preds = np.concatenate(preds, axis=0)
            # Squeeze if single output dimension
            if all_preds.ndim == 2 and all_preds.shape[1] == 1:
                all_preds = all_preds.squeeze(-1)
            return all_preds
    
    
    def save_model(self, path: str) -> None:
        """Save the model state to the specified path."""
        print(f"Saving model to {path}")
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load the model state from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))