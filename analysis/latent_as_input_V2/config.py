from dataclasses import dataclass
from math import e
from typing import Optional
import numpy as np
import torch

@dataclass
class Config:
    # Train / val / test split
    train_frac: float = 0.8
    val_frac: float = 0.1

    # Neighbour graph
    use_taxonomy: bool = True           # set to True to use taxonomic distances
    use_embedding: bool = False         # TODO: implement embedding-based neighbors
    neighbor_mode: str = "knn"    # "threshold" for distance-based, "knn" for K-nearest neighbors
    K: int = 10                         # number of neighbors (used when neighbor_mode="knn")
    dist_thres: int = 4                 # max taxonomic distance (used when neighbor_mode="threshold")
    emb_radius: float = 1.0             # max embedding distance (used when neighbor_mode="threshold")
    kernel_q: Optional[float] = None    # kernel q parameter (if None, computed adaptively)

    # Latent configuration - for latent-as-input variant
    latent_dim: int = 4                 # Dimension of latent embedding per BIN
    latent_init_std: float = 0.0        # Initialize with zeros to match baseline
    latent_lr: float = 5e-3             # Learning rate for Z (embedding) optimization
    latent_scalar_lr: float = 5e-3      # Learning rate for D (scalar latent vec) optimization
    # Staged Phase A: Z is optimised first (z_frac of total steps), then D.
    # This prevents D from hijacking all gradients before Z has a chance to develop.
    latent_steps: int = 100             # Total gradient steps per EM cycle
    latent_z_frac: float = 0.6          # Fraction of steps dedicated to Z (remainder goes to D)
    latent_norm_reg: float = 1e-2       # L2 norm regularisation for D (lambda_norm)
    # Z regularisation is scaled by 1/latent_dim so per-element strength equals D.
    # Without this, Z gets 4x more total penalty than D, suppressing it to near-zero.
    latent_z_norm_reg_factor: float = 1.0  # Multiplier on latent_norm_reg/latent_dim for Z norm
    latent_smooth_reg: float = 1e-3     # Smoothness regularisation (lambda_smooth)
    latent_present_only: bool = False   # If True, only fit latent on y > 0 observations
    
    # Legacy parameters (kept for compatibility but not used in gradient-based solver)
    latent_l2_reg: float = 1e-2         # Kept for reference
    cg_tol: float = 1e-6                # Kept for reference
    cg_maxiter: int = 500               # Kept for reference

    # Training - Adjusted for better convergence
    device: str = "cpu"  # Force CPU to avoid MPS issues with nn.Embedding
    batch_size_bin: int = 1024          # Batch size (in number of observations not samples)
    batch_size_sample: int = 32         # Batch size in number of samples (Issue #2: increased from 8 for conservative fix)
    lr: float = 5e-4                    # Learning rate - increased for faster initial learning
    weight_decay: float = 1e-5          # Light regularization
    epochs_init: int = 25               # Initial epochs training only MLP (reduced from 100 to avoid distribution shift)
    epochs: int = 10                    # Epochs per training phase - increased
    max_cycles: int = 100              # Max training cycles (matched to baseline)
    dropout: float = 0.15               # Dropout rate in MLP - slightly increased
    grad_clip: Optional[float] = 1.0    # Gradient clipping value (None to disable)
    patience: Optional[int] = 20        # Patience for early stopping in number of cycles - increased


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
