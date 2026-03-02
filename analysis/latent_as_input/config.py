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
    latent_lr: float = 2e-2             # Learning rate for latent embedding (separate param group)
    latent_norm_reg: float = 1e-3       # L2 norm regularization on latent
    latent_smooth_reg: float = 1e-3     # Smoothness regularization (lambda_smooth)
    latent_present_only: bool = False   # If True, only fit latent on observations where y > 0 (useful with loss='logistic' to avoid distribution shift)

    # Legacy parameters (kept for compatibility but not used in gradient-based solver)
    latent_l2_reg: float = 1e-2         # Kept for reference
    cg_tol: float = 1e-6                # Kept for reference
    cg_maxiter: int = 500               # Kept for reference

    # Training - joint optimization of MLP and latent in a single pass
    device: str = "cpu"  # Force CPU to avoid MPS issues with nn.Embedding
    batch_size_bin: int = 1024          # Batch size (in number of observations not samples)
    batch_size_sample: int = 32         # Batch size in number of samples
    lr: float = 5e-4                    # Learning rate for MLP parameters
    weight_decay: float = 1e-5          # Weight decay for MLP parameters
    epochs: int = 300                   # Total training epochs (joint MLP + latent)
    dropout: float = 0.15               # Dropout rate in MLP
    grad_clip: Optional[float] = 1.0    # Gradient clipping value (None to disable)
    patience: Optional[int] = 30        # Patience for early stopping (in epochs)

    # Latent pre-warmup: freeze MLP and train only the latent embedding for K epochs before
    # joint training begins. This prevents the MLP from learning to ignore the latent signal
    # (feature-deactivation problem: zero-init latent → MLP never sees informative latent →
    # latent input weights decay toward zero and are never recovered in joint training).
    latent_warmup_epochs: int = 20      # Epochs of latent-only pre-training (0 = disabled)

    # Diagnostics: track latent importance during training
    diag_ablation_interval: int = 20    # Compute ablation delta every N epochs (0 = disabled)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
