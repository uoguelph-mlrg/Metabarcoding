from dataclasses import dataclass
from math import e
from typing import Optional, Literal
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

    # Latent solver - regularization settings
    latent_l2_reg: float = 1e-2         # L2 regularization on H (parameter r)
    latent_smooth_reg: float = 1e-3     # smoothness regularization (parameter λ)
    latent_present_only: bool = True    # If True, only fit latent on observations where y > 0
    cg_tol: float = 1e-6                # conjugate gradient tolerance (i.e., stopping criterion)
    cg_maxiter: int = 500               # conjugate gradient max iterations

    # Architecture - New parameters for multiplicative gating
    embed_dim: int = 8                  # Embedding dimension d for vector latent
    gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid"  # Gating function type (sigmoid is primary)
    gating_alpha: float = 0.5           # Scaling factor for scaled_exp gating (in (0,1])
    gating_kappa: float = 0.5           # Scaling factor for tanh gating
    gating_epsilon: float = 0.693       # Offset for softplus gating (log(2), so g(0)=1)
    final_linear_wd: float = 1e-3       # Weight decay specifically for final linear layer w

    # Training - Adjusted for better convergence
    device: str = (
        "mps" if torch.backends.mps.is_available() else 
        "cuda" if torch.cuda.is_available() else 
        "cpu"
    )
    batch_size_bin: int = 1024          # Batch size (in number of observations not samples)
    batch_size_sample: int = 8          # Batch size in number of samples - reduced for stability
    lr: float = 5e-4                    # Learning rate - increased for faster initial learning
    weight_decay: float = 1e-5          # Light regularization
    epochs_init: int = 100              # Initial epochs training only MLP - increased
    epochs: int = 10                    # Epochs per training phase - increased
    max_cycles: int = 100               # Max training cycles - reduced (early stopping will kick in)
    dropout: float = 0.15               # Dropout rate in MLP - slightly increased
    grad_clip: Optional[float] = 1.0    # Gradient clipping value (None to disable)
    patience: Optional[int] = 5         # Patience for early stopping in number of cycles - increased


def set_seed(seed: int = 14) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
