from dataclasses import dataclass
from math import e
from typing import Optional, Literal
import numpy as np
import torch

@dataclass
class Config:
    # Data path
    data_path: str = "data/ecuador_training_data.csv"    # Path to raw data CSV file (e.g. data/ecuador_training_data.csv)
    results_dir: str = "../results" # Directory to save results (e.g. results/results_2023-01-01_12-00.pkl)
    
    # Train / val / test split
    train_frac: float = 0.8
    val_frac: float = 0.1

    # Neighbour graph
    use_taxonomy: bool = True           # set to True to use taxonomic distances
    use_embedding: bool = False         # set to True to use DNA embedding-based neighbors
    neighbor_mode: str = "knn"          # "threshold" for distance-based, "knn" for K-nearest neighbors
    K: int = 10                         # number of neighbors (used when neighbor_mode="knn")
    dist_thres: int = 4                 # max taxonomic distance (used when neighbor_mode="threshold")
    emb_radius: float = 1.0             # max embedding distance (used when neighbor_mode="threshold")
    kernel_q: Optional[float] = None    # kernel q parameter (if None, computed adaptively)

    # DNA embedding settings (used when use_embedding=True)
    embedding_path: Optional[str] = None       # path to precomputed embeddings (.npy dict: bin_uri->vector)
    barcode_data_path: Optional[str] = None    # path to TSV with 'bin_uri' and 'seq' columns
    emb_distance_metric: str = "cosine"        # distance metric: "cosine" or "euclidean"

    # Latent configuration - for latent-as-input variant
    latent_dim: int = 4                 # Dimension of latent embedding per BIN
    latent_init_std: float = 0.0        # Initialize with zeros to match baseline
    latent_lr: float = 2e-2             # Learning rate for latent embedding (separate param group)
    latent_norm_reg: float = 1e-6       # L2 norm regularization on latent
    latent_smooth_reg: float = 1e-4     # Smoothness regularization (parameter λ_smooth)
    latent_present_only: bool = False   # If True, only fit latent on observations where y > 0 (useful with loss='logistic' to avoid distribution shift)

    # Legacy parameters (kept for compatibility & reference but not used in gradient-based solver)
    latent_l2_reg: float = 1e-3         # L2 norm regularization on D (parameter r)
    cg_tol: float = 1e-6                # conjugate gradient tolerance (i.e., stopping criterion)
    cg_maxiter: int = 2000              # conjugate gradient max iterations
    
    # Architecture - New parameters for multiplicative gating
    embed_dim: int = 4                 # Embedding dimension d for vector latent
    gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid"  # Gating function type (sigmoid is primary)
    gating_alpha: float = 0.5           # Scaling factor for scaled_exp gating (in (0,1])
    gating_kappa: float = 0.5           # Scaling factor for tanh gating
    gating_epsilon: float = 0.693       # Offset for softplus gating (log(2), so g(0)=1)
    final_linear_wd: float = 1e-3       # Weight decay specifically for final linear layer w

    # Training - joint optimization of MLP and latent in a single pass
    device: str = "cpu"  # Force CPU to avoid MPS issues with nn.Embedding
    batch_size_bin: int = 1024          # Batch size (in number of observations not samples)
    batch_size_sample: int = 8          # Batch size in number of samples
    lr: float = 5e-4                    # Learning rate for MLP parameters
    weight_decay: float = 1e-5          # Weight decay for MLP parameters
    epochs: int = 500                   # Total training epochs (joint MLP + latent)
    max_cycles: int = 100               # Max training cycles
    dropout: float = 0.15               # Dropout rate in MLP
    grad_clip: Optional[float] = 1.0    # Gradient clipping value (None to disable)

    # Diagnostics: track latent importance during training
    diag_ablation_interval: int = 20    # Compute ablation delta every N epochs (0 = disabled)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
