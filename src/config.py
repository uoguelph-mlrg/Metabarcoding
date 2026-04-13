from dataclasses import dataclass
from typing import Optional, Literal, List
import os
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@dataclass
class Config:
    # Run configuration
    data_path: str = os.path.join(PROJECT_ROOT, "data", "data_merged.csv")  # Path to raw data CSV file
    results_dir: str = "../results"             # Directory where run artifacts are saved
    checkpoint_every: int = 5                   # Save periodic checkpoint every N epochs
    diag_ablation_interval: int = 20            # Compute latent ablation delta every N epochs (0 = disabled)

    # Train / val / test split
    train_frac: float = 0.8
    val_frac: float = 0.1
    
    # Basic training settings
    loss_type: Literal["cross_entropy", "logistic"] = "cross_entropy"
    device: str = (
        "mps" if torch.backends.mps.is_available() else 
        "cuda" if torch.cuda.is_available() else 
        "cpu"
    )
    batch_size_bin: int = 1024                  # Batch size (in number of observations not samples)
    batch_size_sample: int = 8                  # Batch size in number of samples
    epochs: int = 200                           # Epochs per training phase
    grad_clip: Optional[float] = 1.0            # Gradient clipping value (None to disable)

    # Neighbour graph
    use_taxonomy: bool = False                  # set to True to use taxonomic distances
    use_embedding: bool = True                  # set to True to use DNA embedding-based neighbors
    neighbor_mode: str = "knn"                  # "threshold" for distance-based, "knn" for K-nearest neighbors
    K: int = 25                                 # number of neighbors (used when neighbor_mode="knn")
    dist_thres: int = 4                         # max taxonomic distance (used when neighbor_mode="threshold")
    emb_radius: float = 1.0                     # max embedding distance (used when neighbor_mode="threshold")
    kernel_q: Optional[float] = None            # kernel q parameter (if None, computed adaptively)
    interpolation_method: Literal["nw", "llr"] = "nw"  # interpolation method for latent solver: "nw" for Nadaraya-Watson, "llr" for locally linear regression

    # DNA embedding settings (used when use_embedding=True)
    embedding_path: Optional[str] = os.path.join(PROJECT_ROOT, "data", "embeddings.npy")  # path to precomputed embeddings (.npy dict: bin_uri->vector)
    barcode_data_path: Optional[str] = None     # path to TSV with 'bin_uri' and 'seq' columns
    emb_distance_metric: str = "cosine"         # distance metric: "cosine" or "euclidean"

    # MLP - architecture & optimization settings
    mlp_hidden_dims : List[int] = [128, 128, 128, 128]  # Hidden layer dimensions for MLP
    mlp_lr: float = 5e-4                        # Learning rate for MLP parameters
    weight_decay: float = 1e-5                  # Weight decay for MLP parameters
    mlp_warmup_start_factor: float = 1e-3       # Initial multiplier for MLP LR warmup
    mlp_warmup_frac: float = 0.1                # Fraction of total training steps used for MLP LR warmup
    mlp_lr_eta_min: float = 1e-6                # Minimum MLP LR reached by cosine decay
    dropout: float = 0.15                       # Dropout rate in MLP

    # Latent solver - regularization settings
    latent_smooth_reg: float = 1e-3             # Smoothness regularization (parameter λ_smooth)
    latent_present_only: bool = False           # If True, only fit latent on observations where y > 0 (useful with loss='logistic' to avoid distribution shift)
    latent_l2_reg: float = 1e-3                 # L2 norm regularization on D (parameter r)
    latent_init_prox_reg: float = 0.0           # Initial proximal regularization weight; annealed to 0 across epochs to stabilize early active-set latent updates.

    # Latent solver - optimization settings
    latent_optim_steps: int = 15                # Number of latent optimization steps per batch / solver call
    latent_lr: float = 1e-2                     # Learning rate for the latent AdamW optimizer
    latent_init_std: float = 0.0                # Standard deviation for initializing latent embeddings (0 for zeros, >0 for Gaussian noise)
    latent_warmup_start_factor: float = 1e-3    # Initial multiplier for latent LR warmup
    latent_warmup_frac: float = 0.2             # Fraction of total latent solves used for warmup
    latent_lr_eta_min: float = 1e-6             # Minimum latent LR reached by cosine decay
    latent_k_hop_mode: Literal["threshold", "knn"] = "threshold"  # Method for selecting subset of neighbors for latent optimization 
    latent_k_hop_threshold: int = 2             # Number of neighbor graph hops to select BINs from (used when latent_k_hop_mode="threshold")
    latent_hop_knn_cap: int = 64                # Max number of neighbors to include in latent optimization (used when latent_k_hop_mode="knn")

    # Training with interpolated latents settings
    interpolated_sample_fraction: float = 0.0   # Fraction of training samples using interpolated latent (set to 0 to disable interpolation during training)
    train_MLP_with_interpolation: bool = False  # Whether to train the MLP on interpolated latents too (instead of only using them in the latent solver)
    inference_with_interpolation: bool = False  # Whether to use interpolated latents during inference (if False, uses BINs own latent)
    include_self_in_interpolation: bool = False # Whether to include the BIN's own latent in the interpolation (instead of only using neighbors)

    # Sizes and combination modalities for latent and intrinsic vectors
    embed_dim: int = 10                         # Embedding dimension d for both latent and intrinsic vectors (set to 1 for scalars)
    gating_fn: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"] = "sigmoid"  # Gating function for combining latent and intrinsic vectors
    gating_alpha: float = 0.5                   # Scaling factor for scaled_exp gating (in (0,1])
    gating_kappa: float = 0.5                   # Scaling factor for tanh gating
    gating_epsilon: float = 0.693               # Offset for softplus gating (log(2), so g(0)=1)
    final_linear_weight_decay: float = 1e-3     # Weight decay specifically for final linear layer w

def cpu_if_mps(device: torch.device) -> torch.device:
    """Return CPU when device is MPS.

    Sparse CSR operations (used by the latent solver and interpolation operators)
    are numerically unstable on MPS. All callers that need a sparse-safe device
    should use this helper instead of duplicating the check.
    """
    return torch.device("cpu") if device.type == "mps" else device


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
