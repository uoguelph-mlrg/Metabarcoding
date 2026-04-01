from dataclasses import dataclass
from math import e
from typing import Optional, Literal
import numpy as np
import torch

@dataclass
class Config:
    # Run configuration
    data_path: str = "data/data_merged.csv"    # Path to raw data CSV file
    results_dir: str = "../results"                      # Directory where run artifacts are saved
    loss_type: Literal["cross_entropy", "logistic"] = "cross_entropy"
    checkpoint_every: int = 5                             # Save periodic checkpoint every N epochs
    
    # Train / val / test split
    train_frac: float = 0.8
    val_frac: float = 0.1

    # Neighbour graph
    use_taxonomy: bool = False           # set to True to use taxonomic distances
    use_embedding: bool = True         # set to True to use DNA embedding-based neighbors
    neighbor_mode: str = "knn"          # "threshold" for distance-based, "knn" for K-nearest neighbors
    K: int = 10                         # number of neighbors (used when neighbor_mode="knn")
    dist_thres: int = 4                 # max taxonomic distance (used when neighbor_mode="threshold")
    emb_radius: float = 1.0             # max embedding distance (used when neighbor_mode="threshold")
    kernel_q: Optional[float] = None    # kernel q parameter (if None, computed adaptively)

    # DNA embedding settings (used when use_embedding=True)
    embedding_path: Optional[str] = None       # path to precomputed embeddings (.npy dict: bin_uri->vector)
    barcode_data_path: Optional[str] = "../../../data/data_merged.csv"    # path to TSV with 'bin_uri' and 'seq' columns
    emb_distance_metric: str = "cosine"        # distance metric: "cosine" or "euclidean"

    # Latent solver - regularization settings
    latent_smooth_reg: float = 1e-3     # Smoothness regularization (parameter λ_smooth)
    latent_present_only: bool = False   # If True, only fit latent on observations where y > 0 (useful with loss='logistic' to avoid distribution shift)
    latent_l2_reg: float = 1e-3         # L2 norm regularization on D (parameter r)
    
    latent_convergence_tol: float = 1e-5    # Stopping tolerance for latent L-BFGS solves
    latent_convergence_maxiter: int = 300   # Max iterations for latent L-BFGS solves
    
    # Architecture - New parameters for multiplicative gating
    embed_dim: int = 10                 # Embedding dimension d for vector latent
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
    batch_size_sample: int = 8          # Batch size in number of samples
    lr: float = 5e-4                    # Learning rate for MLP parameters
    latent_lr: float = 1e-2             # Latent learning rate (rescaled, new parameter)
    weight_decay: float = 1e-5          # Weight decay for MLP parameters
    latent_warmup_frac: float = 0.2     # Fraction of epochs over which proximal damping decays from ρ0 → 0
    latent_prox_scale: float = 50.0     # ρ0 = latent_prox_scale × latent_l2_reg at epoch 0
    epochs: int = 200                   # Epochs per training phase
    dropout: float = 0.15               # Dropout rate in MLP
    grad_clip: Optional[float] = 1.0    # Gradient clipping value (None to disable)
    patience: Optional[int] = 25        # Patience for early stopping in number of cycles 

    # Location embedding settings
    use_location_embedding: bool = False
    location_embedder_model: str = "satclip"
    keep_raw_gps_features: bool = False
    location_embedding_prefix: str = "loc_emb"
    location_embedder_device: str = "cpu"
    location_embedder_batch_size: int = 2048
    satclip_ckpt_path: Optional[str] = None
    range_db_path: Optional[str] = 'third_party/RANGE/pretrained/range_db_med.npz'
    range_model_name: str = "RANGE+"
    range_beta: float = 0.5
    alphaearth_year: int = 2024
    alphaearth_scale_meters: int = 10
    alphaearth_project: Optional[str] = 'metabarcoding-491221'


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
