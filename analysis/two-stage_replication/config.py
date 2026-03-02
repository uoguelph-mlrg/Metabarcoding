"""
Configuration for Random Forest Regressor.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RandomForestConfig:
    # Random Forest hyperparameters
    n_estimators: int = 100             # Number of trees in the forest
    max_depth: Optional[int] = None     # Maximum depth of trees (None = unlimited)
    min_samples_split: int = 2          # Minimum samples required to split a node
    min_samples_leaf: int = 1           # Minimum samples required in a leaf node
    max_features: str = "sqrt"          # Number of features to consider for best split
    n_jobs: int = -1                    # Number of parallel jobs (-1 uses all cores)
    random_state: int = 42              # Random seed for reproducibility
    
    # Data split fractions
    train_frac: float = 0.8
    val_frac: float = 0.1
    
    # Data paths (relative to project root)
    data_dir: str = "data"
    
    # Output paths
    model_dir: str = "models"
    results_dir: str = "results"
    
    # Features to exclude (non-numeric or ID columns)
    exclude_columns: tuple = ("sample_id", "bin_uri")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
