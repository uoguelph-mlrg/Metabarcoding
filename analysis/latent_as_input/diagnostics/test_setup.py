#!/usr/bin/env python
"""
Quick test to verify imports and basic functionality of the latent-as-input variant.
"""

import sys
import os

# Add src to path AFTER current directory
# This ensures local config.py takes precedence over src/config.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    
    try:
        from config import Config, set_seed
        print("✓ Local config imported")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from model import Model
        print("✓ Local model imported")
    except ImportError as e:
        print(f"✗ Failed to import model: {e}")
        return False
    
    try:
        from latent_solver import LatentSolver
        print("✓ Local latent_solver imported")
    except ImportError as e:
        print(f"✗ Failed to import latent_solver: {e}")
        return False
    
    try:
        from neighbor_graph import NeighbourGraph
        from dataset import MBDataset
        from loss import Loss
        from mlp import MLPModel
        from utils import load
        print("✓ All src modules imported")
    except ImportError as e:
        print(f"✗ Failed to import from src: {e}")
        return False
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        print("✓ External dependencies imported")
    except ImportError as e:
        print(f"✗ Failed to import external dependency: {e}")
        return False
    
    return True


def test_config():
    """Test that config has all required parameters."""
    print("\nTesting config...")
    from config import Config
    
    cfg = Config()
    required_attrs = [
        'latent_input_dim', 'latent_init_std', 'latent_lr', 'latent_steps',
        'latent_norm_reg', 'latent_smooth_reg', 'latent_present_only'
    ]
    
    for attr in required_attrs:
        if not hasattr(cfg, attr):
            print(f"✗ Config missing attribute: {attr}")
            return False
    
    print(f"✓ Config complete (latent_input_dim={cfg.latent_input_dim}, latent_lr={cfg.latent_lr})")
    return True


def test_model_creation():
    """Test that model can be created with correct architecture."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from model import Model
        from mlp import MLPModel
        from latent_solver import LatentSolver
        from neighbor_graph import NeighbourGraph
        from config import Config
        import pandas as pd
        import numpy as np
        
        cfg = Config()
        
        # Create dummy neighbor graph
        bins_df = pd.DataFrame({
            'bin_uri': [f'BIN_{i}' for i in range(10)],
            'species': [f'species_{i % 3}' for i in range(10)],
            'genus': [f'genus_{i % 2}' for i in range(10)],
        })
        ng = NeighbourGraph(cfg, bins_df)
        ng.build()
        
        # Create latent solver
        solver = LatentSolver(cfg, ng)
        
        # Create MLP (with adjusted input dimension)
        input_dim = 10
        mlp_input_dim = input_dim + cfg.latent_input_dim
        mlp = MLPModel(mlp_input_dim, hidden_dims=cfg.mlp_hidden_dims, dropout=0.1)
        
        # Create model
        model = Model(
            mlp=mlp,
            latent_solver=solver,
            n_bins=10,
            latent_input_dim=cfg.latent_input_dim,
            latent_init_std=cfg.latent_init_std,
            device=torch.device('cpu')
        )
        
        print(f"✓ Model created successfully")
        print(f"  - MLP input dim: {mlp_input_dim}")
        print(f"  - Latent embedding: {model.latent_embedding.weight.shape}")
        print(f"  - Number of BINs: {model.n_bins}")
        
        # Test forward pass
        x = torch.randn(5, input_dim)
        bin_ids = torch.randint(0, 10, (5,))
        output = model(x, bin_ids)
        print(f"  - Forward pass output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("Latent-as-Input Variant - Import and Architecture Test")
    print("="*60)
    
    success = True
    success &= test_imports()
    success &= test_config()
    success &= test_model_creation()
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
        print("\nYou can now run training with:")
        print("  python train.py --loss_type cross_entropy")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*60)
