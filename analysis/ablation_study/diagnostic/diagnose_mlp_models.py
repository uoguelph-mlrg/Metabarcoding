#!/usr/bin/env python3
"""
Diagnostic script to understand why MLP without taxonomy outperforms MLP with taxonomy.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import pandas as pd
from config import Config, set_seed

# Import the data loading function
from ablation_study import load_data_with_taxonomy, TAXONOMY_COLS

def main():
    print("=" * 70)
    print("DIAGNOSING MLP MODEL DATA")
    print("=" * 70)
    
    data_path = os.path.join(os.path.dirname(__file__), '../../data/ecuador_training_data.csv')
    cfg = Config()
    set_seed(14)
    
    # Load data WITHOUT taxonomy
    print("\n" + "=" * 70)
    print("DATA WITHOUT TAXONOMY")
    print("=" * 70)
    
    data_no_tax, _, bin_index, sample_index, _, split_indices = load_data_with_taxonomy(
        data_path, cfg, include_taxonomy=False
    )
    
    X_train_no_tax = data_no_tax["train"]["X"]
    y_train_no_tax = data_no_tax["train"]["y"]
    
    print(f"\nTrain X shape: {X_train_no_tax.shape}")
    print(f"Feature columns: {list(X_train_no_tax.columns)}")
    print(f"Train y shape: {y_train_no_tax.shape}")
    print(f"Train y stats: mean={y_train_no_tax.mean():.6f}, std={y_train_no_tax.std():.6f}")
    print(f"  min={y_train_no_tax.min():.6f}, max={y_train_no_tax.max():.6f}")
    print(f"  % zeros: {(y_train_no_tax == 0).mean()*100:.1f}%")
    
    # Load data WITH taxonomy
    print("\n" + "=" * 70)
    print("DATA WITH TAXONOMY")
    print("=" * 70)
    
    set_seed(14)
    data_with_tax, taxonomy_df, _, _, encoders, _ = load_data_with_taxonomy(
        data_path, cfg, include_taxonomy=True, fixed_split_indices=split_indices
    )
    
    X_train_with_tax = data_with_tax["train"]["X"]
    y_train_with_tax = data_with_tax["train"]["y"]
    
    print(f"\nTrain X shape: {X_train_with_tax.shape}")
    print(f"Feature columns: {list(X_train_with_tax.columns)}")
    print(f"Train y shape: {y_train_with_tax.shape}")
    
    # Check which taxonomy levels are included
    print(f"\nTaxonomy encoders: {list(encoders.keys()) if encoders else 'None'}")
    if encoders:
        for col, le in encoders.items():
            print(f"  {col}: {len(le.classes_)} unique values")
    
    # Check feature value ranges
    print("\n" + "=" * 70)
    print("FEATURE STATISTICS (Train set)")
    print("=" * 70)
    
    # Original features (should be same in both)
    original_features = [c for c in X_train_no_tax.columns if not c.endswith('_encoded')]
    
    print("\n--- Original features (shared) ---")
    for col in original_features[:5]:
        no_tax_vals = X_train_no_tax[col] if col in X_train_no_tax.columns else None
        with_tax_vals = X_train_with_tax[col] if col in X_train_with_tax.columns else None
        
        if no_tax_vals is not None and with_tax_vals is not None:
            match = np.allclose(no_tax_vals.values, with_tax_vals.values, rtol=1e-5, equal_nan=True)
            print(f"  {col}: identical={match}")
    
    # Check taxonomy features
    print("\n--- Taxonomy features (with taxonomy only) ---")
    tax_features = [c for c in X_train_with_tax.columns if c.endswith('_encoded')]
    for col in tax_features:
        vals = X_train_with_tax[col]
        print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")
    
    # Check if targets are the same
    print("\n" + "=" * 70)
    print("TARGET COMPARISON")
    print("=" * 70)
    
    # Align indices for comparison
    common_idx = y_train_no_tax.index.intersection(y_train_with_tax.index)
    print(f"Common training indices: {len(common_idx)}")
    
    y_no_tax_aligned = y_train_no_tax.loc[common_idx]
    y_with_tax_aligned = y_train_with_tax.loc[common_idx]
    
    targets_match = np.allclose(y_no_tax_aligned.values, y_with_tax_aligned.values)
    print(f"Targets identical: {targets_match}")
    
    if not targets_match:
        diff = np.abs(y_no_tax_aligned.values - y_with_tax_aligned.values)
        print(f"  Max diff: {diff.max():.10f}")
        print(f"  Mean diff: {diff.mean():.10f}")
    
    # Check model capacity
    print("\n" + "=" * 70)
    print("MODEL CAPACITY COMPARISON")
    print("=" * 70)
    
    n_features_no_tax = X_train_no_tax.shape[1]
    n_features_with_tax = X_train_with_tax.shape[1]
    n_bins = len(bin_index)
    embedding_dim = 32
    hidden_dims = [128, 64]
    
    # MLPWithBinEmbedding: input = n_features + embedding_dim
    input_no_tax = n_features_no_tax + embedding_dim
    input_with_tax = n_features_with_tax + embedding_dim
    
    # Count parameters
    def count_params(input_dim, hidden_dims, n_bins, embedding_dim):
        params = n_bins * embedding_dim  # Embedding
        prev = input_dim
        for h in hidden_dims:
            params += prev * h + h  # Linear + bias
            prev = h
        params += prev + 1  # Output layer
        return params
    
    params_no_tax = count_params(input_no_tax, hidden_dims, n_bins, embedding_dim)
    params_with_tax = count_params(input_with_tax, hidden_dims, n_bins, embedding_dim)
    
    print(f"MLP no taxonomy:")
    print(f"  Input features: {n_features_no_tax}")
    print(f"  Input to MLP (features + embedding): {input_no_tax}")
    print(f"  Approx parameters: {params_no_tax:,}")
    
    print(f"\nMLP with taxonomy:")
    print(f"  Input features: {n_features_with_tax}")
    print(f"  Input to MLP (features + embedding): {input_with_tax}")
    print(f"  Approx parameters: {params_with_tax:,}")
    
    print(f"\nDifference: +{params_with_tax - params_no_tax:,} parameters ({n_features_with_tax - n_features_no_tax} extra features)")
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("""
The taxonomy features are:
1. Label-encoded integers (normalized) representing categorical levels
2. They add ~7 features (phylum, class, order, family, subfamily, genus, species)
3. This increases model capacity but may also add noise

Possible reasons MLP without taxonomy performs better:
1. Overfitting: More features = more capacity = more overfitting on 80 train samples
2. Redundancy: Taxonomy info may already be captured by the bin embedding
3. Noise: Normalized label encodings may not capture taxonomic relationships well
4. Sample size: With only 80 training samples, simpler models may generalize better

The bin embedding (32 dimensions) already captures bin-specific patterns,
which implicitly includes taxonomic information since related species have
similar evolutionary histories and likely similar amplification patterns.
""")


if __name__ == "__main__":
    main()
