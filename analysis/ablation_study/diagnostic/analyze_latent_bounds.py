#!/usr/bin/env python3
"""
Analyze whether extreme latent values are driven by rare BINs,
and test whether bounding the amplification factor helps.

The biological argument:
- Amplification factor d_b should depend only on genetic/taxonomic factors
- Reasonable amplification biases are typically 0.1x to 10x (not 4 million!)
- Rare BINs with few observations can have unconstrained latent values
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from train import Trainer, Config
from config import Config as BaseConfig


def main():
    print("=" * 70)
    print("ANALYZING LATENT FACTOR BOUNDS AND BIN FREQUENCY")
    print("=" * 70)
    
    # Load data directly
    data_path = Path(__file__).parent.parent.parent / "data" / "ecuador_training_data.csv"
    df = pd.read_csv(data_path)
    
    # Get unique samples and bins
    sample_ids = df['sample-eventid'].unique()
    bin_names = df['bin_uri'].unique()
    
    print(f"\nData: {len(sample_ids)} samples, {len(bin_names)} bins, {len(df)} observations")
    
    # Count observations per BIN
    bin_counts = df.groupby('bin_uri').size().to_dict()
    bin_count_array = np.array([bin_counts.get(b, 0) for b in bin_names])
    
    print(f"\nBIN observation counts:")
    print(f"  Min: {bin_count_array.min()}")
    print(f"  Max: {bin_count_array.max()}")
    print(f"  Mean: {bin_count_array.mean():.1f}")
    print(f"  Median: {np.median(bin_count_array):.1f}")
    
    # Train a model and get latent factors
    print("\n" + "=" * 70)
    print("TRAINING MODEL TO ANALYZE LATENT FACTORS")
    print("=" * 70)
    
    cfg = Config()
    cfg.max_epochs = 100
    cfg.no_wandb = True
    
    trainer = Trainer(cfg, str(data_path))
    trainer.run(use_wandb=False)
    
    # Get latent factors from the model's latent_vec parameter
    d_b = trainer.model.latent_vec.detach().cpu().numpy()  # Shape: (num_bins,)
    
    # Get bin_index from trainer to match bin_names order
    bin_index = trainer.bin_index
    bin_names_ordered = sorted(bin_index.keys(), key=lambda x: bin_index[x])
    bin_count_array = np.array([bin_counts.get(b, 0) for b in bin_names_ordered])
    amplification = np.exp(d_b)
    
    print(f"\nLatent factors (d_b):")
    print(f"  Min: {d_b.min():.4f} → amplification {amplification.min():.6f}")
    print(f"  Max: {d_b.max():.4f} → amplification {amplification.max():.2f}")
    print(f"  Mean: {d_b.mean():.4f}")
    print(f"  Std: {d_b.std():.4f}")
    
    # Analyze correlation between BIN frequency and extreme latent values
    print("\n" + "=" * 70)
    print("CORRELATION: BIN FREQUENCY vs LATENT MAGNITUDE")
    print("=" * 70)
    
    # Correlation between count and |d_b|
    corr_abs = np.corrcoef(bin_count_array, np.abs(d_b))[0, 1]
    corr_val = np.corrcoef(bin_count_array, d_b)[0, 1]
    
    print(f"\nCorrelation(bin_count, |d_b|): {corr_abs:.4f}")
    print(f"Correlation(bin_count, d_b): {corr_val:.4f}")
    
    # Look at extreme latent values
    extreme_positive = d_b > 5  # exp(5) ≈ 148x amplification
    extreme_negative = d_b < -5  # exp(-5) ≈ 0.007x amplification
    
    print(f"\nExtreme latent values (|d_b| > 5, i.e., amplification >148x or <0.007x):")
    print(f"  Positive extreme: {extreme_positive.sum()} BINs")
    print(f"  Negative extreme: {extreme_negative.sum()} BINs")
    
    if extreme_positive.sum() > 0:
        print(f"\n  BINs with extreme POSITIVE latent (high amplification):")
        print(f"    Mean observation count: {bin_count_array[extreme_positive].mean():.1f}")
        print(f"    vs overall mean: {bin_count_array.mean():.1f}")
        
    if extreme_negative.sum() > 0:
        print(f"\n  BINs with extreme NEGATIVE latent (low amplification):")
        print(f"    Mean observation count: {bin_count_array[extreme_negative].mean():.1f}")
        print(f"    vs overall mean: {bin_count_array.mean():.1f}")
    
    # Stratify by observation count
    print("\n" + "=" * 70)
    print("LATENT STATISTICS BY BIN FREQUENCY")
    print("=" * 70)
    
    # Quartiles of bin counts
    q25, q50, q75 = np.percentile(bin_count_array, [25, 50, 75])
    
    rare_mask = bin_count_array <= q25
    common_mask = bin_count_array >= q75
    medium_mask = ~rare_mask & ~common_mask
    
    print(f"\nRare BINs (≤{q25:.0f} obs, n={rare_mask.sum()}):")
    print(f"  d_b mean: {d_b[rare_mask].mean():.4f}, std: {d_b[rare_mask].std():.4f}")
    print(f"  Amplification range: [{amplification[rare_mask].min():.6f}, {amplification[rare_mask].max():.2f}]")
    
    print(f"\nMedium BINs ({q25:.0f}-{q75:.0f} obs, n={medium_mask.sum()}):")
    print(f"  d_b mean: {d_b[medium_mask].mean():.4f}, std: {d_b[medium_mask].std():.4f}")
    print(f"  Amplification range: [{amplification[medium_mask].min():.6f}, {amplification[medium_mask].max():.2f}]")
    
    print(f"\nCommon BINs (≥{q75:.0f} obs, n={common_mask.sum()}):")
    print(f"  d_b mean: {d_b[common_mask].mean():.4f}, std: {d_b[common_mask].std():.4f}")
    print(f"  Amplification range: [{amplification[common_mask].min():.6f}, {amplification[common_mask].max():.2f}]")
    
    # Create visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter: BIN count vs latent value
    ax = axes[0, 0]
    ax.scatter(bin_count_array, d_b, alpha=0.3, s=10)
    ax.set_xlabel('Observation count per BIN')
    ax.set_ylabel('Latent factor (d_b)')
    ax.set_title(f'BIN Frequency vs Latent Factor\n(corr={corr_val:.3f})')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='|d|=5')
    ax.axhline(y=-5, color='orange', linestyle='--', alpha=0.5)
    
    # 2. Histogram of latent values
    ax = axes[0, 1]
    ax.hist(d_b, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Latent factor (d_b)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Latent Factors')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=np.log(10), color='green', linestyle='--', alpha=0.5, label='10x amp')
    ax.axvline(x=-np.log(10), color='green', linestyle='--', alpha=0.5)
    
    # 3. Amplification factor distribution (log scale)
    ax = axes[1, 0]
    log_amp = np.log10(amplification + 1e-10)
    ax.hist(log_amp, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('log10(Amplification Factor)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Amplification Factors')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='1x')
    ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='10x')
    ax.axvline(x=-1, color='green', linestyle='--', alpha=0.5, label='0.1x')
    
    # 4. Box plot by frequency category
    ax = axes[1, 1]
    data_to_plot = [d_b[rare_mask], d_b[medium_mask], d_b[common_mask]]
    bp = ax.boxplot(data_to_plot, labels=['Rare', 'Medium', 'Common'])
    ax.set_xlabel('BIN Frequency Category')
    ax.set_ylabel('Latent factor (d_b)')
    ax.set_title('Latent Values by BIN Frequency')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    fig_path = Path(__file__).parent / "figures" / "latent_frequency_analysis.png"
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close()
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    print("""
The biological model is valid:
- Amplification factor should depend on genetic/taxonomic factors (BIN identity)
- Sample features affect base prediction, not amplification

The implementation issue:
- Latent factors are unconstrained, leading to extreme values
- Rare BINs can have arbitrary latent values (overfitting)

Potential fixes:
1. Stronger L2 regularization on latent factors
2. Bound latent factors: d_b ∈ [-log(10), log(10)] for 0.1x to 10x
3. Use taxonomy to share information across related BINs (hierarchical prior)
4. Only estimate latent for BINs with sufficient observations
""")
    
    # What fraction of BINs have "reasonable" amplification?
    reasonable_mask = (amplification >= 0.1) & (amplification <= 10)
    print(f"\nBINs with 'reasonable' amplification (0.1x to 10x): {reasonable_mask.sum()}/{len(amplification)} ({100*reasonable_mask.mean():.1f}%)")
    
    # Suggest: what if we clip latent values?
    d_b_clipped = np.clip(d_b, -np.log(10), np.log(10))  # 0.1x to 10x
    print(f"\nIf we clip d_b to [-{np.log(10):.2f}, {np.log(10):.2f}]:")
    print(f"  {(d_b != d_b_clipped).sum()} BINs would be affected")


if __name__ == "__main__":
    main()
