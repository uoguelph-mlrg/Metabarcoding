# Gating Function Comparison

This analysis compares the baseline additive architecture with multiple multiplicative gating functions.

## Architecture Comparison

### Baseline (Additive)
- **Formula**: `ŷ = sigmoid(m(x) + d_bin)`
- Simple additive latent adjustment per BIN

### Multiplicative Gating Architectures

All multiplicative architectures use the form: `ŷ = sigmoid(w^T (m(x) ⊙ g(h_bin)))`

where `g(h)` is one of the following gating functions:

1. **Exponential (Primary)**: `g(h) = exp(h)`, g(0) = 1
2. **Scaled Exponential**: `g(h) = exp(α·h)`, α ∈ (0,1], g(0) = 1
3. **Additive**: `g(h) = 1 + h`, g(0) = 1
4. **Softplus**: `g(h) = 1 + softplus(h) - ε`, ε = log(2), g(0) = 1
5. **Tanh**: `g(h) = 1 + tanh(h)·κ`, g(0) = 1
6. **Sigmoid**: `g(h) = 2·σ(h)`, g(0) = 1
7. **Dot Product**: `z = m(x) · h_bin` (direct inner product between MLP output and latent vector)

Gating functions 1–6 satisfy `g(0) = 1` to ensure identity modulation at initialization. The dot product variant (7) is structurally distinct: it bypasses the final linear layer `w` entirely and uses the inner product `∑_i m_i(x) · h_i` as the raw logit directly.

## Usage

### 1. Run Comparison (Train All Architectures)

Train baseline and all 7 gating functions:

```bash
cd /Users/emmaboehly/Documents/Vector/Metabarcoding/analysis/dimensionality_increase

python dimensionality_increase.py \
    --data_path ../../data/ecuador_training_data.csv \
    --output_dir results \
    --no_wandb
```

Or train only specific gating functions:

```bash
python dimensionality_increase.py \
    --data_path ../../data/ecuador_training_data.csv \
    --gating_functions exp scaled_exp additive \
    --output_dir results \
    --no_wandb
```

### 2. Generate Visualizations

After training completes:

```bash
python dimensionality_increase_visualize.py \
    --results_path results/gating_comparison_results.pkl \
    --output_dir figures
```

## Output

### Results
- `results/gating_comparison_results.pkl` - Pickled results for all architectures

### Figures
- `figures/metrics_comparison.png` - Bar plots comparing key metrics
- `figures/scatter_predicted_vs_actual.png` - Scatter plots for all architectures
- `figures/scatter_zoomed.png` - Zoomed scatter plots (GT < 1%)
- `figures/scatter_loglog_predicted_vs_actual.png` - Log-log scatter plots
- `figures/error_by_range.png` - MAE by abundance range
- `figures/error_by_range_zoomed.png` - MAE by abundance range (< 1%)
- `figures/rae_by_range.png` - Relative absolute error by range
- `figures/residual_distribution.png` - Residual distributions
- `figures/zero_vs_nonzero_comparison.png` - Zero vs non-zero performance
- `figures/summary_table.png` - Summary table with all metrics
- `figures/architecture_comparison_results.csv` - CSV with numerical results

## Key Metrics

- **RMSE**: Root mean squared error (lower is better)
- **MAE**: Mean absolute error (lower is better)
- **Absolute Relative Error**: Mean relative error on non-zero values (lower is better)
- **KL Divergence**: Distribution divergence (lower is better)
- **Correlation**: Pearson correlation coefficient (higher is better)
- **MAE (zeros)**: MAE on zero ground truth values
- **MAE (non-zeros)**: MAE on non-zero ground truth values

## Implementation Notes

### Key Fixes Applied
1. **Removed bias from final linear layer**: `z = w^T m̃` (no bias term)
2. **All latent vectors initialize to h=0**: Ensures g(0)=1 for all gating functions
3. **Separate weight decay for w**: Stronger constraint on final linear weights
4. **Proper gating gradient computation**: Each function has correct derivative

### Architecture Differences
- **Baseline**: Single scalar latent per BIN, additive adjustment
- **Multiplicative**: Vector latent (d=8) per BIN, multiplicative gating with learned projection w

The multiplicative architectures are expected to outperform the baseline due to:
1. Higher expressiveness (d-dimensional vs scalar latent)
2. Feature-wise modulation (element-wise gating)
3. Learned readout (w) balances contributions across dimensions
