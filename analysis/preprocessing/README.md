# Read Count Preprocessing Analysis

This directory contains scripts to investigate the effect of different preprocessing methods on read count features in the metabarcoding model.

## Overview

The model uses four read count features: `total_reads_norm`, `avg_reads_norm`, `max_reads_norm`, and `min_reads_norm`. This analysis compares three preprocessing approaches:

1. **Original** (normalize + log): Normalize by sample total, then apply log1p transform
   ```python
   df["total_reads_norm"] = np.log1p(df["total_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
   ```

2. **Normalized** (normalize only): Only normalize by sample total, no log transform
   ```python
   df["total_reads_norm"] = df["total_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
   ```

3. **Logarithm** (log only): Only apply log1p transform, no sample normalization
   ```python
   df["total_reads_norm"] = np.log1p(df["total_reads"])
   ```

## Files

- `utils_test.py`: Modified version of `src/utils.py` with support for different preprocessing methods
- `read_count_preprocessing.py`: Training script that runs the model on all three preprocessing versions
- `preprocessing_visualization.py`: Visualization script to compare results
- `data/`: Contains the raw training data and subdirectories for each preprocessing method
  - `data/original/`: Datasets with original preprocessing (normalize + log)
  - `data/normalized/`: Datasets with normalized-only preprocessing
  - `data/logarithm/`: Datasets with logarithm-only preprocessing

## Usage

### Step 1: Generate Preprocessed Datasets

First, generate the three versions of the dataset with different preprocessing:

```bash
PYTHONPATH=../../src python utils_test.py --data_path data/ecuador_training_data.csv --seed 14
```

This will create three subdirectories in `data/` (original, normalized, logarithm), each containing:
- X_train.csv, X_val.csv, X_test.csv
- y_train.csv, y_val.csv, y_test.csv
- taxonomic_data.csv

### Step 2: Train Models on All Preprocessing Versions

Train the MLP + Latent model on each preprocessed dataset:

```bash
# With W&B logging (requires wandb to be installed and configured)
PYTHONPATH=../../src python read_count_preprocessing.py

# Without W&B logging
PYTHONPATH=../../src python read_count_preprocessing.py --no_wandb
```

This will:
- Load each preprocessed dataset
- Train the model with the same configuration and random seed
- Save results to `results/preprocessing_results.pkl`

**Note**: Training all three versions will take significant time (similar to running the full training pipeline 3 times).

### Step 3: Visualize Results

Generate comparison plots:

```bash
python preprocessing_visualization.py --results_path results/preprocessing_results.pkl --output_dir figures
```

This will create several visualization plots in the `figures/` directory:
- `metrics_comparison.png`: Bar charts comparing key metrics across all methods
- `scatter_actual_vs_predicted.png`: Scatter plots of actual vs predicted values
- `residual_distribution.png`: Distribution of prediction residuals
- `zero_vs_nonzero_mae.png`: MAE comparison for zero vs non-zero ground truth values
- `summary_table.png`: Comprehensive table of all metrics

## Metrics

The analysis computes the following metrics for each preprocessing method:

- **RMSE (micro)**: Root Mean Squared Error (overall)
- **MAE (micro)**: Mean Absolute Error (overall)
- **R² Score**: Coefficient of determination
- **Correlation**: Pearson correlation between predictions and ground truth
- **KL Divergence**: Kullback-Leibler divergence
- **Absolute Relative Error**: Mean relative error for non-zero values
- **RMSE/MAE (zeros)**: Metrics for zero ground truth values
- **RMSE/MAE (non-zeros)**: Metrics for non-zero ground truth values

## Expected Outcomes

This analysis will help determine:
1. Whether the log transformation helps stabilize variance
2. Whether sample normalization is necessary for model performance
3. Which preprocessing approach provides the best predictive accuracy
4. How each method affects prediction quality for rare vs common species

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- torch (for model training)

All dependencies should already be available if you can run the main training pipeline.
