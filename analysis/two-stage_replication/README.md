# Random Forest Models for Metabarcoding

Random Forest baseline models for predicting relative abundance from metabarcoding data.

## Overview

This module implements two approaches:

1. **Single-Stage Model** (`train.py`): A simple Random Forest Regressor
2. **Two-Stage Model** (`two_stage_model.py`): Handles zero-inflation in the data
   - **Stage 1**: Classifier predicts zero vs non-zero abundance
   - **Stage 2**: Regressor predicts abundance for non-zero samples

The two-stage approach is recommended due to ~58% zero-inflation in the target variable.

## Files

- `config.py` - Configuration dataclass for hyperparameters
- `train.py` - Single-stage Random Forest Regressor
- `evaluate.py` - Evaluation and visualization utilities
- `hyperparameter_search.py` - Grid/randomized search for single-stage model
- `two_stage_model.py` - **Two-stage classifier + regressor model**
- `two_stage_hyperparam_search.py` - Hyperparameter search for two-stage model

## Two-Stage Model (Recommended)

### Training

```bash
cd random_forest
python two_stage_model.py --save
```

With custom parameters:

```bash
python two_stage_model.py \
    --clf_n_estimators 200 \
    --clf_max_depth 20 \
    --reg_n_estimators 200 \
    --reg_max_depth 30 \
    --threshold 0.5 \
    --save
```

### Hyperparameter Search

Search for both stages:

```bash
python two_stage_hyperparam_search.py --stage both
```

Search only classifier or regressor:

```bash
python two_stage_hyperparam_search.py --stage classifier
python two_stage_hyperparam_search.py --stage regressor
```

### Evaluation Metrics

**Classifier (Stage 1)**:
- Accuracy
- Precision
- Recall
- **F1-Score** (primary metric)

**Regressor (Stage 2)**:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of determination)
- **MAPE** (Mean Absolute Percentage Error)
- **Median AE** (Median Absolute Error)

---

## Single-Stage Model

### Training

```bash
python train.py --save
```

Training with custom parameters:

```bash
python train.py \
    --n_estimators 200 \
    --max_depth 20 \
    --min_samples_split 5 \
    --min_samples_leaf 2 \
    --max_features sqrt \
    --save
```

### Evaluation

```bash
python evaluate.py --model_path models/rf_model_YYYY-MM-DD_HH-MM.pkl
```

### Hyperparameter Search

```bash
python hyperparameter_search.py --method random --n_iter 50
```

---

## Model Comparison

| Model | Test R² | Test MAE | Notes |
|-------|---------|----------|-------|
| Single-Stage RF | 0.506 | 0.00106 | Struggles with zeros |
| Two-Stage RF | 0.450 | 0.00125 | Better zero handling, F1=0.85 |

The two-stage model has slightly worse overall metrics but handles the zero-inflated nature of the data much better, correctly classifying ~89% of true zeros.

## Data

Uses the preprocessed data from `../data/`:
- `X_train.csv`, `X_val.csv`, `X_test.csv` - Features
- `y_train.csv`, `y_val.csv`, `y_test.csv` - Target (relative abundance)

**Zero-inflation**: ~58% of samples have zero relative abundance.

## Output

### Models
Saved to `models/` directory as pickle files.

### Results  
Saved to `results/` directory:
- CSV with metrics
- Prediction plots
- Feature importance
