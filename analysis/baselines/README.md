# Metabarcoding Baseline Models

This folder contains a set of baseline models for predicting relative abundance from metabarcoding data.

## Models Included

### Simple Baselines
- **Mean Baseline**: Always predicts the training mean
- **Median Baseline**: Always predicts the training median
- **Zero Baseline**: Always predicts zero (useful sanity check for zero-inflated data)

### Linear Models
- **Linear Regression**: Standard OLS regression
- **Ridge Regression**: L2-regularized linear regression
- **ElasticNet**: L1 + L2 regularized linear regression

### Tree-Based Models
- **Decision Tree**: Single decision tree regressor
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential boosting ensemble

### Other Models
- **KNN**: K-Nearest Neighbors regression

### Zero-Inflated Models (specialized for handling excess zeros)
- **Two-Stage (Hurdle) Model**: Combines classification (presence/absence) + regression (abundance given presence)
- **Zero-Inflated Ridge**: Logistic regression for zeros + Ridge regression on log-transformed positives
- **Tweedie Regression**: GLM with Tweedie distribution (naturally handles zeros)
- **Log-Transform Model**: Random Forest on log-transformed targets
- **Quantile Random Forest**: Predicts median (robust to zero-inflation)

## Usage

### Run All Models
```bash
cd baselines
python run_baselines.py --data_path data/ecuador_training_data.csv
```

### Run Specific Models
```bash
python run_baselines.py --data_path data/ecuador_training_data.csv --models ridge random_forest two_stage
```

### Run Only Zero-Inflated Models
```bash
python run_baselines.py --data_path data/ecuador_training_data.csv --zero_inflated_only
```

### Command Line Options
```
--data_path       Path to ecuador_training_data.csv
--models          List of model names to run
--zero_inflated_only  Only run models designed for zero-inflation
--output_dir      Directory to save results (default: results/)
--seed            Random seed for reproducibility
--quiet           Suppress verbose output
```

## Output

Results are saved to the `results/` directory:
- `baseline_results_test_TIMESTAMP.csv`: Test set metrics for all models
- `baseline_results_val_TIMESTAMP.csv`: Validation set metrics
- `baseline_summary_TIMESTAMP.txt`: Human-readable summary report

## Metrics Computed

### Standard Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

### Zero-Inflation Metrics
- **MSE (zeros)**: MSE on zero-valued samples only
- **MSE (non-zeros)**: MSE on positive samples only
- **Zero Recall**: Fraction of true zeros correctly predicted as ~0
- **Zero Precision**: Fraction of predicted zeros that are truly zero

### Sample-Level Metrics
- **Sample R²/RMSE/MAE**: Per-sample aggregated metrics

## File Structure
```
baselines/
├── data/
│   └── ecuador_training_data.csv
├── results/           # Output directory
├── preprocess.py      # Data preprocessing
├── models.py          # Model definitions
├── evaluate.py        # Evaluation metrics
├── run_baselines.py   # Main script
├── requirements.txt
└── README.md
```
