# Diagnostic Suite for Latent-as-Input Model

This directory contains comprehensive diagnostic scripts to investigate why the latent-as-input model underperforms compared to the baseline additive latent model.

## Diagnostic Scripts

### 1. `verify_correctness.py`
**Purpose:** Fundamental correctness checks
- Verifies loss computation is consistent across different methods
- Checks for data leakage between train/val/test splits
- Verifies gradient flow through MLP and latent embeddings
- Validates that predictions and targets are properly normalized

**Run:** 
```bash
cd /path/to/latent_as_input/diagnostics
python verify_correctness.py
```

### 2. `check_latent_convergence.py`
**Purpose:** Test if 5 gradient steps is sufficient for latent optimization
- Compares convergence with 5, 10, 20, 50, 100 gradient steps
- Tracks loss, latent norm, and latent change per step
- Measures final validation loss for each configuration
- Generates convergence plots

**Key Question:** Is the default `latent_steps=5` too small?

**Run:**
```bash
python check_latent_convergence.py
```

**Output:** `figures/latent_convergence_analysis.png`

### 3. `check_gradient_masking.py`
**Purpose:** Test if gradient masking limits learning
- Compares training with and without gradient masking
- Gradient masking restricts updates to bins in current batch + neighbors
- Tracks gradient sparsity and number of bins updated per batch
- Measures impact on final performance

**Key Question:** Does restricting gradient updates hurt performance?

**Run:**
```bash
python check_gradient_masking.py
```

**Output:** `figures/gradient_masking_analysis.png`

### 4. `check_learning_rate.py`
**Purpose:** Find optimal learning rate for latent optimization
- Tests learning rates: 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2
- Tracks convergence speed and final validation loss
- Identifies if current `lr=1e-3` is appropriate

**Key Question:** Is the learning rate optimal?

**Run:**
```bash
python check_learning_rate.py
```

**Output:** `figures/learning_rate_analysis.png`

### 5. `compare_models.py`
**Purpose:** Direct comparison of baseline vs latent-as-input training dynamics
- Trains both models with same configuration
- Tracks train/val loss, latent statistics over time
- Visualizes differences in learning dynamics
- Quantifies performance gap

**Key Question:** What differs in the training dynamics?

**Run:**
```bash
python compare_models.py
```

**Output:** `figures/model_comparison_dynamics.png`

## Expected Issues to Investigate

Based on code review, these are the hypothesized issues:

1. **Insufficient Gradient Steps**: Baseline uses convergence-based solver (up to 500 CG iterations), while latent-as-input uses only 5 gradient steps per cycle.

2. **Gradient Masking**: Latent gradients are masked to only update bins in batch + neighbors, potentially limiting learning.

3. **Learning Rate**: Fixed `lr=1e-3` may not be optimal for the gradient-based latent optimization.

4. **Loss Computation**: Potential mismatch in how cross-entropy loss is applied in the gradient-based solver vs baseline.

5. **Regularization Strength**: Different effective regularization between closed-form (baseline) and gradient-based (latent-input) optimization.

## Running All Diagnostics

To run all diagnostics sequentially:

```bash
# Create figures directory
mkdir -p figures

# Run each diagnostic
python verify_correctness.py
python check_latent_convergence.py
python check_gradient_masking.py  
python check_learning_rate.py
python compare_models.py
```

## Interpreting Results

### If convergence diagnostic shows improvement with more steps:
→ Increase `cfg.latent_steps` in `config.py`

### If gradient masking shows improvement without masking:
→ Remove or modify masking logic in `latent_solver.py:solve_gradient_based()`

### If learning rate diagnostic identifies better lr:
→ Update `cfg.latent_lr` in `config.py`

### If loss computation shows inconsistencies:
→ Fix loss computation in `latent_solver.py` or `loss.py`

## Notes

- All scripts use `seed=14` for reproducibility
- Scripts are designed to work with the Ecuador training dataset
- Each diagnostic saves visualization to `figures/` directory
- Logs provide detailed numerical summaries
