# Critical Bugs Found and Fixed

## Summary of Issues
The latent-as-input_V2 model was performing catastrophically (correlation: 0.04 vs baseline: 0.88) due to **two critical implementation bugs** in the Phase A (latent optimization) solver.

---

## Bug #1: Missing Regularization for Scalar Latent Vector

**Location:** `latent_solver.py`, lines 481-486

### Problem
The gradient-based solver regularizes the latent embedding Z with:
- L2 norm: `lambda_norm * ||Z||²` (strength: 1e-2)
- Smoothness: `lambda_smooth * ||Z - HZ||²` (strength: 1e-3)

**But `latent_vec` (D) had ZERO regularization**, allowing it to:
- Grow unrestricted during Phase A optimization
- Accumulate large positive or negative values
- Become misaligned with the MLP's learned weights
- Destabilize Phase B (MLP training) when frozen

### Impact
Without regularization on `latent_vec`:
- During Phase A: gradients push D to extreme values to minimize loss
- During Phase B: frozen D forces MLP to learn pathologic weights
- Result: Loss of all learned structure (correlation drops 0.88 → 0.04)

### Fix Applied
```python
# BEFORE (Missing D regularization)
norm_loss = lambda_norm * torch.sum(Z ** 2)

# AFTER (Regularize both Z and D equally)
norm_loss_embedding = lambda_norm * torch.sum(Z ** 2)
norm_loss_scalar = lambda_norm * torch.sum(model.latent_vec ** 2)
norm_loss = norm_loss_embedding + norm_loss_scalar
```

---

## Bug #2: Wrong Loss Function in Solver (Sample Mode)

**Location:** `latent_solver.py`, lines 430-460

### Problem
**In Phase B (train_epoch):**
- Outputs are reshaped [B*max_bins] → [B, max_bins]
- Loss applies softmax **per sample** to enforce probabilities sum to 1
- Uses: `Loss.cross_entropy_soft_targets(outputs, targets, mask)`

**But Phase A solver:**
- Left outputs flat [B*max_bins]
- Applied BCE directly to flat predictions
- **Computed loss incorrectly!**

### Why This Matters
- Cross-entropy loss: softmax per sample ensures probabilities sum to 1
  ```
  loss = -sum_b(target_b * log(softmax(logits)_b)) per sample
  ```
- BCE loss: treats each observation independently
  ```
  loss = -[target * log(sigmoid(logit)) + (1-target) * log(1-sigmoid(logit))]
  ```

These compute **completely different gradients** for the same model!

### Impact
The solver optimized latent parameters for the wrong loss objective, leading to:
- Misaligned Z and D parameters
- Poor initial conditions for Phase B
- Divergence in overall training

### Fix Applied
```python
# BEFORE (Wrong loss - BCE on flat data)
output_flat = model(x_flat, bin_idx_flat)
ce_loss = bce_criterion(output_masked, targets_masked.float())

# AFTER (Correct loss - cross-entropy per sample)
output_flat = model(x_flat, bin_idx_flat)
output = output_flat.view(B, max_bins)  # Reshape for correct loss
if mask is not None:
    output = output.masked_fill(mask == 0, float('-inf'))

log_probs = F.log_softmax(output, dim=-1)
if mask is not None:
    log_probs_safe = torch.where(mask_bool, log_probs, torch.zeros_like(log_probs))
else:
    log_probs_safe = log_probs

loss_per_sample = -torch.sum(targets * log_probs_safe, dim=-1)
ce_loss = loss_per_sample.mean()
```

---

## Verification Checklist

Before re-running comparison:
- [x] Regularization applied to both Z and D equally
- [x] All hardcoded values use config parameters
- [x] Loss computation matches Phase B training
- [x] Mask handling preserves numerical stability (-inf → softmax → 0)
- [x] Bin mode still uses BCE (correct for logistic mode)
- [x] Logging enhanced to track D updates

---

## Expected Improvement
With both fixes applied:
- **Regularization:** Prevents latent_vec from diverging
- **Loss function:** Ensures consistent gradients with Phase B
- **Combined:** Should significantly improve model convergence and performance

The baseline achieved correlation: 0.885, RMSE: 0.0025.
The updated model should now be competitive with the baseline.

---

## Remaining Considerations (Not Bugs)

1. **Latent dimension increase:** MLP input goes from N → N+4 features (via Z_b embedding)
   - This is intentional for the latent-as-input variant
   - Extra 4 dimensions should be learnable

2. **Hyperparameter tuning:** May benefit from:
   - Learning rate scheduling for latent parameters
   - Adjusted regularization strengths based on performance
   - Different latent dimension sizes

These are design choices, not bugs.
