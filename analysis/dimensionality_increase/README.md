# Multiplicative Gating Architecture

This folder contains the implementation of the **vector latent with multiplicative gating** architecture as specified in the design document.

## Architecture Overview

### Original Model (Scalar Additive)
- **Intrinsic**: `m(x) ∈ ℝ` (scalar per observation)
- **Latent**: `d_b ∈ ℝ` (scalar per bin)
- **Combination**: `ŷ = sigmoid(m(x) + d_b)` (additive in logit space)

### New Model (Vector Multiplicative)
- **Intrinsic**: `m(x) ∈ ℝᵈ` (d-dimensional log-feature embedding)
- **Latent**: `h_b ∈ ℝᵈ` (d-dimensional bias field per bin)
- **Gating**: `g(h) : ℝᵈ → ℝᵈ₊` (positive gating function: sigmoid, softplus, or exp)
- **Modulation**: `m̃ = m(x) ⊙ g(h_b)` (element-wise multiplication)
- **Final prediction**: `ŷ = sigmoid(wᵀ m̃)` where `w ∈ ℝᵈ` is learned

### Key Properties

1. **Latent cannot predict alone**: If `m(x) = 0` → `ŷ = 0`
2. **Asymmetric roles**: Intrinsic provides signal, latent modulates it
3. **Preserved log-space structure**: Modulation happens before final projection
4. **Controlled expressivity**: Gating function bounds the modulation factor

## Files Modified

### 1. [`config.py`](config.py)
Added architecture parameters:
- `embed_dim: int = 8` — embedding dimension d
- `gating_fn: Literal["sigmoid", "softplus", "exp"] = "sigmoid"` — gating function type

### 2. [`mlp.py`](mlp.py)
Modified to support vector output:
- Changed `forward()` to only squeeze when `output_dim == 1`
- Allows MLP to output d-dimensional embeddings

### 3. [`model.py`](model.py)
Complete rewrite for multiplicative gating:
- **Latent shape**: `(n_bins, embed_dim)` instead of `(n_bins,)`
- **Added gating function**: sigmoid, softplus, or exp
- **Added final linear layer**: `w: ℝᵈ → ℝ` for scalar prediction
- **Updated forward pass**:
  ```python
  intrinsic = mlp(x)              # [N, d]
  latent = latent_vec[bin_ids]    # [N, d]
  gate = gating_fn(latent)        # [N, d]
  modulated = intrinsic * gate     # [N, d]
  output = final_linear(modulated) # [N]
  ```
- **Updated `predict_MLP_only`**: Returns `(N_obs, d)` instead of `(N_obs,)`

### 4. [`latent_solver.py`](latent_solver.py)
Major refactoring for vector latent:
- **Latent matrix**: `H ∈ ℝⁿ_bins × d` instead of scalar vector
- **Non-linear optimization**: Uses L-BFGS for both loss types (was CG for logistic)
- **Updated `solve()`** signature:
  ```python
  def solve(
      y: np.ndarray,                    # (N_obs,)
      intrinsic_vec: np.ndarray,        # (N_obs, d) - VECTOR!
      final_weights: np.ndarray,        # (d,) - from model.final_linear
      bin_ids: np.ndarray,              # (N_obs,)
      sample_ids: Optional[np.ndarray], # for cross_entropy
      loss_type: str,
      x0: Optional[np.ndarray],         # (n_bins, d)
  ) -> np.ndarray:                      # (n_bins, d)
  ```
- **Gradient computation**: Backpropagates through gating function and modulation

### 5. [`train.py`](train.py)
Modified trainer for new architecture:
- **MLP instantiation**: `output_dim=cfg.embed_dim`
- **Model instantiation**: Passes `embed_dim` and `gating_fn`
- **Optimizer**: Includes `final_linear` parameters
- **Latent solving**: Passes `final_weights` to solver

## Usage

### Basic Training

```python
from config import Config, set_seed
from train import Trainer

set_seed(14)
cfg = Config(
    embed_dim=8,           # embedding dimension
    gating_fn="sigmoid",   # gating function
    latent_l2_reg=1e-2,
    latent_smooth_reg=1e-3,
)

trainer = Trainer(
    cfg,
    data_path="../../data/ecuador_training_data.csv",
    loss_type="cross_entropy",  # or "logistic"
)

results = trainer.run(use_wandb=True)
```

### Configuration Parameters

**Architecture**:
- `embed_dim`: Embedding dimension (default: 8)
- `gating_fn`: "sigmoid", "softplus", or "exp" (default: "sigmoid")

**Regularization**:
- `latent_l2_reg`: L2 penalty on H (default: 1e-2)
- `latent_smooth_reg`: Smoothness penalty on H (default: 1e-3)

**Gating Functions**:
- **sigmoid**: `g(h) = 1/(1+exp(-h))` ∈ (0, 1) - bounded, smooth
- **softplus**: `g(h) = log(1+exp(h))` ∈ (0, ∞) - unbounded, smooth
- **exp**: `g(h) = exp(h)` ∈ (0, ∞) - unbounded, sensitive

## Initialization

The latent matrix H is initialized such that `g(H) ≈ 1` (identity modulation):
- **sigmoid**: `H_init = 2.0` → sigmoid(2) ≈ 0.88
- **softplus**: `H_init = 1.0` → softplus(1) ≈ 1.31
- **exp**: `H_init = 0.0` → exp(0) = 1.0

The final linear layer `w` is initialized with equal weights: `w_i = 1/d`.

## Comparison with Original Model

To compare with the original additive model, use:

```python
# Run comparison script from analysis/latent_as_input/
python latent_as_input.py --data_path ../../data/ecuador_training_data.csv

# Visualize results
python latent_as_input_visualisation.py --results_path results/model_comparison_results.pkl
```

## Mathematical Details

### Forward Pass
```
m(x) = MLP(x)                    # intrinsic embedding
h = H[bin_id]                     # latent for bin
g = gating_fn(h)                  # positive gate
m̃ = m ⊙ g                         # modulation
ŷ = sigmoid(w^T m̃)               # final prediction
```

### Latent Optimization (Logistic Loss)
```
min_H  ∑ᵢ BCE(yᵢ, sigmoid(wᵀ(mᵢ ⊙ g(H[binᵢ])))) 
       + (r/2)||H||² 
       + (λ/2)||(I - H_smooth)H||²
```

Solved via L-BFGS with:
- **Objective**: Binary cross-entropy loss
- **Regularization**: L2 + neighbor smoothness
- **Gradient**: Chain rule through gating and modulation

### Latent Optimization (Cross-Entropy Loss)
```
min_H  ∑ₛ CE(yₛ, softmax_within_sample(wᵀ(m ⊙ g(H)))) 
       + (r/2)||H||² 
       + (λ/2)||(I - H_smooth)H||²
```

Same optimization approach, but softmax is computed per-sample instead of globally.

## Design Rationale

See the original design document for full details. Key points:

1. **Motivation**: Scalar latent + scalar MLP is too restrictive
2. **Vector generalization**: Requires explicit semantics for interaction
3. **Multiplicative choice**: Latent modulates intrinsic, doesn't compete
4. **Identifiability**: Asymmetric roles prevent rotational invariance
5. **Interpretability**: Gating as bias field is conceptually clear

## Known Limitations

- **Non-convex optimization**: L-BFGS can find local minima
- **Computational cost**: L-BFGS is slower than CG (original logistic solver)
- **Hyperparameter sensitivity**: `embed_dim`, `gating_fn`, regularization need tuning
- **Initialization matters**: Poor initialization may lead to vanishing/exploding gates

## Recommendations

1. **Start small**: Use `embed_dim=8` or less initially
2. **Use sigmoid**: Most stable gating function (bounded)
3. **Monitor gates**: Check that `g(H)` stays in reasonable range (not all near 0 or 1)
4. **Regularize**: Keep L2 and smoothness penalties active
5. **Warm start**: Use previous latent as `x0` in solver

## Future Extensions

- **Learned gating function**: Replace fixed function with small MLP
- **Dimension-specific regularization**: Different penalties per dimension
- **Structured latent**: Encourage certain dimensions to specialize
- **Attention mechanism**: Learn dimension-wise importance weights
