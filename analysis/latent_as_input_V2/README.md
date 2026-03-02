# Latent-as-Input Model Variant

This folder contains a variant of the metabarcoding model where BIN-specific latent factors are **concatenated to the input features** rather than added to the output.

## Architecture Overview

### Model Structure
- **Latent Embedding**: `Z ∈ ℝ^{n_bins × latent_dim}` stored as `nn.Embedding(n_bins, latent_dim)`
  - Initialized with small Gaussian noise (std ∈ [10⁻³, 10⁻²])
  - Optimized in unconstrained Euclidean space (not log-space)
  - Dimension is configurable (default: 1)

- **Forward Pass**:
  1. Retrieve latent embedding for each BIN: `z_b ∈ ℝ^{latent_dim}`
  2. Concatenate with input features: `x_augmented = [x, z_b]`
  3. Pass through MLP: `output = MLP(x_augmented)`
  4. Output is in log-space (logits for cross-entropy)

### Loss Function
```
L(θ, Z) = L_CE(θ, Z) + λ_smooth * R_smooth(Z) + λ_norm * ||Z||²
```

Where:
- **L_CE**: Cross-entropy loss over samples and bins
- **R_smooth**: Neighborhood smoothness regularization
  - `R_smooth(Z) = Σ_b ||z_b - Σ_j w_bj * z_j||²`
  - Uses same neighbor graph (taxonomic K-NN or threshold) as original model
  - Weights `w_bj` from Nadaraya-Watson kernel
- **λ_norm**: L2 norm regularization on embeddings

### Optimization: EM-like Alternation

#### Phase A — Latent Update
- Freeze MLP parameters θ
- Optimize Z using gradient descent (AdamW) for `latent_steps` iterations
- Minimizes: `L_CE(θ_fixed, Z) + λ_smooth * R_smooth(Z) + λ_norm * ||Z||²`
- Only updates BINs present in training batches (efficient)
- Gradient computation uses PyTorch autograd

#### Phase B — MLP Update
- Freeze latent embeddings Z
- Train MLP parameters θ using standard backprop
- Uses cross-entropy loss with frozen latent

## Key Implementation Details

### Modified Files
1. **config.py**: Added hyperparameters
   - `latent_dim`: Embedding dimension (default: 1)
   - `latent_init_std`: Initialization std (default: 1e-3)
   - `latent_lr`: Learning rate for Phase A (default: 1e-3)
   - `latent_steps`: Gradient steps per EM cycle (default: 5)
   - `latent_norm_reg`: λ_norm weight (default: 1e-2)
   - `latent_smooth_reg`: λ_smooth weight (default: 1e-3)

2. **model.py**: Latent-as-input architecture
   - Uses `nn.Embedding` instead of `nn.Parameter`
   - Concatenates latent to input in `forward()`
   - MLP input dimension adjusted: `input_dim + latent_dim`

3. **latent_solver.py**: Gradient-based optimization
   - New method: `solve_gradient_based()`
   - Implements Phase A via torch autograd
   - Computes smoothness using sparse H matrix
   - Supports multi-dimensional latent

4. **train.py**: EM alternation
   - Creates separate optimizer for latent (`latent_optimizer`)
   - `solve_latent()` calls gradient-based solver
   - Phase A and B alternate each cycle

### Reused from src/
- `neighbor_graph.py`: Taxonomic neighbor graph construction
- `dataset.py`: Data loading and preprocessing
- `loss.py`: Cross-entropy loss with soft targets
- `mlp.py`: Multi-layer perceptron architecture
- `utils.py`: Data loading utilities

## Usage

```python
# Run training with default hyperparameters
python train.py --loss_type cross_entropy

# Or use specific data directory
python train.py --data_dir ../../data --loss_type cross_entropy

# Enable verbose logging
python train.py --loss_type cross_entropy -v
```

All hyperparameters are configured in `config.py`. No command-line arguments needed for hyperparameter tuning—modify the config file directly.

## Key Differences from Original Model

| Aspect | Original Model | Latent-as-Input Variant |
|--------|---------------|-------------------------|
| **Latent representation** | Scalar per BIN, in log-space | Vector per BIN, in Euclidean space |
| **Architecture** | Output: `MLP(x) + d_b` | Output: `MLP([x, z_b])` |
| **Latent optimization** | Closed-form solve (CG/L-BFGS) | Gradient descent (AdamW) |
| **Loss** | CE/logistic + smoothness | CE + smoothness + norm |
| **Interpretation** | Additive abundance bias | Learned BIN embedding feature |

## Advantages of This Variant

1. **Non-additive interactions**: Latent can modulate MLP computation non-linearly
2. **Multi-dimensional latent**: Can capture richer BIN-specific information
3. **Flexible optimization**: Gradient-based allows easy extension (e.g., different regularizers)
4. **Unified framework**: Both θ and Z optimized via gradient descent

## Hyperparameter Tuning Guidance

- **latent_dim**: Start with 1, increase if underfitting (try 2, 4, 8)
- **latent_init_std**: Keep small (1e-3 to 1e-2) to avoid initialization issues
- **latent_steps**: Balance accuracy vs. speed (1-5 steps recommended)
- **latent_lr**: Typically lower than MLP lr (1e-4 to 1e-3)
- **λ_smooth**: Controls neighbor similarity (1e-4 to 1e-2)
- **λ_norm**: Prevents unbounded growth (1e-3 to 1e-1)

## Unseen BINs (Validation/Test)

BINs not present in training are handled via interpolation from the neighborhood graph:
- Uses kernel-weighted average of neighbor latent embeddings
- Implemented in `NeighbourGraph.nw_weights_for_node()`
- Already supported in the original codebase (reused here)

## Future Extensions

1. **Multi-dimensional latent**: Increase `latent_dim` to capture richer structures
2. **Attention mechanisms**: Use latent as query/key for sample-bin interactions
3. **Hierarchical latent**: Different dimensions for different taxonomic levels
4. **Joint optimization**: Train θ and Z simultaneously (remove alternation)
5. **Latent visualization**: Project embeddings to 2D/3D for interpretation
