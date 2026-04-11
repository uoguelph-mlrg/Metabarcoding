Cross-Entropy Verification and Benchmark

This folder includes a reproducible script to verify that:
1. MLP cross-entropy in src/loss.py
2. Latent cross-entropy in src/latent_solver.py

compute the same objective under equivalent inputs, and to benchmark their runtime.

File
- verify_ce_equivalence.py

What it checks
- Forward-value equivalence between the two CE implementations.
- Gradient equivalence with respect to logits.
- Interpolation masking logic in LatentSolver._logits_from_latent.
- Micro-benchmarks (forward + backward) for both CE paths.

Run
From the Metabarcoding root:
python analysis/loss_comparison/verify_ce_equivalence.py --device cpu

Optional args
- --shapes 32x128,64x256,128x512
- --repeats 20
- --warmup 5
- --valid-fraction 0.7
- --seed 7
- --device cpu|mps|cuda

Interpretation
- abs_loss_diff near 0 and max_abs_grad_diff near 0 means CE math/gradients match.
- latent_over_mlp > 1 means latent CE path is slower for that shape.
