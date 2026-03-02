"""
Diagnostic analysis for V2 model: why doesn't it outperform baseline?
"""
import pickle
import numpy as np
import sys
import os

PKL = os.path.join(os.path.dirname(__file__), '..', 'results', 'model_comparison_results.pkl')

with open(PKL, 'rb') as f:
    results = pickle.load(f)

# ── 1. Cycle counts ─────────────────────────────────────────────────────────
baseline_cycles = len(results['baseline']['cycle_val_losses'])
v2_cycles       = len(results['latent_as_input']['cycle_val_losses'])
print("=" * 60)
print("FINDING 1 – Cycle counts")
print("=" * 60)
print(f"  Baseline : {baseline_cycles} cycles  (max_cycles in src/config.py)")
print(f"  V2       : {v2_cycles} cycles  (max_cycles in latent_as_input_V2/config.py)")

# ── 2. Was V2 still converging? ─────────────────────────────────────────────
v2_val   = np.array(results['latent_as_input']['cycle_val_losses'])[:,1]
base_val = np.array(results['baseline']['cycle_val_losses'])[:,1]

print("\n" + "=" * 60)
print("FINDING 2 – Convergence state at cutoff")
print("=" * 60)
print(f"  V2 last 5 val losses : {v2_val[-5:].round(6)}")
print(f"  Baseline last 5 val  : {base_val[-5:].round(6)}")
delta_v2   = v2_val[-5] - v2_val[-1]
delta_base = base_val[-5] - base_val[-1]
print(f"  V2   improvement over last 5 cycles : {delta_v2:.6f}")
print(f"  Base improvement over last 5 cycles : {delta_base:.6f}")
print(f"  V2 still converging? {'YES' if delta_v2 > 1e-5 else 'no'}")

# ── 3. Latent parameter magnitudes ──────────────────────────────────────────
z = np.array(results['latent_as_input']['latent_embeddings'])   # [n_bins, latent_dim]
d = np.array(results['baseline']['latent_vector'])               # [n_bins]

print("\n" + "=" * 60)
print("FINDING 3 – Latent parameter norms")
print("=" * 60)
print(f"  Baseline D (scalar/bin) : std={d.std():.5f}  max_abs={np.abs(d).max():.5f}")
z_norms = np.linalg.norm(z, axis=1)
print(f"  V2 Z (4-d/bin)          : mean_norm={z_norms.mean():.5f}  "
      f"max_norm={z_norms.max():.5f}  overall_std={z.std():.6f}")
print(f"  >> Z is {'NEAR-ZERO (barely learned)' if z.std() < 0.01 else 'active'}")

# NOTE: V2 does not save D (latent_vec) in results — we only have Z.
# This is a gap in observability. We can infer from train dynamics.

# ── 4. Loss gap analysis ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINDING 4 – Loss gap")
print("=" * 60)
print(f"  Baseline best_val_loss : {results['baseline']['best_val_loss']:.6f}")
print(f"  V2       best_val_loss : {results['latent_as_input']['best_val_loss']:.6f}")
print(f"  V2 final val_loss      : {v2_val[-1]:.6f}")
print(f"  Gap (final)            : {v2_val[-1] - base_val[-1]:.6f}")

# Project V2 trajectory: does it look like it would reach baseline if given more cycles?
# Fit a simple exponential decay to the last 20 cycles
if v2_cycles >= 20:
    recent = v2_val[-20:]
    x = np.arange(len(recent), dtype=float)
    # linear fit in log-residual space relative to minimum so far
    floor = recent.min()
    y = np.log(recent - floor + 1e-9)
    slope, intercept = np.polyfit(x, y, 1)
    proj_50_more = np.exp(slope * 30 + y[-1]) + floor
    print(f"\n  Extrapolating V2 trend (linear in log space):")
    print(f"  Approx val_loss after 30 more cycles: {proj_50_more:.6f}")

# ── 5. Initialization gap: init phase losses ─────────────────────────────────
def get_init_losses(timeline):
    arr = np.array(timeline)
    init_mask = arr[:, 0] == 'init'
    return arr[init_mask, 3].astype(float)

base_init_val = get_init_losses(results['baseline']['timeline_val_losses'])
v2_init_val   = get_init_losses(results['latent_as_input']['timeline_val_losses'])
print("\n" + "=" * 60)
print("FINDING 5 – Starting point (init phase)")
print("=" * 60)
print(f"  Baseline init val start : {base_init_val[0]:.4f}  end: {base_init_val[-1]:.4f}")
print(f"  V2       init val start : {v2_init_val[0]:.4f}  end: {v2_init_val[-1]:.4f}")
print(f"  >> V2 starts from a worse initial point"
      if v2_init_val[-1] > base_init_val[-1] + 0.05 else
      "  >> Both converge similarly in init phase")

# ── 6. Effective capacity utilisation ────────────────────────────────────────
print("\n" + "=" * 60)
print("FINDING 6 – Z embedding utilisation")
print("=" * 60)
z_flat = z.flatten()
pct_near_zero = (np.abs(z_flat) < 1e-3).mean() * 100
print(f"  % of Z values nearly zero (<1e-3): {pct_near_zero:.1f}%")
print(f"  Z std per dimension:")
for i in range(z.shape[1]):
    print(f"    dim {i}: std={z[:,i].std():.6f}")

# ── 7. Hypothesis summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HYPOTHESIS SUMMARY")
print("=" * 60)
print("""
H1 – Premature stopping:
   V2 ran only half as many cycles as baseline.
   Was the model still converging? See FINDING 2.

H2 – Z is near-zero (embedding not learning):
   If Z stays near zero, the MLP never learns to use the
   latent-as-input signal. D (output scalar) would then do all
   the work, making V2 identical to the baseline.
   See FINDING 3 and 6.

H3 – D dominates / Z is redundant in softmax mode:
   In cross-entropy (softmax-per-sample) mode, a per-bin scalar D
   captures cross-sample relative biases directly. Z tries to do
   the same thing but through the MLP non-linearity — harder to
   learn, same information. The optimizer takes the path of least
   resistance: large D, small Z.

H4 – Initialisation asymmetry:
   V2 has more parameters and a harder optimisation landscape.
   The init phase may not converge as well, giving a worse starting
   point for the EM cycles. See FINDING 5.

H5 – Regularisation balance:
   Both Z and D share the same lambda_norm (1e-2).
   D is scalar (1 dof/bin), Z is 4-d (4 dof/bin).
   Per-unit regularisation is 4x stronger on Z than D,
   penalising Z more and discouraging it from growing.
""")
