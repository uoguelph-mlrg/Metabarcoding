"""
Standalone script: plot the MLP learning-rate schedule and the latent
update weight (alpha) side-by-side, using only the values in Config.

Run from any directory:
    python Metabarcoding/src/plot_lr_schedule.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Allow running from anywhere
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config

cfg = Config()

# ---------------------------------------------------------------------------
# MLP LR schedule (mirrors Trainer.__init__)
total_steps   = cfg.max_cycles * cfg.epochs
warmup_steps  = max(1, int(0.1 * total_steps))

dummy_model = nn.Linear(1, 1)
optimizer   = torch.optim.AdamW(dummy_model.parameters(), lr=cfg.lr)

warmup_sched = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
)
sched = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
)

mlp_lrs = []
for _ in range(total_steps):
    mlp_lrs.append(optimizer.param_groups[0]["lr"])
    sched.step()
mlp_lrs = np.array(mlp_lrs)

# Per-epoch x-axis: one MLP scheduler step per epoch
mlp_steps = np.arange(total_steps)  # 0 … total_steps-1

# ---------------------------------------------------------------------------
# Proximal weight schedule (mirrors Trainer.run)
warmup_cycles = max(1, int(cfg.latent_warmup_frac * cfg.max_cycles))
rho_0 = cfg.latent_prox_scale * cfg.latent_l2_reg  # initial proximal weight

cycles     = np.arange(cfg.max_cycles)
alphas     = np.minimum(1.0, cycles / warmup_cycles)
rho_vals   = rho_0 * (1.0 - alphas)  # ρ decays from ρ₀ → 0

# Plot ρ at the MLP step when each latent solve happens (start of each cycle)
rho_steps  = cycles * cfg.epochs  # same x-axis as MLP

# ---------------------------------------------------------------------------
# Plot
fig, ax1 = plt.subplots(figsize=(11, 4))

# MLP LR — left axis
color_mlp = "#1f77b4"
ax1.plot(mlp_steps, mlp_lrs, color=color_mlp, linewidth=1.8, label="MLP learning rate")
ax1.set_xlabel("MLP training step  (cycle × epochs/cycle)", fontsize=11)
ax1.set_ylabel("MLP learning rate", color=color_mlp, fontsize=11)
ax1.tick_params(axis="y", labelcolor=color_mlp)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
ax1.set_xlim(0, total_steps - 1)

# Shade the MLP warmup region
ax1.axvspan(0, warmup_steps, color=color_mlp, alpha=0.07, label="MLP warmup region")

# Latent proximal weight ρ — right axis
ax2 = ax1.twinx()
color_lat = "#d62728"
ax2.step(rho_steps, rho_vals, where="post",
         color=color_lat, linewidth=1.8, linestyle="--", label=f"Latent proximal weight ρ")
ax2.set_ylabel("Latent proximal weight  ρ", color=color_lat, fontsize=11)
ax2.tick_params(axis="y", labelcolor=color_lat)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
ax2.set_ylim(-rho_0 * 0.05, rho_0 * 1.15)

# Mark where ρ reaches 0
warmup_end_mlp_step = warmup_cycles * cfg.epochs
ax2.axvline(warmup_end_mlp_step, color=color_lat, linestyle=":", linewidth=1.2,
            alpha=0.6, label=f"ρ reaches 0  (cycle {warmup_cycles})")
ax2.axhline(0, color=color_lat, linestyle="-", linewidth=0.6, alpha=0.3)

# Shade latent warmup region
ax2.axvspan(0, warmup_end_mlp_step, color=color_lat, alpha=0.04, label="Latent proximal warmup")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="center right", fontsize=9, framealpha=0.9)

# Annotate config values
info = (
    f"max_cycles={cfg.max_cycles}, epochs/cycle={cfg.epochs}\n"
    f"lr={cfg.lr}, eta_min=1e-6\n"
    f"MLP warmup={warmup_steps} steps  ({100*warmup_steps/total_steps:.0f}% of total)\n"
    f"latent_warmup_frac={cfg.latent_warmup_frac}  → {warmup_cycles} cycles\n"
    f"ρ₀ = latent_prox_scale × latent_l2_reg = {cfg.latent_prox_scale} × {cfg.latent_l2_reg} = {rho_0:.4f}"
)
ax1.text(0.99, 0.97, info, transform=ax1.transAxes,
         fontsize=8, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))

ax1.set_title("LR schedule: MLP (left) vs Latent proximal weight ρ (right)", fontsize=12)
ax1.grid(True, alpha=0.25)
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "lr_schedule.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
