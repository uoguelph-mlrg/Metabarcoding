"""
Standalone script: plot the current learning-rate schedulers used by Trainer
for both optimizers (MLP and latent), based on values in Config.

Run from any directory:
    python Metabarcoding/analysis/random_figures/plot_lr_schedule.py

Notes:
- In train.py, total scheduler steps depend on len(train_loader).
- This script takes that value as and applies the exact
  formulas from Trainer.__init__.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
sys.path.insert(0, SRC_DIR)

from config import Config


def _build_lr_curve(
    *,
    base_lr: float,
    warmup_frac: float,
    warmup_start_factor: float,
    eta_min: float,
    total_steps: int,
) -> tuple[np.ndarray, int]:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=base_lr)

    warmup_steps = max(1, int(warmup_frac * total_steps))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=eta_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    return np.asarray(lrs), warmup_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MLP and latent LR schedules from current training config")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure without opening an interactive window",
    )
    args = parser.parse_args()

    cfg = Config()

    mlp_total_steps = max(1, cfg.epochs)
    latent_total_steps = max(1, cfg.epochs)

    mlp_lrs, mlp_warmup_steps = _build_lr_curve(
        base_lr=cfg.mlp_lr,
        warmup_frac=cfg.mlp_warmup_frac,
        warmup_start_factor=cfg.mlp_warmup_start_factor,
        eta_min=cfg.mlp_lr_eta_min,
        total_steps=mlp_total_steps,
    )

    latent_lrs, latent_warmup_steps = _build_lr_curve(
        base_lr=cfg.latent_lr,
        warmup_frac=cfg.latent_warmup_frac,
        warmup_start_factor=cfg.latent_warmup_start_factor,
        eta_min=cfg.latent_lr_eta_min,
        total_steps=latent_total_steps,
    )

    mlp_steps = np.arange(mlp_total_steps)
    latent_steps = np.arange(latent_total_steps)

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    color_mlp = "#1f77b4"
    color_lat = "#d62728"

    ax1.plot(mlp_steps, mlp_lrs, color=color_mlp, linewidth=1.8, label="MLP learning rate")
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("MLP learning rate", color=color_mlp, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_mlp)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
    ax1.set_xlim(0, max(mlp_total_steps, latent_total_steps) - 1)
    ax1.axvspan(0, mlp_warmup_steps, color=color_mlp, alpha=0.07, label="MLP warmup")

    ax2 = ax1.twinx()
    ax2.plot(latent_steps, latent_lrs, color=color_lat, linewidth=1.8, linestyle="--", label="Latent learning rate")
    ax2.set_ylabel("Latent learning rate", color=color_lat, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_lat)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
    ax2.axvspan(0, latent_warmup_steps, color=color_lat, alpha=0.05, label="Latent warmup")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9, framealpha=0.9)

    info = (
        f"epochs={cfg.epochs}\n"
        f"$\\mathbf{{MLP}}$:\nlr={cfg.mlp_lr}\nwarmup_frac={cfg.mlp_warmup_frac}\n"
        f"start_factor={cfg.mlp_warmup_start_factor}\neta_min={cfg.mlp_lr_eta_min}\n"
        f"$\\mathbf{{Latent}}$:\nlr={cfg.latent_lr}\noptim_steps={cfg.latent_optim_steps}\n"
        f"warmup_frac={cfg.latent_warmup_frac}\nstart_factor={cfg.latent_warmup_start_factor}\n"
        f"eta_min={cfg.latent_lr_eta_min}"
    )
    ax1.text(
        0.85,
        0.7,
        info,
        transform=ax1.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85),
    )

    ax1.set_title("Trainer LR schedules: MLP (left axis) and Latent (right axis)", fontsize=12)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()

    out_dir = os.path.join(SCRIPT_DIR, "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lr_schedule.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
