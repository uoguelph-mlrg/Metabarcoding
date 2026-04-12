#!/usr/bin/env python
"""
Diagnostic script to investigate:
1. Why there are loss peaks at each EM cycle (loss measurement timing & loss computation)
2. Whether the model should train longer
3. Why initial validation/training losses differ for latent-as-input

Usage:
    python investigate_training_dynamics.py --data_dir ../../data --no_wandb
"""
import argparse
import os
import sys
import pickle
import logging as log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup paths
local_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(local_dir, "..", "..", "src"))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)
if src_path not in sys.path:
    sys.path.append(src_path)

# Clear cached modules
for mod in ["config", "train", "model", "latent_solver"]:
    if mod in sys.modules:
        del sys.modules[mod]

import torch
from config import Config, set_seed
from Metabarcoding.analysis.latent_as_in_and_output.latent_as_in_and_output import load_variant_trainer, load_baseline_trainer


def analyze_loss_patterns(results_path: str):
    """Analyze the training dynamics from saved results."""
    print("\n" + "="*80)
    print("INVESTIGATING TRAINING DYNAMICS")
    print("="*80)

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 80)

        timeline_train = model_results.get("timeline_train_losses", [])
        timeline_val = model_results.get("timeline_val_losses", [])
        cycle_train = model_results.get("cycle_train_losses", [])
        cycle_val = model_results.get("cycle_val_losses", [])

        if not timeline_train:
            print("  No timeline data available")
            continue

        # Convert to DataFrame for easier analysis
        train_df = pd.DataFrame(
            [(phase, cycle, step, loss) for phase, cycle, step, loss in timeline_train],
            columns=["phase", "cycle", "step", "loss"]
        )
        val_df = pd.DataFrame(
            [(phase, cycle, step, loss) for phase, cycle, step, loss in timeline_val],
            columns=["phase", "cycle", "step", "loss"]
        )

        print("\n1. INITIALIZATION PHASE:")
        init_train = train_df[train_df["cycle"] == -1]
        init_val = val_df[val_df["cycle"] == -1]
        if len(init_train) > 0:
            print(f"   Training losses: {init_train['loss'].min():.6f} -> {init_train['loss'].max():.6f} (final: {init_train['loss'].iloc[-1]:.6f})")
            print(f"   Validation losses: {init_val['loss'].min():.6f} -> {init_val['loss'].max():.6f} (final: {init_val['loss'].iloc[-1]:.6f})")
            print(f"   Initial train loss: {init_train['loss'].iloc[0]:.6f}")
            print(f"   Initial val loss: {init_val['loss'].iloc[0]:.6f}")
            print(f"   ⚠️  Initial gap (val - train): {init_val['loss'].iloc[0] - init_train['loss'].iloc[0]:.6f}")

        print("\n2. EM CYCLE DYNAMICS (investigating loss peaks):")
        n_cycles = train_df[train_df["phase"] != "init"]["cycle"].max() + 1
        for cycle in range(min(3, n_cycles)):  # Show first 3 cycles in detail
            print(f"\n   Cycle {cycle}:")

            # Latent phase
            latent_train = train_df[(train_df["cycle"] == cycle) & (train_df["phase"] == "latent")]
            latent_val = val_df[(val_df["cycle"] == cycle) & (val_df["phase"] == "latent")]
            if len(latent_train) > 0:
                print(f"     [LATENT] train: {latent_train['loss'].iloc[0]:.6f}, val: {latent_val['loss'].iloc[0]:.6f}")

            # MLP phase first epoch
            mlp_epochs = train_df[(train_df["cycle"] == cycle) & (train_df["phase"] == "mlp")]
            mlp_val_epochs = val_df[(val_df["cycle"] == cycle) & (val_df["phase"] == "mlp")]
            if len(mlp_epochs) > 0:
                first_mlp_train = mlp_epochs['loss'].iloc[0]
                first_mlp_val = mlp_val_epochs['loss'].iloc[0]
                latent_val_loss = latent_val['loss'].iloc[0] if len(latent_val) > 0 else None
                print(f"     [MLP Epoch 0] train: {first_mlp_train:.6f}, val: {first_mlp_val:.6f}")
                if latent_val_loss is not None:
                    print(f"     🔍 Change after latent optimization (val): {first_mlp_val - latent_val_loss:+.6f}")

                # End of cycle
                last_mlp_train = mlp_epochs['loss'].iloc[-1]
                last_mlp_val = mlp_val_epochs['loss'].iloc[-1]
                print(f"     [MLP Final] train: {last_mlp_train:.6f}, val: {last_mlp_val:.6f}")
                print(f"     Improvement during cycle (val): {latent_val_loss - last_mlp_val:+.6f}")

        print("\n3. TRAINING LENGTH ANALYSIS:")
        # Check if loss has plateaued
        last_n_epochs = 20
        all_cycles_mlp = train_df[train_df["phase"] == "mlp"]
        if len(all_cycles_mlp) > last_n_epochs:
            recent_losses = all_cycles_mlp["loss"].tail(last_n_epochs).values
            early_losses = all_cycles_mlp["loss"].head(last_n_epochs).values
            recent_mean = recent_losses.mean()
            early_mean = early_losses.mean()
            improvement = (early_mean - recent_mean) / early_mean * 100
            print(f"   Training losses (first {last_n_epochs} epochs): mean = {early_mean:.6f}")
            print(f"   Training losses (last {last_n_epochs} epochs): mean = {recent_mean:.6f}")
            print(f"   Improvement: {improvement:.1f}%")
            if improvement < 1:
                print("   ⚠️  Training has largely plateaued - extending training may not help significantly")
            else:
                print("   ✓ Training still improving - could benefit from longer training")

        print("\n4. LOSS CONSISTENCY CHECK:")
        print(f"   Are train and val losses computed on same data at same time?")
        print(f"   Timeline length - train: {len(timeline_train)}, val: {len(timeline_val)}")
        if len(timeline_train) == len(timeline_val):
            print("   ✓ Same number of measurement points")
            # Check first few entries
            for i in range(min(3, len(timeline_train))):
                if timeline_train[i][:3] == timeline_val[i][:3]:
                    print(f"   ✓ Entry {i}: (phase={timeline_train[i][0]}, cycle={timeline_train[i][1]}, step={timeline_train[i][2]})")
                else:
                    print(f"   ❌ Entry {i}: MISMATCH")
                    print(f"      train: {timeline_train[i][:3]}")
                    print(f"      val:   {timeline_val[i][:3]}")

        print("\n5. CYCLE-LEVEL SUMMARY:")
        if cycle_train and cycle_val:
            print("   Cycle | Train Loss   | Val Loss     | Gap")
            print("   ------|--------------|--------------|-------")
            for (c_t, loss_t), (c_v, loss_v) in zip(cycle_train, cycle_val):
                gap = loss_v - loss_t
                print(f"   {c_t:3d}   | {loss_t:.8f} | {loss_v:.8f} | {gap:+.8f}")
            print()
            final_train = cycle_train[-1][1]
            final_val = cycle_val[-1][1]
            print(f"   Final validation loss: {final_val:.6f}")
            print(f"   Final train-val gap: {final_val - final_train:+.6f}")


def main():
    parser = argparse.ArgumentParser(description="Investigate training dynamics")
    parser.add_argument("--results_path", type=str, default=None,
                        help="Path to saved results pickle")
    args = parser.parse_args()

    if args.results_path and os.path.exists(args.results_path):
        analyze_loss_patterns(args.results_path)
    else:
        print("Results file not found. Please provide --results_path")
        print("\nUsage: python investigate_training_dynamics.py --results_path <path_to_results.pkl>")


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
