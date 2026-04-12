"""
Comprehensive diagnostic comparing baseline vs latent-as-input training dynamics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging as log

# Import baseline trainer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from config import Config as BaselineConfig, set_seed
from train import Trainer as BaselineTrainer

# Import latent-as-input trainer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Config as LatentInputConfig
from train import Trainer as LatentInputTrainer

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compare_training_dynamics():
    """
    Compare training dynamics between baseline and latent-as-input models.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'ecuador_training_data.csv')
    
    results = {}
    
    # ===== BASELINE MODEL =====
    log.info(f"\n{'='*60}")
    log.info("Training BASELINE model (Additive Latent)")
    log.info(f"{'='*60}")
    
    set_seed()
    baseline_cfg = BaselineConfig()
    baseline_cfg.epochs_init = 50
    baseline_cfg.epochs = 5
    baseline_cfg.max_cycles = 5
    
    baseline_trainer = BaselineTrainer(
        cfg=baseline_cfg,
        data_path=data_path,
    )
    
    baseline_metrics = {
        'train_losses': [],
        'val_losses': [],
        'latent_norms': [],
        'latent_means': [],
        'latent_stds': [],
    }
    
    # Initial training
    log.info("Phase 0: Initial MLP training...")
    for epoch in tqdm(range(baseline_cfg.epochs_init), desc="Baseline init"):
        train_loss = baseline_trainer.train_epoch()
        val_loss = baseline_trainer.validate(split="val")
        baseline_metrics['train_losses'].append(train_loss)
        baseline_metrics['val_losses'].append(val_loss)
        baseline_metrics['latent_norms'].append(0.0)
        baseline_metrics['latent_means'].append(0.0)
        baseline_metrics['latent_stds'].append(0.0)
    
    # Alternating training
    log.info("Starting alternating training...")
    for cycle in tqdm(range(baseline_cfg.max_cycles), desc="Baseline cycles"):
        # Phase A: Solve latent
        latent_vec = baseline_trainer.solve_latent()
        latent_norm = np.linalg.norm(latent_vec)
        latent_mean = np.mean(latent_vec)
        latent_std = np.std(latent_vec)
        
        log.info(f"Cycle {cycle}: Latent solved - norm={latent_norm:.4f}, mean={latent_mean:.4f}, std={latent_std:.4f}")
        
        # Phase B: Train MLP
        for epoch in range(baseline_cfg.epochs):
            train_loss = baseline_trainer.train_epoch()
            val_loss = baseline_trainer.validate(split="val")
            baseline_metrics['train_losses'].append(train_loss)
            baseline_metrics['val_losses'].append(val_loss)
            baseline_metrics['latent_norms'].append(latent_norm)
            baseline_metrics['latent_means'].append(latent_mean)
            baseline_metrics['latent_stds'].append(latent_std)
    
    results['baseline'] = baseline_metrics
    
    # ===== LATENT-AS-INPUT MODEL =====
    log.info(f"\n{'='*60}")
    log.info("Training LATENT-AS-INPUT model")
    log.info(f"{'='*60}")
    
    set_seed()
    latent_cfg = LatentInputConfig()
    latent_cfg.epochs_init = 50
    latent_cfg.epochs = 5
    latent_cfg.max_cycles = 5
    latent_cfg.latent_steps = 5  # Current default
    
    latent_trainer = LatentInputTrainer(
        cfg=latent_cfg,
        data_path=data_path,
    )
    
    latent_metrics = {
        'train_losses': [],
        'val_losses': [],
        'latent_norms': [],
        'latent_means': [],
        'latent_stds': [],
    }
    
    # Initial training
    log.info("Phase 0: Initial MLP training...")
    for epoch in tqdm(range(latent_cfg.epochs_init), desc="Latent-input init"):
        train_loss = latent_trainer.train_epoch()
        val_loss = latent_trainer.validate(split="val")
        latent_metrics['train_losses'].append(train_loss)
        latent_metrics['val_losses'].append(val_loss)
        Z = latent_trainer.model.get_latent()
        latent_metrics['latent_norms'].append(np.linalg.norm(Z))
        latent_metrics['latent_means'].append(np.mean(Z))
        latent_metrics['latent_stds'].append(np.std(Z))
    
    # Alternating training
    log.info("Starting alternating training...")
    for cycle in tqdm(range(latent_cfg.max_cycles), desc="Latent-input cycles"):
        # Phase A: Solve latent with gradient descent
        Z = latent_trainer.solve_latent()
        latent_norm = np.linalg.norm(Z)
        latent_mean = np.mean(Z)
        latent_std = np.std(Z)
        
        log.info(f"Cycle {cycle}: Latent solved - norm={latent_norm:.4f}, mean={latent_mean:.4f}, std={latent_std:.4f}")
        
        # Phase B: Train MLP
        for epoch in range(latent_cfg.epochs):
            train_loss = latent_trainer.train_epoch()
            val_loss = latent_trainer.validate(split="val")
            latent_metrics['train_losses'].append(train_loss)
            latent_metrics['val_losses'].append(val_loss)
            latent_metrics['latent_norms'].append(latent_norm)
            latent_metrics['latent_means'].append(latent_mean)
            latent_metrics['latent_stds'].append(latent_std)
    
    results['latent_input'] = latent_metrics
    
    # ===== PLOTTING =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = {'baseline': 'blue', 'latent_input': 'red'}
    labels = {'baseline': 'Baseline (Additive)', 'latent_input': 'Latent-as-Input'}
    
    # Plot 1: Training loss
    ax = axes[0, 0]
    for model_name, metrics in results.items():
        ax.plot(metrics['train_losses'], label=labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Validation loss
    ax = axes[0, 1]
    for model_name, metrics in results.items():
        ax.plot(metrics['val_losses'], label=labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Final comparison
    ax = axes[0, 2]
    final_vals = [results[m]['val_losses'][-1] for m in ['baseline', 'latent_input']]
    bars = ax.bar(['Baseline', 'Latent-as-Input'], final_vals, 
                   color=[colors['baseline'], colors['latent_input']])
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, final_vals):
        ax.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Latent norm
    ax = axes[1, 0]
    for model_name, metrics in results.items():
        ax.plot(metrics['latent_norms'], label=labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Latent Norm ||D|| or ||Z||')
    ax.set_title('Latent Magnitude Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Latent mean
    ax = axes[1, 1]
    for model_name, metrics in results.items():
        ax.plot(metrics['latent_means'], label=labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Latent Mean')
    ax.set_title('Latent Mean Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 6: Latent std
    ax = axes[1, 2]
    for model_name, metrics in results.items():
        ax.plot(metrics['latent_stds'], label=labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Latent Std')
    ax.set_title('Latent Standard Deviation Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison_dynamics.png', dpi=300, bbox_inches='tight')
    log.info(f"\nSaved comparison to figures/model_comparison_dynamics.png")
    
    # Print summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY: Model Comparison")
    log.info(f"{'='*60}")
    for model_name in ['baseline', 'latent_input']:
        metrics = results[model_name]
        log.info(f"\n{labels[model_name]}:")
        log.info(f"  Final Train Loss: {metrics['train_losses'][-1]:.6f}")
        log.info(f"  Final Val Loss: {metrics['val_losses'][-1]:.6f}")
        log.info(f"  Final Latent Norm: {metrics['latent_norms'][-1]:.4f}")
        log.info(f"  Final Latent Mean: {metrics['latent_means'][-1]:.4f}")
        log.info(f"  Final Latent Std: {metrics['latent_stds'][-1]:.4f}")
    
    # Compute improvement
    baseline_val = results['baseline']['val_losses'][-1]
    latent_val = results['latent_input']['val_losses'][-1]
    improvement = (baseline_val - latent_val) / baseline_val * 100
    
    if improvement > 0:
        log.info(f"\n✓ Baseline is BETTER by {-improvement:.2f}%")
    else:
        log.info(f"\n✗ Latent-as-Input is WORSE by {-improvement:.2f}%")
    
    return results


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    results = compare_training_dynamics()
