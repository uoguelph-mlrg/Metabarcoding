import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Load results
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'knn_k_tuning_results.csv')
df = pd.read_csv(data_path)

# Create figures directory
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Plot MSE vs K
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['K'], df['val_mse'], linewidth=2, color='#2E86AB', alpha=0.8, label='Validation MSE')
ax.scatter(df['K'], df['val_mse'], s=20, color='#2E86AB', alpha=0.6, zorder=3)

# Highlight best K
best_idx = df['val_mse'].idxmin()
best_k = int(df.loc[best_idx, 'K'])
best_mse = df.loc[best_idx, 'val_mse']
ax.scatter([best_k], [best_mse], color='#A23B72', s=200, marker='*', 
           label=f'Optimal K={best_k} (MSE={best_mse:.3f})', zorder=5, edgecolors='black', linewidths=1.5)

ax.set_xlabel('Number of Neighbors (K)', fontsize=13, fontweight='bold')
ax.set_ylabel('Validation MSE on Residuals', fontsize=13, fontweight='bold')
ax.set_title('KNN Tuning for Latent Smoothing (Taxonomic Features Only)', fontsize=14, fontweight='bold', pad=15)
ax.legend(frameon=True, fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'knn_k_tuning_plot.png'), dpi=300, bbox_inches='tight')
print(f"Figure saved to: {os.path.join(figures_dir, 'knn_k_tuning_plot.png')}")

# Print justification
print("\n" + "=" * 70)
print("OPTIMAL K SELECTION")
print("=" * 70)
print(f"Best K: {best_k}")
print(f"Validation MSE on residuals: {best_mse:.5f}")
print("\nJustification:")
print("- K was tuned using ONLY taxonomic features (latent space)")
print("- Target was residuals after removing intrinsic MLP predictions")
print("- This ensures K optimizes the latent smoothing, not the raw target")
print("=" * 70)
