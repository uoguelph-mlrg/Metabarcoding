"""
Tune K for the latent smoothing model.

This script correctly tunes K by:
1. Training the intrinsic MLP to remove the dominant signal
2. Computing residuals = log(target) - intrinsic_pred
3. Building KNN using ONLY taxonomic features (latent space)
4. Evaluating K by predicting residuals (what the latent model should explain)

This ensures K is optimized for the latent residual structure, not for the raw target.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils import load_processed
from config import Config
from mlp_only import MLPOnly

print("=" * 70)
print("TUNING K FOR LATENT SMOOTHING MODEL (MLP ONLY)")
print("=" * 70)

# Set paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

# Config for splits (should match MLP+latent model)
config = Config()
epochs = 100
batch_size = 1024
lr = 5e-4
patience = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n[1/5] Loading data...")
# Load processed data (splits, features, etc.)
splits, bins_df, bin_index, sample_index, split_indices = load_processed(data_dir)

# Extract intrinsic features (used by MLP)
intrinsic_features = [col for col in splits['train']['X'].columns]
print(f"   Intrinsic features: {intrinsic_features}")

# Extract taxonomic features (used for latent smoothing)
taxonomic_features = ['phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'species']
available_taxonomic = [col for col in taxonomic_features if col in bins_df.columns]
print(f"   Taxonomic features: {available_taxonomic}")

# Prepare data
X_train = splits['train']['X'].values.astype(np.float32)
X_val = splits['val']['X'].values.astype(np.float32)
X_test = splits['test']['X'].values.astype(np.float32)
y_train = splits['train']['y'].values.astype(np.float32)
y_val = splits['val']['y'].values.astype(np.float32)
y_test = splits['test']['y'].values.astype(np.float32)

print("\n[2/5] Training standalone MLP (no latent)...")
mlp = MLPOnly(input_dim=X_train.shape[1], hidden_dims=config.mlp_hidden_dims, output_dim=1, dropout=0.1).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

best_val = float('inf')
patience_counter = 0
for epoch in range(epochs):
    mlp.train()
    permutation = np.random.permutation(X_train.shape[0])
    train_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        idx = permutation[i:i+batch_size]
        xb = torch.from_numpy(X_train[idx]).to(device)
        yb = torch.from_numpy(y_train[idx]).to(device)
        optimizer.zero_grad()
        pred = mlp(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= X_train.shape[0]

    # Validation
    mlp.eval()
    with torch.no_grad():
        val_pred = mlp(torch.from_numpy(X_val).to(device)).cpu().numpy()
        val_loss = np.mean((val_pred - y_val) ** 2)
    if epoch % 20 == 0:
        print(f"   Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > patience:
            print(f"   Early stopping at epoch {epoch}")
            break
print(f"   MLP trained for {epoch+1} epochs (best_val_loss={best_val:.4f})")

print("\n[3/5] Computing residuals (target - intrinsic_pred)...")
mlp.eval()
with torch.no_grad():
    train_preds = mlp(torch.from_numpy(X_train).to(device)).cpu().numpy()
    val_preds = mlp(torch.from_numpy(X_val).to(device)).cpu().numpy()
    test_preds = mlp(torch.from_numpy(X_test).to(device)).cpu().numpy()

# Residuals = target - intrinsic_pred (both in logit space)
residuals_train = y_train - train_preds
residuals_val = y_val - val_preds
residuals_test = y_test - test_preds

print(f"   Train residuals: mean={np.mean(residuals_train):.4f}, std={np.std(residuals_train):.4f}")
print(f"   Val residuals: mean={np.mean(residuals_val):.4f}, std={np.std(residuals_val):.4f}")

print("\n[4/5] Building taxonomic feature matrix for KNN...")
# For each observation, we need the taxonomic features of its bin
# splits['train']['X'] has MultiIndex (sample_id, bin_uri), we need to map bin_uri -> taxonomy

# Create mapping from bin_uri to taxonomic features
bin_to_taxonomy = bins_df.set_index('bin_uri')[available_taxonomic]

# Encode taxonomic features as integers for KNN
from sklearn.preprocessing import LabelEncoder
taxonomy_encoded = bins_df[['bin_uri'] + available_taxonomic].copy()
for col in available_taxonomic:
    le = LabelEncoder()
    taxonomy_encoded[col] = le.fit_transform(taxonomy_encoded[col].fillna('MISSING'))
taxonomy_encoded = taxonomy_encoded.set_index('bin_uri')

def get_taxonomy_features(X_df):
    bin_uris = X_df.index.get_level_values('bin_uri')
    return taxonomy_encoded.loc[bin_uris].values

X_train_taxonomy = get_taxonomy_features(splits['train']['X'])
X_val_taxonomy = get_taxonomy_features(splits['val']['X'])
X_test_taxonomy = get_taxonomy_features(splits['test']['X'])

print(f"   Taxonomy feature shape: {X_train_taxonomy.shape}")

print("\n[5/5] Tuning K by predicting residuals...")
k_values = list(range(1, 1001))
results = []
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    knn.fit(X_train_taxonomy, residuals_train)
    residual_pred = knn.predict(X_val_taxonomy)
    mse = mean_squared_error(residuals_val, residual_pred)
    results.append({'K': k, 'val_mse': mse})
    if k % 10 == 0 or k <= 5 or k == 200:
        print(f"   K={k}: Val MSE={mse:.5f}")

df_results = pd.DataFrame(results)
output_path = os.path.join(os.path.dirname(__file__), 'knn_k_tuning_results.csv')
df_results.to_csv(output_path, index=False)

best_row = df_results.loc[df_results['val_mse'].idxmin()]
print("\n" + "=" * 70)
print(f"OPTIMAL K: {int(best_row['K'])} (Val MSE on residuals={best_row['val_mse']:.5f})")
print("=" * 70)
print(f"\nResults saved to: {output_path}")
print("Run visualization: python visualize_knn_k.py")
