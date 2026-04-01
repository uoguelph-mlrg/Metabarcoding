from typing import Any, Dict, Optional, Literal
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
from tqdm import tqdm
import logging as log

# Import from src folder (reusing existing infrastructure)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from neighbor_graph import NeighbourGraph

# Import from local folder (modified for latent-as-input)
from config import Config

class LatentSolver:
    """
    Build sparse matrices and solve the latent vector optimization:
    
    min_D (1/2) sum_{s,b} (y_{s,b} - m(s,b) - d_b)^2 + (r/2) ||D||^2 + (λ/2) sum_b (d_b - h_b(D))^2
    
    Solution: (V^T V + r*I + λ*(I-H)^T(I-H)) D = V^T (y - m)
    
    Where:
    - V: observation-to-bin indicator matrix (N_obs x n_bins)
    - H: neighbor smoothing matrix (n_bins x n_bins), H[b,:] = NW weights for bin b
    - r: L2 regularization strength (latent_l2_reg)
    - λ: smoothness regularization strength (latent_smooth_reg)
    """

    def __init__(self, cfg: Config, neighbour_graph: NeighbourGraph):
        self.cfg = cfg
        self.ng = neighbour_graph
        self.n_bins = neighbour_graph.n_bins

        # cached sparse matrices
        self.V: Optional[sparse.csr_matrix] = None  # observation-to-bin indicator
        self.H: Optional[sparse.csr_matrix] = None  # neighbor smoothing matrix
        self.A: Optional[sparse.csc_matrix] = None  # precomputed LHS matrix
        self.I_minus_H: Optional[sparse.csr_matrix] = None  # (I - H)
        self.S: Optional[sparse.csc_matrix] = None  # (I - H)^T (I - H)

    def build_V_and_H(
        self, 
        X: pd.DataFrame, 
        bin_index: Dict[Any, int], 
        method: str = "nw"
    ) -> None:
        """
        Build V (observation-to-bin indicator) and H (neighbor smoothing) matrices.
        
        V: (N_obs x n_bins) where V[i, b] = 1 if observation i is for bin b
        H: (n_bins x n_bins) where H[b, :] contains NW weights for neighbors of bin b
        """        
        N_obs = len(X)
        
        # Step 1: Build V (observation-to-bin indicator matrix)
        bin_indices = X.index.get_level_values("bin_uri").map(bin_index).to_numpy(dtype=np.int32)
        
        rows_V = np.arange(N_obs, dtype=np.int32)
        cols_V = bin_indices
        vals_V = np.ones(N_obs, dtype=np.float64)
        
        V = sparse.csr_matrix((vals_V, (rows_V, cols_V)), shape=(N_obs, self.n_bins))
        self.V = V
        
        # Step 2: Build H (neighbor smoothing matrix)
        rows_H = []
        cols_H = []
        vals_H = []
        
        for b in tqdm(range(self.n_bins), desc="Building H matrix", unit="bin", leave=False):
            if method == "nw":
                neigh, w = self.ng.nw_weights_for_node(b)
            else:
                neigh, w = self.ng.llr_coeffs_for_node(b)
            
            if len(neigh) == 0:
                # No neighbors: h_b(D) = d_b (identity) 
                log.warning(f"Bin {b} has no neighbors; using identity smoothing")
                rows_H.append(b)
                cols_H.append(b)
                vals_H.append(1.0)
            else:
                for j, wj in zip(neigh, w):
                    rows_H.append(b)
                    cols_H.append(j)
                    vals_H.append(wj)
        
        H = sparse.csr_matrix((
            np.array(vals_H), (np.array(rows_H, dtype=np.int32), np.array(cols_H, dtype=np.int32))
        ), shape=(self.n_bins, self.n_bins)
        )
        self.H = H
        
        # Step 3: Precompute A = V^T V + r*I + λ*(I - H)^T (I - H)
        VtV = (V.T @ V).tocsc()
        
        I = sparse.identity(self.n_bins, format="csr")
        I_minus_H = I - H
        smoothness_term = (I_minus_H.T @ I_minus_H).tocsc()
        l2_term = I.tocsc()
        
        A = VtV + self.cfg.latent_l2_reg * l2_term + self.cfg.latent_smooth_reg * smoothness_term
        self.A = A
        self.I_minus_H = I_minus_H.tocsr()  # type: ignore
        self.S = smoothness_term
        
        log.info(f"Latent solver: V={V.shape}, H={H.shape}, L2={self.cfg.latent_l2_reg}, smooth={self.cfg.latent_smooth_reg}")


    def solve(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        bin_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        loss_type: Literal["cross_entropy", "logistic"] = "logistic",
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            y: target probabilities (N_obs,) in [0, 1]
            intrinsic_vec: MLP predictions m(s,b) in logit space (N_obs,)
            bin_ids: bin index per observation (N_obs,), required for cross_entropy
            sample_ids: sample index per observation (N_obs,), required for cross_entropy
            loss_type: "logistic" (sigmoid/logit latent solve) or "cross_entropy" (softmax latent solve)
            x0: warm-start initial D (n_bins,)
        
        Returns:
            D: latent vector (n_bins,) in logit space
        """
        if loss_type == "logistic":
            return self._solve_logistic(y=y, intrinsic_vec=intrinsic_vec, x0=x0)
        else:
            if bin_ids is None or sample_ids is None:
                raise ValueError("bin_ids and sample_ids are required for cross_entropy latent solving")
            return self._solve_cross_entropy_lbfgs(
                y=y,
                intrinsic_vec=intrinsic_vec,
                bin_ids=bin_ids,
                sample_ids=sample_ids,
                x0=x0,
            )

    def _solve_logistic(self, y: np.ndarray, intrinsic_vec: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Logistic/sigmoid latent solve via linear system + CG:
            (V^T V + rI + λ(I-H)^T(I-H)) D = V^T (logit(y) - m)
        """
        if self.V is None or self.A is None or self.H is None:
            raise RuntimeError("Matrices not built; call build_V_and_H first")

        # Ensure y is a numpy array
        y = np.asarray(y)
        intrinsic_vec = np.asarray(intrinsic_vec)

        # Handle present-only mode: filter to observations where y > 0
        if self.cfg.latent_present_only:
            present_mask = y > 0
            y_filtered = y[present_mask]
            intrinsic_filtered = intrinsic_vec[present_mask]
            V_filtered = self.V[present_mask, :]
            
            log.debug(f"Present-only: {present_mask.sum()}/{len(y)} obs ({100*present_mask.mean():.1f}%)")
            
            # Recompute A matrix with filtered V
            VtV = (V_filtered.T @ V_filtered).tocsc()
            I = sparse.identity(self.n_bins, format="csr")
            I_minus_H = I - self.H
            smoothness_term = (I_minus_H.T @ I_minus_H).tocsc()
            l2_term = I.tocsc()
            A_filtered = VtV + self.cfg.latent_l2_reg * l2_term + self.cfg.latent_smooth_reg * smoothness_term
            
            y_use = y_filtered
            intrinsic_use = intrinsic_filtered
            V_use = V_filtered
            A_use = A_filtered
        else:
            y_use = y
            intrinsic_use = intrinsic_vec
            V_use = self.V
            A_use = self.A

        # Clip y to avoid log(0) and convert to logits
        y_clipped = np.clip(y_use, 1e-7, 1 - 1e-7)
        y_logit = np.log(y_clipped / (1 - y_clipped))  # logit transform
        
        # Compute residual in logit space: D should satisfy intrinsic + D ≈ logit(y)
        # So residual = logit(y) - intrinsic
        residual = y_logit - intrinsic_use
        
        # Compute b = V^T residual
        b = V_use.T.dot(residual)

        # Solve A D = b using conjugate gradient
        if x0 is None:
            x0_use = np.zeros(self.n_bins, dtype=float)
        else:
            x0_use = np.asarray(x0, dtype=float).reshape(-1)
            if x0_use.shape[0] != self.n_bins:
                raise ValueError(f"x0 has shape {x0_use.shape}, expected ({self.n_bins},)")

        D, info = cg(
            A_use, 
            b, 
            x0=x0_use,
            atol=self.cfg.latent_convergence_tol, 
            maxiter=self.cfg.latent_convergence_maxiter
        )
        
        if info != 0:
            log.warning(f"CG did not converge (info={info})")
        
        # Log summary statistics (only at debug level)
        log.debug(f"Latent D: mean={D.mean():.3f}, std={D.std():.3f}, range=[{D.min():.3f}, {D.max():.3f}]")
        
        return D

    def _solve_cross_entropy_lbfgs(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        bin_ids: np.ndarray,
        sample_ids: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Cross-entropy latent solve (softmax over bins within each sample) with L-BFGS.

        We minimize:
            sum_s CE(y_s, softmax(m_s + D_bins)) + (r/2)||D||^2 + (λ/2)|| (I-H)D ||^2

        This matches the model's cross-entropy training mode.
        """
        if self.I_minus_H is None or self.H is None:
            raise RuntimeError("Matrices not built; call build_V_and_H first")
        if self.cfg.latent_present_only:
            # present-only filtering breaks the meaning of a within-sample distribution unless renormalized
            log.warning("latent_present_only=True is ignored for cross_entropy latent solving")

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        intrinsic_vec = np.asarray(intrinsic_vec, dtype=np.float64).reshape(-1)
        bin_ids = np.asarray(bin_ids, dtype=np.int64).reshape(-1)
        sample_ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)

        if not (len(y) == len(intrinsic_vec) == len(bin_ids) == len(sample_ids)):
            raise ValueError("y, intrinsic_vec, bin_ids, sample_ids must have the same length")

        # Build per-sample segments (stable, no Python dict iteration order issues)
        order = np.argsort(sample_ids, kind="mergesort")
        y_s = y[order]
        m_s = intrinsic_vec[order]
        b_s = bin_ids[order]
        s_s = sample_ids[order]

        # Segment boundaries for each sample id
        unique_s, starts = np.unique(s_s, return_index=True)
        ends = np.append(starts[1:], len(s_s))
        n_samples = len(unique_s)

        r = float(self.cfg.latent_l2_reg)
        lam = float(self.cfg.latent_smooth_reg)
        I_minus_H = self.I_minus_H  # csr

        # Warm start
        if x0 is None:
            x0_use = np.zeros(self.n_bins, dtype=np.float64)
        else:
            x0_use = np.asarray(x0, dtype=np.float64).reshape(-1)
            if x0_use.shape[0] != self.n_bins:
                raise ValueError(f"x0 has shape {x0_use.shape}, expected ({self.n_bins},)")

        def fun_and_jac(D_flat: np.ndarray) -> tuple[float, np.ndarray]:
            D_flat = D_flat.astype(np.float64, copy=False)

            # z_i = m_i + d_{bin(i)} in the sorted-by-sample order
            z = m_s + D_flat[b_s]

            # Gradient accumulator in observation space, then scatter-add into bins
            grad_bins = np.zeros(self.n_bins, dtype=np.float64)
            loss = 0.0

            for st, en in zip(starts, ends):
                z_seg = z[st:en]
                y_seg = y_s[st:en]

                # stable logsumexp
                z_max = float(np.max(z_seg)) if len(z_seg) > 0 else 0.0
                exp = np.exp(z_seg - z_max)
                denom = float(exp.sum()) + 1e-300
                p = exp / denom

                # CE with soft targets: -sum(y * log_softmax(z))
                # log_softmax = z - logsumexp(z)
                logsumexp = z_max + np.log(denom)
                loss += float(-(y_seg * (z_seg - logsumexp)).sum())

                # grad wrt z is (p - y)
                g_z = (p - y_seg)
                np.add.at(grad_bins, b_s[st:en], g_z)

            # Regularization
            if r > 0:
                loss += 0.5 * r * float(D_flat @ D_flat)
                grad_bins += r * D_flat

            if lam > 0:
                diff = I_minus_H.dot(D_flat)  # (I-H)D
                loss += 0.5 * lam * float(diff @ diff)
                grad_bins += lam * I_minus_H.T.dot(diff)

            # Match training scale (average over samples) to keep magnitudes stable
            if n_samples > 0:
                loss /= n_samples
                grad_bins /= n_samples

            return loss, grad_bins

        def fun(D_flat: np.ndarray) -> float:
            f, _ = fun_and_jac(D_flat)
            return f

        def jac(D_flat: np.ndarray) -> np.ndarray:
            _, g = fun_and_jac(D_flat)
            return g

        res = minimize(
            fun=fun,
            x0=x0_use,
            jac=jac,
            method="L-BFGS-B",
            options={
                "maxiter": int(self.cfg.latent_convergence_maxiter),
                "ftol": float(self.cfg.latent_convergence_tol),
            },
        )

        if not res.success:
            log.warning(f"L-BFGS did not converge: {res.message}")

        D = np.asarray(res.x, dtype=np.float64)
        log.debug(f"Latent D (CE): mean={D.mean():.3f}, std={D.std():.3f}, range=[{D.min():.3f}, {D.max():.3f}]")
        return D

    def solve_gradient_based(
        self,
        model,  # torch model with latent_embedding
        data_loader,  # torch DataLoader
        optimizer,  # torch optimizer for latent
        n_steps: int,
        loss_mode: str = "sample",
        val_loss_fn=None,   # Optional[Callable[[], float]]: called after each step for convergence check
        patience: Optional[int] = None,  # early stopping patience (in steps); None = no early stopping
        on_step_end=None,   # Optional[Callable[[int, float], None]]: callback(step, val_loss) for logging
    ) -> np.ndarray:
        """
        Gradient-based latent optimization for latent-as-input variant.
        
        Optimizes: L(theta, Z) = L_CE(theta, Z) + lambda_smooth * R_smooth(Z) + lambda_norm * ||Z||^2
        
        Where:
        - L_CE: cross-entropy loss computed over all observations
        - R_smooth: neighborhood smoothness: sum_b ||z_b - sum_j w_bj * z_j||^2
        - Norm: sum_b ||z_b||^2
        
        Only updates BINs present in the batch + their graph neighbors.
        
        Args:
            model: torch Model with latent_embedding attribute
            data_loader: DataLoader for computing CE loss
            optimizer: torch optimizer (e.g., AdamW) for latent_embedding.weight
            n_steps: number of gradient steps
            loss_mode: "sample" or "bin" mode
            
        Returns:
            Updated latent embeddings as numpy array [n_bins, latent_dim]
        """
        import torch
        import torch.nn.functional as F
        
        if self.H is None:
            raise ValueError("H matrix not built. Call build_V_and_H first.")
        
        model.train()
        device = model.device

        # Early stopping state (used only when val_loss_fn is provided)
        _best_val = float('inf')
        _no_improve = 0
        
        # Enable gradient for latent embedding
        model.latent_embedding.weight.requires_grad = True
        
        # Convert H to torch sparse tensor for efficient smoothness computation
        H_coo = self.H.tocoo()
        H_indices = torch.LongTensor(np.vstack([H_coo.row, H_coo.col]))
        H_values = torch.FloatTensor(H_coo.data)
        H_torch = torch.sparse_coo_tensor(
            H_indices, H_values, size=H_coo.shape, device=device
        )
        
        lambda_smooth = float(self.cfg.latent_smooth_reg)
        lambda_norm = float(self.cfg.latent_norm_reg)
        
        # Loss criteria
        criterion = torch.nn.CrossEntropyLoss()
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        for step in range(n_steps):
            # Restore training state at the top of each step (validate() sets model.eval())
            model.train()
            model.latent_embedding.weight.requires_grad = True
            total_loss = 0.0
            n_batches = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # Unpack batch based on loss_mode
                if loss_mode == "sample":
                    x = batch["input"].to(device)  # [B, max_bins, n_feat]
                    targets = batch["target"].to(device)  # [B, max_bins] or [B, max_bins, n_classes]
                    bin_idx = batch["bin_idx"].to(device)  # [B, max_bins]
                    mask = batch.get("mask")  # [B, max_bins]
                    
                    B, max_bins, n_feat = x.shape
                    x_flat = x.view(B * max_bins, n_feat)
                    bin_idx_flat = bin_idx.view(B * max_bins)

                    if targets.dim() == 3:
                        n_classes = targets.shape[2]
                        targets_flat = targets.view(B * max_bins, n_classes)
                    else:
                        targets_flat = targets.view(B * max_bins)
                    
                    # Forward pass
                    output_flat = model(x_flat, bin_idx_flat)  # [B*max_bins, n_classes]
                    
                    # Apply mask if present
                    if mask is not None:
                        mask_flat = mask.view(B * max_bins).to(device).bool()
                        output_masked = output_flat[mask_flat]
                        targets_masked = targets_flat[mask_flat]
                        bin_idx_masked = bin_idx_flat[mask_flat]
                    else:
                        output_masked = output_flat
                        targets_masked = targets_flat
                        bin_idx_masked = bin_idx_flat
                    
                    # CE loss (or BCE for 1D logits)
                    if output_masked.dim() == 1:
                        ce_loss = bce_criterion(output_masked, targets_masked.float())
                    elif targets_masked.dim() == 2:
                        if targets_masked.shape == output_masked.shape:
                            ce_loss = criterion(output_masked, targets_masked.float())
                        else:
                            ce_loss = criterion(output_masked, targets_masked.argmax(dim=-1))
                    else:
                        ce_loss = criterion(output_masked, targets_masked.long())
                    
                    # Track bins in this batch
                    bins_in_batch = set(bin_idx_masked.detach().cpu().numpy().tolist())
                    
                else:  # bin mode
                    x = batch["input"].to(device)
                    targets = batch["target"].to(device)  # [N, n_classes] or [N]
                    bin_idx = batch["bin_idx"].to(device)
                    
                    # Forward pass
                    output = model(x, bin_idx)
                    
                    # CE loss
                    if targets.dim() == 2:
                        ce_loss = criterion(output, targets.argmax(dim=-1))
                    else:
                        ce_loss = criterion(output, targets.long())
                    
                    # Track bins in this batch
                    bins_in_batch = set(bin_idx.detach().cpu().numpy().tolist())
                
                # Smoothness regularization: ||Z - HZ||^2
                Z = model.latent_embedding.weight  # [n_bins, latent_dim]
                HZ = torch.sparse.mm(H_torch, Z)  # [n_bins, latent_dim]
                smooth_loss = lambda_smooth * torch.sum((Z - HZ) ** 2)
                
                # Norm regularization: ||Z||^2
                norm_loss = lambda_norm * torch.sum(Z ** 2)
                
                # Total loss
                loss = ce_loss + smooth_loss + norm_loss
                
                # Backward and optimize
                loss.backward()

                # AGGRESSIVE FIX: Remove gradient masking to allow all bins to be updated
                # This allows comprehensive learning across all bins, not just those in current batch
                # Previous conservative fix only increased batch size, but masking still limited learning
                # Comment: Lines 489-501 (gradient masking) have been removed
                
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(1, n_batches)
            if step == 0 or (step + 1) % max(1, n_steps // 5) == 0:
                log.debug(f"Latent step {step+1}/{n_steps}: loss={avg_loss:.6f}")

            # Convergence check: evaluate val loss and apply early stopping
            if val_loss_fn is not None:
                val_loss = val_loss_fn()  # sets model.eval() internally
                model.train()  # restore for next step
                model.latent_embedding.weight.requires_grad = True
                if on_step_end is not None:
                    on_step_end(step, val_loss)
                if val_loss < _best_val - 1e-4:
                    _best_val = val_loss
                    _no_improve = 0
                else:
                    _no_improve += 1
                    if patience is not None and _no_improve >= patience:
                        log.info(f"Latent solve converged at step {step + 1} (patience={patience}).")
                        break

        # Disable gradient for latent embedding
        model.latent_embedding.weight.requires_grad = False
        
        # Return updated latent embeddings
        Z_final = model.latent_embedding.weight.detach().cpu().numpy()
        log.debug(f"Latent Z: mean={Z_final.mean():.3f}, std={Z_final.std():.3f}, "
                  f"range=[{Z_final.min():.3f}, {Z_final.max():.3f}]")
        
        return Z_final

