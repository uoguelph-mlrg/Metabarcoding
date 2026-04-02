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
        prox_weight: float = 0.0,
        x_anchor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            y: target probabilities (N_obs,) in [0, 1]
            intrinsic_vec: MLP predictions m(s,b) in logit space (N_obs,)
            bin_ids: bin index per observation (N_obs,), required for cross_entropy
            sample_ids: sample index per observation (N_obs,), required for cross_entropy
            loss_type: "logistic" (sigmoid/logit latent solve) or "cross_entropy" (softmax latent solve)
            x0: warm-start initial D (n_bins,)
            prox_weight: proximal regularization weight ρ (adds ρ||D - x_anchor||^2 term)
            x_anchor: anchor point for proximal term (defaults to zeros if None)
        
        Returns:
            D: latent vector (n_bins,) in logit space
        """
        if loss_type == "logistic":
            return self._solve_logistic(y=y, intrinsic_vec=intrinsic_vec, x0=x0, prox_weight=prox_weight, x_anchor=x_anchor)
        else:
            if bin_ids is None or sample_ids is None:
                raise ValueError("bin_ids and sample_ids are required for cross_entropy latent solving")
            return self._solve_cross_entropy_lbfgs(
                y=y,
                intrinsic_vec=intrinsic_vec,
                bin_ids=bin_ids,
                sample_ids=sample_ids,
                x0=x0,
                prox_weight=prox_weight,
                x_anchor=x_anchor,
            )

    def _solve_logistic(self, y: np.ndarray, intrinsic_vec: np.ndarray, x0: Optional[np.ndarray] = None, prox_weight: float = 0.0, x_anchor: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Logistic/sigmoid latent solve via linear system + CG:
            (V^T V + rI + λ(I-H)^T(I-H)) D = V^T (logit(y) - m)
        With proximal term: adds ρI to A and ρ*x_anchor to b.
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

        # Proximal anchoring: (A + ρI) D = b + ρ x_anchor
        if prox_weight > 0.0:
            x_anch = np.asarray(x_anchor, dtype=float).reshape(-1) if x_anchor is not None else np.zeros(self.n_bins, dtype=float)
            A_use = A_use + prox_weight * sparse.identity(self.n_bins, format="csc")
            b = b + prox_weight * x_anch

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
        prox_weight: float = 0.0,
        x_anchor: Optional[np.ndarray] = None,
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

        # Prepare proximal anchor (evaluated once, closed over in fun_and_jac)
        x_anchor_use: Optional[np.ndarray] = None
        if prox_weight > 0.0:
            x_anchor_use = np.asarray(x_anchor, dtype=np.float64).reshape(-1) if x_anchor is not None else np.zeros(self.n_bins, dtype=np.float64)

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

            # Proximal anchoring term: ρ/2 * ||D - x_anchor||^2
            if prox_weight > 0.0 and x_anchor_use is not None:
                diff_anch = D_flat - x_anchor_use
                loss += 0.5 * prox_weight * float(diff_anch @ diff_anch)
                grad_bins += prox_weight * diff_anch

            # Match training scale (average over samples) to keep magnitudes stable
            if n_samples > 0:
                loss /= n_samples
                grad_bins /= n_samples

            return loss, grad_bins

        cache_x: Optional[np.ndarray] = None
        cache_f: Optional[float] = None
        cache_g: Optional[np.ndarray] = None

        def _eval_cached(D_flat: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal cache_x, cache_f, cache_g
            if cache_x is not None and np.array_equal(D_flat, cache_x):
                if cache_f is None or cache_g is None:
                    raise RuntimeError("Invalid objective cache state")
                return cache_f, cache_g

            f_val, g_val = fun_and_jac(D_flat)
            cache_x = D_flat.copy()
            cache_f = float(f_val)
            cache_g = g_val.copy()
            return cache_f, cache_g

        def fun(D_flat: np.ndarray) -> float:
            f_val, _ = _eval_cached(D_flat)
            return f_val

        def jac(D_flat: np.ndarray) -> np.ndarray:
            _, g_val = _eval_cached(D_flat)
            return g_val

        res = minimize(
            fun=fun,
            x0=x0_use,
            jac=jac,
            method="L-BFGS-B",
            options={
                "maxiter": int(self.cfg.latent_convergence_maxiter),
                "ftol": float(self.cfg.latent_convergence_tol),
                "gtol": float(self.cfg.latent_convergence_gtol),
                "maxfun": int(self.cfg.latent_convergence_maxfun),
            },
        )

        if not res.success:
            log.warning(f"L-BFGS did not converge: {res.message}")

        D = np.asarray(res.x, dtype=np.float64)
        log.debug(f"Latent D (CE): mean={D.mean():.3f}, std={D.std():.3f}, range=[{D.min():.3f}, {D.max():.3f}]")
        return D

    def solve_gradient_based(
        self,
        model,
        data_loader,
        z_optimizer,      # optimizer for Z (embedding); None to skip Z phase
        d_optimizer,      # optimizer for D (scalar vec); None to skip D phase
        n_z_steps: int,
        n_d_steps: int,
        loss_mode: str = "sample",
    ) -> np.ndarray:
        """
        Staged Phase A latent optimisation.

        Motivation: when Z (embedding, input side) and D (scalar, output side) are
        optimised jointly, D always wins because it has a direct gradient path to the
        output. Z stays near-zero and the MLP never learns to use it, making the model
        equivalent to the plain baseline.

        Fix: optimise Z first (D frozen), then D (Z frozen). This gives Z the budget
        to develop a signal before D is allowed to compete.

        Also applies per-element balanced regularisation on Z:
            lambda_norm / latent_dim   (so the per-element penalty equals that on D).

        Args:
            model: torch Model with latent_embedding and latent_vec attributes
            data_loader: DataLoader for computing CE loss
            z_optimizer: AdamW (or similar) for latent_embedding.weight; None to skip
            d_optimizer: AdamW (or similar) for latent_vec; None to skip
            n_z_steps: number of gradient steps dedicated to Z
            n_d_steps: number of gradient steps dedicated to D
            loss_mode: "sample" or "bin"

        Returns:
            Updated Z embeddings as numpy array [n_bins, latent_dim]
        """
        import torch
        import torch.nn.functional as F

        if self.H is None:
            raise ValueError("H matrix not built. Call build_V_and_H first.")

        model.train()
        device = model.device

        # Build sparse H once
        H_coo = self.H.tocoo()
        H_indices = torch.LongTensor(np.vstack([H_coo.row, H_coo.col]))
        H_values = torch.FloatTensor(H_coo.data)
        H_torch = torch.sparse_coo_tensor(
            H_indices, H_values, size=H_coo.shape, device=device
        )

        lambda_smooth = float(self.cfg.latent_smooth_reg)
        lambda_norm_d = float(self.cfg.latent_norm_reg)
        # Per-element regularisation for Z equals D: divide by latent_dim
        lambda_norm_z = (lambda_norm_d / float(self.cfg.latent_dim)
                         * float(self.cfg.latent_z_norm_reg_factor))

        bce_criterion = torch.nn.BCEWithLogitsLoss()

        def _compute_ce_loss(batch):
            """Compute correct CE loss for one batch (matches Phase B Loss class)."""
            if loss_mode == "sample":
                x = batch["input"].to(device)
                targets = batch["target"].to(device)
                bin_idx = batch["bin_idx"].to(device)
                mask = batch.get("mask")

                B, max_bins, n_feat = x.shape
                x_flat = x.view(B * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(B * max_bins)

                output_flat = model(x_flat, bin_idx_flat)
                output = output_flat.view(B, max_bins)

                if mask is not None:
                    output = output.masked_fill(mask == 0, float('-inf'))

                log_probs = F.log_softmax(output, dim=-1)
                if mask is not None:
                    log_probs = torch.where(mask.bool(), log_probs,
                                            torch.zeros_like(log_probs))

                # targets shape: [B, max_bins] (soft, sum to 1 per sample)
                loss_per_sample = -torch.sum(targets * log_probs, dim=-1)
                return loss_per_sample.mean()
            else:
                x = batch["input"].to(device)
                targets = batch["target"].to(device)
                bin_idx = batch["bin_idx"].to(device)
                output = model(x, bin_idx)
                return bce_criterion(output, targets.float())

        def _run_steps(n_steps, optimizer, update_z: bool, update_d: bool, phase_name: str):
            """Run n_steps gradient steps updating only the selected parameter(s)."""
            model.latent_embedding.weight.requires_grad = update_z
            model.latent_vec.requires_grad = update_d

            for step in range(n_steps):
                total_loss = 0.0
                n_batches = 0

                for batch in data_loader:
                    optimizer.zero_grad()
                    ce_loss = _compute_ce_loss(batch)

                    # Regularisation (only penalise active parameters)
                    Z = model.latent_embedding.weight
                    D = model.latent_vec

                    if update_z:
                        HZ = torch.sparse.mm(H_torch, Z)
                        smooth_loss = lambda_smooth * torch.sum((Z - HZ) ** 2)
                        norm_loss_z = lambda_norm_z * torch.sum(Z ** 2)
                    else:
                        smooth_loss = torch.tensor(0.0, device=device)
                        norm_loss_z = torch.tensor(0.0, device=device)

                    norm_loss_d = (lambda_norm_d * torch.sum(D ** 2)
                                   if update_d else torch.tensor(0.0, device=device))

                    loss = ce_loss + smooth_loss + norm_loss_z + norm_loss_d
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                avg_loss = total_loss / max(1, n_batches)
                if step == 0 or (step + 1) % max(1, n_steps // 5) == 0:
                    Z_np = Z.detach().cpu().numpy()
                    D_np = D.detach().cpu().numpy()
                    log.debug(
                        f"[{phase_name}] step {step+1}/{n_steps}: loss={avg_loss:.6f} | "
                        f"Z std={Z_np.std():.5f} | D std={D_np.std():.5f}"
                    )

            # Freeze both after phase
            model.latent_embedding.weight.requires_grad = False
            model.latent_vec.requires_grad = False

        # ── Phase A-Z: develop Z before D competes ───────────────────────────
        if n_z_steps > 0 and z_optimizer is not None:
            _run_steps(n_z_steps, z_optimizer, update_z=True, update_d=False, phase_name="Z")

        # ── Phase A-D: now allow D to refine residuals ───────────────────────
        if n_d_steps > 0 and d_optimizer is not None:
            _run_steps(n_d_steps, d_optimizer, update_z=False, update_d=True, phase_name="D")

        Z_final = model.latent_embedding.weight.detach().cpu().numpy()
        log.debug(
            f"Phase A complete | Z std={Z_final.std():.5f} | "
            f"D std={model.latent_vec.detach().cpu().numpy().std():.5f}"
        )
        return Z_final

