from typing import Any, Dict, Optional, Literal
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
from neighbor_graph import NeighbourGraph
from config import Config
from tqdm import tqdm
import logging as log

class LatentSolver:
    """
    Build sparse matrices and solve the latent vector optimization with multiplicative gating:
    
    Model: ŷ = w^T (m(x) ⊙ g(h[bin]))
    
    min_H sum_i loss(y_i, ŷ_i) + (r/2) ||H||^2 + (λ/2) sum_b ||h_b - ∑_j w_j h_j||^2
    
    Where:
    - H ∈ R^{n_bins × d}: latent matrix (d = embed_dim)
    - g: positive gating function (sigmoid, softplus, exp)
    - m(x) ∈ R^d: intrinsic MLP embedding
    - w ∈ R^d: final linear weights
    - V: observation-to-bin indicator matrix (N_obs × n_bins)
    - H_smooth: neighbor smoothing matrix (n_bins × n_bins)
    - r: L2 regularization strength
    - λ: smoothness regularization strength
    """

    def __init__(self, cfg: Config, neighbour_graph: NeighbourGraph, embed_dim: int, gating_fn: str = "exp"):
        self.cfg = cfg
        self.ng = neighbour_graph
        self.n_bins = neighbour_graph.n_bins
        self.embed_dim = embed_dim
        self.gating_fn = gating_fn

        # cached sparse matrices
        self.V: Optional[sparse.csr_matrix] = None  # observation-to-bin indicator
        self.H_smooth: Optional[sparse.csr_matrix] = None  # neighbor smoothing matrix (renamed to avoid confusion with latent H)
        self.I_minus_H_smooth: Optional[sparse.csr_matrix] = None  # (I - H_smooth)

    def build_V_and_H(
        self, 
        X: pd.DataFrame, 
        bin_index: Dict[Any, int], 
        method: str = "nw"
    ) -> None:
        """
        Build V (observation-to-bin indicator) and H_smooth (neighbor smoothing) matrices.
        
        V: (N_obs × n_bins) where V[i, b] = 1 if observation i is for bin b
        H_smooth: (n_bins × n_bins) where H_smooth[b, :] contains NW weights for neighbors of bin b
        """        
        N_obs = len(X)
        
        # Step 1: Build V (observation-to-bin indicator matrix)
        bin_indices = X.index.get_level_values("bin_uri").map(bin_index).to_numpy(dtype=np.int32)
        
        rows_V = np.arange(N_obs, dtype=np.int32)
        cols_V = bin_indices
        vals_V = np.ones(N_obs, dtype=np.float64)
        
        V = sparse.csr_matrix((vals_V, (rows_V, cols_V)), shape=(N_obs, self.n_bins))
        self.V = V
        
        # Step 2: Build H_smooth (neighbor smoothing matrix)
        rows_H = []
        cols_H = []
        vals_H = []
        
        for b in tqdm(range(self.n_bins), desc="Building H matrix", unit="bin", leave=False):
            if method == "nw":
                neigh, w = self.ng.nw_weights_for_node(b)
            else:
                neigh, w = self.ng.llr_coeffs_for_node(b)
            
            if len(neigh) == 0:
                # No neighbors: h_b(H) = h_b (identity) 
                log.warning(f"Bin {b} has no neighbors; using identity smoothing")
                rows_H.append(b)
                cols_H.append(b)
                vals_H.append(1.0)
            else:
                for j, wj in zip(neigh, w):
                    rows_H.append(b)
                    cols_H.append(j)
                    vals_H.append(wj)
        
        H_smooth = sparse.csr_matrix((
            np.array(vals_H), (np.array(rows_H, dtype=np.int32), np.array(cols_H, dtype=np.int32))
        ), shape=(self.n_bins, self.n_bins)
        )
        self.H_smooth = H_smooth
        
        # Step 3: Precompute (I - H_smooth) for smoothness regularization
        I = sparse.identity(self.n_bins, format="csr")
        self.I_minus_H_smooth = I - H_smooth
        
        log.info(f"Latent solver: V={V.shape}, H_smooth={H_smooth.shape}, embed_dim={self.embed_dim}, "
                 f"L2={self.cfg.latent_l2_reg}, smooth={self.cfg.latent_smooth_reg}")


    def solve(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        final_weights: np.ndarray,
        bin_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        loss_type: Literal["cross_entropy", "logistic"] = "logistic",
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve for latent matrix H ∈ R^{n_bins × d} with multiplicative gating architecture.
        
        Model: ŷ = sigmoid(w^T (m(x) ⊙ g(h[bin])))
        
        Args:
            y: target probabilities (N_obs,) in [0, 1]
            intrinsic_vec: MLP embeddings m(s,b) ∈ R^{N_obs × d}
            final_weights: final linear layer weights w ∈ R^d (from model.final_linear.weight)
            bin_ids: bin index per observation (N_obs,), required for mapping observations to bins
            sample_ids: sample index per observation (N_obs,), required for cross_entropy
            loss_type: "logistic" (sigmoid loss) or "cross_entropy" (softmax loss)
            x0: warm-start initial H (n_bins × d), if None uses default initialization
        
        Returns:
            H: latent matrix (n_bins, embed_dim) before gating
        """
        if bin_ids is None:
            raise ValueError("bin_ids are required for latent solving")
        
        if loss_type == "logistic":
            return self._solve_logistic(
                y=y,
                intrinsic_vec=intrinsic_vec,
                final_weights=final_weights,
                bin_ids=bin_ids,
                x0=x0,
            )
        else:
            if sample_ids is None:
                raise ValueError("sample_ids are required for cross_entropy latent solving")
            return self._solve_cross_entropy_lbfgs(
                y=y,
                intrinsic_vec=intrinsic_vec,
                final_weights=final_weights,
                bin_ids=bin_ids,
                sample_ids=sample_ids,
                x0=x0,
            )

    def _solve_logistic(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        final_weights: np.ndarray,
        bin_ids: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Logistic/sigmoid latent solve with multiplicative gating via L-BFGS.
        
        Model: ŷ_i = sigmoid(w^T (m_i ⊙ g(h[bin_i])))
        Objective: BCE(y, ŷ) + (r/2)||H||^2 + (λ/2)||( I - H_smooth)H||^2
        
        Args:
            y: target probabilities (N_obs,)
            intrinsic_vec: MLP embeddings (N_obs, d)
            final_weights: final linear weights (d,)
            bin_ids: bin indices (N_obs,)
            x0: warm-start (n_bins, d) or None
        
        Returns:
            H: latent matrix (n_bins, d)
        """
        if self.V is None or self.H_smooth is None:
            raise RuntimeError("Matrices not built; call build_V_and_H first")

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        intrinsic_vec = np.asarray(intrinsic_vec, dtype=np.float64)  # (N_obs, d)
        final_weights = np.asarray(final_weights, dtype=np.float64).reshape(-1)  # (d,)
        bin_ids = np.asarray(bin_ids, dtype=np.int64).reshape(-1)
        
        if intrinsic_vec.shape != (len(y), self.embed_dim):
            raise ValueError(f"intrinsic_vec has shape {intrinsic_vec.shape}, expected ({len(y)}, {self.embed_dim})")
        if final_weights.shape[0] != self.embed_dim:
            raise ValueError(f"final_weights has shape {final_weights.shape}, expected ({self.embed_dim},)")

        # Handle present-only mode: filter to observations where y > 0
        if self.cfg.latent_present_only:
            present_mask = y > 0
            y_use = y[present_mask]
            intrinsic_use = intrinsic_vec[present_mask]
            bin_ids_use = bin_ids[present_mask]
            log.debug(f"Present-only: {present_mask.sum()}/{len(y)} obs ({100*present_mask.mean():.1f}%)")
        else:
            y_use = y
            intrinsic_use = intrinsic_vec
            bin_ids_use = bin_ids

        N_obs = len(y_use)
        r = float(self.cfg.latent_l2_reg)
        lam = float(self.cfg.latent_smooth_reg)

        # Initialize H to h=0 so that g(0) = 1 for all gating functions
        if x0 is None:
            x0_use = np.zeros((self.n_bins, self.embed_dim), dtype=np.float64)
        else:
            x0_use = np.asarray(x0, dtype=np.float64)
            if x0_use.shape != (self.n_bins, self.embed_dim):
                raise ValueError(f"x0 has shape {x0_use.shape}, expected ({self.n_bins}, {self.embed_dim})")

        # Gating function and its derivative
        gating_alpha = float(getattr(self.cfg, 'gating_alpha', 0.5))
        gating_kappa = float(getattr(self.cfg, 'gating_kappa', 0.5))
        gating_epsilon = float(getattr(self.cfg, 'gating_epsilon', 0.693))
        
        if self.gating_fn == "exp":
            # (A) Exponential: exp(h), g(0) = 1
            def gate_fn(h):
                return np.exp(h)
            def gate_grad(h):
                return np.exp(h)
        elif self.gating_fn == "scaled_exp":
            # (B) Scaled exponential: exp(α·h), g(0) = 1
            def gate_fn(h):
                return np.exp(gating_alpha * h)
            def gate_grad(h):
                return gating_alpha * np.exp(gating_alpha * h)
        elif self.gating_fn == "additive":
            # (C) Additive-multiplicative: 1 + h, g(0) = 1
            def gate_fn(h):
                return 1.0 + h
            def gate_grad(h):
                return np.ones_like(h)
        elif self.gating_fn == "softplus":
            # (D) Non-negative additive: 1 + softplus(h) - ε, g(0) = 1
            def gate_fn(h):
                return 1.0 + np.log1p(np.exp(h)) - gating_epsilon
            def gate_grad(h):
                return 1.0 / (1.0 + np.exp(-h))  # derivative of softplus
        elif self.gating_fn == "tanh":
            # (E) Bounded tanh: 1 + tanh(h)·κ, g(0) = 1
            def gate_fn(h):
                return 1.0 + np.tanh(h) * gating_kappa
            def gate_grad(h):
                return gating_kappa * (1.0 - np.tanh(h)**2)
        elif self.gating_fn == "sigmoid":
            # (E) Bounded sigmoid: 2·σ(h), g(0) = 1
            def gate_fn(h):
                return 2.0 / (1.0 + np.exp(-h))
            def gate_grad(h):
                s = 1.0 / (1.0 + np.exp(-h))
                return 2.0 * s * (1 - s)
        elif self.gating_fn == "dot_product":
            # (F) Direct dot product: no gating, h acts directly as weight
            def gate_fn(h):
                return np.ones_like(h)  # Identity: used symbolically
            def gate_grad(h):
                return np.ones_like(h)  # Derivative is constant
        else:
            raise ValueError(f"Unknown gating function: {self.gating_fn}")

        def fun_and_jac(H_flat):
            H_flat = H_flat.astype(np.float64, copy=False)
            H = H_flat.reshape(self.n_bins, self.embed_dim)  # (n_bins, d)
            
            # Get latent for each observation
            h_obs = H[bin_ids_use]  # (N_obs, d)
            
            # Compute logits based on gating function
            if self.gating_fn == "dot_product":
                # Direct dot product: logits = m · h
                logits = np.sum(intrinsic_use * h_obs, axis=1)  # (N_obs,)
            else:
                # Standard architecture: logits = w^T (m ⊙ g(h))
                g_h = gate_fn(h_obs)  # (N_obs, d)
                # Modulated embedding: m ⊙ g(h)
                m_tilde = intrinsic_use * g_h  # (N_obs, d)
                # Logits: w^T m_tilde
                logits = m_tilde @ final_weights  # (N_obs,)
            
            # Predictions: sigmoid(logits)
            logits_stable = np.clip(logits, -20, 20)
            p = 1.0 / (1.0 + np.exp(-logits_stable))  # (N_obs,)
            
            # BCE loss: -[y log(p) + (1-y) log(1-p)]
            p_clip = np.clip(p, 1e-7, 1 - 1e-7)
            bce = -(y_use * np.log(p_clip) + (1 - y_use) * np.log(1 - p_clip))
            loss = float(bce.sum())
            
            # Gradient of BCE w.r.t. logits: p - y
            grad_logits = p - y_use  # (N_obs,)
            
            # Backprop through w^T m_tilde
            if self.gating_fn == "dot_product":
                # For dot product: d(m·h)/dh = m
                grad_h_obs = grad_logits[:, np.newaxis] * intrinsic_use  # (N_obs, d)
            else:
                # For standard: d(w^T (m ⊙ g(h)))/dh = m ⊙ g'(h) ⊙ w
                grad_m_tilde = np.outer(grad_logits, final_weights)  # (N_obs, d)
                # Backprop through m ⊙ g(h)
                # d(m ⊙ g(h))/dh = m ⊙ g'(h)
                grad_g_h = intrinsic_use * grad_m_tilde  # (N_obs, d)
                grad_h_obs = grad_g_h * gate_grad(h_obs)  # (N_obs, d)
            
            # Accumulate gradients into H
            grad_H = np.zeros((self.n_bins, self.embed_dim), dtype=np.float64)
            np.add.at(grad_H, bin_ids_use, grad_h_obs)
            
            # L2 regularization: (r/2) ||H||^2
            if r > 0:
                loss += 0.5 * r * float(np.sum(H ** 2))
                grad_H += r * H
            
            # Smoothness regularization: (λ/2) ||(I - H_smooth) H||^2
            if lam > 0:
                diff = self.I_minus_H_smooth.dot(H)  # (n_bins, d)
                loss += 0.5 * lam * float(np.sum(diff ** 2))
                grad_H += lam * (self.I_minus_H_smooth.T.dot(diff))
            
            # Average over observations for stability
            if N_obs > 0:
                loss /= N_obs
                grad_H /= N_obs
            
            return loss, grad_H.ravel()

        def fun(H_flat):
            f, _ = fun_and_jac(H_flat)
            return f

        def jac(H_flat):
            _, g = fun_and_jac(H_flat)
            return g

        res = minimize(
            fun=fun,
            x0=x0_use.ravel(),
            jac=jac,
            method="L-BFGS-B",
            options={
                "maxiter": int(self.cfg.cg_maxiter),
                "ftol": float(self.cfg.cg_tol),
            },
        )

        if not res.success:
            log.warning(f"L-BFGS did not converge: {res.message}")

        H = res.x.reshape(self.n_bins, self.embed_dim)
        log.debug(f"Latent H: mean={H.mean():.3f}, std={H.std():.3f}, range=[{H.min():.3f}, {H.max():.3f}]")
        return H

    def _solve_cross_entropy_lbfgs(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        final_weights: np.ndarray,
        bin_ids: np.ndarray,
        sample_ids: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Cross-entropy latent solve (softmax over bins within each sample) with L-BFGS.
        
        Model: p_i = softmax(w^T (m_i ⊙ g(h[bin_i])))_within_sample
        Objective: CE(y, p) + (r/2)||H||^2 + (λ/2)||(I - H_smooth)H||^2

        This matches the model's cross-entropy training mode.
        
        Args:
            y: target probabilities (N_obs,)
            intrinsic_vec: MLP embeddings (N_obs, d)
            final_weights: final linear weights (d,)
            bin_ids: bin indices (N_obs,)
            sample_ids: sample indices (N_obs,)
            x0: warm-start (n_bins, d) or None
        
        Returns:
            H: latent matrix (n_bins, d)
        """
        if self.I_minus_H_smooth is None or self.H_smooth is None:
            raise RuntimeError("Matrices not built; call build_V_and_H first")
        if self.cfg.latent_present_only:
            # present-only filtering breaks the meaning of a within-sample distribution
            log.warning("latent_present_only=True is ignored for cross_entropy latent solving")

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        intrinsic_vec = np.asarray(intrinsic_vec, dtype=np.float64)  # (N_obs, d)
        final_weights = np.asarray(final_weights, dtype=np.float64).reshape(-1)  # (d,)
        bin_ids = np.asarray(bin_ids, dtype=np.int64).reshape(-1)
        sample_ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)

        if not (len(y) == len(intrinsic_vec) == len(bin_ids) == len(sample_ids)):
            raise ValueError("y, intrinsic_vec, bin_ids, sample_ids must have the same length")
        if intrinsic_vec.shape[1] != self.embed_dim:
            raise ValueError(f"intrinsic_vec has shape {intrinsic_vec.shape}, expected (*, {self.embed_dim})")
        if final_weights.shape[0] != self.embed_dim:
            raise ValueError(f"final_weights has shape {final_weights.shape}, expected ({self.embed_dim},)")

        # Build per-sample segments (stable sort for determinism)
        order = np.argsort(sample_ids, kind="mergesort")
        y_s = y[order]
        m_s = intrinsic_vec[order]  # (N_obs, d)
        b_s = bin_ids[order]
        s_s = sample_ids[order]

        # Segment boundaries for each sample
        unique_s, starts = np.unique(s_s, return_index=True)
        ends = np.append(starts[1:], len(s_s))
        n_samples = len(unique_s)

        r = float(self.cfg.latent_l2_reg)
        lam = float(self.cfg.latent_smooth_reg)

        # Initialize H to h=0 so that g(0) = 1 for all gating functions
        if x0 is None:
            x0_use = np.zeros((self.n_bins, self.embed_dim), dtype=np.float64)
        else:
            x0_use = np.asarray(x0, dtype=np.float64)
            if x0_use.shape != (self.n_bins, self.embed_dim):
                raise ValueError(f"x0 has shape {x0_use.shape}, expected ({self.n_bins}, {self.embed_dim})")

        # Gating function and its derivative (same as in _solve_logistic)
        gating_alpha = float(getattr(self.cfg, 'gating_alpha', 0.5))
        gating_kappa = float(getattr(self.cfg, 'gating_kappa', 0.5))
        gating_epsilon = float(getattr(self.cfg, 'gating_epsilon', 0.693))
        
        if self.gating_fn == "exp":
            def gate_fn(h):
                return np.exp(h)
            def gate_grad(h):
                return np.exp(h)
        elif self.gating_fn == "scaled_exp":
            def gate_fn(h):
                return np.exp(gating_alpha * h)
            def gate_grad(h):
                return gating_alpha * np.exp(gating_alpha * h)
        elif self.gating_fn == "additive":
            def gate_fn(h):
                return 1.0 + h
            def gate_grad(h):
                return np.ones_like(h)
        elif self.gating_fn == "softplus":
            def gate_fn(h):
                return 1.0 + np.log1p(np.exp(h)) - gating_epsilon
            def gate_grad(h):
                return 1.0 / (1.0 + np.exp(-h))
        elif self.gating_fn == "tanh":
            def gate_fn(h):
                return 1.0 + np.tanh(h) * gating_kappa
            def gate_grad(h):
                return gating_kappa * (1.0 - np.tanh(h)**2)
        elif self.gating_fn == "sigmoid":
            def gate_fn(h):
                return 2.0 / (1.0 + np.exp(-h))
            def gate_grad(h):
                s = 1.0 / (1.0 + np.exp(-h))
                return 2.0 * s * (1 - s)
        elif self.gating_fn == "dot_product":
            def gate_fn(h):
                return np.ones_like(h)  # Identity: used symbolically
            def gate_grad(h):
                return np.ones_like(h)  # Derivative is constant
        else:
            raise ValueError(f"Unknown gating function: {self.gating_fn}")

        def fun_and_jac(H_flat):
            H_flat = H_flat.astype(np.float64, copy=False)
            H = H_flat.reshape(self.n_bins, self.embed_dim)  # (n_bins, d)
            
            # Get latent for each observation
            h_obs = H[b_s]  # (N_obs, d)
            
            # Compute modulated embeddings based on gating function
            if self.gating_fn == "dot_product":
                # Direct dot product: h acts directly as weights
                m_tilde = h_obs  # Use h directly
            else:
                # Standard architecture: m ⊙ g(h)
                g_h = gate_fn(h_obs)  # (N_obs, d)
                # Modulated embedding: m ⊙ g(h)
                m_tilde = m_s * g_h  # (N_obs, d)
            
            # For standard architecture: logits = w^T m_tilde
            # For dot_product: logits = m · h (so we need to multiply m_tilde with m)
            if self.gating_fn == "dot_product":
                logits = np.sum(m_s * h_obs, axis=1)  # (N_obs,)
            else:
                logits = m_tilde @ final_weights  # (N_obs,)
            
            # Accumulate loss and gradients per sample
            grad_H = np.zeros((self.n_bins, self.embed_dim), dtype=np.float64)
            loss = 0.0

            for st, en in zip(starts, ends):
                z_seg = logits[st:en]  # logits for this sample
                y_seg = y_s[st:en]
                m_seg = m_s[st:en]
                h_obs_seg = h_obs[st:en]
                b_seg = b_s[st:en]
                
                # Stable softmax
                z_max = float(np.max(z_seg)) if len(z_seg) > 0 else 0.0
                exp_z = np.exp(z_seg - z_max)
                denom = float(exp_z.sum()) + 1e-300
                p = exp_z / denom  # softmax probabilities
                
                # CE with soft targets: -sum(y * log_softmax(z))
                logsumexp = z_max + np.log(denom)
                loss += float(-(y_seg * (z_seg - logsumexp)).sum())
                
                # Gradient w.r.t. logits: p - y
                grad_logits_seg = p - y_seg  # (n_bins_in_sample,)
                
                # Backprop through w^T m_tilde
                if self.gating_fn == "dot_product":
                    # For dot product: d(m·h)/dh = m
                    grad_h_obs_seg = grad_logits_seg[:, np.newaxis] * m_seg  # (n_bins_in_sample, d)
                else:
                    # For standard: d(w^T (m ⊙ g(h)))/dh = m ⊙ g'(h) ⊙ w
                    m_tilde_seg = m_tilde[st:en]
                    grad_m_tilde_seg = np.outer(grad_logits_seg, final_weights)  # (n_bins_in_sample, d)
                    # Backprop through m ⊙ g(h)
                    grad_g_h_seg = m_seg * grad_m_tilde_seg  # (n_bins_in_sample, d)
                    grad_h_obs_seg = grad_g_h_seg * gate_grad(h_obs_seg)  # (n_bins_in_sample, d)
                
                # Accumulate into H
                np.add.at(grad_H, b_seg, grad_h_obs_seg)

            # Regularization
            if r > 0:
                loss += 0.5 * r * float(np.sum(H ** 2))
                grad_H += r * H

            if lam > 0:
                diff = self.I_minus_H_smooth.dot(H)  # (n_bins, d)
                loss += 0.5 * lam * float(np.sum(diff ** 2))
                grad_H += lam * (self.I_minus_H_smooth.T.dot(diff))

            # Match training scale (average over samples) to keep magnitudes stable
            if n_samples > 0:
                loss /= n_samples
                grad_H /= n_samples

            return loss, grad_H.ravel()

        def fun(H_flat):
            f, _ = fun_and_jac(H_flat)
            return f

        def jac(H_flat):
            _, g = fun_and_jac(H_flat)
            return g

        res = minimize(
            fun=fun,
            x0=x0_use.ravel(),
            jac=jac,
            method="L-BFGS-B",
            options={
                "maxiter": int(self.cfg.cg_maxiter),
                "ftol": float(self.cfg.cg_tol),
            },
        )

        if not res.success:
            log.warning(f"L-BFGS did not converge: {res.message}")

        H = res.x.reshape(self.n_bins, self.embed_dim)
        log.debug(f"Latent H (CE): mean={H.mean():.3f}, std={H.std():.3f}, range=[{H.min():.3f}, {H.max():.3f}]")
        return H

