from typing import Any, Dict, Optional, Literal
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import minimize
from neighbor_graph import NeighbourGraph
from config import Config
from gating_functions import make_gating_function
from tqdm import tqdm
import logging as log

GatingFnName = Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"]

class LatentSolver:
    """
    Build sparse matrices and solve the latent vector optimization.

    Scalar mode (embed_dim == 1):
        min_D (1/2) sum_{s,b} (y_{s,b} - m(s,b) - (H_smooth D)_b)^2 + (r/2)||D||^2 + (λ/2)||(I-H_smooth)D||^2
        The interpolated latent (H_smooth @ D)[bin] is used instead of the raw D[bin].
        Solution via linear system + CG (logistic) or L-BFGS (cross-entropy).

    Vector mode (embed_dim > 1):
        min_H BCE(y, ŷ) + (r/2)||H||^2 + (λ/2)||(I-H_smooth)H||^2
        where ŷ = sigmoid(w^T (m(x) ⊙ g((H_smooth H)[bin])))
        The interpolated latent (H_smooth @ H)[bin] is used instead of the raw H[bin].
        Solution via L-BFGS.

    Where:
    - V: observation-to-bin indicator matrix (N_obs × n_bins)
    - H_smooth: neighbor smoothing matrix (n_bins × n_bins), H_smooth[b,:] = NW weights for bin b
    - r: L2 regularization strength (latent_l2_reg)
    - λ: smoothness regularization strength (latent_smooth_reg)
    """

    def __init__(self, cfg: Config, neighbour_graph: NeighbourGraph, embed_dim: int = 1, gating_fn: GatingFnName = "exp"):
        self.cfg = cfg
        self.ng = neighbour_graph
        self.n_bins = neighbour_graph.n_bins
        self.embed_dim = embed_dim

        if embed_dim > 1:
            self.gating = make_gating_function(
                gating_fn,
                alpha=cfg.gating_alpha,
                kappa=cfg.gating_kappa,
                epsilon=cfg.gating_epsilon,
            )

        # cached sparse matrices (both modes)
        self.V: Optional[sparse.csr_matrix] = None         # observation-to-bin indicator
        self.H_smooth: Optional[sparse.csr_matrix] = None  # neighbor smoothing matrix
        self.I_minus_H_smooth: Optional[sparse.csr_matrix] = None  # (I - H_smooth)

        # scalar-mode precomputed matrices (embed_dim == 1 only)
        self.A: Optional[sparse.csc_matrix] = None    # precomputed LHS (V@H_smooth)^T(V@H_smooth) + reg
        self.S: Optional[sparse.csc_matrix] = None    # (I - H_smooth)^T (I - H_smooth)
        self.VH: Optional[sparse.csr_matrix] = None   # V @ H_smooth, used for RHS in CG solve

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
                # No neighbors: identity smoothing
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

        I = sparse.identity(self.n_bins, format="csr")
        I_minus_H_smooth = I - H_smooth
        self.I_minus_H_smooth = sparse.csr_matrix(I_minus_H_smooth)

        if self.embed_dim == 1:
            # Step 3 (scalar only): Precompute A = (VH)^T(VH) + r*I + λ*(I-H_smooth)^T(I-H_smooth)
            # VH = V @ H_smooth so that the data-fidelity term uses the interpolated latent (H D)[b].
            VH = (V @ H_smooth).tocsr()
            self.VH = VH
            VHtVH = (VH.T @ VH).tocsc()
            smoothness_term = (I_minus_H_smooth.T @ I_minus_H_smooth).tocsc()
            l2_term = I.tocsc()
            self.A = VHtVH + self.cfg.latent_l2_reg * l2_term + self.cfg.latent_smooth_reg * smoothness_term
            self.S = smoothness_term
        
        log.info(f"Latent solver: V={V.shape}, H_smooth={H_smooth.shape}, embed_dim={self.embed_dim}, "
                 f"L2={self.cfg.latent_l2_reg}, smooth={self.cfg.latent_smooth_reg}")


    def solve(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        final_weights: Optional[np.ndarray] = None,
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
            intrinsic_vec: MLP predictions — shape (N_obs,) for scalar mode,
                           (N_obs, embed_dim) for vector mode
            final_weights: final linear layer weights w ∈ R^d (required for embed_dim > 1)
            bin_ids: bin index per observation (N_obs,)
            sample_ids: sample index per observation (N_obs,), required for cross_entropy
            loss_type: "logistic" (sigmoid loss) or "cross_entropy" (softmax loss)
            x0: warm-start initial latent
            prox_weight: proximal regularization weight ρ ≥ 0.  Adds (ρ/2)||D - x_anchor||²
                         to the objective, anchoring the update near the current latent.
                         Annealed to zero over the warmup phase (standard damped/proximal EM).
            x_anchor: anchor point in model-space (same shape as x0).  Defaults to x0.

        Returns:
            latent: shape (n_bins,) for scalar mode, (n_bins, embed_dim) for vector mode
        """
        # Default anchor to warm-start if not explicitly provided
        anchor = x_anchor if x_anchor is not None else x0

        if self.embed_dim > 1:
            if final_weights is None:
                raise ValueError("final_weights are required for embed_dim > 1")
            if bin_ids is None:
                raise ValueError("bin_ids are required for embed_dim > 1")
            if loss_type == "logistic":
                return self._solve_logistic_vector(
                    y=y, intrinsic_vec=intrinsic_vec, final_weights=final_weights,
                    bin_ids=bin_ids, x0=x0, prox_weight=prox_weight, x_anchor=anchor,
                )
            else:
                if sample_ids is None:
                    raise ValueError("sample_ids are required for cross_entropy latent solving")
                return self._solve_cross_entropy_lbfgs_vector(
                    y=y, intrinsic_vec=intrinsic_vec, final_weights=final_weights,
                    bin_ids=bin_ids, sample_ids=sample_ids, x0=x0,
                    prox_weight=prox_weight, x_anchor=anchor,
                )
        else:
            if loss_type == "logistic":
                return self._solve_logistic(
                    y=y, intrinsic_vec=intrinsic_vec, x0=x0,
                    prox_weight=prox_weight, x_anchor=anchor,
                )
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
                    x_anchor=anchor,
                )

    def _solve_logistic(self, y: np.ndarray, intrinsic_vec: np.ndarray, x0: Optional[np.ndarray] = None, prox_weight: float = 0.0, x_anchor: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Scalar logistic/sigmoid latent solve via linear system + CG.
        Uses interpolated latent (H_smooth @ D)[bin] matching the forward pass.

        System: ((VH)^T(VH) + rI + λ(I-H_smooth)^T(I-H_smooth)) D = (VH)^T (logit(y) - m)
        where VH = V @ H_smooth.

        With proximal regularization (ρ > 0):
            System becomes (A + ρI) D = b + ρ * (anchor / latent_lr),
            anchoring the output-space solution near x_anchor.
        """
        if self.V is None or self.A is None or self.H_smooth is None or self.I_minus_H_smooth is None:
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
            # Recompute A matrix with filtered VH = V_filtered @ H_smooth
            VH_filtered = (V_filtered @ self.H_smooth).tocsr()
            VHtVH = (VH_filtered.T @ VH_filtered).tocsc()
            I = sparse.identity(self.n_bins, format="csr")
            I_minus_H_smooth = self.I_minus_H_smooth
            smoothness_term = (I_minus_H_smooth.T @ I_minus_H_smooth).tocsc()
            l2_term = I.tocsc()
            A_filtered = VHtVH + self.cfg.latent_l2_reg * l2_term + self.cfg.latent_smooth_reg * smoothness_term

            y_use = y_filtered
            intrinsic_use = intrinsic_filtered
            VH_use = VH_filtered
            A_use = A_filtered
        else:
            y_use = y
            intrinsic_use = intrinsic_vec
            VH_use = self.VH
            A_use = self.A

        # Clip y to avoid log(0) and convert to logits
        y_clipped = np.clip(y_use, 1e-7, 1 - 1e-7)
        y_logit = np.log(y_clipped / (1 - y_clipped))  # logit transform
        
        # Compute residual in logit space: D should satisfy intrinsic + D ≈ logit(y)
        # So residual = logit(y) - intrinsic
        residual = y_logit - intrinsic_use
        
        # Compute b = V^T residual
        # Compute b = (VH)^T residual  (interpolated RHS matching the new LHS)
        b = VH_use.T.dot(residual)

        # Solve A D = b using conjugate gradient
        if x0 is None:
            x0_use = np.zeros(self.n_bins, dtype=float)
        else:
            x0_use = np.asarray(x0, dtype=float).reshape(-1)
            if x0_use.shape[0] != self.n_bins:
                raise ValueError(f"x0 has shape {x0_use.shape}, expected ({self.n_bins},)")

        # Proximal regularization: adds (ρ/2)||D*latent_lr - anchor||² to the objective.
        # In solver space (D before latent_lr scaling): (A + ρI) D = b + ρ * (anchor / latent_lr).
        if prox_weight > 0 and x_anchor is not None:
            lr = max(float(self.cfg.latent_lr), 1e-8)
            anchor_solver = np.asarray(x_anchor, dtype=float).reshape(-1) / lr
            def _matvec(v, _A=A_use, _p=prox_weight): return _A.dot(v) + _p * v
            A_eff = LinearOperator(A_use.shape, matvec=_matvec, dtype=float)
            b_eff = b + prox_weight * anchor_solver
        else:
            A_eff, b_eff = A_use, b

        # Rescale latent learning rate
        D, info = cg(
            A_eff,
            b_eff,
            x0=x0_use,
            atol=self.cfg.cg_tol,
            maxiter=self.cfg.cg_maxiter
        )
        D = D * self.cfg.latent_lr  # Rescale the solution by the latent learning rate to keep magnitudes stable across different regularization strengths
        
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
        Uses interpolated latent (H_smooth @ D)[bin] matching the forward pass.

        We minimize:
            sum_s CE(y_s, softmax(m_s + (H_smooth D)_bins)) + (r/2)||D||^2 + (λ/2)||(I-H)D||^2

        Gradient chain rule: dL_data/dD = H_smooth^T @ (dL_data / d(H_smooth D))
        """
        if self.I_minus_H_smooth is None or self.H_smooth is None:
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
        I_minus_H_smooth = self.I_minus_H_smooth  # csr

        # Warm start
        if x0 is None:
            x0_use = np.zeros(self.n_bins, dtype=np.float64)
        else:
            x0_use = np.asarray(x0, dtype=np.float64).reshape(-1)
            if x0_use.shape[0] != self.n_bins:
                raise ValueError(f"x0 has shape {x0_use.shape}, expected ({self.n_bins},)")

        # Proximal anchor in parameter space (no latent_lr scaling for L-BFGS solvers)
        x_anchor_flat = np.asarray(x_anchor, dtype=np.float64).reshape(-1) if (prox_weight > 0 and x_anchor is not None) else None

        H_smooth = self.H_smooth  # captured for closure

        def fun_and_jac(D_flat: np.ndarray) -> tuple[float, np.ndarray]:
            D_flat = D_flat.astype(np.float64, copy=False)

            # Interpolate: D_interp = H_smooth @ D  (matches forward pass)
            D_interp = H_smooth.dot(D_flat)

            # z_i = m_i + (H_smooth D)_{bin(i)}
            z = m_s + D_interp[b_s]

            # Accumulate dL_data/d(D_interp) per bin; chain-ruled to D below
            grad_D_interp = np.zeros(self.n_bins, dtype=np.float64)
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
                np.add.at(grad_D_interp, b_s[st:en], g_z)

            # Chain rule: dL_data/dD = H_smooth^T @ (dL_data/d(H_smooth D))
            grad_bins = H_smooth.T.dot(grad_D_interp)

            # Regularization (on D directly, not on the interpolated latent)
            if r > 0:
                loss += 0.5 * r * float(D_flat @ D_flat)
                grad_bins += r * D_flat

            if lam > 0:
                diff = I_minus_H_smooth.dot(D_flat)  # (I-H_smooth)D
                loss += 0.5 * lam * float(diff @ diff)
                grad_bins += lam * I_minus_H_smooth.T.dot(diff)

            # Proximal term: (ρ/2)||D - anchor||²  — anchors update near previous latent
            if prox_weight > 0 and x_anchor_flat is not None:
                d_prox = D_flat - x_anchor_flat
                loss += 0.5 * prox_weight * float(d_prox @ d_prox)
                grad_bins += prox_weight * d_prox

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
                "maxiter": int(self.cfg.cg_maxiter),
                "ftol": float(self.cfg.cg_tol),
            },
        )

        if not res.success:
            log.warning(f"L-BFGS did not converge: {res.message}")

        D = np.asarray(res.x, dtype=np.float64)
        log.debug(f"Latent D (CE): mean={D.mean():.3f}, std={D.std():.3f}, range=[{D.min():.3f}, {D.max():.3f}]")
        return D

    # ------------------------------------------------------------------
    # Vector-mode solvers (embed_dim > 1)
    # ------------------------------------------------------------------

    def _solve_logistic_vector(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        final_weights: np.ndarray,
        bin_ids: np.ndarray,
        x0: Optional[np.ndarray] = None,
        prox_weight: float = 0.0,
        x_anchor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vector logistic/sigmoid latent solve via L-BFGS with multiplicative gating.
        Uses interpolated latent (H_smooth @ H)[bin] matching the forward pass.

        Model: ŷ_i = sigmoid(w^T (m_i ⊙ g((H_smooth H)[bin_i])))
        Objective: BCE(y, ŷ) + (r/2)||H||^2 + (λ/2)||(I-H_smooth)H||^2
        Gradient chain rule: dL_data/dH = H_smooth^T @ (dL_data / d(H_smooth H))
        """
        if self.V is None or self.H_smooth is None or self.I_minus_H_smooth is None:
            raise RuntimeError("Matrices not built; call build_V_and_H first")

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        intrinsic_vec = np.asarray(intrinsic_vec, dtype=np.float64)  # (N_obs, d)
        final_weights = np.asarray(final_weights, dtype=np.float64).reshape(-1)  # (d,)
        bin_ids = np.asarray(bin_ids, dtype=np.int64).reshape(-1)

        if intrinsic_vec.shape != (len(y), self.embed_dim):
            raise ValueError(f"intrinsic_vec has shape {intrinsic_vec.shape}, expected ({len(y)}, {self.embed_dim})")
        if final_weights.shape[0] != self.embed_dim:
            raise ValueError(f"final_weights has shape {final_weights.shape}, expected ({self.embed_dim},)")

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

        if x0 is None:
            x0_use = np.zeros((self.n_bins, self.embed_dim), dtype=np.float64)
        else:
            x0_use = np.asarray(x0, dtype=np.float64)
            if x0_use.shape != (self.n_bins, self.embed_dim):
                raise ValueError(f"x0 has shape {x0_use.shape}, expected ({self.n_bins}, {self.embed_dim})")

        I_minus_H_smooth = self.I_minus_H_smooth  # local var for closure type safety
        H_smooth = self.H_smooth  # captured for closure
        x_anchor_mat = np.asarray(x_anchor, dtype=np.float64).reshape(self.n_bins, self.embed_dim) if (prox_weight > 0 and x_anchor is not None) else None

        def fun_and_jac(H_flat):
            H_flat = H_flat.astype(np.float64, copy=False)
            H = H_flat.reshape(self.n_bins, self.embed_dim)

            # Interpolate: H_interp = H_smooth @ H  (matches forward pass)
            H_interp = H_smooth.dot(H)                                        # (n_bins, d)
            h_obs = H_interp[bin_ids_use]                                     # (N_obs, d)
            m_tilde = intrinsic_use * self.gating.gate_np(h_obs)              # (N_obs, d)
            logits = m_tilde @ final_weights                                   # (N_obs,)

            logits_stable = np.clip(logits, -20, 20)
            p = 1.0 / (1.0 + np.exp(-logits_stable))
            p_clip = np.clip(p, 1e-7, 1 - 1e-7)
            loss = float(-(y_use * np.log(p_clip) + (1 - y_use) * np.log(1 - p_clip)).sum())

            grad_logits = p - y_use                                           # (N_obs,)
            grad_m_tilde = np.outer(grad_logits, final_weights)               # (N_obs, d)
            grad_h_obs = intrinsic_use * grad_m_tilde * self.gating.gate_grad_np(h_obs)  # (N_obs, d)

            # Accumulate dL_data/d(H_interp) then chain-rule back to H
            grad_H_interp = np.zeros((self.n_bins, self.embed_dim), dtype=np.float64)
            np.add.at(grad_H_interp, bin_ids_use, grad_h_obs)
            # Chain rule: dL_data/dH = H_smooth^T @ (dL_data/d(H_interp))
            grad_H = H_smooth.T.dot(grad_H_interp)

            if r > 0:
                loss += 0.5 * r * float(np.sum(H ** 2))
                grad_H += r * H

            if lam > 0:
                diff = I_minus_H_smooth.dot(H)  # (n_bins, d)
                loss += 0.5 * lam * float(np.sum(diff ** 2))
                grad_H += lam * (I_minus_H_smooth.T.dot(diff))

            # Proximal term: (ρ/2)||H - anchor||²  — anchors update near previous latent
            if prox_weight > 0 and x_anchor_mat is not None:
                d_prox = H - x_anchor_mat
                loss += 0.5 * prox_weight * float(np.sum(d_prox ** 2))
                grad_H += prox_weight * d_prox

            if N_obs > 0:
                loss /= N_obs
                grad_H /= N_obs

            return loss, grad_H.ravel()

        def fun(H_flat): f, _ = fun_and_jac(H_flat); return f
        def jac(H_flat): _, g = fun_and_jac(H_flat); return g

        res = minimize(
            fun=fun, x0=x0_use.ravel(), jac=jac, method="L-BFGS-B",
            options={"maxiter": int(self.cfg.cg_maxiter), "ftol": float(self.cfg.cg_tol)},
        )
        if not res.success:
            log.warning(f"L-BFGS did not converge: {res.message}")

        H = res.x.reshape(self.n_bins, self.embed_dim)
        log.debug(f"Latent H: mean={H.mean():.3f}, std={H.std():.3f}, range=[{H.min():.3f}, {H.max():.3f}]")
        return H

    def _solve_cross_entropy_lbfgs_vector(
        self,
        y: np.ndarray,
        intrinsic_vec: np.ndarray,
        final_weights: np.ndarray,
        bin_ids: np.ndarray,
        sample_ids: np.ndarray,
        x0: Optional[np.ndarray] = None,
        prox_weight: float = 0.0,
        x_anchor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vector cross-entropy latent solve (softmax over bins within each sample) via L-BFGS.
        Uses interpolated latent (H_smooth @ H)[bin] matching the forward pass.

        Model: p_i = softmax(w^T (m_i ⊙ g((H_smooth H)[bin_i])))_within_sample
        Objective: CE(y, p) + (r/2)||H||^2 + (λ/2)||(I-H_smooth)H||^2
        Gradient chain rule: dL_data/dH = H_smooth^T @ (dL_data / d(H_smooth H))
        """
        if self.I_minus_H_smooth is None or self.H_smooth is None:
            raise RuntimeError("Matrices not built; call build_V_and_H first")
        if self.cfg.latent_present_only:
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

        order = np.argsort(sample_ids, kind="mergesort")
        y_s = y[order]
        m_s = intrinsic_vec[order]  # (N_obs, d)
        b_s = bin_ids[order]
        s_s = sample_ids[order]

        unique_s, starts = np.unique(s_s, return_index=True)
        ends = np.append(starts[1:], len(s_s))
        n_samples = len(unique_s)

        r = float(self.cfg.latent_l2_reg)
        lam = float(self.cfg.latent_smooth_reg)

        if x0 is None:
            x0_use = np.zeros((self.n_bins, self.embed_dim), dtype=np.float64)
        else:
            x0_use = np.asarray(x0, dtype=np.float64)
            if x0_use.shape != (self.n_bins, self.embed_dim):
                raise ValueError(f"x0 has shape {x0_use.shape}, expected ({self.n_bins}, {self.embed_dim})")

        I_minus_H_smooth = self.I_minus_H_smooth  # local var for closure type safety
        H_smooth = self.H_smooth  # captured for closure
        x_anchor_mat = np.asarray(x_anchor, dtype=np.float64).reshape(self.n_bins, self.embed_dim) if (prox_weight > 0 and x_anchor is not None) else None

        def fun_and_jac(H_flat):
            H_flat = H_flat.astype(np.float64, copy=False)
            H = H_flat.reshape(self.n_bins, self.embed_dim)

            # Interpolate: H_interp = H_smooth @ H  (matches forward pass)
            H_interp = H_smooth.dot(H)                            # (n_bins, d)
            h_obs = H_interp[b_s]                                 # (N_obs, d)
            m_tilde = m_s * self.gating.gate_np(h_obs)            # (N_obs, d)
            logits = m_tilde @ final_weights                       # (N_obs,)

            grad_H_interp = np.zeros((self.n_bins, self.embed_dim), dtype=np.float64)
            loss = 0.0

            for st, en in zip(starts, ends):
                z_seg = logits[st:en]
                y_seg = y_s[st:en]
                m_seg = m_s[st:en]
                h_obs_seg = h_obs[st:en]
                b_seg = b_s[st:en]

                z_max = float(np.max(z_seg)) if len(z_seg) > 0 else 0.0
                exp_z = np.exp(z_seg - z_max)
                denom = float(exp_z.sum()) + 1e-300
                p = exp_z / denom

                logsumexp = z_max + np.log(denom)
                loss += float(-(y_seg * (z_seg - logsumexp)).sum())

                grad_logits_seg = p - y_seg
                grad_m_tilde_seg = np.outer(grad_logits_seg, final_weights)             # (seg, d)
                grad_h_obs_seg = m_seg * grad_m_tilde_seg * self.gating.gate_grad_np(h_obs_seg)  # (seg, d)
                np.add.at(grad_H_interp, b_seg, grad_h_obs_seg)

            # Chain rule: dL_data/dH = H_smooth^T @ (dL_data/d(H_interp))
            grad_H = H_smooth.T.dot(grad_H_interp)

            if r > 0:
                loss += 0.5 * r * float(np.sum(H ** 2))
                grad_H += r * H

            if lam > 0:
                diff = I_minus_H_smooth.dot(H)  # (n_bins, d)
                loss += 0.5 * lam * float(np.sum(diff ** 2))
                grad_H += lam * (I_minus_H_smooth.T.dot(diff))

            # Proximal term: (ρ/2)||H - anchor||²  — anchors update near previous latent
            if prox_weight > 0 and x_anchor_mat is not None:
                d_prox = H - x_anchor_mat
                loss += 0.5 * prox_weight * float(np.sum(d_prox ** 2))
                grad_H += prox_weight * d_prox

            if n_samples > 0:
                loss /= n_samples
                grad_H /= n_samples

            return loss, grad_H.ravel()

        def fun(H_flat): f, _ = fun_and_jac(H_flat); return f
        def jac(H_flat): _, g = fun_and_jac(H_flat); return g

        res = minimize(
            fun=fun, x0=x0_use.ravel(), jac=jac, method="L-BFGS-B",
            options={"maxiter": int(self.cfg.cg_maxiter), "ftol": float(self.cfg.cg_tol)},
        )
        if not res.success:
            log.warning(f"L-BFGS did not converge: {res.message}")

        H = res.x.reshape(self.n_bins, self.embed_dim)
        log.debug(f"Latent H (CE): mean={H.mean():.3f}, std={H.std():.3f}, range=[{H.min():.3f}, {H.max():.3f}]")
        return H
