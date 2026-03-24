from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F


class GatingFunction(ABC):
    """
    Abstract base for gating functions g(h) with the property g(0) = 1.

    Used for multiplicative modulation: m̃ = m(x) ⊙ g(h[bin])
    Logit: z = w^T m̃

    Subclasses implement the forward and gradient in both numpy (for the
    L-BFGS latent solver) and PyTorch (for the MLP training forward pass).
    """

    @abstractmethod
    def gate_np(self, h: np.ndarray) -> np.ndarray:
        """Compute g(h) — numpy, used by LatentSolver."""

    @abstractmethod
    def gate_grad_np(self, h: np.ndarray) -> np.ndarray:
        """Compute g'(h) — numpy gradient, used by LatentSolver."""

    @abstractmethod
    def gate_torch(self, h: torch.Tensor) -> torch.Tensor:
        """Compute g(h) — PyTorch, used by Model.forward."""


# ---------------------------------------------------------------------------
# Concrete gating functions
# ---------------------------------------------------------------------------

class ExpGating(GatingFunction):
    """g(h) = exp(h),  g(0) = 1."""

    def gate_np(self, h):
        return np.exp(h)

    def gate_grad_np(self, h):
        return np.exp(h)

    def gate_torch(self, h):
        return torch.exp(h)


class ScaledExpGating(GatingFunction):
    """g(h) = exp(α·h),  g(0) = 1.  α ∈ (0, 1] controls curvature."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def gate_np(self, h):
        return np.exp(self.alpha * h)

    def gate_grad_np(self, h):
        return self.alpha * np.exp(self.alpha * h)

    def gate_torch(self, h):
        return torch.exp(self.alpha * h)


class AdditiveGating(GatingFunction):
    """g(h) = 1 + h,  g(0) = 1.  Linear, unbounded."""

    def gate_np(self, h):
        return 1.0 + h

    def gate_grad_np(self, h):
        return np.ones_like(h)

    def gate_torch(self, h):
        return 1.0 + h


class SoftplusGating(GatingFunction):
    """g(h) = 1 + softplus(h) − ε,  g(0) = 1 when ε = log(2) ≈ 0.693."""

    def __init__(self, epsilon: float = 0.693):
        self.epsilon = epsilon

    def gate_np(self, h):
        return 1.0 + np.log1p(np.exp(h)) - self.epsilon

    def gate_grad_np(self, h):
        return 1.0 / (1.0 + np.exp(-h))  # σ(h) = derivative of softplus

    def gate_torch(self, h):
        return 1.0 + F.softplus(h) - self.epsilon


class TanhGating(GatingFunction):
    """g(h) = 1 + κ·tanh(h),  g(0) = 1.  Bounded to [1−κ, 1+κ]."""

    def __init__(self, kappa: float = 0.5):
        self.kappa = kappa

    def gate_np(self, h):
        return 1.0 + self.kappa * np.tanh(h)

    def gate_grad_np(self, h):
        return self.kappa * (1.0 - np.tanh(h) ** 2)

    def gate_torch(self, h):
        return 1.0 + self.kappa * torch.tanh(h)


class SigmoidGating(GatingFunction):
    """g(h) = 2·σ(h),  g(0) = 1.  Bounded to (0, 2)."""

    def gate_np(self, h):
        return 2.0 / (1.0 + np.exp(-h))

    def gate_grad_np(self, h):
        s = 1.0 / (1.0 + np.exp(-h))
        return 2.0 * s * (1.0 - s)

    def gate_torch(self, h):
        return 2.0 * torch.sigmoid(h)


class DotProductGating(GatingFunction):
    """
    g(h) = h  →  m̃ = m ⊙ h,  z = w^T (m ⊙ h).

    Note: g(0) = 0, so the latent must be non-zero to produce non-trivial
    predictions.  Unlike the other gating functions, this variant passes
    through the final linear layer (w) just like all other gating functions.
    """

    def gate_np(self, h):
        return h

    def gate_grad_np(self, h):
        return np.ones_like(h)

    def gate_torch(self, h):
        return h


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GATING_MAP: dict[str, type] = {
    "exp":          ExpGating,
    "scaled_exp":   ScaledExpGating,
    "additive":     AdditiveGating,
    "softplus":     SoftplusGating,
    "tanh":         TanhGating,
    "sigmoid":      SigmoidGating,
    "dot_product":  DotProductGating,
}


def make_gating_function(
    name: Literal["exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid", "dot_product"],
    alpha: float = 0.5,    # for ScaledExpGating
    kappa: float = 0.5,    # for TanhGating
    epsilon: float = 0.693, # for SoftplusGating  (log(2) ensures g(0) = 1)
) -> GatingFunction:
    """
    Instantiate a GatingFunction by name.

    Parameters are forwarded only to the functions that use them:
        alpha   → ScaledExpGating
        kappa   → TanhGating
        epsilon → SoftplusGating

    All default values match the Config defaults and satisfy g(0) = 1.
    """
    if name not in _GATING_MAP:
        raise ValueError(f"Unknown gating function '{name}'. Choose from: {list(_GATING_MAP)}")

    if name == "scaled_exp":
        return ScaledExpGating(alpha=alpha)
    if name == "tanh":
        return TanhGating(kappa=kappa)
    if name == "softplus":
        return SoftplusGating(epsilon=epsilon)
    return _GATING_MAP[name]()
