from __future__ import annotations

import os
import sys

# Reuse the canonical torch latent solver from src so this analysis
# stays aligned with active-set Adam/stateful behavior.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from latent_solver import LatentSolver as LatentSolver

__all__ = ["LatentSolver"]
