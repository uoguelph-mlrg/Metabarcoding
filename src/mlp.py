import numpy as np
import torch
import torch.nn as nn
from typing import List

class MLPModel(nn.Module):
    """
    MLP that predicts m_theta(s,b) given sample and bin features.
    Simple architecture with light regularization for small datasets.
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int] = [128, 128, 128, 128],
        output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always returns [N, output_dim]; callers squeeze when needed.
        return self.net(x)