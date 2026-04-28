"""
DQN Neural Network.
Simple MLP that maps state -> Q-values for each action.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """3-layer MLP with ReLU activations."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (batch, state_dim) -> Output: (batch, action_dim)"""
        return self.net(x)
