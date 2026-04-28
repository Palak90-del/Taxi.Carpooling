"""
DQN Agent with experience replay, target network, and epsilon-greedy.
"""
from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from network import QNetwork


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ReplayBuffer:
    """Fixed-size replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.dqn_cfg = config.dqn
        self.action_dim = config.total_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reproducibility
        if self.dqn_cfg.seed is not None:
            set_seed(self.dqn_cfg.seed)

        # Networks
        self.q_net = QNetwork(
            self.dqn_cfg.state_dim,
            self.action_dim,
            self.dqn_cfg.hidden_dim,
        ).to(self.device)
        self.target_net = QNetwork(
            self.dqn_cfg.state_dim,
            self.action_dim,
            self.dqn_cfg.hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.dqn_cfg.learning_rate)
        self.replay_buffer = ReplayBuffer(self.dqn_cfg.buffer_size)

        self.epsilon: float = self.dqn_cfg.epsilon_start
        self.train_step: int = 0

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[List[int]] = None,
    ) -> int:
        """Epsilon-greedy action selection over valid actions."""
        if random.random() < self.epsilon:
            if valid_actions:
                return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t).squeeze(0)

            if valid_actions:
                mask = torch.full((self.action_dim,), float("-inf"), device=self.device)
                for a in valid_actions:
                    mask[a] = 0.0
                q_values = q_values + mask

            return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step_fn(self) -> Optional[float]:
        """One gradient step. Returns loss or None if not enough data."""
        if len(self.replay_buffer) < self.dqn_cfg.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.dqn_cfg.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q
        q_values = self.q_net(states_t)
        q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            next_q_max = next_q.max(1)[0]
            target = rewards_t + (1 - dones_t) * self.dqn_cfg.gamma * next_q_max

        loss = nn.functional.mse_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step += 1

        # Update target network
        if self.train_step % self.dqn_cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        self.epsilon = max(
            self.dqn_cfg.epsilon_end,
            self.epsilon * self.dqn_cfg.epsilon_decay,
        )

        return loss.item()

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step": self.train_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.train_step = ckpt["train_step"]
