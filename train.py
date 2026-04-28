"""
Training pipeline with logging and checkpointing.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config
from environment import TaxiEnv
from dqn_agent import DQNAgent


def _seed_env(env: TaxiEnv, config: Config, episode: int) -> None:
    """Apply per-episode seed derived from base seed + episode offset."""
    base = config.env.seed if config.env.seed is not None else 0
    env.e.seed = base + episode


def train(config: Config) -> DQNAgent:
    """Train DQN agent."""
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    os.makedirs(config.train.log_dir, exist_ok=True)

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=config.train.log_dir)
    except ImportError:
        writer = None
        print("[WARN] TensorBoard not available. Logging to console only.")

    env = TaxiEnv(config)
    agent = DQNAgent(config)

    print(f"State dim: {config.dqn.state_dim}, Action dim: {config.total_actions}")
    print(f"Training for {config.train.num_episodes} episodes...")
    if config.train.seed is not None:
        print(f"Seed: {config.train.seed}")
    print("-" * 60)

    all_rewards: List[float] = []
    all_losses: List[float] = []
    best_reward = float("-inf")

    for episode in range(1, config.train.num_episodes + 1):
        # Per-episode seeding for reproducibility
        if config.env.seed is not None:
            _seed_env(env, config, episode)

        state = env.reset()
        episode_reward = 0.0
        episode_losses: List[float] = []
        steps = 0
        info: Dict[str, Any] = {}

        while not env.done:
            valid_actions = env.get_candidate_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step_fn()
            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            state = next_state
            steps += 1

        all_rewards.append(episode_reward)
        if episode_losses:
            all_losses.append(float(np.mean(episode_losses)))

        # --- Logging ---
        if episode % config.train.log_interval == 0:
            avg_reward = np.mean(all_rewards[-config.train.log_interval :])
            avg_loss = np.mean(all_losses[-50:]) if all_losses else 0.0
            line = (
                f"Episode {episode:5d} | "
                f"Reward: {avg_reward:8.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Steps: {steps}"
            )
            if info:
                line += (
                    f" | Pickups: {info.get('pickups', 0)} "
                    f"| Dropoffs: {info.get('dropoffs', 0)} "
                    f"| AvgWait: {info.get('avg_wait', 0):.1f}"
                )
            print(line)

            if writer:
                writer.add_scalar("train/avg_reward", avg_reward, episode)
                writer.add_scalar("train/avg_loss", avg_loss, episode)
                writer.add_scalar("train/epsilon", agent.epsilon, episode)
                if info:
                    writer.add_scalar("train/pickups", info.get("pickups", 0), episode)
                    writer.add_scalar("train/dropoffs", info.get("dropoffs", 0), episode)
                    writer.add_scalar(
                        "train/avg_wait", info.get("avg_wait", 0), episode
                    )
                    writer.add_scalar(
                        "train/on_time_rate",
                        info.get("on_time_rate", 0),
                        episode,
                    )
                    writer.add_scalar(
                        "train/avg_occupancy",
                        info.get("avg_occupancy", 0),
                        episode,
                    )

        # --- Checkpoint ---
        if episode % config.train.save_interval == 0:
            path = os.path.join(config.train.checkpoint_dir, f"ckpt_ep{episode}.pt")
            agent.save(path)
            print(f"  -> Saved checkpoint: {path}")

        # --- Best model ---
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(config.train.checkpoint_dir, "best.pt"))

    # Final save
    agent.save(os.path.join(config.train.checkpoint_dir, "final.pt"))
    if writer:
        writer.close()

    print("-" * 60)
    print(f"Training complete. Best episode reward: {best_reward:.2f}")
    return agent


# ---------------------------------------------------------------------------
# Metrics keys
# ---------------------------------------------------------------------------
METRIC_KEYS = [
    "total_reward", "normalized_reward", "pickups", "dropoffs",
    "avg_wait", "on_time_rate", "avg_occupancy",
]


def evaluate(
    agent: DQNAgent,
    config: Config,
    num_episodes: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate agent (greedy, no exploration).
    Returns mean of each metric across episodes."""
    env = TaxiEnv(config)
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    per_episode: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        state = env.reset(seed=ep_seed)
        info: Dict[str, Any] = {}

        while not env.done:
            valid_actions = env.get_candidate_actions()
            action = agent.select_action(state, valid_actions)
            state, _, done, info = env.step(action)

        # Always capture the final info (env.step now always returns it)
        for key in METRIC_KEYS:
            per_episode[key].append(info.get(key, 0.0))

    agent.epsilon = old_epsilon

    results = {k: float(np.mean(v)) for k, v in per_episode.items()}
    return results


def evaluate_multi_seed(
    agent: DQNAgent,
    config: Config,
    num_episodes: int = 10,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate across multiple seeds, return mean, std, ci95 per metric."""
    if seeds is None:
        seeds = [42, 123, 456]

    all_results: List[Dict[str, float]] = []
    for seed in seeds:
        r = evaluate(agent, config, num_episodes=num_episodes, seed=seed)
        all_results.append(r)

    summary: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in all_results]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / max(len(vals) ** 0.5, 1e-9)
        summary[key] = {"mean": mean, "std": std, "ci95": ci95}
    return summary
