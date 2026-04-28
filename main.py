"""
Main entry point for the RL Taxi Dispatch system.
Usage:
    python main.py train          # Train DQN
    python main.py eval           # Evaluate trained DQN
    python main.py baselines      # Run non-RL baselines
    python main.py compare        # Compare DQN vs baselines
    python main.py demo           # Quick demo (200 episodes)
    python main.py multi-seed     # Multi-seed DQN evaluation
    python main.py ablation       # Reward ablation study
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config
from environment import TaxiEnv
from dqn_agent import DQNAgent
from train import train, evaluate, evaluate_multi_seed, METRIC_KEYS
from baselines import BASELINES


def _make_config(seed: Optional[int] = None, ablation: Optional[str] = None) -> Config:
    """Build config with optional seed and ablation."""
    config = Config()
    if seed is not None:
        config.apply_seed(seed)
    if ablation is not None:
        _apply_ablation(config, ablation)
    return config


def _apply_ablation(config: Config, name: str) -> None:
    """Apply named ablation to reward config."""
    r = config.reward
    if name == "neutral":
        # Only pickup/dropoff rewards, no penalties
        r.use_occupancy_bonus = False
        r.use_waiting_penalty = False
        r.use_eta_violation = False
        r.use_detour_penalty = False
        r.use_idle_penalty = False
    elif name == "no_detour":
        r.use_detour_penalty = False
    elif name == "no_idle":
        r.use_idle_penalty = False
    elif name == "no_waiting":
        r.use_waiting_penalty = False
    elif name == "no_occupancy":
        r.use_occupancy_bonus = False
    elif name == "no_eta":
        r.use_eta_violation = False
    else:
        print(f"[WARN] Unknown ablation: {name}. Using default rewards.")


def run_baseline(
    config: Config,
    name: str,
    num_episodes: int = 50,
    seed: Optional[int] = None,
) -> dict:
    """Run a baseline policy for N episodes."""
    env = TaxiEnv(config)
    policy = BASELINES[name]()

    metrics_accum: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        env.reset(seed=ep_seed)
        info: Dict[str, Any] = {}
        while not env.done:
            action = policy.select_action(env)
            _, _, _, info = env.step(action)
        for key in METRIC_KEYS:
            metrics_accum[key].append(info.get(key, 0.0))

    results = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
    return results


def run_baseline_multi_seed(
    config: Config,
    name: str,
    num_episodes: int = 20,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run baseline across multiple seeds, return mean/std/ci95."""
    if seeds is None:
        seeds = [42, 123, 456]

    all_results: List[Dict[str, float]] = []
    for seed in seeds:
        r = run_baseline(config, name, num_episodes=num_episodes, seed=seed)
        all_results.append(r)

    summary: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in all_results]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / max(len(vals) ** 0.5, 1e-9)
        summary[key] = {"mean": mean, "std": std, "ci95": ci95}
    return summary


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(args) -> None:
    seed = args.seed if hasattr(args, "seed") and args.seed else None
    config = _make_config(seed=seed)
    train(config)


def cmd_eval(args) -> None:
    seed = args.seed if hasattr(args, "seed") and args.seed else None
    config = _make_config(seed=seed)
    agent = DQNAgent(config)
    ckpt = args.checkpoint or "checkpoints/best.pt"
    try:
        agent.load(ckpt)
    except FileNotFoundError:
        print(f"Checkpoint not found: {ckpt}")
        print("Train first: python main.py train")
        sys.exit(1)

    results = evaluate(agent, config, num_episodes=20, seed=seed)
    print("\n=== DQN Evaluation (20 episodes) ===")
    for k, v in results.items():
        print(f"  {k:>20s}: {v:.4f}")


def cmd_baselines(args) -> None:
    seed = args.seed if hasattr(args, "seed") and args.seed else None
    config = _make_config(seed=seed)
    num_ep = args.episodes or 50
    print(f"\nRunning baselines ({num_ep} episodes each)...\n")
    print(f"{'Baseline':<20s} {'Reward':>8s} {'Pickups':>8s} {'Dropoffs':>9s} "
          f"{'AvgWait':>8s} {'OnTime':>8s} {'Occup':>8s}")
    print("-" * 75)
    for name in BASELINES:
        results = run_baseline(config, name, num_episodes=num_ep, seed=seed)
        print(
            f"{name:<20s} {results['total_reward']:8.2f} {results['pickups']:8.1f} "
            f"{results['dropoffs']:9.1f} {results['avg_wait']:8.2f} "
            f"{results['on_time_rate']:8.3f} {results['avg_occupancy']:8.3f}"
        )


def cmd_compare(args) -> None:
    seed = args.seed if hasattr(args, "seed") and args.seed else None
    config = _make_config(seed=seed)
    num_ep = args.episodes or 50

    print(f"\n=== Comparison ({num_ep} episodes) ===\n")

    # Baselines
    rows = []
    for name in BASELINES:
        results = run_baseline(config, name, num_episodes=num_ep, seed=seed)
        rows.append((name, results))

    # DQN
    agent = DQNAgent(config)
    ckpt = args.checkpoint or "checkpoints/best.pt"
    try:
        agent.load(ckpt)
        dqn_results = evaluate(agent, config, num_episodes=num_ep, seed=seed)
        rows.append(("DQN (trained)", dqn_results))
    except FileNotFoundError:
        print("[WARN] No DQN checkpoint found, skipping DQN eval.")

    print(f"{'Policy':<20s} {'Reward':>8s} {'Pickups':>8s} {'Dropoffs':>9s} "
          f"{'AvgWait':>8s} {'OnTime':>8s} {'Occup':>8s}")
    print("-" * 75)
    for name, r in rows:
        print(
            f"{name:<20s} {r['total_reward']:8.2f} {r['pickups']:8.1f} "
            f"{r['dropoffs']:9.1f} {r['avg_wait']:8.2f} "
            f"{r['on_time_rate']:8.3f} {r['avg_occupancy']:8.3f}"
        )


def cmd_demo(args) -> None:
    """Quick demo - train 200 episodes, then show results."""
    seed = args.seed if hasattr(args, "seed") and args.seed else None
    config = _make_config(seed=seed)
    config.train.num_episodes = 200
    config.train.log_interval = 50
    config.train.save_interval = 100

    print("=" * 60)
    print("  RL TAXI DISPATCH - QUICK DEMO (200 episodes)")
    if seed is not None:
        print(f"  Seed: {seed}")
    print("=" * 60)

    agent = train(config)

    # Evaluate
    print("\nEvaluating trained agent...")
    results = evaluate(agent, config, num_episodes=20, seed=seed)
    print("\n=== DQN Results ===")
    for k, v in results.items():
        print(f"  {k:>20s}: {v:.4f}")

    # Baselines
    print("\n=== Baselines ===")
    for name in BASELINES:
        bresults = run_baseline(config, name, num_episodes=20, seed=seed)
        print(f"\n  [{name}]")
        for k, v in bresults.items():
            print(f"    {k:>18s}: {v:.4f}")


def cmd_multi_seed(args) -> None:
    """Evaluate DQN and baselines across multiple seeds."""
    config = _make_config()
    num_ep = args.episodes or 20
    seeds = [42, 123, 456]
    ckpt = args.checkpoint or "checkpoints/best.pt"

    print(f"\n=== Multi-Seed Evaluation ({num_ep} episodes x {len(seeds)} seeds) ===\n")

    # Baselines
    rows = []
    for name in BASELINES:
        summary = run_baseline_multi_seed(config, name, num_episodes=num_ep, seeds=seeds)
        rows.append((name, summary))

    # DQN
    agent = DQNAgent(config)
    try:
        agent.load(ckpt)
        dqn_summary = evaluate_multi_seed(agent, config, num_episodes=num_ep, seeds=seeds)
        rows.append(("DQN (trained)", dqn_summary))
    except FileNotFoundError:
        print("[WARN] No DQN checkpoint found, skipping DQN eval.")

    print(f"{'Policy':<20s} {'Reward':>16s} {'AvgWait':>16s} {'OnTime':>16s}")
    print("-" * 72)
    for name, s in rows:
        r = s["total_reward"]
        w = s["avg_wait"]
        o = s["on_time_rate"]
        print(
            f"{name:<20s} {r['mean']:7.1f}±{r['ci95']:5.1f} "
            f"{w['mean']:7.2f}±{w['ci95']:5.2f} "
            f"{o['mean']:7.3f}±{o['ci95']:5.3f}"
        )

    # Save JSON
    out = {name: {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in s.items()} for name, s in rows}
    with open("results_multi_seed.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results_multi_seed.json")


def cmd_ablation(args) -> None:
    """Run ablation study: compare different reward configurations."""
    seed = args.seed if hasattr(args, "seed") and args.seed else None
    num_ep = args.episodes or 30

    ablations = ["default", "neutral", "no_detour", "no_idle", "no_waiting", "no_occupancy"]

    print(f"\n=== Reward Ablation Study ({num_ep} episodes) ===\n")
    print(f"{'Ablation':<20s} {'Reward':>8s} {'Pickups':>8s} {'Dropoffs':>9s} "
          f"{'AvgWait':>8s} {'OnTime':>8s} {'Occup':>8s}")
    print("-" * 75)

    for abl in ablations:
        config = _make_config(seed=seed, ablation=abl if abl != "default" else None)
        config.train.num_episodes = num_ep * 3  # quick train per ablation
        config.train.log_interval = max(num_ep * 3, 50)
        config.train.save_interval = config.train.num_episodes + 1  # no save

        # Quick train
        agent = DQNAgent(config)
        env = TaxiEnv(config)
        for ep in range(config.train.num_episodes):
            if config.env.seed is not None:
                env.e.seed = config.env.seed + ep
            state = env.reset()
            while not env.done:
                valid = env.get_candidate_actions()
                action = agent.select_action(state, valid)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step_fn()
                state = next_state

        # Evaluate
        results = evaluate(agent, config, num_episodes=num_ep, seed=seed)
        print(
            f"{abl:<20s} {results['total_reward']:8.2f} {results['pickups']:8.1f} "
            f"{results['dropoffs']:9.1f} {results['avg_wait']:8.2f} "
            f"{results['on_time_rate']:8.3f} {results['avg_occupancy']:8.3f}"
        )

    print("\nAblation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL Taxi Dispatch System")
    sub = parser.add_subparsers(dest="command")

    # train
    p = sub.add_parser("train", help="Train DQN agent")
    p.add_argument("--seed", type=int, default=None)

    # eval
    p = sub.add_parser("eval", help="Evaluate trained DQN agent")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    # baselines
    p = sub.add_parser("baselines", help="Run non-RL baselines")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)

    # compare
    p = sub.add_parser("compare", help="Compare DQN vs baselines")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)

    # demo
    p = sub.add_parser("demo", help="Quick demo (200 episodes)")
    p.add_argument("--seed", type=int, default=None)

    # multi-seed
    p = sub.add_parser("multi-seed", help="Multi-seed DQN + baseline evaluation")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--episodes", type=int, default=20)

    # ablation
    p = sub.add_parser("ablation", help="Reward ablation study")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    commands = {
        "train": cmd_train,
        "eval": cmd_eval,
        "baselines": cmd_baselines,
        "compare": cmd_compare,
        "demo": cmd_demo,
        "multi-seed": cmd_multi_seed,
        "ablation": cmd_ablation,
    }
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
