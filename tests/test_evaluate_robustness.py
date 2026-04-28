"""
Test evaluate() robustness: always returns well-formed metrics dict.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from dqn_agent import DQNAgent
from train import evaluate, evaluate_multi_seed, METRIC_KEYS


def test_evaluate_returns_all_keys():
    """evaluate() must return all METRIC_KEYS."""
    config = Config()
    config.train.num_episodes = 5
    agent = DQNAgent(config)

    results = evaluate(agent, config, num_episodes=3, seed=42)

    for key in METRIC_KEYS:
        assert key in results, f"Missing key: {key}"
        assert isinstance(results[key], float), f"Key {key} should be float"

    print("[PASS] test_evaluate_returns_all_keys")


def test_evaluate_multi_seed():
    """evaluate_multi_seed() must return mean, std, ci95 for each metric."""
    config = Config()
    agent = DQNAgent(config)

    summary = evaluate_multi_seed(agent, config, num_episodes=2, seeds=[1, 2])

    for key in METRIC_KEYS:
        assert key in summary, f"Missing metric: {key}"
        for sub in ["mean", "std", "ci95"]:
            assert sub in summary[key], f"Missing {sub} in {key}"
            assert isinstance(summary[key][sub], float)

    print("[PASS] test_evaluate_multi_seed")


def test_evaluate_with_seed():
    """evaluate() with seed should be deterministic."""
    config = Config()
    agent = DQNAgent(config)

    r1 = evaluate(agent, config, num_episodes=3, seed=99)
    r2 = evaluate(agent, config, num_episodes=3, seed=99)

    for key in METRIC_KEYS:
        assert r1[key] == r2[key], f"Key {key} differs: {r1[key]} vs {r2[key]}"

    print("[PASS] test_evaluate_with_seed")


def test_evaluate_handles_no_passengers():
    """evaluate() should not crash even if no passengers spawn."""
    config = Config()
    config.env.passenger_spawn_prob = 0.0
    agent = DQNAgent(config)

    results = evaluate(agent, config, num_episodes=1, seed=1)

    for key in METRIC_KEYS:
        assert key in results

    print("[PASS] test_evaluate_handles_no_passengers")


if __name__ == "__main__":
    test_evaluate_returns_all_keys()
    test_evaluate_multi_seed()
    test_evaluate_with_seed()
    test_evaluate_handles_no_passengers()
    print("\nAll evaluate robustness tests passed.")
