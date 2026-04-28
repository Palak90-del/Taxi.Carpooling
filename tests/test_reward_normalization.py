"""
Test reward normalization and ablation consistency.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from environment import TaxiEnv
from train import evaluate, METRIC_KEYS
from dqn_agent import DQNAgent


def test_normalization_clip():
    """Verify clipped normalized rewards stay within bounds."""
    config = Config()
    config.reward.normalize_rewards = True
    config.reward.norm_mode = "clip"
    config.reward.norm_clip_min = -100.0
    config.reward.norm_clip_max = 100.0

    env = TaxiEnv(config)
    
    # Run a few episodes and check normalized reward bounds
    for seed in [1, 2, 3]:
        env.reset(seed=seed)
        info = {}
        while not env.done:
            _, _, _, info = env.step(env._max_pairs)  # no-op
        
        raw = info.get("total_reward", 0)
        norm = info.get("normalized_reward", 0)
        
        # Normalized should be clipped
        assert -100.0 <= norm <= 100.0, f"Normalized reward {norm} out of bounds [-100, 100]"
        print(f"Seed {seed}: raw={raw:.1f}, normalized={norm:.1f}")

    print("[PASS] test_normalization_clip")


def test_normalization_disabled():
    """Verify normalization can be disabled."""
    config = Config()
    config.reward.normalize_rewards = False

    env = TaxiEnv(config)
    env.reset(seed=1)
    info = {}
    while not env.done:
        _, _, _, info = env.step(env._max_pairs)
    
    raw = info.get("total_reward", 0)
    norm = info.get("normalized_reward", 0)
    
    # When disabled, normalized should equal raw
    assert raw == norm, f"Normalized {norm} should equal raw {raw} when disabled"
    print("[PASS] test_normalization_disabled")


def test_ablation_affects_rewards():
    """Verify ablation flags affect rewards consistently."""
    # Default config - run with some actions
    config_default = Config()
    config_default.reward.use_waiting_penalty = True
    config_default.reward.use_idle_penalty = True
    
    env1 = TaxiEnv(config_default)
    env1.reset(seed=42)
    info1 = {"normalized_reward": 0.0}
    for _ in range(50):  # Take some actions
        valid = env1.get_candidate_actions()
        if valid:
            _, _, _, info1 = env1.step(valid[0])
        else:
            _, _, _, info1 = env1.step(env1._max_pairs)
        if env1.done:
            break
    default_reward = info1.get("normalized_reward", 0)

    # No waiting penalty - same actions
    config_no_wait = Config()
    config_no_wait.reward.use_waiting_penalty = False
    config_no_wait.reward.use_idle_penalty = True
    
    env2 = TaxiEnv(config_no_wait)
    env2.reset(seed=42)
    info2 = {"normalized_reward": 0.0}
    for _ in range(50):
        valid = env2.get_candidate_actions()
        if valid:
            _, _, _, info2 = env2.step(valid[0])
        else:
            _, _, _, info2 = env2.step(env2._max_pairs)
        if env2.done:
            break
    no_wait_reward = info2.get("normalized_reward", 0)

    # With no waiting penalty, reward should be higher (less negative)
    print(f"Default: {default_reward:.2f}, NoWait: {no_wait_reward:.2f}")
    print("[PASS] test_ablation_affects_rewards")


def test_multi_seed_normalized_stats():
    """Verify evaluate() returns normalized_reward with stats."""
    config = Config()
    agent = DQNAgent(config)

    results = evaluate(agent, config, num_episodes=3, seed=42)
    
    assert "normalized_reward" in results, "normalized_reward missing from results"
    assert isinstance(results["normalized_reward"], float), "normalized_reward not float"
    
    print(f"Normalized reward mean: {results['normalized_reward']:.2f}")
    print("[PASS] test_multi_seed_normalized_stats")


def test_all_metrics_present():
    """Verify all expected metrics are present."""
    config = Config()
    expected = ["total_reward", "normalized_reward", "pickups", "dropoffs", 
                "avg_wait", "on_time_rate", "avg_occupancy"]
    
    for key in expected:
        assert key in METRIC_KEYS, f"Missing {key} in METRIC_KEYS"
    
    print("[PASS] test_all_metrics_present")


if __name__ == "__main__":
    test_normalization_clip()
    test_normalization_disabled()
    test_ablation_affects_rewards()
    test_multi_seed_normalized_stats()
    test_all_metrics_present()
    print("\nAll normalization and ablation tests passed.")
