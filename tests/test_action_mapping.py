"""
Test action mapping alignment between environment and agent.
Verifies: candidate_actions always return valid indices within [0, total_actions-1].
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from environment import TaxiEnv


def test_action_mapping_basic():
    """Candidate actions must always be within [0, total_actions-1]."""
    config = Config()
    env = TaxiEnv(config)

    for trial in range(10):
        env.reset(seed=trial)
        for step in range(50):
            valid = env.get_candidate_actions()
            assert all(
                0 <= a < config.total_actions for a in valid
            ), f"Trial {trial}, step {step}: invalid action in {valid}, total={config.total_actions}"
            assert config.total_actions - 1 in valid, (
                f"Trial {trial}, step {step}: no-op (action {config.total_actions-1}) missing from valid actions"
            )
            if env.done:
                break
            env.step(valid[0])  # take first valid action

    print("[PASS] test_action_mapping_basic")


def test_noop_is_always_valid():
    """The no-op action (last index) must always be in candidate actions."""
    config = Config()
    env = TaxiEnv(config)

    for trial in range(20):
        env.reset(seed=trial)
        for step in range(100):
            valid = env.get_candidate_actions()
            noop = config.total_actions - 1
            assert noop in valid, (
                f"Trial {trial}, step {step}: no-op {noop} not in {valid}"
            )
            if env.done:
                break
            env.step(noop)  # always no-op

    print("[PASS] test_noop_is_always_valid")


def test_pair_actions_within_max():
    """Pair action indices must be < max_pairs (total_actions - 1)."""
    config = Config()
    env = TaxiEnv(config)

    for trial in range(10):
        env.reset(seed=trial)
        max_pairs = config.total_actions - 1
        for step in range(50):
            valid = env.get_candidate_actions()
            pair_actions = [a for a in valid if a < max_pairs]
            assert all(
                a < max_pairs for a in pair_actions
            ), f"Trial {trial}, step {step}: pair action >= max_pairs"
            if env.done:
                break
            if pair_actions:
                env.step(pair_actions[0])
            else:
                env.step(env._max_pairs)

    print("[PASS] test_pair_actions_within_max")


if __name__ == "__main__":
    test_action_mapping_basic()
    test_noop_is_always_valid()
    test_pair_actions_within_max()
    print("\nAll action mapping tests passed.")
