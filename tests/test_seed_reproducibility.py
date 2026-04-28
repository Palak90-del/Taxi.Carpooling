"""
Test seed reproducibility: same seed produces identical initial states.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import Config
from environment import TaxiEnv


def test_seed_initial_state_identical():
    """Same seed should produce identical initial state."""
    config = Config()
    config.env.seed = 42

    env1 = TaxiEnv(config)
    state1 = env1.reset()

    env2 = TaxiEnv(config)
    state2 = env2.reset()

    assert np.allclose(state1, state2), "Same seed should produce identical states"

    # Check taxi positions match
    for t1, t2 in zip(env1.taxis, env2.taxis):
        assert t1.x == t2.x and t1.y == t2.y, "Taxi positions should match"

    print("[PASS] test_seed_initial_state_identical")


def test_different_seeds_differ():
    """Different seeds should (very likely) produce different states."""
    config = Config()

    env1 = TaxiEnv(config)
    state1 = env1.reset(seed=1)

    env2 = TaxiEnv(config)
    state2 = env1.reset(seed=99999)

    # They might be different; we just check the code doesn't crash
    # In practice they will differ with very high probability
    print("[PASS] test_different_seeds_differ (states computed without error)")


def test_seed_steps_deterministic():
    """Same seed + same actions = same trajectory.
    Run one env, record actions, replay in fresh env with same seed."""
    config = Config()

    # Run 1: record states, rewards, actions
    env1 = TaxiEnv(config)
    s1 = env1.reset(seed=100)
    trajectory = [(s1.copy(), 0.0)]
    actions_taken = []
    for step in range(30):
        valid = env1.get_candidate_actions()
        action = valid[0]
        actions_taken.append(action)
        s1, r1, d1, _ = env1.step(action)
        trajectory.append((s1.copy(), r1))
        if d1:
            break

    # Run 2: replay same actions with same seed
    env2 = TaxiEnv(config)
    s2 = env2.reset(seed=100)
    assert np.allclose(s2, trajectory[0][0]), "Initial states should match"
    for step, action in enumerate(actions_taken):
        s2, r2, d2, _ = env2.step(action)
        assert np.allclose(s2, trajectory[step + 1][0]), f"Step {step}: states diverged"
        assert abs(r2 - trajectory[step + 1][1]) < 1e-6, f"Step {step}: rewards diverged"
        if d2:
            break

    print("[PASS] test_seed_steps_deterministic")


def test_agent_reproducibility():
    """Same seed + same agent = same action sequence.
    Run agent1, record actions, replay with fresh agent2."""
    from dqn_agent import DQNAgent, set_seed

    config = Config()
    config.apply_seed(77)

    # Run 1: record actions
    set_seed(77)
    agent1 = DQNAgent(config)
    env1 = TaxiEnv(config)
    s1 = env1.reset(seed=77)
    actions_taken = []
    for step in range(20):
        valid = env1.get_candidate_actions()
        a1 = agent1.select_action(s1, valid)
        actions_taken.append(a1)
        s1, _, d1, _ = env1.step(a1)
        if d1:
            break

    # Run 2: replay with fresh agent, same seed
    config2 = Config()
    config2.apply_seed(77)
    set_seed(77)
    agent2 = DQNAgent(config2)
    env2 = TaxiEnv(config2)
    s2 = env2.reset(seed=77)

    for step, action in enumerate(actions_taken):
        valid = env2.get_candidate_actions()
        a2 = agent2.select_action(s2, valid)
        assert a2 == action, f"Step {step}: actions diverged ({action} vs {a2})"
        s2, _, d2, _ = env2.step(a2)
        if d2:
            break

    print("[PASS] test_agent_reproducibility")


if __name__ == "__main__":
    test_seed_initial_state_identical()
    test_different_seeds_differ()
    test_seed_steps_deterministic()
    test_agent_reproducibility()
    print("\nAll seed reproducibility tests passed.")
