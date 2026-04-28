#!/usr/bin/env python3
"""
Quick sanity check: run 3 seeds with 20 episodes each.
Verifies all commands work end-to-end without crashes.
"""
import subprocess
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

SEEDS = [42, 123, 456]
EPISODES = 20


def run(cmd: str, desc: str) -> bool:
    """Run a command and report pass/fail."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd, shell=True, cwd=PROJECT_DIR,
        capture_output=False, text=True,
    )
    if result.returncode != 0:
        print(f"[FAIL] {desc} (exit code {result.returncode})")
        return False
    print(f"[PASS] {desc}")
    return True


def main() -> None:
    all_passed = True

    # 1. Baselines with seed
    for seed in SEEDS:
        ok = run(
            f"{PYTHON} main.py baselines --episodes {EPISODES} --seed {seed}",
            f"Baselines (seed={seed})",
        )
        all_passed = all_passed and ok

    # 2. Train short episode
    ok = run(
        f"{PYTHON} main.py train --seed 42",
        "Short DQN training (default episodes with seed)",
    )
    all_passed = all_passed and ok

    # 3. Evaluate
    ok = run(
        f"{PYTHON} main.py eval --seed 42",
        "DQN evaluation",
    )
    all_passed = all_passed and ok

    # 4. Compare
    ok = run(
        f"{PYTHON} main.py compare --episodes {EPISODES} --seed 42",
        "DQN vs baselines comparison",
    )
    all_passed = all_passed and ok

    # 5. Tests
    for test_file in [
        "tests/test_action_mapping.py",
        "tests/test_seed_reproducibility.py",
        "tests/test_evaluate_robustness.py",
    ]:
        ok = run(f"{PYTHON} {test_file}", f"Unit test: {test_file}")
        all_passed = all_passed and ok

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
