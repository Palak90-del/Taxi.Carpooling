# Reproducibility Guide

## Environment Setup

```bash
pip install -r requirements.txt
```

Required packages: `torch>=2.0.0`, `numpy>=1.24.0`, `tensorboard>=2.14.0`

## Seed Control

Every command supports a `--seed` flag for reproducibility:

```bash
python main.py train --seed 42
python main.py eval --seed 42
python main.py baselines --seed 42
python main.py demo --seed 42
```

When a seed is set:
- Python `random`, NumPy, and PyTorch are seeded globally.
- The environment uses the seed for taxi positions, passenger spawning, and random decisions.
- Per-episode variation is derived as `base_seed + episode_number`.

## Quick Sanity Check (CI)

Run all unit tests and a short training loop:

```bash
python scripts/run_quick_checks.py
```

This runs 3 seeds with 20 episodes each, trains a short DQN, and verifies no crashes.

## Running Experiments

### Baselines

```bash
python main.py baselines --episodes 50 --seed 42
```

### DQN Training

```bash
python main.py train --seed 42
# Checkpoints saved to checkpoints/
# TensorBoard logs saved to runs/
```

### Multi-Seed Evaluation

```bash
python main.py multi-seed --episodes 20
```

Runs 3 seeds (42, 123, 456) and computes mean, std, 95% CI for each metric.
Saves results to `results_multi_seed.json`.

### Reward Ablation Study

```bash
python main.py ablation --episodes 30 --seed 42
```

Compares: default, neutral (no penalties), no_detour, no_idle, no_waiting, no_occupancy.

### Compare DQN vs Baselines

```bash
python main.py compare --episodes 50 --seed 42 --checkpoint checkpoints/best.pt
```

## Unit Tests

```bash
python tests/test_action_mapping.py
python tests/test_seed_reproducibility.py
python tests/test_evaluate_robustness.py
```

## Configuration

All hyperparameters are in `config.py`:

- `EnvConfig`: grid size, taxis, passengers, spawn rate, seed
- `RewardConfig`: reward weights + ablation flags (`use_*`) + normalization settings
- `DQNConfig`: learning rate, epsilon schedule, buffer size, seed
- `TrainConfig`: episodes, logging intervals, seed
- `ActionConfig`: top-k taxis/passengers for action space

### Reward Normalization

The system includes configurable reward normalization to bound extreme end-of-episode totals:

```python
# In RewardConfig:
normalize_rewards: bool = True   # Enable normalization
norm_mode: str = "clip"          # "clip", "standardize", or "none"
norm_clip_min: float = -100.0    # Clip range for clip mode
norm_clip_max: float = 100.0
```

- **clip**: Clips rewards to `[norm_clip_min, norm_clip_max]`
- **standardize**: (reserved for future use)
- **none**: No normalization applied

Normalization is applied **only at episode end** for reporting. Raw cumulative rewards are still tracked for learning.

### Ablation Study

Available ablations (via flags in RewardConfig):
- `use_occupancy_bonus`: Enable/disable occupancy utilization bonus
- `use_waiting_penalty`: Enable/disable passenger waiting time penalty
- `use_eta_violation`: Enable/disable ETA violation penalty
- `use_detour_penalty`: Enable/disable route detour penalty
- `use_idle_penalty`: Enable/disable idle taxi penalty

To run specific ablations programmatically:

```python
from config import Config

config = Config()
config.reward.use_waiting_penalty = False  # Disable waiting penalty
config.reward.use_idle_penalty = False     # Disable idle penalty
```

## Expected Outputs

After `python main.py compare --episodes 50 --seed 42`:

```
Policy               Reward  Pickups  Dropoffs  AvgWait   OnTime    Occup
---------------------------------------------------------------------------
greedy_nearest      ...     ...      ...       ...      ...      ...
fcfs                ...     ...      ...       ...      ...      ...
rule_pooling        ...     ...      ...       ...      ...      ...
random              ...     ...      ...       ...      ...      ...
eta_nearest         ...     ...      ...       ...      ...      ...
DQN (trained)       ...     ...      ...       ...      ...      ...
```

After `python main.py multi-seed --episodes 20`:

```
Policy               Reward          AvgWait          OnTime
------------------------------------------------------------------------
greedy_nearest      xxx.x±xx.x    xx.xx±x.xx    x.xxx±x.xxx
...
```
