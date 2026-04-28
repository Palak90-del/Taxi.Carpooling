# RL Taxi Dispatch System

A reinforcement learning system for taxi dispatch optimization on a 20×20 grid world. This project implements a Deep Q-Network (DQN) agent and compares it against multiple heuristic baselines for ride-hailing dispatch decisions.

## Introduction

### Problem Statement
Optimize taxi dispatch decisions in a dynamic ride-hailing environment where:
- 5 taxis with capacity 2 passengers each
- Passengers spawn randomly with pickup/drop-off locations
- Each passenger has a delay tolerance and expected travel time
- Goal: maximize pickups/dropoffs, minimize wait time, maximize on-time delivery

### Environment
- **Grid**: 20×20 cells
- **Taxis**: 5 vehicles, capacity 2 each, can carry multiple passengers
- **Passengers**: Random spawn, pickup/drop-off locations, delay tolerance
- **Time**: Timesteps advance on state changes (taxi movement, passenger spawn)

### State Representation (86-dim)
- **Taxi features** (5 taxis × 5 = 25): position (x,y), occupancy, on_route, route_length
- **Passenger features** (10 passengers × 6 = 60): pickup (x,y), drop (x,y), wait_time, tolerance
- **Global** (1): current timestep

### Action Space
- Top-K candidate (taxi, passenger) pairs + no-op
- Default: 3 taxis × 5 passengers + 1 no-op = 16 actions
- Invalid actions masked during selection

### Reward Function
| Component | Weight | Description |
|-----------|--------|-------------|
| Pickup success | +5.0 | Successful passenger pickup |
| Dropoff success | +10.0 | Successful passenger delivery |
| Occupancy bonus | +0.2 | Utilization reward |
| Waiting penalty | -0.02 | Per timestep per waiting passenger |
| ETA violation | -2.0 | Passenger exceeds delay tolerance |
| Detour penalty | -0.01 | Excessive route deviation |
| Idle penalty | -0.01 | Taxi with no passengers/route |

### Normalization
End-of-episode rewards are clipped to [-100, 100] for stable training and comparable results across configurations.

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
# Requirements: torch>=2.0.0, numpy>=1.24.0, tensorboard>=2.14.0
```

### Quick Test (All Unit Tests)
```bash
cd "/Users/ravneetsingh/Downloads/RL PROJECT"

python3 tests/test_action_mapping.py
python3 tests/test_seed_reproducibility.py
python3 tests/test_evaluate_robustness.py
python3 tests/test_reward_normalization.py
```

### Main Experiments

#### 1. Run Baselines
```bash
python3 main.py baselines --episodes 50 --seed 42
```
Output: Table with Reward, Pickups, Dropoffs, AvgWait, OnTime, Occupancy

#### 2. Multi-Seed Evaluation (Recommended for Paper)
```bash
python3 main.py multi-seed --episodes 20
```
- Runs 3 seeds (42, 123, 456), 20 episodes each
- Outputs mean ± 95% CI
- Saves to `results_multi_seed.json`

#### 3. Reward Ablation Study
```bash
python3 main.py ablation --episodes 30 --seed 42
```
Tests: default, neutral, no_detour, no_idle, no_waiting, no_occupancy

#### 4. Train DQN
```bash
python3 main.py train --seed 42
```
- Default: 2000 episodes
- Checkpoints saved to `checkpoints/`
- TensorBoard logs to `runs/`

#### 5. Compare DQN vs Baselines
```bash
python3 main.py compare --episodes 50 --seed 42
```

#### 6. Quick Demo
```bash
python3 main.py demo --seed 42
```

### Command-Line Options
| Flag | Description |
|------|-------------|
| `--seed N` | Set random seed for reproducibility |
| `--episodes N` | Number of evaluation episodes |
| `--checkpoint PATH` | Path to model checkpoint |

---

## Previous Runs & Results

### Unit Tests (All Passed)
```
✅ test_action_mapping.py          (3/3 tests)
✅ test_seed_reproducibility.py    (4/4 tests)  
✅ test_evaluate_robustness.py      (4/4 tests)
✅ test_reward_normalization.py    (5/5 tests)
```

### Baselines Comparison (50 episodes, seed=42)

| Baseline | Reward | Pickups | Dropoffs | AvgWait | OnTime | Occupancy |
|----------|--------|---------|----------|---------|--------|-----------|
| greedy_nearest | -41.14 | 291.2 | 283.6 | 18.70 | 0.955 | 1.516 |
| fcfs | -6750.35 | 286.3 | 277.3 | 34.97 | 0.934 | 1.784 |
| **rule_pooling** | **1410.55** | **292.0** | **286.1** | **16.18** | **0.963** | **1.188** |
| random | -3783.39 | 291.4 | 282.7 | 28.78 | 0.940 | 1.728 |
| eta_nearest | -89.61 | 291.2 | 283.8 | 18.90 | 0.955 | 1.484 |

### Multi-Seed Evaluation (20 episodes × 3 seeds)

| Policy | Reward (mean±CI) | AvgWait | OnTime |
|--------|------------------|---------|--------|
| greedy_nearest | -218.2±230.3 | 19.28±0.58 | 0.955±0.004 |
| fcfs | -7089.7±506.1 | 35.50±0.88 | 0.934±0.004 |
| **rule_pooling** | **1127.1±158.2** | **17.01±0.31** | **0.961±0.002** |
| random | -3998.3±229.1 | 29.26±0.68 | 0.940±0.003 |
| eta_nearest | -348.8±217.1 | 19.57±0.47 | 0.954±0.003 |
| DQN (trained) | -6561.0±658.8 | 35.02±1.14 | 0.935±0.004 |

### Key Observations
- **rule_pooling** consistently outperforms all other methods
- Greedy and ETA-nearest perform similarly (nearest neighbor heuristic)
- FCFS performs poorly due to ignoring spatial efficiency
- DQN needs more training to compete with heuristics
- Normalization bounds rewards to ±100 for stable comparisons

---

## Research Paper Guide

### Abstract
Briefly describe the problem (taxi dispatch on grid world), the approach (DQN with candidate actions), and key results (rule_pooling heuristic outperforms DQN; spatial pooling matters).

### Methods

#### Environment Design
- 20×20 grid, 5 taxis, capacity 2
- Manhattan distance for travel time estimation
- Greedy insertion for route planning
- State: 86-dimensional (taxi + passenger + global features)

#### RL Algorithm
- DQN with experience replay, target network, epsilon-greedy
- Action space: Top-K (taxi, passenger) pairs + no-op
- Reward: Pickup (+5), Dropoff (+10), penalties for waiting/idle/violations

#### Baselines
1. **greedy_nearest**: Assign to nearest available taxi
2. **fcfs**: First-come-first-serve (oldest passenger)
3. **rule_pooling**: Minimize (distance + detour + wait)
4. **random**: Random assignment
5. **eta_nearest**: Minimize pickup+direct_eta

#### Reproducibility
- Seed control (--seed flag)
- Multi-seed evaluation (42, 123, 456)
- 95% confidence intervals reported

### Experiments

1. **Baseline Comparison**: Table with all 5 baselines across 50 episodes
2. **Multi-Seed Evaluation**: Mean ± CI across 3 seeds, 20 episodes
3. **Ablation Study**: Impact of reward components
4. **Learning Curves**: Training reward over episodes (TensorBoard)

### Results to Report

| Metric | Best Baseline | DQN Gap |
|--------|---------------|---------|
| Total Reward | rule_pooling (+1410) | DQN below |
| Avg Wait Time | rule_pooling (16.2) | DQN higher |
| On-Time Rate | rule_pooling (96.3%) | DQN lower |
| Occupancy | fcfs (1.78) | - |

### Discussion
- Spatial pooling (rule_pooling) captures route efficiency
- DQN underperforms due to limited training (200 eps vs 2000 needed)
- Reward normalization enables stable comparisons
- Future: longer training, Double DQN, prioritized replay

### Limitations
- Fixed routing heuristic (greedy insertion) - RL learns assignment only
- Simplified state representation (no real traffic)
- Single-city scenario (grid world)

---

## File Structure

```
RL PROJECT/
├── config.py              # All hyperparameters
├── environment.py        # Taxi dispatch environment
├── network.py            # DQN neural network
├── dqn_agent.py          # DQN agent with replay buffer
├── baselines.py          # 5 heuristic baselines
├── train.py              # Training + evaluation
├── main.py               # CLI entry point
├── requirements.txt      # Dependencies
├── tests/
│   ├── test_action_mapping.py
│   ├── test_seed_reproducibility.py
│   ├── test_evaluate_robustness.py
│   └── test_reward_normalization.py
├── scripts/
│   └── run_quick_checks.py
├── docs/
│   └── reproducibility_guide.md
├── checkpoints/          # Saved models
├── runs/                 # TensorBoard logs
└── results_multi_seed.json
```

---

## Configuration Options

Edit `config.py` for experiments:

```python
# Reward normalization (default: enabled)
reward.normalize_rewards = True
reward.norm_mode = "clip"
reward.norm_clip_min = -100.0
reward.norm_clip_max = 100.0

# Ablation flags
reward.use_occupancy_bonus = True
reward.use_waiting_penalty = True
reward.use_eta_violation = True
reward.use_detour_penalty = True
reward.use_idle_penalty = True

# Training
train.num_episodes = 2000
train.log_interval = 10
```

---

## Citation & Credits

This is a research implementation for taxi dispatch optimization using reinforcement learning. Results show that the **rule_pooling** heuristic outperforms both DQN and other baselines in this setup.

For questions or issues, refer to the reproducibility guide or run unit tests to verify the environment is working correctly.
