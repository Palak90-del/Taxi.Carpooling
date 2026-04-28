"""
Configuration for RL Taxi Dispatch System.
All hyperparameters and settings in one place.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Environment settings."""
    grid_size: int = 20
    num_taxis: int = 5
    taxi_capacity: int = 2
    max_passengers_waiting: int = 10
    max_timesteps: int = 500
    passenger_spawn_prob: float = 0.3
    alpha_direct_eta: float = 2.0
    max_delay_tolerance: int = 30
    max_steps_per_episode: int = 1000
    seed: Optional[int] = None


@dataclass
class RewardConfig:
    """Reward weights - all configurable."""
    pickup_success: float = 5.0
    dropoff_success: float = 10.0
    occupancy_bonus: float = 0.2
    waiting_time_penalty: float = -0.02
    eta_violation_penalty: float = -2.0
    detour_penalty: float = -0.01
    idle_penalty: float = -0.01
    no_op_penalty: float = -0.005
    invalid_action_penalty: float = -0.5

    # Ablation flags: set to False to zero-out that reward component
    use_occupancy_bonus: bool = True
    use_waiting_penalty: bool = True
    use_eta_violation: bool = True
    use_detour_penalty: bool = True
    use_idle_penalty: bool = True

    # Reward normalization settings
    normalize_rewards: bool = True  # Apply normalization to end-of-episode rewards
    norm_mode: str = "clip"  # "clip", "standardize", or "none"
    norm_clip_min: float = -100.0  # Clip range for clip mode
    norm_clip_max: float = 100.0

    def effective(self) -> "RewardConfig":
        """Return a copy with ablation flags applied (zeroed where disabled)."""
        r = RewardConfig(
            pickup_success=self.pickup_success,
            dropoff_success=self.dropoff_success,
            occupancy_bonus=self.occupancy_bonus,
            waiting_time_penalty=self.waiting_time_penalty,
            eta_violation_penalty=self.eta_violation_penalty,
            detour_penalty=self.detour_penalty,
            idle_penalty=self.idle_penalty,
            no_op_penalty=self.no_op_penalty,
            invalid_action_penalty=self.invalid_action_penalty,
            use_occupancy_bonus=self.use_occupancy_bonus,
            use_waiting_penalty=self.use_waiting_penalty,
            use_eta_violation=self.use_eta_violation,
            use_detour_penalty=self.use_detour_penalty,
            use_idle_penalty=self.use_idle_penalty,
            normalize_rewards=self.normalize_rewards,
            norm_mode=self.norm_mode,
            norm_clip_min=self.norm_clip_min,
            norm_clip_max=self.norm_clip_max,
        )
        if not self.use_occupancy_bonus:
            r.occupancy_bonus = 0.0
        if not self.use_waiting_penalty:
            r.waiting_time_penalty = 0.0
        if not self.use_eta_violation:
            r.eta_violation_penalty = 0.0
        if not self.use_detour_penalty:
            r.detour_penalty = 0.0
        if not self.use_idle_penalty:
            r.idle_penalty = 0.0
        return r

    def normalize_episode_reward(self, total_reward: float) -> float:
        """Normalize end-of-episode total reward."""
        if not self.normalize_rewards or self.norm_mode == "none":
            return total_reward
        
        if self.norm_mode == "clip":
            return max(self.norm_clip_min, min(self.norm_clip_max, total_reward))
        
        return total_reward


@dataclass
class DQNConfig:
    """DQN agent settings."""
    state_dim: int = 0
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.9999
    target_update_freq: int = 10
    batch_size: int = 64
    buffer_size: int = 50000
    min_buffer_size: int = 500
    seed: Optional[int] = None


@dataclass
class TrainConfig:
    """Training pipeline settings."""
    num_episodes: int = 2000
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 10
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    seed: Optional[int] = None


@dataclass
class ActionConfig:
    """Action space settings."""
    top_k_passengers: int = 5
    top_k_taxis: int = 3
    include_noop: bool = True


@dataclass
class Config:
    """Master config combining all sub-configs."""
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    def __post_init__(self):
        # Actions = top_k_taxis * top_k_passengers + 1 (no-op)
        self.total_actions = (
            self.action.top_k_taxis * self.action.top_k_passengers
            + (1 if self.action.include_noop else 0)
        )
        # State dim: taxis (x,y,cap,occ,on_route) + passengers (px,py,dx,dy,wait,ttl) + global (timestep)
        taxi_state = self.env.num_taxis * 5
        passenger_state = self.env.max_passengers_waiting * 6
        global_state = 1
        self.dqn.state_dim = taxi_state + passenger_state + global_state

    def apply_seed(self, seed: int) -> None:
        """Propagate a seed to all sub-configs."""
        self.env.seed = seed
        self.dqn.seed = seed
        self.train.seed = seed
