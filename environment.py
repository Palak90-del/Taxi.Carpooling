"""
Taxi Dispatch Environment.
20x20 grid, multiple taxis, random passengers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import Config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Passenger:
    """A ride request."""
    id: int
    pickup_x: int
    pickup_y: int
    drop_x: int
    drop_y: int
    request_time: int
    delay_tolerance: int
    picked_up: bool = False
    completed: bool = False
    dropped_off: bool = False
    assigned_taxi: Optional[int] = None

    @property
    def direct_eta(self) -> int:
        return (
            abs(self.pickup_x - self.drop_x)
            + abs(self.pickup_y - self.drop_y)
        )

    @property
    def status(self) -> str:
        if self.dropped_off:
            return "completed"
        if self.picked_up:
            return "onboard"
        return "waiting"


@dataclass
class Taxi:
    """A taxi with a route."""
    id: int
    x: int
    y: int
    capacity: int
    occupancy: int = 0
    on_route: bool = False
    route: List[Tuple[int, int, str]] = field(default_factory=list)
    total_distance: int = 0

    @property
    def has_capacity(self) -> bool:
        return self.occupancy < self.capacity


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class TaxiEnv:
    """Grid-based taxi dispatch environment."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.e = config.env
        self.r = config.reward.effective()  # apply ablation flags
        self.grid_size: int = self.e.grid_size
        self._next_passenger_id: int = 0

        # State
        self.taxis: List[Taxi] = []
        self.passengers: List[Passenger] = []
        self.timestep: int = 0
        self.done: bool = False
        self._max_pairs: int = config.total_actions - 1

        # Metrics (per episode)
        self.metrics: dict = {}

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def _random_pos(self) -> Tuple[int, int]:
        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        )

    def _spawn_passenger(self) -> Passenger:
        p_x, p_y = self._random_pos()
        d_x, d_y = self._random_pos()
        while d_x == p_x and d_y == p_y:
            d_x, d_y = self._random_pos()
        direct = self.manhattan(p_x, p_y, d_x, d_y)
        tol = min(
            int(self.e.alpha_direct_eta * direct),
            self.e.max_delay_tolerance,
        )
        p = Passenger(
            id=self._next_passenger_id,
            pickup_x=p_x,
            pickup_y=p_y,
            drop_x=d_x,
            drop_y=d_y,
            request_time=self.timestep,
            delay_tolerance=tol,
        )
        self._next_passenger_id += 1
        return p

    # ---- greedy insertion -------------------------------------------------
    def _insert_route(
        self, taxi: Taxi, passenger: Passenger
    ) -> List[Tuple[int, int, str]]:
        """Try all pickup/dropoff insert positions, return best route."""
        px, py = passenger.pickup_x, passenger.pickup_y
        dx, dy = passenger.drop_x, passenger.drop_y
        existing = list(taxi.route)

        best_route: List[Tuple[int, int, str]] = []
        best_cost = float("inf")

        n = len(existing)
        for pi in range(n + 1):
            for di in range(pi + 1, n + 2):
                candidate = (
                    existing[:pi]
                    + [(px, py, "pickup")]
                    + existing[pi : di - 1]
                    + [(dx, dy, "dropoff")]
                    + existing[di - 1 :]
                )
                cost = self._route_cost(taxi.x, taxi.y, candidate)
                if cost < best_cost:
                    best_cost = cost
                    best_route = candidate
        return best_route

    def _route_cost(
        self, sx: int, sy: int, route: List[Tuple[int, int, str]]
    ) -> int:
        cost = 0
        cx, cy = sx, sy
        for rx, ry, _ in route:
            cost += self.manhattan(cx, cy, rx, ry)
            cx, cy = rx, ry
        return cost

    # ---- movement ---------------------------------------------------------
    def _move_taxis(self) -> None:
        for taxi in self.taxis:
            if not taxi.route:
                taxi.on_route = False
                continue
            tx, ty, _ = taxi.route[0]
            if taxi.x == tx and taxi.y == ty:
                continue  # already there, will be processed
            # move one step Manhattan
            if taxi.x < tx:
                taxi.x += 1
            elif taxi.x > tx:
                taxi.x -= 1
            elif taxi.y < ty:
                taxi.y += 1
            elif taxi.y > ty:
                taxi.y -= 1
            taxi.total_distance += 1

    def _process_arrivals(self, reward: float) -> float:
        """Handle pickup / dropoff when taxi reaches waypoint."""
        for taxi in self.taxis:
            if not taxi.route:
                continue
            wx, wy, wtype = taxi.route[0]
            if taxi.x == wx and taxi.y == wy:
                taxi.route.pop(0)
                if not taxi.route:
                    taxi.on_route = False
                if wtype == "pickup":
                    for p in self.passengers:
                        if (
                            p.pickup_x == wx
                            and p.pickup_y == wy
                            and not p.picked_up
                            and not p.dropped_off
                            and p.assigned_taxi == taxi.id
                        ):
                            p.picked_up = True
                            taxi.occupancy += 1
                            wait = self.timestep - p.request_time
                            self.metrics["pickups"] += 1
                            self.metrics["total_wait"] += wait
                            reward += self.r.pickup_success
                            break
                elif wtype == "dropoff":
                    for p in self.passengers:
                        if (
                            p.drop_x == wx
                            and p.drop_y == wy
                            and p.picked_up
                            and not p.dropped_off
                        ):
                            p.dropped_off = True
                            p.completed = True
                            taxi.occupancy -= 1
                            self.metrics["dropoffs"] += 1
                            reward += self.r.dropoff_success
                            break
        return reward

    # ---- reward -----------------------------------------------------------
    def _compute_reward(self) -> float:
        reward = 0.0
        # occupancy utilisation
        total_occ = sum(t.occupancy for t in self.taxis)
        total_cap = sum(t.capacity for t in self.taxis)
        if total_cap > 0:
            reward += self.r.occupancy_bonus * (total_occ / total_cap)
        # waiting penalty
        for p in self.passengers:
            if p.status == "waiting":
                wait = self.timestep - p.request_time
                reward += self.r.waiting_time_penalty
                if wait > p.delay_tolerance:
                    reward += self.r.eta_violation_penalty
                    self.metrics["eta_violations"] += 1
        # idle penalty
        for taxi in self.taxis:
            if not taxi.on_route and taxi.occupancy == 0:
                reward += self.r.idle_penalty
        # detour penalty (route length vs direct)
        for taxi in self.taxis:
            if taxi.route:
                direct = self.manhattan(
                    taxi.x, taxi.y, taxi.route[0][0], taxi.route[0][1]
                )
                route_len = self._route_cost(taxi.x, taxi.y, taxi.route)
                if route_len > direct * 1.5:
                    reward += self.r.detour_penalty
        return reward

    # ---- public API -------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment, return initial state.
        Optionally seed RNG for reproducibility."""
        # Seed handling
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        elif self.e.seed is not None:
            random.seed(self.e.seed)
            np.random.seed(self.e.seed)

        self.taxis = [
            Taxi(
                id=i,
                x=random.randint(0, self.grid_size - 1),
                y=random.randint(0, self.grid_size - 1),
                capacity=self.e.taxi_capacity,
            )
            for i in range(self.e.num_taxis)
        ]
        self.passengers = []
        self._next_passenger_id = 0
        self.timestep = 0
        self.done = False
        self.metrics = {
            "pickups": 0,
            "dropoffs": 0,
            "total_wait": 0,
            "eta_violations": 0,
            "total_reward": 0.0,
        }
        self._max_pairs = self.cfg.total_actions - 1  # reserve last for no-op
        # spawn initial passengers
        for _ in range(random.randint(1, 3)):
            self.passengers.append(self._spawn_passenger())
        return self.get_state()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one timestep.
        action:
          - 0 .. len(pairs)-1 : (taxi, passenger) pair index
          - _max_pairs        : no-op (always equals total_actions - 1)
        Returns: (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        reward = 0.0

        # --- decode & apply action ---
        taxi_pass = self._get_candidate_pairs()
        noop_action = self._max_pairs  # == total_actions - 1

        if action < len(taxi_pass) and action < noop_action:
            t_idx, p_idx = taxi_pass[action]
            taxi = self.taxis[t_idx]
            passenger = self.passengers[p_idx]
            if (
                taxi.has_capacity
                and not passenger.picked_up
                and not passenger.dropped_off
                and passenger.assigned_taxi is None
            ):
                passenger.assigned_taxi = taxi.id
                taxi.route = self._insert_route(taxi, passenger)
                taxi.on_route = True
            else:
                reward += self.r.invalid_action_penalty
        # else: no-op

        # --- advance time ---
        self._move_taxis()
        reward += self._process_arrivals(0.0)
        self.timestep += 1

        # --- spawn passengers ---
        if random.random() < self.e.passenger_spawn_prob:
            self.passengers.append(self._spawn_passenger())

        # --- timeout check ---
        for p in self.passengers:
            if p.status == "waiting":
                if self.timestep - p.request_time > p.delay_tolerance:
                    p.completed = True  # timed out

        # --- reward ---
        reward += self._compute_reward()
        self.metrics["total_reward"] += reward

        # --- done ---
        self.done = self.timestep >= self.e.max_steps_per_episode

        # Apply normalization only at end of episode for reporting
        if self.done:
            raw_total_reward = self.metrics["total_reward"]
            normalized_reward = self.r.normalize_episode_reward(raw_total_reward)
            self.metrics["normalized_reward"] = normalized_reward
        else:
            # Not done yet - normalized reward is same as raw for reporting
            self.metrics["normalized_reward"] = self.metrics["total_reward"]

        # Always return info (even mid-episode) for robust metrics capture
        total = len(self.passengers) if self.passengers else 1
        info = {
            "pickups": self.metrics["pickups"],
            "dropoffs": self.metrics["dropoffs"],
            "avg_wait": (
                self.metrics["total_wait"] / max(self.metrics["pickups"], 1)
            ),
            "on_time_rate": self.metrics["dropoffs"] / max(total, 1),
            "avg_occupancy": sum(t.occupancy for t in self.taxis)
            / max(len(self.taxis), 1),
            "total_reward": self.metrics["total_reward"],
            "normalized_reward": self.metrics["normalized_reward"],
        }

        return self.get_state(), reward, self.done, info

    # ---- state / action helpers -------------------------------------------
    def _waiting_passengers(self) -> List[int]:
        """Indices of passengers that are still waiting."""
        return [
            i
            for i, p in enumerate(self.passengers)
            if p.status == "waiting" and p.assigned_taxi is None
        ]

    def _get_candidate_pairs(self) -> List[Tuple[int, int]]:
        """Get valid (taxi_idx, passenger_idx) pairs."""
        pairs: List[Tuple[int, int]] = []
        waiting = self._waiting_passengers()
        for t_idx, taxi in enumerate(self.taxis):
            if not taxi.has_capacity:
                continue
            for p_idx in waiting:
                pairs.append((t_idx, p_idx))
        return pairs

    def get_candidate_actions(self) -> List[int]:
        """Return indices of valid actions (for masking).
        Pairs occupy 0..min(len(pairs), max_pairs)-1, no-op is max_pairs."""
        pairs = self._get_candidate_pairs()
        n = min(len(pairs), self._max_pairs)
        return list(range(n)) + [self._max_pairs]

    def get_state(self) -> np.ndarray:
        """Flat state vector (fixed size)."""
        parts: List[float] = []
        # Taxi features
        for taxi in self.taxis:
            parts.extend([
                taxi.x / self.grid_size,
                taxi.y / self.grid_size,
                taxi.occupancy / max(taxi.capacity, 1),
                1.0 if taxi.on_route else 0.0,
                len(taxi.route) / 10.0,
            ])
        # Passenger features (padded to max_passengers_waiting)
        waiting = self._waiting_passengers()
        sorted_pass = sorted(
            waiting,
            key=lambda i: self.manhattan(
                self.taxis[0].x if self.taxis else 0,
                self.taxis[0].y if self.taxis else 0,
                self.passengers[i].pickup_x,
                self.passengers[i].pickup_y,
            ),
        )
        for i in range(self.e.max_passengers_waiting):
            if i < len(sorted_pass):
                p = self.passengers[sorted_pass[i]]
                parts.extend([
                    p.pickup_x / self.grid_size,
                    p.pickup_y / self.grid_size,
                    p.drop_x / self.grid_size,
                    p.drop_y / self.grid_size,
                    min((self.timestep - p.request_time) / 50.0, 1.0),
                    min(p.delay_tolerance / 50.0, 1.0),
                ])
            else:
                parts.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Global features
        parts.append(self.timestep / self.e.max_steps_per_episode)
        return np.array(parts, dtype=np.float32)
