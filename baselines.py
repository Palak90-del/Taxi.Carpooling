"""
Non-RL baselines for taxi dispatch.
1. Greedy Nearest Taxi
2. First-Come-First-Serve (FCFS)
3. Rule-Based Pooling
4. Random Policy
5. ETA-Nearest (assign passenger with shortest ETA to nearest taxi)
"""
from __future__ import annotations

import random
from typing import List, Tuple

from environment import TaxiEnv, Taxi, Passenger


class GreedyNearestTaxi:
    """Assign each waiting passenger to the nearest taxi with capacity."""

    def select_action(self, env: TaxiEnv) -> int:
        pairs = env._get_candidate_pairs()
        if not pairs:
            return env._max_pairs  # no-op

        best_idx, best_dist = 0, float("inf")
        for i, (t_idx, p_idx) in enumerate(pairs):
            if i >= env._max_pairs:
                break
            taxi = env.taxis[t_idx]
            p = env.passengers[p_idx]
            dist = env.manhattan(taxi.x, taxi.y, p.pickup_x, p.pickup_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx


class FirstComeFirstServe:
    """Assign the oldest waiting passenger to the first available taxi."""

    def select_action(self, env: TaxiEnv) -> int:
        pairs = env._get_candidate_pairs()
        if not pairs:
            return env._max_pairs  # no-op

        # Find passenger with earliest request time
        best_idx, earliest = 0, float("inf")
        for i, (t_idx, p_idx) in enumerate(pairs):
            if i >= env._max_pairs:
                break
            p = env.passengers[p_idx]
            if p.request_time < earliest:
                earliest = p.request_time
                best_idx = i
        return best_idx


class RuleBasedPooling:
    """
    Pool nearby passengers and assign to taxis that minimise combined travel.
    Simple heuristic: for each taxi, pick the passenger that minimises
    (distance_to_pickup + detour_cost) and assign the best overall pair.
    """

    def select_action(self, env: TaxiEnv) -> int:
        pairs = env._get_candidate_pairs()
        if not pairs:
            return env._max_pairs  # no-op

        best_idx, best_cost = 0, float("inf")
        for i, (t_idx, p_idx) in enumerate(pairs):
            if i >= env._max_pairs:
                break
            taxi = env.taxis[t_idx]
            p = env.passengers[p_idx]

            dist_to_pickup = env.manhattan(taxi.x, taxi.y, p.pickup_x, p.pickup_y)
            dist_pickup_to_drop = env.manhattan(
                p.pickup_x, p.pickup_y, p.drop_x, p.drop_y
            )

            # Detour: how much extra distance if inserted into current route
            if taxi.route:
                existing_cost = env._route_cost(taxi.x, taxi.y, taxi.route)
                new_route = env._insert_route(taxi, p)
                new_cost = env._route_cost(taxi.x, taxi.y, new_route)
                detour = max(new_cost - existing_cost, 0)
            else:
                detour = 0

            # Weighted cost: penalise long waits
            wait = env.timestep - p.request_time
            cost = dist_to_pickup + detour + 0.5 * wait
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        return best_idx


class RandomPolicy:
    """Randomly pick one valid (taxi, passenger) pair each step."""

    def select_action(self, env: TaxiEnv) -> int:
        pairs = env._get_candidate_pairs()
        if not pairs:
            return env._max_pairs
        return random.randint(0, min(len(pairs) - 1, env._max_pairs - 1))


class ETANearest:
    """
    Assign the passenger with the shortest direct ETA (pickup-to-drop distance)
    to the nearest available taxi.
    """

    def select_action(self, env: TaxiEnv) -> int:
        pairs = env._get_candidate_pairs()
        if not pairs:
            return env._max_pairs

        best_idx, best_score = 0, float("inf")
        for i, (t_idx, p_idx) in enumerate(pairs):
            if i >= env._max_pairs:
                break
            taxi = env.taxis[t_idx]
            p = env.passengers[p_idx]

            dist_to_pickup = env.manhattan(taxi.x, taxi.y, p.pickup_x, p.pickup_y)
            eta = p.direct_eta
            # Score: how quickly can this passenger be served
            score = dist_to_pickup + eta
            if score < best_score:
                best_score = score
                best_idx = i
        return best_idx


BASELINES = {
    "greedy_nearest": GreedyNearestTaxi,
    "fcfs": FirstComeFirstServe,
    "rule_pooling": RuleBasedPooling,
    "random": RandomPolicy,
    "eta_nearest": ETANearest,
}
