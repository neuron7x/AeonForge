from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"


@dataclass(frozen=True)
class State:
    eoi: float
    fatigue: float  # [0,1]
    task_complexity: float  # [0,1]


@dataclass(frozen=True)
class Action:
    task_type: TaskType
    ai_autonomy: float  # [0,1]


@dataclass(frozen=True)
class Observation:
    hrv_measurement: float
    self_reported_load: str  # low/medium/high
    task_completion_time: float


class Particle:
    def __init__(self, state: State, weight: float = 1.0):
        self.state = state
        self.weight = weight


class BeliefState:
    def __init__(self, particles: List[Particle]):
        self.particles = particles
        self._normalize()

    def _normalize(self) -> None:
        total = sum(p.weight for p in self.particles)
        if total <= 0:
            n = len(self.particles)
            if n == 0:
                return
            w = 1.0 / n
            for p in self.particles:
                p.weight = w
            return
        for p in self.particles:
            p.weight /= total

    def sample_state(self) -> State:
        weights = [p.weight for p in self.particles]
        idx = int(np.random.choice(len(self.particles), p=weights))
        return self.particles[idx].state

    def mean_eoi(self) -> float:
        return float(sum(p.state.eoi * p.weight for p in self.particles))

    def mean_fatigue(self) -> float:
        return float(sum(p.state.fatigue * p.weight for p in self.particles))


class TransitionModel:
    def __init__(self) -> None:
        self.eoi_reduction_rate = 0.15
        self.fatigue_acc_rate = 0.05
        self.noise_std = 0.1

    def sample(self, state: State, action: Action) -> State:
        eoi_reduction = self.eoi_reduction_rate * action.ai_autonomy
        new_eoi = max(0.0, state.eoi - eoi_reduction + float(np.random.normal(0, self.noise_std)))

        fatigue_inc = self.fatigue_acc_rate * (1 - action.ai_autonomy * 0.5)
        new_fatigue = min(1.0, state.fatigue + fatigue_inc + float(np.random.normal(0, self.noise_std * 0.5)))

        new_comp = float(np.clip(state.task_complexity + np.random.normal(0, 0.05), 0, 1))
        return State(eoi=new_eoi, fatigue=new_fatigue, task_complexity=new_comp)


class ObservationModel:
    def __init__(self) -> None:
        self.hrv_noise_std = 10.0

    def sample(self, state: State, action: Action) -> Observation:
        true_hrv = 100 - state.eoi * 30
        measured = float(true_hrv + np.random.normal(0, self.hrv_noise_std))

        if state.eoi < 0.5:
            sr = "low"
        elif state.eoi < 1.0:
            sr = "medium"
        elif state.eoi < 1.5:
            sr = "high" if np.random.rand() > 0.3 else "medium"
        else:
            sr = "high"

        base = 1.0
        t = base * (1 + state.eoi * 0.3 + state.fatigue * 0.2) * (1 - action.ai_autonomy * 0.4)
        return Observation(hrv_measurement=measured, self_reported_load=sr, task_completion_time=float(t))

    def probability(self, obs: Observation, state: State, action: Action) -> float:
        expected_hrv = 100 - state.eoi * 30
        hrv_prob = math.exp(-((obs.hrv_measurement - expected_hrv) / self.hrv_noise_std) ** 2)

        if state.eoi < 0.5:
            report_prob = 0.8 if obs.self_reported_load == "low" else 0.1
        elif state.eoi < 1.0:
            report_prob = {"low": 0.2, "medium": 0.7, "high": 0.1}.get(obs.self_reported_load, 0.01)
        elif state.eoi < 1.5:
            report_prob = {"low": 0.1, "medium": 0.3, "high": 0.6}.get(obs.self_reported_load, 0.01)
        else:
            report_prob = 0.9 if obs.self_reported_load == "high" else 0.05

        expected_time = 1.0 * (1 + state.eoi * 0.3 + state.fatigue * 0.2) * (1 - action.ai_autonomy * 0.4)
        time_prob = math.exp(-((obs.task_completion_time - expected_time) / 0.2) ** 2)

        return float(hrv_prob * report_prob * time_prob)


class RewardModel:
    def __init__(self, productivity_weight: float = 0.6, wellbeing_weight: float = 0.4) -> None:
        self.productivity_weight = productivity_weight
        self.wellbeing_weight = wellbeing_weight

    def reward(self, state: State, action: Action) -> float:
        productivity = action.ai_autonomy * 10
        wellbeing_penalty = -(state.eoi ** 2 + state.fatigue ** 2)
        complexity_bonus = action.ai_autonomy * state.task_complexity * 2
        return float(self.productivity_weight * (productivity + complexity_bonus) + self.wellbeing_weight * wellbeing_penalty)


class UCTNode:
    def __init__(self, belief: BeliefState, parent: Optional["UCTNode"] = None) -> None:
        self.belief = belief
        self.parent = parent
        self.children: Dict[Action, "UCTNode"] = {}
        self.visits = 0
        self.action_counts: Dict[Action, int] = {}
        self.action_values: Dict[Action, float] = {}

    def is_fully_expanded(self, actions: List[Action]) -> bool:
        return len(self.children) == len(actions)

    def best_action(self, c: float) -> Action:
        best, best_ucb = None, -float("inf")
        for a, n in self.action_counts.items():
            if n == 0:
                return a
            q = self.action_values[a] / n
            u = c * math.sqrt(math.log(self.visits + 1) / n)
            v = q + u
            if v > best_ucb:
                best_ucb, best = v, a
        # fallback if no stats
        if best is None and self.action_counts:
            best = next(iter(self.action_counts))
        return best  # type: ignore[return-value]


class POMDPPlanner:
    def __init__(self, num_particles: int = 500, num_simulations: int = 300, max_depth: int = 8, discount: float = 0.95, exploration_const: float = 50.0) -> None:
        self.num_particles = num_particles
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.discount = discount
        self.exploration_const = exploration_const

        self.transition_model = TransitionModel()
        self.observation_model = ObservationModel()
        self.reward_model = RewardModel()

        self.actions = self._create_action_space()

    def _create_action_space(self) -> List[Action]:
        actions: List[Action] = []
        for t in TaskType:
            for autonomy in [0.2, 0.5, 0.8, 1.0]:
                actions.append(Action(t, autonomy))
        return actions

    def initialize_belief(self, initial_eoi: float, initial_fatigue: float = 0.3, task_complexity: float = 0.5) -> BeliefState:
        parts = []
        for _ in range(self.num_particles):
            eoi = float(np.clip(np.random.normal(initial_eoi, 0.2), 0, 3))
            fat = float(np.clip(np.random.normal(initial_fatigue, 0.1), 0, 1))
            comp = float(np.clip(np.random.normal(task_complexity, 0.1), 0, 1))
            parts.append(Particle(State(eoi=eoi, fatigue=fat, task_complexity=comp), weight=1.0))
        return BeliefState(parts)

    def _resample(self, particles: List[Particle]) -> List[Particle]:
        weights = np.array([p.weight for p in particles], dtype=float)
        s = weights.sum()
        if s <= 0:
            n = len(particles)
            return [Particle(p.state, 1.0) for p in particles]
        weights = weights / s
        idx = np.random.choice(len(particles), size=self.num_particles, p=weights)
        return [Particle(particles[i].state, 1.0) for i in idx]

    def update_belief(self, belief: BeliefState, action: Action, observation: Observation) -> BeliefState:
        new_particles: List[Particle] = []
        for p in belief.particles:
            ns = self.transition_model.sample(p.state, action)
            like = self.observation_model.probability(observation, ns, action)
            new_particles.append(Particle(ns, p.weight * like))

        sum_w = sum(p.weight for p in new_particles)
        sum_sq = sum((p.weight ** 2) for p in new_particles)
        ess = ((sum_w ** 2) / sum_sq) if sum_sq > 0 else 0.0
        if ess < self.num_particles / 2:
            new_particles = self._resample(new_particles)
        return BeliefState(new_particles)

    def plan(self, belief: BeliefState, task_type: TaskType) -> Action:
        valid = [a for a in self.actions if a.task_type == task_type]
        root = UCTNode(belief)
        # initialize counts
        for a in valid:
            root.action_counts[a] = 0
            root.action_values[a] = 0.0
        for _ in range(self.num_simulations):
            s = belief.sample_state()
            self._simulate(root, s, 0, valid)
        # pick best by mean value
        best = max(valid, key=lambda a: (root.action_values[a] / max(1, root.action_counts[a])))
        return best

    def _simulate(self, node: UCTNode, state: State, depth: int, actions: List[Action]) -> float:
        if depth >= self.max_depth:
            return 0.0
        # expansion
        for a in actions:
            if a not in node.children:
                node.children[a] = None  # marker
        unexplored = [a for a, child in node.children.items() if child is None]
        if unexplored:
            a = unexplored[0]
        else:
            a = node.best_action(self.exploration_const)
        # rollout
        ns = self.transition_model.sample(state, a)
        r = self.reward_model.reward(state, a)
        obs = self.observation_model.sample(ns, a)
        # child
        if node.children[a] is None:
            new_belief = BeliefState([Particle(ns, 1.0) for _ in range(50)])
            node.children[a] = UCTNode(new_belief, parent=node)
            node.action_counts[a] = 0
            node.action_values[a] = 0.0
        child = node.children[a]
        fut = self._simulate(child, ns, depth + 1, actions) if isinstance(child, UCTNode) else 0.0

        total = r + self.discount * fut
        node.visits += 1
        node.action_counts[a] += 1
        node.action_values[a] += total
        return total
