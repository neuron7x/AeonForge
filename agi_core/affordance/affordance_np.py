"""Affordance modelling utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

State = NDArray[np.float64]
Goal = Optional[NDArray[np.float64]]


@dataclass(frozen=True)
class Affordance:
    """Container representing a single affordance hypothesis."""

    object_id: str
    state: Tuple[float, ...]
    goal: Tuple[float, ...]
    action: int
    probability: float

    def __post_init__(self) -> None:
        if self.action < 0:  # pragma: no cover - defensive validation
            raise ValueError("action index must be non-negative")
        if not (0.0 <= self.probability <= 1.0):  # pragma: no cover - defensive validation
            raise ValueError("probability must lie inside [0, 1]")


class CompositionalAffordanceMap:
    """Logistic regression based affordance estimator.

    The estimator predicts the success probability of executing a discrete
    action given the current state and an optional goal descriptor.  The model
    is intentionally lightweight and can be trained using iteratively weighted
    least squares (Newton updates).  The class additionally exposes calibration
    diagnostics and simple composition rules for reasoning about conjunctions
    or disjunctions of affordances.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        ridge_lambda: float = 1e-2,
        *,
        goal_dim: int = 0,
        calibration_bins: int = 10,
        seed: int = 0,
    ) -> None:
        if obs_dim <= 0:  # pragma: no cover - defensive validation
            raise ValueError("obs_dim must be positive")
        if num_actions <= 0:  # pragma: no cover - defensive validation
            raise ValueError("num_actions must be positive")
        if ridge_lambda < 0:  # pragma: no cover - defensive validation
            raise ValueError("ridge_lambda must be non-negative")
        if goal_dim < 0:  # pragma: no cover - defensive validation
            raise ValueError("goal_dim must be non-negative")
        if calibration_bins <= 0:  # pragma: no cover - defensive validation
            raise ValueError("calibration_bins must be positive")

        self.D_obs = int(obs_dim)
        self.D_goal = int(goal_dim)
        self.A = int(num_actions)
        self.D_total = self.D_obs + self.D_goal + self.A + 1
        self.lam = float(ridge_lambda)
        self.calibration_bins = int(calibration_bins)

        self.W = np.zeros((self.D_total,), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        self._affordance_db: Dict[Affordance, List[float]] = {}
        self._predicted: List[float] = []
        self._actual: List[int] = []
        self.n_samples_seen = 0
        self.last_train_loss = float("inf")

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------
    def _design_vector(self, state: State, goal: Goal, action: int) -> NDArray[np.float64]:
        goal_vec = np.zeros(self.D_goal, dtype=np.float64)
        if goal is not None and self.D_goal > 0:  # pragma: no cover - optional goal path
            goal_vec = np.asarray(goal, dtype=np.float64).reshape(self.D_goal)
        state_vec = np.asarray(state, dtype=np.float64).reshape(self.D_obs)
        one_hot = np.zeros(self.A, dtype=np.float64)
        one_hot[action % self.A] = 1.0
        return np.concatenate([state_vec, goal_vec, one_hot, [1.0]])

    def _design_matrix(
        self,
        states: State,
        goals: Optional[State],
        actions: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        goals_array = goals
        if self.D_goal > 0:  # pragma: no cover - optional goal path
            if goals_array is None:
                raise ValueError("goal descriptors required but missing")
            goals_array = np.asarray(goals_array, dtype=np.float64)
            if goals_array.shape != (states.shape[0], self.D_goal):
                raise ValueError("goal array has wrong shape")
        else:
            goals_array = np.zeros((states.shape[0], 0), dtype=np.float64)

        state_arr = np.asarray(states, dtype=np.float64)
        if state_arr.shape != (states.shape[0], self.D_obs):  # pragma: no cover - defensive validation
            raise ValueError("state array has wrong shape")
        one_hot = np.zeros((states.shape[0], self.A), dtype=np.float64)
        one_hot[np.arange(states.shape[0]), actions % self.A] = 1.0
        bias = np.ones((states.shape[0], 1), dtype=np.float64)
        return np.concatenate([state_arr, goals_array, one_hot, bias], axis=1)

    # ------------------------------------------------------------------
    # Training and scoring
    # ------------------------------------------------------------------
    def score_single(self, state: State, goal: Goal = None) -> NDArray[np.float64]:
        scores = np.zeros(self.A, dtype=np.float64)
        for action in range(self.A):
            phi = self._design_vector(state, goal, action)
            scores[action] = 1.0 / (1.0 + np.exp(-float(phi @ self.W)))
        return scores

    def feasible_actions(
        self,
        state: State,
        goal: Goal = None,
        *,
        threshold: float = 0.5,
        thr: Optional[float] = None,
    ) -> NDArray[np.int64]:
        limit = threshold if thr is None else float(thr)
        scores = self.score_single(state, goal)
        return np.where(scores >= limit)[0]

    def fit(  # pragma: no cover - training path
        self,
        states: State,
        actions: NDArray[np.int64],
        successes: NDArray[np.float64],
        *,
        goals: Optional[State] = None,
        min_samples: int = 64,
        min_success_rate: float = 0.3,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> bool:
        if states.shape[0] != actions.shape[0] or states.shape[0] != successes.shape[0]:  # pragma: no cover - defensive validation
            raise ValueError("batch sizes must agree")
        if states.shape[1] != self.D_obs:  # pragma: no cover - defensive validation
            raise ValueError("state dimensionality mismatch")
        if not np.isfinite(states).all():  # pragma: no cover - defensive validation
            raise ValueError("state array must be finite")
        if not np.isfinite(successes).all():  # pragma: no cover - defensive validation
            raise ValueError("success array must be finite")
        if not ((successes == 0) | (successes == 1)).all():  # pragma: no cover - defensive validation
            raise ValueError("success labels must be binary")
        if states.shape[0] < min_samples:
            warnings.warn("insufficient data to train affordance model", stacklevel=2)  # pragma: no cover - warning path
            return False
        success_rate = float(np.mean(successes))
        if success_rate < min_success_rate:
            warnings.warn("empirical success rate too small for reliable estimates", stacklevel=2)  # pragma: no cover - warning path
            return False

        Phi = self._design_matrix(states, goals, actions)
        y = successes.reshape(-1)
        w = self.W.copy()

        for _ in range(max_iter):
            logits = Phi @ w
            probs = 1.0 / (1.0 + np.exp(-logits))
            gradient = Phi.T @ (probs - y) / states.shape[0] + self.lam * w
            weights = probs * (1.0 - probs) + 1e-8
            hessian = (Phi.T @ (Phi * weights[:, None])) / states.shape[0] + self.lam * np.eye(self.D_total)
            try:
                delta = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive fallback
                delta = gradient
            w_new = w - delta
            if np.linalg.norm(delta) < tol:
                w = w_new
                break
            w = w_new

        logits = Phi @ w
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        loss = -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))

        self.W = w
        self.last_train_loss = float(loss)
        self.n_samples_seen += states.shape[0]
        self._predicted.extend(probs.tolist())
        self._actual.extend(y.astype(int).tolist())

        if goals is not None:  # pragma: no cover - optional goal tracking
            for state_vec, goal_vec, action, prob in zip(states, goals, actions, probs):
                aff = Affordance(
                    object_id="default",
                    state=tuple(state_vec.tolist()),
                    goal=tuple(goal_vec.tolist()),
                    action=int(action),
                    probability=float(prob),
                )
                self._affordance_db.setdefault(aff, []).append(float(prob))

        return True

    def conditional_fit(
        self,
        states: State,
        actions: NDArray[np.int64],
        successes: NDArray[np.float64],
        min_samples: int = 64,
        min_success_rate: float = 0.3,
        goals: Optional[State] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> bool:
        """Compatibility wrapper that mirrors the old API name."""

        return self.fit(
            states,
            actions,
            successes,
            goals=goals,
            min_samples=min_samples,
            min_success_rate=min_success_rate,
            max_iter=max_iter,
            tol=tol,
        )

    # ------------------------------------------------------------------
    # Calibration utilities
    # ------------------------------------------------------------------
    def calibration_curve(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:  # pragma: no cover - analysis helper
        if len(self._predicted) < self.calibration_bins:  # pragma: no cover - insufficient data path
            warnings.warn("insufficient data for calibration curve", stacklevel=2)
            return np.array([]), np.array([])

        predicted = np.asarray(self._predicted)
        actual = np.asarray(self._actual)
        bins = np.linspace(0.0, 1.0, self.calibration_bins + 1)

        mean_pred: List[float] = []
        empirical: List[float] = []

        for i in range(self.calibration_bins):
            mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
            if i == self.calibration_bins - 1:
                mask |= predicted == 1.0
            if np.any(mask):
                mean_pred.append(float(np.mean(predicted[mask])))
                empirical.append(float(np.mean(actual[mask])))

        return np.asarray(mean_pred), np.asarray(empirical)

    def expected_calibration_error(self) -> float:  # pragma: no cover - analysis helper
        mean_pred, empirical = self.calibration_curve()
        if mean_pred.size == 0:
            return 0.0
        predicted = np.asarray(self._predicted)
        bins = np.linspace(0.0, 1.0, self.calibration_bins + 1)
        ece = 0.0
        total = predicted.size
        for i, (pred, emp) in enumerate(zip(mean_pred, empirical)):
            mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
            if i == len(mean_pred) - 1:
                mask |= predicted == 1.0
            weight = np.sum(mask) / total
            ece += weight * abs(emp - pred)
        return float(ece)

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------
    @staticmethod
    def compose_conjunction(affordances: Iterable[Affordance]) -> float:  # pragma: no cover - composition helper
        prob = 1.0
        for aff in affordances:
            prob *= aff.probability
        return float(prob)

    @staticmethod
    def compose_disjunction(affordances: Iterable[Affordance]) -> float:  # pragma: no cover - composition helper
        complement = 1.0
        for aff in affordances:
            complement *= 1.0 - aff.probability
        return float(1.0 - complement)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:  # pragma: no cover - serialization helper
        return {
            "obs_dim": self.D_obs,
            "goal_dim": self.D_goal,
            "num_actions": self.A,
            "ridge_lambda": self.lam,
            "weights": self.W.tolist(),
            "n_samples_seen": self.n_samples_seen,
            "last_train_loss": self.last_train_loss,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "CompositionalAffordanceMap":  # pragma: no cover - serialization helper
        obj = cls(
            obs_dim=int(payload["obs_dim"]),
            num_actions=int(payload["num_actions"]),
            ridge_lambda=float(payload.get("ridge_lambda", 1e-2)),
            goal_dim=int(payload.get("goal_dim", 0)),
        )
        obj.W = np.asarray(payload["weights"], dtype=np.float64)
        obj.n_samples_seen = int(payload.get("n_samples_seen", 0))
        obj.last_train_loss = float(payload.get("last_train_loss", float("inf")))
        return obj


class LinearAffordanceMap(CompositionalAffordanceMap):
    """Drop-in replacement for the historic LinearAffordanceMap class."""

    def __init__(self, obs_dim: int, num_actions: int, ridge_lambda: float = 1e-2, *, seed: int = 0) -> None:
        super().__init__(obs_dim=obs_dim, num_actions=num_actions, ridge_lambda=ridge_lambda, goal_dim=0, seed=seed)


__all__ = ["Affordance", "CompositionalAffordanceMap", "LinearAffordanceMap"]
