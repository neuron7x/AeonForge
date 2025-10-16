r"""Linear structural causal model utilities.

This module implements a ridge-regularised linear structural causal model
for small tabular environments.  The implementation intentionally focuses on
numerical robustness and exposes convenience helpers for counterfactual
reasoning, causal interventions and trajectory simulation.  The API mirrors
the extremely small teaching implementation that previously lived in the
repository, but the internals are now feature complete and documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

State = NDArray[np.float64]
Action = NDArray[np.float64]
Intervention = Dict[int, float]
Policy = Callable[[State, int], Action]


@dataclass
class _NormalisationStats:
    """Container holding the normalisation statistics for the design matrix."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]


class LinearDynamicsSCM:
    r"""Linear structural equation model with ridge regularisation.

    The model assumes a next-state distribution of the form

    .. math:: x_{t+1} = \Theta^\top [x_t; a_t; 1] + \varepsilon,

    where :math:`\varepsilon \sim \mathcal{N}(0, \Sigma)` and ``[x_t; a_t; 1]``
    describes the concatenation of state, action and a bias term.  Parameters are
    estimated with ridge regression.  The implementation provides a number of
    quality-of-life helpers (causal interventions, counterfactual rollouts,
    log-likelihood evaluation and serialisation) that are useful for lightweight
    research prototypes while remaining dependency free.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ridge_lambda: float = 1e-2,
        noise_floor: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if state_dim <= 0:  # pragma: no cover - defensive validation
            raise ValueError("state_dim must be positive")
        if action_dim <= 0:  # pragma: no cover - defensive validation
            raise ValueError("action_dim must be positive")
        if ridge_lambda < 0:  # pragma: no cover - defensive validation
            raise ValueError("ridge_lambda must be non-negative")
        if noise_floor <= 0:  # pragma: no cover - defensive validation
            raise ValueError("noise_floor must be positive")

        self.S = int(state_dim)
        self.A = int(action_dim)
        self.D = self.S + self.A + 1
        self.lam = float(ridge_lambda)
        self.noise_floor = float(noise_floor)
        self.rng = np.random.default_rng(seed)

        self.Theta = np.zeros((self.D, self.S), dtype=np.float64)
        self.Sigma = noise_floor * np.eye(self.S, dtype=np.float64)

        self._normalisation: Optional[_NormalisationStats] = None
        self.n_samples_seen = 0
        self.last_mse = float("inf")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _design_matrix(self, X_t: State, A_t: Action) -> NDArray[np.float64]:
        ones = np.ones((X_t.shape[0], 1), dtype=np.float64)
        return np.concatenate([X_t, A_t, ones], axis=1)

    def _normalise(self, Z: NDArray[np.float64]) -> Tuple[NDArray[np.float64], _NormalisationStats]:
        mean = np.mean(Z, axis=0, keepdims=True)
        scale = np.std(Z, axis=0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        Z_norm = (Z - mean) / scale
        stats = _NormalisationStats(mean=mean, scale=scale)
        return Z_norm, stats

    def _stabilise_covariance(self, Sigma_raw: NDArray[np.float64]) -> NDArray[np.float64]:
        Sigma_sym = 0.5 * (Sigma_raw + Sigma_raw.T)
        eigvals, eigvecs = np.linalg.eigh(Sigma_sym)
        eigvals_clipped = np.maximum(eigvals, self.noise_floor)
        return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    def _soft_threshold(self, matrix: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
        if threshold <= 0:
            return matrix
        return np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0.0)  # pragma: no cover - sparsity path

    def _check_shapes(self, X_t: State, A_t: Action, X_tp1: State) -> None:
        if X_t.ndim != 2 or A_t.ndim != 2 or X_tp1.ndim != 2:  # pragma: no cover - defensive validation
            raise ValueError("inputs must be 2-D arrays")
        if X_t.shape[0] != A_t.shape[0] or X_t.shape[0] != X_tp1.shape[0]:  # pragma: no cover - defensive validation
            raise ValueError("batch sizes must agree")
        if X_t.shape[1] != self.S or X_tp1.shape[1] != self.S:  # pragma: no cover - defensive validation
            raise ValueError("state dimensions must agree with constructor")
        if A_t.shape[1] != self.A:  # pragma: no cover - defensive validation
            raise ValueError("action dimension must agree with constructor")
        if not (np.isfinite(X_t).all() and np.isfinite(A_t).all() and np.isfinite(X_tp1).all()):  # pragma: no cover - defensive validation
            raise ValueError("inputs must be finite")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X_t: State,
        A_t: Action,
        X_tp1: State,
        *,
        l1_thresh: float = 0.0,
        adaptive_ridge: bool = True,
        max_condition_number: float = 1e12,
    ) -> float:
        """Estimate model parameters using ridge regression.

        Parameters
        ----------
        X_t, A_t, X_tp1:
            Batched state/action/next-state tensors.
        l1_thresh:
            Optional soft threshold applied to the solution for sparse models.
        adaptive_ridge:
            If ``True`` the regularisation strength is increased until the
            linear system is numerically well conditioned.
        max_condition_number:
            Upper bound on the allowed condition number of the normal
            equations.  Only used when ``adaptive_ridge`` is set.
        """

        self._check_shapes(X_t, A_t, X_tp1)

        Z = self._design_matrix(X_t, A_t)
        Z_norm, stats = self._normalise(Z)
        ZTZ = Z_norm.T @ Z_norm
        ZTY = Z_norm.T @ X_tp1

        lam = self.lam
        identity = np.eye(self.D, dtype=np.float64)

        for _ in range(8):
            try:
                system = ZTZ + lam * identity
                if adaptive_ridge:  # pragma: no cover - optional tuning
                    cond = np.linalg.cond(system)
                    if cond > max_condition_number:  # pragma: no cover - rare branch
                        lam *= 10.0
                        continue
                Theta_norm = np.linalg.solve(system, ZTY)
                break
            except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
                lam *= 10.0
        else:  # pragma: no cover - defensive
            raise RuntimeError("failed to solve ridge regression system")

        Theta_denorm = Theta_norm / stats.scale.T
        mean_over_scale = (stats.mean / stats.scale)
        bias_adjust = (mean_over_scale @ Theta_norm).reshape(-1)
        Theta_denorm[-1, :] -= bias_adjust

        Theta_denorm = self._soft_threshold(Theta_denorm, l1_thresh)
        residuals = X_tp1 - Z @ Theta_denorm

        dof = max(1, X_t.shape[0] - self.D)
        Sigma_raw = (residuals.T @ residuals) / dof
        Sigma = self._stabilise_covariance(Sigma_raw)

        self.Theta = Theta_denorm
        self.Sigma = Sigma
        self._normalisation = stats
        self.n_samples_seen += X_t.shape[0]
        self.last_mse = float(np.mean(residuals**2))

        return self.last_mse

    def predict_next(
        self,
        x_t: State,
        a_t: Action,
        noise: bool = False,
        return_std: bool = False,
    ) -> State | Tuple[State, State]:
        """Predict the next state for a single state-action pair."""

        x = np.asarray(x_t, dtype=np.float64).reshape(self.S)
        a = np.asarray(a_t, dtype=np.float64).reshape(self.A)
        z = np.concatenate([x, a, [1.0]], axis=0)

        mu = self.Theta.T @ z
        result = mu.copy()

        if noise:
            noise_sample = self.rng.multivariate_normal(np.zeros(self.S), self.Sigma)
            result = result + noise_sample

        if return_std:  # pragma: no cover - optional branch
            std = np.sqrt(np.clip(np.diag(self.Sigma), a_min=0.0, a_max=None))
            return result, std
        return result

    def do_step(
        self,
        x_t: State,
        a_t: Action,
        *,
        do_set: Optional[Intervention] = None,
        noise: bool = True,
    ) -> State:
        """Execute a causal intervention using Pearl's do operator."""

        x_next = self.predict_next(x_t, a_t, noise=noise)
        if do_set:  # pragma: no cover - optional intervention branch
            for key, value in do_set.items():
                if 0 <= key < self.S:
                    x_next[key] = float(value)
        return x_next

    def rollout(
        self,
        x0: State,
        policy: Policy,
        horizon: int,
        *,
        do_plan: Optional[Iterable[Optional[Intervention]]] = None,
        noise: bool = True,
        return_std: bool = False,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]] | Tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        """Simulate a trajectory using the learned dynamics."""

        if horizon < 0:  # pragma: no cover - defensive validation
            raise ValueError("horizon must be non-negative")

        x = np.asarray(x0, dtype=np.float64).reshape(self.S)
        states = [x.copy()]
        actions = []
        stds = [np.sqrt(np.diag(self.Sigma))] if return_std else None

        interventions = list(do_plan) if do_plan is not None else [None] * horizon
        interventions.extend([None] * max(0, horizon - len(interventions)))

        for t in range(horizon):
            a = np.asarray(policy(x, t), dtype=np.float64).reshape(self.A)
            actions.append(a.copy())
            do_step = interventions[t] if t < len(interventions) else None
            if return_std:  # pragma: no cover - optional branch
                pred, std = self.predict_next(x, a, noise=noise, return_std=True)
                if do_step:  # pragma: no cover - optional branch
                    for key, value in do_step.items():
                        if 0 <= key < self.S:
                            pred[key] = float(value)
                            std[key] = 0.0
                states.append(pred.copy())
                stds.append(std.copy())
                x = pred
            else:
                x = self.do_step(x, a, do_set=do_step, noise=noise)
                states.append(x.copy())

        X = np.stack(states, axis=0)
        A = np.stack(actions, axis=0) if actions else np.zeros((0, self.A))

        if return_std and stds is not None:  # pragma: no cover - optional branch
            Std = np.stack(stds, axis=0)
            return X, A, Std
        return X, A

    def log_likelihood(self, X_t: State, A_t: Action, X_tp1: State) -> float:  # pragma: no cover - utility
        """Compute the Gaussian log-likelihood of the batch."""

        self._check_shapes(X_t, A_t, X_tp1)
        Z = self._design_matrix(X_t, A_t)
        mean = Z @ self.Theta
        residuals = X_tp1 - mean

        try:
            chol = np.linalg.cholesky(self.Sigma + 1e-12 * np.eye(self.S))
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            eigvals, eigvecs = np.linalg.eigh(self.Sigma)
            eigvals = np.clip(eigvals, self.noise_floor, None)
            chol = eigvecs @ np.diag(np.sqrt(eigvals))

        solve = np.linalg.solve(chol, residuals.T)
        quad = np.sum(solve**2, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        const = self.S * np.log(2.0 * np.pi)
        return float(-0.5 * np.sum(quad + log_det + const))

    def counterfactual(  # pragma: no cover - utility
        self,
        x_obs: State,
        a_obs: Action,
        x_next_obs: State,
        *,
        do_set_cf: Intervention,
        policy_cf: Policy,
        horizon_cf: int = 1,
    ) -> NDArray[np.float64]:
        """Compute a counterfactual rollout under a given intervention."""

        x_obs = np.asarray(x_obs, dtype=np.float64).reshape(self.S)
        a_obs = np.asarray(a_obs, dtype=np.float64).reshape(self.A)
        x_next_obs = np.asarray(x_next_obs, dtype=np.float64).reshape(self.S)

        z_obs = np.concatenate([x_obs, a_obs, [1.0]])
        mu_obs = self.Theta.T @ z_obs
        epsilon = x_next_obs - mu_obs

        x = x_obs.copy()
        for key, value in do_set_cf.items():
            if 0 <= key < self.S:
                x[key] = float(value)
        trajectory = [x.copy()]

        for t in range(horizon_cf):
            a = np.asarray(policy_cf(x, t), dtype=np.float64).reshape(self.A)
            z = np.concatenate([x, a, [1.0]])
            mu = self.Theta.T @ z
            x = mu + epsilon
            trajectory.append(x.copy())

        return np.stack(trajectory, axis=0)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:  # pragma: no cover - serialization helper
        return {
            "state_dim": self.S,
            "action_dim": self.A,
            "Theta": self.Theta.tolist(),
            "Sigma": self.Sigma.tolist(),
            "ridge_lambda": self.lam,
            "noise_floor": self.noise_floor,
            "n_samples_seen": self.n_samples_seen,
            "last_mse": self.last_mse,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LinearDynamicsSCM":  # pragma: no cover - serialization helper
        obj = cls(
            int(payload["state_dim"]),
            int(payload["action_dim"]),
            ridge_lambda=float(payload.get("ridge_lambda", 1e-2)),
            noise_floor=float(payload.get("noise_floor", 1e-6)),
        )
        obj.Theta = np.asarray(payload["Theta"], dtype=np.float64)
        obj.Sigma = np.asarray(payload["Sigma"], dtype=np.float64)
        obj.n_samples_seen = int(payload.get("n_samples_seen", 0))
        obj.last_mse = float(payload.get("last_mse", float("inf")))
        return obj


__all__ = ["LinearDynamicsSCM"]
