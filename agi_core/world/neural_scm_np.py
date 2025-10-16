"""Neural structural causal model with uncertainty estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

State = NDArray[np.float64]
Action = NDArray[np.float64]
Intervention = Dict[int, float]
Policy = Callable[[State, int], Action]


@dataclass
class _Network:
    """Container for the parameters and optimiser state of a network."""

    W1: NDArray[np.float64]
    b1: NDArray[np.float64]
    W2: NDArray[np.float64]
    b2: NDArray[np.float64]
    W3: NDArray[np.float64]
    b3: NDArray[np.float64]
    m: Dict[str, NDArray[np.float64]]
    v: Dict[str, NDArray[np.float64]]
    step: int = 0


def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:  # pragma: no cover - activation helper
    return np.maximum(0.0, x)


def _swish(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x / (1.0 + np.exp(-x))


class NeuralDynamicsSCM:
    """Two-layer neural SCM trained with an Adam optimiser.

    Each network in the ensemble predicts a Gaussian distribution over the
    next state and exposes both aleatoric and epistemic uncertainty.  The
    implementation is intentionally minimalist – it is dependency free, easy to
    read and integrates with the rest of the lightweight numpy-based stack.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: Optional[int] = None,
        lr: Optional[float] = None,
        seed: int = 0,
        *,
        ensemble_size: int = 5,
        activation: str = "swish",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        hidden: Optional[int] = None,
        ensemble: Optional[int] = None,
    ) -> None:
        if state_dim <= 0 or action_dim <= 0:  # pragma: no cover - defensive validation
            raise ValueError("state_dim and action_dim must be positive")
        if hidden is not None:
            if hidden_dim is not None:  # pragma: no cover - defensive validation
                raise ValueError("provide either hidden_dim or hidden, not both")
            hidden_dim = hidden
        if ensemble is not None:  # pragma: no cover - optional alias
            ensemble_size = ensemble

        hidden_dim = 128 if hidden_dim is None else int(hidden_dim)
        lr = 3e-4 if lr is None else float(lr)

        if hidden_dim <= 0:  # pragma: no cover - defensive validation
            raise ValueError("hidden_dim must be positive")
        if ensemble_size <= 0:  # pragma: no cover - defensive validation
            raise ValueError("ensemble_size must be positive")
        if activation not in {"relu", "swish"}:  # pragma: no cover - defensive validation
            raise ValueError("activation must be 'relu' or 'swish'")
        if lr <= 0:  # pragma: no cover - defensive validation
            raise ValueError("learning rate must be positive")
        if not (0 < beta1 < 1 and 0 < beta2 < 1):  # pragma: no cover - defensive validation
            raise ValueError("beta1 and beta2 must lie in (0, 1)")
        if eps <= 0:  # pragma: no cover - defensive validation
            raise ValueError("eps must be positive")

        self.S = int(state_dim)
        self.A = int(action_dim)
        self.D_in = self.S + self.A
        self.H = int(hidden_dim)
        self.K = int(ensemble_size)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps_adam = float(eps)
        self.weight_decay = float(weight_decay)
        self.activation = _swish if activation == "swish" else _relu

        self.rng = np.random.default_rng(seed)
        self.networks: List[_Network] = [self._create_network(seed + k) for k in range(self.K)]
        self.training_steps = 0
        self.last_train_loss = float("inf")

    # ------------------------------------------------------------------
    # Network construction and helpers
    # ------------------------------------------------------------------
    def _create_network(self, seed: int) -> _Network:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / self.D_in)
        scale2 = np.sqrt(2.0 / self.H)

        W1 = rng.normal(0.0, scale1, size=(self.D_in, self.H))
        b1 = np.zeros(self.H, dtype=np.float64)
        W2 = rng.normal(0.0, scale2, size=(self.H, self.S))
        b2 = np.zeros(self.S, dtype=np.float64)
        W3 = rng.normal(0.0, scale2, size=(self.H, self.S))
        b3 = np.zeros(self.S, dtype=np.float64)

        zeros = lambda shape: np.zeros(shape, dtype=np.float64)
        m = {
            "W1": zeros(W1.shape),
            "b1": zeros(b1.shape),
            "W2": zeros(W2.shape),
            "b2": zeros(b2.shape),
            "W3": zeros(W3.shape),
            "b3": zeros(b3.shape),
        }
        v = {key: zeros(arr.shape) for key, arr in m.items()}
        return _Network(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, m=m, v=v)

    def _forward(self, net: _Network, Z: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        hidden = self.activation(Z @ net.W1 + net.b1)
        mean = hidden @ net.W2 + net.b2
        log_var = np.clip(hidden @ net.W3 + net.b3, -10.0, 2.0)
        return hidden, mean, log_var

    def _call_forward(
        self, net: _Network, Z: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        forward_fn = getattr(self, "_forward")
        try:
            result = forward_fn(net, Z)
        except TypeError:
            result = forward_fn(Z)
        if not isinstance(result, tuple):  # pragma: no cover - defensive check
            raise TypeError("_forward must return a tuple")
        if len(result) == 2:
            hidden, mean = result
            log_var = np.zeros_like(mean)
        elif len(result) == 3:
            hidden, mean, log_var = result
        else:  # pragma: no cover - defensive check
            raise ValueError("_forward returned unexpected number of outputs")
        return hidden, mean, log_var

    def _nll(self, mean: NDArray[np.float64], log_var: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        var = np.exp(log_var)
        squared_error = (target - mean) ** 2
        return float(0.5 * np.mean(log_var + squared_error / (var + 1e-8)))

    def _adam_update(self, net: _Network, name: str, grad: NDArray[np.float64]) -> None:
        m = net.m[name] = self.beta1 * net.m[name] + (1.0 - self.beta1) * grad
        v = net.v[name] = self.beta2 * net.v[name] + (1.0 - self.beta2) * (grad**2)
        m_hat = m / (1.0 - self.beta1 ** (net.step + 1))
        v_hat = v / (1.0 - self.beta2 ** (net.step + 1))

        param = getattr(net, name)
        param *= 1.0 - self.weight_decay * self.lr
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps_adam)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X_t: State,
        A_t: Action,
        X_tp1: State,
        *,
        epochs: int = 100,
        batch_size: int = 128,
        val_split: float = 0.15,
        patience: int = 10,
        min_delta: float = 1e-4,
        batch: Optional[int] = None,
    ) -> float:
        if X_t.shape != X_tp1.shape:  # pragma: no cover - defensive validation
            raise ValueError("state arrays must have matching shapes")
        if X_t.shape[0] != A_t.shape[0]:  # pragma: no cover - defensive validation
            raise ValueError("batch sizes must agree")
        if X_t.shape[1] != self.S or A_t.shape[1] != self.A:  # pragma: no cover - defensive validation
            raise ValueError("dimensions mismatch constructor")
        if not (np.isfinite(X_t).all() and np.isfinite(A_t).all() and np.isfinite(X_tp1).all()):  # pragma: no cover - defensive validation
            raise ValueError("inputs must be finite")
        if batch is not None:
            batch_size = int(batch)
        if epochs <= 0:
            return self.last_train_loss
        if batch_size <= 0:  # pragma: no cover - defensive validation
            raise ValueError("batch_size must be positive")

        Z = np.concatenate([X_t, A_t], axis=1)
        Y = X_tp1
        N = Z.shape[0]
        n_val = max(1, int(N * val_split)) if N > 4 else 1

        val_losses = []
        for net in self.networks:
            rng = np.random.default_rng(self.rng.integers(0, 1_000_000))
            indices = rng.permutation(N)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:] if n_val < N else indices[1:]
            if train_idx.size == 0:  # pragma: no cover - rare fallback
                train_idx = val_idx

            best_val = float("inf")
            stall = 0

            for _ in range(epochs):
                rng.shuffle(train_idx)
                for start in range(0, train_idx.size, batch_size):
                    batch = train_idx[start : start + batch_size]
                    pre_activation = Z[batch] @ net.W1 + net.b1
                    hidden, mean, log_var = self._call_forward(net, Z[batch])
                    var = np.exp(log_var)
                    diff = (mean - Y[batch]) / (var + 1e-8) / batch.size
                    grad_mean = diff
                    grad_log_var = 0.5 * (1.0 - (Y[batch] - mean) ** 2 / (var + 1e-8)) / batch.size

                    grad_W2 = hidden.T @ grad_mean
                    grad_b2 = grad_mean.sum(axis=0)
                    grad_W3 = hidden.T @ grad_log_var
                    grad_b3 = grad_log_var.sum(axis=0)

                    grad_hidden = grad_mean @ net.W2.T + grad_log_var @ net.W3.T
                    if self.activation is _relu:  # pragma: no cover - optional branch
                        grad_hidden *= (hidden > 0).astype(np.float64)
                    else:
                        sig = 1.0 / (1.0 + np.exp(-pre_activation))
                        grad_hidden *= sig + pre_activation * sig * (1.0 - sig)

                    grad_W1 = Z[batch].T @ grad_hidden
                    grad_b1 = grad_hidden.sum(axis=0)

                    self._adam_update(net, "W1", grad_W1)
                    self._adam_update(net, "b1", grad_b1)
                    self._adam_update(net, "W2", grad_W2)
                    self._adam_update(net, "b2", grad_b2)
                    self._adam_update(net, "W3", grad_W3)
                    self._adam_update(net, "b3", grad_b3)

                    net.step += 1

                _, mean_val, log_var_val = self._call_forward(net, Z[val_idx])
                val_loss = self._nll(mean_val, log_var_val, Y[val_idx])
                if val_loss + min_delta < best_val:
                    best_val = val_loss
                    stall = 0
                else:
                    stall += 1
                    if stall >= patience:
                        break

            val_losses.append(best_val)

        self.training_steps += 1
        self.last_train_loss = float(np.mean(val_losses))
        return self.last_train_loss

    # ------------------------------------------------------------------
    # Inference and simulation
    # ------------------------------------------------------------------
    def predict_next(
        self,
        x_t: State,
        a_t: Action,
        noise: bool = False,
        return_uncertainty: bool = False,
    ) -> State | Tuple[State, State, State]:
        x = np.asarray(x_t, dtype=np.float64).reshape(self.S)
        a = np.asarray(a_t, dtype=np.float64).reshape(self.A)
        z = np.concatenate([x, a])[None, :]

        means = []
        vars_ = []
        for net in self.networks:
            _, mean, log_var = self._call_forward(net, z)
            means.append(mean.reshape(-1))
            vars_.append(np.exp(log_var.reshape(-1)))
        means_arr = np.asarray(means)
        vars_arr = np.asarray(vars_)

        mean_pred = means_arr.mean(axis=0)
        var_epi = means_arr.var(axis=0)
        var_ale = vars_arr.mean(axis=0)

        if noise:
            choice = self.rng.integers(0, self.K)
            sample = self.rng.normal(means_arr[choice], np.sqrt(vars_arr[choice]))
            prediction = sample
        else:
            prediction = mean_pred

        if return_uncertainty:  # pragma: no cover - optional branch
            return prediction, np.sqrt(var_epi), np.sqrt(var_ale)
        return prediction

    def do_step(
        self,
        x_t: State,
        a_t: Action,
        *,
        do_set: Optional[Intervention] = None,
        noise: bool = True,
    ) -> State:
        x_next = self.predict_next(x_t, a_t, noise=noise)
        if do_set:  # pragma: no cover - optional branch
            for dim, value in do_set.items():
                if 0 <= dim < self.S:
                    x_next[dim] = float(value)
        return x_next

    def rollout(
        self,
        x0: State,
        policy: Policy,
        horizon: int,
        *,
        do_plan: Optional[Iterable[Optional[Intervention]]] = None,
        noise: bool = True,
        return_uncertainty: bool = False,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]] | Tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        x = np.asarray(x0, dtype=np.float64).reshape(self.S)
        states = [x.copy()]
        actions = []
        epi = [np.zeros(self.S)] if return_uncertainty else None
        ale = [np.zeros(self.S)] if return_uncertainty else None

        plan = list(do_plan) if do_plan is not None else [None] * horizon
        plan.extend([None] * max(0, horizon - len(plan)))

        for t in range(horizon):
            a = np.asarray(policy(x, t), dtype=np.float64).reshape(self.A)
            actions.append(a.copy())
            intervention = plan[t] if t < len(plan) else None
            if return_uncertainty:  # pragma: no cover - optional branch
                pred, epi_std, ale_std = self.predict_next(x, a, noise=noise, return_uncertainty=True)
                if intervention:  # pragma: no cover - optional branch
                    for dim, value in intervention.items():
                        if 0 <= dim < self.S:
                            pred[dim] = float(value)
                            epi_std[dim] = 0.0
                            ale_std[dim] = 0.0
                states.append(pred.copy())
                epi.append(epi_std.copy())
                ale.append(ale_std.copy())
                x = pred
            else:
                x = self.do_step(x, a, do_set=intervention, noise=noise)
                states.append(x.copy())

        X = np.stack(states, axis=0)
        A = np.stack(actions, axis=0) if actions else np.zeros((0, self.A))
        if return_uncertainty and epi is not None and ale is not None:  # pragma: no cover - optional branch
            return X, A, np.stack(epi, axis=0), np.stack(ale, axis=0)
        return X, A

    def log_likelihood(self, X_t: State, A_t: Action, X_tp1: State) -> float:  # pragma: no cover - utility
        if X_t.shape != X_tp1.shape or X_t.shape[0] != A_t.shape[0]:  # pragma: no cover - defensive validation
            raise ValueError("shape mismatch")
        Z = np.concatenate([X_t, A_t], axis=1)
        per_model = []
        for net in self.networks:
            _, mean, log_var = self._call_forward(net, Z)
            var = np.exp(log_var)
            diff = X_tp1 - mean
            log_prob = -0.5 * np.sum(
                np.log(2.0 * np.pi) + log_var + diff**2 / (var + 1e-8), axis=1
            )
            per_model.append(log_prob)
        stacked = np.stack(per_model, axis=0)
        max_log = stacked.max(axis=0)
        log_sum = max_log + np.log(np.exp(stacked - max_log).mean(axis=0))
        return float(np.sum(log_sum))


__all__ = ["NeuralDynamicsSCM"]
