"""Contextual bandit algorithms for edge and Kelly decisions.

The module provides a small API to plug different contextual bandit
algorithms into the backtesting framework.  The action space is the
cartesian product of a discrete set of ``edge`` thresholds and ``kelly``
caps.  Each algorithm exposes a common interface through ``BanditBase``
with ``select`` and ``update`` methods.

Only a minimal feature set is implemented to keep the code lightweight
while remaining easily extensible.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Action = Tuple[float, float]


@dataclass
class BanditBase:
    """Abstract base class for contextual bandits."""

    actions: Sequence[Action]
    epsilon: float = 0.0
    decay: float = 1.0

    def select(self, x: np.ndarray) -> Dict[str, object]:
        """Return an action for the provided context vector ``x``."""

        raise NotImplementedError

    def update(self, x: np.ndarray, action: Action, reward: float) -> None:
        """Update the model with the observed ``reward`` for ``action``."""

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence helpers
    def state_dict(self) -> Dict[str, object]:  # pragma: no cover - trivial
        return {}

    def load_state_dict(self, sd: Dict[str, object]) -> None:  # pragma: no cover
        pass


class LinUCB(BanditBase):
    """Linear UCB with separate models for each action."""

    def __init__(
        self,
        actions: Sequence[Action],
        alpha: float = 1.0,
        epsilon: float = 0.0,
        prior_lambda: float = 1.0,
        decay: float = 1.0,
    ) -> None:
        super().__init__(actions, epsilon, decay)
        self.alpha = alpha
        self.prior_lambda = prior_lambda
        self.d: int | None = None
        self.A: Dict[Action, np.ndarray] = {}
        self.b: Dict[Action, np.ndarray] = {}

    def _ensure_init(self, d: int) -> None:
        if self.d is not None:
            return
        self.d = d
        I = np.eye(d)
        for a in self.actions:
            self.A[a] = self.prior_lambda * I.copy()
            self.b[a] = np.zeros(d)

    def select(self, x: np.ndarray) -> Dict[str, object]:
        self._ensure_init(len(x))
        if np.random.rand() < self.epsilon:
            action = self.actions[np.random.randint(len(self.actions))]
            return {"action": action, "scores": None}

        scores: List[float] = []
        for a in self.actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(x @ theta)
            bonus = self.alpha * float(np.sqrt(x @ A_inv @ x))
            scores.append(mean + bonus)
        idx = int(np.argmax(scores))
        action = self.actions[idx]
        return {"action": action, "scores": scores}

    def update(self, x: np.ndarray, action: Action, reward: float) -> None:
        self._ensure_init(len(x))
        A = self.A[action]
        b = self.b[action]
        A *= self.decay
        b *= self.decay
        A += np.outer(x, x)
        b += reward * x
        self.A[action] = A
        self.b[action] = b

    def state_dict(self) -> Dict[str, object]:  # pragma: no cover - simple
        out = {}
        for a in self.actions:
            out[str(a)] = {"A": self.A[a].tolist(), "b": self.b[a].tolist()}
        return out

    def load_state_dict(self, sd: Dict[str, object]) -> None:  # pragma: no cover
        for key, val in sd.items():
            a = eval(key)  # noqa: S307 - controlled input
            self.A[a] = np.array(val["A"])
            self.b[a] = np.array(val["b"])
        self.d = len(next(iter(self.A.values())))


class ThompsonGaussian(BanditBase):
    """Gaussian Thompson Sampling with independent linear models."""

    def __init__(
        self,
        actions: Sequence[Action],
        epsilon: float = 0.0,
        prior_lambda: float = 1.0,
        decay: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(actions, epsilon, decay)
        self.prior_lambda = prior_lambda
        self.d: int | None = None
        self.A: Dict[Action, np.ndarray] = {}
        self.b: Dict[Action, np.ndarray] = {}
        self.rng = np.random.default_rng(seed)

    def _ensure_init(self, d: int) -> None:
        if self.d is not None:
            return
        self.d = d
        I = np.eye(d)
        for a in self.actions:
            self.A[a] = self.prior_lambda * I.copy()
            self.b[a] = np.zeros(d)

    def select(self, x: np.ndarray) -> Dict[str, object]:
        self._ensure_init(len(x))
        if self.rng.random() < self.epsilon:
            action = self.actions[self.rng.integers(len(self.actions))]
            return {"action": action, "scores": None}

        sampled: List[float] = []
        for a in self.actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = self.rng.multivariate_normal(A_inv @ self.b[a], A_inv)
            sampled.append(float(x @ theta))
        idx = int(np.argmax(sampled))
        action = self.actions[idx]
        return {"action": action, "scores": sampled}

    def update(self, x: np.ndarray, action: Action, reward: float) -> None:
        self._ensure_init(len(x))
        A = self.A[action]
        b = self.b[action]
        A *= self.decay
        b *= self.decay
        A += np.outer(x, x)
        b += reward * x
        self.A[action] = A
        self.b[action] = b

    def state_dict(self) -> Dict[str, object]:  # pragma: no cover - simple
        out = {}
        for a in self.actions:
            out[str(a)] = {"A": self.A[a].tolist(), "b": self.b[a].tolist()}
        return out

    def load_state_dict(self, sd: Dict[str, object]) -> None:  # pragma: no cover
        for key, val in sd.items():
            a = eval(key)  # noqa: S307 - controlled input
            self.A[a] = np.array(val["A"])
            self.b[a] = np.array(val["b"])
        self.d = len(next(iter(self.A.values())))


def make_action_space(edge_thr: Iterable[float], kelly_cap: Iterable[float]) -> List[Action]:
    """Create a cartesian product action space."""

    return [(e, k) for e, k in itertools.product(edge_thr, kelly_cap)]


def make_bandit(cfg) -> BanditBase:
    """Factory that builds a bandit from a configuration object.

    Parameters
    ----------
    cfg : Any
        Configuration with fields ``algo`` and others specific to the
        algorithm.  ``cfg.action_space`` must contain ``edge_thr`` and
        ``kelly_cap`` sequences.
    """

    actions = make_action_space(cfg.action_space.edge_thr, cfg.action_space.kelly_cap)
    common = dict(
        actions=actions,
        epsilon=cfg.epsilon,
        decay=cfg.decay,
        prior_lambda=cfg.prior_lambda,
    )
    algo = cfg.algo.lower()
    if algo == "linucb":
        return LinUCB(alpha=cfg.alpha, **common)
    if algo == "thompson":
        return ThompsonGaussian(**common, seed=getattr(cfg, "seed", None))
    raise ValueError(f"Unknown bandit algo {cfg.algo}")
