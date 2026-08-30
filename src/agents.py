"""Agents tabulaires : Q-learning (hors-politique) et SARSA (sur-politique)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TabularAgent(ABC):
    """Agent epsilon-glouton sur une Q-table NumPy."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        q_init: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.Q = np.full((n_states, n_actions), q_init, dtype=np.float64)
        # Compteur de visites par état (malédiction de la dimension).
        self.visit_count = np.zeros(n_states, dtype=np.int64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        # En cas d'égalité, tirage uniforme parmi les argmax (évite un biais à gauche).
        q_row = self.Q[state]
        max_q = q_row.max()
        candidates = np.flatnonzero(q_row == max_q)
        return int(self.rng.choice(candidates))

    @abstractmethod
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool, next_action: int | None = None) -> None:
        raise NotImplementedError

    def n_visited_states(self) -> int:
        return int(np.count_nonzero(self.visit_count))

    def visit_fraction(self) -> float:
        return self.n_visited_states() / float(self.n_states)


class QLearningAgent(TabularAgent):
    """Contrôle TD hors-politique : cible = R + gamma * max_a Q(S', a)."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool, next_action: int | None = None) -> None:
        self.visit_count[state] += 1
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error


class SarsaAgent(TabularAgent):
    """Contrôle TD sur-politique : cible = R + gamma * Q(S', A')."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool, next_action: int | None = None) -> None:
        if next_action is None and not done:
            raise ValueError("SARSA exige next_action lorsque l'épisode n'est pas terminé")
        self.visit_count[state] += 1
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, int(next_action)]
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
