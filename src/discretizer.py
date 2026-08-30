"""Discrétisation uniforme (binning) d'un vecteur d'observation continu."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class UniformDiscretizer:
    """Grille uniforme par dimension, encodage en un indice d'état entier.

    Pour chaque dimension i, l'intervalle [low_i, high_i] est découpé en
    ``n_bins`` intervalles de largeur égale. Les observations hors bornes
    sont écrêtées (clipping), ce qui est indispensable pour les vitesses
    non bornées de CartPole.
    """

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        n_bins: int | Sequence[int],
    ) -> None:
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        if self.low.shape != self.high.shape:
            raise ValueError("low et high doivent avoir la même dimension")
        self.n_dims = int(self.low.shape[0])

        if isinstance(n_bins, (int, np.integer)):
            self.n_bins = np.full(self.n_dims, int(n_bins), dtype=np.int64)
        else:
            self.n_bins = np.asarray(n_bins, dtype=np.int64)
            if self.n_bins.shape != (self.n_dims,):
                raise ValueError("n_bins doit être un entier ou un vecteur de même dim que low")
        if np.any(self.n_bins < 2):
            raise ValueError("Il faut au moins 2 intervalles par dimension")

        # Frontières internes pour np.digitize : n_bins-1 seuils par dimension.
        self.edges: list[np.ndarray] = []
        for i in range(self.n_dims):
            # linspace inclut les bornes ; on retire les extrémités pour digitize.
            full = np.linspace(self.low[i], self.high[i], int(self.n_bins[i]) + 1)
            self.edges.append(full[1:-1])

        self.n_states = int(np.prod(self.n_bins))
        # Strides pour l'encodage mixte (équivalent à ravel_multi_index).
        self._strides = np.ones(self.n_dims, dtype=np.int64)
        for i in range(self.n_dims - 2, -1, -1):
            self._strides[i] = self._strides[i + 1] * self.n_bins[i + 1]

    def clip(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(obs, dtype=np.float64), self.low, self.high)

    def discretize(self, obs: np.ndarray) -> int:
        """Retourne l'indice d'état dans {0, ..., n_states-1}."""
        x = self.clip(obs)
        idx = 0
        for i in range(self.n_dims):
            b = int(np.digitize(x[i], self.edges[i]))
            idx += b * int(self._strides[i])
        return idx

    def bin_indices(self, obs: np.ndarray) -> np.ndarray:
        """Indices de case par dimension (utile pour le débogage)."""
        x = self.clip(obs)
        bins = np.empty(self.n_dims, dtype=np.int64)
        for i in range(self.n_dims):
            bins[i] = int(np.digitize(x[i], self.edges[i]))
        return bins

    def to_dict(self) -> dict:
        return {
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "n_bins": self.n_bins.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UniformDiscretizer":
        return cls(low=data["low"], high=data["high"], n_bins=data["n_bins"])
