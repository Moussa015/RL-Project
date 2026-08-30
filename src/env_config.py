"""Formalisation MDP et bornes de discrétisation pour CartPole-v1 et MountainCar-v0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    """Spécification MDP et bornes utilisées pour le binning."""

    env_id: str
    n_actions: int
    obs_low: np.ndarray
    obs_high: np.ndarray
    gamma_default: float
    max_episode_steps: int
    success_threshold: float  # récompense cumulée (évaluation gloutonne)
    description: str


# CartPole-v1 : les vitesses sont non bornées dans Gymnasium.
# On écrête sur un intervalle couvrant les trajectoires typiques
# (un peu plus large que les seuils de chute |x|=2.4 et |theta|=12°).
# Concentrer les bornes de position/angle sur les seuils de terminaison
# évite de gaspiller des cases hors de la région visitable.
CARTPOLE = EnvSpec(
    env_id="CartPole-v1",
    n_actions=2,
    obs_low=np.array([-2.4, -3.0, -0.2095, -3.5], dtype=np.float64),
    obs_high=np.array([2.4, 3.0, 0.2095, 3.5], dtype=np.float64),
    gamma_default=0.99,
    max_episode_steps=500,
    success_threshold=475.0,  # critère Gymnasium « solved » (moyenne 100 épisodes)
    description=(
        "Pendule inversé sur un chariot. Observation (x, x_dot, theta, theta_dot). "
        "Actions : pousser à gauche (0) ou à droite (1). Récompense +1 par pas. "
        "terminated si |x|>2.4 ou |theta|>12° ; truncated à 500 pas."
    ),
)

# MountainCar-v0 : bornes officielles, déjà finies.
MOUNTAINCAR = EnvSpec(
    env_id="MountainCar-v0",
    n_actions=3,
    obs_low=np.array([-1.2, -0.07], dtype=np.float64),
    obs_high=np.array([0.6, 0.07], dtype=np.float64),
    gamma_default=0.99,
    max_episode_steps=200,
    success_threshold=-110.0,  # critère classique « solved » Gymnasium
    description=(
        "Voiture sous-motorisée dans une vallée. Observation (position, vitesse). "
        "Actions : gauche (0), neutre (1), droite (2). Récompense -1 par pas. "
        "terminated si position >= 0.5 ; truncated à 200 pas."
    ),
)

SPECS = {
    "CartPole-v1": CARTPOLE,
    "MountainCar-v0": MOUNTAINCAR,
}


def get_spec(env_id: str) -> EnvSpec:
    if env_id not in SPECS:
        raise ValueError(f"Environnement non supporté : {env_id}. Choisir parmi {list(SPECS)}")
    return SPECS[env_id]
