"""Rejeu d'un agent sauvegardé, sans réentraînement.

Écran unique (tous les scénarios) :
    python play.py

Commande exigée par le sujet (un modèle précis) :
    python play.py --model models/cartpole_best.npz
    python play.py --model models/mountaincar_best.npz --episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.discretizer import UniformDiscretizer
from src.env_config import get_spec
from src.train import success_of_episode


def load_model(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "Q": data["Q"],
        "env_id": str(data["env_id"]),
        "algo": str(data["algo"]),
        "n_bins": int(data["n_bins"]),
        "low": data["low"],
        "high": data["high"],
        "bins": data["bins"],
    }


def play(
    model_path: Path,
    n_episodes: int = 5,
    seed: int = 0,
    render: bool = True,
    fps: int = 50,
) -> None:
    payload = load_model(model_path)
    spec = get_spec(payload["env_id"])
    discretizer = UniformDiscretizer(payload["low"], payload["high"], payload["bins"])
    Q = payload["Q"]
    rng = np.random.default_rng(seed)

    render_mode = "human" if render else None
    env = gym.make(payload["env_id"], render_mode=render_mode)
    env.reset(seed=seed)

    print(
        f"Rejeu {payload['algo']} sur {payload['env_id']} "
        f"(grille {payload['n_bins']} bins/dim, {discretizer.n_states} états)"
    )

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        length = 0
        terminated = False
        truncated = False
        while True:
            state = discretizer.discretize(obs)
            q_row = Q[state]
            action = int(rng.choice(np.flatnonzero(q_row == q_row.max())))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            length += 1
            if terminated or truncated:
                break
        ok = success_of_episode(payload["env_id"], total, length, terminated, spec.max_episode_steps)
        returns.append(total)
        print(f"  épisode {ep+1}: R={total:.1f}  longueur={length}  succès={ok}")

    env.close()
    print(f"Moyenne : {np.mean(returns):.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Rejouer un agent tabulaire sauvegardé")
    p.add_argument(
        "--model",
        default=None,
        help="Fichier .npz : mode ligne de commande (exigence du sujet). "
        "Sans cet argument, ouvre l'écran unique de démonstration.",
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--gui", action="store_true", help="Forcer l'écran unique")
    args = p.parse_args()

    use_gui = args.gui or (args.model is None and not args.no_render)
    if use_gui:
        from src.demo_gui import run_gui

        run_gui(seed=args.seed)
        return

    model = Path(args.model or "models/cartpole_best.npz")
    play(model, args.episodes, args.seed, render=not args.no_render)


if __name__ == "__main__":
    main()
