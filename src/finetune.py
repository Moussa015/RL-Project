"""Poursuite d'entraînement à faible ε à partir d'une Q-table déjà apprise."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import QLearningAgent, SarsaAgent
from src.discretizer import UniformDiscretizer
from src.env_config import get_spec
from src.train import (
    epsilon_at,
    evaluate_greedy,
    save_metrics,
    save_model,
    success_of_episode,
)


def finetune_one(
    npz_path: Path,
    extra_episodes: int,
    eps_start: float,
    eps_min: float,
    alpha: float | None,
    seed: int,
) -> dict:
    data = np.load(npz_path)
    env_id = str(data["env_id"])
    algo = str(data["algo"])
    n_bins = int(data["n_bins"])
    spec = get_spec(env_id)
    disc = UniformDiscretizer(data["low"], data["high"], data["bins"])
    gamma = float(data["gamma"]) if "gamma" in data.files else 0.99
    alpha = float(alpha) if alpha is not None else float(data["alpha"])

    rng = np.random.default_rng(seed + 50_000)
    cls = QLearningAgent if algo == "qlearning" else SarsaAgent
    agent = cls(
        n_states=disc.n_states,
        n_actions=spec.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=eps_start,
        q_init=0.0,
        rng=rng,
    )
    agent.Q = data["Q"].astype(np.float64).copy()
    if "visit_count" in data.files:
        agent.visit_count = data["visit_count"].astype(np.int64).copy()

    env = gym.make(env_id)
    env.reset(seed=seed + 50_000)
    env.action_space.seed(seed + 50_000)

    returns = np.empty(extra_episodes, dtype=np.float64)
    lengths = np.empty(extra_episodes, dtype=np.int32)
    successes = np.empty(extra_episodes, dtype=np.int8)
    first_success = None

    for ep in range(extra_episodes):
        agent.epsilon = epsilon_at(ep, eps_start, eps_min, extra_episodes)
        obs, _ = env.reset()
        state = disc.discretize(obs)
        action = agent.select_action(state, greedy=False)
        total = 0.0
        length = 0
        terminated = False
        truncated = False
        while True:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            length += 1
            next_state = disc.discretize(next_obs)
            if isinstance(agent, SarsaAgent):
                if terminated:
                    agent.update(state, action, float(reward), next_state, True, 0)
                else:
                    next_action = agent.select_action(next_state, greedy=False)
                    agent.update(state, action, float(reward), next_state, False, next_action)
            else:
                agent.update(state, action, float(reward), next_state, bool(terminated), None)
                next_action = None
            if terminated or truncated:
                break
            state = next_state
            action = next_action if isinstance(agent, SarsaAgent) else agent.select_action(next_state, greedy=False)

        returns[ep] = total
        lengths[ep] = length
        ok = success_of_episode(env_id, total, length, terminated, spec.max_episode_steps)
        successes[ep] = int(ok)
        if ok and first_success is None:
            first_success = ep + 1

    env.close()

    metrics_path = ROOT / "results" / f"{npz_path.stem}_metrics.json"
    old = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    old_returns = np.asarray(old.get("returns", []), dtype=np.float64)
    old_lengths = np.asarray(old.get("lengths", []), dtype=np.int32)
    old_succ = np.asarray(old.get("successes", []), dtype=np.int8)

    merged_returns = np.concatenate([old_returns, returns]) if old_returns.size else returns
    merged_lengths = np.concatenate([old_lengths, lengths]) if old_lengths.size else lengths
    merged_succ = np.concatenate([old_succ, successes]) if old_succ.size else successes

    result = {
        "env_id": env_id,
        "algo": algo,
        "n_bins": n_bins,
        "seed": int(data["seed"]) if "seed" in data.files else seed,
        "n_episodes": int(len(merged_returns)),
        "alpha": alpha,
        "gamma": gamma,
        "eps_start": eps_start,
        "eps_min": eps_min,
        "decay_episodes": extra_episodes,
        "q_init": 0.0,
        "n_states": disc.n_states,
        "n_actions": spec.n_actions,
        "n_visited": agent.n_visited_states(),
        "visit_fraction": agent.visit_fraction(),
        "returns": merged_returns,
        "lengths": merged_lengths,
        "successes": merged_succ,
        "first_success_episode": old.get("first_success_episode") or first_success,
        "discretizer": disc.to_dict(),
        "Q": agent.Q,
        "visit_count": agent.visit_count,
    }
    eval_stats = evaluate_greedy(env_id, agent.Q, disc, n_episodes=100, seed=seed)
    save_model(npz_path, result)
    save_metrics(metrics_path, result, eval_stats)
    print(
        f"OK {npz_path.name}  eval={eval_stats['eval_mean']:.1f}  "
        f"succès={eval_stats['eval_success_rate']*100:.0f}%  "
        f"visites={agent.visit_fraction()*100:.2f}%",
        flush=True,
    )
    return {"key": npz_path.stem, "eval": eval_stats, "n_episodes": result["n_episodes"]}


from concurrent.futures import ProcessPoolExecutor, as_completed


def _worker(args: tuple) -> dict:
    path, extra, eps_start, eps_min, seed = args
    return finetune_one(path, extra, eps_start, eps_min, alpha=None, seed=seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="cartpole_qlearning_bins12_seed*.npz")
    p.add_argument("--extra", type=int, default=2500)
    p.add_argument("--eps-start", type=float, default=0.12)
    p.add_argument("--eps-min", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    paths = sorted((ROOT / "models").glob(args.pattern))
    if not paths:
        raise SystemExit(f"Aucun modèle pour {args.pattern}")
    jobs = []
    for path in paths:
        seed = int(path.stem.split("seed")[-1])
        jobs.append((path, args.extra, args.eps_start, args.eps_min, seed))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, job) for job in jobs]
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    main()
