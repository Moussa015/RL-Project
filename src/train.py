"""Boucle d'entraînement / d'évaluation pour un agent tabulaire discrétisé."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.agents import QLearningAgent, SarsaAgent, TabularAgent
from src.discretizer import UniformDiscretizer
from src.env_config import get_spec


def make_agent(
    algo: str,
    n_states: int,
    n_actions: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    q_init: float,
    seed: int,
) -> TabularAgent:
    rng = np.random.default_rng(seed)
    cls = QLearningAgent if algo == "qlearning" else SarsaAgent
    return cls(
        n_states=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        q_init=q_init,
        rng=rng,
    )


def epsilon_at(episode: int, eps_start: float, eps_min: float, decay_episodes: int) -> float:
    """Décroissance linéaire de epsilon sur ``decay_episodes`` épisodes."""
    if decay_episodes <= 0:
        return eps_min
    frac = min(1.0, episode / float(decay_episodes))
    return eps_start + frac * (eps_min - eps_start)


def run_episode(
    env: gym.Env,
    agent: TabularAgent,
    discretizer: UniformDiscretizer,
    greedy: bool,
    learn: bool,
) -> tuple[float, int, bool]:
    """Un épisode. Retourne (retour, longueur, succès_atteint_le_but).

    Distinction terminated / truncated (Gymnasium) :
    - terminated : vrai état absorbant (chute, drapeau) → pas de bootstrap TD.
    - truncated : coupure artificielle d'horizon → on bootstrap encore.
    """
    obs, _ = env.reset()
    state = discretizer.discretize(obs)
    action = agent.select_action(state, greedy=greedy)
    total = 0.0
    length = 0
    reached_goal = False

    while True:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        length += 1
        next_state = discretizer.discretize(next_obs)

        if learn:
            if isinstance(agent, SarsaAgent):
                if terminated:
                    next_action = 0  # ignoré : pas de bootstrap
                    agent.update(state, action, float(reward), next_state, True, next_action)
                else:
                    # Truncation : bootstrap avec A' (l'horizon n'est pas un vrai terminal).
                    next_action = agent.select_action(next_state, greedy=greedy)
                    agent.update(state, action, float(reward), next_state, False, next_action)
            else:
                agent.update(state, action, float(reward), next_state, bool(terminated), None)

        if terminated or truncated:
            reached_goal = bool(terminated) and float(reward) >= 0.0 and length < env.spec.max_episode_steps if env.spec else bool(terminated)
            # Heuristique plus propre par environnement :
            # CartPole : succès = truncated à 500 (tige tenue).
            # MountainCar : succès = terminated (drapeau).
            break

        state = next_state
        if isinstance(agent, SarsaAgent) and learn and not terminated:
            action = next_action  # type: ignore[assignment]
        else:
            action = agent.select_action(next_state, greedy=greedy)

    return total, length, bool(terminated or truncated)


def success_of_episode(env_id: str, ret: float, length: int, terminated: bool, spec_max: int) -> bool:
    if env_id == "CartPole-v1":
        return length >= spec_max
    if env_id == "MountainCar-v0":
        # Succès = atteinte du drapeau (terminated). La troncature à 200 pas n'en est pas un.
        return bool(terminated)
    return False


def train(
    env_id: str,
    algo: str,
    n_bins: int,
    seed: int,
    n_episodes: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_min: float,
    decay_episodes: int | None = None,
    q_init: float = 0.0,
    eval_every: int = 0,
    verbose: bool = True,
) -> dict:
    spec = get_spec(env_id)
    decay_episodes = decay_episodes if decay_episodes is not None else n_episodes
    discretizer = UniformDiscretizer(spec.obs_low, spec.obs_high, n_bins)
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    agent = make_agent(
        algo, discretizer.n_states, spec.n_actions, alpha, gamma, eps_start, q_init, seed
    )

    returns = np.empty(n_episodes, dtype=np.float64)
    lengths = np.empty(n_episodes, dtype=np.int32)
    successes = np.empty(n_episodes, dtype=np.int8)
    first_success = None

    for ep in range(n_episodes):
        agent.epsilon = epsilon_at(ep, eps_start, eps_min, decay_episodes)
        obs, _ = env.reset()
        state = discretizer.discretize(obs)
        action = agent.select_action(state, greedy=False)
        total = 0.0
        length = 0
        terminated = False
        truncated = False

        while True:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            length += 1
            next_state = discretizer.discretize(next_obs)

            if isinstance(agent, SarsaAgent):
                if terminated:
                    agent.update(state, action, float(reward), next_state, True, 0)
                    next_action = action
                else:
                    next_action = agent.select_action(next_state, greedy=False)
                    agent.update(state, action, float(reward), next_state, False, next_action)
            else:
                agent.update(state, action, float(reward), next_state, bool(terminated), None)
                next_action = None

            if terminated or truncated:
                break
            state = next_state
            if isinstance(agent, SarsaAgent):
                action = next_action  # type: ignore[assignment]
            else:
                action = agent.select_action(next_state, greedy=False)

        returns[ep] = total
        lengths[ep] = length
        ok = success_of_episode(env_id, total, length, terminated, spec.max_episode_steps)
        successes[ep] = int(ok)
        if ok and first_success is None:
            first_success = ep + 1

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            window = returns[max(0, ep - 99) : ep + 1]
            print(
                f"[{env_id} {algo} bins={n_bins} seed={seed}] "
                f"ep {ep+1}/{n_episodes}  "
                f"R_ma100={window.mean():.1f}  "
                f"eps={agent.epsilon:.3f}  "
                f"visites={agent.visit_fraction()*100:.2f}%"
            )

    env.close()
    return {
        "env_id": env_id,
        "algo": algo,
        "n_bins": int(n_bins),
        "seed": int(seed),
        "n_episodes": int(n_episodes),
        "alpha": alpha,
        "gamma": gamma,
        "eps_start": eps_start,
        "eps_min": eps_min,
        "decay_episodes": decay_episodes,
        "q_init": q_init,
        "n_states": discretizer.n_states,
        "n_actions": spec.n_actions,
        "n_visited": agent.n_visited_states(),
        "visit_fraction": agent.visit_fraction(),
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
        "first_success_episode": first_success,
        "discretizer": discretizer.to_dict(),
        "Q": agent.Q,
        "visit_count": agent.visit_count,
    }


def evaluate_greedy(
    env_id: str,
    Q: np.ndarray,
    discretizer: UniformDiscretizer,
    n_episodes: int = 100,
    seed: int = 0,
) -> dict:
    """Évaluation strictement gloutonne, sans mise à jour (protocole du sujet)."""
    spec = get_spec(env_id)
    env = gym.make(env_id)
    env.reset(seed=seed + 10_000)
    rng = np.random.default_rng(seed + 10_000)

    returns = []
    lengths = []
    n_success = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        length = 0
        terminated = False
        truncated = False
        while True:
            state = discretizer.discretize(obs)
            q_row = Q[state]
            max_q = q_row.max()
            action = int(rng.choice(np.flatnonzero(q_row == max_q)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            length += 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(length)
        if success_of_episode(env_id, total, length, terminated, spec.max_episode_steps):
            n_success += 1
    env.close()
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "eval_n": n_episodes,
        "eval_mean": float(arr.mean()),
        "eval_std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "eval_min": float(arr.min()),
        "eval_max": float(arr.max()),
        "eval_success_rate": n_success / float(n_episodes),
        "eval_mean_length": float(np.mean(lengths)),
        "eval_returns": arr,
    }


def save_model(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Q=result["Q"],
        visit_count=result["visit_count"],
        env_id=result["env_id"],
        algo=result["algo"],
        n_bins=result["n_bins"],
        seed=result["seed"],
        alpha=result["alpha"],
        gamma=result["gamma"],
        low=np.asarray(result["discretizer"]["low"]),
        high=np.asarray(result["discretizer"]["high"]),
        bins=np.asarray(result["discretizer"]["n_bins"]),
    )
    # Copie .npy de la seule Q-table, exigée par le sujet.
    np.save(path.with_suffix(".npy"), result["Q"])


def save_metrics(path: Path, result: dict, eval_stats: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k not in ("Q", "visit_count")}
    payload["returns"] = result["returns"].tolist()
    payload["lengths"] = result["lengths"].tolist()
    payload["successes"] = result["successes"].tolist()
    if eval_stats is not None:
        payload["eval"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in eval_stats.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement Q-learning / SARSA sur état discrétisé")
    p.add_argument("--env", default="CartPole-v1", choices=["CartPole-v1", "MountainCar-v0"])
    p.add_argument("--algo", default="qlearning", choices=["qlearning", "sarsa"])
    p.add_argument("--bins", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-min", type=float, default=0.02)
    p.add_argument("--q-init", type=float, default=0.0)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--decay-episodes", type=int, default=0, help="0 = égal au nombre d'épisodes")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    decay = args.decay_episodes or None
    result = train(
        env_id=args.env,
        algo=args.algo,
        n_bins=args.bins,
        seed=args.seed,
        n_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        decay_episodes=decay,
        q_init=args.q_init,
        verbose=True,
    )
    discretizer = UniformDiscretizer.from_dict(result["discretizer"])
    eval_stats = evaluate_greedy(args.env, result["Q"], discretizer, args.eval_episodes, args.seed)
    print(
        f"Évaluation gloutonne ({args.eval_episodes} ép.) : "
        f"R={eval_stats['eval_mean']:.1f} ± {eval_stats['eval_std']:.1f}  "
        f"succès={eval_stats['eval_success_rate']*100:.1f}%"
    )

    stem = args.out or f"models/{args.env.split('-')[0].lower()}_{args.algo}_bins{args.bins}_seed{args.seed}"
    save_model(Path(stem + ".npz"), result)
    save_metrics(Path("results") / (Path(stem).name + "_metrics.json"), result, eval_stats)
    print(f"Modèle sauvegardé : {stem}.npz")


if __name__ == "__main__":
    main()
