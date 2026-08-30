"""Protocole expérimental multi-graines du Projet 6.

Lance :
  - Q-learning pour 3 granularités (6, 12, 24) × 5 graines × 2 environnements
  - SARSA sur la meilleure grille de chaque environnement × 5 graines

Usage :
    python -m src.experiments
    python -m src.experiments --quick   # fumée (1 graine, peu d'épisodes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Permet d'exécuter le fichier comme script OU comme module.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.discretizer import UniformDiscretizer
from src.train import evaluate_greedy, save_metrics, save_model, train

FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

SEEDS = [0, 1, 2, 3, 4]
BINS_LIST = [6, 12, 24]

# Budgets d'épisodes : plus de cases ⇒ plus d'échantillons nécessaires.
EPISODES = {
    ("CartPole-v1", 6): 2500,
    ("CartPole-v1", 12): 4000,
    ("CartPole-v1", 24): 6000,
    ("MountainCar-v0", 6): 4000,
    ("MountainCar-v0", 12): 5000,
    ("MountainCar-v0", 24): 8000,
}

ALPHAS = {
    "CartPole-v1": 0.15,
    "MountainCar-v0": 0.30,
}

GAMMAS = {
    "CartPole-v1": 0.99,
    "MountainCar-v0": 0.99,
}


def job_key(env_id: str, algo: str, n_bins: int, seed: int) -> str:
    short = env_id.split("-")[0].lower()
    return f"{short}_{algo}_bins{n_bins}_seed{seed}"


def run_one(payload: dict) -> dict:
    """Worker processus : entraîne, évalue, sauvegarde, retourne les métriques légères."""
    result = train(
        env_id=payload["env_id"],
        algo=payload["algo"],
        n_bins=payload["n_bins"],
        seed=payload["seed"],
        n_episodes=payload["n_episodes"],
        alpha=payload["alpha"],
        gamma=payload["gamma"],
        eps_start=payload["eps_start"],
        eps_min=payload["eps_min"],
        decay_episodes=payload.get("decay_episodes"),
        q_init=payload["q_init"],
        verbose=payload.get("verbose", False),
    )
    discretizer = UniformDiscretizer.from_dict(result["discretizer"])
    eval_stats = evaluate_greedy(
        payload["env_id"], result["Q"], discretizer, n_episodes=100, seed=payload["seed"]
    )
    key = job_key(payload["env_id"], payload["algo"], payload["n_bins"], payload["seed"])
    save_model(MODELS / f"{key}.npz", result)
    save_metrics(RESULTS / f"{key}_metrics.json", result, eval_stats)

    returns = result["returns"]
    window = 100 if len(returns) >= 100 else max(1, len(returns) // 10)
    ma = np.convolve(returns, np.ones(window) / window, mode="valid")
    light = {
        "key": key,
        "env_id": result["env_id"],
        "algo": result["algo"],
        "n_bins": result["n_bins"],
        "seed": result["seed"],
        "n_states": result["n_states"],
        "n_visited": result["n_visited"],
        "visit_fraction": result["visit_fraction"],
        "first_success_episode": result["first_success_episode"],
        "n_episodes": result["n_episodes"],
        "returns": returns.tolist(),
        "eval": {k: v for k, v in eval_stats.items() if k != "eval_returns"},
        "train_ma_final": float(ma[-1]) if len(ma) else float(returns[-1]),
    }
    return light


def build_jobs(quick: bool) -> list[dict]:
    jobs = []
    seeds = [0] if quick else SEEDS
    bins_list = [8] if quick else BINS_LIST
    scale = 0.15 if quick else 1.0

    for env_id in ("CartPole-v1", "MountainCar-v0"):
        for n_bins in bins_list:
            n_ep = 300 if quick else EPISODES[(env_id, n_bins)]
            n_ep = max(50, int(n_ep * scale)) if quick else n_ep
            for seed in seeds:
                jobs.append(
                    {
                        "env_id": env_id,
                        "algo": "qlearning",
                        "n_bins": n_bins,
                        "seed": seed,
                        "n_episodes": n_ep,
                        "alpha": ALPHAS[env_id],
                        "gamma": GAMMAS[env_id],
                        "eps_start": 1.0,
                        "eps_min": 0.05 if env_id.startswith("Mountain") else 0.02,
                        "q_init": 0.0,
                        "verbose": False,
                        "decay_episodes": max(1, int(0.5 * n_ep)),
                    }
                )
    return jobs


def pick_best_bins(summaries: list[dict], env_id: str, algo: str = "qlearning") -> int:
    """Meilleure grille = plus haute moyenne d'évaluation gloutonne."""
    by_bins: dict[int, list[float]] = {}
    for s in summaries:
        if s["env_id"] == env_id and s["algo"] == algo:
            by_bins.setdefault(s["n_bins"], []).append(s["eval"]["eval_mean"])
    means = {b: float(np.mean(v)) for b, v in by_bins.items()}
    return max(means, key=means.get)


def add_sarsa_jobs(summaries: list[dict], quick: bool) -> list[dict]:
    jobs = []
    seeds = [0] if quick else SEEDS
    for env_id in ("CartPole-v1", "MountainCar-v0"):
        best = 8 if quick else pick_best_bins(summaries, env_id)
        n_ep = 300 if quick else EPISODES[(env_id, best)]
        for seed in seeds:
            jobs.append(
                {
                    "env_id": env_id,
                    "algo": "sarsa",
                    "n_bins": best,
                    "seed": seed,
                    "n_episodes": n_ep,
                    "alpha": ALPHAS[env_id],
                    "gamma": GAMMAS[env_id],
                    "eps_start": 1.0,
                    "eps_min": 0.05 if env_id.startswith("Mountain") else 0.02,
                    "q_init": 0.0,
                    "verbose": False,
                    "decay_episodes": max(1, int(0.5 * n_ep)),
                }
            )
    return jobs


def execute_jobs(jobs: list[dict], workers: int) -> list[dict]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out: list[dict] = []
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {job_key(job['env_id'], job['algo'], job['n_bins'], job['seed'])}  ({job['n_episodes']} ép.)")
            out.append(run_one(job))
    else:
        print(f"Lancement de {len(jobs)} jobs sur {workers} processus…", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                job = futs[fut]
                try:
                    res = fut.result()
                    out.append(res)
                    print(
                        f"[{done}/{len(jobs)}] OK {res['key']}  "
                        f"eval={res['eval']['eval_mean']:.1f}  "
                        f"visites={res['visit_fraction']*100:.2f}%",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[{done}/{len(jobs)}] ÉCHEC {job}: {exc!r}")
                    raise
    print(f"Terminé en {(time.time()-t0)/60:.1f} min")
    return out


def persist_summary(summaries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-sarsa", action="store_true")
    args = p.parse_args()

    q_jobs = build_jobs(args.quick)
    q_sum = execute_jobs(q_jobs, args.workers)
    persist_summary(q_sum, RESULTS / "summary_qlearning.json")

    all_sum = list(q_sum)
    if not args.skip_sarsa:
        s_jobs = add_sarsa_jobs(q_sum, args.quick)
        s_sum = execute_jobs(s_jobs, args.workers)
        persist_summary(s_sum, RESULTS / "summary_sarsa.json")
        all_sum.extend(s_sum)

    persist_summary(all_sum, RESULTS / "summary_all.json")
    print("Résumé écrit dans results/summary_all.json")


if __name__ == "__main__":
    main()
