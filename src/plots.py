"""Figures du rapport : courbes moyenne ± écart-type, tableaux, occupation de table."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

ENV_LABEL = {"CartPole-v1": "CartPole-v1", "MountainCar-v0": "MountainCar-v0"}


def rebuild_summary_from_metrics() -> list[dict]:
    """Reconstruit summary_all.json depuis les JSON de métriques (après finetune)."""
    summaries = []
    for path in sorted(RESULTS.glob("*_metrics.json")):
        if "bins8" in path.name:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "key": path.stem.replace("_metrics", ""),
                "env_id": d["env_id"],
                "algo": d["algo"],
                "n_bins": d["n_bins"],
                "seed": d["seed"],
                "n_states": d["n_states"],
                "n_visited": d["n_visited"],
                "visit_fraction": d["visit_fraction"],
                "first_success_episode": d.get("first_success_episode"),
                "n_episodes": d["n_episodes"],
                "returns": d["returns"],
                "eval": {k: v for k, v in d.get("eval", {}).items() if k != "eval_returns"},
            }
        )
    (RESULTS / "summary_all.json").write_text(json.dumps(summaries), encoding="utf-8")
    return summaries


def load_all() -> list[dict]:
    return rebuild_summary_from_metrics()


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(x))
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def group_returns(summaries: list[dict], env_id: str, algo: str, n_bins: int) -> np.ndarray:
    """Matrice (n_seeds, n_episodes) des retours."""
    series = [
        np.asarray(s["returns"], dtype=np.float64)
        for s in summaries
        if s["env_id"] == env_id and s["algo"] == algo and s["n_bins"] == n_bins
    ]
    if not series:
        return np.empty((0, 0))
    L = min(len(a) for a in series)
    return np.stack([a[:L] for a in series], axis=0)


def plot_mean_std(ax, matrix: np.ndarray, window: int, label: str, color: str) -> None:
    if matrix.size == 0:
        return
    smoothed = np.stack([moving_average(row, window) for row in matrix], axis=0)
    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0, ddof=1) if smoothed.shape[0] > 1 else np.zeros_like(mean)
    x = np.arange(1, len(mean) + 1)
    ax.plot(x, mean, color=color, lw=1.8, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def fig_granularity(summaries: list[dict]) -> None:
    colors = {6: "#1f77b4", 12: "#ff7f0e", 24: "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, env_id, window in zip(
        axes, ("CartPole-v1", "MountainCar-v0"), (50, 100)
    ):
        for n_bins in (6, 12, 24, 8):
            mat = group_returns(summaries, env_id, "qlearning", n_bins)
            if mat.size == 0:
                continue
            plot_mean_std(ax, mat, window, f"{n_bins} bins / dim.", colors.get(n_bins, "#9467bd"))
        ax.set_title(ENV_LABEL[env_id])
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Retour (moyenne glissante)")
        ax.legend(frameon=True)
    fig.suptitle("Q-learning : effet de la granularité de discrétisation (moyenne ± écart-type)")
    fig.tight_layout()
    fig.savefig(FIGURES / "courbes_granularite.png", bbox_inches="tight")
    plt.close(fig)


def fig_q_vs_sarsa(summaries: list[dict]) -> None:
    colors = {"qlearning": "#1f77b4", "sarsa": "#d62728"}
    labels = {"qlearning": "Q-learning", "sarsa": "SARSA"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, env_id, window in zip(
        axes, ("CartPole-v1", "MountainCar-v0"), (50, 100)
    ):
        sarsa_bins = {s["n_bins"] for s in summaries if s["env_id"] == env_id and s["algo"] == "sarsa"}
        if not sarsa_bins:
            ax.set_title(f"{ENV_LABEL[env_id]} (pas de SARSA)")
            continue
        n_bins = next(iter(sarsa_bins))
        mats = {}
        min_len = None
        for algo in ("qlearning", "sarsa"):
            mats[algo] = group_returns(summaries, env_id, algo, n_bins)
            if mats[algo].size:
                L = mats[algo].shape[1]
                min_len = L if min_len is None else min(min_len, L)
        for algo in ("qlearning", "sarsa"):
            mat = mats[algo]
            if min_len and mat.size:
                mat = mat[:, :min_len]
            plot_mean_std(ax, mat, window, labels[algo], colors[algo])
        ax.set_title(f"{ENV_LABEL[env_id]}  —  {n_bins} bins / dim.")
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Retour (moyenne glissante)")
        ax.legend()
    fig.suptitle("Comparaison Q-learning / SARSA sur la meilleure discrétisation")
    fig.tight_layout()
    fig.savefig(FIGURES / "courbes_q_vs_sarsa.png", bbox_inches="tight")
    plt.close(fig)


def fig_eval_bars(summaries: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, env_id in zip(axes, ("CartPole-v1", "MountainCar-v0")):
        bins_vals = sorted({s["n_bins"] for s in summaries if s["env_id"] == env_id and s["algo"] == "qlearning"})
        means, stds = [], []
        for b in bins_vals:
            vals = [
                s["eval"]["eval_mean"]
                for s in summaries
                if s["env_id"] == env_id and s["algo"] == "qlearning" and s["n_bins"] == b
            ]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        x = np.arange(len(bins_vals))
        ax.bar(x, means, yerr=stds, capsize=4, color="#4c78a8", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bins_vals])
        ax.set_xlabel("Nombre d'intervalles par dimension")
        ax.set_ylabel("Retour d'évaluation gloutonne")
        ax.set_title(ENV_LABEL[env_id])
    fig.suptitle("Performance d'évaluation (100 épisodes gloutons) selon la grille")
    fig.tight_layout()
    fig.savefig(FIGURES / "eval_par_grille.png", bbox_inches="tight")
    plt.close(fig)


def fig_occupation(summaries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    markers = {"CartPole-v1": "o", "MountainCar-v0": "s"}
    colors = {"CartPole-v1": "#1f77b4", "MountainCar-v0": "#ff7f0e"}
    for env_id in ("CartPole-v1", "MountainCar-v0"):
        rows = defaultdict(list)
        for s in summaries:
            if s["env_id"] == env_id and s["algo"] == "qlearning":
                rows[s["n_bins"]].append((s["n_states"], s["visit_fraction"]))
        xs, ys = [], []
        for b in sorted(rows):
            n_states = rows[b][0][0]
            frac = np.mean([v[1] for v in rows[b]])
            xs.append(n_states)
            ys.append(100.0 * frac)
        ax.plot(
            xs,
            ys,
            marker=markers[env_id],
            color=colors[env_id],
            label=ENV_LABEL[env_id],
            lw=1.8,
        )
        for x, y, b in zip(xs, ys, sorted(rows)):
            ax.annotate(f"{b} bins", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Taille de la Q-table (nombre d'états discrets)")
    ax.set_ylabel("Fraction d'états visités (%)")
    ax.set_title("Malédiction de la dimension : occupation réelle de la table")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "occupation_table.png", bbox_inches="tight")
    plt.close(fig)


def write_tables(summaries: list[dict]) -> None:
    """Tableau récapitulatif CSV pour le rapport."""
    lines = [
        "env,algo,n_bins,n_states,visit_fraction_mean,eval_mean,eval_std,eval_success,first_success_median"
    ]
    groups = defaultdict(list)
    for s in summaries:
        groups[(s["env_id"], s["algo"], s["n_bins"])].append(s)
    for key in sorted(groups):
        rows = groups[key]
        env, algo, bins = key
        n_states = rows[0]["n_states"]
        vf = np.mean([r["visit_fraction"] for r in rows])
        em = np.mean([r["eval"]["eval_mean"] for r in rows])
        es = np.std([r["eval"]["eval_mean"] for r in rows], ddof=1) if len(rows) > 1 else 0.0
        sr = np.mean([r["eval"]["eval_success_rate"] for r in rows])
        firsts = [r["first_success_episode"] for r in rows if r["first_success_episode"] is not None]
        fs = float(np.median(firsts)) if firsts else float("nan")
        lines.append(
            f"{env},{algo},{bins},{n_states},{vf:.4f},{em:.2f},{es:.2f},{sr:.3f},{fs:.1f}"
        )
    (RESULTS / "tableau_recap.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summaries = load_all()
    fig_granularity(summaries)
    fig_q_vs_sarsa(summaries)
    fig_eval_bars(summaries)
    fig_occupation(summaries)
    write_tables(summaries)
    print(f"Figures écrites dans {FIGURES}")
    print(f"Tableau : {RESULTS / 'tableau_recap.csv'}")


if __name__ == "__main__":
    main()
