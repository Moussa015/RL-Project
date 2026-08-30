"""Captures et vidéo de démonstration (agent aléatoire vs entraîné)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.discretizer import UniformDiscretizer
from src.env_config import get_spec

VIDEOS = ROOT / "videos"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "C:\\Windows\\Fonts\\arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def title_card(text: str, size: tuple[int, int], n_frames: int, fps: int) -> list[np.ndarray]:
    w, h = size
    img = Image.new("RGB", (w, h), (18, 28, 48))
    draw = ImageDraw.Draw(img)
    font = _font(28)
    small = _font(18)
    lines = text.split("\n")
    y = h // 2 - 20 * len(lines)
    for i, line in enumerate(lines):
        f = font if i == 0 else small
        bbox = draw.textbbox((0, 0), line, font=f)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, y), line, fill=(240, 240, 245), font=f)
        y += (bbox[3] - bbox[1]) + 10
    frame = np.asarray(img)
    return [frame] * max(1, int(n_frames))


def overlay_banner(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    draw.rectangle([(0, 0), (w, 36)], fill=(0, 0, 0, 150))
    draw.text((12, 8), text, fill=(255, 255, 255, 255), font=_font(16))
    return np.asarray(img.convert("RGB"))


def rollout_frames(
    env_id: str,
    Q: np.ndarray | None,
    discretizer: UniformDiscretizer | None,
    n_episodes: int,
    seed: int,
    banner: str,
    max_steps: int | None = None,
) -> list[np.ndarray]:
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    spec = get_spec(env_id)
    frames: list[np.ndarray] = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        for t in range(max_steps or spec.max_episode_steps):
            raw = env.render()
            frames.append(overlay_banner(raw, f"{banner}  |  ép. {ep+1}  t={t+1}  R={total:.0f}"))
            if Q is None:
                action = int(env.action_space.sample())
            else:
                state = discretizer.discretize(obs)
                q_row = Q[state]
                action = int(rng.choice(np.flatnonzero(q_row == q_row.max())))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                raw = env.render()
                frames.append(overlay_banner(raw, f"{banner}  |  FIN  R={total:.0f}"))
                break
    env.close()
    return frames


def save_png(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def curve_frames(summary_path: Path, size: tuple[int, int], n_frames: int) -> list[np.ndarray]:
    """Quelques secondes sur la courbe d'apprentissage (image statique)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_subplot(111)
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        shown = False
        for env_id, color in (("CartPole-v1", "#1f77b4"), ("MountainCar-v0", "#ff7f0e")):
            rows = [s for s in data if s["env_id"] == env_id and s["algo"] == "qlearning"]
            if not rows:
                continue
            # Meilleure moyenne d'évaluation
            best = max(rows, key=lambda s: s["eval"]["eval_mean"])
            r = np.asarray(best["returns"])
            win = min(100, max(5, len(r) // 20))
            ma = np.convolve(r, np.ones(win) / win, mode="valid")
            ax.plot(ma, color=color, lw=2, label=f"{env_id} ({best['n_bins']} bins)")
            shown = True
        if shown:
            ax.legend()
    ax.set_title("Courbes d'apprentissage (moyenne glissante)")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Retour")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    # Redimensionner exactement à size
    pil = Image.fromarray(img).resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(pil)
    return [arr] * n_frames


def pick_best_model(env_id: str) -> Path | None:
    short = env_id.split("-")[0].lower()
    preferred = MODELS / f"{short}_best.npz"
    if preferred.exists():
        return preferred
    cands = sorted(MODELS.glob(f"{env_id.split('-')[0].lower()}_qlearning_bins*_seed*.npz"))
    if not cands:
        return None
    # Préférer le modèle dont le nom contient le plus grand nombre de bins parmi ceux existants,
    # puis seed 0 — le rapport sélectionnera le meilleur via summary.
    summary = ROOT / "results" / "summary_all.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        rows = [s for s in data if s["env_id"] == env_id and s["algo"] == "qlearning"]
        if rows:
            best = max(rows, key=lambda s: s["eval"]["eval_mean"])
            path = MODELS / f"{best['key']}.npz"
            if path.exists():
                return path
    return cands[0]


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore

    # Harmoniser la taille
    h, w = frames[0].shape[:2]
    normed = []
    for f in frames:
        if f.shape[0] != h or f.shape[1] != w:
            f = np.asarray(Image.fromarray(f).resize((w, h), Image.Resampling.BILINEAR))
        normed.append(f)
    imageio.mimsave(path, normed, fps=fps)
    print(f"Vidéo : {path}  ({len(normed)/fps:.1f} s, {len(normed)} frames)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    VIDEOS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    all_frames: list[np.ndarray] = []
    canvas = None

    all_frames.extend(title_card(
        "Projet RL 6 — Q-learning discrétisé\nCartPole-v1 et MountainCar-v0\nKOMI · TAKOANO · COULIBALY  —  M2 IA 2025-2026",
        (600, 400),
        n_frames=args.fps * 6,
        fps=args.fps,
    ))

    for env_id, n_rand, n_trained in (("CartPole-v1", 2, 4), ("MountainCar-v0", 2, 8)):
        spec = get_spec(env_id)
        all_frames.extend(title_card(f"{env_id}\nAgent aléatoire", (600, 400), args.fps * 2, args.fps))
        rand_frames = rollout_frames(env_id, None, None, n_rand, seed=0, banner=f"{env_id}  ALÉATOIRE")
        if rand_frames:
            save_png(rand_frames[len(rand_frames)//2], FIGURES / f"screenshot_{env_id.split('-')[0].lower()}_random.png")
        all_frames.extend(rand_frames)

        model_path = pick_best_model(env_id)
        all_frames.extend(title_card(
            f"{env_id}\nAgent entraîné" + (f"\n{model_path.name}" if model_path else "\n(modèle manquant)"),
            (600, 400),
            args.fps * 2,
            args.fps,
        ))
        if model_path and model_path.exists():
            data = np.load(model_path)
            disc = UniformDiscretizer(data["low"], data["high"], data["bins"])
            trained = rollout_frames(
                env_id,
                data["Q"],
                disc,
                n_trained,
                seed=1,
                banner=f"{env_id}  ENTRAÎNÉ  ({data['algo']})",
            )
            if trained:
                save_png(trained[min(len(trained)//2, len(trained)-1)], FIGURES / f"screenshot_{env_id.split('-')[0].lower()}_trained.png")
            all_frames.extend(trained)
        else:
            print(f"Pas de modèle pour {env_id}, captures entraînées ignorées.")

    all_frames.extend(title_card("Courbes d'apprentissage", (600, 400), args.fps * 2, args.fps))
    all_frames.extend(curve_frames(ROOT / "results" / "summary_all.json", (600, 400), args.fps * 15))

    all_frames.extend(title_card(
        "Fin de la démonstration\nUniversité Aube Nouvelle — M2 IA",
        (600, 400),
        args.fps * 3,
        args.fps,
    ))

    write_mp4(all_frames, VIDEOS / "demo_projet6.mp4", fps=args.fps)


if __name__ == "__main__":
    main()
