"""Écran unique de démonstration : tous les scénarios, sans réentraînement.

Fenêtre Tkinter (Gymnasium utilise pygame uniquement hors écran).
Ne remplace pas ``python play.py --model ...`` (livrable du sujet).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk

import gymnasium as gym
import numpy as np

from src.discretizer import UniformDiscretizer
from src.env_config import get_spec
from src.train import success_of_episode

ROOT = Path(__file__).resolve().parent.parent
NAVY = "#121e3a"
PANEL = "#182a4a"
ACCENT = "#2e7cd6"
GREEN = "#2ea86e"
TEXT = "#f0f4fa"
MUTED = "#a0b0c4"
BTN = "#26385c"
BTN_ACTIVE_CP = "#2e7cd6"
BTN_ACTIVE_MC = "#2ea86e"

VIEW_W, VIEW_H = 720, 480
TICK_MS = 20  # ~50 FPS


def _patch_gym_close() -> None:
    """Gymnasium appelle pygame.quit() dans close(), ce qui cassait la fenêtre."""

    def _soft_close(self) -> None:
        self.screen = None
        self.clock = None
        if hasattr(self, "surf"):
            self.surf = None
        self.isopen = False

    import gymnasium.envs.classic_control.cartpole as cartpole
    import gymnasium.envs.classic_control.mountain_car as mountain_car

    cartpole.CartPoleEnv.close = _soft_close
    mountain_car.MountainCarEnv.close = _soft_close


@dataclass
class Scenario:
    label: str
    env_id: str
    model_path: Path | None
    group: str


def _existing(*rel_paths: str) -> Path | None:
    for rel in rel_paths:
        path = ROOT / rel
        if path.exists():
            return path
    return None


def all_scenarios() -> list[Scenario]:
    return [
        Scenario("Aléatoire", "CartPole-v1", None, "CartPole"),
        Scenario(
            "Q-learning",
            "CartPole-v1",
            _existing(
                "models/cartpole_qlearning_bins12_seed2.npz",
                "models/cartpole_qlearning_bins12_seed0.npz",
            ),
            "CartPole",
        ),
        Scenario(
            "SARSA (meilleur)",
            "CartPole-v1",
            _existing("models/cartpole_best.npz", "models/cartpole_sarsa_bins12_seed3.npz"),
            "CartPole",
        ),
        Scenario("Aléatoire", "MountainCar-v0", None, "MountainCar"),
        Scenario(
            "Q-learning",
            "MountainCar-v0",
            _existing(
                "models/mountaincar_qlearning_bins24_seed0.npz",
                "models/mountaincar_qlearning_bins12_seed0.npz",
            ),
            "MountainCar",
        ),
        Scenario(
            "SARSA (meilleur)",
            "MountainCar-v0",
            _existing("models/mountaincar_best.npz", "models/mountaincar_sarsa_bins24_seed3.npz"),
            "MountainCar",
        ),
    ]


class Player:
    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.Q: np.ndarray | None = None
        self.discretizer: UniformDiscretizer | None = None
        self.algo = "aléatoire"
        if scenario.model_path is not None:
            data = np.load(scenario.model_path)
            self.Q = data["Q"]
            self.discretizer = UniformDiscretizer(data["low"], data["high"], data["bins"])
            self.algo = str(data["algo"])
        self.spec = get_spec(scenario.env_id)
        self.env = gym.make(scenario.env_id, render_mode="rgb_array")
        self.episode = 0
        self.reset()

    def close(self) -> None:
        self.env.close()

    def reset(self) -> None:
        self.obs, _ = self.env.reset(seed=self.seed + self.episode)
        self.total = 0.0
        self.length = 0
        self.done = False
        self.success = False
        self.episode += 1
        self.end_pause = 0

    def act(self) -> int:
        if self.Q is None or self.discretizer is None:
            return int(self.env.action_space.sample())
        state = self.discretizer.discretize(self.obs)
        q_row = self.Q[state]
        return int(self.rng.choice(np.flatnonzero(q_row == q_row.max())))

    def step(self) -> None:
        if self.done:
            self.end_pause += 1
            if self.end_pause > 40:
                self.reset()
            return
        action = self.act()
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.total += float(reward)
        self.length += 1
        if terminated or truncated:
            self.done = True
            self.success = success_of_episode(
                self.scenario.env_id, self.total, self.length, terminated, self.spec.max_episode_steps
            )

    def frame(self) -> np.ndarray:
        raw = self.env.render()
        return np.ascontiguousarray(raw)


def _to_photo(frame: np.ndarray, width: int, height: int) -> tk.PhotoImage:
    """Convertit une frame RGB en PhotoImage Tk (PPM), redimensionnée."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    ys = (np.linspace(0, h - 1, nh)).astype(np.int32)
    xs = (np.linspace(0, w - 1, nw)).astype(np.int32)
    resized = frame[ys][:, xs]
    header = f"P6 {nw} {nh} 255\n".encode("ascii")
    return tk.PhotoImage(data=header + resized.tobytes(), format="PPM")


class DemoApp:
    def __init__(self, seed: int = 0) -> None:
        _patch_gym_close()
        self.seed = seed
        self.scenarios = all_scenarios()
        self.current = 2 if self.scenarios[2].model_path else 0
        self.paused = False
        self.player = Player(self.scenarios[self.current], seed=seed)
        self.buttons: list[tk.Button] = []

        self.root = tk.Tk()
        self.root.title("Projet 6 — CartPole & MountainCar")
        self.root.configure(bg=NAVY)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        header = tk.Frame(self.root, bg=PANEL, padx=16, pady=12)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Projet 6  ·  discrétisation Q-learning / SARSA",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Cliquez un scénario.  Espace = pause   R = nouvel épisode   Esc = quitter",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 8))

        grid = tk.Frame(header, bg=PANEL)
        grid.pack(fill="x")
        tk.Label(grid, text="CartPole", bg=PANEL, fg=ACCENT, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=3
        )
        tk.Label(grid, text="MountainCar", bg=PANEL, fg=GREEN, font=("Segoe UI", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=3
        )
        for i, sc in enumerate(self.scenarios):
            row, col = divmod(i, 3)
            btn = tk.Button(
                grid,
                text=sc.label,
                font=("Segoe UI", 9),
                relief="flat",
                cursor="hand2",
                command=lambda idx=i: self.select(idx),
                width=18,
            )
            btn.grid(row=row, column=col + 1, padx=4, pady=3)
            self.buttons.append(btn)
        self._refresh_buttons()

        box = tk.Frame(self.root, bg="#0a101c", width=VIEW_W, height=VIEW_H)
        box.pack_propagate(False)
        box.pack(padx=16, pady=(8, 4))
        self.view = tk.Label(box, bg="#0a101c")
        self.view.pack(expand=True)
        self.photo: tk.PhotoImage | None = None

        self.status = tk.Label(
            self.root, bg=NAVY, fg=TEXT, font=("Consolas", 10), anchor="w", padx=16, pady=8
        )
        self.status.pack(fill="x")

        self.root.bind("<space>", lambda _e: self._toggle_pause())
        self.root.bind("<r>", lambda _e: self.player.reset())
        self.root.bind("<R>", lambda _e: self.player.reset())
        self.root.bind("<Escape>", lambda _e: self._on_close())
        for n in range(6):
            self.root.bind(str(n + 1), lambda e, idx=n: self.select(idx))

        self.root.after(TICK_MS, self._tick)

    def _refresh_buttons(self) -> None:
        for i, btn in enumerate(self.buttons):
            sc = self.scenarios[i]
            missing = sc.label != "Aléatoire" and sc.model_path is None
            if missing:
                btn.configure(state="disabled", bg="#3a2a2a", fg="#c08080")
            elif i == self.current:
                bg = BTN_ACTIVE_CP if sc.group == "CartPole" else BTN_ACTIVE_MC
                btn.configure(state="normal", bg=bg, fg="white", activebackground=bg)
            else:
                btn.configure(state="normal", bg=BTN, fg=TEXT, activebackground="#345078")

    def select(self, idx: int) -> None:
        sc = self.scenarios[idx]
        if sc.label != "Aléatoire" and sc.model_path is None:
            return
        self.player.close()
        self.current = idx
        self.player = Player(sc, seed=self.seed)
        self.paused = False
        self._refresh_buttons()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _tick(self) -> None:
        if not self.paused:
            self.player.step()
        frame = self.player.frame()
        self.photo = _to_photo(frame, VIEW_W, VIEW_H)
        self.view.configure(image=self.photo)

        extra = ""
        if self.player.done:
            extra = "   ·   SUCCÈS" if self.player.success else "   ·   fin d'épisode"
        if self.paused:
            extra += "   ·   PAUSE"
        self.status.configure(
            text=(
                f"{self.player.scenario.env_id}  |  {self.player.scenario.label} "
                f"({self.player.algo})  |  épisode {self.player.episode}  "
                f"t={self.player.length}  R={self.player.total:.0f}{extra}"
            )
        )
        self.root.after(TICK_MS, self._tick)

    def _on_close(self) -> None:
        try:
            self.player.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_gui(seed: int = 0) -> None:
    DemoApp(seed=seed).run()
