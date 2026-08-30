# Projet 6 — Discrétisation d'espaces continus

**Q-learning et SARSA tabulaires sur CartPole-v1 et MountainCar-v0**

Université Aube Nouvelle — Master 2 Intelligence Artificielle — 2025-2026  
Module : Apprentissage par renforcement (Dr SOMDA Augustin)  
Groupe : **KOMI Mouhamed Ramadan**, **TAKOANO Elie**, **COULIBALY Moussa**

---

## 1. Installation

Python 3.10 ou supérieur.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Arborescence

```
src/                 # code (discrétisation, Q-learning, SARSA, expériences)
play.py              # rejeu sans réentraînement
models/              # Q-tables (.npz pour rejouer, .npy des meilleurs agents)
results/             # métriques JSON + tableau récapitulatif
figures/             # courbes et captures
videos/              # démonstration mp4
rapport/             # rapport DOCX
```

## 3. Reproduire les expériences du rapport

Protocole complet (3 grilles × 5 graines × Q-learning, puis SARSA sur la meilleure grille) :

```bash
python -m src.experiments --workers 6
python -m src.plots
python -m src.record_demo
```

Une expérience isolée (une commande par courbe du rapport) :

```bash
python -m src.train --env CartPole-v1 --algo qlearning --bins 12 --seed 0 --episodes 4000
python -m src.train --env CartPole-v1 --algo sarsa     --bins 12 --seed 0 --episodes 4000
python -m src.train --env MountainCar-v0 --algo qlearning --bins 12 --seed 0 --episodes 5000 --alpha 0.3
python -m src.train --env MountainCar-v0 --algo sarsa     --bins 12 --seed 0 --episodes 5000 --alpha 0.3
```

Graines utilisées : `0 1 2 3 4`. Grilles : `6, 12, 24` intervalles par dimension.

## 4. Rejouer un agent (sans réentraînement)

```bash
python play.py --model models/cartpole_best.npz
python play.py --model models/mountaincar_best.npz --episodes 5
python play.py --model models/cartpole_best.npz --no-render --episodes 20
```

Les fichiers `models/*.npz` contiennent la Q-table et les bornes de discrétisation.  
`models/cartpole_best.npy` et `models/mountaincar_best.npy` sont les Q-tables des meilleurs agents (exigence du sujet).

## 5. Vidéo

Après les expériences :

```bash
python -m src.record_demo
```

Fichier produit : `videos/demo_projet6.mp4` (agent aléatoire, agent entraîné, courbes).

## 6. Rapport

- `rapport/Rapport_Projet6_Discretisation_QLearning.docx`
- `rapport/Rapport_Projet6_Discretisation_QLearning.pdf` (16 pages, structure imposée)

## 7. Git

Le sujet exige un dépôt Git (contributions des trois membres) et l'adresse du dépôt au rendu.

```bash
git add README.md requirements.txt play.py src models figures videos rapport
git commit -m "Projet 6 : Q-learning discrétisé CartPole et MountainCar"
```

Ne versionnez pas `.venv/` (déjà dans `.gitignore`).

## 8. Reproductibilité

- Aucune bibliothèque RL (Stable-Baselines, etc.).
- Q-learning et SARSA en NumPy pur.
- Graines fixées ; l'évaluation gloutonne (100 épisodes) est **séparée** de l'entraînement.
- Distinction `terminated` (vrai terminal, pas de bootstrap) / `truncated` (horizon, bootstrap conservé).
