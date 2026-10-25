"""Paths and constants shared by the pipeline, the models and the API."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get("SCOUTSIGNAL_ARTIFACTS", ROOT / "artifacts"))
DB_PATH = ARTIFACTS / "warehouse.sqlite"
TRACKING_PATH = ARTIFACTS / "tracking.npz"

# Rink geometry in NHL API units (feet). Every shot is stored in "attacking"
# coordinates: the shooting team attacks the net at (+NET_X, 0).
NET_X = 89.0
BLUE_LINE_X = 25.0

# Synthetic league.
SEED = 7
SEASONS = (2021, 2022, 2023, 2024, 2025)
RULE_CHANGE_SEASON = 2024  # smaller goalie equipment: shooting % jumps league-wide
RULE_CHANGE_LOGIT = 0.30
N_TEAMS = 32
SKATERS_PER_TEAM = 18
GAMES_PER_SEASON = 600
SHOTS_PER_GAME = 45
DRAFT_CLASSES = tuple(range(2015, 2025))
PROSPECTS_PER_CLASS = 150
LAST_OBSERVED_SEASON = 2025
MAX_DEV_YEARS = 8

# Tracking: 1.5 s before release at 10 Hz. Entity order in the frames array.
HZ = 10
N_FRAMES = 15
ENTITIES = ("shooter", "puck", "passer", "defender", "goalie")

SHOT_TYPES = ("wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around")
SKILLS = ("skating", "shooting", "passing", "hockey_sense", "compete", "physicality", "defense")
