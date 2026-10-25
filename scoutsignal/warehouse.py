"""SQLite warehouse: one schema for synthetic and real NHL data.

SQLite keeps the deployed image dependency-free; the SQL is plain enough to move to
Postgres or DuckDB unchanged (window functions, CTEs, no vendor extensions).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     INTEGER PRIMARY KEY,
    game_id      INTEGER NOT NULL,
    season       INTEGER NOT NULL,
    period       INTEGER NOT NULL,
    sort_order   INTEGER NOT NULL,
    game_seconds INTEGER NOT NULL,
    event_type   TEXT NOT NULL,
    team_id      INTEGER NOT NULL,
    x REAL, y REAL,                 -- attacking coordinates of the event's owner
    shooter_id   INTEGER,
    goalie_id    INTEGER,
    shot_type    TEXT,
    strength     TEXT,
    is_goal      INTEGER NOT NULL DEFAULT 0,
    source       TEXT NOT NULL DEFAULT 'synthetic'
);
CREATE INDEX IF NOT EXISTS ix_events_game ON events (game_id, sort_order);
CREATE INDEX IF NOT EXISTS ix_events_shooter ON events (shooter_id, season);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY, name TEXT, team_id INTEGER, position TEXT, shoots TEXT
);
CREATE TABLE IF NOT EXISTS shot_xg (
    event_id INTEGER PRIMARY KEY, model TEXT, xg REAL
);
CREATE TABLE IF NOT EXISTS prospects (
    prospect_id INTEGER PRIMARY KEY, name TEXT, draft_year INTEGER, draft_round INTEGER,
    position TEXT, league TEXT, dev_years_observed INTEGER, made_nhl INTEGER
);
CREATE TABLE IF NOT EXISTS scouting_reports (
    prospect_id INTEGER PRIMARY KEY, text TEXT
);
"""

# Shooter-season totals for the finishing model. Joins each shot to its xG and keeps
# shooters with a minimum shot count.
SHOOTER_SEASON_SQL = """
SELECT e.shooter_id AS player_id, p.name, p.position, e.season,
       COUNT(*)              AS shots,
       SUM(e.is_goal)        AS goals,
       SUM(x.xg)             AS xg
FROM events e
JOIN shot_xg x USING (event_id)
LEFT JOIN players p ON p.player_id = e.shooter_id
WHERE e.event_type IN ('shot-on-goal', 'missed-shot', 'goal')
GROUP BY e.shooter_id, e.season
HAVING COUNT(*) >= :min_shots
"""

# Team share of expected goals per season: the usual process metric for front offices.
TEAM_XG_SQL = """
WITH shot AS (
    SELECT e.season, e.game_id, e.team_id, x.xg, e.is_goal
    FROM events e JOIN shot_xg x USING (event_id)
), game_teams AS (
    SELECT DISTINCT season, game_id, team_id FROM events
)
SELECT g.season, g.team_id,
       SUM(CASE WHEN s.team_id = g.team_id THEN s.xg ELSE 0 END) AS xgf,
       SUM(CASE WHEN s.team_id <> g.team_id THEN s.xg ELSE 0 END) AS xga,
       SUM(CASE WHEN s.team_id = g.team_id THEN s.is_goal ELSE 0 END) AS gf,
       SUM(CASE WHEN s.team_id <> g.team_id THEN s.is_goal ELSE 0 END) AS ga
FROM game_teams g JOIN shot s ON s.season = g.season AND s.game_id = g.game_id
GROUP BY g.season, g.team_id
ORDER BY g.season, xgf * 1.0 / (xgf + xga) DESC
"""


def connect(path: Path = C.DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def reset(path: Path = C.DB_PATH) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    return connect(path)


def load_league(conn: sqlite3.Connection, league) -> None:
    league.events.assign(source="synthetic").to_sql("events", conn, if_exists="append", index=False)
    league.skaters[["player_id", "name", "team_id", "position", "shoots"]].to_sql(
        "players", conn, if_exists="append", index=False)
    league.prospects[["prospect_id", "name", "draft_year", "draft_round", "position", "league",
                      "dev_years_observed", "made_nhl"]].to_sql(
        "prospects", conn, if_exists="append", index=False)
    league.reports[["prospect_id", "text"]].to_sql(
        "scouting_reports", conn, if_exists="append", index=False)
    # Ground truth stays in its own tables so nothing downstream reads it by accident.
    league.shot_truth.to_sql("sim_shot_truth", conn, if_exists="replace", index=False)
    league.skaters[["player_id", "finishing"]].to_sql("sim_skater_truth", conn, if_exists="replace", index=False)
    league.prospects.to_sql("sim_prospect_truth", conn, if_exists="replace", index=False)
    league.reports[["prospect_id", "skills_mentioned"]].to_sql(
        "sim_report_truth", conn, if_exists="replace", index=False)
    conn.commit()


def load_nhl(conn: sqlite3.Connection, events: pd.DataFrame, players: pd.DataFrame) -> None:
    events.assign(source="nhl").to_sql("events", conn, if_exists="append", index=False)
    known = set(pd.read_sql_query("SELECT player_id FROM players", conn).player_id)
    players[~players.player_id.isin(known)].drop_duplicates("player_id").to_sql(
        "players", conn, if_exists="append", index=False)
    conn.commit()


def save_xg(conn: sqlite3.Connection, event_ids: np.ndarray, xg: np.ndarray, model: str) -> None:
    conn.execute("DELETE FROM shot_xg")
    conn.executemany("INSERT INTO shot_xg VALUES (?, ?, ?)",
                     zip(map(int, event_ids), [model] * len(xg), map(float, xg)))
    conn.commit()


def shooter_seasons(conn: sqlite3.Connection, min_shots: int = 1) -> pd.DataFrame:
    return pd.read_sql_query(SHOOTER_SEASON_SQL, conn, params={"min_shots": min_shots})


def team_xg(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(TEAM_XG_SQL, conn)


def save_tracking(frames: np.ndarray, event_ids: np.ndarray, path: Path = C.TRACKING_PATH) -> None:
    # float16 keeps positions to ~0.06 ft and halves the file.
    np.savez_compressed(path, frames=frames.astype(np.float16), event_ids=event_ids)


def load_tracking(path: Path = C.TRACKING_PATH):
    data = np.load(path)
    return data["frames"].astype(np.float32), data["event_ids"]
