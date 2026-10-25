"""Shot features from two sources.

* Play-by-play: location, shot type and the event sequence before the shot. Public NHL
  data has all of this, so these features work on real games.
* Tracking: positions of the shooter, puck, passer, nearest defender and goalie in the
  1.5 s before release. Public NHL data does not have this; the simulator does.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import config as C

S, P, PA, D, G = range(len(C.ENTITIES))

PBP_NUMERIC = ["distance", "angle", "is_rebound", "is_rush", "secs_since_last", "period"]
PBP_CATEGORICAL = ["shot_type", "strength", "last_event_type"]
TRACKING_NUMERIC = [
    "def_dist", "shooter_speed", "puck_dy_1s", "royal_road", "goalie_err", "possession_time",
]

# Sequence features in SQL. LAG over the game's event order gives the event right before
# each shot; its location is flipped into the shooting team's frame when the other team
# owned it. Rebound: a saved shot by the same team <= 3 s earlier. Rush: the previous event
# was outside the offensive zone <= 4 s earlier.
PBP_FEATURES_SQL = """
WITH ordered AS (
    SELECT e.*,
           LAG(event_type)   OVER w AS prev_type,
           LAG(team_id)      OVER w AS prev_team,
           LAG(x)            OVER w AS prev_x,
           LAG(game_seconds) OVER w AS prev_seconds,
           LAG(period)       OVER w AS prev_period
    FROM events e
    WINDOW w AS (PARTITION BY game_id ORDER BY sort_order)
)
SELECT event_id, game_id, season, period, team_id, shooter_id, goalie_id,
       x, y, shot_type, strength, is_goal,
       SQRT((:net_x - x) * (:net_x - x) + y * y)                  AS distance,
       ABS(ATAN2(y, :net_x - x))                                    AS angle,
       COALESCE(prev_type, 'none')                                  AS last_event_type,
       CASE WHEN prev_period = period THEN game_seconds - prev_seconds ELSE 1200 END
                                                                    AS secs_since_last,
       CASE WHEN prev_type = 'shot-on-goal' AND prev_team = team_id
             AND prev_period = period AND game_seconds - prev_seconds <= 3
            THEN 1 ELSE 0 END                                       AS is_rebound,
       CASE WHEN prev_period = period AND game_seconds - prev_seconds <= 4
             AND (CASE WHEN prev_team = team_id THEN prev_x ELSE -prev_x END) < :blue_line
            THEN 1 ELSE 0 END                                       AS is_rush
FROM ordered
WHERE event_type IN ('shot-on-goal', 'missed-shot', 'goal')
ORDER BY event_id
"""


def pbp_features(conn: sqlite3.Connection) -> pd.DataFrame:
    conn.create_function("ATAN2", 2, lambda a, b: float(np.arctan2(a, b)), deterministic=True)
    conn.create_function("SQRT", 1, lambda v: float(np.sqrt(v)), deterministic=True)
    return pd.read_sql_query(
        PBP_FEATURES_SQL, conn, params={"net_x": C.NET_X, "blue_line": C.BLUE_LINE_X}
    )


def goalie_line_y(puck: np.ndarray, goalie_x: np.ndarray) -> np.ndarray:
    """Where the goalie should stand laterally: on the puck-to-net-centre line at goalie_x."""
    run = np.maximum(C.NET_X - puck[..., 0], 1.0)
    return puck[..., 1] * (C.NET_X - goalie_x) / run


def tracking_features(frames: np.ndarray) -> pd.DataFrame:
    """frames: (n_shots, N_FRAMES, n_entities, 2), last frame is the release."""
    shooter, puck, defender, goalie = frames[:, :, S], frames[:, :, P], frames[:, :, D], frames[:, :, G]
    one_second = C.HZ + 1

    def_dist = np.linalg.norm(shooter[:, -1] - defender[:, -1], axis=1)
    shooter_speed = np.linalg.norm(shooter[:, -1] - shooter[:, -4], axis=1) * C.HZ / 3
    recent_y = puck[:, -one_second:, 1]
    puck_dy_1s = np.abs(recent_y[:, -1] - recent_y[:, 0])
    royal_road = ((recent_y.min(axis=1) < -3) & (recent_y.max(axis=1) > 3)).astype(int)
    goalie_err = np.abs(goalie[:, -1, 1] - goalie_line_y(puck[:, -1], goalie[:, -1, 0]))

    # Time the shooter has had the puck: consecutive frames, counting back from release,
    # with the puck on the shooter's stick. Short possession = one-timer or quick catch.
    on_stick = np.linalg.norm(puck - shooter, axis=2) < 1.5
    broken = ~on_stick[:, ::-1]
    first_break = np.where(broken.any(axis=1), broken.argmax(axis=1), C.N_FRAMES)
    possession_time = first_break / C.HZ

    return pd.DataFrame({
        "def_dist": def_dist,
        "shooter_speed": shooter_speed,
        "puck_dy_1s": puck_dy_1s,
        "royal_road": royal_road,
        "goalie_err": goalie_err,
        "possession_time": possession_time,
    })
