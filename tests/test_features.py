import sqlite3

import numpy as np
import pandas as pd

from scoutsignal import config as C
from scoutsignal import warehouse as W
from scoutsignal.features import pbp_features, tracking_features


def _events(rows):
    cols = ["event_id", "game_id", "season", "period", "sort_order", "game_seconds", "event_type",
            "team_id", "x", "y", "shooter_id", "goalie_id", "shot_type", "strength", "is_goal"]
    return pd.DataFrame(rows, columns=cols)


def test_rebound_and_rush_from_sql():
    conn = sqlite3.connect(":memory:")
    conn.executescript(W.SCHEMA)
    _events([
        (1, 1, 2024, 1, 0, 100, "shot-on-goal", 10, 60, 5, 1, 9, "wrist", "EV", 0),
        (2, 1, 2024, 1, 1, 102, "shot-on-goal", 10, 85, 2, 2, 9, "tip-in", "EV", 0),  # rebound
        (3, 1, 2024, 1, 2, 200, "takeaway", 20, 70, 0, None, None, None, "EV", 0),   # team 20 in its O-zone
        (4, 1, 2024, 1, 3, 203, "goal", 10, 75, -3, 1, 9, "snap", "EV", 1),          # = -70 for team 10: rush
        (5, 1, 2024, 2, 4, 1201, "missed-shot", 10, 80, 0, 1, 9, "wrist", "EV", 0),  # new period
    ]).assign(source="test").to_sql("events", conn, if_exists="append", index=False)
    f = pbp_features(conn).set_index("event_id")
    assert f.is_rebound.to_dict() == {1: 0, 2: 1, 4: 0, 5: 0}
    assert f.is_rush.to_dict() == {1: 0, 2: 0, 4: 1, 5: 0}
    assert f.loc[5, "secs_since_last"] == 1200
    assert np.isclose(f.loc[2, "distance"], np.hypot(4, 2))


def test_tracking_features_royal_road_and_one_timer():
    n = C.N_FRAMES
    frames = np.zeros((1, n, len(C.ENTITIES), 2), np.float32)
    frames[0, :, 0] = [70, -10]                                   # shooter stands still
    frames[0, :, 1, 0] = 70
    frames[0, :, 1, 1] = np.linspace(20, -10, n)                  # puck crosses the slot
    frames[0, -1, 1] = [70, -10]                                  # arrives on release
    frames[0, :, 3] = [72, -12]                                   # defender
    frames[0, :, 4] = [85, 0]                                     # goalie still centred
    f = tracking_features(frames).iloc[0]
    assert f.royal_road == 1
    assert f.possession_time == 0.1
    assert np.isclose(f.def_dist, np.hypot(2, 2), atol=1e-5)
    assert f.goalie_err > 1  # puck at y=-10 means the goalie should have shifted
