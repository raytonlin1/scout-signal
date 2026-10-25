"""Pull real games from the public NHL API into the warehouse schema.

Endpoint: https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play
Regular-season game ids are {season}02{nnnn}, e.g. 2024020001.

The API's homeTeamDefendingSide flag is not reliable for every play, so each team's
attacking direction is inferred per period from where its unblocked shots were taken.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from . import config as C

URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
SHOTS = ("shot-on-goal", "missed-shot", "goal")
SKIP = {"period-start", "period-end", "game-end", "shootout-complete", "game-official"}


def _clock(mmss: str) -> int:
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def _strength(code: str | None, is_home: bool) -> str:
    # situationCode digits: away goalie, away skaters, home skaters, home goalie.
    if not code or len(code) != 4:
        return "EV"
    away_g, away_s, home_s, home_g = map(int, code)
    own, opp, opp_goalie = (home_s, away_s, away_g) if is_home else (away_s, home_s, home_g)
    if opp_goalie == 0:
        return "EN"
    return "EV" if own == opp else "PP" if own > opp else "SH"


def parse_game(pbp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_id = int(pbp["id"])
    season = int(str(pbp["season"])[:4])
    home_id = pbp["homeTeam"]["id"]
    rows = []
    for play in pbp["plays"]:
        kind = play["typeDescKey"]
        period = play["periodDescriptor"]
        if kind in SKIP or period.get("periodType") == "SO":
            continue
        d = play.get("details", {})
        team = d.get("eventOwnerTeamId")
        rows.append(dict(
            event_id=game_id * 10_000 + int(play["eventId"]),
            game_id=game_id, season=season, period=int(period["number"]),
            sort_order=int(play.get("sortOrder", 0)),
            game_seconds=(int(period["number"]) - 1) * 1200 + _clock(play["timeInPeriod"]),
            event_type=kind, team_id=int(team) if team else 0,
            x=d.get("xCoord"), y=d.get("yCoord"),
            shooter_id=d.get("shootingPlayerId") or d.get("scoringPlayerId"),
            goalie_id=d.get("goalieInNetId"),
            shot_type=d.get("shotType") if kind in SHOTS else None,
            strength=_strength(play.get("situationCode"), team == home_id) if team else None,
            is_goal=int(kind == "goal"),
            home_defends=play.get("homeTeamDefendingSide"),
        ))
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev, pd.DataFrame(columns=["player_id", "name", "team_id", "position", "shoots"])

    # Attacking direction per (team, period): most unblocked shots come from the
    # offensive zone, so the median shot x tells you which net the team attacked.
    shots = ev[ev.event_type.isin(SHOTS) & ev.x.notna()]
    sign = np.sign(shots.groupby(["team_id", "period"]).x.median()).replace(0, 1)
    fallback = np.where(
        (ev.team_id == home_id) == (ev.home_defends == "left"), 1.0, -1.0)
    s = ev.set_index(["team_id", "period"]).index.map(lambda k: sign.get(k, np.nan))
    s = np.where(np.isnan(np.asarray(s, float)), fallback, s)
    ev["x"] = ev.x * s
    ev["y"] = ev.y * s
    ev = ev.drop(columns="home_defends")

    players = pd.DataFrame([dict(
        player_id=r["playerId"],
        name=f'{r["firstName"]["default"]} {r["lastName"]["default"]}',
        team_id=r["teamId"], position=r["positionCode"], shoots=None,
    ) for r in pbp.get("rosterSpots", [])])
    return ev, players


def fetch_season(season: int, max_games: int, cache: Path = C.ARTIFACTS / "nhl_raw",
                 pause: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or read cached) regular-season games and parse them."""
    cache.mkdir(parents=True, exist_ok=True)
    events, players = [], []
    with httpx.Client(timeout=20, headers={"User-Agent": "scoutsignal/0.1"}) as client:
        for n in range(1, max_games + 1):
            game_id = int(f"{season}02{n:04d}")
            path = cache / f"{game_id}.json"
            if not path.exists():
                r = client.get(URL.format(game_id=game_id))
                if r.status_code == 404:
                    break
                r.raise_for_status()
                path.write_text(r.text)
                time.sleep(pause)
            pbp = json.loads(path.read_text())
            if pbp.get("gameState") not in ("OFF", "FINAL"):
                continue
            e, p = parse_game(pbp)
            events.append(e)
            players.append(p)
    return pd.concat(events, ignore_index=True), pd.concat(players, ignore_index=True)
