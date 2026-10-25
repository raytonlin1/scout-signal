"""ScoutSignal API and dashboard.

    uvicorn app.main:app --port 7860
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from scoutsignal import config as C
from scoutsignal.extract import extract
from scoutsignal.models.development import report_features

STATIC = Path(__file__).parent / "static"


class Artifacts:
    def __init__(self, root: Path = C.ARTIFACTS):
        if not (root / "report.json").exists():
            raise RuntimeError(f"No artifacts in {root}. Run: python -m scoutsignal.pipeline all")
        self.report = json.loads((root / "report.json").read_text())
        self.brief = (root / "brief.md").read_text()
        self.xg = {fs: joblib.load(root / f"xg_{fs}.joblib")
                   for fs in ("pbp", "tracking") if (root / f"xg_{fs}.joblib").exists()}
        dev = root / "development.joblib"
        self.development = joblib.load(dev) if dev.exists() else None
        self.db_path = root / "warehouse.sqlite"

    def query(self, sql: str, params=()) -> pd.DataFrame:
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            return pd.read_sql_query(sql, conn, params=params)


@lru_cache
def artifacts() -> Artifacts:
    return Artifacts()


@asynccontextmanager
async def lifespan(_: FastAPI):
    artifacts()  # fail at startup, not on the first request
    yield


app = FastAPI(title="ScoutSignal", version="0.1.0", lifespan=lifespan,
              description="Scouting-report extraction, xG, finishing talent and prospect "
                          "development models for hockey player evaluation.")


class Shot(BaseModel):
    x: float = Field(..., ge=-100, le=100, description="Feet; the attacked net is at (89, 0)")
    y: float = Field(..., ge=-42.5, le=42.5)
    shot_type: str = "wrist"
    strength: str = Field("EV", pattern="^(EV|PP|SH)$")
    is_rebound: bool = False
    is_rush: bool = False
    secs_since_last: float = 10.0
    period: int = Field(1, ge=1, le=5)
    last_event_type: str = "faceoff"
    # Tracking features. Give all of them to use the tracking model.
    def_dist: float | None = Field(None, description="Nearest defender at release, ft")
    shooter_speed: float | None = None
    puck_dy_1s: float | None = Field(None, description="Puck lateral travel in the last 1 s, ft")
    royal_road: bool | None = Field(None, description="Puck crossed the middle of the slot in the last 1 s")
    goalie_err: float | None = Field(None, description="Goalie distance off the puck-to-net line, ft")
    possession_time: float | None = Field(None, description="Seconds on the shooter's stick")


class ReportIn(BaseModel):
    text: str = Field(..., min_length=10, max_length=10_000)
    draft_round: int = Field(4, ge=1, le=7)


@app.get("/api/health")
def health():
    a = artifacts()
    return {"status": "ok", "models": sorted(a.xg) + (["development"] if a.development else []),
            "data": a.report["data"]}


@app.post("/api/xg")
def expected_goals(shot: Shot):
    a = artifacts()
    tracking_fields = ["def_dist", "shooter_speed", "puck_dy_1s", "royal_road", "goalie_err", "possession_time"]
    has_tracking = all(getattr(shot, f) is not None for f in tracking_fields)
    fs = "tracking" if has_tracking and "tracking" in a.xg else "pbp"
    row = shot.model_dump()
    row["distance"] = float(np.hypot(C.NET_X - shot.x, shot.y))
    row["angle"] = float(abs(np.arctan2(shot.y, C.NET_X - shot.x)))
    df = pd.DataFrame([row])
    xg = float(a.xg[fs].predict(df)[0])
    return {"xg": round(xg, 4), "model": fs, "distance_ft": round(row["distance"], 1),
            "angle_deg": round(float(np.degrees(row["angle"])), 1)}


@app.post("/api/scouting/extract")
def scouting_extract(body: ReportIn):
    return extract(body.text).to_dict()


@app.post("/api/prospects/project")
def prospect_project(body: ReportIn):
    a = artifacts()
    if a.development is None:
        raise HTTPException(503, "Development model not trained (synthetic build needed)")
    parsed = extract(body.text)
    feats = report_features([body.text], [body.draft_round])
    curve = a.development.debut_curve(feats)[0]
    return {"extraction": parsed.to_dict(), "draft_round": body.draft_round,
            "p_debut_by_year": [round(float(p), 4) for p in curve],
            "p_debut_within_4_years": round(float(curve[3]), 4)}


@app.get("/api/finishing")
def finishing(limit: int = Query(25, ge=1, le=500), buy_low: bool = False,
              position: str | None = Query(None, pattern="^(F|D)$")):
    sql = "SELECT * FROM finishing WHERE 1=1"
    params: list = []
    if buy_low:
        sql += " AND buy_low = 1"
    if position:
        sql += " AND position = ?"
        params.append(position)
    sql += " ORDER BY theta_mean DESC LIMIT ?"
    params.append(limit)
    df = artifacts().query(sql, params)
    return {"alpha": artifacts().report["finishing"]["alpha"],
            "players": json.loads(df.to_json(orient="records"))}


@app.get("/api/players")
def players(q: str = Query("", max_length=50), limit: int = Query(20, le=100)):
    df = artifacts().query(
        "SELECT player_id, name, position, shots, goals, ROUND(xg, 1) AS xg, ROUND(theta_mean, 3) AS theta_mean "
        "FROM finishing WHERE name LIKE ? ORDER BY shots DESC LIMIT ?", (f"%{q}%", limit))
    return json.loads(df.to_json(orient="records"))


@app.get("/api/players/{player_id}/shots")
def player_shots(player_id: int, season: int | None = None):
    sql = ("SELECT e.season, e.x, e.y, e.shot_type, e.is_goal, ROUND(s.xg, 4) AS xg "
           "FROM events e JOIN shot_xg s USING (event_id) WHERE e.shooter_id = ?")
    params: list = [player_id]
    if season:
        sql += " AND e.season = ?"
        params.append(season)
    df = artifacts().query(sql, params)
    if df.empty:
        raise HTTPException(404, "No shots for that player")
    return json.loads(df.to_json(orient="records"))


@app.get("/api/teams")
def teams(season: int | None = None):
    a = artifacts()
    season = season or a.report["data"]["seasons"][-1]
    df = a.query("SELECT *, xgf / (xgf + xga) AS xg_share FROM team_xg WHERE season = ? "
                 "ORDER BY xg_share DESC", (season,))
    return json.loads(df.to_json(orient="records"))


@app.get("/api/report")
def report():
    return artifacts().report


@app.get("/api/brief", response_class=PlainTextResponse)
def brief():
    return artifacts().brief


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
