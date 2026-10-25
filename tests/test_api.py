import pytest
from fastapi.testclient import TestClient

from scoutsignal import config as C

pytestmark = pytest.mark.skipif(not (C.ARTIFACTS / "report.json").exists(),
                                reason="run `python -m scoutsignal.pipeline all` first")


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and "pbp" in r["models"]


def test_xg_tracking_vs_pbp(client):
    base = {"x": 75, "y": 5, "shot_type": "wrist"}
    slot = client.post("/api/xg", json=base).json()
    point = client.post("/api/xg", json={**base, "x": 30, "y": 20}).json()
    assert slot["model"] == "pbp" and slot["xg"] > point["xg"]
    tracked = dict(base, def_dist=12, shooter_speed=10, puck_dy_1s=25, royal_road=True,
                   goalie_err=5, possession_time=0.1)
    covered = dict(tracked, def_dist=2, royal_road=False, goalie_err=0.2, possession_time=1.2, puck_dy_1s=1)
    a, b = client.post("/api/xg", json=tracked).json(), client.post("/api/xg", json=covered).json()
    assert a["model"] == "tracking" and a["xg"] > b["xg"]


def test_prospect_projection(client):
    text = "Kasper Lindqvist | LW | Shoots L | 6'1\" 187 lbs | SHL. Elite stride. Reads are excellent."
    good = client.post("/api/prospects/project", json={"text": text, "draft_round": 1}).json()
    weak = client.post("/api/prospects/project",
                       json={"text": text.replace("Elite", "Poor").replace("excellent", "poor"), "draft_round": 6}).json()
    assert good["extraction"]["skills"]["skating"]["grade"] == 80
    assert len(good["p_debut_by_year"]) == C.MAX_DEV_YEARS
    assert good["p_debut_within_4_years"] > weak["p_debut_within_4_years"]


def test_finishing_and_shots(client):
    players = client.get("/api/finishing?limit=5").json()["players"]
    assert len(players) == 5
    assert players[0]["theta_mean"] >= players[-1]["theta_mean"]
    shots = client.get(f"/api/players/{players[0]['player_id']}/shots").json()
    assert shots and {"x", "y", "xg", "is_goal"} <= shots[0].keys()


def test_dashboard_and_brief(client):
    assert "ScoutSignal" in client.get("/").text
    assert client.get("/api/brief").text.startswith("# ScoutSignal brief")
