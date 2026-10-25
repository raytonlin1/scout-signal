"""End-to-end pipeline: build the warehouse, train and audit every model, write artifacts.

    python -m scoutsignal.pipeline build                      # synthetic league (default)
    python -m scoutsignal.pipeline build --source nhl --season 2024 --max-games 300
    python -m scoutsignal.pipeline train                      # models, audit, brief
    python -m scoutsignal.pipeline all                        # build + train

Everything the API serves is written to artifacts/: the SQLite warehouse, fitted models
(joblib), report.json with every metric, and brief.md for front-office readers.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from . import config as C
from . import warehouse as W
from .backtest import recalibrate, rolling_origin, rule_change_audit
from .brief import write_brief
from .extract import evaluate as evaluate_extraction
from .features import TRACKING_NUMERIC, pbp_features, tracking_features
from .metrics import binary_report, concordance_index
from .models.development import BASELINE, DevelopmentModel, kaplan_meier, report_features
from .models.finishing import backtest_finishing, leaderboard
from .models.xg import XGModel, modelling_shots


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(source: str = "synthetic", season: int = 2024, max_games: int = 300) -> None:
    conn = W.reset()
    if source == "synthetic":
        from .simulate import simulate
        log("simulating league")
        league = simulate()
        W.load_league(conn, league)
        W.save_tracking(league.tracking, league.tracking_event_ids)
        log(f"{len(league.events):,} events, {len(league.tracking):,} tracked shots, "
            f"{len(league.reports):,} scouting reports")
    else:
        from .nhl_api import fetch_season
        log(f"fetching up to {max_games} NHL games from {season}-{season + 1}")
        events, players = fetch_season(season, max_games)
        W.load_nhl(conn, events, players)
        if C.TRACKING_PATH.exists():
            C.TRACKING_PATH.unlink()
        log(f"{len(events):,} events from {events.game_id.nunique()} games")


def load_shots(conn) -> pd.DataFrame:
    shots = pbp_features(conn)
    if C.TRACKING_PATH.exists():
        frames, ids = W.load_tracking()
        tf = tracking_features(frames).assign(event_id=ids)
        shots = shots.merge(tf, on="event_id", how="left")
    else:
        for c in TRACKING_NUMERIC:
            shots[c] = np.nan
    truth = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sim_shot_truth'", conn)
    if len(truth):
        shots = shots.merge(pd.read_sql_query("SELECT * FROM sim_shot_truth", conn),
                            on="event_id", how="left")
    return modelling_shots(shots)


def out_of_fold_xg(shots: pd.DataFrame, feature_set: str, folds: int = 5) -> np.ndarray:
    """xG for every shot from a model that never saw that shot's game, then shifted per
    season so league goals = league xG. Finishing talent is measured against that."""
    oof = np.zeros(len(shots))
    for tr, te in GroupKFold(folds).split(shots, groups=shots.game_id):
        oof[te] = XGModel(feature_set).fit(shots.iloc[tr]).predict(shots.iloc[te])
    for s in shots.season.unique():
        m = (shots.season == s).to_numpy()
        z = np.log(oof[m] / (1 - oof[m])) + recalibrate(oof[m], shots.is_goal.to_numpy()[m])
        oof[m] = 1 / (1 + np.exp(-z))
    return oof


def train_xg(conn, shots: pd.DataFrame, report: dict) -> str:
    has_tracking = shots[TRACKING_NUMERIC[0]].notna().all()
    best = "tracking" if has_tracking else "pbp"
    if "true_rebound" in shots:
        report["sequence_feature_agreement"] = {
            "is_rebound": round(float((shots.is_rebound == shots.true_rebound).mean()), 4),
            "is_rush": round(float((shots.is_rush == shots.true_rush).mean()), 4),
        }
    log("rolling-origin backtest")
    report["xg_backtest"] = rolling_origin(shots, ("pbp", "tracking") if has_tracking else ("pbp",))
    if shots.season.nunique() > 2 and C.RULE_CHANGE_SEASON in shots.season.values:
        log("rule-change audit")
        report["rule_change_audit"] = rule_change_audit(shots, feature_set=best)

    log("final xG models")
    latest = shots.season.max()
    for fs in (["pbp", "tracking"] if has_tracking else ["pbp"]):
        model = XGModel(fs).fit(shots)
        # Serve predictions on the current league scoring level.
        cur = shots[shots.season == latest]
        model.offset = recalibrate(model.predict(cur), cur.is_goal.to_numpy())
        joblib.dump(model, C.ARTIFACTS / f"xg_{fs}.joblib")

    log("out-of-fold xG for finishing")
    shots["xg"] = out_of_fold_xg(shots, best)
    W.save_xg(conn, shots.event_id.to_numpy(), shots.xg.to_numpy(), best)
    report["xg_model_for_finishing"] = best
    report["xg_oof"] = binary_report(shots.is_goal, shots.xg)
    return best


def train_finishing(conn, report: dict) -> None:
    log("finishing model")
    seasons = W.shooter_seasons(conn)
    latest = int(seasons.season.max())
    board = leaderboard(seasons, latest, min_shots=50 if seasons.shots.sum() > 20_000 else 15)
    board.to_sql("finishing", conn, if_exists="replace", index=False)
    report["finishing"] = {"alpha": round(board.attrs["alpha"], 1), "through_season": latest,
                           "n_shooters": len(board), "n_buy_low": int(board.buy_low.sum())}
    if seasons.season.nunique() > 1:
        report["finishing"]["backtest"] = backtest_finishing(seasons)
    truth = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE name='sim_skater_truth'", conn)
    if len(truth):
        t = pd.read_sql_query("SELECT * FROM sim_skater_truth", conn)
        b = board.merge(t, on="player_id")
        report["finishing"]["rank_corr_with_true_talent"] = {
            "raw_goals_over_xg": round(float(stats.spearmanr(b.raw_ratio, b.finishing)[0]), 3),
            "posterior_mean": round(float(stats.spearmanr(b.theta_mean, b.finishing)[0]), 3),
        }
    W.team_xg(conn).to_sql("team_xg", conn, if_exists="replace", index=False)


def train_development(conn, report: dict) -> None:
    prospects = pd.read_sql_query(
        "SELECT p.*, r.text FROM prospects p JOIN scouting_reports r USING (prospect_id)", conn)
    if prospects.empty:
        return
    log(f"extracting {len(prospects):,} scouting reports")
    feats = report_features(prospects.text, prospects.draft_round)
    df = pd.concat([prospects[["prospect_id", "draft_year", "dev_years_observed", "made_nhl"]], feats], axis=1)

    rng = np.random.default_rng(C.SEED)
    test = rng.random(len(df)) < 0.3
    train_df, test_df = df[~test].reset_index(drop=True), df[test].reset_index(drop=True)
    results = {}
    for name, features in [("draft_round_only", BASELINE), ("draft_round_plus_report", None)]:
        m = DevelopmentModel(features) if features else DevelopmentModel()
        m.fit(train_df)
        risk = m.debut_curve(test_df)[:, 3]
        results[name] = {"c_index": round(concordance_index(
            test_df.dev_years_observed, test_df.made_nhl, risk), 4)}
    final = DevelopmentModel().fit(df)
    joblib.dump(final, C.ARTIFACTS / "development.joblib")
    coefs = final.clf_[-1].coef_[0][: len(final.features)]
    results["standardized_coefficients"] = dict(sorted(
        ((f, round(float(c), 3)) for f, c in zip(final.features, coefs)),
        key=lambda kv: -abs(kv[1]))[:8])
    km = {int(r): kaplan_meier(g.dev_years_observed, g.made_nhl).to_dict("records")
          for r, g in df.groupby("draft_round")}
    results["kaplan_meier_by_round"] = km
    report["development"] = results

    log("evaluating extraction")
    truth = pd.read_sql_query(
        "SELECT t.*, m.skills_mentioned, r.text FROM sim_prospect_truth t "
        "JOIN sim_report_truth m USING (prospect_id) JOIN scouting_reports r USING (prospect_id)", conn)
    labels = [{"position": row.position, "shoots": row.shoots, "height_in": row.height_in,
               "weight_lb": row.weight_lb, "league": row.league,
               "skills": {s: getattr(row, f"scout_{s}") for s in row.skills_mentioned.split(",")}}
              for row in truth.itertuples()]
    report["extraction_synthetic"] = evaluate_extraction(truth.text, labels)
    hand = [json.loads(l) for l in open(C.ROOT / "scoutsignal/data/handwritten_reports.jsonl")]
    report["extraction_handwritten"] = evaluate_extraction(
        [h["text"] for h in hand], [h["labels"] for h in hand])


def train() -> dict:
    conn = W.connect()
    report: dict = {"generated": time.strftime("%Y-%m-%d %H:%M")}
    shots = load_shots(conn)
    report["data"] = {"shots": len(shots), "goals": int(shots.is_goal.sum()),
                      "seasons": sorted(map(int, shots.season.unique())),
                      "source": pd.read_sql_query("SELECT DISTINCT source FROM events", conn).source.tolist()}
    train_xg(conn, shots, report)
    train_finishing(conn, report)
    train_development(conn, report)
    (C.ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2, default=float))
    write_brief(conn, report, C.ARTIFACTS / "brief.md")
    log("done: artifacts/report.json, artifacts/brief.md")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(prog="scoutsignal")
    ap.add_argument("command", choices=["build", "train", "all"])
    ap.add_argument("--source", choices=["synthetic", "nhl"], default="synthetic")
    ap.add_argument("--season", type=int, default=2024, help="NHL season start year")
    ap.add_argument("--max-games", type=int, default=300)
    args = ap.parse_args()
    if args.command in ("build", "all"):
        build(args.source, args.season, args.max_games)
    if args.command in ("train", "all"):
        train()


if __name__ == "__main__":
    main()
