"""One-page brief for front-office readers, written from report.json after every run.

Numbers a GM can act on, in plain language, with the uncertainty attached. No log loss.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def write_brief(conn: sqlite3.Connection, report: dict, path: Path) -> str:
    lines = ["# ScoutSignal brief", "",
             f"_Generated {report['generated']} from {report['data']['shots']:,} unblocked shots, "
             f"seasons {report['data']['seasons'][0]}–{report['data']['seasons'][-1]}._", ""]

    board = pd.read_sql_query("SELECT * FROM finishing", conn)
    buys = board[board.buy_low.astype(bool)].head(5)
    fin = report["finishing"]
    lines += ["## 1. Buy-low finishers", "",
              "Shooters the model rates as above-average finishers whose goal totals this season "
              "fell short of their expected goals. A market that reads raw goals will price them low.", "",
              "| Player | Pos | Career shots | Goals | xG | This season G − xG | Finishing (80% interval) | P(above avg) |",
              "|---|---|---:|---:|---:|---:|---|---:|"]
    for r in buys.itertuples():
        lines.append(f"| {r.name} | {r.position} | {r.shots} | {r.goals} | {r.xg:.1f} | "
                     f"{r.latest_goals_minus_xg:+.1f} | "
                     f"{r.theta_mean:.2f} ({r.theta_lo:.2f}–{r.theta_hi:.2f}) | {_pct(r.p_above_avg)} |")
    if buys.empty:
        lines.append("| none this season | | | | | | | |")
    bt = fin.get("backtest", {}).get("mean", {})
    if bt:
        lines += ["", f"Finishing 1.00 = scores exactly as expected. Estimates are shrunk toward "
                  f"league average (prior worth about {fin['alpha']:.0f} expected goals). "
                  f"Backtested on next-season goals (RMSE): shrunk {bt['shrunk']['rmse']:.2f}, "
                  f"everyone-is-average {bt['league']['rmse']:.2f}, raw goals-over-expected "
                  f"{bt['raw']['rmse']:.2f}. Raw finishing numbers predict worse than ignoring "
                  "finishing altogether; shrunk ones beat both."]
    if "rank_corr_with_true_talent" in fin:
        rc = fin["rank_corr_with_true_talent"]
        lines += [f"On simulated data, where true talent is known, the shrunk estimate ranks "
                  f"shooters with correlation {rc['posterior_mean']:.2f} against {rc['raw_goals_over_xg']:.2f} "
                  "for raw goals-over-expected."]

    audit = report.get("rule_change_audit")
    if audit:
        lines += ["", "## 2. Is the xG model still trustworthy?", ""]
        for r in audit["by_season"]:
            tag = "training" if r["in_training"] else "after change"
            lines.append(f"- {r['season']} ({tag}): {r['obs_over_exp']:.2f} goals per expected goal")
        lines += ["", f"**Verdict:** {audit['verdict']}."]
        rem = audit.get("remedies")
        if rem:
            lines += [f"Shifting the model's baseline using the first post-change season brings "
                      f"{rem['evaluated_on']} from {rem['frozen']['obs_over_exp']:.2f} to "
                      f"{rem['recalibrated']['obs_over_exp']:.2f} goals per expected goal without "
                      "retraining."]

    bt = [r for r in report["xg_backtest"] if r["model"] in ("pbp", "tracking")]
    if any(r["model"] == "tracking" for r in bt):
        df = pd.DataFrame(bt).pivot(index="test_season", columns="model", values="log_loss")
        lift = (1 - df.tracking / df.pbp).mean()
        auc = pd.DataFrame(bt).pivot(index="test_season", columns="model", values="auc").mean()
        lines += ["", "## 3. What tracking data adds", "",
                  f"Adding pre-shot movement (passes across the slot, goalie position, defender gap, "
                  f"time on the stick) improves out-of-season log loss by {_pct(lift)} over "
                  f"play-by-play alone, and AUC from {auc.pbp:.3f} to {auc.tracking:.3f}: two shots "
                  "from the same spot are not the same chance if one follows a cross-slot pass."]

    dev = report.get("development")
    if dev:
        a, b = dev["draft_round_only"]["c_index"], dev["draft_round_plus_report"]["c_index"]
        skills = [k for k in dev["standardized_coefficients"] if k.startswith("g_")][:2]
        top = " and ".join(k[2:].replace("_", " ") for k in skills)
        lines += ["", "## 4. Reading the scouting reports", "",
                  f"Grades extracted from free-text reports improve ranking of who reaches the NHL "
                  f"first (concordance {a:.2f} from draft round alone to {b:.2f} with the reports). "
                  f"Beyond draft round and projected role, the skills that predict an earlier debut "
                  f"are {top}."]

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text
