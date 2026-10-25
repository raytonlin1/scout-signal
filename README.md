---
title: ScoutSignal
emoji: 🏒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ScoutSignal

Hockey player evaluation from three kinds of evidence: free-text scouting reports,
play-by-play, and spatio-temporal tracking. The project extracts structured data from each,
fits models on it, backtests them, checks for drift after a rule change, and serves the
results through an API, a dashboard and a one-page brief for front-office readers.

| Question a front office asks | Model | Technique |
|---|---|---|
| How dangerous was that shot? | Expected goals (xG) | Gradient-boosted trees with monotonic constraints; play-by-play and tracking feature sets |
| Is this shooter a good finisher or just lucky? | Finishing talent | Bayesian hierarchical Gamma-Poisson; prior fitted by marginal likelihood; exact posterior intervals |
| What do our scouts' reports say, in numbers? | Report extraction | Rule-based information extraction: 20–80 skill grades plus an evidence span for each |
| Will this prospect make the NHL, and when? | Development curve | Discrete-time survival model with right-censoring; features come from the extracted reports |
| Can we still trust the model after the rules changed? | Model audit | Rolling-origin backtest, PSI feature drift, calibration-in-the-large, recalibration |

## Results

From `python -m scoutsignal.pipeline all` on the synthetic league: 133k unblocked shots
over 5 seasons, 1,500 prospects. Every number is out of sample. See
[artifacts/brief.md](artifacts/brief.md) after a run.

**Tracking data adds signal play-by-play cannot see.** Rolling-origin backtest (train on
earlier seasons, test on the next), averaged over 2023–2025:

| xG model | Log loss | AUC | ECE |
|---|---:|---:|---:|
| Play-by-play features | 0.315 | 0.725 | 0.016 |
| + tracking features | **0.307** | **0.748** | 0.015 |
| Oracle (true probabilities) | 0.299 | 0.763 | 0.004 |

Tracking closes about half the gap between play-by-play and the oracle. On 120 real
2024-25 NHL games, the play-by-play model reaches an out-of-fold AUC of 0.746, in line with
public xG models.

**The audit catches the rule change and names the right fix.** A smaller-goalie-equipment
rule in 2024 raises scoring. A model frozen on 2021–23 under-predicts: 1.28 and 1.22 goals
per expected goal in 2024 and 2025. Feature PSI stays under 0.001: the shots look the
same, they just go in more often. That is concept drift, not feature drift. A one-parameter
intercept shift fitted on 2024 brings 2025 to 0.96 and halves calibration error
(ECE 0.021 → 0.008) with no retraining.

**Raw finishing numbers are worse than ignoring finishing.** Predicting next-season goals
from xG × finishing estimate:

| Finishing estimate | RMSE | Poisson deviance |
|---|---:|---:|
| Raw goals / xG | 2.64 | 1.64 |
| Everyone average (θ = 1) | 2.47 | 1.13 |
| Hierarchical posterior mean | **2.36** | **1.07** |

Against the simulator's true talent, the posterior ranks shooters at Spearman 0.71, against
0.68 for raw goals over xG. The buy-low list holds shooters with P(above average) ≥ 70%
whose goals this season fell short of their xG.

**Scouting text predicts development beyond draft round.** On held-out prospects, adding
the extracted grades raises concordance from 0.752 (draft round only) to 0.767. Skating and
hockey sense come out as the strongest skill signals, which matches the effects built into
the simulator.

**Extraction quality.** On 1,500 synthetic reports the grade MAE is 0.9 points on the 20–80
scale, and every metadata field is 100% correct. The 12 hand-written reports in
[scoutsignal/data/handwritten_reports.jsonl](scoutsignal/data/handwritten_reports.jsonl)
were written alongside the lexicon, so treat them as a smoke test, not a benchmark.

## Why synthetic data, and what is real

Public NHL play-by-play has shot locations and event sequences. It has no puck/player
tracking, and nobody publishes scouting reports. The simulator
([scoutsignal/simulate.py](scoutsignal/simulate.py)) draws latent talent for every player and
generates play-by-play, 10 Hz tracking frames and free-text reports from it. It keeps the
truth in separate `sim_*` tables, so every model can be scored against the answer, which real
data never gives you. The rule change, the noisy scouts and the right-censored draft classes
are all built in on purpose.

The real-data path is fully wired. [scoutsignal/nhl_api.py](scoutsignal/nhl_api.py) pulls
games from `api-web.nhle.com` into the same warehouse schema. The API's
`homeTeamDefendingSide` flag is unreliable, so the parser infers each team's attacking
direction per period from its shot locations. It also parses strength state from
`situationCode`, drops empty-net shots, and feeds the same SQL feature query and models.

```bash
python -m scoutsignal.pipeline all --source nhl --season 2024 --max-games 1312
```

On real data, tracking features and the development model are skipped because those inputs
don't exist publicly. With one season and few shots per player, the finishing prior
correctly shrinks almost everyone to average. Multiple seasons are needed before finishing
talent separates from noise, and that is itself the point of the model.

## How it works

```
simulate.py / nhl_api.py ──► SQLite warehouse (events, players, prospects, reports)
                                   │
         features.py: SQL window functions (LAG over game order) → rebound, rush, time since last event
                      numpy over tracking frames → defender gap, cross-slot pass, goalie offset, time on stick
                                   │
 models/xg.py ── backtest.py (rolling origin, PSI, recalibration)
      │ out-of-fold xG, normalised per season
 models/finishing.py (Gamma-Poisson)        extract.py ──► models/development.py (discrete-time hazard)
                                   │
            artifacts/: warehouse.sqlite, *.joblib, report.json, brief.md
                                   │
                app/main.py (FastAPI) + app/static/index.html (dashboard)
```

Design choices worth defending in an interview:

- **Sequence features live in SQL** ([features.py](scoutsignal/features.py)): `LAG()` over
  each game's event order, flipping the previous event's x into the shooting team's frame.
  A test checks that the SQL reproduces the simulator's true rebound and rush flags exactly
  (100% agreement).
- **Monotonic constraints on xG** for distance, angle, defender gap and goalie offset. A
  coach who sees a farther shot rated higher stops trusting the model.
- **Out-of-fold, season-normalised xG for finishing.** Each shot's xG comes from a model
  that never saw that game, scaled so league goals equal league xG every season. Without
  this, the rule change would show up as every 2024 shooter getting "better".
- **Conjugate posterior instead of MCMC.** The Gamma-Poisson model is exact and fits in
  milliseconds, so the API can recompute it on any filter. The PyMC version would be the
  same model with slower sampling.
- **Survival, not classification, for development.** "Hasn't made it yet" for a 2023
  draftee is censored, not a failure. Treating it as a 0 would bias every recent class
  downward.
- **Rules, not a learned NER model, for reports.** There are no public labelled scouting
  reports, the vocabulary is small, and every grade has to point at the phrase a scout
  can check. Contrastive splitting ("but", "although"), negation ("not elite" ≈ 35),
  hedges and aspect masking ("strength on the puck is a real strength") handle most of
  the difficulty.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m scoutsignal.pipeline all        # ~30 s: simulate, train, audit, write artifacts
uvicorn app.main:app --port 7860
```

Open http://localhost:7860 for the dashboard and http://localhost:7860/docs for the API.

```bash
curl -X POST localhost:7860/api/prospects/project -H 'content-type: application/json' \
  -d '{"text": "Teemu Lehtonen | RW | Shoots R | 5'\''10\" 172 lbs | Liiga. Release is elite. Not a strong skater.", "draft_round": 2}'
curl -X POST localhost:7860/api/xg -H 'content-type: application/json' \
  -d '{"x": 78, "y": -6, "shot_type": "wrist", "royal_road": true, "goalie_err": 4, "def_dist": 10, "shooter_speed": 12, "puck_dy_1s": 20, "possession_time": 0.1}'
curl 'localhost:7860/api/finishing?buy_low=true'
```

| Endpoint | Returns |
|---|---|
| `POST /api/scouting/extract` | Position, handedness, size, league, projected role, 20–80 grades with evidence |
| `POST /api/prospects/project` | The extraction plus P(NHL debut by year 1–8) |
| `POST /api/xg` | Shot xG; uses the tracking model when all tracking fields are given |
| `GET /api/finishing` | Finishing leaderboard with credible intervals and buy-low flags |
| `GET /api/players/{id}/shots` | Shot locations and xG for the shot map |
| `GET /api/teams` | Team xG share by season |
| `GET /api/report`, `/api/brief` | Every metric; the front-office brief |

## Tests

```bash
pytest
```

The tests cover extraction edge cases (negation, contrast, aspect masking, metadata in both
header styles), the SQL sequence features on hand-built games, NHL parsing (direction
flips, strength codes), finishing-model recovery and interval coverage, survival
bookkeeping, and the API. API tests need artifacts; CI builds them first.

## Deploy

- **Hugging Face Spaces:** the block at the top of this file is the Space config. Push the
  repo to a Docker Space. The image trains everything at build time with a fixed seed, so
  it serves immediately and every replica is identical.
- **Anywhere with Docker** (Fly.io, Render, Cloud Run): `docker build -t scoutsignal . &&
  docker run -p 7860:7860 scoutsignal`.
- **CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the pipeline, the tests,
  uploads `report.json` and `brief.md` as build artifacts (a model report on every
  commit), then builds the image and smoke-tests `/api/health`.

## Where this goes next

- Real tracking: the NHL's `pptReplayUrl` sprite files hold puck and player positions for
  goals. Those are goals only, so they are biased, but enough to validate the tracking features.
- Swap the rule-based extractor for an LLM extractor behind the same `Report` schema, and use
  the rule-based one as its regression test.
- Aging curves for skill growth: a hierarchical model on the extracted grades over repeated
  reports per prospect.
