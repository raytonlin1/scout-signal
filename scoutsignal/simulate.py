"""A synthetic league with known ground truth.

Real NHL play-by-play is public, but tracking data and scouting reports are not, and real
data never tells you the right answer. The simulator draws latent talent for every player,
generates play-by-play, tracking frames and free-text scouting reports from it, and keeps
the truth so each model can be checked against it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .features import goalie_line_y, tracking_features

SHOT_TYPE_LOGIT = {
    "wrist": 0.0, "snap": 0.10, "slap": 0.05, "backhand": -0.25,
    "tip-in": 0.20, "deflected": 0.10, "wrap-around": -0.60,
}
STRENGTH_LOGIT = {"EV": 0.0, "PP": 0.30, "SH": -0.10}
BASE_LOGIT = -1.75

FIRST = ["Liam", "Noah", "Owen", "Jack", "Lucas", "Ethan", "Mason", "Logan", "Carter", "Cole",
         "Nolan", "Ryan", "Eli", "Tyler", "Brady", "Dylan", "Adam", "Matias", "Jonas", "Oskar",
         "Emil", "Viktor", "Anton", "Nikita", "Pavel", "Kasper", "Aleksi", "Teemu", "Gabe", "Max"]
LAST = ["Anderson", "Bouchard", "Carlsson", "Dubois", "Ellis", "Fischer", "Gagnon", "Hughes",
        "Ivanov", "Jensen", "Kowalski", "Lindqvist", "MacLeod", "Novak", "Olsen", "Pelletier",
        "Quinn", "Roy", "Sorensen", "Tremblay", "Virtanen", "Walsh", "Young", "Zetterlund",
        "Bergeron", "Lehtonen", "Morozov", "Halvorsen", "Strand", "Kerr", "Doyle", "Becker"]


@dataclass
class League:
    skaters: pd.DataFrame       # latent finishing talent per skater
    goalies: pd.DataFrame
    events: pd.DataFrame        # play-by-play, attacking coordinates, NHL-like schema
    tracking: np.ndarray        # (n_shots, N_FRAMES, n_entities, 2)
    tracking_event_ids: np.ndarray
    shot_truth: pd.DataFrame    # true goal probability per shot
    prospects: pd.DataFrame     # latent skills and observed NHL-debut outcome
    reports: pd.DataFrame       # one free-text scouting report per prospect


def _names(rng, n):
    return [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]


def _roster(rng):
    n = C.N_TEAMS * C.SKATERS_PER_TEAM
    is_d = np.tile(np.arange(C.SKATERS_PER_TEAM) >= 12, C.N_TEAMS)
    skaters = pd.DataFrame({
        "player_id": 8_000_000 + np.arange(n),
        "name": _names(rng, n),
        "team_id": np.repeat(np.arange(1, C.N_TEAMS + 1), C.SKATERS_PER_TEAM),
        "position": np.where(is_d, "D", "F"),
        "shoots": rng.choice(["L", "R"], n, p=[0.62, 0.38]),
        # Finishing talent on the log-odds scale; defencemen shoot from worse spots
        # and are slightly worse finishers.
        "finishing": np.where(is_d, rng.normal(-0.10, 0.15, n), rng.normal(0.0, 0.25, n)),
        "volume": np.where(is_d, rng.gamma(4, 0.5, n), rng.gamma(4, 1.0, n)),
    })
    goalies = pd.DataFrame({
        "goalie_id": 8_900_000 + np.arange(2 * C.N_TEAMS),
        "name": _names(rng, 2 * C.N_TEAMS),
        "team_id": np.repeat(np.arange(1, C.N_TEAMS + 1), 2),
        "saving": rng.normal(0.0, 0.12, 2 * C.N_TEAMS),
    })
    return skaters, goalies


def _shot_locations(rng, n, rebound):
    dist = np.where(rebound, 4 + rng.gamma(2, 3.0, n), 6 + rng.gamma(2.2, 12.0, n)).clip(3, 70)
    ang = rng.normal(0, 0.6, n).clip(-1.45, 1.45)
    # Shots come from inside the offensive zone.
    x = np.maximum(C.NET_X - dist * np.cos(ang), C.BLUE_LINE_X + 1)
    y = (dist * np.sin(ang)).clip(-41, 41)
    return x, y, np.hypot(C.NET_X - x, y)


def _shot_types(rng, dist, rebound):
    # Close shots are tips and backhands; far shots are slap shots.
    close = np.clip((25 - dist) / 20, 0, 1)
    w = np.stack([
        np.full_like(dist, 4.0), np.full_like(dist, 2.0), 0.2 + 2.0 * (1 - close),
        0.5 + 1.0 * close, 0.1 + 1.5 * close + rebound, 0.1 + 0.6 * close, 0.05 + 0.3 * close,
    ], axis=1)
    w /= w.sum(axis=1, keepdims=True)
    idx = (w.cumsum(axis=1) > rng.random(len(dist))[:, None]).argmax(axis=1)
    return np.array(C.SHOT_TYPES)[idx]


def _tracking(rng, x, y, rebound):
    """Positions of shooter, puck, passer, defender and goalie for each shot."""
    n = len(x)
    t = np.linspace(-(C.N_FRAMES - 1) / C.HZ, 0, C.N_FRAMES)
    release = np.stack([x, y], axis=1)

    to_net = np.arctan2(-y, C.NET_X - x)
    heading = to_net + rng.normal(0, 0.9, n)
    speed = rng.gamma(4, 3.0, n)
    vel = speed[:, None] * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    shooter = release[:, None, :] + vel[:, None, :] * t[None, :, None]

    has_pass = ~rebound & (rng.random(n) < 0.55)
    royal = has_pass & (rng.random(n) < 0.30)
    side = np.where(y >= 0, 1.0, -1.0)
    passer_y = np.where(royal, -side * rng.uniform(8, 35, n), rng.uniform(-38, 38, n))
    passer_x = rng.uniform(35, 82, n)
    passer = np.stack([passer_x, passer_y], axis=1)[:, None, :] + rng.normal(0, 3, (n, 1, 2)) * t[None, :, None]

    # Puck: on the passer's stick, in flight, then on the shooter's stick. On rebounds it
    # comes off the goalie's pad instead of a teammate's stick.
    t_recv = np.where(has_pass | rebound, rng.uniform(-0.9, -0.05, n), -np.inf)
    origin = np.where(rebound[:, None], np.stack([np.full(n, 86.5), rng.uniform(-3, 3, n)], 1),
                      passer[:, 0, :])
    recv_point = release + vel * np.where(np.isfinite(t_recv), t_recv, 0)[:, None]
    flight = np.linalg.norm(recv_point - origin, axis=1) / rng.uniform(45, 75, n)
    flight = np.where(rebound, rng.uniform(0.15, 0.35, n), flight)
    t_pass = t_recv - flight
    frac = ((t[None, :] - t_pass[:, None]) / flight[:, None]).clip(0, 1)
    in_flight = origin[:, None, :] + frac[..., None] * (recv_point - origin)[:, None, :]
    before = t[None, :] < t_pass[:, None]
    after = t[None, :] >= t_recv[:, None]
    puck = np.where(after[..., None], shooter, in_flight)
    puck = np.where((before & ~rebound[:, None])[..., None], passer, puck)
    puck = np.where((before & rebound[:, None])[..., None], origin[:, None, :], puck)

    # Nearest defender closes on the shooter.
    def_dist = 2 + rng.gamma(2, 3.5, n)
    def_ang = to_net + rng.normal(0, 0.9, n)
    offset = def_dist[:, None] * np.stack([np.cos(def_ang), np.sin(def_ang)], axis=1)
    defender = shooter + offset[:, None, :] * (1 + 0.5 * -t)[None, :, None]

    # Goalie tracks the puck on the puck-to-net line, reacting with a lag; cross-ice passes
    # and rebounds leave the goalie behind the play.
    lag_frames = rng.integers(1, 5, n)
    idx = np.clip(np.arange(C.N_FRAMES)[None, :] - lag_frames[:, None], 0, None)
    lagged = np.take_along_axis(puck, idx[..., None], axis=1)
    gx = np.full((n, C.N_FRAMES), 85.0)
    gy = goalie_line_y(lagged, gx).clip(-5, 5) + rng.normal(0, 0.4, (n, 1))
    goalie = np.stack([gx, gy], axis=2)

    return np.stack([shooter, puck, passer, defender, goalie], axis=2).astype(np.float32)


def _season(rng, season, skaters, goalies, first_event_id):
    team_skaters = {t: g for t, g in skaters.groupby("team_id")}
    rows, frames_all, truth_rows = [], [], []
    eid = first_event_id
    season_form = rng.normal(0, 0.05, len(skaters))  # small year-to-year wobble in finishing
    form = dict(zip(skaters.player_id, skaters.finishing + season_form))
    saving = dict(zip(goalies.goalie_id, goalies.saving))

    for g in range(C.GAMES_PER_SEASON):
        game_id = season * 1_000_000 + 20_000 + g + 1
        home, away = rng.choice(np.arange(1, C.N_TEAMS + 1), 2, replace=False)
        starters = {home: 8_900_000 + 2 * (home - 1) + rng.integers(0, 2),
                    away: 8_900_000 + 2 * (away - 1) + rng.integers(0, 2)}
        n = rng.poisson(C.SHOTS_PER_GAME)
        times = np.sort(rng.integers(5, 3590, n))
        team = rng.choice([home, away], n)
        rebound = np.zeros(n, bool)
        for i in range(1, n):
            if rng.random() < 0.07:
                rebound[i] = True
                team[i] = team[i - 1]
                times[i] = times[i - 1] + rng.integers(1, 4)
            else:
                # Leave room for a preceding event so only true rebounds look like rebounds.
                times[i] = max(times[i], times[i - 1] + 2)
        period = np.minimum(times // 1200 + 1, 3)
        rebound[1:] &= period[1:] == period[:-1]

        # The event before each non-rebound shot: a neutral/defensive-zone play for rushes,
        # otherwise an offensive-zone faceoff, hit or puck battle.
        gap = np.diff(times, prepend=0)
        rush = ~rebound & (rng.random(n) < 0.18)
        pre_dt = np.where(rush, rng.integers(1, 5, n), 5 + rng.exponential(15, n).astype(int))
        pre_dt = np.minimum(pre_dt, np.maximum(gap - 1, 1))
        pre_time = times - pre_dt
        rush &= (pre_time // 1200 + 1) == period

        x, y, dist = _shot_locations(rng, n, rebound)
        stype = _shot_types(rng, dist, rebound.astype(float))
        strength = rng.choice(["EV", "PP", "SH"], n, p=[0.80, 0.15, 0.05])
        frames = _tracking(rng, x, y, rebound)
        tf = tracking_features(frames)

        shooters = np.empty(n, dtype=np.int64)
        for i in range(n):
            roster = team_skaters[team[i]]
            w = roster.volume.to_numpy()
            if rebound[i]:
                w = np.where(roster.position.to_numpy() == "F", w, 0.2 * w)
            shooters[i] = rng.choice(roster.player_id.to_numpy(), p=w / w.sum())
        goalie = np.array([starters[away if t == home else home] for t in team])

        ang = np.abs(np.arctan2(y, C.NET_X - x))
        z = (BASE_LOGIT - 0.055 * dist - 0.75 * ang
             + np.vectorize(SHOT_TYPE_LOGIT.get)(stype)
             + np.vectorize(STRENGTH_LOGIT.get)(strength)
             + 0.75 * rebound + 0.35 * rush
             + 0.30 * np.minimum(tf.goalie_err.to_numpy(), 6)
             + 0.40 * tf.royal_road.to_numpy()
             + 0.60 * (1 - np.exp(-tf.def_dist.to_numpy() / 5))
             + 0.45 * (tf.possession_time.to_numpy() < 0.35)
             + np.array([form[s] for s in shooters])
             - np.array([saving[k] for k in goalie])
             + (C.RULE_CHANGE_LOGIT if season >= C.RULE_CHANGE_SEASON else 0.0))
        p_true = 1 / (1 + np.exp(-z))
        goal = rng.random(n) < p_true
        # A rebound needs a saved shot in front of it: drop rebounds of goals (and rebounds
        # of dropped shots) rather than force saves, which would bias the goal rate.
        keep = np.ones(n, bool)
        for i in range(1, n):
            keep[i] = not (rebound[i] and (goal[i - 1] or not keep[i - 1]))
        rebound &= keep
        etype = np.where(goal, "goal", np.where(rng.random(n) < 0.70, "shot-on-goal", "missed-shot"))
        etype[:-1] = np.where(rebound[1:], "shot-on-goal", etype[:-1])

        for i in np.flatnonzero(keep):
            if not rebound[i]:
                ptype = rng.choice(["takeaway", "giveaway", "faceoff"] if rush[i]
                                   else ["faceoff", "hit", "giveaway", "takeaway"])
                px = rng.uniform(-60, 20) if rush[i] else rng.uniform(30, 85)
                other = away if team[i] == home else home
                pteam = team[i] if ptype == "takeaway" or rng.random() < 0.5 else other
                sign = 1 if pteam == team[i] else -1
                rows.append(dict(event_id=eid, game_id=game_id, season=season,
                                 game_seconds=int(pre_time[i]), event_type=ptype,
                                 team_id=int(pteam), x=sign * px, y=rng.uniform(-38, 38),
                                 shooter_id=None, goalie_id=None, shot_type=None,
                                 strength=strength[i], is_goal=0))
                eid += 1
            rows.append(dict(event_id=eid, game_id=game_id, season=season,
                             game_seconds=int(times[i]), event_type=etype[i], team_id=int(team[i]),
                             x=float(x[i]), y=float(y[i]), shooter_id=int(shooters[i]),
                             goalie_id=int(goalie[i]), shot_type=stype[i], strength=strength[i],
                             is_goal=int(goal[i])))
            truth_rows.append(dict(event_id=eid, p_true=p_true[i], true_rebound=int(rebound[i]),
                                   true_rush=int(rush[i])))
            frames_all.append(frames[i])
            eid += 1

    return rows, frames_all, truth_rows, eid


def _prospects(rng):
    n = len(C.DRAFT_CLASSES) * C.PROSPECTS_PER_CLASS
    draft_year = np.repeat(C.DRAFT_CLASSES, C.PROSPECTS_PER_CLASS)
    g = rng.normal(0, 1, n)
    skills = {s: (50 + 10 * (0.6 * g + 0.8 * rng.normal(0, 1, n))).clip(20, 80) for s in C.SKILLS}
    # Scouts see skills with noise; draft position reflects their consensus.
    consensus = g + rng.normal(0, 0.6, n)
    rank = pd.Series(consensus).groupby(draft_year).rank(ascending=False).to_numpy()
    draft_round = np.minimum(7, 1 + ((rank - 1) // (C.PROSPECTS_PER_CLASS / 7)).astype(int))
    position = rng.choice(["C", "LW", "RW", "D"], n, p=[0.3, 0.2, 0.2, 0.3])
    league = rng.choice(["OHL", "WHL", "QMJHL", "USHL", "NCAA", "SHL", "Liiga"], n)

    # Discrete-time hazard of NHL debut in development year k. Skating and hockey sense
    # matter beyond overall talent, which is what the text extractor should pick up.
    year_effect = np.array([-1.2, 0.0, 0.5, 0.4, 0.0, -0.5, -1.0, -1.5])
    lin = (-3.2 + 0.9 * g + 0.04 * (skills["skating"] - 50) + 0.03 * (skills["hockey_sense"] - 50)
           + np.where(position == "D", -0.3, 0.0))
    window = np.minimum(C.LAST_OBSERVED_SEASON - draft_year, C.MAX_DEV_YEARS)
    time = window.copy()
    event = np.zeros(n, int)
    for k in range(1, C.MAX_DEV_YEARS + 1):
        h = 1 / (1 + np.exp(-(lin + year_effect[k - 1])))
        hit = (event == 0) & (k <= window) & (rng.random(n) < h)
        time[hit], event[hit] = k, 1

    df = pd.DataFrame({
        "prospect_id": 1 + np.arange(n), "name": _names(rng, n), "draft_year": draft_year,
        "draft_round": draft_round, "position": position, "league": league,
        "shoots": rng.choice(["L", "R"], n, p=[0.6, 0.4]),
        "height_in": rng.normal(72.5, 2.2, n).round().astype(int),
        "weight_lb": rng.normal(188, 12, n).round().astype(int),
        "latent_talent": g, "dev_years_observed": time, "made_nhl": event,
    })
    for s in C.SKILLS:
        df[f"true_{s}"] = skills[s]
        df[f"scout_{s}"] = (5 * np.round((skills[s] + rng.normal(0, 5, n)) / 5)).clip(20, 80).astype(int)
    return df


# Report phrasing lives here, apart from the extractor's lexicon, so evaluation on
# synthetic reports is not a pure round trip.
GRADE_WORDS = {
    80: ["elite", "game-breaking", "high-end"],
    70: ["plus-plus", "excellent", "high-end"],
    65: ["very good", "a real strength", "well above average"],
    60: ["above-average", "good", "strong"],
    55: ["solid", "a tick above average", "reliable"],
    50: ["average", "fine", "NHL-average"],
    45: ["a bit below average", "inconsistent", "fringe"],
    40: ["below-average", "a concern", "underwhelming"],
    30: ["poor", "a major weakness", "well below NHL standard"],
    20: ["a glaring hole", "very poor"],
}
SKILL_NOUNS = {
    "skating": ["skating", "stride", "edgework", "first-step quickness", "acceleration"],
    "shooting": ["shot", "release", "shooting", "finishing ability", "one-timer"],
    "passing": ["passing", "playmaking", "vision", "puck distribution"],
    "hockey_sense": ["hockey sense", "hockey IQ", "decision-making", "reads", "anticipation"],
    "compete": ["compete level", "motor", "work ethic", "battle level"],
    "physicality": ["physicality", "strength on the puck", "net-front play", "physical game"],
    "defense": ["defensive game", "gap control", "own-zone play", "backchecking", "stick detail"],
}
POS_WORDS = {"C": ["centre", "center", "pivot"], "LW": ["left winger", "LW"],
             "RW": ["right winger", "RW"], "D": ["defenceman", "defenseman", "blueliner"]}


def _grade_word(rng, grade, adjective=False):
    key = min(GRADE_WORDS, key=lambda k: abs(k - grade))
    words = GRADE_WORDS[key]
    if adjective:
        # "Elite stride." reads fine; "A concern stride." does not.
        words = [w for w in words if not w.startswith(("a ", "well ", "NHL", "fine"))] or ["average"]
    return str(rng.choice(words))


def _report(rng, row):
    pos_word = str(rng.choice(POS_WORDS[row.position]))
    feet, inches = divmod(int(row.height_in), 12)
    shot = "left" if row.shoots == "L" else "right"
    if rng.random() < 0.5:
        header = (f"{row.name} | {pos_word} | Shoots {row.shoots} | {feet}'{inches}\" "
                  f"{row.weight_lb} lbs | {row.league}.")
    else:
        header = (f"{row.name}, a {feet}-foot-{inches}, {row.weight_lb}-pound {shot}-shot "
                  f"{pos_word} from the {row.league}.")
    mentioned = rng.choice(list(C.SKILLS), size=int(rng.integers(4, 7)), replace=False)
    sentences = []
    i = 0
    while i < len(mentioned):
        s = mentioned[i]
        grade = getattr(row, f"scout_{s}")
        noun = str(rng.choice(SKILL_NOUNS[s]))
        word = _grade_word(rng, grade)
        r = rng.random()
        if r < 0.15 and i + 1 < len(mentioned):
            s2 = mentioned[i + 1]
            noun2 = str(rng.choice(SKILL_NOUNS[s2]))
            word2 = _grade_word(rng, getattr(row, f"scout_{s2}"))
            grade2 = getattr(row, f"scout_{s2}")
            joiner = "but" if abs(grade - grade2) >= 10 else "and"
            sentences.append(f"{noun.capitalize()} is {word}, {joiner} {noun2} is {word2}.")
            i += 2
            continue
        if r < 0.25 and grade <= 45:
            better = _grade_word(rng, 65)
            sentences.append(f"{noun.capitalize()} is not {better}.")
        elif r < 0.35 and grade >= 60:
            sentences.append(f"{noun.capitalize()} is never an issue.")
        elif r < 0.60:
            sentences.append(f"{_grade_word(rng, grade, adjective=True).capitalize()} {noun}.")
        else:
            sentences.append(f"{noun.capitalize()} is {word}.")
        i += 1
    rng.shuffle(sentences)
    overall = np.mean([getattr(row, f"scout_{s}") for s in C.SKILLS])
    if row.position == "D":
        role = "top-pair" if overall >= 60 else "second-pair" if overall >= 52 else "third-pair" if overall >= 45 else "depth"
        role_text = f"{role} defenceman"
    else:
        role = "top-six" if overall >= 60 else "middle-six" if overall >= 52 else "bottom-six" if overall >= 45 else "depth"
        role_text = f"{role} forward"
    return f"{header} " + " ".join(sentences) + f" Projects as a {role_text}.", sorted(mentioned)


def simulate(seed: int = C.SEED, seasons=C.SEASONS) -> League:
    rng = np.random.default_rng(seed)
    skaters, goalies = _roster(rng)
    rows, frames, truth = [], [], []
    eid = 1
    for season in seasons:
        r, f, t, eid = _season(rng, season, skaters, goalies, eid)
        rows += r
        frames += f
        truth += t
    events = pd.DataFrame(rows)
    events["period"] = np.minimum(events.game_seconds // 1200 + 1, 3)
    events["sort_order"] = events.groupby("game_id").cumcount()
    events = events[["event_id", "game_id", "season", "period", "sort_order", "game_seconds",
                     "event_type", "team_id", "x", "y", "shooter_id", "goalie_id",
                     "shot_type", "strength", "is_goal"]]
    shot_truth = pd.DataFrame(truth)

    prospects = _prospects(rng)
    texts, mentioned = zip(*(_report(rng, row) for row in prospects.itertuples()))
    reports = pd.DataFrame({"prospect_id": prospects.prospect_id, "text": texts,
                            "skills_mentioned": [",".join(m) for m in mentioned]})
    return League(skaters, goalies, events, np.stack(frames), shot_truth.event_id.to_numpy(),
                  shot_truth, prospects, reports)
