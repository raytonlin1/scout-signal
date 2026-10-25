"""Information extraction from free-text scouting reports.

Turns a report like

    "Kasper Lindqvist | LW | Shoots L | 6'1" 187 lbs | SHL. Elite stride. Shot is not
     very good, but reads are excellent. Projects as a top-six forward."

into structured fields (position, handedness, size, league, projected role) and 20-80
grades per skill, each with the text span that produced it.

Rules, not a learned model: scouting vocabulary is small and stable, every grade needs
to be traceable to a phrase a scout can check, and there are no public labelled reports
to train on. The pieces:

1. Sentences are split into segments on contrastive words ("but", "although", ";") and
   on "and" / commas only when both sides name a skill *and* a judgement, so "elite
   skating and passing" shares one judgement while "skating is elite and passing is
   fine" gets two.
2. Skill aspects and judgement phrases are matched longest-first from lexicons.
3. Negators ("not", "isn't", "never") flip a judgement towards the other side of
   average; intensifiers push it away from 50 and hedges pull it back.
4. Each aspect takes the judgement(s) in its segment, nearest first when there are several.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config as C

ASPECTS = {
    "skating": ["skating", "skater", "stride", "edges", "edgework", "first-step quickness",
                "first step", "quickness", "acceleration", "top speed", "speed", "mobility",
                "footwork", "feet"],
    "shooting": ["shot", "shooter", "release", "shooting", "finishing ability", "finishing",
                 "finisher", "one-timer", "wrister", "scoring touch"],
    "passing": ["passing", "passer", "playmaking", "playmaker", "vision", "puck distribution",
                "distribution", "distributor", "setup ability"],
    "hockey_sense": ["hockey sense", "sense", "hockey iq", "iq", "decision-making",
                     "decision making", "decisions", "reads", "anticipation", "awareness",
                     "processing", "smarts"],
    "compete": ["compete level", "compete", "motor", "work ethic", "battle level", "effort",
                "engine", "competitiveness", "battle"],
    "physicality": ["physicality", "physical game", "physical play", "strength on the puck",
                    "net-front play", "net-front presence", "net front", "board play",
                    "puck protection", "hitting"],
    "defense": ["defensive game", "defensive play", "defensive zone", "defence", "defense",
                "gap control", "gaps", "own-zone play", "own-zone coverage", "own zone",
                "backchecking", "backcheck", "stick detail", "positioning", "coverage"],
}

JUDGEMENTS = {
    80: ["elite", "game-breaking", "game breaking", "world-class", "world class", "exceptional",
         "generational"],
    75: ["high-end", "high end", "outstanding", "special", "dynamic"],
    70: ["plus-plus", "excellent", "superb", "terrific", "dangerous"],
    65: ["very good", "great", "a real strength", "a big strength", "well above average",
         "well above-average", "impressive", "high-level", "never an issue", "a weapon"],
    60: ["above average", "above-average", "good", "strong", "plus", "effective", "a strength",
         "sharp", "quick", "smooth", "heavy", "no concerns", "polished"],
    55: ["solid", "reliable", "a tick above average", "decent", "capable", "steady", "sound",
         "dependable"],
    50: ["average", "nhl-average", "fine", "adequate", "okay", "passable", "middling",
         "serviceable", "ordinary"],
    45: ["a bit below average", "a tick below average", "inconsistent", "fringe",
         "a work in progress", "raw", "so-so", "unrefined", "streaky", "average at best"],
    40: ["below average", "below-average", "a concern", "underwhelming", "limited", "mediocre",
         "suspect", "a question mark", "needs work", "needs improvement", "sloppy", "slow",
         "heavy-footed", "lacking"],
    30: ["poor", "a major weakness", "well below nhl standard", "weak", "bad", "a liability",
         "a weakness", "a problem", "a red flag"],
    20: ["a glaring hole", "very poor", "terrible", "awful", "nonexistent", "non-existent"],
}

NEGATORS = re.compile(r"(?<![\w-])(not|isn't|isnt|wasn't|aren't|never|no|hardly|far from|doesn't)(?![\w-])")
INTENSIFIERS = re.compile(r"(?<![\w-])(truly|extremely|really|exceptionally|incredibly|very)\s+$")
HEDGES = re.compile(r"(?<![\w-])(somewhat|slightly|fairly|reasonably|mostly|relatively)\s+$")
LACKS = re.compile(r"(?<![\w-])(lacks|lacking|without)\s+(?:\w+\s+){0,2}$")
CONTRAST = re.compile(r"\s*(?:;|,?\s+\b(?:but|although|though|however|while|whereas|yet)\b,?)\s+")
SOFT_SPLIT = re.compile(r"\s*(?:,\s*and|,|\band\b)\s+")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")


def _phrase_regex(phrases):
    alts = sorted(set(phrases), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(p) for p in alts) + r")(?![\w-])")


ASPECT_OF = {p: s for s, ps in ASPECTS.items() for p in ps}
GRADE_OF = {p: g for g, ps in JUDGEMENTS.items() for p in ps}
ASPECT_RE = _phrase_regex(ASPECT_OF)
GRADE_RE = _phrase_regex(GRADE_OF)

LEAGUES = ["OHL", "WHL", "QMJHL", "USHL", "NCAA", "SHL", "Liiga", "KHL", "AHL", "USNTDP",
           "NTDP", "Allsvenskan", "Mestis", "DEL", "MHL", "BCHL"]
POSITIONS = [
    (re.compile(r"\b(?:centre|center|pivot)\b", re.I), "C"),
    (re.compile(r"\bleft[- ]wing(?:er)?\b", re.I), "LW"),
    (re.compile(r"\bright[- ]wing(?:er)?\b", re.I), "RW"),
    (re.compile(r"\b(?:defen[cs]eman|blueliner|rearguard|d-man)\b", re.I), "D"),
    (re.compile(r"\b(LW|RW)\b"), None),
    # A bare "C" or "D" only counts as its own header field, e.g. "| C |".
    (re.compile(r"(?:^|\|)\s*(C|D)\s*(?=\||$)", re.M), None),
    (re.compile(r"\bwinger\b", re.I), "W"),
]
ROLES = {
    "top-line": 1, "first-line": 1, "top-six": 1, "top-pair": 1, "first-pair": 1,
    "top-four": 2, "second-line": 2, "middle-six": 2, "second-pair": 2,
    "third-line": 3, "bottom-six": 3, "third-pair": 3, "bottom-pair": 3,
    "fourth-line": 4, "depth": 4, "fringe": 4,
}
ROLE_RE = re.compile(r"(?:projects? as|projection:?|ceiling:?|profiles as)[^.]*?\b("
                     + "|".join(map(re.escape, ROLES)) + r")\b", re.I)


@dataclass
class SkillGrade:
    grade: int
    evidence: list[str] = field(default_factory=list)


@dataclass
class Report:
    name: str | None = None
    position: str | None = None
    shoots: str | None = None
    height_in: int | None = None
    weight_lb: int | None = None
    league: str | None = None
    projected_role: str | None = None
    role_tier: int | None = None
    skills: dict[str, SkillGrade] = field(default_factory=dict)

    def grades(self) -> dict[str, int]:
        return {k: v.grade for k, v in self.skills.items()}

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("name", "position", "shoots", "height_in", "weight_lb", "league",
              "projected_role", "role_tier")}
        d["skills"] = {k: {"grade": v.grade, "evidence": v.evidence} for k, v in self.skills.items()}
        return d


def _negate(g: float) -> float:
    # "not elite" is closer to average than to poor; "not poor" is a bit above average.
    return 50 - 0.5 * (g - 50)


def _judgement(seg: str, m: re.Match) -> float:
    g = float(GRADE_OF[m.group(1)])
    before = seg[: m.start()]
    window = " ".join(before.split()[-4:])
    if INTENSIFIERS.search(before) and m.group(1) not in ("very good", "very poor"):
        g = 50 + (g - 50) * 1.3
    if HEDGES.search(before):
        g = 50 + (g - 50) * 0.5
    if NEGATORS.search(window):
        g = _negate(g)
    return g


def _has_both(text: str) -> bool:
    return bool(ASPECT_RE.search(text)) and bool(GRADE_RE.search(text))


def _segments(sentence: str) -> list[str]:
    out = []
    for clause in CONTRAST.split(sentence):
        parts = SOFT_SPLIT.split(clause)
        # Only split on "and"/commas when every piece carries its own judgement.
        if len(parts) > 1 and all(_has_both(p) for p in parts):
            out.extend(parts)
        else:
            out.append(clause)
    return [s for s in out if s.strip()]


def _score_segment(seg: str) -> list[tuple[str, float, str]]:
    aspects = list(ASPECT_RE.finditer(seg))
    if not aspects:
        return []
    # Mask aspect spans so "strength on the puck" does not also read as "a strength".
    masked = seg
    for a in aspects:
        masked = masked[: a.start()] + " " * (a.end() - a.start()) + masked[a.end():]
    judgements = [(m, _judgement(masked, m)) for m in GRADE_RE.finditer(masked)]
    found = []
    for a in aspects:
        skill = ASPECT_OF[a.group(1)]
        if judgements and len(aspects) == 1:
            g = sum(j for _, j in judgements) / len(judgements)
        elif judgements:
            g = min(judgements, key=lambda j: abs(j[0].start() - a.start()))[1]
        elif LACKS.search(seg[: a.start()]):
            g = 35.0
        else:
            continue
        found.append((skill, g, seg.strip()))
    return found


def _metadata(text: str, r: Report) -> None:
    head = text.split("|")[0] if "|" in text else text
    m = re.match(r"\s*([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+){1,2})(?=\s*(?:[|,(]|$))", head)
    if m:
        r.name = m.group(1)

    m = (re.search(r"\b([4-7])\s*['’]\s*(\d{1,2})\s*(?:\"|”|'')?", text)
         or re.search(r"\b([4-7])-(?:foot|ft)-?(\d{1,2})?\b", text, re.I))
    if m:
        r.height_in = int(m.group(1)) * 12 + int(m.group(2) or 0)
    m = re.search(r"\b(1\d{2}|2\d{2})\s*-?\s*(?:lbs?\.?|pounds?)\b", text, re.I)
    if m:
        r.weight_lb = int(m.group(1))

    m = (re.search(r"\bshoots:?\s*(L|R|left|right)\b", text, re.I)
         or re.search(r"\b(left|right)[- ](?:shot|shooting|handed)\b", text, re.I))
    if m:
        r.shoots = m.group(1)[0].upper()

    for pattern, code in POSITIONS:
        m = pattern.search(text)
        if m:
            r.position = code or m.group(1)
            break

    m = re.search(r"\b(" + "|".join(LEAGUES) + r")\b", text)
    if m:
        r.league = m.group(1)

    m = ROLE_RE.search(text)
    if m:
        r.projected_role = m.group(1).lower()
        r.role_tier = ROLES[r.projected_role]


def extract(text: str) -> Report:
    r = Report()
    _metadata(text, r)
    raw: dict[str, list[tuple[float, str]]] = {}
    for sentence in SENTENCE.split(text.strip()):
        for seg in _segments(sentence.lower()):
            for skill, g, evidence in _score_segment(seg):
                raw.setdefault(skill, []).append((g, evidence))
    for skill in C.SKILLS:
        if skill in raw:
            g = sum(v for v, _ in raw[skill]) / len(raw[skill])
            grade = int(min(80, max(20, 5 * round(g / 5))))
            r.skills[skill] = SkillGrade(grade, [e for _, e in raw[skill]])
    return r


def evaluate(reports, truth) -> dict:
    """Score extraction against labelled reports.

    reports: iterable of (text, labels) where labels has 'skills' {skill: grade} and any
    of the metadata fields.
    """
    tp = fp = fn = 0
    abs_err, within5 = [], 0
    meta_hits = {k: [0, 0] for k in ("position", "shoots", "height_in", "weight_lb", "league",
                                     "projected_role")}
    for text, labels in zip(reports, truth):
        got = extract(text)
        g, t = got.grades(), labels["skills"]
        tp += len(g.keys() & t.keys())
        fp += len(g.keys() - t.keys())
        fn += len(t.keys() - g.keys())
        for k in g.keys() & t.keys():
            abs_err.append(abs(g[k] - t[k]))
            within5 += abs(g[k] - t[k]) <= 5
        for k, hit in meta_hits.items():
            if labels.get(k) is not None:
                hit[1] += 1
                hit[0] += getattr(got, k) == labels[k]
    p = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n_reports": len(truth),
        "skill_mention_precision": round(p, 3),
        "skill_mention_recall": round(rec, 3),
        "skill_mention_f1": round(2 * p * rec / max(p + rec, 1e-9), 3),
        "grade_mae": round(sum(abs_err) / max(len(abs_err), 1), 2),
        "grade_within_5": round(within5 / max(len(abs_err), 1), 3),
        "metadata_accuracy": {k: round(h / n, 3) for k, (h, n) in meta_hits.items() if n},
    }
