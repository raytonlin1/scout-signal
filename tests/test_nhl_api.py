from scoutsignal.nhl_api import _strength, parse_game


def _play(eid, kind, period, clock, x, y, team, code="1551", **details):
    return {"eventId": eid, "sortOrder": eid, "typeDescKey": kind, "timeInPeriod": clock,
            "periodDescriptor": {"number": period, "periodType": "REG"}, "situationCode": code,
            "homeTeamDefendingSide": "left",
            "details": {"xCoord": x, "yCoord": y, "eventOwnerTeamId": team, **details}}


def test_parse_normalises_attacking_direction():
    pbp = {"id": 2024020001, "season": 20242025, "homeTeam": {"id": 1}, "awayTeam": {"id": 7},
           "rosterSpots": [{"playerId": 11, "firstName": {"default": "A"}, "lastName": {"default": "B"},
                            "teamId": 1, "positionCode": "C"}],
           "plays": [
               _play(1, "shot-on-goal", 1, "01:00", -70, 10, 1, shotType="wrist", shootingPlayerId=11),
               _play(2, "goal", 1, "02:00", -80, -5, 1, shotType="snap", scoringPlayerId=11),
               _play(3, "missed-shot", 1, "03:00", 60, 0, 7, shotType="slap", shootingPlayerId=22),
               _play(4, "period-end", 1, "20:00", None, None, None),
           ]}
    ev, players = parse_game(pbp)
    assert len(ev) == 3
    assert (ev.x > 0).all()              # every shot now attacks +x
    assert ev.loc[0, "y"] == -10         # flipped with x
    assert ev.loc[1, "is_goal"] == 1 and ev.loc[1, "shooter_id"] == 11
    assert ev.loc[0, "game_seconds"] == 60
    assert players.name.tolist() == ["A B"]


def test_strength_from_situation_code():
    assert _strength("1551", True) == "EV"
    assert _strength("1451", True) == "PP"   # away down a skater
    assert _strength("1451", False) == "SH"
    assert _strength("0651", True) == "EN"   # away pulled the goalie
