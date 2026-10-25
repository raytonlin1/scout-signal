from scoutsignal.extract import extract


def test_metadata_pipe_header():
    r = extract("Kasper Lindqvist | LW | Shoots L | 6'1\" 187 lbs | SHL. Elite stride. "
                "Projects as a top-six forward.")
    assert (r.name, r.position, r.shoots, r.height_in, r.weight_lb, r.league) == \
        ("Kasper Lindqvist", "LW", "L", 73, 187, "SHL")
    assert r.projected_role == "top-six" and r.role_tier == 1


def test_metadata_prose():
    r = extract("Brody Kerr, a 6-foot-3, 210-pound left-shot defenceman from the WHL. Gap control is strong.")
    assert (r.position, r.shoots, r.height_in, r.weight_lb, r.league) == ("D", "L", 75, 210, "WHL")
    # "left-shot" must not be read as a comment on the shot.
    assert "shooting" not in r.skills


def test_contrast_splits_judgements():
    g = extract("Shot is not very good, but reads are excellent.").grades()
    assert g["shooting"] < 50 < g["hockey_sense"]


def test_shared_judgement_and_separate_judgements():
    assert extract("Elite skating and passing.").grades() == {"skating": 80, "passing": 80}
    g = extract("Skating is elite and passing is average.").grades()
    assert g == {"skating": 80, "passing": 50}


def test_negation_moves_toward_other_side_of_average():
    assert extract("Skating is not elite.").grades()["skating"] == 35
    assert extract("Compete is not a concern.").grades()["compete"] == 55
    assert extract("Motor is never an issue.").grades()["compete"] == 65


def test_aspect_mask_and_lacks():
    assert extract("Strength on the puck is a real strength.").grades() == {"physicality": 65}
    assert extract("Lacks any defensive game.").grades() == {"defense": 35}


def test_evidence_and_unmentioned_skills():
    r = extract("Release is elite.")
    assert r.skills["shooting"].evidence == ["release is elite."]
    assert set(r.skills) == {"shooting"}
