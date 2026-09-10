"""Injury regime tests.

Fixtures are real cases sampled from the live league, because the two failure
modes that matter both look reasonable in the abstract: treating everyone with a
designation as damaged, and treating nobody as damaged at all.
"""

from __future__ import annotations

import pytest

from fantasylineup.model.health import (
    DEFAULT_MULTIPLIERS,
    Regime,
    classify,
    is_structural,
    ros_multiplier,
    weekly_multiplier,
)

# (status, body_part, notes, has_weekly_projection)
KITTLE = ("Questionable", "Achilles", "Surgery", True)
MAHOMES = ("Questionable", "Knee - ACL", "Surgery", True)
NABERS = ("Questionable", "Knee - ACL", "Surgery", True)
NACUA = ("Questionable", "Undisclosed", None, True)
CHASE = ("Questionable", "Knee", None, True)
PENIX = ("Out", "Knee - ACL", "Surgery", False)
HILL = ("Questionable", "Knee - ACL", "Surgery", False)
DELL = ("IR", "Knee - ACL + MCL", "Surgery", False)
CHARBONNET = ("PUP", "Knee - ACL", "Surgery", False)
ANKLE_TWEAK = ("Questionable", "Ankle", None, False)
HEALTHY = (None, None, None, True)


def test_playing_after_surgery_is_diminished_not_out():
    """Kittle's case, and the reason availability is the discriminator.

    He had Achilles surgery and is playing. Classifying him as out would zero a
    player who is on the field every week; classifying him as healthy would
    price him as the player he was before the injury.
    """
    health = classify(*KITTLE)
    assert health.regime is Regime.PLAYING_DIMINISHED
    assert health.structural
    assert health.is_tradeable
    assert ros_multiplier(health) == pytest.approx(0.90)


def test_not_playing_after_structural_surgery_is_season_over():
    """Recovery from an ACL or Achilles outruns any remaining schedule.

    These four all carried non-zero rest-of-season projections upstream while
    being unavailable -- Penix at 143.2 points, Hill at 93.7 -- which is the
    exact figure the tool used to trade on.
    """
    for case in (PENIX, HILL, DELL, CHARBONNET):
        health = classify(*case)
        assert health.regime is Regime.OUT_LONG, case
        assert not health.is_tradeable, case
        assert ros_multiplier(health) == 0.0, case


def test_questionable_stars_are_not_penalised():
    """The regression a blanket injury_status rule would cause.

    "Questionable" skews heavily toward good players, because good players get
    reported: median rest-of-season projection 95.3 against 38.7 for players
    with no designation. Haircutting the tag itself would quietly dock every
    star for an undisclosed knock.
    """
    for case in (NACUA, CHASE):
        health = classify(*case)
        assert health.regime is Regime.PLAYING_DIMINISHED, case
        assert not health.structural, case
        assert ros_multiplier(health) == 1.0, case


def test_short_term_absence_is_discounted_not_zeroed():
    """Out for now with a soft-tissue problem is not out for the year."""
    health = classify(*ANKLE_TWEAK)
    assert health.regime is Regime.OUT_SHORT
    assert health.is_tradeable
    assert 0.0 < ros_multiplier(health) < 1.0


def test_healthy_player_is_untouched():
    health = classify(*HEALTHY)
    assert health.regime is Regime.HEALTHY
    assert not health.has_designation
    assert ros_multiplier(health) == 1.0


def test_compound_body_parts_are_recognised():
    """Sleeper writes "Knee - ACL + MCL"; substring matching has to cope."""
    assert is_structural("Knee - ACL + MCL", None)
    assert is_structural("Knee - ACL", None)
    assert is_structural("Achilles", None)
    assert not is_structural("Hamstring", "Strain")


def test_surgery_note_alone_implies_structural():
    """The body part is sometimes vague where the note is not."""
    assert is_structural("Lower Body", "Surgery")
    assert is_structural(None, "Surgery")


def test_multipliers_are_overridable():
    """The numbers are judgment, so they live in config, not in code."""
    health = classify(*KITTLE)
    assert ros_multiplier(health, {"playing_diminished_structural": 0.5}) == pytest.approx(0.5)
    assert set(DEFAULT_MULTIPLIERS) == {
        "healthy",
        "playing_diminished_structural",
        "playing_diminished_soft",
        "out_short",
        "out_long",
    }


# ------------------------------------------------------------------ disclosure


def test_disclosure_matches_what_the_model_actually_did():
    """The note must not claim a haircut that was never applied.

    A structural case is discounted, so saying the projection is down is true.
    A soft-tissue "Questionable" is not discounted at all, so the same sentence
    would be a false statement about our own numbers.
    """
    structural = classify(*KITTLE).sentence("George Kittle")
    assert "down from where he was" in structural
    assert "Achilles" in structural or "achilles" in structural

    soft = classify(*CHASE).sentence("Ja'Marr Chase")
    assert "down from where he was" not in soft
    assert "still projected to play" in soft


def test_disclosure_preserves_acronyms():
    """"knee - acl surgery" reads as a typo in a message to another person."""
    assert "ACL" in classify(*PENIX).sentence("Michael Penix")
    assert "MCL" in classify(*DELL).sentence("Tank Dell")


def test_disclosure_omits_undisclosed_body_part():
    """Repeating "Undisclosed" pads the note without informing anyone."""
    text = classify(*NACUA).sentence("Puka Nacua")
    assert "ndisclosed" not in text
    assert "Questionable" in text


def test_healthy_player_has_no_disclosure():
    assert classify(*HEALTHY).sentence("Anyone") == ""


def test_article_agreement():
    assert "an ankle issue" in classify(*ANKLE_TWEAK).sentence("Someone")
    assert "a knee issue" in classify(*CHASE).sentence("Someone")


def test_disclosure_capitalises_proper_nouns():
    """"achilles surgery" reads as a typo; the tendon is named after a person."""
    assert "Achilles" in classify(*KITTLE).sentence("George Kittle")
    # Common nouns still read naturally in lower case.
    assert "a knee issue" in classify(*CHASE).sentence("Ja'Marr Chase")


# ------------------------------------------------- weekly availability


def test_a_player_ruled_out_is_worth_nothing_this_week():
    """The bug: Sleeper's projections endpoint keeps a full number for him.

    Its app shows zero, so the discrepancy is invisible unless you read the raw
    feed. Sampled live: Sam Darnold 17.1 while Out with a hip injury, Brock
    Bowers 16.0 while Doubtful after meniscus surgery. Left alone the optimiser
    started a doubtful tight end over a healthy one.
    """
    out = classify("Out", "Hip", None, has_weekly_projection=True)
    assert weekly_multiplier(out) == 0.0


def test_long_term_designations_are_also_zero_for_the_week():
    for status in ("IR", "PUP", "NA", "Sus", "DNR"):
        health = classify(status, "Knee", None, has_weekly_projection=False)
        assert weekly_multiplier(health) == 0.0, status


def test_doubtful_is_near_zero_but_not_zero():
    health = classify("Doubtful", "Knee - Meniscus", "Surgery", has_weekly_projection=True)
    assert 0.0 < weekly_multiplier(health) < 0.25


def test_questionable_keeps_most_of_his_value():
    """Most questionable players suit up; benching them all would be worse.

    This is the number to calibrate first, and the one place where the
    module's argument against blunt use of `injury_status` does not apply:
    for a single week the designation is a statement about availability.
    """
    health = classify("Questionable", "Ankle", None, has_weekly_projection=True)
    assert 0.7 <= weekly_multiplier(health) <= 0.9


def test_an_undesignated_player_is_untouched():
    assert weekly_multiplier(classify(None, None, None, has_weekly_projection=True)) == 1.0


def test_an_unrecognised_designation_is_not_treated_as_healthy():
    """A new upstream code should not silently become full value."""
    health = classify("Probable", "Ankle", None, has_weekly_projection=True)
    assert weekly_multiplier(health) < 1.0


def test_the_weekly_and_season_haircuts_are_different_questions():
    """A player back from surgery and playing keeps season value but is
    still a weekly risk while carrying a designation; one who is out for a
    week keeps season value that the weekly number must not."""
    playing_back = classify("Questionable", "Achilles", "Surgery", has_weekly_projection=True)
    assert ros_multiplier(playing_back) > 0.5
    assert weekly_multiplier(playing_back) < 1.0


def test_grading_a_finished_week_is_not_haircut_by_today_s_designation():
    """Calibration must score the projection that was acted on.

    The weekly haircut is on by default so a new forward-looking caller is safe,
    which makes the recap the one place it has to be turned off: an injury
    designation carried today says nothing about who was available in a week
    already played, and applying it would quietly bias the fit that grades the
    model.
    """
    import sqlite3

    from fantasylineup.db import init_db
    from fantasylineup.pipeline import blended_projections

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.execute(
        """INSERT INTO players (sleeper_id, full_name, position, fantasy_positions,
                                team, injury_status, updated_at)
           VALUES ('1', 'Ruled Out', 'TE', '["TE"]', 'LV', 'Out', '2026-09-10')"""
    )
    conn.execute(
        """INSERT INTO projections (source, season, week, sleeper_id, as_of, mean, stats)
           VALUES ('sleeper', 2026, 1, '1', '2026-09-10', 16.0, '{"rec": 6.0}')"""
    )
    conn.commit()

    scoring = {"rec": 1.0}
    forward, _ = blended_projections(conn, {"1"}, 2026, 1, scoring, {}, 0.0)
    graded, _ = blended_projections(
        conn, {"1"}, 2026, 1, scoring, {}, 0.0, apply_weekly_health=False
    )
    conn.close()

    assert forward[0].points == 0.0, "a forward-looking week must rule him out"
    assert graded[0].points == pytest.approx(6.0), "a finished week keeps what was projected"
