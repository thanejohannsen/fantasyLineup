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
