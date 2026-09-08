"""Lineup optimizer tests."""

from __future__ import annotations

import pytest

from fantasylineup.engine.lineup import (
    PlayerProjection,
    lineup_value,
    marginal_value,
    optimize_lineup,
    starting_slots,
)

ZOO_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"]


def p(pid: str, pos: str, pts: float, **kw) -> PlayerProjection:
    return PlayerProjection(
        sleeper_id=pid,
        name=pid,
        position=pos,
        points=pts,
        fantasy_positions=frozenset({pos}),
        **kw,
    )


def test_starting_slots_strips_bench(roster_positions):
    slots = starting_slots(roster_positions)
    assert slots == ZOO_SLOTS
    assert len(slots) == 10
    assert "BN" not in slots


def test_dedicated_slot_takes_the_only_eligible_player():
    """A dedicated slot constrains who is left for FLEX."""
    slots = ["RB", "FLEX"]
    players = [p("rb1", "RB", 9.0), p("wr1", "WR", 8.5), p("wr2", "WR", 8.4)]
    lineup = optimize_lineup(players, slots)
    # RB slot can only take rb1; FLEX takes the best remaining, wr1.
    assert lineup.assignments[0].sleeper_id == "rb1"
    assert lineup.assignments[1].sleeper_id == "wr1"
    assert lineup.total_points == pytest.approx(17.5)


def test_flex_reserves_scarce_position():
    """A high-scoring WR must not squat in FLEX if that strands the WR slots."""
    slots = ["WR", "WR", "FLEX"]
    players = [
        p("wr1", "WR", 20.0),
        p("wr2", "WR", 15.0),
        p("wr3", "WR", 14.0),
        p("rb1", "RB", 14.5),
    ]
    lineup = optimize_lineup(players, slots)
    ids = {pl.sleeper_id for pl in lineup.starters}
    # Best total is wr1 + wr2 in the WR slots and rb1 in FLEX = 49.5,
    # beating wr1+wr2+wr3 = 49.0.
    assert ids == {"wr1", "wr2", "rb1"}
    assert lineup.total_points == pytest.approx(49.5)


def test_full_zoo_lineup():
    players = [
        p("qb", "QB", 22.0),
        p("rb1", "RB", 18.0),
        p("rb2", "RB", 12.0),
        p("rb3", "RB", 11.0),
        p("wr1", "WR", 17.0),
        p("wr2", "WR", 14.0),
        p("wr3", "WR", 13.0),
        p("te1", "TE", 9.0),
        p("k", "K", 8.0),
        p("def", "DEF", 7.0),
        p("benchwr", "WR", 4.0),
    ]
    lineup = optimize_lineup(players, ZOO_SLOTS)
    assert len(lineup.assignments) == 10
    assert not lineup.unfilled
    # Both FLEX slots go to the best leftovers: rb3 (11.0) and wr3 (13.0).
    flex_ids = {lineup.assignments[6].sleeper_id, lineup.assignments[7].sleeper_id}
    assert flex_ids == {"rb3", "wr3"}
    assert lineup.total_points == pytest.approx(131.0)
    assert [b.sleeper_id for b in lineup.bench] == ["benchwr"]


def test_bye_week_hole_leaves_slot_empty():
    """No eligible kicker means an empty slot, not an illegal one.

    Sleeper allows an empty slot. Filling it with an ineligible player would
    inflate the projected total and produce advice that cannot be followed.
    """
    players = [p("qb", "QB", 20.0), p("rb1", "RB", 10.0)]
    lineup = optimize_lineup(players, ["QB", "RB", "K"])
    assert lineup.unfilled == [2]
    assert lineup.total_points == pytest.approx(30.0)
    assert 2 not in lineup.assignments


def test_forced_assignment_pins_locked_player():
    """A player whose game kicked off cannot be moved.

    The optimizer must route around him rather than proposing a lineup that is
    no longer legal to set.
    """
    players = [
        p("rb_locked", "RB", 3.0, locked=True),
        p("rb_better", "RB", 20.0),
        p("wr1", "WR", 15.0),
    ]
    lineup = optimize_lineup(players, ["RB", "FLEX"], forced={0: "rb_locked"})
    assert lineup.assignments[0].sleeper_id == "rb_locked"
    # FLEX takes the best remaining, even though rb_better outscores the pin.
    assert lineup.assignments[1].sleeper_id == "rb_better"
    assert lineup.total_points == pytest.approx(23.0)


def test_forced_unknown_player_is_an_error():
    with pytest.raises(ValueError, match="not in the supplied roster"):
        optimize_lineup([p("a", "RB", 1.0)], ["RB"], forced={0: "ghost"})


def test_marginal_value_sees_positional_surplus():
    """The core insight behind the waiver and trade engines.

    A fifth good running back adds almost nothing to a roster already deep at
    the position, while an equally-projected wide receiver fills a real hole.
    Raw projections cannot tell these apart; marginal lineup value can.
    """
    slots = ["RB", "RB", "WR", "WR", "FLEX"]
    rb_heavy = [
        p("rb1", "RB", 18.0),
        p("rb2", "RB", 17.0),
        p("rb3", "RB", 16.0),
        p("rb4", "RB", 15.0),
        p("wr1", "WR", 12.0),
    ]
    another_rb = p("rb5", "RB", 14.0)
    a_receiver = p("wr2", "WR", 14.0)

    assert marginal_value(rb_heavy, slots, a_receiver) > marginal_value(rb_heavy, slots, another_rb)


def test_lineup_value_of_empty_roster_is_zero():
    assert lineup_value([], ZOO_SLOTS) == 0.0


def test_no_slots_benches_everyone():
    lineup = optimize_lineup([p("a", "RB", 5.0)], [])
    assert lineup.total_points == 0.0
    assert len(lineup.bench) == 1


def test_multi_position_eligibility():
    """Sleeper lists some players at more than one position."""
    swiss = PlayerProjection(
        sleeper_id="swiss",
        name="swiss",
        position="RB",
        points=10.0,
        fantasy_positions=frozenset({"RB", "WR"}),
    )
    lineup = optimize_lineup([swiss], ["WR"])
    assert lineup.assignments[0].sleeper_id == "swiss"


def test_canonicalization_prefers_dedicated_slot():
    """The best running back belongs in RB, not FLEX, when both are optimal.

    The optimizer is indifferent -- both assignments score identically -- but
    advice that buries the star in FLEX reads like a mistake and invites the
    reader to override correct advice.
    """
    slots = ["RB", "FLEX"]
    players = [p("star", "RB", 21.0), p("scrub", "RB", 12.0)]
    lineup = optimize_lineup(players, slots)
    assert lineup.assignments[0].sleeper_id == "star"
    assert lineup.assignments[1].sleeper_id == "scrub"


def test_canonicalization_preserves_total():
    """Reordering is presentation only and must never change the score."""
    slots = ["RB", "WR", "FLEX", "FLEX"]
    players = [
        p("rb1", "RB", 20.0),
        p("rb2", "RB", 9.0),
        p("wr1", "WR", 18.0),
        p("wr2", "WR", 11.0),
        p("te1", "TE", 6.0),
    ]
    lineup = optimize_lineup(players, slots)
    assert lineup.total_points == pytest.approx(58.0)
    assert lineup.assignments[0].sleeper_id == "rb1"
    assert lineup.assignments[1].sleeper_id == "wr1"


def test_canonicalization_never_moves_a_locked_player():
    """A player whose game kicked off must stay exactly where he is.

    Reshuffling him would produce a lineup that can no longer legally be set.
    """
    slots = ["RB", "FLEX"]
    players = [p("locked_scrub", "RB", 4.0, locked=True), p("star", "RB", 22.0)]
    lineup = optimize_lineup(players, slots, forced={0: "locked_scrub"})
    assert lineup.assignments[0].sleeper_id == "locked_scrub"
    assert lineup.assignments[1].sleeper_id == "star"
