"""Ingest tests."""

from __future__ import annotations

from fantasylineup.sync import is_fantasy_relevant


def test_hybrid_player_is_relevant():
    """Travis Hunter regression.

    His primary ``position`` is DB, but ``fantasy_positions`` includes WR and he
    is rostered and started as a receiver. Filtering on the primary position
    dropped him from the player table entirely -- a rostered player silently
    absent from lineup optimization, with nothing to indicate anything was
    wrong.
    """
    hunter = {"position": "DB", "fantasy_positions": ["DB", "WR"], "full_name": "Travis Hunter"}
    assert is_fantasy_relevant(hunter)


def test_pure_defensive_player_is_not_relevant():
    assert not is_fantasy_relevant({"position": "CB", "fantasy_positions": ["DB"]})


def test_offensive_lineman_is_not_relevant():
    assert not is_fantasy_relevant({"position": "OL", "fantasy_positions": []})


def test_missing_fantasy_positions_falls_back_to_position():
    """Some records carry no fantasy_positions array at all."""
    assert is_fantasy_relevant({"position": "WR", "fantasy_positions": None})
    assert is_fantasy_relevant({"position": "DEF"})


def test_empty_record_is_not_relevant():
    assert not is_fantasy_relevant({})


def test_real_dump_slice_includes_hybrids(players_slice):
    """The captured dump must still exercise the hybrid path."""
    hybrids = [
        p
        for p in players_slice.values()
        if p.get("position") not in {"QB", "RB", "WR", "TE", "K", "DEF"}
        and is_fantasy_relevant(p)
    ]
    assert hybrids, "fixture no longer covers hybrid players"
    assert any(p["full_name"] == "Travis Hunter" for p in hybrids)
