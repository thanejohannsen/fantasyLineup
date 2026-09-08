"""Availability tests.

Availability is the thing this project most needs to get right, because the
naive alternative -- replaying the draft forward through a transaction log --
drifts silently and is wrong forever once it misses a single event. These tests
pin the stateless property: whatever the newest roster snapshot says is the
answer, regardless of what came before it.
"""

from __future__ import annotations

import json

import pytest

from fantasylineup.db import connect, init_db
from fantasylineup.sync import compute_availability

LEAGUE = "test-league"


@pytest.fixture
def conn():
    c = connect(":memory:")
    init_db(c)
    yield c
    c.close()


def _add_players(conn, ids, team="BUF", position="RB", active=1):
    conn.executemany(
        """INSERT INTO players (sleeper_id, full_name, team, position, fantasy_positions,
                                active, updated_at)
           VALUES (?,?,?,?,?,?,'now')""",
        [(pid, f"Player {pid}", team, position, json.dumps([position]), active) for pid in ids],
    )
    conn.commit()


def _snapshot(conn, roster_map, taken_at):
    """roster_map: {roster_id: [player ids]}"""
    cur = conn.execute(
        "INSERT INTO roster_snapshots (league_id, taken_at, content_hash) VALUES (?,?,?)",
        (LEAGUE, taken_at, taken_at),
    )
    sid = cur.lastrowid
    rows = [
        (sid, rid, f"owner{rid}", pid, 0, None, 0)
        for rid, pids in roster_map.items()
        for pid in pids
    ]
    conn.executemany(
        """INSERT INTO roster_players
           (snapshot_id, roster_id, owner_id, sleeper_id, is_starter, slot_index, on_reserve)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return sid


def test_availability_is_universe_minus_rosters(conn):
    _add_players(conn, ["a", "b", "c", "d", "e"])
    _snapshot(conn, {1: ["a", "b"], 5: ["c"]}, "2026-09-01T00:00:00+00:00")

    avail = compute_availability(conn, LEAGUE, my_roster_id=5)

    assert avail.rostered_count == 3
    assert avail.my_players == {"c"}
    assert avail.free_agents == {"d", "e"}


def test_only_the_newest_snapshot_counts(conn):
    """The whole point: history is never replayed, only the latest state read.

    An earlier snapshot showing a player rostered must not leak into the answer
    after he has been dropped.
    """
    _add_players(conn, ["a", "b", "c"])
    _snapshot(conn, {1: ["a", "b", "c"]}, "2026-09-01T00:00:00+00:00")
    _snapshot(conn, {1: ["a"]}, "2026-09-08T00:00:00+00:00")

    avail = compute_availability(conn, LEAGUE, my_roster_id=1)

    assert avail.my_players == {"a"}
    assert avail.free_agents == {"b", "c"}


def test_gap_in_history_does_not_corrupt_state(conn):
    """Skipping weeks entirely is harmless -- nothing is accumulated.

    This is the failure mode that motivated the design: a bot that is offline
    for three weeks must come back correct with no reconciliation step.
    """
    _add_players(conn, ["a", "b", "c", "d"])
    _snapshot(conn, {1: ["a"], 2: ["b"]}, "2026-09-01T00:00:00+00:00")
    # Weeks pass with no observation at all, then the league looks like this:
    _snapshot(conn, {1: ["c"], 2: ["d"]}, "2026-10-15T00:00:00+00:00")

    avail = compute_availability(conn, LEAGUE, my_roster_id=1)

    assert avail.my_players == {"c"}
    assert avail.free_agents == {"a", "b"}


def test_inactive_and_teamless_players_are_not_free_agents(conn):
    """Retired and unsigned players should not be suggested as pickups."""
    _add_players(conn, ["active1"])
    _add_players(conn, ["retired"], active=0)
    conn.execute(
        """INSERT INTO players (sleeper_id, full_name, team, position, fantasy_positions,
                                active, updated_at)
           VALUES ('unsigned','Free Agent',NULL,'WR','[\"WR\"]',1,'now')"""
    )
    conn.commit()
    _snapshot(conn, {1: []}, "2026-09-01T00:00:00+00:00")

    avail = compute_availability(conn, LEAGUE, my_roster_id=1)

    assert avail.free_agents == {"active1"}


def test_player_on_two_rosters_is_refused(conn):
    """An inconsistent view from Sleeper must fail loudly, not be guessed at."""
    _add_players(conn, ["a"])
    _snapshot(conn, {1: ["a"], 2: ["a"]}, "2026-09-01T00:00:00+00:00")

    with pytest.raises(ValueError, match="multiple rosters"):
        compute_availability(conn, LEAGUE, my_roster_id=1)


def test_missing_snapshot_is_a_clear_error(conn):
    with pytest.raises(LookupError, match="run `fl sync`"):
        compute_availability(conn, LEAGUE, my_roster_id=1)
