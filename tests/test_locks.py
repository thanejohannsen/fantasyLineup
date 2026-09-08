"""Lock-state tests.

The scenario that matters is Thursday night: one starter has played, the rest of
the roster is still live, and advice must respect the difference. Proposing a
change that can no longer be made is worse than proposing nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fantasylineup.db import connect, init_db
from fantasylineup.engine.lineup import optimize_lineup
from fantasylineup.engine.locks import (
    format_countdown,
    locked_slot_assignments,
    next_deadline,
    player_lock_states,
    team_lock_states,
)
from tests.test_lineup import p

SEASON = 2026
WEEK = 1
NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)  # Friday noon UTC


@pytest.fixture
def conn():
    c = connect(":memory:")
    init_db(c)
    yield c
    c.close()


def _add_game(conn, home, away, kickoff, status="pre", game_id=None):
    conn.execute(
        """INSERT INTO games (game_id, season, week, game_date, home, away, status,
                              kickoff_utc, updated_at)
           VALUES (?,?,?,?,?,?,?,?,'now')""",
        (
            game_id or f"{away}{home}",
            SEASON,
            WEEK,
            kickoff.date().isoformat(),
            home,
            away,
            status,
            kickoff.isoformat(),
        ),
    )
    conn.commit()


def test_pre_game_is_unlocked(conn):
    _add_game(conn, "KC", "BUF", NOW + timedelta(days=2))
    states = team_lock_states(conn, SEASON, WEEK, now=NOW)
    assert not states["KC"].locked
    assert not states["BUF"].locked
    assert states["KC"].opponent == "BUF"


def test_in_progress_game_is_locked(conn):
    _add_game(conn, "KC", "BUF", NOW + timedelta(days=2), status="in")
    states = team_lock_states(conn, SEASON, WEEK, now=NOW)
    assert states["KC"].locked


def test_passed_kickoff_locks_even_if_state_is_stale(conn):
    """Either signal alone is sufficient.

    Live state can lag minutes behind an actual kickoff. Waiting for it before
    treating a player as locked would hand out advice that is already void.
    """
    _add_game(conn, "KC", "BUF", NOW - timedelta(minutes=5), status="pre")
    states = team_lock_states(conn, SEASON, WEEK, now=NOW)
    assert states["KC"].locked
    assert states["KC"].reason == "kickoff has passed"


def test_missing_kickoff_time_still_locks_on_state(conn):
    """A stale or absent kickoff must not leave a played game unlocked."""
    conn.execute(
        """INSERT INTO games (game_id, season, week, home, away, status, updated_at)
           VALUES ('x',?,?,'KC','BUF','post','now')""",
        (SEASON, WEEK),
    )
    conn.commit()
    states = team_lock_states(conn, SEASON, WEEK, now=NOW)
    assert states["KC"].locked


def test_bye_week_player_is_unlocked_with_no_deadline(conn):
    """A player with no game is on a bye, not frozen."""
    _add_game(conn, "KC", "BUF", NOW + timedelta(days=2))
    states = player_lock_states(conn, [p("bye_guy", "RB", 0.0, team="DET")], SEASON, WEEK, now=NOW)
    state = states["bye_guy"]
    assert not state.locked
    assert state.kickoff_utc is None
    assert "bye" in state.reason


def test_next_deadline_ignores_locked_and_past_games(conn):
    _add_game(conn, "KC", "BUF", NOW - timedelta(hours=3), status="post")
    _add_game(conn, "SF", "LAR", NOW + timedelta(hours=5))
    _add_game(conn, "NYG", "DAL", NOW + timedelta(days=3))

    players = [
        p("a", "RB", 1.0, team="KC"),
        p("b", "WR", 1.0, team="SF"),
        p("c", "TE", 1.0, team="DAL"),
    ]
    states = player_lock_states(conn, players, SEASON, WEEK, now=NOW)
    assert next_deadline(states, now=NOW) == NOW + timedelta(hours=5)


def test_no_open_games_means_no_deadline(conn):
    _add_game(conn, "KC", "BUF", NOW - timedelta(hours=3), status="post")
    states = player_lock_states(conn, [p("a", "RB", 1.0, team="KC")], SEASON, WEEK, now=NOW)
    assert next_deadline(states, now=NOW) is None


def test_thursday_starter_is_pinned_and_rest_reoptimized(conn):
    """The whole point of lock awareness.

    A weak starter played Thursday and cannot be moved. A better player is on
    the bench. The engine must keep the Thursday player where he is and improve
    only the slots that are still open, rather than proposing an impossible swap.
    """
    _add_game(conn, "KC", "BUF", NOW - timedelta(days=1), status="post")  # Thursday, done
    _add_game(conn, "SF", "LAR", NOW + timedelta(days=2))  # Sunday, open

    snapshot = conn.execute(
        "INSERT INTO roster_snapshots (league_id, taken_at, content_hash) VALUES ('L','t','h')"
    ).lastrowid
    roster = [
        ("thursday_dud", 0, 1),
        ("sunday_starter", 1, 1),
        ("sunday_stud", None, 0),
    ]
    conn.executemany(
        """INSERT INTO roster_players
           (snapshot_id, roster_id, owner_id, sleeper_id, is_starter, slot_index, on_reserve)
           VALUES (?,5,'o',?,?,?,0)""",
        [(snapshot, pid, is_start, idx) for pid, idx, is_start in roster],
    )
    conn.commit()

    players = [
        p("thursday_dud", "RB", 3.0, team="KC"),
        p("sunday_starter", "RB", 8.0, team="SF"),
        p("sunday_stud", "RB", 20.0, team="LAR"),
    ]
    states = player_lock_states(conn, players, SEASON, WEEK, now=NOW)
    forced = locked_slot_assignments(conn, snapshot, 5, states)
    assert forced == {0: "thursday_dud"}

    lineup = optimize_lineup(players, ["RB", "RB"], forced=forced)
    assert lineup.assignments[0].sleeper_id == "thursday_dud"
    assert lineup.assignments[1].sleeper_id == "sunday_stud"
    assert lineup.total_points == pytest.approx(23.0)


def test_countdown_formatting():
    now = NOW
    assert format_countdown(now + timedelta(days=2, hours=9), now) == "2d 9h"
    assert format_countdown(now + timedelta(hours=4, minutes=12), now) == "4h 12m"
    assert format_countdown(now + timedelta(minutes=45), now) == "45m"
    assert format_countdown(now - timedelta(minutes=1), now) == "locked"
    assert format_countdown(None, now) == "no deadline"


def test_espn_scoreboard_parses_to_kickoffs(monkeypatch):
    """Pin the shape of ESPN's public scoreboard payload.

    Also pins the WSH -> WAS normalisation: Washington is the only club the two
    sources spell differently, and an unnormalised code would silently make
    every Washington player look like a bye.
    """
    import json
    from pathlib import Path

    import fantasylineup.sources.kickoffs as kickoffs

    payload = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "espn_scoreboard_2026_w1.json").read_text()
    )

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def get(self, *a, **kw):
            return _Resp()

    monkeypatch.setattr(kickoffs.httpx, "Client", lambda **kw: _Client())

    games = kickoffs.fetch_kickoffs(2026, 1)
    assert len(games) == 16
    assert all(g.kickoff_utc.tzinfo is not None for g in games)
    assert all(not g.started for g in games)

    teams = {t for g in games for t in (g.home, g.away)}
    assert "WSH" not in teams, "ESPN's WSH must be normalised to Sleeper's WAS"
    assert "WAS" in teams

    opener = min(games, key=lambda g: g.kickoff_utc)
    assert opener.kickoff_utc == datetime(2026, 9, 10, 0, 20, tzinfo=UTC)
    assert {opener.home, opener.away} == {"SEA", "NE"}
