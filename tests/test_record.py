"""Grading the contrarian rule against what was visible before kickoff.

The record is derived from stored snapshots rather than accumulated, so these
tests are really about *which reading decides*. Three boundaries carry the
whole design: a reading after kickoff must never count, a reading too far
before kickoff is not gameday evidence, and the reading that does count is the
last one before the game -- which is what makes "it fell off the board during
the week" enforce itself with no bookkeeping.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from fantasylineup.db import init_db
from fantasylineup.engine.record import MAX_STALENESS, settle, tally
from fantasylineup.sources.splits import (
    GameSplits,
    Side,
    parse_games,
    store_results,
    store_splits,
)

KICK = datetime(2026, 9, 27, 17, 0, tzinfo=UTC)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    init_db(c)
    yield c
    c.close()


def _snapshot(conn, at, *, home_tickets=90, away=None, home=None, line=-3.5, kickoff=KICK):
    """One reading of one spread, with the home side carrying the tickets."""
    game = GameSplits(
        away=away or "AAA",
        home=home or "BBB",
        kickoff_utc=kickoff,
        status="scheduled",
        num_bets=5000,
        sides={
            ("spread", "home"): Side("spread", "home", line, -110, home_tickets, 88),
            ("spread", "away"): Side("spread", "away", -line, -110, 100 - home_tickets, 12),
        },
    )
    store_splits(conn, [game], fetched_at=at)


def _result(conn, away_points, home_points, away="AAA", home="BBB", kickoff=KICK):
    store_results(
        conn,
        [
            GameSplits(
                away=away, home=home, kickoff_utc=kickoff, status="complete",
                num_bets=0, away_points=away_points, home_points=home_points,
            )
        ],
    )


# ------------------------------------------------- which reading decides


def test_a_game_that_fell_off_the_board_is_never_counted():
    """The point of grading on the gameday reading.

    Lopsided on Tuesday, level by Sunday: the bet was not on the board when it
    mattered, so it must not turn up in the record either way.
    """
    c = sqlite3.connect(":memory:")
    init_db(c)
    _snapshot(c, KICK - timedelta(days=3), home_tickets=90)
    _snapshot(c, KICK - timedelta(minutes=30), home_tickets=55)
    _result(c, 30, 20)

    assert settle(c) == []


def test_a_game_that_only_qualifies_late_is_counted():
    """And the converse, so the test above cannot pass by grading nothing."""
    c = sqlite3.connect(":memory:")
    init_db(c)
    _snapshot(c, KICK - timedelta(days=3), home_tickets=55)
    _snapshot(c, KICK - timedelta(minutes=30), home_tickets=90)
    _result(c, 30, 20)

    (s,) = settle(c)
    assert s.fade.public.tickets == 90
    assert s.observed_at == KICK - timedelta(minutes=30)


def test_a_reading_taken_after_kickoff_never_decides(conn):
    """Once the game is under way the tickets include people betting it live,
    which is not the public the rule is about."""
    _snapshot(conn, KICK - timedelta(minutes=30), home_tickets=55)
    _snapshot(conn, KICK + timedelta(hours=1), home_tickets=95)
    _result(conn, 30, 20)

    assert settle(conn) == []


# ----------------------------------------------------------- staleness


@pytest.mark.parametrize(
    "lead, graded",
    [
        (timedelta(hours=3), True),
        (MAX_STALENESS, True),
        (MAX_STALENESS + timedelta(minutes=1), False),
        (timedelta(hours=5), False),
    ],
)
def test_a_reading_too_far_from_kickoff_is_not_gameday_evidence(lead, graded):
    """The scheduler drops runs, so the nearest reading can be hours old. Past
    the limit it says nothing about gameday and the bet is dropped rather than
    graded on a guess -- the boundary itself is asserted, since an off-by-one
    here silently changes how much of the record exists."""
    c = sqlite3.connect(":memory:")
    init_db(c)
    _snapshot(c, KICK - lead, home_tickets=90)
    _result(c, 30, 20)

    assert bool(settle(c)) is graded


# ------------------------------------------------------------- grading


def test_the_unpopular_side_covering_is_a_win(conn):
    """Public on the home team -3.5; we take away +3.5 and they lose by 3."""
    _snapshot(conn, KICK - timedelta(minutes=30), home_tickets=90, line=-3.5)
    _result(conn, 20, 23)

    (s,) = settle(conn)
    assert s.fade.line_label == "AAA +3.5"
    assert s.result == "win"


def test_the_unpopular_side_failing_to_cover_is_a_loss(conn):
    _snapshot(conn, KICK - timedelta(minutes=30), home_tickets=90, line=-3.5)
    _result(conn, 20, 30)

    (s,) = settle(conn)
    assert s.result == "loss"


def test_a_spread_landing_on_the_number_is_a_push_and_not_a_decision(conn):
    """A push returns the stake, so it belongs in the log but not in the win
    rate -- counting it either way would misstate the record."""
    _snapshot(conn, KICK - timedelta(minutes=30), home_tickets=90, line=-3.0)
    _result(conn, 20, 23)

    (s,) = settle(conn)
    assert s.result == "push"

    r = tally([s])
    assert (r.wins, r.losses, r.pushes) == (0, 0, 1)
    assert r.decided == 0 and r.win_rate is None
    assert r.units == 0


def test_a_total_is_graded_on_the_combined_score(conn):
    game = GameSplits(
        away="AAA", home="BBB", kickoff_utc=KICK, status="scheduled", num_bets=5000,
        sides={
            ("total", "under"): Side("total", "under", 47.5, -110, 93, 95),
            ("total", "over"): Side("total", "over", 47.5, -110, 7, 5),
        },
    )
    store_splits(conn, [game], fetched_at=KICK - timedelta(minutes=30))
    _result(conn, 30, 24)  # 54, over 47.5

    (s,) = settle(conn)
    assert s.fade.line_label == "over 47.5"
    assert s.result == "win"


# ----------------------------------------------- the real week, end to end


@pytest.fixture
def week3(conn):
    """Week 3 of 2026, stored the way `fl fades-backfill` stores it."""
    path = pathlib.Path(__file__).parent.parent / "fixtures" / "action_network_w3.json"
    games = [g for g in parse_games(json.loads(path.read_text())) if g.final]
    for g in games:
        store_splits(conn, [g], fetched_at=g.kickoff_utc, is_final=True)
    store_results(conn, games)
    return conn


def test_week_three_reproduces_the_bets_the_rule_actually_made(week3):
    """The one test that pins the rule to a known-good outcome.

    These four are the bets the strategy this board implements made in week 3,
    reported independently of this code. If a future change to the threshold,
    the market filter, or the qualification test quietly alters what gets
    selected, nothing else here would notice -- the live board has no right
    answer to compare against, but this week does.
    """
    settled = settle(week3)
    assert {(s.game, s.fade.line_label) for s in settled} == {
        ("KC @ MIA", "MIA +9.5"),
        ("SEA @ WAS", "WAS +8.5"),
        ("ARI @ SF", "ARI +7.5"),
        ("BAL @ DAL", "over 54.5"),
    }

    r = tally(settled)
    assert (r.wins, r.losses) == (3, 1)
    assert {s.game for s in settled if s.result == "loss"} == {"KC @ MIA"}


def test_a_backfilled_week_is_marked_as_such(week3):
    """The record merges both bases into one number, so the page can only be
    honest about it if the rows remember which they are."""
    settled = settle(week3)
    assert tally(settled).from_final_tally == len(settled) == 4


def test_no_moneyline_reaches_the_record(week3):
    """Week 3 had moneylines at 98% on two games. They are excluded from the
    board by design, and the record must not quietly readmit them."""
    assert all(s.fade.market != "moneyline" for s in settle(week3))
