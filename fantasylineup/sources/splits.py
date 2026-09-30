"""How the betting public is distributed across each side of a market.

Source is Action Network's public web scoreboard, which reports, per game and
per market, what share of **tickets** and what share of **money** sits on each
side. Those two are different questions and the gap between them is the whole
point: eighty per cent of tickets alongside half the money means the minority is
betting far larger, which is the only thing that distinguishes a crowd from an
informed one.

No key, no auth. It is also undocumented and not ours, so it is wrapped here
behind one module with a fixture-backed test, and every failure degrades the
page rather than breaking the refresh -- the same treatment ESPN's scoreboard
gets in ``kickoffs.py``.

The totals look wrong, and are not
----------------------------------
Sampled across 2026 weeks 1-4, this feed puts the **under** ahead on tickets in
52 of 64 games, mean 64%. That inverts the best-documented bias in betting --
the public buys overs -- while the same payload puts spread favourites at 59% of
tickets, matching expectation exactly. One field looking right beside a
neighbour looking transposed is worth checking, and it mattered: flipping the
totals labels moves a three-week backtest of the contrarian rule from +26.6% ROI
to -1.3%.

Checked on 2026-09-30 against independent public boards, on two live games
pointing in *opposite* directions -- which rules a systematic transposition out,
rather than merely failing to find one:

    IND @ WAS 47.5   ours: under 93% / over 7%   VegasInsider:  under 93% / over 7%
    PIT @ CLE 38.5   ours: over 85%              ScoresAndOdds: over 85% / under 15%

The money share agreed too (95% on the under at IND @ WAS), and that independent
board showed unders leading in 12 of its own 16 games the same day. So the
lopsided pattern belongs to published consensus data this season, not to this
parser, and why the crowd sits on unders is a real question but not ours.

Both sides and their shares are still carried through to the page. That is what
made this check a glance instead of a project, and it keeps the next one cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

log = logging.getLogger(__name__)

SCOREBOARD_URL = "https://api.actionnetwork.com/web/v2/scoreboard/nfl"

# Every book id returns the same consensus figures, so these are requested only
# because the endpoint expects some. 15 is preferred when present purely to make
# the choice deterministic rather than dependent on dict ordering.
BOOK_IDS = (15, 30, 68, 69, 71, 75)
PREFERRED_BOOK = "15"

# A browser agent: the endpoint serves a web app and is unreliable without one.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}

MARKETS = ("spread", "total", "moneyline")

# Sides that oppose each other, so a fade can find its counterpart.
OPPOSITE = {"home": "away", "away": "home", "over": "under", "under": "over"}


@dataclass(frozen=True)
class Side:
    """One side of one market, and how the public is split across it."""

    market: str
    side: str
    line: float | None
    odds: int | None
    tickets: int
    money: int | None


@dataclass(frozen=True)
class GameSplits:
    away: str
    home: str
    kickoff_utc: datetime | None
    status: str
    num_bets: int
    sides: dict[tuple[str, str], Side] = field(default_factory=dict)
    # Final score, once there is one. The same payload carries the boxscore, so
    # grading a fade needs no second source.
    away_points: int | None = None
    home_points: int | None = None

    @property
    def label(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def started(self) -> bool:
        return self.status not in ("scheduled", "pre", "")

    @property
    def final(self) -> bool:
        return self.away_points is not None and self.home_points is not None

    def opposite(self, side: Side) -> Side | None:
        return self.sides.get((side.market, OPPOSITE.get(side.side, "")))


def _parse_kickoff(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        log.warning("Unparseable kickoff %r", raw)
        return None


def parse_games(payload: dict) -> list[GameSplits]:
    """Turn one scoreboard payload into games, split-by-split.

    Kept separate from the fetch so it can be tested against a captured
    response without the network.
    """
    out: list[GameSplits] = []
    for game in payload.get("games") or []:
        teams = {t.get("id"): t.get("abbr") for t in game.get("teams") or []}
        away = teams.get(game.get("away_team_id"))
        home = teams.get(game.get("home_team_id"))
        if not away or not home:
            continue

        sides: dict[tuple[str, str], Side] = {}
        for book_id, periods in (game.get("markets") or {}).items():
            for market in MARKETS:
                for entry in (periods.get("event") or {}).get(market) or []:
                    info = entry.get("bet_info") or {}
                    tickets = (info.get("tickets") or {}).get("percent")
                    side = entry.get("side")
                    if tickets is None or not side:
                        continue
                    key = (market, side)
                    # First writer wins unless the preferred book turns up.
                    if key in sides and str(book_id) != PREFERRED_BOOK:
                        continue
                    sides[key] = Side(
                        market=market,
                        side=side,
                        line=entry.get("value"),
                        odds=entry.get("odds"),
                        tickets=int(tickets),
                        money=(info.get("money") or {}).get("percent"),
                    )

        box = game.get("boxscore") or {}
        out.append(
            GameSplits(
                away=away,
                home=home,
                kickoff_utc=_parse_kickoff(game.get("start_time")),
                status=str(game.get("status") or ""),
                num_bets=int(game.get("num_bets") or 0),
                sides=sides,
                away_points=box.get("total_away_points"),
                home_points=box.get("total_home_points"),
            )
        )
    return out


def fetch_splits(
    timeout: float = 30.0, week: int | None = None, season: int | None = None
) -> list[GameSplits]:
    """Ticket and money splits for every NFL game on the board.

    With no ``week`` this is the current slate. Naming a completed week returns
    it with final scores attached, which is how the record gets backfilled --
    though the percentages a finished week serves are the closing tally, not
    what was showing before kickoff. ``store_splits(..., is_final=True)`` is
    what keeps that distinction.

    Returns an empty list on any failure. A betting board that cannot be built
    should leave its tab empty, not take the lineup advice down with it.
    """
    params: dict[str, str | int] = {
        "period": "game",
        "bookIds": ",".join(str(b) for b in BOOK_IDS),
    }
    if week is not None:
        params["week"] = week
    if season is not None:
        params["season"] = season
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_HEADERS) as client:
            resp = client.get(SCOREBOARD_URL, params=params)
            resp.raise_for_status()
            games = parse_games(resp.json())
    except Exception as exc:  # noqa: BLE001 - degrade rather than fail the run
        log.warning("Betting splits unavailable: %s", exc)
        return []

    quoted = sum(len(g.sides) for g in games)
    log.info("Betting splits: %d games, %d quoted sides", len(games), quoted)
    return games


def store_splits(
    conn,
    games: list[GameSplits],
    fetched_at: datetime | None = None,
    is_final: bool = False,
) -> int:
    """Append what this refresh saw, one row per quoted side.

    Append-only and keyed by fetch time, so the history is a record of what was
    observable before kickoff. That is exactly what the fade record is graded
    from, and grading against splits pulled after the fact would be scoring the
    rule on information it never had.

    ``is_final`` marks the exception: rows backfilled from a completed week,
    whose percentages are the closing tally because that is all the feed still
    serves for a finished game. They are stamped with the kickoff they belong to
    so the ordinary grading path picks them up, and flagged so the page can say
    how much of the record rests on them.
    """
    stamp = (fetched_at or datetime.now(UTC)).isoformat(timespec="seconds")
    rows = [
        (
            stamp,
            g.kickoff_utc.isoformat() if g.kickoff_utc else None,
            g.away,
            g.home,
            s.market,
            s.side,
            s.line,
            s.odds,
            s.tickets,
            s.money,
            g.num_bets,
            int(is_final),
        )
        for g in games
        for s in g.sides.values()
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO betting_splits
               (fetched_at, kickoff_utc, away, home, market, side,
                line, odds, tickets_pct, money_pct, num_bets, is_final)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def store_results(conn, games: list[GameSplits], recorded_at: datetime | None = None) -> int:
    """Record the final score of every game that has one.

    Without a kickoff there is nothing to match a snapshot against, so those are
    skipped rather than stored under a null key.
    """
    stamp = (recorded_at or datetime.now(UTC)).isoformat(timespec="seconds")
    rows = [
        (g.away, g.home, g.kickoff_utc.isoformat(), g.away_points, g.home_points, stamp)
        for g in games
        if g.final and g.kickoff_utc is not None
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO game_results
               (away, home, kickoff_utc, away_points, home_points, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)
