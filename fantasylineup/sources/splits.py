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

    @property
    def label(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def started(self) -> bool:
        return self.status not in ("scheduled", "pre", "")

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

        out.append(
            GameSplits(
                away=away,
                home=home,
                kickoff_utc=_parse_kickoff(game.get("start_time")),
                status=str(game.get("status") or ""),
                num_bets=int(game.get("num_bets") or 0),
                sides=sides,
            )
        )
    return out


def fetch_splits(timeout: float = 30.0) -> list[GameSplits]:
    """Current ticket and money splits for every NFL game on the board.

    Returns an empty list on any failure. A betting board that cannot be built
    should leave its tab empty, not take the lineup advice down with it.
    """
    params = {"period": "game", "bookIds": ",".join(str(b) for b in BOOK_IDS)}
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
    conn, games: list[GameSplits], fetched_at: datetime | None = None
) -> int:
    """Append what this refresh saw, one row per quoted side.

    Append-only and keyed by fetch time, so the history is a record of what was
    observable before kickoff. Grading a contrarian rule against splits pulled
    after the fact would be scoring it on information it never had.
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
        )
        for g in games
        for s in g.sides.values()
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO betting_splits
               (fetched_at, kickoff_utc, away, home, market, side,
                line, odds, tickets_pct, money_pct, num_bets)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)
