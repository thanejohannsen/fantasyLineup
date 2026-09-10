"""Kickoff times and live game state.

Source is ESPN's **public sports** scoreboard API. That is a different service
from the ESPN *fantasy* API this project deliberately does not use: the fantasy
one needs ``espn_s2``/``SWID`` cookies scraped from a browser and expires
yearly, whereas this endpoint is public, unauthenticated and stable.

Why not the sources already in use:

Sleeper's ``/schedule`` carries a live ``status`` per game but only a *date*,
with no kickoff time -- enough to know a game has started, not enough to say
"locks in 4h 12m".

Kalshi's markets do carry timestamps, but ``occurrence_datetime`` is
approximately game *end*, not kickoff: ATL/PIT reads 20:00Z for a 1pm ET
kickoff, and every game follows the same roughly +3h pattern. Deriving kickoff
by subtracting three hours would be guesswork that silently breaks on overtime
or a schedule change.

ESPN gives the exact kickoff (``2026-09-10T00:20Z``) alongside a live state, so
it answers both questions directly.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

log = logging.getLogger(__name__)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# ESPN spells Washington differently from Sleeper. Verified as the only
# disagreement across all 32 clubs.
_TEAM_ALIASES = {"WSH": "WAS"}

# ESPN status states. 'pre' is the only one where a roster slot is still open.
STATE_PRE = "pre"
STATE_POST = "post"

# A regulation game is four fifteen-minute quarters.
_QUARTER_SECONDS = 900.0
_REGULATION_SECONDS = 4 * _QUARTER_SECONDS


def normalize_team(abbr: str | None) -> str | None:
    if abbr is None:
        return None
    return _TEAM_ALIASES.get(abbr, abbr)


def remaining_fraction(state: str | None, period: int | None, clock: float | None) -> float:
    """How much of a game is still to be played, from 1.0 to 0.0.

    A player mid-game has banked part of his projection and still owns the rest,
    so neither treating him as finished nor as untouched is right. Overtime
    clamps to zero: it is unscheduled time no projection anticipated, and
    crediting a player for more than a full game because his game ran long would
    be worse than crediting him for none of it.
    """
    if state is None or state == STATE_PRE:
        return 1.0
    if state == STATE_POST:
        return 0.0
    left = (4 - (period or 0)) * _QUARTER_SECONDS + (clock or 0.0)
    return max(0.0, min(1.0, left / _REGULATION_SECONDS))


@dataclass(frozen=True)
class Kickoff:
    week: int
    home: str
    away: str
    kickoff_utc: datetime
    state: str
    period: int = 0
    clock: float = 0.0

    @property
    def started(self) -> bool:
        return self.state != STATE_PRE

    @property
    def remaining_fraction(self) -> float:
        return remaining_fraction(self.state, self.period, self.clock)

    def involves(self, team: str | None) -> bool:
        return team is not None and team in (self.home, self.away)


def fetch_kickoffs(season: int, week: int, timeout: float = 30.0) -> list[Kickoff]:
    params = {"dates": season, "seasontype": 2, "week": week}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(SCOREBOARD_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

    out: list[Kickoff] = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competitors = {
            c.get("homeAway"): normalize_team((c.get("team") or {}).get("abbreviation"))
            for c in competitions[0].get("competitors", [])
        }
        home, away = competitors.get("home"), competitors.get("away")
        if not home or not away:
            continue
        try:
            kickoff = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            log.warning("Unparseable kickoff for %s", event.get("id"))
            continue
        out.append(
            Kickoff(
                week=week,
                home=home,
                away=away,
                kickoff_utc=kickoff,
                state=((event.get("status") or {}).get("type") or {}).get("state", STATE_PRE),
                period=int((event.get("status") or {}).get("period") or 0),
                clock=float((event.get("status") or {}).get("clock") or 0.0),
            )
        )
    return out


def sync_kickoffs(conn: sqlite3.Connection, season: int, week: int) -> int:
    """Store kickoff time and live state onto the existing games rows.

    Matched on (week, home, away) rather than a shared game id, because ESPN and
    Sleeper number games differently.
    """
    kickoffs = fetch_kickoffs(season, week)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    updated = 0
    for k in kickoffs:
        cur = conn.execute(
            """UPDATE games SET kickoff_utc = ?, status = ?, period = ?, clock = ?,
                   updated_at = ?
               WHERE season = ? AND week = ? AND home = ? AND away = ?""",
            (
                k.kickoff_utc.isoformat(), k.state, k.period, k.clock, now,
                season, week, k.home, k.away,
            ),
        )
        updated += cur.rowcount
    conn.commit()
    if updated < len(kickoffs):
        log.warning(
            "Only matched %d of %d ESPN games to the Sleeper schedule for week %d",
            updated,
            len(kickoffs),
            week,
        )
    return updated
