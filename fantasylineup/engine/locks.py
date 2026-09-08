"""Which roster decisions are still open, and how long is left to make them.

A fantasy week is not one deadline but several. Each player locks at his own
game's kickoff, so by Sunday evening most of the roster is frozen while a
Monday-night player is still live. Advice that ignores this is wrong in both
directions: it proposes swaps that can no longer be made, and it hides the ones
that still can.

Locking is judged on two independent signals, and either is sufficient:

* the game's live state has left ``pre``
* the current time has passed the recorded kickoff

Both are kept because each fails differently. Live state can lag a few minutes
behind an actual kickoff, and a stale or missing kickoff time would otherwise
freeze a player who is still startable. Treating either signal as decisive is
the safe direction to be wrong in -- proposing a change that cannot be made is
worse than declining to propose one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from .lineup import PlayerProjection

# Games whose live state is anything other than this have begun.
STATE_PRE = "pre"


@dataclass(frozen=True)
class LockState:
    """Whether one player's slot can still be changed."""

    locked: bool
    kickoff_utc: datetime | None
    opponent: str | None = None
    reason: str = ""

    def seconds_remaining(self, now: datetime) -> float | None:
        if self.locked or self.kickoff_utc is None:
            return None
        return max(0.0, (self.kickoff_utc - now).total_seconds())


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def team_lock_states(
    conn: sqlite3.Connection, season: int, week: int, now: datetime | None = None
) -> dict[str, LockState]:
    """Lock state for every team playing in the given week."""
    now = now or datetime.now(UTC)
    states: dict[str, LockState] = {}

    for row in conn.execute(
        "SELECT home, away, status, kickoff_utc FROM games WHERE season = ? AND week = ?",
        (season, week),
    ):
        kickoff = _parse(row["kickoff_utc"])
        started_by_state = row["status"] is not None and row["status"] != STATE_PRE
        started_by_clock = kickoff is not None and now >= kickoff

        if started_by_state and started_by_clock:
            reason = "game in progress"
        elif started_by_state:
            reason = f"game state is {row['status']}"
        elif started_by_clock:
            reason = "kickoff has passed"
        else:
            reason = ""

        for team, opponent in ((row["home"], row["away"]), (row["away"], row["home"])):
            if team:
                states[team] = LockState(
                    locked=started_by_state or started_by_clock,
                    kickoff_utc=kickoff,
                    opponent=opponent,
                    reason=reason,
                )
    return states


def player_lock_states(
    conn: sqlite3.Connection,
    players: list[PlayerProjection],
    season: int,
    week: int,
    now: datetime | None = None,
) -> dict[str, LockState]:
    """Lock state per player, derived from his team's game.

    A player whose team has no game this week is on a bye. He is reported as
    unlocked with no kickoff: there is no deadline to act on, and he should not
    be treated as frozen.
    """
    by_team = team_lock_states(conn, season, week, now=now)
    out: dict[str, LockState] = {}
    for player in players:
        state = by_team.get(player.team or "")
        out[player.sleeper_id] = state or LockState(
            locked=False, kickoff_utc=None, reason="no game this week (bye)"
        )
    return out


def next_deadline(
    lock_states: dict[str, LockState], now: datetime | None = None
) -> datetime | None:
    """Earliest kickoff among players who are still changeable."""
    now = now or datetime.now(UTC)
    upcoming = [
        s.kickoff_utc
        for s in lock_states.values()
        if not s.locked and s.kickoff_utc is not None and s.kickoff_utc > now
    ]
    return min(upcoming) if upcoming else None


def format_countdown(target: datetime | None, now: datetime | None = None) -> str:
    if target is None:
        return "no deadline"
    now = now or datetime.now(UTC)
    seconds = (target - now).total_seconds()
    if seconds <= 0:
        return "locked"
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def locked_slot_assignments(
    conn: sqlite3.Connection,
    snapshot_id: int,
    roster_id: int,
    lock_states: dict[str, LockState],
) -> dict[int, str]:
    """Slot index -> player id for starters who can no longer be moved.

    Feeds ``optimize_lineup(forced=...)`` so the remaining slots are optimised
    around choices that are already final.
    """
    forced: dict[int, str] = {}
    for row in conn.execute(
        """SELECT sleeper_id, slot_index FROM roster_players
           WHERE snapshot_id = ? AND roster_id = ? AND is_starter = 1""",
        (snapshot_id, roster_id),
    ):
        state = lock_states.get(row["sleeper_id"])
        if row["slot_index"] is not None and state is not None and state.locked:
            forced[int(row["slot_index"])] = row["sleeper_id"]
    return forced
