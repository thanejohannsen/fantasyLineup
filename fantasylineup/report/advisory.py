"""The advisory: what to start, and what it would gain over the current lineup.

Deliberately expressed as a *delta* against the lineup currently set in Sleeper.
Sleeper's API is read-only, so nothing here can be applied automatically -- the
output has to be worth acting on by hand, which means saying plainly what to
change and what it is worth, not just printing an optimal lineup and leaving
the diff as an exercise.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..engine.lineup import Lineup, PlayerProjection, optimize_lineup, starting_slots


@dataclass(frozen=True)
class LineupChange:
    slot: str
    start: PlayerProjection
    sit: PlayerProjection | None
    gain: float


@dataclass
class Advisory:
    week: int
    slots: list[str]
    optimal: Lineup
    current: Lineup
    changes: list[LineupChange]

    @property
    def gain(self) -> float:
        return self.optimal.total_points - self.current.total_points


def current_lineup(
    conn: sqlite3.Connection,
    snapshot_id: int,
    roster_id: int,
    players: list[PlayerProjection],
    slots: list[str],
) -> Lineup:
    """Reconstruct the lineup as actually set in Sleeper.

    ``slot_index`` is the player's position within the roster's ``starters``
    array, which aligns with ``roster_positions`` after bench slots are removed.
    """
    by_id = {p.sleeper_id: p for p in players}
    rows = conn.execute(
        """SELECT sleeper_id, slot_index FROM roster_players
           WHERE snapshot_id = ? AND roster_id = ? AND is_starter = 1""",
        (snapshot_id, roster_id),
    ).fetchall()

    lineup = Lineup(slots=list(slots))
    started: set[str] = set()
    for r in rows:
        idx = r["slot_index"]
        player = by_id.get(r["sleeper_id"])
        if player is None or idx is None or idx >= len(slots):
            continue
        lineup.assignments[idx] = player
        started.add(player.sleeper_id)

    lineup.bench = sorted(
        (p for p in players if p.sleeper_id not in started),
        key=lambda p: p.points,
        reverse=True,
    )
    lineup.unfilled = [i for i in range(len(slots)) if i not in lineup.assignments]
    return lineup


def diff_lineups(optimal: Lineup, current: Lineup, slots: list[str]) -> list[LineupChange]:
    """Slot-by-slot changes, most valuable first.

    Only slots whose occupant actually changes are reported; a player who merely
    moves between two interchangeable slots (RB to FLEX) is not a change worth
    acting on.
    """
    optimal_ids = {p.sleeper_id for p in optimal.assignments.values()}
    current_ids = {p.sleeper_id for p in current.assignments.values()}
    if optimal_ids == current_ids:
        return []

    changes = []
    for idx, slot in enumerate(slots):
        new = optimal.assignments.get(idx)
        old = current.assignments.get(idx)
        if new is None:
            continue
        # Ignore pure reshuffles: the player was already starting somewhere.
        if new.sleeper_id in current_ids:
            continue
        changes.append(
            LineupChange(
                slot=slot,
                start=new,
                sit=old if old is not None and old.sleeper_id not in optimal_ids else None,
                gain=new.points - (old.points if old else 0.0),
            )
        )
    changes.sort(key=lambda c: c.gain, reverse=True)
    return changes


def build_advisory(
    conn: sqlite3.Connection,
    snapshot_id: int,
    roster_id: int,
    players: list[PlayerProjection],
    roster_positions: list[str],
    week: int,
) -> Advisory:
    slots = starting_slots(roster_positions)
    optimal = optimize_lineup(players, slots)
    current = current_lineup(conn, snapshot_id, roster_id, players, slots)
    return Advisory(
        week=week,
        slots=slots,
        optimal=optimal,
        current=current,
        changes=diff_lineups(optimal, current, slots),
    )
