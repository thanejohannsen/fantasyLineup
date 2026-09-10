"""The advisory: what to start, and what it would gain over the current lineup.

Deliberately expressed as a *delta* against the lineup currently set in Sleeper.
Sleeper's API is read-only, so nothing here can be applied automatically -- the
output has to be worth acting on by hand, which means saying plainly what to
change and what it is worth, not just printing an optimal lineup and leaving the
diff as an exercise.

Where an opponent is known the recommendation optimises win probability rather
than expected points, so the advice shifts toward the floor when ahead and the
ceiling when behind.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..engine.lineup import Lineup, PlayerProjection, optimize_lineup, starting_slots
from ..engine.locks import LockState, locked_slot_assignments, next_deadline, player_lock_states
from ..engine.simulate import LiveState, SimulatedOutcome
from ..engine.winprob import optimize_for_win_probability, posture


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
    generated_at: datetime
    lock_states: dict[str, LockState] = field(default_factory=dict)
    deadline: datetime | None = None
    opponent_name: str | None = None
    outcome: SimulatedOutcome | None = None
    ev_outcome: SimulatedOutcome | None = None
    winprob_swaps: int = 0
    my_banked: float = 0.0
    opponent_banked: float = 0.0

    @property
    def gain(self) -> float:
        return self.optimal.total_points - self.current.total_points

    @property
    def locked_players(self) -> list[PlayerProjection]:
        return [
            p
            for p in self.current.assignments.values()
            if self.lock_states.get(p.sleeper_id, LockState(False, None)).locked
        ]

    @property
    def posture(self) -> str | None:
        if self.outcome is None:
            return None
        return posture(self.outcome.win_probability)


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
        if new is None or new.sleeper_id in current_ids:
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
    season: int,
    week: int,
    now: datetime | None = None,
    opponent_starters: list[PlayerProjection] | None = None,
    opponent_name: str | None = None,
    sds: dict[str, float] | None = None,
    my_banked: float = 0.0,
    opponent_banked: float = 0.0,
    live: dict[str, LiveState] | None = None,
    draws: int = 20000,
) -> Advisory:
    """Recommend a lineup, respecting what has already locked.

    Starters whose games have kicked off are pinned and the remaining slots
    optimised around them, so every change proposed is one that can still
    actually be made in the app. With an opponent supplied the objective becomes
    win probability; without one it stays expected points.
    """
    now = now or datetime.now(UTC)
    slots = starting_slots(roster_positions)

    lock_states = player_lock_states(conn, players, season, week, now=now)
    forced = locked_slot_assignments(conn, snapshot_id, roster_id, lock_states)

    outcome = ev_outcome = None
    swaps = 0
    if opponent_starters and sds:
        result = optimize_for_win_probability(
            players=players,
            slots=slots,
            opponent_starters=opponent_starters,
            sds=sds,
            forced=forced,
            draws=draws,
            live=live,
        )
        optimal, outcome, ev_outcome, swaps = (
            result.lineup,
            result.outcome,
            result.ev_outcome,
            result.swaps_applied,
        )
    else:
        optimal = optimize_lineup(players, slots, forced=forced)

    current = current_lineup(conn, snapshot_id, roster_id, players, slots)
    return Advisory(
        week=week,
        slots=slots,
        optimal=optimal,
        current=current,
        changes=diff_lineups(optimal, current, slots),
        generated_at=now,
        lock_states=lock_states,
        deadline=next_deadline(lock_states, now=now),
        opponent_name=opponent_name,
        outcome=outcome,
        ev_outcome=ev_outcome,
        winprob_swaps=swaps,
        my_banked=my_banked,
        opponent_banked=opponent_banked,
    )
