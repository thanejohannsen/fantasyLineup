"""Optimal starting lineup selection.

Filling a fantasy lineup is an assignment problem, not a sorting problem. With
two FLEX slots a greedy "start the highest projected players" pass can be
strictly worse than optimal: taking the best available WR into FLEX may strand
a slot that only an RB could have filled. So this solves it exactly, as
maximum-weight bipartite matching between players and slots via the Hungarian
algorithm.

The same routine underpins waivers and trades. A player's worth to a team is
``LV(roster + p) - LV(roster)`` -- his *marginal* contribution to the best legal
lineup, not his raw projection. That is what lets the engine see that a team
with four good running backs gains little from a fifth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# Which fantasy positions may fill each roster slot. Slot names follow Sleeper's
# ``roster_positions`` vocabulary; unknown slots are treated as unfillable
# rather than silently accepting anyone.
SLOT_ELIGIBILITY: dict[str, frozenset[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "DEF": frozenset({"DEF"}),
    "DL": frozenset({"DL"}),
    "LB": frozenset({"LB"}),
    "DB": frozenset({"DB"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "WRRB_FLEX": frozenset({"RB", "WR"}),
    "REC_FLEX": frozenset({"WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
    "IDP_FLEX": frozenset({"DL", "LB", "DB"}),
}

# Slots that never hold an active starter.
NON_STARTING_SLOTS = frozenset({"BN", "IR", "TAXI"})

# Finite stand-in for "illegal". Must dominate any real point total but stay
# finite: linear_sum_assignment cannot work with infinities.
_ILLEGAL = 1e6


@dataclass(frozen=True)
class PlayerProjection:
    """One player's candidacy for a lineup slot."""

    sleeper_id: str
    name: str
    position: str
    points: float
    fantasy_positions: frozenset[str] = frozenset()
    locked: bool = False
    team: str | None = None
    opponent: str | None = None

    def eligible_for(self, slot: str) -> bool:
        allowed = SLOT_ELIGIBILITY.get(slot)
        if allowed is None:
            return False
        positions = self.fantasy_positions or frozenset({self.position})
        return bool(positions & allowed)


@dataclass
class Lineup:
    """A solved lineup: slot index -> player, plus anything left over."""

    slots: list[str]
    assignments: dict[int, PlayerProjection] = field(default_factory=dict)
    bench: list[PlayerProjection] = field(default_factory=list)
    unfilled: list[int] = field(default_factory=list)

    @property
    def total_points(self) -> float:
        return sum(p.points for p in self.assignments.values())

    @property
    def starters(self) -> list[PlayerProjection]:
        return [self.assignments[i] for i in sorted(self.assignments)]

    def describe(self) -> list[tuple[str, PlayerProjection | None]]:
        return [(slot, self.assignments.get(i)) for i, slot in enumerate(self.slots)]


def starting_slots(roster_positions: list[str]) -> list[str]:
    """Strip bench/IR/taxi entries, leaving only slots that score."""
    return [s for s in roster_positions if s not in NON_STARTING_SLOTS]


def optimize_lineup(
    players: list[PlayerProjection],
    slots: list[str],
    forced: dict[int, str] | None = None,
) -> Lineup:
    """Best legal assignment of players to slots.

    ``forced`` pins specific slot indices to specific player ids -- used for
    players whose games have already kicked off, since those choices can no
    longer be changed and must be treated as fixed while the rest is
    re-optimised around them.

    Slots with no eligible player left (a bye-week hole, or a position wiped out
    by injury) are reported in ``unfilled`` rather than being filled illegally.
    Sleeper permits an empty slot, and pretending otherwise would quietly
    inflate the projected total.
    """
    lineup = Lineup(slots=list(slots))
    if not slots:
        lineup.bench = list(players)
        return lineup

    forced = forced or {}
    by_id = {p.sleeper_id: p for p in players}

    # Pin forced assignments first, then solve over what remains.
    remaining_slots = []
    for idx, slot in enumerate(slots):
        pinned_id = forced.get(idx)
        if pinned_id is not None:
            player = by_id.get(pinned_id)
            if player is None:
                raise ValueError(f"Forced player {pinned_id} is not in the supplied roster")
            lineup.assignments[idx] = player
        else:
            remaining_slots.append(idx)

    used = {p.sleeper_id for p in lineup.assignments.values()}
    candidates = [p for p in players if p.sleeper_id not in used]

    if not remaining_slots:
        lineup.bench = candidates
        return lineup
    if not candidates:
        lineup.unfilled = remaining_slots
        return lineup

    # Rows are candidates, columns are the slots still to fill. Cost is negated
    # points so the minimising solver maximises the lineup total.
    cost = np.full((len(candidates), len(remaining_slots)), _ILLEGAL, dtype=float)
    for r, player in enumerate(candidates):
        for c, slot_idx in enumerate(remaining_slots):
            if player.eligible_for(slots[slot_idx]):
                cost[r, c] = -player.points

    rows, cols = linear_sum_assignment(cost)

    assigned_players: set[str] = set()
    for r, c in zip(rows, cols):
        slot_idx = remaining_slots[c]
        if cost[r, c] >= _ILLEGAL:
            # No eligible player existed for this slot; leave it empty.
            lineup.unfilled.append(slot_idx)
            continue
        player = candidates[r]
        lineup.assignments[slot_idx] = player
        assigned_players.add(player.sleeper_id)

    # A slot can also go unfilled when there are fewer candidates than slots.
    filled = set(lineup.assignments) | set(lineup.unfilled)
    lineup.unfilled.extend(i for i in remaining_slots if i not in filled)
    lineup.unfilled.sort()

    lineup.bench = [p for p in candidates if p.sleeper_id not in assigned_players]
    lineup.bench.sort(key=lambda p: p.points, reverse=True)
    return lineup


def lineup_value(players: list[PlayerProjection], slots: list[str]) -> float:
    """Expected points from optimally starting this roster.

    The primitive behind every marginal-value calculation in the project.
    """
    return optimize_lineup(players, slots).total_points


def marginal_value(
    players: list[PlayerProjection], slots: list[str], candidate: PlayerProjection
) -> float:
    """How much adding ``candidate`` would improve the best legal lineup."""
    base = lineup_value(players, slots)
    return lineup_value([*players, candidate], slots) - base
