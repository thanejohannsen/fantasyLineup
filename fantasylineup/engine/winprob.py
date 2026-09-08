"""Choosing a lineup to maximise win probability rather than expected points.

The search is a hill climb seeded with the expected-points optimum. That seed is
not arbitrary: when a matchup is close the two objectives coincide, so the EV
lineup is already the answer and the climb terminates immediately. The climb
only travels when the matchup is lopsided enough that risk posture matters,
which is exactly when it should.

Candidate moves are single swaps between a starter and an eligible bench player.
Each candidate is scored by simulation against the opponent's projected lineup,
including any points already banked on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lineup import Lineup, PlayerProjection, optimize_lineup
from .simulate import SimulatedOutcome, win_probability

# Below this the swap is noise, not signal: the simulation's own standard error
# at 20k draws is a few tenths of a percent.
MIN_IMPROVEMENT = 0.005


@dataclass
class WinProbLineup:
    lineup: Lineup
    outcome: SimulatedOutcome
    ev_outcome: SimulatedOutcome
    swaps_applied: int

    @property
    def improved(self) -> bool:
        return self.swaps_applied > 0

    @property
    def gain(self) -> float:
        return self.outcome.win_probability - self.ev_outcome.win_probability


def _swap(lineup: Lineup, slot: int, incoming: PlayerProjection) -> Lineup:
    replaced = lineup.assignments.get(slot)
    new = Lineup(
        slots=list(lineup.slots),
        assignments=dict(lineup.assignments),
        bench=[p for p in lineup.bench if p.sleeper_id != incoming.sleeper_id],
        unfilled=list(lineup.unfilled),
    )
    new.assignments[slot] = incoming
    if replaced is not None:
        new.bench.append(replaced)
    return new


def optimize_for_win_probability(
    players: list[PlayerProjection],
    slots: list[str],
    opponent_starters: list[PlayerProjection],
    sds: dict[str, float],
    forced: dict[int, str] | None = None,
    draws: int = 20000,
    my_banked: float = 0.0,
    opponent_banked: float = 0.0,
    max_iterations: int = 12,
) -> WinProbLineup:
    """Hill climb from the expected-points optimum toward the best win chance."""
    forced = forced or {}
    current = optimize_lineup(players, slots, forced=forced)

    def evaluate(lineup: Lineup) -> SimulatedOutcome:
        return win_probability(
            lineup.starters,
            opponent_starters,
            sds,
            draws=draws,
            my_banked=my_banked,
            opponent_banked=opponent_banked,
        )

    ev_outcome = evaluate(current)
    best_outcome = ev_outcome
    swaps = 0

    for _ in range(max_iterations):
        best_move: tuple[int, PlayerProjection, SimulatedOutcome] | None = None

        for slot_idx, slot in enumerate(current.slots):
            if slot_idx in forced:
                continue  # already played; cannot be changed
            for candidate in current.bench:
                if not candidate.eligible_for(slot):
                    continue
                trial = _swap(current, slot_idx, candidate)
                outcome = evaluate(trial)
                if outcome.win_probability > best_outcome.win_probability + MIN_IMPROVEMENT and (
                    best_move is None or outcome.win_probability > best_move[2].win_probability
                ):
                    best_move = (slot_idx, candidate, outcome)

        if best_move is None:
            break

        slot_idx, candidate, outcome = best_move
        current = _swap(current, slot_idx, candidate)
        best_outcome = outcome
        swaps += 1

    return WinProbLineup(
        lineup=current,
        outcome=best_outcome,
        ev_outcome=ev_outcome,
        swaps_applied=swaps,
    )


def posture(win_prob: float) -> str:
    """Plain-language read on what the matchup asks for.

    Thresholds are a presentation choice, not a modelling one -- the search
    optimises the continuous probability regardless. They exist so the report
    can say *why* it is recommending a lower-projected player.
    """
    if win_prob >= 0.65:
        return "heavy favourite: protect the floor"
    if win_prob <= 0.35:
        return "heavy underdog: chase the ceiling"
    return "close matchup: maximise expected points"
