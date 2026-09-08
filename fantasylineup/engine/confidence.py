"""Whether a trade's advantage survives being wrong about something.

A marginal-lineup-value gain is a difference of two optimisation results, and
differences of optimisations are unstable in a way the number itself hides. A
projection moving a point or two can flip which bench player occupies a FLEX
slot, and the reported gain jumps discretely. One live proposal was quoted at
+23.9, then +8.3, then -7.8 across successive refreshes of the same sources
before dropping out entirely -- the same trade, the same rosters, no new
information. Printed to one decimal place it read like an edge; it was noise.

Two independent things are checked here, because a gain can fail either way.

**Stability.** Perturb every projection by a plausible error and recompute the
gain a few hundred times. A real advantage survives; an artefact of one slot
assignment does not. Reporting a range rather than a point is the honest form
when the range is wide.

**Contingency.** Ask which player the gain actually rests on. When a trade
sends two starters for one, the lineup hole is filled by promoting someone off
the bench, and the whole advantage can belong to that promoted player rather
than to the players being exchanged. In the live case the entire +8.3 came from
promoting a tight end returning from Achilles surgery, and it collapsed to -32.1
if he underperformed by a quarter. Trading on that is trading on an unproven
projection while calling it a lineup upgrade.

Contingency also has a cliff shape worth understanding: once the promoted player
falls below the next bench alternative, further decline costs nothing, because
the alternative simply takes the slot. So the downside is bounded but arrives
quickly, and a gain that only exists at 100% of a projection has no margin at
all.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .lineup import PlayerProjection, lineup_value, optimize_lineup
from .trades import TradeProposal

# Plausible relative error on a rest-of-season projection. Deliberately modest:
# refresh-to-refresh movement on the live blend was of this order, and the point
# is to test the gain against ordinary drift rather than against catastrophe.
DEFAULT_JITTER = 0.03
DEFAULT_DRAWS = 200

# Games of current-season evidence before a trade may lean on a player carrying
# an injury designation. Snap share and target share stabilise inside this
# window where fantasy points, being touchdown-driven, do not.
DEFAULT_MIN_GAMES = 3


@dataclass(frozen=True)
class Contingency:
    """A player the trade's advantage depends on, and what happens without him."""

    player: PlayerProjection
    gain_without: float
    games_played: int
    min_games: int = DEFAULT_MIN_GAMES

    @property
    def is_injured(self) -> bool:
        return bool(self.player.injury_status)

    @property
    def is_unproven(self) -> bool:
        """Carrying a designation and with too little evidence to lean on.

        Only relevant for a player we would be *relying* on. A player being sent
        is never tested this way: selling an injured asset is best done before
        the market prices the injury, so requiring evidence there would suppress
        exactly the trades worth making first.
        """
        return self.is_injured and self.games_played < self.min_games


@dataclass(frozen=True)
class TradeConfidence:
    """How much of the reported gain is real."""

    gain: float
    p10: float
    p90: float
    contingencies: tuple[Contingency, ...]
    require_stable: bool = True

    @property
    def is_stable(self) -> bool:
        """Positive across the great majority of plausible projection errors."""
        return self.p10 > 0 or not self.require_stable

    @property
    def blocking(self) -> tuple[Contingency, ...]:
        """Contingencies on players we have no business relying on yet."""
        return tuple(c for c in self.contingencies if c.is_unproven)

    @property
    def is_offerable(self) -> bool:
        return self.is_stable and not self.blocking

    @property
    def spread(self) -> float:
        return self.p90 - self.p10

    def describe(self) -> str:
        if not self.is_stable:
            return f"unstable: {self.p10:+.0f} to {self.p90:+.0f} under normal projection error"
        if self.blocking:
            names = ", ".join(c.player.name for c in self.blocking)
            return f"depends on {names}, with too little evidence this season"
        return f"holds up: {self.p10:+.0f} to {self.p90:+.0f}"


def _swap(
    roster: list[PlayerProjection],
    out_players: tuple[PlayerProjection, ...],
    in_players: tuple[PlayerProjection, ...],
) -> list[PlayerProjection]:
    outgoing = {p.sleeper_id for p in out_players}
    return [p for p in roster if p.sleeper_id not in outgoing] + list(in_players)


def find_contingencies(
    proposal: TradeProposal,
    my_roster: list[PlayerProjection],
    slots: list[str],
    games_played: dict[str, int] | None = None,
    min_gain: float = 3.0,
    min_games: int = DEFAULT_MIN_GAMES,
) -> tuple[Contingency, ...]:
    """Players promoted off the bench whom the gain actually rests on.

    A player is a contingency when removing his contribution takes the gain
    below the threshold that justified proposing the trade. He is tested by
    zeroing him and re-solving, so the optimiser backfills with the real next
    alternative rather than an assumed one.
    """
    games_played = games_played or {}
    before = {p.sleeper_id for p in optimize_lineup(my_roster, slots).starters}
    after_roster = _swap(my_roster, proposal.give, proposal.get)
    after_lineup = optimize_lineup(after_roster, slots)
    acquired = {p.sleeper_id for p in proposal.get}

    promoted = [
        p
        for p in after_lineup.starters
        if p.sleeper_id not in before and p.sleeper_id not in acquired
    ]
    if not promoted:
        return ()

    base = lineup_value(my_roster, slots)
    out: list[Contingency] = []
    for player in promoted:
        without = [
            dataclasses.replace(p, points=0.0) if p.sleeper_id == player.sleeper_id else p
            for p in after_roster
        ]
        gain_without = lineup_value(without, slots) - base
        if gain_without < min_gain:
            out.append(
                Contingency(
                    player=player,
                    gain_without=gain_without,
                    games_played=games_played.get(player.sleeper_id, 0),
                    min_games=min_games,
                )
            )
    return tuple(out)


def measure_stability(
    proposal: TradeProposal,
    my_roster: list[PlayerProjection],
    slots: list[str],
    jitter: float = DEFAULT_JITTER,
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260908,
) -> tuple[float, float]:
    """Tenth and ninetieth percentile of the gain under plausible projection error.

    Each draw perturbs every player once and uses that same perturbation on both
    sides of the difference, so what is measured is the sensitivity of the
    *slot assignment* rather than added noise. Seeded, because advice that
    flickers between refreshes is worse than advice that is merely uncertain.
    """
    rng = np.random.default_rng(seed)
    # Sorted, not a set. Seeding the generator fixes the *sequence* of draws;
    # it does not fix which player each draw lands on. Iterating a set of string
    # ids does that, and CPython salts string hashes per process, so the same
    # inputs gave a different range on every run -- p90 moved by four points
    # across five processes. The seed looked like it was working because a test
    # compared two calls inside one process, where set order is stable.
    involved = sorted({p.sleeper_id for p in (*my_roster, *proposal.give, *proposal.get)})

    gains = np.empty(draws)
    for i in range(draws):
        factors = {
            pid: max(0.0, float(f))
            for pid, f in zip(involved, rng.normal(1.0, jitter, len(involved)))
        }

        def shake(players):
            return [
                dataclasses.replace(p, points=p.points * factors.get(p.sleeper_id, 1.0))
                for p in players
            ]

        mine = shake(my_roster)
        after = _swap(mine, tuple(shake(proposal.give)), tuple(shake(proposal.get)))
        gains[i] = lineup_value(after, slots) - lineup_value(mine, slots)

    return float(np.percentile(gains, 10)), float(np.percentile(gains, 90))


def assess(
    proposal: TradeProposal,
    my_roster: list[PlayerProjection],
    slots: list[str],
    games_played: dict[str, int] | None = None,
    min_gain: float = 3.0,
    jitter: float = DEFAULT_JITTER,
    draws: int = DEFAULT_DRAWS,
    min_games: int = DEFAULT_MIN_GAMES,
    require_stable: bool = True,
) -> TradeConfidence:
    """Full confidence assessment for one proposal."""
    p10, p90 = measure_stability(proposal, my_roster, slots, jitter=jitter, draws=draws)
    return TradeConfidence(
        gain=proposal.my_gain,
        p10=p10,
        p90=p90,
        contingencies=find_contingencies(
            proposal,
            my_roster,
            slots,
            games_played=games_played,
            min_gain=min_gain,
            min_games=min_games,
        ),
        require_stable=require_stable,
    )


def assess_all(
    proposals: list[TradeProposal],
    my_roster: list[PlayerProjection],
    slots: list[str],
    games_played: dict[str, int] | None = None,
    **kwargs,
) -> dict[int, TradeConfidence]:
    """Confidence for a shortlist, keyed by position in that list."""
    return {
        i: assess(p, my_roster, slots, games_played=games_played, **kwargs)
        for i, p in enumerate(proposals)
    }
