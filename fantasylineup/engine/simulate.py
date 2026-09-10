"""Monte Carlo over lineup outcomes, and the win probability that follows.

Maximising expected points is the wrong objective in a lopsided matchup. A heavy
favourite should protect a lead by lowering variance; a heavy underdog has to
raise it, because an average week loses. Both fall out of maximising ``P(win)``
rather than ``E[points]``, and the two objectives only agree when the matchup is
close.

The same machinery handles mid-week state. Once a Thursday game is final those
points are *known*, not projected, so they enter as a constant and only the
remaining players are simulated. That is what makes the engine chase after a
Thursday bust and protect after a Thursday smash without any special-case logic:
conditioning on banked points changes the optimal risk posture automatically.

Known limitation -- correlation
-------------------------------
Players are drawn independently. Real fantasy scores are correlated: a
quarterback and his own receiver rise together, and a defence falls as the
opposing offence rises. Independent draws therefore understate the spread of a
stacked lineup and overstate how reliably a favourite wins.

No correlation term is applied because there is no fitted value for one yet, and
an invented coefficient would be a confident-looking guess embedded in every
recommendation. The calibration step measures realised lineup spread against
predicted, which is what would justify a number here.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from .lineup import PlayerProjection

# Fantasy scores are non-negative and right-skewed, so offensive players are
# drawn from a gamma matched to the projected mean and standard deviation.
# Defences can score negative points and get a normal instead.
_SIGNED_POSITIONS = frozenset({"DEF"})


@dataclass(frozen=True)
class LiveState:
    """What a player has already scored, and how much of his game is left.

    Both halves are needed. Points already on the board are known and must not
    be re-drawn, but a player at half time still owns half a projection, so
    treating "has scored something" as "is finished" would understate every
    lineup mid-slate.
    """

    scored: float = 0.0
    remaining: float = 1.0


_NOT_STARTED = LiveState()


@dataclass(frozen=True)
class SimulatedOutcome:
    win_probability: float
    mean: float
    p10: float
    p90: float
    draws: int


def simulate_players(
    players: list[PlayerProjection],
    sds: dict[str, float],
    draws: int,
    rng: np.random.Generator,
    live: dict[str, LiveState] | None = None,
) -> np.ndarray:
    """Draw ``draws`` samples of each player's score. Shape (draws, n_players).

    A player partway through his game contributes what he has scored plus a
    draw over the fraction still to play. Scaling the mean by that fraction and
    the standard deviation by its square root is not an approximation: both the
    gamma and the normal are additive in exactly this way, so two halves of a
    game sum to the same distribution as the whole.
    """
    if not players:
        return np.zeros((draws, 0))

    live = live or {}
    out = np.empty((draws, len(players)))
    for i, player in enumerate(players):
        state = live.get(player.sleeper_id, _NOT_STARTED)
        if state.remaining <= 0.0:
            # Finished. These points are known, so re-drawing them would be
            # inventing uncertainty that no longer exists.
            out[:, i] = state.scored
            continue

        mean = float(player.points) * state.remaining
        sd = max(1e-6, float(sds.get(player.sleeper_id, 1.0)) * math.sqrt(state.remaining))

        if player.position in _SIGNED_POSITIONS or mean <= 0:
            out[:, i] = state.scored + rng.normal(mean, sd, draws)
        else:
            # Gamma matched on the first two moments: shape (mean/sd)^2,
            # scale sd^2/mean. Non-negative and right-skewed, like the real
            # distribution of a weekly fantasy score.
            shape = (mean / sd) ** 2
            scale = sd**2 / mean
            out[:, i] = state.scored + rng.gamma(shape, scale, draws)
    return out


def simulate_lineup_total(
    starters: list[PlayerProjection],
    sds: dict[str, float],
    draws: int,
    rng: np.random.Generator,
    live: dict[str, LiveState] | None = None,
) -> np.ndarray:
    """Total points for a lineup across draws.

    Points already scored arrive inside the per-player draws, not as a separate
    team total added on top. Adding a team's banked points to a full simulation
    of all ten starters counts everyone who has already played twice -- once as
    what he actually scored and again as what he was projected to score.
    """
    return simulate_players(starters, sds, draws, rng, live=live).sum(axis=1)


def win_probability(
    my_starters: list[PlayerProjection],
    opponent_starters: list[PlayerProjection],
    sds: dict[str, float],
    draws: int = 20000,
    live: dict[str, LiveState] | None = None,
    seed: int | None = 12345,
) -> SimulatedOutcome:
    """Probability this lineup outscores the opponent's.

    A fixed seed by default so that re-running the advisory without new data
    returns the same recommendation. Advice that flickers between refreshes on
    simulation noise alone is worse than useless.
    """
    rng = np.random.default_rng(seed)
    mine = simulate_lineup_total(my_starters, sds, draws, rng, live=live)
    theirs = simulate_lineup_total(opponent_starters, sds, draws, rng, live=live)

    # A tie counts as half a win; Sleeper leagues are rarely tied but the
    # convention keeps the probability symmetric.
    wins = float(np.mean((mine > theirs) + 0.5 * (mine == theirs)))
    return SimulatedOutcome(
        win_probability=wins,
        mean=float(mine.mean()),
        p10=float(np.percentile(mine, 10)),
        p90=float(np.percentile(mine, 90)),
        draws=draws,
    )
