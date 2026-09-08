"""Win-probability tests.

The claim under test is that risk posture does not need to be programmed. Given
a win-probability objective, a favourite should reach for the safer player and
an underdog for the riskier one, from the same code path, purely because of who
they are playing.
"""

from __future__ import annotations

import numpy as np
import pytest

from fantasylineup.engine.lineup import PlayerProjection
from fantasylineup.engine.simulate import win_probability
from fantasylineup.engine.winprob import optimize_for_win_probability, posture
from fantasylineup.model.variance import default_model


def player(pid: str, pos: str, pts: float) -> PlayerProjection:
    return PlayerProjection(
        sleeper_id=pid, name=pid, position=pos, points=pts, fantasy_positions=frozenset({pos})
    )


# A safe 10-point receiver and a volatile 9-point one. Expected points prefer
# the safe player; only the matchup decides whether that is right.
SAFE = player("safe", "WR", 10.0)
VOLATILE = player("volatile", "WR", 9.0)
SDS = {"safe": 2.0, "volatile": 12.0}


def _opponent(total: float) -> list[PlayerProjection]:
    return [player("opp", "WR", total)]


def test_underdog_chases_the_ceiling():
    """Facing a score an average week cannot beat, take the variance.

    The safe player is projected higher but essentially cannot reach 40. The
    volatile one usually loses by more and occasionally wins, which is worth
    more than losing reliably.
    """
    result = optimize_for_win_probability(
        players=[SAFE, VOLATILE],
        slots=["WR"],
        opponent_starters=_opponent(40.0),
        sds={**SDS, "opp": 1.0},
        draws=40000,
    )
    assert result.lineup.assignments[0].sleeper_id == "volatile"
    assert result.improved
    assert result.gain > 0


def test_favourite_protects_the_floor():
    """Holding a winning hand, refuse the variance."""
    result = optimize_for_win_probability(
        players=[SAFE, VOLATILE],
        slots=["WR"],
        opponent_starters=_opponent(2.0),
        sds={**SDS, "opp": 1.0},
        draws=40000,
    )
    assert result.lineup.assignments[0].sleeper_id == "safe"
    assert not result.improved, "the expected-points lineup was already right"


def test_close_matchup_keeps_the_expected_points_lineup():
    """When the matchup is even, the two objectives agree and nothing moves."""
    result = optimize_for_win_probability(
        players=[SAFE, VOLATILE],
        slots=["WR"],
        opponent_starters=_opponent(10.0),
        sds={**SDS, "opp": 6.0},
        draws=40000,
    )
    assert result.lineup.assignments[0].sleeper_id == "safe"
    assert result.swaps_applied == 0


def test_banked_points_flip_the_posture():
    """The Thursday-night case, and the reason banked points are modelled.

    Same roster, same opponent, same remaining slot. The only difference is what
    already happened on Thursday. After a bust the correct play is the volatile
    receiver; after a smash it is the safe one. Nothing in the engine special
    cases this -- conditioning on known points is enough.
    """
    common = dict(
        players=[SAFE, VOLATILE],
        slots=["WR"],
        opponent_starters=_opponent(20.0),
        sds={**SDS, "opp": 3.0},
        draws=40000,
    )
    after_bust = optimize_for_win_probability(**common, my_banked=0.0, opponent_banked=25.0)
    after_smash = optimize_for_win_probability(**common, my_banked=40.0, opponent_banked=0.0)

    assert after_bust.lineup.assignments[0].sleeper_id == "volatile"
    assert after_smash.lineup.assignments[0].sleeper_id == "safe"


def test_locked_slot_is_never_swapped():
    result = optimize_for_win_probability(
        players=[SAFE, VOLATILE],
        slots=["WR"],
        opponent_starters=_opponent(40.0),
        sds={**SDS, "opp": 1.0},
        forced={0: "safe"},
        draws=20000,
    )
    assert result.lineup.assignments[0].sleeper_id == "safe"
    assert result.swaps_applied == 0


def test_win_probability_is_deterministic_across_runs():
    """Advice must not flicker between refreshes on simulation noise alone."""
    args = ([SAFE], [player("opp", "WR", 10.0)], {"safe": 5.0, "opp": 5.0})
    first = win_probability(*args, draws=5000)
    second = win_probability(*args, draws=5000)
    assert first.win_probability == second.win_probability


def test_win_probability_bounds_and_symmetry():
    sds = {"a": 4.0, "b": 4.0}
    mirror = win_probability(
        [player("a", "WR", 12.0)], [player("b", "WR", 12.0)], sds, draws=40000
    )
    assert 0.45 < mirror.win_probability < 0.55
    assert mirror.p10 < mirror.mean < mirror.p90


def test_gamma_draws_are_non_negative():
    """Offensive players cannot score negative points; defences can."""
    rng = np.random.default_rng(0)
    from fantasylineup.engine.simulate import simulate_players

    samples = simulate_players([player("wr", "WR", 8.0)], {"wr": 9.0}, 5000, rng)
    assert samples.min() >= 0.0

    dst = simulate_players([player("d", "DEF", 6.0)], {"d": 6.0}, 5000, rng)
    assert dst.min() < 0.0, "a defence can post a negative week"


def test_variance_model_matches_the_fitted_shape():
    """Quarterback spread is flat; skill-position spread scales."""
    model = default_model()
    qb_low = model.sd_for("QB", 14.0)
    qb_high = model.sd_for("QB", 21.0)
    assert qb_high - qb_low < 0.5, "fitted QB slope is ~0.02, essentially flat"

    rb_low = model.sd_for("RB", 3.0)
    rb_high = model.sd_for("RB", 17.0)
    assert rb_high > 2 * rb_low


def test_market_sd_never_shrinks_the_fitted_spread():
    """A partial market view is a floor on uncertainty, not a measurement."""
    model = default_model()
    fitted = model.sd_for("WR", 12.0)
    assert model.blend_market_sd("WR", 12.0, market_sd=1.0) == fitted
    assert model.blend_market_sd("WR", 12.0, market_sd=fitted + 3) == pytest.approx(fitted + 3)


def test_posture_labels():
    assert "favourite" in posture(0.80)
    assert "underdog" in posture(0.20)
    assert "close" in posture(0.50)


def test_calibration_refits_the_shipped_defaults():
    """The shipped coefficients must be what the fitter produces.

    Guards against the constants and the calibration path drifting apart --
    a divergence that would otherwise show up only as slowly wrong advice.
    """
    from fantasylineup.model.calibration import Observation, fit_variance
    from fantasylineup.model.variance import POSITION_VARIANCE

    rng = np.random.default_rng(7)
    observations = []
    for position, (intercept, slope, _floor) in POSITION_VARIANCE.items():
        for _ in range(1200):
            projected = float(rng.uniform(2, 22))
            sd = intercept + slope * projected
            observations.append(
                Observation(position, projected, projected + rng.normal(0, sd), "sleeper")
            )

    fitted = fit_variance(observations)
    for position, (intercept, slope, _floor) in POSITION_VARIANCE.items():
        got_intercept, got_slope = fitted[position]
        assert abs(got_intercept - intercept) < 1.5, position
        assert abs(got_slope - slope) < 0.12, position


def test_calibration_scores_a_perfect_source_at_zero():
    from fantasylineup.model.calibration import Observation, score_source

    perfect = [Observation("WR", 10.0, 10.0, "oracle") for _ in range(50)]
    score = score_source(perfect)
    assert score.mae == 0.0 and score.bias == 0.0 and score.n == 50


def test_calibration_detects_a_biased_source():
    from fantasylineup.model.calibration import Observation, score_source

    optimistic = [Observation("RB", 12.0, 9.0, "hype") for _ in range(80)]
    score = score_source(optimistic)
    assert score.bias == pytest.approx(-3.0)
    assert score.mae == pytest.approx(3.0)
