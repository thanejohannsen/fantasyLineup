"""Ladder-to-distribution tests.

Cases are real Kalshi quotes rather than invented ones, because the failure mode
that matters here is a market that looks fine and is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from fantasylineup.model.ladder import (
    Rung,
    fit_count,
    fit_lognormal,
    fit_normal,
    pava_decreasing,
    usable_rungs,
)

# Dak Prescott, passing yards, DAL at NYG. A liquid ladder: ten rungs, tight
# spreads, prices spanning 0.835 down to 0.03.
DAK = [
    Rung(174.5, 0.80, 0.87),
    Rung(199.5, 0.71, 0.83),
    Rung(224.5, 0.58, 0.72),
    Rung(249.5, 0.46, 0.59),
    Rung(274.5, 0.31, 0.44),
    Rung(299.5, 0.16, 0.32),
    Rung(324.5, 0.08, 0.21),
    Rung(349.5, 0.06, 0.11),
    Rung(374.5, 0.06, 0.10),
    Rung(399.5, 0.02, 0.04),
]

# Jahan Dotson, receiving yards. Illiquid: the low rungs are quoted 18c against
# 60c, so only upper-tail rungs survive filtering.
DOTSON = [
    Rung(14.5, 0.18, 0.60),
    Rung(24.5, 0.18, 0.41),
    Rung(39.5, 0.18, 0.22),
    Rung(49.5, 0.12, 0.13),
    Rung(59.5, 0.08, 0.09),
    Rung(69.5, 0.05, 0.06),
]


def test_liquid_ladder_fits_a_credible_distribution():
    fit = fit_normal(DAK)
    assert fit is not None
    assert fit.is_trustworthy
    # A starting quarterback, not a decimal-point error.
    assert 200 < fit.mean < 300
    assert 50 < fit.sd < 110
    assert fit.r_squared > 0.98


def test_illiquid_ladder_is_rejected_despite_excellent_fit():
    """The trap this gate exists for.

    Filtering leaves only upper-tail rungs. A straight line through them fits at
    R-squared 0.993 -- better than the liquid ladder -- while implying a mean of
    about 5 receiving yards, because nothing constrains the line below the
    lowest surviving strike. Goodness of fit alone would wave this through.
    """
    fit = fit_normal(DOTSON)
    assert fit is not None
    assert fit.r_squared > 0.98, "the artefact really does fit well"
    assert not fit.brackets_median
    assert not fit.is_trustworthy


def test_spread_filter_drops_uninformative_rungs():
    kept = usable_rungs(DOTSON)
    assert [r.strike for r in kept] == [39.5, 49.5, 59.5, 69.5]
    assert all(r.spread <= 0.20 for r in kept)
    # The 14.5 rung, quoted 18c against 60c, carries no information at all.
    assert 14.5 not in [r.strike for r in kept]


def test_too_few_rungs_returns_none():
    assert fit_normal([Rung(10.5, 0.5, 0.55), Rung(20.5, 0.3, 0.35)]) is None


def test_flat_ladder_returns_none():
    """Identical prices carry no shape and must not produce a fit."""
    flat = [Rung(s, 0.30, 0.34) for s in (10.5, 20.5, 30.5, 40.5)]
    assert fit_normal(flat) is None


def test_pava_enforces_monotone_decreasing():
    out = pava_decreasing([0.9, 0.5, 0.6, 0.3])
    assert np.all(np.diff(out) <= 1e-12)
    # The violating pair is averaged, not discarded.
    assert out[1] == pytest.approx(0.55)
    assert out[2] == pytest.approx(0.55)
    assert out[0] == pytest.approx(0.9)


def test_pava_leaves_monotone_input_untouched():
    values = [0.9, 0.7, 0.4, 0.1]
    assert np.allclose(pava_decreasing(values), values)


def test_pava_on_empty_input():
    assert len(pava_decreasing([])) == 0


def test_small_inversion_between_adjacent_rungs_still_fits():
    """Independently quoted rungs invert by a cent or two; that is not a fault.

    The inversion here is between neighbouring strikes whose true probabilities
    are close, which is what actually happens in live quotes. PAVA averages the
    offending pair and the fit is unharmed.
    """
    noisy = [
        Rung(100.5, 0.83, 0.87),
        Rung(125.5, 0.76, 0.80),
        Rung(150.5, 0.68, 0.72),
        Rung(175.5, 0.69, 0.73),  # one cent above its neighbour
        Rung(200.5, 0.50, 0.54),
        Rung(225.5, 0.38, 0.42),
    ]
    fit = fit_normal(noisy)
    assert fit is not None and fit.is_trustworthy
    # The median sits between the 175.5 and 200.5 rungs, so the mean should too.
    assert 175 < fit.mean < 225


def test_touchdown_count_is_exact():
    """E[N] = sum P(N >= k) needs no distributional assumption."""
    fit = fit_count([Rung(0.5, 0.42, 0.46), Rung(1.5, 0.10, 0.13), Rung(2.5, 0.01, 0.03)])
    assert fit is not None
    # 0.44 + 0.115 + 0.02
    assert fit.mean == pytest.approx(0.575, abs=1e-6)
    assert fit.method == "count-survival"
    assert 0 < fit.sd < 1.5


def test_count_variance_matches_the_identity():
    rungs = [Rung(0.5, 0.60, 0.60), Rung(1.5, 0.20, 0.20)]
    fit = fit_count(rungs)
    # E[N] = 0.8; E[N^2] = 1*0.6 + 3*0.2 = 1.2; Var = 1.2 - 0.64 = 0.56
    assert fit.mean == pytest.approx(0.8)
    assert fit.sd == pytest.approx(0.56**0.5)


def test_count_with_no_usable_rungs():
    assert fit_count([Rung(0.5, 0.05, 0.95)]) is None


# Noah Gray, receiving yards. The market prices his median at exactly 9.5, so
# any sane fit must return a mean in that neighbourhood.
NOAH_GRAY = [
    Rung(9.5, 0.49, 0.51),
    Rung(14.5, 0.18, 0.39),
    Rung(24.5, 0.18, 0.22),
    Rung(39.5, 0.08, 0.09),
    Rung(49.5, 0.08, 0.09),
]


def test_lognormal_beats_normal_on_skewed_yardage():
    """Yardage is right-skewed, and assuming otherwise understates volume.

    This ladder prices P(>= 9.5) at exactly 0.50, so the median is 9.5 and a
    right-skewed mean should sit above it. A normal returns 5.4 -- below the
    median it was fitted to, which is self-contradictory -- because the steep
    upper tail drags the line. The lognormal returns 17.9, agreeing closely with
    the independent Sleeper projection of 18.19.
    """
    normal = fit_normal(NOAH_GRAY)
    lognormal = fit_lognormal(NOAH_GRAY)
    assert normal is not None and lognormal is not None

    assert normal.mean < 9.5, "the normal fit falls below its own median"
    assert lognormal.mean > 9.5
    assert 15 < lognormal.mean < 21
    assert lognormal.r_squared > normal.r_squared


def test_count_fit_refuses_a_ladder_that_misses_one_or_more():
    """E[N] = sum P(N >= k) needs the k=1 term.

    Reception ladders are quoted from "2+", so summing what is there silently
    drops the largest term. Measured on a live slate that understated receivers
    by a factor of three -- a five-catch receiver implied as 0.6 catches -- with
    nothing to signal anything was wrong.
    """
    starts_at_two = [Rung(1.5, 0.70, 0.74), Rung(2.5, 0.50, 0.54), Rung(3.5, 0.30, 0.34)]
    assert fit_count(starts_at_two) is None

    starts_at_one = [Rung(0.5, 0.70, 0.74), Rung(1.5, 0.50, 0.54), Rung(2.5, 0.30, 0.34)]
    assert fit_count(starts_at_one) is not None


def test_spread_filter_keeps_informative_middle_rungs():
    """The threshold must not delete the rungs that bracket the median.

    Real ladders quote their middle rungs at 16-17c, which a 15c filter removes
    entirely, leaving only tails and forcing a rejection. Drake London's ladder
    is exactly that shape.
    """
    london = [
        Rung(39.5, 0.61, 0.78),
        Rung(49.5, 0.51, 0.67),
        Rung(59.5, 0.40, 0.56),
        Rung(69.5, 0.29, 0.45),
        Rung(79.5, 0.20, 0.35),
        Rung(89.5, 0.12, 0.27),
    ]
    fit = fit_lognormal(london)
    assert fit is not None and fit.is_trustworthy
    assert fit.brackets_median
    assert 40 < fit.mean < 90


def test_lognormal_rejects_non_positive_strikes():
    """log(0) is undefined; such a ladder must fall through, not crash."""
    assert fit_lognormal([Rung(0.0, 0.9, 0.92), Rung(-5.0, 0.8, 0.82)]) is None
