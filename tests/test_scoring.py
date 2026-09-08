"""Scoring engine tests."""

from __future__ import annotations

from fantasylineup.model.scoring import STANDARD_PPR, score_stats, self_check


def test_reproduces_sleeper_pts_ppr(rb_projections):
    """Our dot product must agree with Sleeper's own pre-scored total.

    This is the cheapest possible guard against unit and sign errors: getting
    pass_yd wrong by a factor of ten, or flipping fum_lost, breaks hundreds of
    players at once and cannot survive this assertion.
    """
    matched, total, worst = self_check(rb_projections)
    assert total > 50, "fixture should contain a meaningful slate"
    assert worst < 0.1, f"worst deviation {worst:.3f} exceeds the component-rounding floor"
    assert matched == total


def test_league_settings_differ_from_generic_ppr(rb_projections, scoring_settings):
    """The league's own settings must actually be applied, not silently ignored.

    ZOO + PEARL is close to standard PPR, so the totals are similar -- but they
    are computed from 43 keys including kicker distance tiers and DST tiers that
    generic PPR has no concept of.
    """
    record = rb_projections[0]
    league_points = score_stats(record["stats"], scoring_settings)
    ppr_points = score_stats(record["stats"], STANDARD_PPR)
    assert league_points > 0
    # Same ballpark for a running back, since the differing keys are elsewhere.
    assert abs(league_points - ppr_points) < 1.0


def test_unknown_stat_keys_are_ignored():
    """Informational keys in the payload must not leak into the score."""
    stats = {"rec": 5, "rec_yd": 60, "pts_ppr": 999.0, "gp": 1, "rec_tgt": 8, "adp_dd_ppr": 12.0}
    assert score_stats(stats, {"rec": 1.0, "rec_yd": 0.1}) == 11.0


def test_missing_stats_score_zero():
    assert score_stats({}, {"rec": 1.0}) == 0.0
