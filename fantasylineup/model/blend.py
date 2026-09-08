"""Blending market-implied stats into the Sleeper baseline.

Sleeper is the anchor and Kalshi is an adjustment, capped so no single thin
market can dominate a projection. The reason is coverage, not distrust: the
market simply does not price everything a fantasy score is made of.

What is priced, and what is not
-------------------------------
Receivers are well covered -- receiving yards, receptions and touchdowns are all
quoted, which is every component of a wide receiver's score.

Quarterbacks are partly covered: passing yards are quoted, passing touchdowns
are not. ``KXNFLTD`` asks whether a player *scores* a touchdown, so for a
quarterback it describes his rushing, not his passing.

Running backs are the weakest case. There is no rushing-yards market at all, so
their largest scoring component is untouched and only receiving work and
touchdowns can be adjusted.

Each player therefore carries a coverage fraction -- how much of his Sleeper
projection sits in components the market actually prices -- and the cap is
scaled by it. A receiver with three liquid ladders can move the full allowance;
a running back with only a touchdown market moves a fraction of it. This keeps a
partial view from masquerading as a complete one.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping

from ..sources.kalshi import COUNT_STATS, MarketQuote
from .ladder import LadderFit, Rung, fit_count, fit_lognormal, fit_normal
from .scoring import score_stats

log = logging.getLogger(__name__)

# Stat keys a "scored touchdown" market informs. The market cannot say whether
# the touchdown was run in or caught, so the implied total is apportioned using
# the split Sleeper already projects.
TD_COMPONENTS = ("rush_td", "rec_td")


@dataclass
class BlendResult:
    """One player's blended projection and why it moved."""

    sleeper_id: str
    baseline: float
    market: float | None
    blended: float
    sd: float | None
    coverage: float
    fits: dict[str, LadderFit] = field(default_factory=dict)

    @property
    def shift(self) -> float:
        return self.blended - self.baseline

    @property
    def has_market(self) -> bool:
        return self.market is not None


def group_quotes(quotes: list[MarketQuote]) -> dict[tuple[str, str], list[Rung]]:
    """Group raw quotes into ladders keyed by (kalshi player id, stat)."""
    ladders: dict[tuple[str, str], list[Rung]] = defaultdict(list)
    for q in quotes:
        if q.kalshi_id is None:
            continue
        ladders[(q.kalshi_id, q.stat)].append(Rung(strike=q.strike, bid=q.bid, ask=q.ask))
    return dict(ladders)


def fit_ladders(ladders: Mapping[tuple[str, str], list[Rung]]) -> dict[tuple[str, str], LadderFit]:
    """Fit each ladder, keeping only fits worth acting on."""
    out: dict[tuple[str, str], LadderFit] = {}
    for key, rungs in ladders.items():
        _, stat = key
        fit = None
        if stat in COUNT_STATS:
            # Exact where the ladder reaches "1 or more"; returns None otherwise
            # (reception ladders start at "2+" and fall through).
            fit = fit_count(rungs)
        if fit is None:
            # Lognormal is the default: these quantities are non-negative and
            # right-skewed. Normal is a last resort for a ladder lognormal
            # cannot fit (a non-positive strike, or a degenerate sigma).
            fit = fit_lognormal(rungs) or fit_normal(rungs)
        if fit is not None and fit.is_trustworthy:
            out[key] = fit
    return out


def _market_stats(
    baseline_stats: Mapping[str, float], fits: Mapping[str, LadderFit]
) -> dict[str, float]:
    """Overlay market-implied values onto Sleeper's component stats."""
    merged = dict(baseline_stats)
    for stat, fit in fits.items():
        if stat == "scored_td":
            # Split the implied touchdown total in Sleeper's projected
            # rush/receive proportion; fall back to receiving when Sleeper
            # projects no touchdowns at all and offers no split to copy.
            projected = {c: float(baseline_stats.get(c) or 0.0) for c in TD_COMPONENTS}
            total = sum(projected.values())
            if total > 0:
                for component, value in projected.items():
                    merged[component] = fit.mean * (value / total)
            else:
                merged["rec_td"] = fit.mean
        else:
            merged[stat] = fit.mean
    return merged


def _coverage(
    baseline_stats: Mapping[str, float],
    fits: Mapping[str, LadderFit],
    scoring_settings: Mapping[str, float],
) -> float:
    """Share of the baseline projection sitting in market-priced components.

    A running back whose points are mostly rushing yards scores near zero here
    even with a liquid touchdown market, which is the honest answer.
    """
    baseline_points = score_stats(baseline_stats, scoring_settings)
    if baseline_points <= 0:
        return 0.0

    covered_keys: set[str] = set()
    for stat in fits:
        covered_keys.update(TD_COMPONENTS if stat == "scored_td" else {stat})

    covered = sum(
        float(scoring_settings.get(k, 0.0)) * float(baseline_stats.get(k) or 0.0)
        for k in covered_keys
    )
    return max(0.0, min(1.0, covered / baseline_points))


def blend_player(
    sleeper_id: str,
    baseline_stats: Mapping[str, float],
    fits: Mapping[str, LadderFit],
    scoring_settings: Mapping[str, float],
    max_shift: float = 0.25,
) -> BlendResult:
    """Combine one player's Sleeper baseline with whatever the market prices.

    The market view is allowed to move the projection by at most
    ``max_shift * coverage``, as a fraction of the baseline. With no usable
    ladder the baseline passes through untouched.
    """
    baseline = score_stats(baseline_stats, scoring_settings)
    if not fits:
        return BlendResult(sleeper_id, baseline, None, baseline, None, 0.0, {})

    coverage = _coverage(baseline_stats, fits, scoring_settings)
    market = score_stats(_market_stats(baseline_stats, fits), scoring_settings)

    allowance = abs(baseline) * max_shift * coverage
    delta = max(-allowance, min(allowance, market - baseline))

    return BlendResult(
        sleeper_id=sleeper_id,
        baseline=baseline,
        market=market,
        blended=baseline + delta,
        sd=_implied_sd(fits, scoring_settings),
        coverage=coverage,
        fits=dict(fits),
    )


def _implied_sd(
    fits: Mapping[str, LadderFit], scoring_settings: Mapping[str, float]
) -> float | None:
    """Fantasy-point standard deviation implied by the priced components.

    Components are combined assuming independence, which understates the true
    spread: a big receiving game correlates with a touchdown. It is a floor on
    uncertainty, not an estimate of it, and covers only the priced components --
    so it is used to inform the variance model rather than replace it.
    """
    if not fits:
        return None
    variance = 0.0
    for stat, fit in fits.items():
        keys = TD_COMPONENTS if stat == "scored_td" else (stat,)
        # Touchdown weights are equal across rush and receive in every sane
        # league, so taking the max is exact there and conservative elsewhere.
        weight = max((abs(float(scoring_settings.get(k, 0.0))) for k in keys), default=0.0)
        variance += (weight * fit.sd) ** 2
    return variance**0.5 if variance > 0 else None
