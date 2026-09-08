"""Fantasy scoring.

Sleeper's projections and stats endpoints return *component* stats
(``rush_yd``, ``rec``, ``rec_td``, ...) using the same key namespace as a
league's ``scoring_settings``. Scoring is therefore a dot product over shared
keys -- which is the whole reason we use those endpoints rather than the
pre-scored ``pts_ppr`` field: this league has 43 scoring keys, and only its own
settings produce the number that will actually appear on the scoreboard.

Keys present in the stat payload but absent from the settings (``pts_ppr``,
``gp``, ``rec_tgt``, ``adp_dd_ppr``, ...) are informational and drop out
naturally.

Known limitation -- defensive points-allowed tiers
--------------------------------------------------
Defensive scoring is tiered: 0 points allowed is worth 10, 1-6 is worth 7,
14-20 is worth 1, and so on. Sleeper's projection expresses this by marking a
single modal bucket with 1.0 (``pts_allow_14_20: 1.0`` alongside
``pts_allow: 16.0``) and omitting every other bucket.

Scoring that literally, as this module does, treats the most likely bucket as
certain. It is right that the term is included -- this league scores that tier
and Sleeper's generic ``pts_ppr`` simply ignores it, which is why our DST
numbers run about a point higher than theirs -- but collapsing the distribution
to its mode understates defensive variance in both directions: a shutout is
worth 10 and a blowout loss is worth -4, and neither shows up.

Doing better needs a distribution over points allowed rather than a point
estimate. Kalshi's game total and spread markets imply exactly that, so this is
revisited once those are wired in; until then DST means are slightly optimistic
and DST variance is badly understated.
"""

from __future__ import annotations

from typing import Any, Mapping

# Sleeper's own PPR interpretation, used only to self-check the engine against
# the ``pts_ppr`` value they ship alongside the components.
STANDARD_PPR: dict[str, float] = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_2pt": 2.0,
    "pass_int": -1.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rush_2pt": 2.0,
    "rec": 1.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "rec_2pt": 2.0,
    "fum_lost": -2.0,
}


def score_stats(stats: Mapping[str, Any], scoring_settings: Mapping[str, float]) -> float:
    """Score one player's component stats under a league's settings."""
    total = 0.0
    for key, weight in scoring_settings.items():
        value = stats.get(key)
        if value:
            total += float(weight) * float(value)
    return total


def score_many(
    records: list[Mapping[str, Any]], scoring_settings: Mapping[str, float]
) -> dict[str, float]:
    """Score a list of projection/stat records, keyed by Sleeper player id."""
    out: dict[str, float] = {}
    for rec in records:
        pid = rec.get("player_id")
        stats = rec.get("stats") or {}
        if pid:
            out[str(pid)] = score_stats(stats, scoring_settings)
    return out


def self_check(records: list[Mapping[str, Any]], tolerance: float = 0.1) -> tuple[int, int, float]:
    """Reproduce Sleeper's ``pts_ppr`` from components under STANDARD_PPR.

    A cheap but effective guard: it catches sign flips, unit errors (0.04 vs
    0.4) and typo'd stat keys immediately, because any of those blow the
    tolerance across hundreds of players at once.

    The tolerance is 0.1 rather than something tighter because Sleeper rounds
    both the components and ``pts_ppr`` before serialising, so summing rounded
    parts drifts slightly from their rounded total. Measured worst case across
    a full position slate is ~0.07, i.e. the rounding floor -- anything much
    above that is a real bug, not noise.

    Returns ``(matched, total, worst_absolute_difference)``.
    """
    matched = 0
    total = 0
    worst = 0.0
    for rec in records:
        stats = rec.get("stats") or {}
        expected = stats.get("pts_ppr")
        if expected is None:
            continue
        total += 1
        diff = abs(score_stats(stats, STANDARD_PPR) - float(expected))
        worst = max(worst, diff)
        if diff <= tolerance:
            matched += 1
    return matched, total, worst
