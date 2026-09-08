"""Turning Kalshi prop ladders into distributions.

A Kalshi player prop is not a point estimate, it is a *ladder* of nested binary
markets -- "250+ passing yards", "275+", "300+" -- each priced independently.
Read together those prices are a survival function, P(X >= k), and a survival
function is a distribution. That is strictly more information than any
projection feed provides: not just what a player is expected to do, but how
uncertain the market is about it.

Three things have to be handled before the prices are usable.

**Vig and noise.** Each rung is quoted with a bid/ask spread, so the midpoint is
the natural probability estimate. Unlike a mutually-exclusive market there is no
sum-to-one normalisation to apply, because the rungs are nested rather than
disjoint: the implied bucket masses ``S(k_i) - S(k_i+1)`` already sum to one by
construction. Monotonicity is the only structural constraint, and it is imposed
with isotonic regression.

**Thin markets.** Some rungs are quoted so wide they carry no information -- a
receiving-yards rung seen at 18c bid against 60c ask, where the "probability"
could be anything from a coin flip to near-certain. Averaging that into a
projection is worse than ignoring it, so rungs wider than a threshold are
discarded and ladders with too few survivors are rejected outright.

**Tails.** The ladder only spans its strikes; the mass below the lowest rung and
above the highest is unobserved. Rather than invent a value for those buckets, a
distribution is fitted to the whole ladder. For a normal, ``S(k) = 1 -
Phi((k-mu)/sigma)`` rearranges to ``Phi^-1(1 - S(k)) = (k - mu)/sigma``, which is
*linear in k* -- so the fit is an ordinary least-squares line through the probit
of the observed survival values, and the tails follow from the fitted shape
instead of a guess.

Counts (touchdowns) are handled separately and exactly: for a non-negative
integer variable, ``E[N] = sum_k P(N >= k)``, which the ladder gives directly
when it starts at 0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# A rung quoted wider than this carries no usable signal. Set at 0.20 rather
# than tighter because real ladders quote their *middle* rungs -- the ones that
# bracket the median and carry the most information -- at 16-17c. Filtering at
# 0.15 deleted exactly those and left only the uninformative tails.
DEFAULT_MAX_SPREAD = 0.20
# Below this many usable rungs a fit is not trustworthy.
MIN_RUNGS = 3
# Probabilities are clipped away from the boundaries so the probit is finite.
_EPS = 1e-4


@dataclass(frozen=True)
class Rung:
    """One rung of a ladder: the market for "X >= strike"."""

    strike: float
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def is_usable(self, max_spread: float = DEFAULT_MAX_SPREAD) -> bool:
        if self.ask <= 0 or self.bid < 0 or self.ask > 1 or self.bid > self.ask:
            return False
        return self.spread <= max_spread


@dataclass(frozen=True)
class LadderFit:
    """A distribution recovered from a ladder."""

    mean: float
    sd: float
    rungs_used: int
    r_squared: float
    method: str
    brackets_median: bool = True

    @property
    def is_trustworthy(self) -> bool:
        """Whether this fit should be allowed to influence a projection.

        Goodness of fit alone is not enough, and trusting it is an easy mistake
        to make. A ladder whose usable rungs all sit in the upper tail -- every
        price below 0.5 -- can produce an excellent R-squared while placing the
        mean far outside the observed strike range, because a straight line
        through three tail points extrapolates freely below them. One real
        receiving-yards ladder fits at R-squared 0.993 and implies a mean of
        4.7 yards, which is not a projection, it is an artefact.

        So the fit must also *bracket the median*: at least one rung priced
        above 0.5 and one below, meaning the centre of the distribution is
        interpolated between observations rather than invented beyond them.
        """
        return (
            self.rungs_used >= MIN_RUNGS
            and self.sd > 0
            and self.r_squared >= 0.9
            and self.brackets_median
            and self.mean >= 0
        )


def pava_decreasing(values: list[float] | np.ndarray) -> np.ndarray:
    """Nearest non-increasing sequence, by pool-adjacent-violators.

    A survival function cannot increase, but independently quoted rungs
    routinely violate that by a cent or two. This projects the observed mids
    onto the closest monotone sequence in least squares, which removes the
    contradiction without discarding information.
    """
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n == 0:
        return y

    # Blocks of (sum, count); merge while the running means increase.
    sums = list(y)
    counts = [1.0] * n
    idx = 0
    while idx < len(sums) - 1:
        if sums[idx] / counts[idx] < sums[idx + 1] / counts[idx + 1]:
            sums[idx] += sums[idx + 1]
            counts[idx] += counts[idx + 1]
            del sums[idx + 1], counts[idx + 1]
            if idx > 0:
                idx -= 1
        else:
            idx += 1

    out = np.empty(n)
    pos = 0
    for total, count in zip(sums, counts):
        out[pos : pos + int(count)] = total / count
        pos += int(count)
    return out


def usable_rungs(rungs: list[Rung], max_spread: float = DEFAULT_MAX_SPREAD) -> list[Rung]:
    keep = [r for r in rungs if r.is_usable(max_spread)]
    return sorted(keep, key=lambda r: r.strike)


def fit_normal(rungs: list[Rung], max_spread: float = DEFAULT_MAX_SPREAD) -> LadderFit | None:
    """Recover (mean, sd) by least squares on the probit of the survival curve.

    Returns ``None`` when the ladder is too thin or too incoherent to fit at
    all. A fit that succeeds but should not be trusted is returned with
    ``is_trustworthy`` false rather than discarded, so callers can report *why*
    a player has no market opinion instead of silently omitting him.
    """
    keep = usable_rungs(rungs, max_spread)
    if len(keep) < MIN_RUNGS:
        return None

    strikes = np.array([r.strike for r in keep], dtype=float)
    survival = pava_decreasing([r.mid for r in keep])
    survival = np.clip(survival, _EPS, 1 - _EPS)

    # S(k) = 1 - Phi((k - mu)/sigma)  =>  Phi^-1(1 - S(k)) = (k - mu)/sigma
    z = norm.ppf(1.0 - survival)
    finite = np.isfinite(z)
    if finite.sum() < MIN_RUNGS:
        return None
    strikes, z = strikes[finite], z[finite]

    # Degenerate when every usable rung has the same price.
    if np.ptp(z) < 1e-9 or np.ptp(strikes) < 1e-9:
        return None

    slope, intercept = np.polyfit(strikes, z, 1)
    if slope <= 0:
        return None

    sigma = 1.0 / slope
    mu = -intercept * sigma

    predicted = slope * strikes + intercept
    ss_res = float(np.sum((z - predicted) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Is the median interpolated between observed rungs, or extrapolated past
    # the end of the ladder? Only the former supports a credible mean.
    brackets = bool(survival.max() >= 0.5 >= survival.min())

    return LadderFit(
        mean=float(mu),
        sd=float(sigma),
        rungs_used=len(strikes),
        r_squared=r_squared,
        method="normal-probit",
        brackets_median=brackets,
    )


def fit_lognormal(rungs: list[Rung], max_spread: float = DEFAULT_MAX_SPREAD) -> LadderFit | None:
    """Fit a lognormal, for quantities that are non-negative and right-skewed.

    Yardage is not symmetric. A receiver's floor is zero and his ceiling is
    open-ended, so the distribution has a long right tail and a mode well below
    its mean. Forcing a normal through that systematically understates
    low-volume players: one real tight-end ladder prices the median at exactly
    9.5 yards, yet a normal fitted to it returns a mean of 5.4, because the
    steep upper tail drags the line down. A lognormal reproduces the same ladder
    with a mean near 18, which is both self-consistent and close to the
    independent projection.

    The algebra is the same trick as the normal, in log space:
    ``S(k) = 1 - Phi((ln k - mu)/sigma)`` gives ``Phi^-1(1 - S(k))`` linear in
    ``ln k``.
    """
    keep = [r for r in usable_rungs(rungs, max_spread) if r.strike > 0]
    if len(keep) < MIN_RUNGS:
        return None

    log_strikes = np.log(np.array([r.strike for r in keep], dtype=float))
    survival = pava_decreasing([r.mid for r in keep])
    survival = np.clip(survival, _EPS, 1 - _EPS)

    z = norm.ppf(1.0 - survival)
    finite = np.isfinite(z)
    if finite.sum() < MIN_RUNGS:
        return None
    log_strikes, z = log_strikes[finite], z[finite]

    if np.ptp(z) < 1e-9 or np.ptp(log_strikes) < 1e-9:
        return None

    slope, intercept = np.polyfit(log_strikes, z, 1)
    if slope <= 0:
        return None

    sigma = 1.0 / slope
    mu = -intercept * sigma
    if sigma > 3:
        # Beyond this the lognormal mean is dominated by an unobserved tail.
        return None

    predicted = slope * log_strikes + intercept
    ss_res = float(np.sum((z - predicted) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mean = math.exp(mu + sigma**2 / 2)
    variance = (math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2)

    return LadderFit(
        mean=float(mean),
        sd=float(math.sqrt(variance)),
        rungs_used=len(log_strikes),
        r_squared=r_squared,
        method="lognormal-probit",
        brackets_median=bool(survival.max() >= 0.5 >= survival.min()),
    )


def fit_count(rungs: list[Rung], max_spread: float = DEFAULT_MAX_SPREAD) -> LadderFit | None:
    """Exact mean and variance for a non-negative integer variable.

    For counts, ``E[N] = sum_{k>=1} P(N >= k)`` and
    ``E[N^2] = sum_{k>=1} (2k - 1) P(N >= k)``. A touchdown ladder quotes
    exactly those terms (strikes at 0.5, 1.5, 2.5 meaning 1+, 2+, 3+), so no
    distributional assumption is needed at all.

    The identity needs *every* term from k=1 upward, so a ladder that does not
    reach down to "1 or more" cannot be evaluated this way and returns None.
    Reception ladders are quoted from 1.5 ("2+") and hit exactly that case:
    silently summing from k=2 drops the single largest term and understates a
    starting receiver by four or five catches. Touchdown ladders start at 0.5
    and are fine.

    The sum is truncated at the top rung. That understates both mean and
    variance slightly, but the omitted terms are the probability of a third or
    fourth touchdown, which is small enough not to move a lineup decision.
    """
    keep = usable_rungs(rungs, max_spread)
    if not keep:
        return None
    if math.ceil(keep[0].strike) != 1:
        # Ladder starts above "1 or more"; the survival sum is not computable.
        return None

    survival = pava_decreasing([r.mid for r in keep])
    mean = 0.0
    second = 0.0
    for rung, prob in zip(keep, survival):
        k = math.ceil(rung.strike)  # strike 0.5 is the market for "1 or more"
        if k < 1:
            continue
        mean += float(prob)
        second += (2 * k - 1) * float(prob)

    variance = max(0.0, second - mean**2)
    return LadderFit(
        mean=mean,
        sd=math.sqrt(variance),
        rungs_used=len(keep),
        r_squared=1.0,  # exact identity, nothing is being fitted
        method="count-survival",
    )
