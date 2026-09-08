"""How wrong a projection is likely to be.

A projection without a spread cannot answer the question that actually decides a
lineup: not "who scores more on average" but "who is more likely to win me this
matchup". Those differ whenever the matchup is lopsided.

The coefficients below were fitted on 6,809 projection-and-outcome pairs from
the 2025 season -- every skill player Sleeper projected, joined to what he
actually scored, priced under this league's own scoring settings. They are
defaults, refitted from live results by the calibration step.

Two things the data says that a guess would have got wrong:

*Quarterback spread is flat.* A quarterback projected for 21 points is no more
variable in absolute terms than one projected for 14: the fitted slope is 0.02,
essentially zero, against a constant of about 7.2. Assuming spread scales with
projection -- as it plainly does for every other position -- would badly
misprice the position with the highest projections on the roster.

*Everyone else fans out.* Running backs, receivers and tight ends carry slopes
of 0.33 to 0.41, so a 17-point running back is roughly two and a half times as
uncertain as a 3-point one.

Sleeper's in-season projections were close to unbiased: mean residuals sat
within a point of zero in every position and projection band, so no bias
correction is applied here.
"""

from __future__ import annotations

from dataclasses import dataclass

# sd(points) ~= intercept + slope * projected_points, fitted per position on
# 2025. Floors keep a near-zero projection from implying near-zero uncertainty:
# a benched player can still break a long touchdown.
POSITION_VARIANCE: dict[str, tuple[float, float, float]] = {
    # position: (intercept, slope, floor)
    "QB": (7.21, 0.020, 3.0),
    "RB": (2.85, 0.337, 1.5),
    "WR": (3.20, 0.329, 1.5),
    "TE": (2.51, 0.412, 1.2),
    "K": (3.60, 0.150, 1.5),
    "DEF": (4.40, 0.204, 2.0),
}

# Used when a player's position is unknown or unfitted; deliberately wide.
DEFAULT_VARIANCE = (3.20, 0.330, 1.5)


@dataclass(frozen=True)
class VarianceModel:
    """Maps a projection to a standard deviation."""

    coefficients: dict[str, tuple[float, float, float]]

    def sd_for(self, position: str | None, projection: float) -> float:
        intercept, slope, floor = self.coefficients.get(position or "", DEFAULT_VARIANCE)
        return max(floor, intercept + slope * max(0.0, projection))

    def blend_market_sd(
        self, position: str | None, projection: float, market_sd: float | None
    ) -> float:
        """Combine the fitted spread with a market-implied one.

        The market's implied spread covers only the components it prices and
        assumes those are independent, so it is a floor on uncertainty rather
        than a measurement of it. Taking the larger of the two keeps a
        confident-looking partial market from shrinking a projection's genuine
        uncertainty.
        """
        fitted = self.sd_for(position, projection)
        if market_sd is None or market_sd <= 0:
            return fitted
        return max(fitted, market_sd)


def default_model() -> VarianceModel:
    return VarianceModel(coefficients=dict(POSITION_VARIANCE))
