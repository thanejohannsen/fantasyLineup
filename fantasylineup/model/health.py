"""Injury regimes, and what they do to a player's remaining value.

The discriminator is **availability, not severity**, and that comes from the
data rather than from a table of body parts.

Sleeper's *rest-of-season* projections do not price injury at all.
Sampled live, Michael Penix carried 143.2 rest-of-season points while out with a
reconstructed ACL, and Tyreek Hill 93.7 in the same state -- while George Kittle,
who is actually playing after Achilles surgery, carried 169.3, a figure that
already reflects his reduced role. Discounting all three the same way would be
as wrong as discounting none of them.

Its *weekly* projections price it only partly, and the exception is the one
that matters for a start/sit. A player on IR, PUP or NA has no weekly record and
so arrives at zero on his own. A player designated **Out** or **Doubtful** keeps
a full one: sampled live, Brock Bowers carried 16.0 for the week while Doubtful
after meniscus surgery, and Sam Darnold 17.1 while Out. Sleeper's *app* shows
those players at zero; the projections endpoint does not, so the haircut has to
be applied here. Assuming otherwise put a doubtful tight end into a recommended
lineup at full value.

For the rest-of-season regime the split is on whether a current-week projection
exists:

*Playing with a designation* means he is back from a prior injury, and the
rest-of-season number already reflects it. He is worth something, and he is
tradeable.

*Not playing* means the rest-of-season number is stale and aspirational. How
stale depends on whether he is week-to-week or done for the year.

Why ``injury_status`` alone must never be the rule
--------------------------------------------------
It is far too blunt. ``injury_start_date`` is 0 of 782 populated upstream, so
there is no timing signal in it at all, and "Questionable" skews heavily toward
stars -- median rest-of-season projection 95.3 against 38.7 for players with no
designation, because good players get reported. A blanket haircut on
"Questionable" would quietly penalise Puka Nacua and Ja'Marr Chase for
undisclosed knocks while doing nothing about the players who are genuinely gone.

Recovery evidence
-----------------
An Achilles rupture returns 66.2% of players at a mean of 10.9 months, and an
ACL reconstruction 61.8% at a mean of 13.6 months. Both exceed any remaining
schedule, so a structural tear sustained *during* a season ends it -- which is
why ``OUT_LONG`` is zero rather than merely discounted. Performance decline
after return is real but concentrated in the first season back, with players
regaining pre-injury levels given more than one season; a player currently
playing after such a surgery is by construction inside that window.

Sources: Achilles systematic review (PMC11682601) and cohort study (PubMed
30886876); ACL updated analysis (PubMed 33553449) and systematic review
(Ross et al. 2020); career outlook review (PubMed 33218267).

The multipliers below are *literature-informed judgment, not fitted values.*
``PLAYING_DIMINISHED`` is the weakest of them: it risks double-counting decline
that Sleeper may already have priced, which is why it is deliberately small. It
is the first parameter ``fl calibrate`` should measure once enough weeks of
stored projections and outcomes accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

# Designations that mean a player is unavailable for an extended period rather
# than week to week.
LONG_TERM_STATUSES = frozenset({"IR", "PUP", "DNR", "NA", "Sus"})

# Body parts whose recovery window exceeds a season. Matched as substrings
# because Sleeper writes compound values like "Knee - ACL + MCL".
STRUCTURAL_PARTS = ("ACL", "Achilles", "Patellar", "Lisfranc")

# A surgery note is itself evidence of a structural problem even when the body
# part is recorded vaguely.
STRUCTURAL_NOTES = frozenset({"Surgery"})


class Regime(str, Enum):
    HEALTHY = "healthy"
    PLAYING_DIMINISHED = "playing_diminished"
    OUT_SHORT = "out_short"
    OUT_LONG = "out_long"


# Rest-of-season multipliers. The weekly haircut is separate, below.
DEFAULT_MULTIPLIERS: dict[str, float] = {
    "healthy": 1.00,
    "playing_diminished_structural": 0.90,
    "playing_diminished_soft": 1.00,
    "out_short": 0.40,
    "out_long": 0.00,
}


# Probability a player carrying each designation actually takes the field. A
# weekly projection is conditional on playing, so this is what turns it into an
# expectation.
#
# "Out" and the long-term designations are definitional rather than judged: the
# player is not playing, and the value is zero. The other two are judgment.
# Doubtful is the NFL's "unlikely to play" and empirically very few do.
# Questionable is the weakest number here and the one to calibrate first --
# most questionable players suit up, so a heavy haircut would bench stars for
# knocks they play through.
#
# This is deliberately *not* the blanket haircut argued against above. That
# argument is about rest-of-season value, where a designation says little about
# a whole season and skews toward stars. For a single week the designation is
# precisely a statement about availability, which is the question being asked.
DEFAULT_AVAILABILITY: dict[str, float] = {
    "out": 0.00,
    "doubtful": 0.10,
    "questionable": 0.80,
    "healthy": 1.00,
}


@dataclass(frozen=True)
class HealthStatus:
    """One player's injury regime and what it does to his value."""

    regime: Regime
    status: str | None
    body_part: str | None
    notes: str | None
    structural: bool

    @property
    def has_designation(self) -> bool:
        return bool(self.status)

    @property
    def is_tradeable(self) -> bool:
        """Whether he should be offered in a trade at all.

        A player with no remaining value should not be packaged in either
        direction: not sold as though he had value, and not acquired as though
        he had value either.
        """
        return self.regime is not Regime.OUT_LONG

    @property
    def short_label(self) -> str:
        """Compact tag for a lineup row, e.g. ``Q - Achilles``."""
        if not self.status:
            return ""
        initial = {"Questionable": "Q", "Doubtful": "D", "Out": "OUT"}.get(
            self.status, self.status
        )
        part = self.body_part if self.body_part and self.body_part != "Undisclosed" else ""
        return f"{initial} - {part}" if part else initial

    def sentence(self, name: str) -> str:
        """A plain statement of the injury, for a trade message.

        Deliberately flat. Selling an injured player is legitimate; dressing the
        injury up as an opportunity is not, and the recipient can see the
        designation in his own app regardless.
        """
        if not self.status:
            return ""

        # "Undisclosed" is the most common body part upstream and says nothing;
        # repeating it makes the note look padded rather than informative.
        part = self.body_part if self.body_part and self.body_part != "Undisclosed" else None
        surgery = (self.notes or "").strip() == "Surgery"

        phrase = _readable_part(part) if part else None
        if surgery and phrase:
            detail = f"he's {self.status}, coming off {phrase} surgery"
        elif surgery:
            detail = f"he's {self.status} after surgery"
        elif phrase:
            article = "an" if phrase[0].lower() in "aeiou" else "a"
            detail = f"he's {self.status} with {article} {phrase} issue"
        else:
            detail = f"he's {self.status}"

        # The tail must match what the model actually did to his value. Saying a
        # projection is "already down" when no haircut was applied would be a
        # false claim about our own numbers.
        if self.regime is Regime.PLAYING_DIMINISHED and self.structural:
            tail = "He's playing, but his projection is down from where he was."
        elif self.regime is Regime.PLAYING_DIMINISHED:
            tail = "He's still projected to play."
        elif self.regime is Regime.OUT_SHORT:
            tail = "He isn't projected to play right now."
        else:
            tail = "He isn't expected back this season."
        return f"Heads up on {name}: {detail}. {tail}"


# Body parts named after people or places keep their capital in ordinary prose.
PROPER_NOUN_PARTS = frozenset({"achilles", "lisfranc", "jones"})


def _readable_part(part: str) -> str:
    """Lower-case a body part without mangling acronyms or proper nouns.

    Sleeper writes values like "Knee - ACL + MCL" and "Achilles". Blanket
    lower-casing yields "knee - acl + mcl" and "achilles", both of which read as
    typos in a message going to another person.
    """

    def fix(token: str) -> str:
        if len(token) >= 2 and token.isupper():
            return token  # ACL, MCL
        if token.lower() in PROPER_NOUN_PARTS:
            return token.capitalize()  # Achilles
        return token.lower()

    return " ".join(fix(token) for token in part.split())


def is_structural(body_part: str | None, notes: str | None) -> bool:
    """Whether the injury is one whose recovery outruns a season."""
    part = body_part or ""
    if any(token.lower() in part.lower() for token in STRUCTURAL_PARTS):
        return True
    return (notes or "").strip() in STRUCTURAL_NOTES


def classify(
    injury_status: str | None,
    injury_body_part: str | None,
    injury_notes: str | None,
    has_weekly_projection: bool,
) -> HealthStatus:
    """Place a player in a regime.

    ``has_weekly_projection`` is the load-bearing argument. It is the only
    signal available that distinguishes "back from last year's surgery and
    playing" from "had that surgery three weeks ago and is finished for the
    year", because ``injury_start_date`` is never populated upstream.
    """
    status = (injury_status or "").strip() or None
    structural = is_structural(injury_body_part, injury_notes)

    if status is None:
        return HealthStatus(Regime.HEALTHY, None, injury_body_part, injury_notes, structural)

    if has_weekly_projection:
        return HealthStatus(
            Regime.PLAYING_DIMINISHED, status, injury_body_part, injury_notes, structural
        )

    if status in LONG_TERM_STATUSES or structural:
        return HealthStatus(
            Regime.OUT_LONG, status, injury_body_part, injury_notes, structural
        )

    return HealthStatus(Regime.OUT_SHORT, status, injury_body_part, injury_notes, structural)


def ros_multiplier(
    health: HealthStatus, multipliers: Mapping[str, float] | None = None
) -> float:
    """How much of a rest-of-season projection survives this injury."""
    table = dict(DEFAULT_MULTIPLIERS)
    if multipliers:
        table.update(multipliers)

    if health.regime is Regime.HEALTHY:
        return table["healthy"]
    if health.regime is Regime.PLAYING_DIMINISHED:
        key = "playing_diminished_structural" if health.structural else "playing_diminished_soft"
        return table[key]
    if health.regime is Regime.OUT_SHORT:
        return table["out_short"]
    return table["out_long"]


def weekly_multiplier(
    health: HealthStatus, availability: Mapping[str, float] | None = None
) -> float:
    """How much of a *weekly* projection survives this designation.

    Applies to the current week only. Rest-of-season value uses
    ``ros_multiplier``, which asks a different question: whether a player is
    available at all over months, rather than on Sunday.
    """
    table = dict(DEFAULT_AVAILABILITY)
    if availability:
        table.update(availability)

    status = (health.status or "").strip()
    if not status:
        return table["healthy"]
    if status in LONG_TERM_STATUSES or status == "Out":
        return table["out"]
    if status == "Doubtful":
        return table["doubtful"]
    if status == "Questionable":
        return table["questionable"]
    # An unrecognised designation is still a designation; treating it as healthy
    # would make a new upstream code silently invisible.
    return table["questionable"]
