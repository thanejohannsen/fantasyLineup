"""Finding trades that both sides actually want.

The idea that makes this work is that a player is not worth the same to every
team. Value him by his *marginal* contribution to a specific roster's best legal
lineup and the same player is worth a lot to a team with a hole at his position
and almost nothing to a team already deep there. That asymmetry is what a trade
exploits, and it is invisible to any ranking based on projected points.

So every candidate is scored twice, once from each side:

    my_gain    = LV(mine - give + get)  - LV(mine)
    their_gain = LV(theirs - get + give) - LV(theirs)

and only deals where **both** numbers are positive are proposed. That is not
politeness, it is what makes an offer likely to be accepted: a trade the other
manager can see is bad for them will be declined, so a one-sided proposal is
wasted effort even when it is nominally "good" for us.

Valuation is on rest-of-season projections, not this week. A trade changes a
roster for the remainder of the season, and judging it on a single week would
chase matchups and ignore the schedule.

Two guards keep the output realistic. Deals involving either side's best player
are skipped, because managers do not trade their anchor for surplus regardless
of what the maths says. And a fairness band rejects proposals that take far more
raw projected value than they give, since the counterparty judges an offer on
rankings rather than on marginal lineup value and will read a lopsided one as an
insult.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from itertools import combinations

from ..model.health import HealthStatus, Regime, is_structural
from .lineup import PlayerProjection, lineup_value, optimize_lineup

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeProposal:
    """One offer, valued from both sides."""

    partner_roster_id: int
    partner_name: str
    give: tuple[PlayerProjection, ...]
    get: tuple[PlayerProjection, ...]
    my_gain: float
    their_gain: float

    @property
    def shape(self) -> str:
        return f"{len(self.give)}-for-{len(self.get)}"

    @property
    def raw_given(self) -> float:
        return sum(p.points for p in self.give)

    @property
    def raw_received(self) -> float:
        return sum(p.points for p in self.get)

    @property
    def joint_gain(self) -> float:
        return self.my_gain + self.their_gain


def _swap(
    roster: list[PlayerProjection],
    out_players: tuple[PlayerProjection, ...],
    in_players: tuple[PlayerProjection, ...],
) -> list[PlayerProjection]:
    outgoing = {p.sleeper_id for p in out_players}
    return [p for p in roster if p.sleeper_id not in outgoing] + list(in_players)


def _looks_fair(proposal_give: float, proposal_get: float, band: float) -> bool:
    """Whether the raw values are close enough for the offer to be taken seriously.

    Measured on raw projected points because that is what the counterparty sees.
    """
    larger = max(proposal_give, proposal_get)
    if larger <= 0:
        return False
    return abs(proposal_give - proposal_get) / larger <= band


def find_trades(
    my_roster: list[PlayerProjection],
    their_roster: list[PlayerProjection],
    slots: list[str],
    partner_roster_id: int,
    partner_name: str,
    min_my_gain: float = 3.0,
    min_their_gain: float = 1.0,
    fairness_band: float = 0.40,
    protect_top_n: int = 1,
    max_package: int = 2,
) -> list[TradeProposal]:
    """All two-sided-positive trades with one partner.

    Searches 1-for-1 and 2-for-1 in both directions. Thresholds are in
    rest-of-season points.
    """
    if not my_roster or not their_roster:
        return []

    my_base = lineup_value(my_roster, slots)
    their_base = lineup_value(their_roster, slots)

    mine_sorted = sorted(my_roster, key=lambda p: p.points, reverse=True)
    theirs_sorted = sorted(their_roster, key=lambda p: p.points, reverse=True)
    untouchable = {p.sleeper_id for p in mine_sorted[:protect_top_n]} | {
        p.sleeper_id for p in theirs_sorted[:protect_top_n]
    }

    def packages(roster: list[PlayerProjection]) -> list[tuple[PlayerProjection, ...]]:
        # Season-ending injuries are excluded explicitly, in both directions: a
        # player with no remaining value must not be sold as though he had any,
        # and must not be acquired as though he had any either. The rest-of-season
        # haircut already zeroes them, but relying on that would break silently
        # the moment the multiplier stopped being exactly zero.
        pool = [
            p
            for p in roster
            if p.sleeper_id not in untouchable
            and p.points > 0
            and p.health_regime != Regime.OUT_LONG.value
        ]
        out: list[tuple[PlayerProjection, ...]] = [(p,) for p in pool]
        if max_package >= 2:
            out.extend(combinations(pool, 2))
        return out

    proposals: list[TradeProposal] = []
    for give in packages(my_roster):
        for get in packages(their_roster):
            # Skip shapes that are not 1-for-1 or 2-for-1 in either direction.
            if len(give) == 2 and len(get) == 2:
                continue
            if not _looks_fair(
                sum(p.points for p in give), sum(p.points for p in get), fairness_band
            ):
                continue

            my_gain = lineup_value(_swap(my_roster, give, get), slots) - my_base
            if my_gain < min_my_gain:
                continue
            their_gain = lineup_value(_swap(their_roster, get, give), slots) - their_base
            if their_gain < min_their_gain:
                continue

            proposals.append(
                TradeProposal(
                    partner_roster_id=partner_roster_id,
                    partner_name=partner_name,
                    give=give,
                    get=get,
                    my_gain=my_gain,
                    their_gain=their_gain,
                )
            )

    proposals.sort(key=lambda t: (t.my_gain, t.their_gain), reverse=True)
    return proposals


def best_trades_across_league(
    my_roster: list[PlayerProjection],
    rosters: dict[int, list[PlayerProjection]],
    names: dict[int, str],
    slots: list[str],
    my_roster_id: int,
    limit: int = 5,
    games_played: dict[str, int] | None = None,
    assess_confidence: bool = True,
    jitter: float = 0.03,
    draws: int = 200,
    min_games: int = 3,
    require_stable: bool = True,
    **kwargs: object,
) -> list[TradeProposal]:
    """Search every opponent and return the strongest offers overall.

    Confidence is assessed only on the ranked shortlist, not on every candidate.
    The screening search evaluates tens of thousands of combinations and the
    assessment costs a few hundred lineup solves apiece; doing it up front would
    be minutes of work to discard almost all of it.
    """
    found: list[TradeProposal] = []
    for roster_id, roster in rosters.items():
        if roster_id == my_roster_id:
            continue
        found.extend(
            find_trades(
                my_roster,
                roster,
                slots,
                partner_roster_id=roster_id,
                partner_name=names.get(roster_id, f"roster {roster_id}"),
                **kwargs,  # type: ignore[arg-type]
            )
        )
    found.sort(key=lambda t: (t.my_gain, t.their_gain), reverse=True)

    # One proposal per partner, so the output is a set of distinct offers to
    # send rather than five variations on the same trade. Where the best offer
    # to a partner fails its confidence check, fall through to the next one
    # rather than dropping that partner entirely.
    from .confidence import assess

    seen: set[int] = set()
    unique: list[TradeProposal] = []
    for proposal in found:
        if proposal.partner_roster_id in seen:
            continue
        if assess_confidence:
            confidence = assess(
                proposal,
                my_roster,
                slots,
                games_played=games_played,
                min_gain=float(kwargs.get("min_my_gain", 3.0)),  # type: ignore[arg-type]
                jitter=jitter,
                draws=draws,
                min_games=min_games,
                require_stable=require_stable,
            )
            if not confidence.is_offerable:
                log.info(
                    "Dropping %s offer: %s",
                    proposal.partner_name,
                    confidence.describe(),
                )
                continue
        seen.add(proposal.partner_roster_id)
        unique.append(proposal)
        if len(unique) >= limit:
            break
    return unique


@dataclass(frozen=True)
class TradeRationale:
    """Why a trade works, and a message that says so honestly.

    Every claim is derived from roster state that both managers can already see
    in the app -- who is rostered, who starts, what the projections say. Nothing
    is invented or overstated, which matters for more than ethics: a pitch that
    misrepresents a roster the other manager can inspect in ten seconds is worse
    than no pitch, and it poisons the next offer too.

    The honest framing is also the persuasive one here, because the engine only
    ever proposes trades that improve both lineups. There is a real mutual
    argument to make, so the message just has to make it clearly.
    """

    why: str
    their_angle: str
    pitch: str
    disclosures: tuple[str, ...] = ()
    # Things true about the deal that the gain does not price. Kept apart from
    # `why` because they argue the other way, and folding them into the case for
    # a trade is how a caveat gets skimmed past.
    caveats: tuple[str, ...] = ()


def _position_counts(roster: list[PlayerProjection]) -> dict[str, int]:
    """How many of each position a roster holds.

    A roster count, not a lineup fact, so it is the same whichever lineup the
    manager chooses to run.
    """
    from collections import Counter

    return dict(Counter(p.position for p in roster))


def lineup_optimal(roster: list[PlayerProjection], slots: list[str]):
    return optimize_lineup(roster, slots)


def _health_of(player: PlayerProjection) -> HealthStatus:
    """Health status from the fields the pipeline attached to a projection."""
    regime = Regime(player.health_regime) if player.health_regime else Regime.HEALTHY
    return HealthStatus(
        regime=regime,
        status=player.injury_status,
        body_part=player.injury_body_part,
        notes=player.injury_notes,
        structural=is_structural(player.injury_body_part, player.injury_notes),
    )


def _agree(players, singular: str, plural: str) -> str:
    """Verb matching a name list: one player starts, two players start."""
    return singular if len(players) == 1 else plural


def _names(players: tuple[PlayerProjection, ...]) -> str:
    if len(players) == 1:
        return players[0].name
    return " and ".join([", ".join(p.name for p in players[:-1]), players[-1].name])


@dataclass(frozen=True)
class _LineupFacts:
    """Every membership fact a rationale needs, resolved in one place.

    This exists because deriving them ad hoc produced the same bug twice. Claims
    were being read from three different sources -- our optimal lineup for a
    manager, our optimal after the trade, and his real lineup -- with no rule
    saying which. Fixing the two facts under examination left a third reading the
    wrong one, and it shipped a statement that a player was out of a lineup he
    was starting in.

    The rule, applied uniformly:

    *Now* is the lineup actually set. *After* is a re-solve seeded from that same
    real lineup. Our idea of another manager's best lineup is never used for a
    claim about him -- the two agree until they don't, and where they diverge he
    is looking at his version.
    """

    my_now: frozenset[str]
    their_now: frozenset[str]
    my_after: frozenset[str]
    their_after: frozenset[str]


def _resolve_lineups(
    proposal: TradeProposal,
    my_roster: list[PlayerProjection],
    their_roster: list[PlayerProjection],
    slots: list[str],
    my_starters: list[PlayerProjection] | None,
    their_starters: list[PlayerProjection] | None,
) -> _LineupFacts:
    """Resolve who starts where, before and after, from real lineups where known."""

    def ids(players) -> frozenset[str]:
        return frozenset(p.sleeper_id for p in players)

    def after(
        current: list[PlayerProjection] | None,
        roster: list[PlayerProjection],
        leaving: tuple[PlayerProjection, ...],
        arriving: tuple[PlayerProjection, ...],
    ) -> frozenset[str]:
        # Seed from the real lineup when there is one, so the post-trade claim
        # is an edit of what he runs rather than of what we would run for him.
        # With no lineup set at all -- early season, or a manager who never sets
        # one -- fall back to the optimal over the whole roster.
        base = current if current else roster
        outgoing = {p.sleeper_id for p in leaving}
        pool = [p for p in base if p.sleeper_id not in outgoing] + list(arriving)
        return ids(optimize_lineup(pool, slots).assignments.values())

    my_now = ids(my_starters) if my_starters else ids(
        optimize_lineup(my_roster, slots).assignments.values()
    )
    their_now = ids(their_starters) if their_starters else ids(
        optimize_lineup(their_roster, slots).assignments.values()
    )
    return _LineupFacts(
        my_now=my_now,
        their_now=their_now,
        my_after=after(my_starters, my_roster, proposal.give, proposal.get),
        their_after=after(their_starters, their_roster, proposal.get, proposal.give),
    )


def explain_trade(
    proposal: TradeProposal,
    my_roster: list[PlayerProjection],
    their_roster: list[PlayerProjection],
    slots: list[str],
    my_starters: list[PlayerProjection] | None = None,
    their_starters: list[PlayerProjection] | None = None,
) -> TradeRationale:
    """Build the rationale and a sendable message for one proposal.

    ``my_starters`` and ``their_starters`` are the lineups as actually set in
    the app. When supplied, every claim about who starts and who gets benched is
    made against those rather than against a computed ideal, so the other
    manager can verify each one against what he is looking at.
    """
    my_before_counts = _position_counts(my_roster)
    their_before_counts = _position_counts(their_roster)

    # Every membership fact comes from here, so no claim can pick a different
    # source than its neighbours.
    facts = _resolve_lineups(
        proposal, my_roster, their_roster, slots, my_starters, their_starters
    )

    sending_benched = [p for p in proposal.give if p.sleeper_id not in facts.my_now]
    receiving_starter = [p for p in proposal.get if p.sleeper_id in facts.my_after]
    they_start_incoming = [p for p in proposal.give if p.sleeper_id in facts.their_after]
    # "Not in their lineup either" is a claim about *now*, so it is read from the
    # lineup he actually set. Reading it from our optimal for him is what
    # produced a note saying a player he starts was not in his lineup.
    their_benched_out = [p for p in proposal.get if p.sleeper_id not in facts.their_now]

    # Who the incoming players push out of his lineup. Naming him is the most
    # checkable form of the argument, which is exactly why it has to come from
    # the real lineup.
    displaced = [
        p
        for p in (their_starters or [])
        if p.sleeper_id not in facts.their_after
        and p.sleeper_id not in {g.sleeper_id for g in proposal.get}
    ]

    # Who the incoming players push out of *my* lineup -- the mirror of
    # `displaced`, and the first question a reader asks. "He starts for you
    # immediately" is only half an answer while the other half, who he replaces,
    # is missing. Read from the real lineup for the same reason every other
    # claim is: measured against our own optimal it names a player the manager
    # can see is already on his bench.
    sent_ids = {g.sleeper_id for g in proposal.give}
    my_displaced = [
        p
        for p in my_roster
        if p.sleeper_id in facts.my_now
        and p.sleeper_id not in facts.my_after
        and p.sleeper_id not in sent_ids
    ]

    # Who is blocking the player I am sending, so my own reason is specific.
    blocked_by = None
    if sending_benched:
        same_position = [
            player
            for player in my_roster
            if player.position == sending_benched[0].position
            and player.sleeper_id in facts.my_now
        ]
        blocked_by = max(same_position, key=lambda x: x.points) if same_position else None

    # --- why this helps me -------------------------------------------------
    parts = []
    if sending_benched:
        surplus = sending_benched[0]
        depth = my_before_counts.get(surplus.position, 0)
        parts.append(
            f"{_names(tuple(sending_benched))} "
            f"{_agree(sending_benched, 'sits', 'sit')} on your bench "
            f"({depth} {surplus.position}s rostered) and "
            f"{_agree(sending_benched, 'does', 'do')} not crack the lineup"
        )
    if receiving_starter:
        clause = (
            f"{_names(tuple(receiving_starter))} "
            f"{_agree(receiving_starter, 'starts', 'start')} for you immediately"
        )
        if my_displaced:
            clause += (
                f", pushing {_names(tuple(my_displaced))} out of your lineup"
            )
        parts.append(clause)
    if not parts:
        parts.append("consolidates depth into a better starter")
    why = "; ".join(parts) + f". Worth +{proposal.my_gain:.0f} points rest of season."

    # Two starters sharing a bye is a week that cannot be covered, and marginal
    # lineup value cannot see it: it prices a season total, not the shape of the
    # schedule behind it. Reported rather than priced, because what a collision
    # costs depends on bench depth at that position in that week -- a different
    # calculation from the one this engine does, and one worth doing properly
    # rather than approximating inside a trade search.
    incoming = {g.sleeper_id for g in proposal.get}
    after_starters = [
        p
        for p in [*my_roster, *proposal.get]
        if p.sleeper_id in facts.my_after
    ]
    by_bye: dict[int, list[PlayerProjection]] = {}
    for player in after_starters:
        if player.bye_week:
            by_bye.setdefault(player.bye_week, []).append(player)

    caveats = tuple(
        f"{_names(tuple(sorted(group, key=lambda x: x.name)))} all share a week "
        f"{week} bye" if len(group) > 2 else
        f"{_names(tuple(sorted(group, key=lambda x: x.name)))} share a week {week} bye"
        for week, group in sorted(by_bye.items())
        # Only a clash this trade would create is worth raising; one the roster
        # already has is not news, and listing it would bury the one that is.
        if len(group) > 1 and any(p.sleeper_id in incoming for p in group)
    )

    # --- why they should say yes -------------------------------------------
    # Depth is reported per player: a package can span two positions, and
    # describing both with one position count would be simply false.
    their_parts = []
    for player in they_start_incoming:
        depth = their_before_counts.get(player.position, 0)
        their_parts.append(
            f"{player.name} starts for them (they roster {depth} at {player.position})"
        )
    if their_benched_out:
        their_parts.append(
            f"{_names(tuple(their_benched_out))} "
            f"{_agree(their_benched_out, 'is', 'are')} not in their lineup either"
        )
    if not their_parts:
        their_parts.append("they gain lineup value at a position they are thin at")
    their_angle = "; ".join(their_parts) + f". Worth +{proposal.their_gain:.0f} to them."

    # --- the message -------------------------------------------------------
    # Terse and specific, in the register a league-mate actually writes in. No
    # greeting, no sign-off, no offer to "adjust the pieces": pleasantries read
    # as filler and make a sound proposal look automated. What earns a reply is
    # the concrete claim -- who the player displaces in *their* lineup, and what
    # it is worth over the rest of the season.
    give_names = _names(proposal.give)
    get_names = _names(proposal.get)

    lines = [f"{give_names} for {get_names}?"]

    # Naming who he beats out is far stronger than naming a position count:
    # it is checkable, and it is the argument they would make themselves.
    if they_start_incoming and displaced:
        # Phrased so the counts cannot be misread: two incoming players do not
        # each displace the same man, they push a set of players to the bench.
        verb = "would both start" if len(they_start_incoming) > 1 else "would start"
        moves = "move" if len(displaced) > 1 else "moves"
        lines.append(
            f"{_names(tuple(they_start_incoming))} {verb} for you - "
            f"{_names(tuple(displaced))} {moves} to your bench. Worth about "
            f"+{proposal.their_gain:.0f} points to your lineup rest of season "
            f"by my numbers."
        )
    elif they_start_incoming:
        lines.append(
            f"{_names(tuple(they_start_incoming))} slots straight into your "
            f"lineup - about +{proposal.their_gain:.0f} points rest of season "
            f"by my numbers."
        )
    else:
        lines.append(
            f"By my numbers it's worth about +{proposal.their_gain:.0f} points "
            f"to your lineup rest of season."
        )

    # A short, true reason for wanting it. Saying plainly what I get out of the
    # deal reads as straightforward rather than as an angle being worked.
    if sending_benched and blocked_by:
        lines.append(
            f"On my end {_names(tuple(sending_benched))} is behind "
            f"{blocked_by.name}, so he doesn't start for me."
        )
    elif sending_benched:
        lines.append(
            f"On my end {_names(tuple(sending_benched))} isn't cracking my lineup."
        )
    elif len(proposal.give) > len(proposal.get):
        lines.append(
            "I'm carrying more depth than I can start and would rather run one "
            "player I can actually use."
        )
    elif receiving_starter:
        lines.append(f"{get_names} fills a hole for me.")

    # Any injury on either side is stated before the closing line. Selling an
    # injured player is legitimate; making an affirmative case for him while
    # omitting a known surgery is not, and the recipient can see the designation
    # in his own app anyway -- so omitting it buys nothing and costs the
    # credibility every future offer depends on.
    disclosures = tuple(
        _health_of(player).sentence(player.name)
        for player in (*proposal.give, *proposal.get)
        if _health_of(player).has_designation
    )
    if disclosures:
        lines[-1:-1] = list(disclosures)

    # Wrapped so it pastes into a messaging app without reflowing into one line.
    pitch = "\n\n".join("\n".join(textwrap.wrap(line, 72)) for line in lines)
    return TradeRationale(
        why=why,
        their_angle=their_angle,
        pitch=pitch,
        disclosures=disclosures,
        caveats=caveats,
    )


@dataclass(frozen=True)
class RankedProposal:
    """A proposal with its priority, and what it rules out.

    Proposals are not independent. Several of the best offers usually route
    through the same surplus player, and the moment he is traded the rest are
    dead. Presenting them as a flat list invites sending all of them and then
    honouring whichever is accepted first, which is how a manager ends up taking
    the worst of three offers he could have had.
    """

    rank: int
    proposal: TradeProposal
    conflicts_with: tuple[int, ...]
    shared_players: tuple[str, ...]

    @property
    def is_priority(self) -> bool:
        """Nothing better already claims a player this deal needs."""
        return not self.conflicts_with


def rank_proposals(proposals: list[TradeProposal]) -> list[RankedProposal]:
    """Order by value to us, flagging deals that compete for the same player.

    Ranking is by our own gain, which is the whole point of the exercise. The
    conflict flag then answers the question a flat list cannot: of these offers,
    which one should actually be sent first.
    """
    ordered = sorted(proposals, key=lambda t: (t.my_gain, t.their_gain), reverse=True)

    ranked: list[RankedProposal] = []
    for index, proposal in enumerate(ordered):
        involved = {p.sleeper_id for p in (*proposal.give, *proposal.get)}
        conflicts: list[int] = []
        shared: list[str] = []
        for better_index, better in enumerate(ordered[:index]):
            overlap = involved & {p.sleeper_id for p in (*better.give, *better.get)}
            if overlap:
                conflicts.append(better_index + 1)
                names = {
                    p.name
                    for p in (*proposal.give, *proposal.get)
                    if p.sleeper_id in overlap
                }
                shared.extend(sorted(names))
        ranked.append(
            RankedProposal(
                rank=index + 1,
                proposal=proposal,
                conflicts_with=tuple(conflicts),
                shared_players=tuple(dict.fromkeys(shared)),
            )
        )
    return ranked
