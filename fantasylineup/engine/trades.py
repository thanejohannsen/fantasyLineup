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

import textwrap
from dataclasses import dataclass
from itertools import combinations

from .lineup import PlayerProjection, lineup_value


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
        pool = [p for p in roster if p.sleeper_id not in untouchable and p.points > 0]
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
    **kwargs: object,
) -> list[TradeProposal]:
    """Search every opponent and return the strongest offers overall."""
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
    # send rather than five variations on the same trade.
    seen: set[int] = set()
    unique: list[TradeProposal] = []
    for proposal in found:
        if proposal.partner_roster_id in seen:
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


def _profile(roster: list[PlayerProjection], slots: list[str]) -> tuple[dict, set[str]]:
    """Positional counts and who actually starts."""
    from collections import Counter

    lineup = lineup_optimal(roster, slots)
    rostered = Counter(p.position for p in roster)
    starters = {p.sleeper_id for p in lineup.assignments.values()}
    return dict(rostered), starters


def lineup_optimal(roster: list[PlayerProjection], slots: list[str]):
    from .lineup import optimize_lineup

    return optimize_lineup(roster, slots)


def _names(players: tuple[PlayerProjection, ...]) -> str:
    if len(players) == 1:
        return players[0].name
    return " and ".join([", ".join(p.name for p in players[:-1]), players[-1].name])


def explain_trade(
    proposal: TradeProposal,
    my_roster: list[PlayerProjection],
    their_roster: list[PlayerProjection],
    slots: list[str],
) -> TradeRationale:
    """Build the rationale and a sendable message for one proposal."""
    my_before_counts, my_before_starters = _profile(my_roster, slots)
    their_before_counts, their_before_starters = _profile(their_roster, slots)

    my_after = _swap(my_roster, proposal.give, proposal.get)
    their_after = _swap(their_roster, proposal.get, proposal.give)
    _, my_after_starters = _profile(my_after, slots)
    _, their_after_starters = _profile(their_after, slots)

    # The crispest possible argument: I am sending players who do not start for
    # me and receiving one who does.
    sending_benched = [p for p in proposal.give if p.sleeper_id not in my_before_starters]
    receiving_starter = [p for p in proposal.get if p.sleeper_id in my_after_starters]
    they_start_incoming = [p for p in proposal.give if p.sleeper_id in their_after_starters]
    their_benched_out = [p for p in proposal.get if p.sleeper_id not in their_before_starters]

    # --- why this helps me -------------------------------------------------
    parts = []
    if sending_benched:
        surplus = sending_benched[0]
        depth = my_before_counts.get(surplus.position, 0)
        parts.append(
            f"{_names(tuple(sending_benched))} sits on your bench "
            f"({depth} {surplus.position}s rostered) and does not crack the lineup"
        )
    if receiving_starter:
        parts.append(f"{_names(tuple(receiving_starter))} starts for you immediately")
    if not parts:
        parts.append("consolidates depth into a better starter")
    why = "; ".join(parts) + f". Worth +{proposal.my_gain:.0f} points rest of season."

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
        their_parts.append(f"{_names(tuple(their_benched_out))} is not in their lineup either")
    if not their_parts:
        their_parts.append("they gain lineup value at a position they are thin at")
    their_angle = "; ".join(their_parts) + f". Worth +{proposal.their_gain:.0f} to them."

    # --- the message -------------------------------------------------------
    give_names = _names(proposal.give)
    get_names = _names(proposal.get)

    opening = f"Hey {proposal.partner_name} - interested in a trade?"

    if sending_benched and they_start_incoming:
        body = (
            f"I'm deep at {sending_benched[0].position} and "
            f"{_names(tuple(sending_benched))} is buried on my bench, so he's "
            f"doing nothing for me. Looking at your roster he'd walk straight "
            f"into your lineup."
        )
    elif len(proposal.give) > len(proposal.get):
        # A 2-for-1 is a genuine two-sided argument: I consolidate into a
        # better starter, they get depth and a free roster spot's worth of
        # value. Say both halves rather than only mine.
        body = (
            f"I'm carrying more depth than I can start, so I'd rather "
            f"consolidate two players into one I can actually use. You'd be "
            f"getting two contributors back for one."
        )
    elif receiving_starter:
        body = (
            f"{get_names} would slot straight into my lineup, and looking at "
            f"your roster I think {give_names} does the same for you."
        )
    else:
        body = "I think there's a deal here that helps us both."

    offer = f"I'd send {give_names} for {get_names}."

    if their_benched_out:
        closing = (
            f"From your side you'd be moving someone who isn't starting for you "
            f"anyway and filling a real hole. Happy to adjust if the shape isn't right."
        )
    else:
        closing = (
            "I think it genuinely helps us both - happy to adjust the pieces if "
            "you'd rather shape it differently."
        )

    # Wrapped to a messaging-app measure so it can be pasted without reflowing
    # into one unreadable line.
    pitch = "\n\n".join(
        "\n".join(textwrap.wrap(part, 72)) for part in (opening, body, offer, closing)
    )
    return TradeRationale(why=why, their_angle=their_angle, pitch=pitch)
