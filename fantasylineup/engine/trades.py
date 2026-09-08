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
