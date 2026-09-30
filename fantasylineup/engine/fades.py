"""Games where the tickets are lopsided enough to bet against.

The rule: when roughly eighty per cent or more of *bet tickets* sit on one side
of a spread or a total, take the other side. Tickets, not money -- a headcount,
where every five-dollar parlay leg weighs the same as a serious position.

Why tickets rather than handle
------------------------------
Because the gap between them is the signal. Eighty per cent of tickets next to
half the money means the twenty per cent are betting far larger, which is the
only thing separating a crowd from an informed minority. Handle alone cannot say
that, and neither can tickets alone -- so both are carried through to the board
even though only the ticket share decides what qualifies.

Why moneylines are excluded
---------------------------
They reach eighty per cent almost automatically: everyone takes the big
favourite for a small payout, so the threshold stops meaning anything. Over 2026
weeks 1-3 the rule produced 35 moneyline bets at a 34% win rate against 8 spread
bets, which is mostly a statement about which market the filter happens to fire
on rather than about the public being wrong.

What this is not
----------------
It is not a claim of edge. Three weeks of backtest put the spread arm at 6-2,
whose confidence interval runs from 45% to 105% -- consistent with anything at
all. This produces a board so a forward record can accumulate against a rule
fixed in advance, which is the only way the question gets answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ..sources.splits import GameSplits, Side

# Moneylines are deliberately absent. See the module docstring.
FADEABLE_MARKETS = ("spread", "total")

DEFAULT_THRESHOLD = 80


@dataclass(frozen=True)
class Fade:
    """One lopsided market, and the side the rule says to take."""

    game: str
    kickoff_utc: datetime | None
    market: str
    public: Side
    bet: Side
    num_bets: int
    # Carried so a spread can be written as "MIA +9.5" rather than "home +9.5".
    # The bet has to name the team you are backing: a reader who has to map
    # home onto a team from the fixture beside it is a reader who can get it
    # the wrong way round, which is the whole failure this board exists to
    # avoid on the over/under side.
    away: str = ""
    home: str = ""

    @property
    def divergence(self) -> int | None:
        """Money share minus ticket share on the popular side.

        Strongly negative means the money is not following the crowd, which is
        the case the rule is really reaching for. Zero or positive means the
        handle agrees, and fading it means fading everyone.
        """
        if self.public.money is None:
            return None
        return self.public.money - self.public.tickets

    @property
    def bet_name(self) -> str:
        """Who you are backing: a team for a spread, over/under for a total."""
        if self.market == "total":
            return self.bet.side
        team = {"away": self.away, "home": self.home}.get(self.bet.side, "")
        return team or self.bet.side

    @property
    def line_label(self) -> str:
        """The bet as it would be written on a ticket."""
        if self.bet.line is None:
            return self.bet_name
        if self.market == "total":
            return f"{self.bet_name} {self.bet.line:g}"
        return f"{self.bet_name} {self.bet.line:+g}"


def find_fades(
    games: list[GameSplits],
    threshold: int = DEFAULT_THRESHOLD,
    now: datetime | None = None,
    markets: tuple[str, ...] = FADEABLE_MARKETS,
) -> list[Fade]:
    """Every spread or total currently carrying a lopsided ticket count.

    Recomputed from scratch on every call rather than maintained as a list.
    Availability is derived the same way everywhere else in this project, and it
    means a game that stops qualifying simply stops appearing -- there is no
    stale entry to remove and nothing to go out of sync.

    Games already under way are dropped: the rule is about a price you can still
    take.
    """
    now = now or datetime.now(UTC)
    out: list[Fade] = []
    for game in games:
        if game.started:
            continue
        if game.kickoff_utc is not None and game.kickoff_utc <= now:
            continue
        for (market, _side), public in game.sides.items():
            if market not in markets or public.tickets < threshold:
                continue
            bet = game.opposite(public)
            if bet is None:
                # Half a market is not a market: without the other side there is
                # no price to take, so there is nothing to show.
                continue
            out.append(
                Fade(
                    game=game.label,
                    kickoff_utc=game.kickoff_utc,
                    market=market,
                    public=public,
                    bet=bet,
                    num_bets=game.num_bets,
                    away=game.away,
                    home=game.home,
                )
            )

    # Soonest first -- the board is read in the order the games need deciding.
    return sorted(
        out,
        key=lambda f: (
            f.kickoff_utc or datetime.max.replace(tzinfo=UTC),
            -f.public.tickets,
            f.game,
        ),
    )
