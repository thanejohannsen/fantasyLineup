"""Waiver targets, ranked by what they would actually add.

The ranking is *value over my replacement*, not value in the abstract. A free
agent is worth exactly what he adds to the best legal lineup this roster can
field -- ``LV(roster + player) - LV(roster)`` -- which is zero for a fifth
running back on a roster already starting four, however good he looks in
isolation. Ranking by projected points instead would recommend exactly those
players.

Because a pickup requires a drop on a full roster, each target is paired with
the cheapest legal drop and scored on the *net* change. That is the number that
matters: adding a 12-point receiver while cutting an 11-point one is not a
12-point gain.

This league runs rolling waiver priority rather than FAAB, so there is no bid to
recommend -- only whether a player is worth spending priority on.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .lineup import PlayerProjection, lineup_value


@dataclass(frozen=True)
class WaiverTarget:
    """A pickup worth considering, and what it would cost to make room."""

    player: PlayerProjection
    drop: PlayerProjection | None
    gain: float
    rostered_gain: float

    @property
    def is_net_positive(self) -> bool:
        return self.gain > 0


def _cheapest_drop(
    roster: list[PlayerProjection],
    slots: list[str],
    protected: set[str],
) -> tuple[PlayerProjection | None, float]:
    """The rostered player whose removal costs the least lineup value."""
    base = lineup_value(roster, slots)
    best: tuple[PlayerProjection | None, float] = (None, float("inf"))
    for candidate in roster:
        if candidate.sleeper_id in protected:
            continue
        without = [p for p in roster if p.sleeper_id != candidate.sleeper_id]
        cost = base - lineup_value(without, slots)
        if cost < best[1]:
            best = (candidate, cost)
    return best


def rank_waiver_targets(
    roster: list[PlayerProjection],
    candidates: list[PlayerProjection],
    slots: list[str],
    roster_limit: int | None = None,
    protected: set[str] | None = None,
    limit: int = 10,
) -> list[WaiverTarget]:
    """Rank free agents by the net lineup value of adding them.

    ``roster_limit`` is the number of active roster spots. When the roster is
    full a drop is forced, and the pairing is chosen to minimise what that drop
    costs.
    """
    protected = protected or set()
    base = lineup_value(roster, slots)
    must_drop = roster_limit is not None and len(roster) >= roster_limit

    results: list[WaiverTarget] = []
    for candidate in candidates:
        added = [*roster, candidate]
        rostered_gain = lineup_value(added, slots) - base

        if not must_drop:
            results.append(WaiverTarget(candidate, None, rostered_gain, rostered_gain))
            continue

        # Choose the drop *after* adding, so a player the newcomer makes
        # redundant becomes the cheapest cut.
        drop, cost = _cheapest_drop(added, slots, protected | {candidate.sleeper_id})
        if drop is None:
            continue
        final = [p for p in added if p.sleeper_id != drop.sleeper_id]
        results.append(
            WaiverTarget(
                player=candidate,
                drop=drop,
                gain=lineup_value(final, slots) - base,
                rostered_gain=rostered_gain,
            )
        )

    results.sort(key=lambda t: t.gain, reverse=True)
    return [t for t in results if t.is_net_positive][:limit]


# How many free agents to consider per position. Roughly proportional to how
# many of each a roster can start, so the pool spans what is actually startable.
PER_POSITION_CANDIDATES = {"RB": 35, "WR": 40, "TE": 20, "QB": 8, "K": 8, "DEF": 12}
DEFAULT_PER_POSITION = 10


def shortlist_candidates(
    conn: sqlite3.Connection,
    free_agent_ids: set[str],
    projections: dict[str, dict],
) -> set[str]:
    """Narrow the free-agent pool before the expensive per-player evaluation.

    Nearly 700 players are unrostered in a 12-team league and almost all project
    near zero, so some shortlist is needed. It has to be taken **per position**,
    though, not from a single ranking by projected points.

    Ranking globally fails badly in a one-quarterback league. Season-long
    quarterback totals dominate every other position -- an unrostered backup
    quarterback outprojects every available running back and receiver -- so a
    global top-N shortlist fills up entirely with quarterbacks who cannot
    improve a lineup that already starts one. Measured on this league, that
    produced a shortlist where every single candidate added exactly zero, and
    genuinely useful running backs were never evaluated at all.
    """
    if not free_agent_ids:
        return set()

    placeholders = ",".join("?" * len(free_agent_ids))
    positions = {
        r["sleeper_id"]: r["position"]
        for r in conn.execute(
            f"SELECT sleeper_id, position FROM players WHERE sleeper_id IN ({placeholders})",
            list(free_agent_ids),
        )
    }

    by_position: dict[str, list[tuple[str, float]]] = {}
    for pid in free_agent_ids:
        points = float((projections.get(pid) or {}).get("mean") or 0.0)
        if points <= 0:
            continue
        by_position.setdefault(positions.get(pid) or "UNK", []).append((pid, points))

    shortlist: set[str] = set()
    for position, players in by_position.items():
        players.sort(key=lambda kv: kv[1], reverse=True)
        keep = PER_POSITION_CANDIDATES.get(position, DEFAULT_PER_POSITION)
        shortlist.update(pid for pid, _ in players[:keep])
    return shortlist


def explain_no_targets(
    roster: list[PlayerProjection],
    candidates: list[PlayerProjection],
    slots: list[str],
    top_n: int = 3,
) -> tuple[PlayerProjection | None, list[PlayerProjection]]:
    """The bar a pickup had to clear, and who came closest.

    "Nothing available improves the lineup" is a correct answer but a useless
    one: it gives no way to tell a healthy roster apart from a broken query.
    Naming the weakest current starter and the best free agents who failed to
    beat him makes the verdict checkable at a glance.

    Quarterbacks, kickers and defences are excluded from the near-miss list. In
    a one-quarterback league a backup quarterback's season total dominates every
    other position while being incapable of improving anything.

    The bar is the weakest *displaceable* starter, not the weakest starter
    outright. Those differ, and using the wrong one is actively misleading: this
    roster's lowest-scoring starter is a kicker on 76 season points, but no
    tight end can take a kicker's slot, so the number a tight end actually has
    to beat is the flex starter on 169.
    """
    from .lineup import optimize_lineup

    displaceable = {"RB", "WR", "TE"}
    lineup = optimize_lineup(roster, slots)
    starters = [p for p in lineup.assignments.values() if p.position in displaceable]
    weakest = min(starters, key=lambda p: p.points) if starters else None
    near = sorted(
        (c for c in candidates if c.position in displaceable),
        key=lambda p: p.points,
        reverse=True,
    )[:top_n]
    return weakest, near
