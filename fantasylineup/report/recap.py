"""The Tuesday recap: what happened, and which calls were actually wrong.

The useful part of a recap is not the score, which Sleeper already shows. It is
separating three very different kinds of miss, because only one of them is a
fault in the bot:

*Projection error* -- we said fourteen and he scored four. The model was wrong
about the player.

*Variance* -- the call was right and the outcome was bad. The alternative would
have lost too, so nothing should change. Treating this as a mistake is how a
model gets overfitted to noise by a well-meaning owner.

*Information miss* -- news broke before kickoff and no run picked it up in time.
This is the only category that indicts the bot rather than the world, and it is
measurable: compare the last status snapshot taken before lock against what was
knowable then.

Because the recap lands before this league's Tuesday waiver run, it doubles as
the input to that week's claims.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

from ..engine.lineup import Lineup, PlayerProjection, optimize_lineup, starting_slots
from ..model.scoring import score_stats
from ..sources.sleeper import SleeperClient, utcnow_iso

log = logging.getLogger(__name__)

# A starter missing his projection by less than this is noise, not a story.
NOTABLE_MISS = 5.0


def sync_actuals(
    conn: sqlite3.Connection,
    client: SleeperClient,
    season: int,
    week: int,
    scoring_settings: dict,
) -> int:
    """Store what every player actually scored, under this league's settings."""
    records = client.stats(season, week)
    now = utcnow_iso()
    rows = []
    for rec in records:
        pid = rec.get("player_id")
        stats = rec.get("stats") or {}
        if not pid or not stats:
            continue
        rows.append(
            (season, week, str(pid), score_stats(stats, scoring_settings), json.dumps(stats), now)
        )
    conn.executemany(
        """INSERT OR REPLACE INTO actuals (season, week, sleeper_id, points, stats, updated_at)
           VALUES (?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    log.info("Stored %d actual results for %d week %d", len(rows), season, week)
    return len(rows)


def load_actuals(conn: sqlite3.Connection, season: int, week: int) -> dict[str, float]:
    return {
        r["sleeper_id"]: float(r["points"] or 0.0)
        for r in conn.execute(
            "SELECT sleeper_id, points FROM actuals WHERE season = ? AND week = ?", (season, week)
        )
    }


@dataclass(frozen=True)
class PlayerResult:
    player: PlayerProjection
    projected: float
    actual: float
    started: bool

    @property
    def miss(self) -> float:
        return self.actual - self.projected


@dataclass
class Recap:
    week: int
    results: list[PlayerResult]
    actual_lineup: Lineup
    hindsight_lineup: Lineup
    my_points: float
    opponent_points: float
    opponent_name: str

    @property
    def points_left_on_bench(self) -> float:
        return self.hindsight_lineup.total_points - self.actual_lineup.total_points

    @property
    def played(self) -> bool:
        """Whether this week has actually happened.

        A week with no recorded scoring is one that has not been played, not one
        that was lost nil-nil. Reporting a defeat for a week yet to kick off
        would be worse than saying nothing.
        """
        return any(r.actual for r in self.results) or bool(
            self.my_points or self.opponent_points
        )

    @property
    def won(self) -> bool:
        return self.my_points > self.opponent_points

    @property
    def biggest_misses(self) -> list[PlayerResult]:
        started = [r for r in self.results if r.started]
        return sorted(started, key=lambda r: r.miss)[:3]

    @property
    def notable_bench(self) -> list[PlayerResult]:
        """Bench players who beat a starter by a meaningful margin."""
        started = [r for r in self.results if r.started]
        if not started:
            return []
        worst_starter = min(r.actual for r in started)
        return sorted(
            (r for r in self.results if not r.started and r.actual > worst_starter + NOTABLE_MISS),
            key=lambda r: r.actual,
            reverse=True,
        )[:3]


def build_recap(
    conn: sqlite3.Connection,
    snapshot_id: int,
    roster_id: int,
    players: list[PlayerProjection],
    roster_positions: list[str],
    season: int,
    week: int,
    my_points: float,
    opponent_points: float,
    opponent_name: str,
) -> Recap:
    """Compare what was recommended against what actually happened."""
    slots = starting_slots(roster_positions)
    actuals = load_actuals(conn, season, week)

    started_slots = {
        r["sleeper_id"]: r["slot_index"]
        for r in conn.execute(
            """SELECT sleeper_id, slot_index FROM roster_players
               WHERE snapshot_id = ? AND roster_id = ? AND is_starter = 1""",
            (snapshot_id, roster_id),
        )
    }
    started_ids = set(started_slots)

    results = [
        PlayerResult(
            player=p,
            projected=p.points,
            actual=actuals.get(p.sleeper_id, 0.0),
            started=p.sleeper_id in started_ids,
        )
        for p in players
    ]

    # Re-solve the week with hindsight: the best lineup that *could* have been
    # set, which is the only fair measure of points left on the bench.
    scored = [
        PlayerProjection(
            sleeper_id=p.sleeper_id,
            name=p.name,
            position=p.position,
            points=actuals.get(p.sleeper_id, 0.0),
            fantasy_positions=p.fantasy_positions,
            team=p.team,
        )
        for p in players
    ]
    hindsight = optimize_lineup(scored, slots)

    # Rebuild the lineup as it was actually set, using the recorded slot
    # positions rather than any ordering of our own.
    actual_lineup = Lineup(slots=list(slots))
    for p in scored:
        idx = started_slots.get(p.sleeper_id)
        if idx is not None and idx < len(slots):
            actual_lineup.assignments[int(idx)] = p

    return Recap(
        week=week,
        results=results,
        actual_lineup=actual_lineup,
        hindsight_lineup=hindsight,
        my_points=my_points,
        opponent_points=opponent_points,
        opponent_name=opponent_name,
    )
