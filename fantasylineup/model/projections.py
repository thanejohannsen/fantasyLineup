"""Projection ingest and assembly.

Sleeper's projections arrive as component stats, which are scored here with the
league's own settings. Every row is stamped with an ``as_of`` timestamp and kept
rather than overwritten, because calibration later has to grade *the projection
the bot actually acted on*. Recomputing a past week's projection from today's
data would quietly leak hindsight and make the measured accuracy meaningless.

Rest-of-season totals are stored under week 0.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Mapping

from ..engine.lineup import PlayerProjection
from ..sources.sleeper import SKILL_POSITIONS, SleeperClient, utcnow_iso
from .scoring import score_stats

log = logging.getLogger(__name__)

REST_OF_SEASON = 0
SOURCE_SLEEPER = "sleeper"


def sync_projections(
    conn: sqlite3.Connection,
    client: SleeperClient,
    season: int,
    week: int | None,
    scoring_settings: Mapping[str, float],
    positions: tuple[str, ...] = SKILL_POSITIONS,
) -> int:
    """Fetch and store projections, scored under this league's settings.

    ``week=None`` stores rest-of-season totals (as week 0), which is the horizon
    trade valuation needs; a week number stores that week, which is what lineup
    decisions need.
    """
    records = client.projections(season, week=week, positions=positions)
    now = utcnow_iso()
    stored_week = REST_OF_SEASON if week is None else week

    rows = []
    for rec in records:
        pid = rec.get("player_id")
        stats = rec.get("stats") or {}
        if not pid or not stats:
            continue
        # Skip records with no actual projection; the endpoint returns the whole
        # position group including players with nothing projected.
        if stats.get("pts_ppr") is None:
            continue
        rows.append(
            (
                SOURCE_SLEEPER,
                season,
                stored_week,
                str(pid),
                now,
                score_stats(stats, scoring_settings),
                None,  # sd is filled in by the variance model in a later phase
                rec.get("opponent"),
                rec.get("game_id"),
                json.dumps(stats),
            )
        )

    conn.executemany(
        """INSERT OR REPLACE INTO projections
           (source, season, week, sleeper_id, as_of, mean, sd, opponent, game_id, stats)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    label = "rest-of-season" if week is None else f"week {week}"
    log.info("Stored %d %s projections for %d", len(rows), label, season)
    return len(rows)


def latest_projections(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    source: str = SOURCE_SLEEPER,
) -> dict[str, dict[str, Any]]:
    """Most recent projection per player for one week.

    Uses the newest ``as_of`` per player rather than a single global timestamp,
    so a partial refresh (one position group re-fetched) does not blank out the
    rest.
    """
    rows = conn.execute(
        """
        SELECT p.sleeper_id, p.mean, p.sd, p.opponent, p.game_id, p.stats, p.as_of
        FROM projections p
        JOIN (SELECT sleeper_id, MAX(as_of) AS mx
              FROM projections
              WHERE source = ? AND season = ? AND week = ?
              GROUP BY sleeper_id) m
          ON m.sleeper_id = p.sleeper_id AND m.mx = p.as_of
        WHERE p.source = ? AND p.season = ? AND p.week = ?
        """,
        (source, season, week, source, season, week),
    )
    return {
        r["sleeper_id"]: {
            "mean": r["mean"],
            "sd": r["sd"],
            "opponent": r["opponent"],
            "game_id": r["game_id"],
            "stats": json.loads(r["stats"]) if r["stats"] else {},
            "as_of": r["as_of"],
        }
        for r in rows
    }


def build_player_projections(
    conn: sqlite3.Connection,
    player_ids: set[str],
    season: int,
    week: int,
    source: str = SOURCE_SLEEPER,
) -> list[PlayerProjection]:
    """Assemble optimizer inputs for a set of players.

    A player with no projection is included at zero rather than dropped. On a
    bye or after a mid-week signing there is genuinely no projection, and
    dropping him would hide the fact that a roster spot is producing nothing.
    """
    if not player_ids:
        return []

    projections = latest_projections(conn, season, week, source=source)
    placeholders = ",".join("?" * len(player_ids))
    rows = conn.execute(
        f"""SELECT sleeper_id, full_name, position, fantasy_positions, team
            FROM players WHERE sleeper_id IN ({placeholders})""",
        list(player_ids),
    ).fetchall()

    out = []
    for r in rows:
        proj = projections.get(r["sleeper_id"], {})
        fantasy_positions = frozenset(json.loads(r["fantasy_positions"] or "[]"))
        out.append(
            PlayerProjection(
                sleeper_id=r["sleeper_id"],
                name=r["full_name"],
                position=r["position"],
                points=float(proj.get("mean") or 0.0),
                fantasy_positions=fantasy_positions or frozenset({r["position"]}),
                team=r["team"],
                opponent=proj.get("opponent"),
            )
        )
    return out
