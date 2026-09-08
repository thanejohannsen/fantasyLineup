"""Assembling one advisory from all the sources.

Kept separate from the CLI so the same assembly serves the terminal report, the
published page and the tests without any of them re-deriving it.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from .engine.lineup import PlayerProjection, starting_slots
from .engine.locks import locked_slot_assignments, player_lock_states
from .model.blend import BlendResult, blend_player, fit_ladders, group_quotes
from .model.projections import latest_projections
from .model.variance import VarianceModel, default_model
from .sources.kalshi import KalshiClient
from .sources.sleeper import SleeperClient
from .sync import compute_availability

log = logging.getLogger(__name__)


@dataclass
class MatchupContext:
    """Who we are playing, and what has already been scored."""

    opponent_roster_id: int | None
    opponent_name: str
    my_banked: float = 0.0
    opponent_banked: float = 0.0


def find_opponent(
    conn: sqlite3.Connection, client: SleeperClient, league_id: str, roster_id: int, week: int
) -> MatchupContext:
    """Resolve this week's opponent from the matchup pairing.

    ``points`` on a matchup entry is what has already been scored this week, so
    it doubles as the banked total once games start.
    """
    entries = client.matchups(league_id, week)
    mine = next((m for m in entries if m.get("roster_id") == roster_id), None)
    if mine is None or mine.get("matchup_id") is None:
        return MatchupContext(None, "unknown", 0.0, 0.0)

    theirs = next(
        (
            m
            for m in entries
            if m.get("matchup_id") == mine["matchup_id"] and m.get("roster_id") != roster_id
        ),
        None,
    )
    if theirs is None:
        return MatchupContext(None, "bye", float(mine.get("points") or 0.0), 0.0)

    opponent_id = int(theirs["roster_id"])
    row = conn.execute(
        """SELECT COALESCE(u.team_name, u.display_name) AS name
           FROM roster_players rp
           LEFT JOIN league_users u
             ON u.user_id = rp.owner_id AND u.league_id = ?
           WHERE rp.roster_id = ? AND rp.snapshot_id = (
               SELECT MAX(snapshot_id) FROM roster_snapshots WHERE league_id = ?)
           LIMIT 1""",
        (league_id, opponent_id, league_id),
    ).fetchone()
    name = row["name"] if row else None

    return MatchupContext(
        opponent_roster_id=opponent_id,
        opponent_name=name or f"roster {opponent_id}",
        my_banked=float(mine.get("points") or 0.0),
        opponent_banked=float(theirs.get("points") or 0.0),
    )


def blended_projections(
    conn: sqlite3.Connection,
    player_ids: set[str],
    season: int,
    week: int,
    scoring_settings: Mapping[str, float],
    kalshi_fits: Mapping[tuple[str, str], Any],
    max_shift: float,
) -> tuple[list[PlayerProjection], dict[str, BlendResult]]:
    """Build optimizer inputs with Kalshi folded into the Sleeper baseline."""
    if not player_ids:
        return [], {}

    projections = latest_projections(conn, season, week)
    placeholders = ",".join("?" * len(player_ids))
    rows = conn.execute(
        f"""SELECT sleeper_id, full_name, position, fantasy_positions, team, kalshi_id
            FROM players WHERE sleeper_id IN ({placeholders})""",
        list(player_ids),
    ).fetchall()

    # Regroup market fits by player so each roster entry can be looked up once.
    by_player: dict[str, dict[str, Any]] = {}
    for (kalshi_id, stat), fit in kalshi_fits.items():
        by_player.setdefault(kalshi_id, {})[stat] = fit

    players: list[PlayerProjection] = []
    blends: dict[str, BlendResult] = {}
    for r in rows:
        proj = projections.get(r["sleeper_id"], {})
        stats = proj.get("stats") or {}
        fits = by_player.get(r["kalshi_id"] or "", {})

        blend = blend_player(r["sleeper_id"], stats, fits, scoring_settings, max_shift)
        blends[r["sleeper_id"]] = blend

        import json as _json

        positions = frozenset(_json.loads(r["fantasy_positions"] or "[]"))
        players.append(
            PlayerProjection(
                sleeper_id=r["sleeper_id"],
                name=r["full_name"],
                position=r["position"],
                points=blend.blended,
                fantasy_positions=positions or frozenset({r["position"]}),
                team=r["team"],
                opponent=proj.get("opponent"),
            )
        )
    return players, blends


def standard_deviations(
    players: list[PlayerProjection],
    blends: Mapping[str, BlendResult],
    model: VarianceModel | None = None,
) -> dict[str, float]:
    model = model or default_model()
    out: dict[str, float] = {}
    for p in players:
        blend = blends.get(p.sleeper_id)
        out[p.sleeper_id] = model.blend_market_sd(
            p.position, p.points, blend.sd if blend else None
        )
    return out


def fetch_market_fits(kalshi_base: str) -> dict[tuple[str, str], Any]:
    """Pull and fit every player ladder Kalshi is currently quoting.

    Failures are logged and swallowed: a Kalshi outage should degrade the
    advisory to Sleeper-only, not break it. The report says which players
    carried market information, so a silent degradation is still visible.
    """
    try:
        with KalshiClient(kalshi_base) as client:
            quotes = client.player_quotes()
    except Exception as exc:  # noqa: BLE001 - degrade rather than fail
        log.warning("Kalshi unavailable, falling back to Sleeper only: %s", exc)
        return {}
    fits = fit_ladders(group_quotes(quotes))
    log.info("Kalshi: %d quotes -> %d trustworthy ladder fits", len(quotes), len(fits))
    return fits


def roster_players_for(
    conn: sqlite3.Connection, league_id: str, roster_id: int
) -> set[str]:
    row = conn.execute(
        "SELECT MAX(snapshot_id) AS sid FROM roster_snapshots WHERE league_id = ?",
        (league_id,),
    ).fetchone()
    if row is None or row["sid"] is None:
        return set()
    return {
        r["sleeper_id"]
        for r in conn.execute(
            "SELECT sleeper_id FROM roster_players WHERE snapshot_id = ? AND roster_id = ?",
            (row["sid"], roster_id),
        )
    }


def lock_context(
    conn: sqlite3.Connection,
    players: list[PlayerProjection],
    season: int,
    week: int,
    snapshot_id: int,
    roster_id: int,
    now: datetime | None = None,
):
    now = now or datetime.now(UTC)
    states = player_lock_states(conn, players, season, week, now=now)
    forced = locked_slot_assignments(conn, snapshot_id, roster_id, states)
    return states, forced


__all__ = [
    "MatchupContext",
    "blended_projections",
    "compute_availability",
    "fetch_market_fits",
    "find_opponent",
    "lock_context",
    "roster_players_for",
    "standard_deviations",
    "starting_slots",
]
