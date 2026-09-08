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
from .model.health import HealthStatus, Regime, classify, is_structural, ros_multiplier
from .model.projections import REST_OF_SEASON, latest_projections
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
    health_multipliers: Mapping[str, float] | None = None,
    weekly_week: int | None = None,
) -> tuple[list[PlayerProjection], dict[str, BlendResult]]:
    """Build optimizer inputs with Kalshi folded into the Sleeper baseline.

    Injury haircuts are applied here because this is the single choke point
    every command shares, so no caller can accidentally reason about a
    season-ending injury as though the player were available.

    They apply to rest-of-season projections only. Weekly projections need no
    haircut: Sleeper simply omits a player who is not playing, so an unavailable
    player already arrives at zero. Rest-of-season numbers carry no such
    treatment -- one quarterback was priced at 143 points while out for the year
    with a reconstructed ACL.
    """
    if not player_ids:
        return [], {}

    projections = latest_projections(conn, season, week)
    # Whether a *weekly* projection exists is what separates "back from last
    # year's surgery and playing" from "had that surgery three weeks ago".
    is_ros = week == REST_OF_SEASON
    weekly = (
        latest_projections(conn, season, weekly_week)
        if is_ros and weekly_week is not None
        else projections
    )

    # An empty weekly table means the data was never fetched, not that every
    # injured player in the league is finished for the year. Without this guard
    # a missing sync silently zeroes every player carrying a designation and
    # quietly drops them from trades -- destructive, and invisible.
    weekly_known = bool(weekly)
    if is_ros and not weekly_known:
        log.warning(
            "No week %s projections stored; treating injury designations as "
            "playing rather than out. Run a weekly projection sync for accurate "
            "injury handling.",
            weekly_week,
        )

    placeholders = ",".join("?" * len(player_ids))
    rows = conn.execute(
        f"""SELECT sleeper_id, full_name, position, fantasy_positions, team, kalshi_id,
                   injury_status, injury_body_part, injury_notes
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

        health = classify(
            r["injury_status"],
            r["injury_body_part"],
            r["injury_notes"],
            # Absent weekly data is unknown, not "out": assume playing.
            has_weekly_projection=(r["sleeper_id"] in weekly) or not weekly_known,
        )
        points = blend.blended
        if is_ros:
            points *= ros_multiplier(health, health_multipliers)

        import json as _json

        positions = frozenset(_json.loads(r["fantasy_positions"] or "[]"))
        players.append(
            PlayerProjection(
                sleeper_id=r["sleeper_id"],
                name=r["full_name"],
                position=r["position"],
                points=points,
                fantasy_positions=positions or frozenset({r["position"]}),
                team=r["team"],
                opponent=proj.get("opponent"),
                injury_status=r["injury_status"],
                injury_body_part=r["injury_body_part"],
                injury_notes=r["injury_notes"],
                health_regime=health.regime.value,
            )
        )
    return players, blends


def health_of(player: PlayerProjection) -> HealthStatus:
    """The health status carried on a projection, without re-deriving it."""
    regime = Regime(player.health_regime) if player.health_regime else Regime.HEALTHY
    return HealthStatus(
        regime=regime,
        status=player.injury_status,
        body_part=player.injury_body_part,
        notes=player.injury_notes,
        structural=is_structural(player.injury_body_part, player.injury_notes),
    )


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
    "LeagueMove",
    "MatchupContext",
    "league_activity",
    "all_rosters",
    "roster_names",
    "blended_projections",
    "compute_availability",
    "health_of",
    "current_starters",
    "fetch_market_fits",
    "find_opponent",
    "lock_context",
    "roster_players_for",
    "standard_deviations",
    "starting_slots",
]


def all_rosters(conn: sqlite3.Connection, league_id: str) -> dict[int, set[str]]:
    """Every team's current roster, from the newest snapshot."""
    row = conn.execute(
        "SELECT MAX(snapshot_id) AS sid FROM roster_snapshots WHERE league_id = ?",
        (league_id,),
    ).fetchone()
    if row is None or row["sid"] is None:
        return {}
    out: dict[int, set[str]] = {}
    for r in conn.execute(
        "SELECT roster_id, sleeper_id FROM roster_players WHERE snapshot_id = ?", (row["sid"],)
    ):
        out.setdefault(int(r["roster_id"]), set()).add(r["sleeper_id"])
    return out


def roster_names(conn: sqlite3.Connection, league_id: str) -> dict[int, str]:
    row = conn.execute(
        "SELECT MAX(snapshot_id) AS sid FROM roster_snapshots WHERE league_id = ?",
        (league_id,),
    ).fetchone()
    if row is None or row["sid"] is None:
        return {}
    return {
        int(r["roster_id"]): r["name"] or f"roster {r['roster_id']}"
        for r in conn.execute(
            """SELECT DISTINCT rp.roster_id,
                      COALESCE(u.team_name, u.display_name) AS name
               FROM roster_players rp
               LEFT JOIN league_users u
                 ON u.user_id = rp.owner_id AND u.league_id = ?
               WHERE rp.snapshot_id = ?""",
            (league_id, row["sid"]),
        )
    }


@dataclass(frozen=True)
class LeagueMove:
    """One completed transaction, in plain language."""

    kind: str          # 'trade' | 'waiver' | 'free_agent'
    when: datetime | None
    summary: str
    involves_me: bool
    player_ids: frozenset[str]


def league_activity(
    conn: sqlite3.Connection,
    client: SleeperClient,
    league_id: str,
    week: int,
    my_roster_id: int,
    limit: int = 8,
) -> list[LeagueMove]:
    """Recent adds, drops and trades across the league.

    This is narrative only. Correctness never depends on it: availability and
    every roster figure are derived from a fresh snapshot on each run, so a
    completed trade is reflected whether or not its transaction record was ever
    read. The feed exists to say *why* something changed -- that a waiver target
    is gone, or that a proposal is void because the player moved.
    """
    try:
        raw = client.transactions(league_id, week)
    except Exception as exc:  # noqa: BLE001 - narrative only, never fatal
        log.warning("Could not read transactions: %s", exc)
        return []

    names = {
        r["sleeper_id"]: r["full_name"]
        for r in conn.execute("SELECT sleeper_id, full_name FROM players")
    }
    teams = roster_names(conn, league_id)

    moves: list[LeagueMove] = []
    for entry in raw:
        if entry.get("status") != "complete":
            continue
        adds = entry.get("adds") or {}
        drops = entry.get("drops") or {}
        roster_ids = [int(r) for r in (entry.get("roster_ids") or [])]
        when = None
        if entry.get("status_updated"):
            when = datetime.fromtimestamp(entry["status_updated"] / 1000, tz=UTC)

        def who(pid: str, mapping: dict) -> str:
            rid = mapping.get(pid)
            return teams.get(int(rid), f"roster {rid}") if rid is not None else "?"

        kind = entry.get("type") or "move"
        if kind == "trade":
            parts = [
                f"{names.get(pid, pid)} to {who(pid, adds)}" for pid in adds
            ]
            summary = "Trade: " + "; ".join(parts) if parts else "Trade completed"
        else:
            added = ", ".join(names.get(pid, pid) for pid in adds)
            dropped = ", ".join(names.get(pid, pid) for pid in drops)
            team = teams.get(roster_ids[0], "someone") if roster_ids else "someone"
            label = "claimed" if kind == "waiver" else "added"
            # A drop with no corresponding add is a plain cut, not an
            # acquisition of nobody.
            if added and dropped:
                summary = f"{team} {label} {added}, dropped {dropped}"
            elif added:
                summary = f"{team} {label} {added}"
            elif dropped:
                summary = f"{team} dropped {dropped}"
            else:
                summary = f"{team} made a roster move"

        moves.append(
            LeagueMove(
                kind=kind,
                when=when,
                summary=summary,
                involves_me=my_roster_id in roster_ids,
                player_ids=frozenset({*adds, *drops}),
            )
        )

    moves.sort(key=lambda m: m.when or datetime.min.replace(tzinfo=UTC), reverse=True)
    return moves[:limit]


def current_starters(
    conn: sqlite3.Connection,
    league_id: str,
    roster_id: int,
    players: list[PlayerProjection],
) -> list[PlayerProjection]:
    """The lineup a team actually has set, in slot order.

    Distinct from the lineup we would choose for them. Any claim made to another
    manager about his own roster has to be measured against this, or it is
    trivially falsifiable.
    """
    row = conn.execute(
        "SELECT MAX(snapshot_id) AS sid FROM roster_snapshots WHERE league_id = ?",
        (league_id,),
    ).fetchone()
    if row is None or row["sid"] is None:
        return []
    by_id = {p.sleeper_id: p for p in players}
    rows = conn.execute(
        """SELECT sleeper_id, slot_index FROM roster_players
           WHERE snapshot_id = ? AND roster_id = ? AND is_starter = 1
           ORDER BY slot_index""",
        (row["sid"], roster_id),
    )
    return [by_id[r["sleeper_id"]] for r in rows if r["sleeper_id"] in by_id]
