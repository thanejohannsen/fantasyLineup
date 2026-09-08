"""Ingest: pull Sleeper state into the local database.

The central invariant is that **availability is derived, never accumulated**.
Every sync writes a complete roster snapshot; who is available is then
``universe - union(all rosters)`` computed fresh from the newest snapshot. A
missed week, a crashed job or a month of downtime costs nothing, because no
state is being replayed forward. The transaction log is read only to explain
*why* something changed, never to decide what is true.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from .sources.sleeper import SleeperClient, utcnow_iso

log = logging.getLogger(__name__)

# Positions that can occupy a fantasy roster spot in this league.
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


def is_fantasy_relevant(player: dict[str, Any]) -> bool:
    """Whether a player can occupy a fantasy roster spot.

    Must be judged on ``fantasy_positions``, not the primary ``position``.
    Sleeper lists Travis Hunter as ``position: "DB"`` with
    ``fantasy_positions: ["DB", "WR"]`` -- filtering on the primary position
    drops him from the player table entirely even though he is rostered and
    started as a wide receiver. Two-way and hybrid players are rare enough to
    miss in testing and prominent enough to matter when missed.
    """
    positions = set(player.get("fantasy_positions") or [])
    if player.get("position"):
        positions.add(player["position"])
    return bool(positions & FANTASY_POSITIONS)


# --------------------------------------------------------------------- players


def sync_players(conn: sqlite3.Connection, client: SleeperClient, ttl_hours: int = 24) -> int:
    """Upsert the canonical player table from Sleeper's full dump.

    Also records a health snapshot for any player whose ``injury_status`` or
    ``news_updated`` changed since we last looked. Those two fields are the only
    usable health signal: ``practice_participation`` exists in the payload but
    is empty for every player we sampled, so relying on it would silently do
    nothing.
    """
    dump = client.players_dump(ttl_hours=ttl_hours)
    now = utcnow_iso()

    rows = []
    for pid, p in dump.items():
        if not is_fantasy_relevant(p):
            continue
        pos = p.get("position")
        rows.append(
            (
                pid,
                p.get("kalshi_id"),
                p.get("espn_id"),
                p.get("gsis_id"),
                p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
                p.get("search_full_name"),
                p.get("team"),
                pos,
                json.dumps(p.get("fantasy_positions") or []),
                1 if p.get("active") else 0,
                p.get("status"),
                p.get("years_exp"),
                p.get("depth_chart_order"),
                p.get("search_rank"),
                now,
            )
        )

    conn.executemany(
        """
        INSERT INTO players (sleeper_id, kalshi_id, espn_id, gsis_id, full_name, search_name,
                             team, position, fantasy_positions, active, status, years_exp,
                             depth_chart_order, search_rank, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(sleeper_id) DO UPDATE SET
            kalshi_id=excluded.kalshi_id, espn_id=excluded.espn_id, gsis_id=excluded.gsis_id,
            full_name=excluded.full_name, search_name=excluded.search_name, team=excluded.team,
            position=excluded.position, fantasy_positions=excluded.fantasy_positions,
            active=excluded.active, status=excluded.status, years_exp=excluded.years_exp,
            depth_chart_order=excluded.depth_chart_order, search_rank=excluded.search_rank,
            updated_at=excluded.updated_at
        """,
        rows,
    )
    _record_status_changes(conn, dump, now)
    conn.commit()
    log.info("Synced %d fantasy-relevant players", len(rows))
    return len(rows)


def _record_status_changes(
    conn: sqlite3.Connection, dump: dict[str, dict[str, Any]], now: str
) -> int:
    """Append a health row only when something actually moved.

    Writing unconditionally would bury real changes in noise; writing on change
    turns this table into a change log, which is exactly what "was he cleared
    since I last checked?" needs.
    """
    latest = {
        r["sleeper_id"]: (r["injury_status"], r["news_updated"])
        for r in conn.execute(
            """
            SELECT s.sleeper_id, s.injury_status, s.news_updated
            FROM player_status_snapshots s
            JOIN (SELECT sleeper_id, MAX(observed_at) AS mx
                  FROM player_status_snapshots GROUP BY sleeper_id) m
              ON m.sleeper_id = s.sleeper_id AND m.mx = s.observed_at
            """
        )
    }

    changed = []
    for pid, p in dump.items():
        if not is_fantasy_relevant(p):
            continue
        cur = (p.get("injury_status"), p.get("news_updated"))
        if latest.get(pid) != cur:
            changed.append((pid, now, p.get("injury_status"), p.get("injury_body_part"), cur[1]))

    if changed:
        conn.executemany(
            """INSERT OR REPLACE INTO player_status_snapshots
               (sleeper_id, observed_at, injury_status, injury_body_part, news_updated)
               VALUES (?,?,?,?,?)""",
            changed,
        )
    log.info("Recorded %d player status changes", len(changed))
    return len(changed)


# ---------------------------------------------------------------------- league


def sync_league(conn: sqlite3.Connection, client: SleeperClient, league_id: str) -> dict[str, Any]:
    league = client.league(league_id)
    if not league:
        raise ValueError(f"Sleeper returned no league for id {league_id}")
    conn.execute(
        """INSERT INTO league_meta (league_id, season, name, payload, updated_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(league_id) DO UPDATE SET
             season=excluded.season, name=excluded.name,
             payload=excluded.payload, updated_at=excluded.updated_at""",
        (league_id, int(league["season"]), league.get("name"), json.dumps(league), utcnow_iso()),
    )
    conn.commit()
    return league


def load_league(conn: sqlite3.Connection, league_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT payload FROM league_meta WHERE league_id = ?", (league_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"League {league_id} not synced yet; run `fl sync` first.")
    return json.loads(row["payload"])


# --------------------------------------------------------------------- rosters


def _roster_hash(rosters: Iterable[dict[str, Any]]) -> str:
    canon = sorted(
        (r["roster_id"], tuple(sorted(r.get("players") or [])), tuple(r.get("starters") or []))
        for r in rosters
    )
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()


def sync_rosters(conn: sqlite3.Connection, client: SleeperClient, league_id: str) -> int:
    """Snapshot every roster. Returns the snapshot id (existing one if unchanged)."""
    rosters = client.rosters(league_id)
    digest = _roster_hash(rosters)

    prev = conn.execute(
        "SELECT snapshot_id, content_hash FROM roster_snapshots "
        "WHERE league_id = ? ORDER BY taken_at DESC LIMIT 1",
        (league_id,),
    ).fetchone()
    if prev and prev["content_hash"] == digest:
        log.info("Rosters unchanged; reusing snapshot %d", prev["snapshot_id"])
        return int(prev["snapshot_id"])

    cur = conn.execute(
        "INSERT INTO roster_snapshots (league_id, taken_at, content_hash) VALUES (?,?,?)",
        (league_id, utcnow_iso(), digest),
    )
    snapshot_id = int(cur.lastrowid)

    rows = []
    for r in rosters:
        starters = r.get("starters") or []
        reserve = set(r.get("reserve") or [])
        starter_index = {pid: i for i, pid in enumerate(starters) if pid and pid != "0"}
        for pid in r.get("players") or []:
            rows.append(
                (
                    snapshot_id,
                    r["roster_id"],
                    r.get("owner_id"),
                    pid,
                    1 if pid in starter_index else 0,
                    starter_index.get(pid),
                    1 if pid in reserve else 0,
                )
            )
    conn.executemany(
        """INSERT OR REPLACE INTO roster_players
           (snapshot_id, roster_id, owner_id, sleeper_id, is_starter, slot_index, on_reserve)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    log.info("Snapshot %d: %d roster entries across %d teams", snapshot_id, len(rows), len(rosters))
    return snapshot_id


def latest_snapshot_id(conn: sqlite3.Connection, league_id: str) -> int:
    row = conn.execute(
        "SELECT snapshot_id FROM roster_snapshots WHERE league_id = ? "
        "ORDER BY taken_at DESC LIMIT 1",
        (league_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"No roster snapshot for league {league_id}; run `fl sync` first.")
    return int(row["snapshot_id"])


# -------------------------------------------------------------------- schedule


def sync_schedule(conn: sqlite3.Connection, client: SleeperClient, season: int) -> int:
    games = client.schedule(season)
    now = utcnow_iso()
    rows = [
        (
            g["game_id"],
            season,
            g.get("week"),
            g.get("date"),
            g.get("home"),
            g.get("away"),
            g.get("status"),
            now,
        )
        for g in games
    ]
    conn.executemany(
        """INSERT INTO games (game_id, season, week, game_date, home, away, status, updated_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(game_id, season) DO UPDATE SET
             week=excluded.week, game_date=excluded.game_date, home=excluded.home,
             away=excluded.away, status=excluded.status, updated_at=excluded.updated_at""",
        rows,
    )
    conn.commit()
    log.info("Synced %d games for %d", len(rows), season)
    return len(rows)


# ---------------------------------------------------------------- availability


@dataclass(frozen=True)
class Availability:
    """Who is rostered where, and who is free, as of one snapshot."""

    snapshot_id: int
    rostered: dict[str, int]          # sleeper_id -> roster_id
    my_players: set[str]
    free_agents: set[str]

    @property
    def rostered_count(self) -> int:
        return len(self.rostered)


def compute_availability(
    conn: sqlite3.Connection, league_id: str, my_roster_id: int
) -> Availability:
    """Derive availability by subtraction from the newest roster snapshot.

    Deliberately stateless: nothing here depends on having observed the draft or
    any intervening transaction. Whatever the rosters say right now is the
    answer.
    """
    snapshot_id = latest_snapshot_id(conn, league_id)

    rostered: dict[str, int] = {}
    duplicates: list[str] = []
    for row in conn.execute(
        "SELECT sleeper_id, roster_id FROM roster_players WHERE snapshot_id = ?", (snapshot_id,)
    ):
        pid = row["sleeper_id"]
        if pid in rostered:
            duplicates.append(pid)
        rostered[pid] = int(row["roster_id"])
    if duplicates:
        # Would mean Sleeper handed us an inconsistent view; refuse to guess.
        raise ValueError(f"{len(duplicates)} player(s) on multiple rosters: {duplicates[:5]}")

    universe = {
        r["sleeper_id"]
        for r in conn.execute(
            "SELECT sleeper_id FROM players WHERE active = 1 AND team IS NOT NULL"
        )
    }
    my_players = {pid for pid, rid in rostered.items() if rid == my_roster_id}
    return Availability(
        snapshot_id=snapshot_id,
        rostered=rostered,
        my_players=my_players,
        free_agents=universe - set(rostered),
    )
