"""SQLite schema and connection helpers.

Two ideas drive this schema:

1. *Availability is derived, never accumulated.* Rosters are snapshotted whole
   on every sync, so who is available is always recomputed from the latest
   snapshot rather than replayed from a draft plus a transaction log. Missing a
   week costs nothing.

2. *Store what was known at decision time.* ``projections`` and ``advisories``
   both carry an ``as_of`` timestamp. Calibration must score the projection the
   bot actually acted on, not one recomputed later with hindsight, or the
   measured accuracy is fiction.

Snapshot tables are written only when content changes, so they double as a
change log -- which is how late injury clearances are detected.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
-- Canonical player identity. Sleeper's dump is the Rosetta Stone: it carries
-- kalshi_id, which joins exactly to Kalshi's custom_strike.football_player.
CREATE TABLE IF NOT EXISTS players (
    sleeper_id        TEXT PRIMARY KEY,
    kalshi_id         TEXT,
    espn_id           INTEGER,     -- spotty upstream (~32% of top 100); do not rely on it
    gsis_id           TEXT,
    full_name         TEXT NOT NULL,
    search_name       TEXT,
    team              TEXT,
    position          TEXT,
    fantasy_positions TEXT,        -- JSON array
    active            INTEGER,
    status            TEXT,
    years_exp         INTEGER,
    depth_chart_order INTEGER,
    search_rank       INTEGER,
    -- Current injury state. Also appended to player_status_snapshots, but that
    -- is a change log; consumers need the latest value on a plain join.
    injury_status     TEXT,
    injury_body_part  TEXT,
    injury_notes      TEXT,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_players_kalshi ON players(kalshi_id);
CREATE INDEX IF NOT EXISTS idx_players_pos_team ON players(position, team);

-- Health / news signal. One row per observed *change*, so this is a change log.
-- practice_participation is deliberately absent: it is 0/300 populated upstream.
CREATE TABLE IF NOT EXISTS player_status_snapshots (
    sleeper_id       TEXT NOT NULL,
    observed_at      TEXT NOT NULL,
    injury_status    TEXT,
    injury_body_part TEXT,
    news_updated     INTEGER,
    PRIMARY KEY (sleeper_id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_status_player
    ON player_status_snapshots(sleeper_id, observed_at DESC);

-- League configuration, cached whole (scoring_settings, roster_positions).
CREATE TABLE IF NOT EXISTS league_meta (
    league_id  TEXT PRIMARY KEY,
    season     INTEGER,
    name       TEXT,
    payload    TEXT NOT NULL,   -- full JSON as returned
    updated_at TEXT NOT NULL
);

-- League members, so reports can name an opponent rather than print an id.
CREATE TABLE IF NOT EXISTS league_users (
    league_id    TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    display_name TEXT,
    team_name    TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (league_id, user_id)
);

-- Roster snapshots. content_hash lets us skip writing unchanged states, so
-- consecutive rows mean a roster genuinely moved.
CREATE TABLE IF NOT EXISTS roster_snapshots (
    snapshot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id    TEXT NOT NULL,
    taken_at     TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rsnap_league ON roster_snapshots(league_id, taken_at DESC);

CREATE TABLE IF NOT EXISTS roster_players (
    snapshot_id INTEGER NOT NULL REFERENCES roster_snapshots(snapshot_id) ON DELETE CASCADE,
    roster_id   INTEGER NOT NULL,
    owner_id    TEXT,
    sleeper_id  TEXT NOT NULL,
    is_starter  INTEGER NOT NULL DEFAULT 0,
    slot_index  INTEGER,          -- position within the starters array, else NULL
    on_reserve  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (snapshot_id, roster_id, sleeper_id)
);
CREATE INDEX IF NOT EXISTS idx_rplayers_snap ON roster_players(snapshot_id);

-- NFL schedule. status flips pre_game -> in progress -> complete and is the
-- authoritative per-player lock signal (no timezone arithmetic required).
CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    game_date   TEXT,
    home        TEXT,
    away        TEXT,
    status      TEXT,
    -- Exact kickoff from ESPN's public scoreboard. Used both for the
    -- countdown and, alongside status, to decide whether a slot has locked.
    kickoff_utc TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (game_id, season)
);
CREATE INDEX IF NOT EXISTS idx_games_week ON games(season, week);

-- Projections, one row per (source, player, week, as_of). Component stats are
-- kept as JSON so the league's own scoring settings can be reapplied later.
CREATE TABLE IF NOT EXISTS projections (
    source     TEXT NOT NULL,     -- 'sleeper' | 'kalshi' | 'blend'
    season     INTEGER NOT NULL,
    -- Week 0 means rest-of-season totals. Deliberately 0 and not NULL: SQLite
    -- treats NULLs as distinct in a PRIMARY KEY, so NULL would silently permit
    -- duplicate rows for the same player.
    week       INTEGER NOT NULL,
    sleeper_id TEXT NOT NULL,
    as_of      TEXT NOT NULL,
    mean       REAL,
    sd         REAL,
    opponent   TEXT,
    game_id    TEXT,
    stats      TEXT,              -- JSON component stats
    PRIMARY KEY (source, season, week, sleeper_id, as_of)
);
CREATE INDEX IF NOT EXISTS idx_proj_lookup ON projections(season, week, source, as_of DESC);

-- Actual results, for calibration and the recap.
CREATE TABLE IF NOT EXISTS actuals (
    season     INTEGER NOT NULL,
    week       INTEGER NOT NULL,
    sleeper_id TEXT NOT NULL,
    points     REAL,
    stats      TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (season, week, sleeper_id)
);

-- Kalshi market state. One row per (ticker, fetched_at).
CREATE TABLE IF NOT EXISTS kalshi_markets (
    ticker       TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    series       TEXT,
    event_ticker TEXT,
    kalshi_id    TEXT,            -- custom_strike.football_player
    team_uuid    TEXT,
    strike_type  TEXT,
    floor_strike REAL,
    yes_bid      REAL,
    yes_ask      REAL,
    title        TEXT,
    close_time   TEXT,
    PRIMARY KEY (ticker, fetched_at)
);
CREATE INDEX IF NOT EXISTS idx_kalshi_player ON kalshi_markets(kalshi_id, fetched_at DESC);

-- Every recommendation the bot emitted, with the deadline it was subject to.
-- This is what makes "did we miss news before lock?" answerable after the fact.
CREATE TABLE IF NOT EXISTS advisories (
    advisory_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id    TEXT NOT NULL,
    season       INTEGER NOT NULL,
    week         INTEGER NOT NULL,
    as_of        TEXT NOT NULL,
    kind         TEXT NOT NULL,   -- 'lineup' | 'waiver' | 'trade'
    payload      TEXT NOT NULL,   -- JSON
    deadline_utc TEXT,
    model_version INTEGER
);
CREATE INDEX IF NOT EXISTS idx_adv_week ON advisories(league_id, season, week, as_of DESC);

-- Fitted model parameters, versioned so a report can name what produced it.
CREATE TABLE IF NOT EXISTS model_params (
    version    INTEGER PRIMARY KEY AUTOINCREMENT,
    fitted_at  TEXT NOT NULL,
    through_week INTEGER,
    params     TEXT NOT NULL,     -- JSON
    notes      TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with sane pragmas and dict-like rows."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def open_db(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()
