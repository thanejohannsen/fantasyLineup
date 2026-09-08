"""Sleeper API client.

The API is free, unauthenticated and **read-only** -- Sleeper's docs state
plainly that "you cannot modify contents via this API". The bot therefore only
ever *suggests*; lineup changes, waiver claims and trade offers are executed by
hand in the app.

Two families of endpoint are used:

``/v1/...``
    Documented and stable: league, rosters, users, matchups, transactions, and
    the full player dump.

``/projections/...``, ``/stats/...``, ``/schedule/...``
    Undocumented but live, and far more useful than the documented surface
    because they return *component* stats (rush_yd, rec, rec_td, ...) rather
    than a single pre-scored total. That lets us apply this league's own 43
    scoring keys instead of trusting a generic PPR number. Being undocumented,
    they may change without notice -- hence the fixture-backed tests, so a
    break is loud and localised here rather than silently wrong downstream.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


class RateLimiter:
    """Simple sliding-window limiter.

    Sleeper's guidance is to stay under 1000 calls/minute or risk an IP block. A
    full sync is roughly 20 calls, so this exists purely as a guard rail against
    a runaway loop.
    """

    def __init__(self, max_per_min: int) -> None:
        self.max_per_min = max_per_min
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= self.max_per_min:
            sleep_for = 60 - (now - self._calls[0]) + 0.05
            log.warning("Rate limit reached, sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)
            return self.acquire()
        self._calls.append(now)


class SleeperClient:
    def __init__(
        self,
        base_url: str,
        cache_dir: Path,
        max_calls_per_min: int = 120,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = RateLimiter(max_calls_per_min)
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "fantasylineup/0.1 (+personal fantasy advisor)"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SleeperClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- caching

    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path:
        key = json.dumps([path, params or {}], sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        safe = path.strip("/").replace("/", "_")[:60] or "root"
        return self.cache_dir / f"{safe}.{digest}.json"

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> Any:
        cache_file = self._cache_path(path, params)
        if ttl_seconds is not None and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < ttl_seconds:
                try:
                    return json.loads(cache_file.read_text())
                except json.JSONDecodeError:
                    log.warning("Corrupt cache at %s, refetching", cache_file)

        self._limiter.acquire()
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        # Write via a temp file so a crash mid-write cannot leave torn JSON.
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(cache_file)
        return data

    # ------------------------------------------------------- documented (v1)

    def state(self) -> dict[str, Any]:
        """Current season/week. ``display_week`` is what the app shows."""
        return self._get("/v1/state/nfl", ttl_seconds=600)

    def league(self, league_id: str) -> dict[str, Any]:
        """League config including ``scoring_settings`` and ``roster_positions``."""
        return self._get(f"/v1/league/{league_id}", ttl_seconds=3600)

    def rosters(self, league_id: str) -> list[dict[str, Any]]:
        """Current rosters for every team. This is the availability source of truth."""
        return self._get(f"/v1/league/{league_id}/rosters", ttl_seconds=0)

    def users(self, league_id: str) -> list[dict[str, Any]]:
        return self._get(f"/v1/league/{league_id}/users", ttl_seconds=3600)

    def matchups(self, league_id: str, week: int) -> list[dict[str, Any]]:
        """Weekly matchups; entries sharing a ``matchup_id`` are opponents."""
        return self._get(f"/v1/league/{league_id}/matchups/{week}", ttl_seconds=0)

    def transactions(self, league_id: str, week: int) -> list[dict[str, Any]]:
        """Adds/drops/trades. Used for narrative only -- never as availability truth."""
        return self._get(f"/v1/league/{league_id}/transactions/{week}", ttl_seconds=300)

    def players_dump(self, ttl_hours: int = 24) -> dict[str, dict[str, Any]]:
        """The full ~14.6MB player universe (~12k players).

        Sleeper asks that this be fetched at most once a day, so the TTL is
        long by default. It is also where ``kalshi_id`` comes from.
        """
        return self._get("/v1/players/nfl", ttl_seconds=ttl_hours * 3600)

    # ----------------------------------------------------------- undocumented

    def schedule(self, season: int, season_type: str = "regular") -> list[dict[str, Any]]:
        """All games for the season with a live ``status`` field.

        ``status`` moves pre_game -> in-progress -> complete, which is the
        authoritative signal for whether a player's roster slot has locked.
        """
        return self._get(f"/schedule/nfl/{season_type}/{season}", ttl_seconds=300)

    def projections(
        self,
        season: int,
        week: int | None = None,
        positions: tuple[str, ...] = SKILL_POSITIONS,
        season_type: str = "regular",
        ttl_seconds: float = 900,
    ) -> list[dict[str, Any]]:
        """Projections with component stats.

        ``week=None`` returns rest-of-season totals, which is the right horizon
        for trade valuation; a week number returns that week only, which is the
        right horizon for lineup decisions.
        """
        path = f"/projections/nfl/{season}" if week is None else f"/projections/nfl/{season}/{week}"
        params = {
            "season_type": season_type,
            "position[]": list(positions),
            "order_by": "pts_ppr",
        }
        return self._get(path, params=params, ttl_seconds=ttl_seconds)

    def stats(
        self,
        season: int,
        week: int,
        positions: tuple[str, ...] = SKILL_POSITIONS,
        season_type: str = "regular",
        ttl_seconds: float = 900,
    ) -> list[dict[str, Any]]:
        """Actual results for a completed week, with the same component stats.

        This is the calibration fuel: roughly 2,000+ skill-player rows a week
        across the whole NFL, which is what makes fitting a handful of model
        parameters defensible when a single fantasy roster never could be.
        """
        params = {
            "season_type": season_type,
            "position[]": list(positions),
            "order_by": "pts_ppr",
        }
        return self._get(f"/stats/nfl/{season}/{week}", params=params, ttl_seconds=ttl_seconds)


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
