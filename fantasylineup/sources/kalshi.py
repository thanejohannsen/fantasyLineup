"""Kalshi market data.

Public market data needs no authentication. Only the series listed in
``PLAYER_SERIES`` and ``GAME_SERIES`` are fetched, because those are the ones
that actually carry open markets: ``KXNFLANYTD`` and ``KXNFLINT`` exist as
series but were empty when surveyed, and ``KXNFLRUSHYDS`` does not exist at all.

That last absence shapes what the market can and cannot contribute. Rushing
yards are the dominant scoring component for a running back and there is no
market for them, so a back's projection leans almost entirely on Sleeper while a
receiver's can be informed on all three of his components. The report has to
show that difference rather than implying uniform confidence.

Players are joined by ``custom_strike.football_player``, a UUID that is exactly
Sleeper's ``kalshi_id`` -- verified identical for the same player in both
sources. There is no name matching anywhere in this project.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

log = logging.getLogger(__name__)

# Series -> the Sleeper stat key its ladders describe.
#
# KXNFLTD is touchdowns *scored*, per its own rules text ("If <player> scores at
# least 1+ touchdowns"), so it maps to rushing plus receiving touchdowns and
# emphatically not to passing touchdowns. Treating it as pass_td would badly
# overrate quarterbacks.
PLAYER_SERIES: dict[str, str] = {
    "KXNFLPASSYDS": "pass_yd",
    "KXNFLRECYDS": "rec_yd",
    "KXNFLREC": "rec",
    "KXNFLTD": "scored_td",
}

# Game-level context: win probability, totals and spreads.
GAME_SERIES = ("KXNFLGAME", "KXNFLTOTAL", "KXNFLSPREAD")

# Stats whose ladders can be evaluated exactly by the survival identity, when
# the ladder reaches "1 or more". Everything else is fitted as a lognormal:
# every quantity here is non-negative and right-skewed, and a normal
# systematically understates low-volume players. Measured against Sleeper on a
# live slate, switching to lognormal moved the median agreement from 0.75 to
# 0.96 for receiving yards and 0.85 to 0.95 for receptions.
COUNT_STATS = frozenset({"scored_td", "rec"})


@dataclass(frozen=True)
class MarketQuote:
    ticker: str
    event_ticker: str
    series: str
    kalshi_id: str | None
    stat: str
    strike: float
    bid: float
    ask: float
    title: str
    close_time: str | None


class KalshiClient:
    def __init__(
        self, base_url: str, timeout: float = 45.0, pause_seconds: float = 0.35
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Kalshi returns 429 when four 1000-market series are pulled
        # back-to-back. A short pause between pages is cheaper than the retry,
        # and a full sweep is only a handful of requests either way.
        self.pause_seconds = pause_seconds
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "fantasylineup/0.1 (+personal fantasy advisor)"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "KalshiClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def markets(self, series_ticker: str, status: str = "open") -> Iterator[dict[str, Any]]:
        """All markets in a series, following the cursor to the end."""
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 1000, "series_ticker": series_ticker}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            resp = self._client.get(f"{self.base_url}/markets", params=params)
            if resp.status_code == 429:
                # One polite retry; beyond that the caller degrades to
                # Sleeper-only rather than hammering.
                log.info("Kalshi rate limited on %s, backing off", series_ticker)
                time.sleep(2.0)
                resp = self._client.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            payload = resp.json()
            batch = payload.get("markets") or []
            yield from batch
            cursor = payload.get("cursor")
            # Kalshi returns the same cursor with an empty page at the end.
            if not cursor or not batch:
                return
            if self.pause_seconds:
                time.sleep(self.pause_seconds)

    def player_quotes(self) -> list[MarketQuote]:
        out: list[MarketQuote] = []
        for series, stat in PLAYER_SERIES.items():
            try:
                markets = list(self.markets(series))
            except httpx.HTTPError as exc:
                log.warning("Kalshi series %s unavailable: %s", series, exc)
                continue
            if self.pause_seconds:
                time.sleep(self.pause_seconds)
            for m in markets:
                quote = _to_quote(m, series, stat)
                if quote is not None:
                    out.append(quote)
            log.info("Kalshi %s: %d markets", series, len(markets))
        return out


def _to_quote(market: dict[str, Any], series: str, stat: str) -> MarketQuote | None:
    strike = market.get("floor_strike")
    if strike is None or market.get("strike_type") != "greater":
        return None
    custom = market.get("custom_strike") or {}
    try:
        bid = float(market.get("yes_bid_dollars") or 0.0)
        ask = float(market.get("yes_ask_dollars") or 0.0)
    except (TypeError, ValueError):
        return None
    return MarketQuote(
        ticker=market["ticker"],
        event_ticker=market.get("event_ticker", ""),
        series=series,
        kalshi_id=custom.get("football_player"),
        stat=stat,
        strike=float(strike),
        bid=bid,
        ask=ask,
        title=market.get("title", ""),
        close_time=market.get("close_time"),
    )
