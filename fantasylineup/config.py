"""Configuration loading.

Everything the bot needs to know about *which* league it is advising lives in
config.toml, so the code carries no hardcoded league identity.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field as dc_field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass(frozen=True)
class LeagueConfig:
    league_id: str
    roster_id: int
    season: int
    previous_league_id: str | None = None


@dataclass(frozen=True)
class PathsConfig:
    db: Path
    cache: Path
    reports: Path
    site: Path

    def ensure(self) -> None:
        self.db.parent.mkdir(parents=True, exist_ok=True)
        for d in (self.cache, self.reports, self.site):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SourcesConfig:
    sleeper_base: str
    kalshi_base: str
    max_calls_per_min: int
    players_dump_ttl_hours: int
    status_ttl_minutes: int


@dataclass(frozen=True)
class ModelConfig:
    kalshi_max_shift: float
    sim_draws: int


@dataclass(frozen=True)
class TradesConfig:
    """How hard a proposed gain must work before it is shown."""

    jitter: float = 0.03
    draws: int = 200
    require_stable: bool = True
    min_games_for_contingency: int = 3


@dataclass(frozen=True)
class HealthConfig:
    """Injury multipliers: rest-of-season regimes, and weekly availability."""

    healthy: float = 1.00
    playing_diminished_structural: float = 0.90
    playing_diminished_soft: float = 1.00
    out_short: float = 0.40
    out_long: float = 0.00
    # Chance a player carrying each designation actually plays this week.
    weekly_out: float = 0.00
    weekly_doubtful: float = 0.10
    weekly_questionable: float = 0.80
    # Player name -> first week he is back. Hand-entered from news, because no
    # feed carries a return date.
    returns: dict[str, int] = dc_field(default_factory=dict)

    def return_weeks(self) -> dict[str, int]:
        """Return weeks keyed by lowercased name, for a forgiving lookup."""
        return {str(k).strip().lower(): int(v) for k, v in self.returns.items()}

    def as_table(self) -> dict[str, float]:
        return {
            "healthy": self.healthy,
            "playing_diminished_structural": self.playing_diminished_structural,
            "playing_diminished_soft": self.playing_diminished_soft,
            "out_short": self.out_short,
            "out_long": self.out_long,
        }

    def availability(self) -> dict[str, float]:
        return {
            "out": self.weekly_out,
            "doubtful": self.weekly_doubtful,
            "questionable": self.weekly_questionable,
            "healthy": 1.00,
        }


@dataclass(frozen=True)
class Config:
    league: LeagueConfig
    paths: PathsConfig
    sources: SourcesConfig
    model: ModelConfig
    trades: TradesConfig
    health: HealthConfig
    root: Path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No config at {path}. Copy config.toml from the repo root and set your league_id."
        )
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    root = path.resolve().parent
    p = raw["paths"]
    paths = PathsConfig(
        db=root / p["db"],
        cache=root / p["cache"],
        reports=root / p["reports"],
        site=root / p["site"],
    )
    return Config(
        league=LeagueConfig(**raw["league"]),
        paths=paths,
        sources=SourcesConfig(**raw["sources"]),
        model=ModelConfig(**raw["model"]),
        trades=TradesConfig(**raw.get("trades", {})),
        health=HealthConfig(**raw.get("health", {})),
        root=root,
    )
