"""Configuration loading.

Everything the bot needs to know about *which* league it is advising lives in
config.toml, so the code carries no hardcoded league identity.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
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
class HealthConfig:
    """Rest-of-season multipliers per injury regime."""

    healthy: float = 1.00
    playing_diminished_structural: float = 0.90
    playing_diminished_soft: float = 1.00
    out_short: float = 0.40
    out_long: float = 0.00

    def as_table(self) -> dict[str, float]:
        return {
            "healthy": self.healthy,
            "playing_diminished_structural": self.playing_diminished_structural,
            "playing_diminished_soft": self.playing_diminished_soft,
            "out_short": self.out_short,
            "out_long": self.out_long,
        }


@dataclass(frozen=True)
class Config:
    league: LeagueConfig
    paths: PathsConfig
    sources: SourcesConfig
    model: ModelConfig
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
        health=HealthConfig(**raw.get("health", {})),
        root=root,
    )
