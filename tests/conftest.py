"""Shared fixtures.

Everything here is real captured API output rather than invented data, so the
tests fail if Sleeper's undocumented endpoints change shape. That is the point:
those endpoints carry no compatibility promise, and a silent shape change would
otherwise surface as quietly wrong advice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="session")
def league() -> dict:
    return _load("league.json")


@pytest.fixture(scope="session")
def scoring_settings(league: dict) -> dict:
    return league["scoring_settings"]


@pytest.fixture(scope="session")
def roster_positions(league: dict) -> list[str]:
    return league["roster_positions"]


@pytest.fixture(scope="session")
def rosters() -> list[dict]:
    return _load("rosters.json")


@pytest.fixture(scope="session")
def rb_projections() -> list[dict]:
    return _load("projections_rb_2026_w1.json")


@pytest.fixture(scope="session")
def players_slice() -> dict:
    return _load("players_slice.json")
