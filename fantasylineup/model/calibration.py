"""Measuring how wrong the model was, and refitting it.

What can honestly be learned, and what cannot
---------------------------------------------
Ten starters a week is far too small a sample to learn anything from. But the
model projects the *whole NFL*, so every completed week yields on the order of
two thousand projection-and-outcome pairs. That supports fitting a handful of
parameters -- per-position spread, per-position bias, an injury haircut -- and
nothing more. Per-player predictive models from one season of a twelve-team
league would be pure overfitting, and are deliberately not attempted.

The Kalshi question cannot be answered yet
------------------------------------------
Whether blending market prices actually beats Sleeper alone is the interesting
question, and it is **not backtestable**. Kalshi retains no usable pre-kickoff
history: settled markets return an empty book (bid 0.00 against ask 1.00), the
candlesticks endpoint returns nothing for them, and every settled NFL
passing-yards market in existence is from 2026 -- the series did not exist during
the 2025 season.

So the blend can only be evaluated *prospectively*, by recording what was
projected before each slate and grading it afterwards. That is what
``calibrate`` accumulates, and it needs several weeks of live data before it says
anything. Until then the capped, Sleeper-anchored blend is a deliberately
conservative default rather than a measured improvement.

Grading uses the projection stored at decision time, never one recomputed later.
Recomputing would leak hindsight and make the measured accuracy fiction.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

log = logging.getLogger(__name__)

# Below this many observations a position's fit is not worth trusting; fall back
# to the shipped defaults rather than chase noise.
MIN_OBSERVATIONS = 200


@dataclass(frozen=True)
class Observation:
    position: str
    projected: float
    actual: float
    source: str

    @property
    def error(self) -> float:
        return self.actual - self.projected


@dataclass
class SourceScore:
    source: str
    n: int
    mae: float
    bias: float
    rmse: float

    def describe(self) -> str:
        return (
            f"{self.source:<10} n={self.n:<6} MAE {self.mae:6.2f}  "
            f"bias {self.bias:+6.2f}  RMSE {self.rmse:6.2f}"
        )


def score_source(observations: list[Observation]) -> SourceScore:
    if not observations:
        return SourceScore("(none)", 0, float("nan"), float("nan"), float("nan"))
    errors = np.array([o.error for o in observations])
    return SourceScore(
        source=observations[0].source,
        n=len(errors),
        mae=float(np.mean(np.abs(errors))),
        bias=float(np.mean(errors)),
        rmse=float(np.sqrt(np.mean(errors**2))),
    )


def fit_variance(observations: list[Observation]) -> dict[str, tuple[float, float]]:
    """Refit ``sd = intercept + slope * projection`` per position.

    Fitted on binned standard deviations rather than raw residuals, because the
    quantity being modelled is the spread itself and binning gives the regression
    something stable to sit on.
    """
    by_position: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        if o.projected >= 1.0:
            by_position[o.position].append(o)

    fitted: dict[str, tuple[float, float]] = {}
    for position, group in by_position.items():
        if len(group) < MIN_OBSERVATIONS:
            log.info("Skipping %s: only %d observations", position, len(group))
            continue
        group.sort(key=lambda o: o.projected)
        bins = min(8, max(3, len(group) // 60))
        xs, ys = [], []
        for i in range(bins):
            chunk = group[i * len(group) // bins : (i + 1) * len(group) // bins]
            if len(chunk) < 10:
                continue
            xs.append(float(np.mean([o.projected for o in chunk])))
            ys.append(float(np.std([o.error for o in chunk])))
        if len(xs) < 3:
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        fitted[position] = (float(intercept), float(slope))
    return fitted


def load_observations(
    conn: sqlite3.Connection, season: int, weeks: list[int], source: str = "sleeper"
) -> list[Observation]:
    """Join stored projections to actuals, using the earliest stored projection.

    The *earliest* ``as_of`` for a week is the one that best represents what was
    known when the lineup decision was made. Taking the latest would grade a
    projection refreshed after news broke, flattering the model.
    """
    if not weeks:
        return []
    placeholders = ",".join("?" * len(weeks))
    rows = conn.execute(
        f"""
        SELECT p.sleeper_id, p.mean AS projected, a.points AS actual, pl.position
        FROM projections p
        JOIN (SELECT season, week, sleeper_id, MIN(as_of) AS first_seen
              FROM projections
              WHERE source = ? AND season = ? AND week IN ({placeholders})
              GROUP BY season, week, sleeper_id) f
          ON f.season = p.season AND f.week = p.week
         AND f.sleeper_id = p.sleeper_id AND f.first_seen = p.as_of
        JOIN actuals a
          ON a.season = p.season AND a.week = p.week AND a.sleeper_id = p.sleeper_id
        JOIN players pl ON pl.sleeper_id = p.sleeper_id
        WHERE p.source = ? AND p.season = ? AND p.week IN ({placeholders})
        """,
        [source, season, *weeks, source, season, *weeks],
    )
    return [
        Observation(
            position=r["position"] or "UNK",
            projected=float(r["projected"] or 0.0),
            actual=float(r["actual"] or 0.0),
            source=source,
        )
        for r in rows
    ]


def save_params(
    conn: sqlite3.Connection,
    variance: dict[str, tuple[float, float]],
    scores: dict[str, SourceScore],
    through_week: int,
    notes: str = "",
) -> int:
    """Version the fit, so any report can name what produced it."""
    payload = {
        "variance": {k: list(v) for k, v in variance.items()},
        "scores": {k: vars(v) for k, v in scores.items()},
    }
    cur = conn.execute(
        """INSERT INTO model_params (fitted_at, through_week, params, notes)
           VALUES (?,?,?,?)""",
        (
            datetime.now(UTC).isoformat(timespec="seconds"),
            through_week,
            json.dumps(payload),
            notes,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def latest_params(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT version, params FROM model_params ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["params"])
    payload["version"] = row["version"]
    return payload
