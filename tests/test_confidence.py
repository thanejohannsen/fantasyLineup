"""Confidence tests.

The failure these guard against is a number that looks settled and is not. A
live proposal was quoted at +23.9, then +8.3, then -7.8 across successive
refreshes of the same sources, with no new information -- the gain is a
difference of two optimisations, so a small projection move can flip a slot
assignment and jump the result.
"""

from __future__ import annotations

import os
import sys

import pytest

from fantasylineup.engine.confidence import (
    assess,
    find_contingencies,
    measure_stability,
)
from fantasylineup.engine.trades import TradeProposal
from tests.test_lineup import p

SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]


def _injured(pid, pos, pts, status="Questionable", part="Achilles", notes="Surgery"):
    from fantasylineup.engine.lineup import PlayerProjection

    return PlayerProjection(
        sleeper_id=pid,
        name=pid,
        position=pos,
        points=pts,
        fantasy_positions=frozenset({pos}),
        injury_status=status,
        injury_body_part=part,
        injury_notes=notes,
        health_regime="playing_diminished",
    )


# A roster where a 2-for-1 fills its lineup hole by promoting a bench player.
def _roster(bench_te_points: float = 152.0, injured_bench: bool = False):
    bench_te = (
        _injured("BenchTE", "TE", bench_te_points)
        if injured_bench
        else p("BenchTE", "TE", bench_te_points)
    )
    return [
        p("Star", "RB", 309.0),
        p("QB1", "QB", 275.0),
        p("TE1", "TE", 235.0),
        p("Send1", "RB", 206.0),
        p("Send2", "WR", 199.0),
        p("WR2", "WR", 161.0),
        p("RB3", "RB", 161.0),
        p("WR3", "WR", 155.0),
        bench_te,
        p("Spare", "RB", 128.0),
    ]


def _two_for_one(roster):
    give = tuple(x for x in roster if x.sleeper_id in {"Send1", "Send2"})
    get = (p("Upgrade", "RB", 261.0),)
    return TradeProposal(7, "them", give, get, 8.3, 1.0)


# ------------------------------------------------------------- contingency


def test_gain_resting_on_one_promoted_player_is_flagged():
    """A 2-for-1's advantage can belong to the bench player who backfills.

    Two starters leave and one arrives, so the lineup hole is filled by a
    promotion. When removing that promoted player takes the gain below the
    threshold, the trade is not really about the players being exchanged.
    """
    roster = _roster()
    contingencies = find_contingencies(_two_for_one(roster), roster, SLOTS)

    assert contingencies, "the promoted bench player should be identified"
    names = {c.player.sleeper_id for c in contingencies}
    assert "BenchTE" in names
    assert all(c.gain_without < 3.0 for c in contingencies)


def test_a_trade_that_stands_on_its_own_has_no_contingency():
    """A straight upgrade promotes nobody, so nothing is contingent."""
    roster = _roster()
    swap_out = next(x for x in roster if x.sleeper_id == "Send1")
    proposal = TradeProposal(7, "them", (swap_out,), (p("Better", "RB", 290.0),), 84.0, 2.0)
    assert find_contingencies(proposal, roster, SLOTS) == ()


def test_contingency_gain_without_reflects_the_real_alternative():
    """Zeroing the player lets the optimiser backfill with who is actually next.

    Assuming a replacement value instead would hide the cliff: once the promoted
    player drops below the next bench option, further decline costs nothing.
    """
    roster = _roster()
    contingencies = find_contingencies(_two_for_one(roster), roster, SLOTS)
    # The spare back (128) is the real fallback, so the loss is bounded by the
    # gap to him rather than by the promoted player's whole projection.
    assert contingencies[0].gain_without > -200.0


# ------------------------------------------------------------ evidence gate


def test_leaning_on_an_unproven_injured_player_blocks_the_offer():
    """Zero games of evidence is not a basis for relying on a projection."""
    roster = _roster(injured_bench=True)
    confidence = assess(
        _two_for_one(roster), roster, SLOTS, games_played={}, draws=60
    )
    assert confidence.blocking, "an injured contingency with no games must block"
    assert not confidence.is_offerable
    assert {c.player.sleeper_id for c in confidence.blocking} == {"BenchTE"}
    # This trade happens to fail the stability check too, and describe() reports
    # that first; the evidence wording is exercised on its own below.


def test_evidence_unblocks_once_he_has_played():
    """Same trade, same player, once there is something to go on."""
    roster = _roster(injured_bench=True)
    proposal = _two_for_one(roster)
    blocked = assess(proposal, roster, SLOTS, games_played={}, draws=60)
    proven = assess(
        proposal, roster, SLOTS, games_played={"BenchTE": 5}, draws=60
    )
    assert blocked.blocking and not proven.blocking


def test_a_healthy_promoted_player_does_not_trip_the_evidence_gate():
    """The gate is about injury uncertainty, not about bench players generally."""
    roster = _roster(injured_bench=False)
    confidence = assess(_two_for_one(roster), roster, SLOTS, games_played={}, draws=60)
    assert confidence.contingencies, "still contingent"
    assert not confidence.blocking, "but not on an injury we cannot judge"


# --------------------------------------------------------------- stability


def test_a_thin_gain_does_not_survive_projection_error():
    """The case that started this: a gain inside the noise of its own inputs."""
    roster = _roster()
    p10, p90 = measure_stability(_two_for_one(roster), roster, SLOTS, draws=300)
    assert p10 < 0 < p90, "a marginal 2-for-1 should straddle zero"


def test_a_real_upgrade_survives_projection_error():
    roster = _roster()
    swap_out = next(x for x in roster if x.sleeper_id == "Send1")
    proposal = TradeProposal(7, "them", (swap_out,), (p("Better", "RB", 290.0),), 84.0, 2.0)
    p10, _ = measure_stability(proposal, roster, SLOTS, draws=300)
    assert p10 > 0


def test_stability_is_deterministic():
    """Advice that flickers between refreshes is worse than advice that is uncertain."""
    roster = _roster()
    proposal = _two_for_one(roster)
    first = measure_stability(proposal, roster, SLOTS, draws=120)
    second = measure_stability(proposal, roster, SLOTS, draws=120)
    assert first == second


# The in-process check above passed while the property it names was false. Seeding
# the generator fixes the sequence of draws, not which player each draw lands on,
# and that mapping came from iterating a set of string ids -- an order CPython
# salts per process. Identical inputs produced a p90 four points apart across five
# runs, so the hourly dashboard quoted a different range every hour from unchanged
# data. Determinism that only holds inside one process is not determinism: the
# refreshes being compared are separate processes.
_DETERMINISM_PROBE = """
import sys
sys.path.insert(0, {root!r})
from fantasylineup.engine.confidence import measure_stability
from fantasylineup.engine.trades import TradeProposal
from tests.test_confidence import SLOTS, _roster, _two_for_one

roster = _roster()
print("%.6f %.6f" % measure_stability(_two_for_one(roster), roster, SLOTS, draws=120))
"""


def test_stability_is_deterministic_across_processes():
    import pathlib
    import subprocess

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    script = _DETERMINISM_PROBE.format(root=root)

    results = set()
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        results.add(out.stdout.strip())

    assert len(results) == 1, f"range moved between processes: {results}"


def test_percentiles_are_ordered_and_bracket_nothing_absurd():
    roster = _roster()
    p10, p90 = measure_stability(_two_for_one(roster), roster, SLOTS, draws=200)
    assert p10 <= p90
    assert -500 < p10 and p90 < 500


def test_require_stable_can_be_disabled():
    """The threshold is a policy choice, so it lives in config."""
    roster = _roster()
    proposal = _two_for_one(roster)
    strict = assess(proposal, roster, SLOTS, draws=60, require_stable=True)
    lenient = assess(proposal, roster, SLOTS, draws=60, require_stable=False)
    assert not strict.is_stable
    assert lenient.is_stable


def test_describe_names_the_evidence_problem_when_that_is_the_only_one():
    """With stability waived, the remaining objection must be stated plainly."""
    roster = _roster(injured_bench=True)
    confidence = assess(
        _two_for_one(roster),
        roster,
        SLOTS,
        games_played={},
        draws=60,
        require_stable=False,
    )
    assert confidence.is_stable  # waived
    assert "too little evidence" in confidence.describe()
    assert "BenchTE" in confidence.describe()


def test_describe_reports_instability_first():
    """Stability is the more fundamental objection, so it leads."""
    roster = _roster(injured_bench=True)
    text = assess(_two_for_one(roster), roster, SLOTS, games_played={}, draws=60).describe()
    assert "unstable" in text


def test_confidence_reports_a_range_not_a_point():
    roster = _roster()
    confidence = assess(_two_for_one(roster), roster, SLOTS, draws=120)
    assert confidence.spread > 0
    assert confidence.p10 == pytest.approx(confidence.p90 - confidence.spread)
