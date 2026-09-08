"""The dashboard must not change when only the clock has.

The hourly job commits `docs/index.html` when it differs from the last commit,
ignoring the "Updated ..." line. That guard is only meaningful if nothing else in
the page moves on its own: the first live run committed because a countdown cell
had drifted from "5d 5h" to "5d 4h" on unchanged data, which would have produced
a commit an hour forever and buried real changes.

Every relative time is now rendered by the browser from a `data-kickoff`
attribute, so the served HTML carries absolute times only.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from fantasylineup.engine.lineup import Lineup
from fantasylineup.engine.locks import LockState
from fantasylineup.report.advisory import Advisory
from fantasylineup.report.render_html import render_dashboard
from tests.test_lineup import p

SLOTS = ["QB", "RB", "WR", "FLEX"]
KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)


def _advisory(generated_at: datetime) -> Advisory:
    players = [
        p("QB1", "QB", 18.0),
        p("RB1", "RB", 21.4),
        p("WR1", "WR", 14.9),
        p("RB2", "RB", 12.5),
    ]
    lineup = Lineup(slots=SLOTS, assignments=dict(enumerate(players)), bench=[])
    return Advisory(
        week=1,
        slots=SLOTS,
        optimal=lineup,
        current=lineup,
        changes=[],
        generated_at=generated_at,
        lock_states={
            x.sleeper_id: LockState(False, KICKOFF, opponent="PIT") for x in players
        },
        deadline=KICKOFF,
        opponent_name="jakered47",
    )


def _without_timestamp(html: str) -> str:
    return re.sub(r'<div class="sub">.*?</div>', "", html, flags=re.S)


def test_only_the_timestamp_moves_between_runs():
    """Four hours apart, on identical inputs, nothing else may differ."""
    first = render_dashboard(_advisory(datetime(2026, 9, 8, 15, 7, tzinfo=UTC)), "ZOO")
    later = render_dashboard(
        _advisory(datetime(2026, 9, 8, 15, 7, tzinfo=UTC) + timedelta(hours=4)), "ZOO"
    )

    assert first != later, "the timestamp itself should still update"
    assert _without_timestamp(first) == _without_timestamp(later)


def test_no_server_rendered_countdown_survives_in_the_page():
    """A relative time in the served HTML is stale the moment it is written."""
    html = render_dashboard(_advisory(datetime(2026, 9, 8, 15, 7, tzinfo=UTC)), "ZOO")
    body = _without_timestamp(html)
    # "5d 1h" / "3h 20m" -- the shapes format_countdown emits.
    assert not re.search(r">\s*\d+d \d+h\s*<", body)
    assert not re.search(r">\s*\d+h \d+m\s*<", body)


def test_the_countdown_prefix_is_not_inside_the_element_the_script_overwrites():
    """The browser replaces the whole textContent, so a prefix inside it is lost."""
    html = render_dashboard(_advisory(datetime(2026, 9, 8, 15, 7, tzinfo=UTC)), "ZOO")
    for span in re.findall(r"<span class=\"when\"[^>]*>(.*?)</span>", html, flags=re.S):
        assert "Next lock" not in span
