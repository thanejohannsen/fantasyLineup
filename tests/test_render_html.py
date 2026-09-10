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
from fantasylineup.engine.trades import TradeProposal, TradeRationale
from fantasylineup.engine.waivers import WaiverTarget
from fantasylineup.report.render_html import (
    TeamView,
    render_dashboard,
    render_moves_panel,
    render_team_body,
)
from tests.test_lineup import p

SLOTS = ["QB", "RB", "WR", "FLEX"]
KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)
_NOW = datetime(2026, 9, 8, 15, 7, tzinfo=UTC)


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
    return re.sub(r'<div class="sub stamp">.*?</div>', "", html, flags=re.S)


def _moves_html(roster_id: int) -> str:
    """A real moves panel, so the ids it emits are actually in the page.

    Building the sections without one would make the id-collision test pass
    vacuously -- there would be no `pitch` elements to collide.
    """
    proposal = TradeProposal(
        partner_roster_id=roster_id + 100,
        partner_name="Some Team",
        give=(p("Mine", "RB", 200.0),),
        get=(p("Theirs", "WR", 220.0),),
        my_gain=12.0,
        their_gain=4.0,
    )
    return render_moves_panel(
        [WaiverTarget(player=p("FA", "TE", 90.0), drop=None, gain=5.0, rostered_gain=5.0)],
        [proposal],
        {0: TradeRationale(why="because", their_angle="they need it", pitch="Mine for Theirs?")},
        None,
        None,
        prefix=f"-{roster_id}-",
    )


def _page(generated_at: datetime, teams: int = 1) -> str:
    """A page with `teams` sections, each carrying a full moves panel."""
    views = [
        TeamView(
            roster_id=rid,
            name=f"Team {rid}",
            advisory=_advisory(generated_at),
            moves_html=_moves_html(rid),
        )
        for rid in range(1, teams + 1)
    ]
    return render_dashboard("ZOO", views, default_roster_id=1)


def test_only_the_timestamp_moves_between_runs():
    """Four hours apart, on identical inputs, nothing else may differ."""
    first = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC))
    later = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC) + timedelta(hours=4))

    assert first != later, "the timestamp itself should still update"
    assert _without_timestamp(first) == _without_timestamp(later)


def test_no_server_rendered_countdown_survives_in_the_page():
    """A relative time in the served HTML is stale the moment it is written."""
    html = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC))
    body = _without_timestamp(html)
    # "5d 1h" / "3h 20m" -- the shapes format_countdown emits.
    assert not re.search(r">\s*\d+d \d+h\s*<", body)
    assert not re.search(r">\s*\d+h \d+m\s*<", body)


def test_the_countdown_prefix_is_not_inside_the_element_the_script_overwrites():
    """The browser replaces the whole textContent, so a prefix inside it is lost."""
    html = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC))
    for span in re.findall(r"<span class=\"when\"[^>]*>(.*?)</span>", html, flags=re.S):
        assert "Next lock" not in span


# ------------------------------------------------------- the team selector


def test_every_element_id_is_unique_across_the_whole_document():
    """The trap that makes twelve sections in one page different from one page.

    `render_moves_panel` numbers its pitch elements from zero, and the count
    restarts in every team's section. Duplicate ids are not a validation nicety
    here: `getElementById` returns the first match, so every Copy button on the
    page would silently copy the first team's message.
    """
    html = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC), teams=12)
    ids = re.findall(r'\bid="([^"]+)"', html)
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"ids repeated across sections: {sorted(duplicates)}"


def test_exactly_one_team_is_visible_and_it_is_the_default():
    html = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC), teams=12)
    sections = re.findall(r'<section class="team" data-team="(\d+)"([^>]*)>', html)

    assert len(sections) == 12
    visible = [rid for rid, attrs in sections if "hidden" not in attrs]
    assert visible == ["1"], "the default team, and only the default team, shows"


def test_the_selector_offers_every_team_and_preselects_the_default():
    html = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC), teams=12)
    options = re.findall(r'<option value="(\d+)"([^>]*)>', html)

    assert [rid for rid, _ in options] == [str(n) for n in range(1, 13)]
    selected = [rid for rid, attrs in options if "selected" in attrs]
    assert selected == ["1"]


def test_a_single_team_page_has_no_selector():
    """`fl advise --publish` writes one team; a dropdown of one is noise."""
    assert "<select" not in _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC), teams=1)


def test_twelve_teams_still_only_move_on_the_timestamp():
    """The commit guard has to survive the page getting twelve times bigger."""
    first = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC), teams=12)
    later = _page(datetime(2026, 9, 8, 15, 7, tzinfo=UTC) + timedelta(hours=4), teams=12)

    assert first != later
    assert _without_timestamp(first) == _without_timestamp(later)


def test_an_unknown_default_falls_back_rather_than_hiding_everything():
    """A roster id that is not on the page must not blank it."""
    views = [
        TeamView(
            roster_id=rid,
            name=f"Team {rid}",
            advisory=_advisory(_NOW),
            moves_html=_moves_html(rid),
        )
        for rid in (3, 7)
    ]
    html = render_dashboard("ZOO", views, default_roster_id=99)
    sections = re.findall(r'<section class="team" data-team="(\d+)"([^>]*)>', html)
    assert [rid for rid, attrs in sections if "hidden" not in attrs] == ["3"]


# ------------------------------------------------- current vs recommended


def _lineups(current_ids: list[str], optimal_ids: list[str], slots: list[str]):
    """Two lineups over the same slots, from player ids."""
    pool = {
        "QB1": p("QB1", "QB", 18.0),
        "RB1": p("RB1", "RB", 21.4),
        "RB2": p("RB2", "RB", 15.3),
        "WR1": p("WR1", "WR", 14.9),
        "BENCH": p("BENCH", "RB", 8.7),
    }
    build = lambda ids: Lineup(  # noqa: E731
        slots=slots,
        assignments={i: pool[x] for i, x in enumerate(ids)},
        bench=[v for k, v in pool.items() if k not in ids],
    )
    return build(current_ids), build(optimal_ids), pool


def _advisory_from(current_ids, optimal_ids, slots):
    current, optimal, pool = _lineups(current_ids, optimal_ids, slots)
    return Advisory(
        week=1,
        slots=slots,
        optimal=optimal,
        current=current,
        changes=[],
        generated_at=_NOW,
        lock_states={x.sleeper_id: LockState(False, KICKOFF) for x in pool.values()},
        deadline=KICKOFF,
        opponent_name="them",
    )


def _changed_rows(html_text: str) -> int:
    return len(re.findall(r'<tr class="[^"]*changed', html_text))


def test_shuffling_the_same_starters_between_equal_slots_is_not_a_change():
    """The bug this caught: two RB slots reporting two changes and +0.0 points.

    Which back the optimiser puts in RB1 versus RB2 is arbitrary. Compared index
    by index, a lineup where nobody moved reads as two changes, which trains the
    reader to ignore the highlight.
    """
    slots = ["QB", "RB", "RB", "WR"]
    advisory = _advisory_from(
        ["QB1", "RB1", "RB2", "WR1"], ["QB1", "RB2", "RB1", "WR1"], slots
    )
    body = render_team_body(TeamView(1, "T", advisory))

    assert _changed_rows(body) == 0
    assert "already matches" in body


def test_a_player_entering_the_lineup_is_a_change():
    slots = ["QB", "RB", "RB", "WR"]
    advisory = _advisory_from(
        ["QB1", "RB1", "RB2", "WR1"], ["QB1", "RB1", "BENCH", "WR1"], slots
    )
    body = render_team_body(TeamView(1, "T", advisory))

    assert _changed_rows(body) == 1
    assert "1 change from" in body


def test_both_lineups_are_shown_side_by_side():
    slots = ["QB", "RB", "RB", "WR"]
    advisory = _advisory_from(
        ["QB1", "RB1", "RB2", "WR1"], ["QB1", "RB1", "BENCH", "WR1"], slots
    )
    body = render_team_body(TeamView(1, "T", advisory))

    assert "<th>Current</th>" in body and "<th>Recommended</th>" in body
    # The player being benched stays visible -- knowing who you move off is
    # half the decision.
    assert "RB2" in body and "BENCH" in body


def test_the_bench_is_listed():
    slots = ["QB", "RB", "RB", "WR"]
    advisory = _advisory_from(
        ["QB1", "RB1", "RB2", "WR1"], ["QB1", "RB1", "RB2", "WR1"], slots
    )
    body = render_team_body(TeamView(1, "T", advisory))
    assert "<h2>Bench</h2>" in body and "BENCH" in body


def test_a_market_moved_projection_is_marked():
    """Sleeper's own number and one Kalshi moved must not look identical."""
    import dataclasses

    slots = ["QB"]
    advisory = _advisory_from(["QB1"], ["QB1"], slots)
    plain = render_team_body(TeamView(1, "T", advisory))
    assert 'class="mkt"' not in plain.split('<p class="note">')[0]

    moved = dataclasses.replace(
        advisory.optimal.assignments[0], market_shift=1.4, market_coverage=0.8
    )
    advisory.optimal.assignments[0] = moved
    advisory.current.assignments[0] = moved
    body = render_team_body(TeamView(1, "T", advisory))
    assert "K+1.4" in body


def test_trade_caveats_reach_the_page():
    """A caveat computed and not rendered is worse than one never computed.

    This shipped exactly that way once: the bye-clash text was built into a
    local variable that no template interpolated, so the engine knew and the
    reader did not.
    """
    from fantasylineup.engine.trades import TradeRationale

    panel = render_moves_panel(
        [],
        [
            TradeProposal(
                partner_roster_id=3,
                partner_name="Them",
                give=(p("Mine", "RB", 200.0),),
                get=(p("Theirs", "WR", 220.0),),
                my_gain=12.0,
                their_gain=4.0,
            )
        ],
        {
            0: TradeRationale(
                why="because",
                their_angle="they need it",
                pitch="Mine for Theirs?",
                caveats=("Theirs and Other share a week 7 bye",),
            )
        },
    )
    assert "week 7 bye" in panel
