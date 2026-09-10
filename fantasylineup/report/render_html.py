"""The dashboard page, written to docs/ for GitHub Pages.

Published as a static page rather than an interactive artifact because the
refresh runs unattended on a schedule: the page has to be rewritten by a cron
job with no session behind it. One bookmarked URL, always current.

Everything is inlined -- no external CSS, fonts or scripts -- so the page renders
identically offline and cannot break because a CDN changed.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import UTC, datetime

from ..engine.locks import LockState
from .advisory import Advisory

_UNKNOWN = LockState(locked=False, kickoff_utc=None)

_STYLE = """
:root {
  --bg: #fbfbf9; --panel: #ffffff; --ink: #1a1a18; --muted: #6b6b64;
  --line: #e5e4de; --accent: #1f6f5c; --warn: #a8501e; --good: #1f6f5c;
  --locked: #9a9a92;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14140f; --panel: #1c1c18; --ink: #ececE4; --muted: #9a9a90;
    --line: #2e2e28; --accent: #6cc0a4; --warn: #d99257; --good: #6cc0a4;
    --locked: #6a6a62;
  }
}
:root[data-theme="dark"] {
  --bg: #14140f; --panel: #1c1c18; --ink: #ececE4; --muted: #9a9a90;
  --line: #2e2e28; --accent: #6cc0a4; --warn: #d99257; --good: #6cc0a4;
  --locked: #6a6a62;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--ink); margin: 0;
  font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 2rem 1.25rem 4rem;
}
.wrap { max-width: 760px; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 .2rem; letter-spacing: -.01em; }
.sub { color: var(--muted); font-size: .85rem; margin-bottom: 1.5rem; }
.panel {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.1rem 1.2rem; margin-bottom: 1rem;
}
.panel h2 {
  font-size: .72rem; text-transform: uppercase; letter-spacing: .09em;
  color: var(--muted); margin: 0 0 .85rem; font-weight: 600;
}
.headline { font-size: 1.9rem; font-weight: 650; letter-spacing: -.02em; }
.headline small { font-size: .8rem; font-weight: 400; color: var(--muted); }
.posture { color: var(--accent); font-size: .9rem; margin-top: .3rem; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
td, th { padding: .32rem 0; text-align: left; border-bottom: 1px solid var(--line); }
th { font-size: .68rem; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); }
tr:last-child td { border-bottom: 0; }
.slot { color: var(--muted); font-size: .8rem; }
/* The fixed width is a table-layout concern only; applying it to the inline
   spans used in the trade cards squeezes them into a one-word column. */
td.slot, th.slot { width: 3.4rem; }
.num { text-align: right; font-variant-numeric: tabular-nums; width: 4rem; }
.when { text-align: right; color: var(--muted); font-size: .78rem; width: 5rem; }
.locked { color: var(--locked); }
.change { color: var(--warn); font-weight: 600; }
.empty { color: var(--muted); font-style: italic; }
.note { color: var(--muted); font-size: .82rem; margin-top: .8rem; }
.scroll { overflow-x: auto; }
.trade { border-top: 1px solid var(--line); padding: .8rem 0; }
.trade:first-of-type { border-top: 0; padding-top: 0; }
.tradehead { margin-bottom: .35rem; }
.why { color: var(--muted); font-size: .84rem; margin: .5rem 0 0; }
.pitchwrap { margin-top: .7rem; }
.pitch {
  background: var(--bg); border: 1px solid var(--line); border-radius: 7px;
  padding: .7rem .8rem; font: .82rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap; margin: .4rem 0 0; overflow-x: auto;
}
.copy {
  background: none; border: 1px solid var(--line); color: var(--muted);
  border-radius: 6px; padding: .2rem .55rem; font-size: .74rem; cursor: pointer;
}
.copy:hover { color: var(--ink); border-color: var(--muted); }
.rank { color: var(--muted); font-size: .8rem; font-variant-numeric: tabular-nums; }
.badge {
  background: var(--accent); color: var(--panel); border-radius: 5px;
  padding: .1rem .45rem; font-size: .68rem; letter-spacing: .04em;
  text-transform: uppercase; font-weight: 600;
}
.conflict { color: var(--warn); font-size: .82rem; margin: .45rem 0 0; }
.injury {
  color: var(--warn); font-size: .7rem; letter-spacing: .03em;
  border: 1px solid var(--warn); border-radius: 4px; padding: 0 .28rem;
  white-space: nowrap;
}
.feed { margin: 0; padding-left: 1.1rem; font-size: .86rem; }
.feed li { margin-bottom: .25rem; }
.feed li.mine { color: var(--accent); font-weight: 600; }
/* The UA default would cover this, but the page carries eleven hidden sections
   and a stylesheet quirk that revealed them would be a mess rather than a
   glitch, so say it explicitly. */
.team[hidden] { display: none; }
.teamsel { display: flex; align-items: center; gap: .5rem; margin: 0 0 1rem; }
.teamsel label {
  color: var(--muted); font-size: .78rem; text-transform: uppercase;
  letter-spacing: .06em;
}
.teamsel select {
  flex: 1; max-width: 22rem; padding: .4rem .5rem; font: inherit;
  color: var(--ink); background: var(--panel);
  border: 1px solid var(--line); border-radius: 6px;
}
/* A changed slot is the only thing on the page worth acting on, so it gets the
   accent rail; the player being benched fades rather than disappearing, because
   knowing who you are moving off is half the decision. */
tr.changed td { background: color-mix(in srgb, var(--accent) 7%, transparent); }
tr.changed td:first-child { box-shadow: inset 2px 0 0 var(--accent); }
td.faded { color: var(--muted); text-decoration: line-through; }
.pts { color: var(--muted); font-size: .78rem; font-variant-numeric: tabular-nums; }
.mkt {
  color: var(--accent); font-size: .68rem; letter-spacing: .02em;
  border: 1px solid var(--accent); border-radius: 4px; padding: 0 .26rem;
  white-space: nowrap;
}
"""


# Kept out of the f-string template: JavaScript braces collide with f-string
# interpolation, and escaping every one of them is a needless hazard.
_SCRIPT = """<script>
// Countdowns are recomputed in the browser from absolute kickoff times, so the
// page stays correct between scheduled refreshes. Without this the publisher
// would have to rewrite and commit the page every hour purely to keep a
// relative time honest. Server-rendered text remains the no-JavaScript
// fallback.
(function () {
  function label(ms) {
    if (ms <= 0) return "locked";
    var m = Math.floor(ms / 60000), d = Math.floor(m / 1440), h = Math.floor((m % 1440) / 60);
    if (d) return d + "d " + h + "h";
    if (h) return h + "h " + (m % 60) + "m";
    return m + "m";
  }
  function tick() {
    var now = Date.now();
    document.querySelectorAll("[data-kickoff]").forEach(function (el) {
      var t = Date.parse(el.getAttribute("data-kickoff"));
      if (!isNaN(t)) el.textContent = label(t - now);
    });
  }
  tick();
  setInterval(tick, 30000);

  // Team switching. Every team is already in the document, so this only
  // toggles visibility -- there is nothing to fetch and no state to rebuild.
  var picker = document.getElementById("team");
  if (picker) {
    var KEY = "fantasylineup.team";
    function show(id) {
      var found = false;
      document.querySelectorAll("section.team").forEach(function (el) {
        var mine = el.getAttribute("data-team") === id;
        el.hidden = !mine;
        if (mine) found = true;
      });
      return found;
    }
    // A remembered team can disappear -- someone leaves the league, or the page
    // is opened in a browser carrying a stale id -- so fall back to the server's
    // choice rather than hiding every section.
    try {
      var saved = localStorage.getItem(KEY);
      if (saved && saved !== picker.value && show(saved)) picker.value = saved;
    } catch (e) {}
    picker.addEventListener("change", function () {
      show(picker.value);
      try { localStorage.setItem(KEY, picker.value); } catch (e) {}
      window.scrollTo(0, 0);
    });
  }

  document.querySelectorAll(".copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var el = document.getElementById(btn.getAttribute("data-copy"));
      if (!el) return;
      navigator.clipboard.writeText(el.textContent).then(function () {
        var was = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = was; }, 1500);
      });
    });
  });
})();
</script>"""


def _esc(text: object) -> str:
    return html.escape(str(text))


def _injury_html(player) -> str:
    """Designation badge, in the warning colour, beside a player's name."""
    status = getattr(player, "injury_status", None)
    if not status:
        return ""
    initial = {"Questionable": "Q", "Doubtful": "D", "Out": "OUT"}.get(status, status)
    part = getattr(player, "injury_body_part", None)
    label = f"{initial} - {part}" if part and part != "Undisclosed" else initial
    return f' <span class="injury">{_esc(label)}</span>'


def _market_html(player) -> str:
    """Marker showing the market touched this number, and which way.

    Without it a projection is unattributable: Sleeper's own figure and one the
    Kalshi ladders moved look identical, and they do not deserve equal trust.
    """
    if not getattr(player, "has_market", False):
        return ""
    return (
        f' <span class="mkt" title="Kalshi moved this {player.market_shift:+.1f} points; '
        f'the market prices {player.market_coverage:.0%} of his scoring">'
        f"K{player.market_shift:+.1f}</span>"
    )


def _lock_cell(state) -> str:
    """Kickoff as an absolute time, rewritten to a countdown by the browser.

    A server-rendered "5d 1h" is replaced before anyone reads it, but it changes
    every hour, which made the page differ from the last commit on unchanged
    data. An absolute time is also the more honest fallback: a cached page
    showing "5d 1h" is wrong, one showing the kickoff stays true however stale.
    """
    if state.locked:
        return '<td class="when">locked</td>'
    if state.kickoff_utc is None:
        return '<td class="when">bye</td>'
    return (
        f'<td class="when" data-kickoff="{state.kickoff_utc.isoformat()}">'
        f"{_esc(f'{state.kickoff_utc:%a %H:%M}')}</td>"
    )


def _aligned_slots(advisory):
    """(slot, current, recommended) per slot, with interchangeable slots paired.

    Two RB slots are the same slot, and which back the optimiser puts in which
    index is arbitrary. Compared index by index, a lineup where nothing moved
    reports two changes and a gain of +0.0 -- which is what it did before this
    existed. Players present in both lineups are paired up first so only genuine
    entries and exits are left to differ.
    """
    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(advisory.slots):
        by_label.setdefault(label, []).append(i)

    out: list[tuple[str, object, object]] = [("", None, None)] * len(advisory.slots)
    for label, idxs in by_label.items():
        now = [advisory.current.assignments.get(i) for i in idxs]
        best = [advisory.optimal.assignments.get(i) for i in idxs]
        now_ids = {p.sleeper_id for p in now if p}
        keep = {p.sleeper_id: p for p in best if p and p.sleeper_id in now_ids}
        arriving = iter([p for p in best if p and p.sleeper_id not in now_ids])

        arranged = [keep.get(p.sleeper_id) if p else None for p in now]
        for k, v in enumerate(arranged):
            if v is None:
                arranged[k] = next(arriving, None)
        for i, n, b in zip(idxs, now, arranged):
            out[i] = (label, n, b)
    return out


def _player_cell(player, *, faded: bool = False) -> str:
    if player is None:
        return '<td class="empty">-</td>'
    cls = "faded" if faded else ""
    opp = f' <span class="slot">vs {_esc(player.opponent)}</span>' if player.opponent else ""
    return (
        f'<td class="{cls}">{_esc(player.name)}{_injury_html(player)}{_market_html(player)}'
        f'{opp} <span class="pts">{player.points:.1f}</span></td>'
    )


@dataclass(frozen=True)
class TeamView:
    """One team's report, ready to render into the shared page."""

    roster_id: int
    name: str
    advisory: Advisory
    moves_html: str = ""


def render_team_body(view: TeamView, hidden: bool = False) -> str:
    """One team's panels, as a section the selector can show or hide."""
    a = view.advisory
    moves_html = view.moves_html

    # Current beside recommended, so a change is visible as a difference rather
    # than as a separate list the reader has to reconcile against the table.
    rows = []
    changed_slots = 0
    # Membership, not slot position: a change is a player entering or leaving
    # the lineup. Shuffling the same starters between equivalent slots costs
    # nothing and is not worth acting on.
    now_ids = {p.sleeper_id for p in a.current.assignments.values()}
    best_ids = {p.sleeper_id for p in a.optimal.assignments.values()}
    for slot, now, best in _aligned_slots(a):
        state = a.lock_states.get(
            (best or now).sleeper_id if (best or now) else "", _UNKNOWN
        )
        differs = (best is not None and best.sleeper_id not in now_ids) or (
            now is not None and now.sleeper_id not in best_ids
        )
        changed_slots += differs
        cls = "changed" if differs else ("locked" if state.locked else "")
        rows.append(
            f'<tr class="{cls}"><td class="slot">{_esc(slot)}</td>'
            + _player_cell(now, faded=differs)
            + _player_cell(best)
            + _lock_cell(state)
            + "</tr>"
        )

    if changed_slots:
        plural = "" if changed_slots == 1 else "s"
        summary = (
            f"{changed_slots} change{plural} from the lineup you have set, worth "
            f"{a.gain:+.1f} projected points. Sleeper's API is read-only, so make "
            f"them in the app."
        )
    else:
        summary = "Your lineup already matches the recommendation."

    bench = sorted(a.optimal.bench, key=lambda x: -x.points)
    bench_rows = "".join(
        f'<tr><td class="slot">{_esc(b.position)}</td>'
        + _player_cell(b)
        + _lock_cell(a.lock_states.get(b.sleeper_id, _UNKNOWN))
        + "</tr>"
        for b in bench
    )
    bench_block = (
        f'<div class="panel"><h2>Bench</h2><div class="scroll"><table>'
        f'<tr><th>Pos</th><th>Player</th><th class="when">Kickoff</th></tr>'
        f"{bench_rows}</table></div></div>"
        if bench
        else ""
    )

    if a.outcome is not None:
        wp = a.outcome.win_probability
        banked = ""
        if a.my_banked or a.opponent_banked:
            banked = (
                f'<div class="note">Banked so far: you {a.my_banked:.1f}, '
                f"them {a.opponent_banked:.1f}</div>"
            )
        headline = (
            f'<div class="panel"><h2>Week {a.week} vs {_esc(a.opponent_name)}</h2>'
            f'<div class="headline">{wp:.0%} <small>win probability</small></div>'
            f'<div class="posture">{_esc(a.posture)}</div>'
            f'<div class="note">Projected {a.outcome.mean:.0f} pts '
            f"({a.outcome.p10:.0f}-{a.outcome.p90:.0f}, 10th-90th percentile)</div>{banked}"
            # Both sides of a matchup can read above 50%, and the page now shows
            # both. It is not an inconsistency: each number assumes that team
            # makes its recommended changes while the opponent plays his
            # highest-points lineup, and both teams can gain by switching.
            f'<div class="note">Assumes you make the changes below and they play '
            f"their highest-points lineup.</div></div>"
        )
    else:
        headline = (
            f'<div class="panel"><h2>Week {a.week}</h2>'
            f'<div class="headline">{a.optimal.total_points:.1f} '
            f"<small>projected points</small></div></div>"
        )

    # The next lock belongs to the team being viewed, not to the page, so it
    # lives inside the section rather than in the shared header.
    deadline = ""
    if a.deadline is not None:
        # "Next lock" sits outside the span: the browser overwrites the whole
        # textContent of a [data-kickoff] element, so a prefix placed inside it
        # was being wiped the moment the script ran.
        deadline = (
            f'<div class="sub">Next lock <span class="when"'
            f' data-kickoff="{a.deadline.isoformat()}">'
            f'{_esc(f"{a.deadline:%a %H:%M UTC}")}</span></div>'
        )

    return (
        f'<section class="team" data-team="{view.roster_id}"{" hidden" if hidden else ""}>'
        f"{deadline}{headline}"
        f'<div class="panel"><h2>Lineup</h2><div class="scroll"><table>'
        f'<tr><th>Slot</th><th>Current</th><th>Recommended</th>'
        f'<th class="when">Kickoff</th></tr>'
        f'{"".join(rows)}'
        f"</table></div>"
        f'<p class="note">{summary} <span class="mkt">K</span> marks a projection '
        f"the Kalshi market moved, and by how much; everything else is Sleeper's "
        f"own number.</p></div>"
        f"{bench_block}"
        f"{moves_html}"
        f"</section>"
    )


def _team_selector(teams: list[TeamView], default_roster_id: int) -> str:
    """Dropdown over the league. Omitted when there is only one team to show."""
    if len(teams) < 2:
        return ""
    options = "".join(
        f'<option value="{t.roster_id}"'
        f'{" selected" if t.roster_id == default_roster_id else ""}>'
        f"{_esc(t.name)}</option>"
        for t in teams
    )
    return (
        f'<div class="teamsel"><label for="team">Viewing</label>'
        f'<select id="team">{options}</select></div>'
    )


def render_dashboard(
    league_name: str, teams: list[TeamView], default_roster_id: int | None = None
) -> str:
    """The whole page: one section per team, one of them visible.

    Every team is rendered into the same document rather than into a file each.
    The report is a few kilobytes per team, so twelve of them cost less than one
    image, and switching becomes instant with nothing to fetch.
    """
    if not teams:
        raise ValueError("a dashboard needs at least one team")
    if default_roster_id is None or all(t.roster_id != default_roster_id for t in teams):
        default_roster_id = teams[0].roster_id

    generated = teams[0].advisory.generated_at.strftime("%a %d %b %Y, %H:%M UTC")
    # Only the default team is visible on load. The script may switch to a
    # remembered choice, but the page is already correct without it.
    sections = "".join(
        render_team_body(t, hidden=t.roster_id != default_roster_id) for t in teams
    )

    # A complete document: GitHub Pages serves the file verbatim, with no
    # wrapper of its own.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>{_esc(league_name)} lineup</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(league_name)}</h1>
  <div class="sub stamp">Updated {generated}</div>
  {_team_selector(teams, default_roster_id)}
  {sections}
  <div class="panel"><h2>How to read this</h2>
    <p class="note">Projections blend Sleeper with Kalshi prop markets, capped by how much
    of a player's scoring the market actually prices. There is no rushing-yards market,
    so running backs lean on Sleeper. Win probability comes from simulating both
    lineups; players are drawn independently, which understates the spread of a
    stacked lineup.</p>
  </div>
</div>
{_SCRIPT}
</body>
</html>
"""


def render_moves_panel(
    targets, proposals, rationales=None, activity=None, confidences=None, prefix: str = ""
) -> str:
    """Waiver and trade panels for one team.

    ``prefix`` namespaces every element id. The page carries a section per team
    and the proposal index restarts at zero in each, so without it every Copy
    button on the page would resolve to the first team's message.
    """
    """Waiver and trade panel, appended to the dashboard."""
    parts = []

    if targets:
        rows = "".join(
            f"<li>Add <strong>{_esc(t.player.name)}</strong> "
            f'<span class="slot">({_esc(t.player.position)})</span>'
            + (f", drop {_esc(t.drop.name)}" if t.drop else "")
            + f' <span class="slot">+{t.gain:.0f} pts rest of season</span></li>'
            for t in targets
        )
        parts.append(f'<div class="panel"><h2>Waiver targets</h2><ul>{rows}</ul></div>')

    if proposals:
        from ..engine.trades import rank_proposals

        rationales = rationales or {}
        blocks = []
        ordered = list(proposals)
        for ranked in rank_proposals(ordered):
            p = ranked.proposal
            i = ordered.index(p)
            r = rationales.get(i)
            why = f'<p class="why"><b>Why:</b> {_esc(r.why)}</p>' if r else ""
            angle = f'<p class="why"><b>Their side:</b> {_esc(r.their_angle)}</p>' if r else ""
            pitch = ""
            if r:
                pitch = (
                    f'<div class="pitchwrap">'
                    f'<button class="copy" data-copy="pitch{prefix}{i}">Copy message</button>'
                    f'<pre class="pitch" id="pitch{prefix}{i}">{_esc(r.pitch)}</pre></div>'
                )
            badge = (
                '<span class="badge">Send this one</span>'
                if ranked.rank == 1
                else ""
            )
            conflict = ""
            if ranked.conflicts_with:
                others = ", ".join(f"#{c}" for c in ranked.conflicts_with)
                conflict = (
                    f'<p class="conflict">Competes with {others} - all need '
                    f"{_esc(', '.join(ranked.shared_players))}. Only one can happen.</p>"
                )
            # A range, not a point. The gain is a difference of two
            # optimisations and jumps discretely when a slot assignment flips,
            # so a bare figure reads more settled than it is.
            conf = (confidences or {}).get(i)
            if conf is not None:
                gains = (
                    f'<div class="slot">you +{p.my_gain:.0f} '
                    f"({conf.p10:+.0f} to {conf.p90:+.0f} under projection error), "
                    f"them +{p.their_gain:.0f} rest-of-season pts</div>"
                )
                rests = "".join(
                    f'<p class="conflict">Rests on {_esc(ct.player.name)}: without him '
                    f"{ct.gain_without:+.0f}"
                    + (f", {ct.games_played} games this season" if ct.is_injured else "")
                    + "</p>"
                    for ct in conf.contingencies
                )
            else:
                gains = (
                    f'<div class="slot">you +{p.my_gain:.0f}, them +{p.their_gain:.0f} '
                    f"rest-of-season pts</div>"
                )
                rests = ""
            blocks.append(
                f'<div class="trade"><div class="tradehead">'
                f'<span class="rank">#{ranked.rank}</span> '
                f"<strong>{_esc(p.partner_name)}</strong> "
                f'<span class="slot">[{_esc(p.shape)}]</span> {badge}</div>'
                f"<div>Send <b>{_esc(', '.join(x.name for x in p.give))}</b></div>"
                f"<div>Get <b>{_esc(', '.join(x.name for x in p.get))}</b></div>"
                f"{gains}{rests}{conflict}{why}{angle}{pitch}</div>"
            )
        parts.append(
            f'<div class="panel"><h2>Trade offers</h2>'
            f'<p class="note">Ranked by value to you. Offers sharing a player are '
            f"mutually exclusive - send the top one first and wait.</p>"
            f'{"".join(blocks)}'
            f'<p class="note">Both sides gain, which is what makes an offer worth '
            f"sending. Every claim in these messages comes from roster data the "
            f"other manager can already see.</p></div>"
        )

    if activity:
        rows = "".join(
            # Class computed outside the f-string: escaped quotes are not
            # permitted inside an f-string expression.
            "<li{}>{}</li>".format(
                ' class="mine"' if m.involves_me else "", _esc(m.summary)
            )
            for m in activity
        )
        parts.append(
            f'<div class="panel"><h2>Recent league activity</h2><ul class="feed">{rows}</ul>'
            f'<p class="note">Narrative only. Rosters are re-read in full every run, '
            f"so a completed trade or waiver is reflected whether or not its "
            f"transaction record was seen.</p></div>"
        )

    return "".join(parts)


def utc_now() -> datetime:
    return datetime.now(UTC)
