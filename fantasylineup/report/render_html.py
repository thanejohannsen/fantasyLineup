"""The dashboard page, written to docs/ for GitHub Pages.

Published as a static page rather than an interactive artifact because the
refresh runs unattended on a schedule: the page has to be rewritten by a cron
job with no session behind it. One bookmarked URL, always current.

Everything is inlined -- no external CSS, fonts or scripts -- so the page renders
identically offline and cannot break because a CDN changed.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

from ..engine.locks import LockState, format_countdown
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


def render_dashboard(advisory: Advisory, team_name: str, moves_html: str = "") -> str:
    a = advisory
    generated = a.generated_at.strftime("%a %d %b %Y, %H:%M UTC")

    rows = []
    for i, (slot, player) in enumerate(a.optimal.describe()):
        if player is None:
            rows.append(
                f'<tr><td class="slot">{_esc(slot)}</td>'
                f'<td class="empty" colspan="3">no eligible player</td></tr>'
            )
            continue
        state = a.lock_states.get(player.sleeper_id, _UNKNOWN)
        when = (
            "locked"
            if state.locked
            else (
                "bye"
                if state.kickoff_utc is None
                else format_countdown(state.kickoff_utc, a.generated_at)
            )
        )
        cls = ' class="locked"' if state.locked else ""
        opp = f"vs {_esc(player.opponent)}" if player.opponent else ""
        # The absolute kickoff travels with the row so the countdown can be
        # recomputed in the browser; the server-rendered text is the fallback.
        stamp = (
            f' data-kickoff="{state.kickoff_utc.isoformat()}"'
            if state.kickoff_utc is not None and not state.locked
            else ""
        )
        rows.append(
            f"<tr{cls}><td class=\"slot\">{_esc(slot)}</td>"
            f"<td>{_esc(player.name)}{_injury_html(player)} "
            f"<span class=\"slot\">{opp}</span></td>"
            f'<td class="num">{player.points:.1f}</td>'
            f'<td class="when"{stamp}>{_esc(when)}</td></tr>'
        )

    if a.changes:
        changes = "".join(
            f"<li class=\"change\">Start {_esc(c.start.name)} at {_esc(c.slot)}"
            + (f" instead of {_esc(c.sit.name)}" if c.sit else "")
            + f" <span class=\"slot\">+{c.gain:.1f} pts</span></li>"
            for c in a.changes
        )
        changes_block = (
            f'<div class="panel"><h2>Change your lineup</h2><ul>{changes}</ul>'
            f'<p class="note">Sleeper\'s API is read-only. Make these changes in the app.</p></div>'
        )
    else:
        changes_block = (
            '<div class="panel"><h2>Lineup</h2>'
            "<p>Your current lineup is already optimal. Nothing to change.</p></div>"
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
            f"({a.outcome.p10:.0f}-{a.outcome.p90:.0f}, 10th-90th percentile)</div>{banked}</div>"
        )
    else:
        headline = (
            f'<div class="panel"><h2>Week {a.week}</h2>'
            f'<div class="headline">{a.optimal.total_points:.1f} '
            f"<small>projected points</small></div></div>"
        )

    deadline = ""
    deadline_attr = ""
    if a.deadline is not None:
        deadline = (
            f"Next lock in {format_countdown(a.deadline, a.generated_at)} "
            f"({a.deadline:%a %H:%M UTC})"
        )
        deadline_attr = f' data-kickoff="{a.deadline.isoformat()}"'
        deadline = f'<span class="when"{deadline_attr}>{_esc(deadline)}</span>'
    

    # A complete document: GitHub Pages serves the file verbatim, with no
    # wrapper of its own.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>{_esc(team_name)} lineup</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(team_name)}</h1>
  <div class="sub">Updated {generated}{" &middot; " + deadline if deadline else ""}</div>
  {headline}
  {changes_block}
  <div class="panel"><h2>Recommended lineup</h2><div class="scroll"><table>
    <tr><th>Slot</th><th>Player</th><th class="num">Proj</th><th class="when">Locks</th></tr>
    {"".join(rows)}
  </table></div></div>
  {moves_html}
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


def render_moves_panel(targets, proposals, rationales=None, activity=None) -> str:
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
                    f'<button class="copy" data-copy="pitch{i}">Copy message</button>'
                    f'<pre class="pitch" id="pitch{i}">{_esc(r.pitch)}</pre></div>'
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
            blocks.append(
                f'<div class="trade"><div class="tradehead">'
                f'<span class="rank">#{ranked.rank}</span> '
                f"<strong>{_esc(p.partner_name)}</strong> "
                f'<span class="slot">[{_esc(p.shape)}]</span> {badge}</div>'
                f"<div>Send <b>{_esc(', '.join(x.name for x in p.give))}</b></div>"
                f"<div>Get <b>{_esc(', '.join(x.name for x in p.get))}</b></div>"
                f'<div class="slot">you +{p.my_gain:.0f}, them +{p.their_gain:.0f} '
                f"rest-of-season pts</div>{conflict}{why}{angle}{pitch}</div>"
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
