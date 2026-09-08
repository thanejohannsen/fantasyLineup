"""Terminal rendering of an advisory."""

from __future__ import annotations

import textwrap

from ..engine.locks import LockState, format_countdown
from ..model.health import Regime, is_structural
from .advisory import Advisory

_UNKNOWN = LockState(locked=False, kickoff_utc=None)


def _injury_tag(player) -> str:
    """Compact designation for a roster line, e.g. ``[Q - Achilles]``."""
    status = getattr(player, "injury_status", None)
    if not status:
        return ""
    initial = {"Questionable": "Q", "Doubtful": "D", "Out": "OUT"}.get(status, status)
    part = getattr(player, "injury_body_part", None)
    if part and part != "Undisclosed":
        return f"[{initial} - {part}]"
    return f"[{initial}]"


def _wrap(text: str, width: int, indent: int) -> str:
    """Wrap prose to a readable measure, aligned under its label."""
    pad = " " * indent
    return ("\n" + pad).join(textwrap.wrap(text, width))


def _player_line(slot: str, player, advisory: Advisory, width: int = 26) -> str:
    if player is None:
        return f"  {slot:<5} {'(empty)':<{width}}      --"
    state = advisory.lock_states.get(player.sleeper_id, _UNKNOWN)
    if state.locked:
        marker = "LOCKED"
    elif state.kickoff_utc is None:
        marker = "BYE"
    else:
        marker = format_countdown(state.kickoff_utc, advisory.generated_at)
    opp = f"vs {player.opponent}" if player.opponent else ""
    tag = _injury_tag(player)
    name = f"{player.name} {tag}".strip() if tag else player.name
    return f"  {slot:<5} {name:<{width}} {player.points:6.2f}  {opp:<7} {marker:>8}"


def render_advisory(advisory: Advisory, team_name: str = "", blends: dict | None = None) -> str:
    lines: list[str] = []
    header = f"Week {advisory.week} lineup"
    if team_name:
        header += f" - {team_name}"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(f"as of {advisory.generated_at:%a %d %b %H:%M UTC}")

    if advisory.deadline is not None:
        countdown = format_countdown(advisory.deadline, advisory.generated_at)
        lines.append(f"next lock in {countdown} ({advisory.deadline:%a %H:%M UTC})")

    if advisory.outcome is not None:
        wp = advisory.outcome.win_probability
        lines.append("")
        lines.append(
            f"vs {advisory.opponent_name}:  win probability {wp:6.1%}   "
            f"({advisory.outcome.p10:.0f} - {advisory.outcome.p90:.0f} pts, 10th-90th)"
        )
        lines.append(f"  {advisory.posture}")
        if advisory.winprob_swaps:
            gain = wp - advisory.ev_outcome.win_probability
            lines.append(
                f"  {advisory.winprob_swaps} swap(s) away from the highest-points lineup, "
                f"worth +{gain:.1%} win probability"
            )
        if advisory.my_banked or advisory.opponent_banked:
            lines.append(
                f"  banked so far: you {advisory.my_banked:.1f}, "
                f"them {advisory.opponent_banked:.1f}"
            )
    lines.append("")

    lines.append(f"OPTIMAL  {advisory.optimal.total_points:.2f} projected")
    for slot, player in advisory.optimal.describe():
        lines.append(_player_line(slot, player, advisory))

    if advisory.optimal.bench:
        lines.append("")
        lines.append("  Bench")
        for p in advisory.optimal.bench:
            state = advisory.lock_states.get(p.sleeper_id, _UNKNOWN)
            marker = "LOCKED" if state.locked else ""
            tag = _injury_tag(p)
            name = f"{p.name} {tag}".strip() if tag else p.name
            lines.append(f"        {name:<26} {p.points:6.2f}          {marker:>8}")

    lines.append("")
    if advisory.optimal.unfilled:
        empty = ", ".join(advisory.slots[i] for i in advisory.optimal.unfilled)
        lines.append(f"! No eligible player for: {empty}")
        lines.append("")

    if not advisory.changes:
        lines.append(
            f"Current lineup is already optimal ({advisory.current.total_points:.2f} projected)."
        )
    else:
        lines.append(
            f"CHANGES  {advisory.current.total_points:.2f} -> "
            f"{advisory.optimal.total_points:.2f}  (+{advisory.gain:.2f})"
        )
        for c in advisory.changes:
            state = advisory.lock_states.get(c.start.sleeper_id, _UNKNOWN)
            by = format_countdown(state.kickoff_utc, advisory.generated_at)
            sit = f" instead of {c.sit.name} ({c.sit.points:.2f})" if c.sit else ""
            lines.append(
                f"  START {c.start.name} ({c.start.points:.2f}) at {c.slot}{sit}"
                f"   [decide within {by}]"
            )

    if blends:
        covered = [b for b in blends.values() if b.has_market]
        if covered:
            lines.append("")
            lines.append(
                f"Market data informed {len(covered)} of {len(blends)} players "
                f"(Kalshi prop ladders; no rushing-yards market exists, so running "
                f"backs lean on Sleeper)."
            )
            names = {
                p.sleeper_id: p.name
                for p in [*advisory.optimal.assignments.values(), *advisory.optimal.bench]
            }
            for b in sorted(covered, key=lambda b: abs(b.shift), reverse=True)[:4]:
                if abs(b.shift) < 0.05:
                    continue
                lines.append(
                    f"    {names.get(b.sleeper_id, b.sleeper_id):<24} "
                    f"{b.baseline:6.2f} -> {b.blended:6.2f} "
                    f"({b.shift:+.2f}, coverage {b.coverage:.0%})"
                )

    locked = advisory.locked_players
    if locked:
        lines.append("")
        lines.append(f"{len(locked)} starter(s) already locked and excluded from changes.")

    return "\n".join(lines)


def render_moves(
    targets, proposals, roster_limit: int, near_miss=None, rationales=None, activity=None
) -> str:
    """Waiver targets and trade offers.

    Gains are rest-of-season points added to the best legal lineup, which is
    what a move is actually worth -- not the incoming player's projection.
    """
    lines: list[str] = ["Roster moves", "============", ""]

    lines.append("WAIVER TARGETS  (net gain after the required drop)")
    if not targets:
        lines.append("  Nothing available improves the lineup.")
        if near_miss:
            weakest, near = near_miss
            if weakest is not None:
                lines.append(
                    f"  Bar to clear: your weakest displaceable starter is {weakest.name} "
                    f"({weakest.position}, {weakest.points:.0f} ROS)."
                )
            for c in near:
                lines.append(f"    closest: {c.name} ({c.position}, {c.points:.0f} ROS)")
    for t in targets:
        drop = f"  drop {t.drop.name} ({t.drop.points:.1f})" if t.drop else ""
        tag = _injury_tag(t.player)
        lines.append(
            f"  +{t.gain:5.1f}  ADD {t.player.name} {tag}".rstrip()
            + f" ({t.player.position}, {t.player.points:.1f} ROS){drop}"
        )

    lines.append("")
    lines.append("TRADE OFFERS  (ranked by value to you; both sides must gain)")
    if not proposals:
        lines.append("  No mutually beneficial trade found.")
        lines.append(
            "  With two FLEX slots a surplus running back or receiver still starts, "
            "so positional imbalance has to be severe before a swap helps either side."
        )
    from ..engine.trades import rank_proposals

    rationales = rationales or {}
    ranked = rank_proposals(list(proposals))
    for r in ranked:
        p = r.proposal
        i = list(proposals).index(p)
        give = ", ".join(
            f"{x.name}{' ' + _injury_tag(x) if _injury_tag(x) else ''} ({x.points:.1f})"
            for x in p.give
        )
        get = ", ".join(
            f"{x.name}{' ' + _injury_tag(x) if _injury_tag(x) else ''} ({x.points:.1f})"
            for x in p.get
        )
        lines.append("")
        flag = "SEND THIS ONE" if r.rank == 1 else ""
        lines.append(f"  #{r.rank}  {p.partner_name}  [{p.shape}]  {flag}")
        if r.conflicts_with:
            others = ", ".join(f"#{c}" for c in r.conflicts_with)
            lines.append(
                f"      -- rules out / ruled out by {others}: "
                f"both need {', '.join(r.shared_players)}"
            )
        lines.append(f"      you send    {give}")
        lines.append(f"      you receive {get}")
        lines.append(f"      you +{p.my_gain:.1f} ROS pts, them +{p.their_gain:.1f}")
        r = rationales.get(i)
        if r:
            lines.append("")
            lines.append(f"      WHY   {_wrap(r.why, 62, 12)}")
            lines.append(f"      THEM  {_wrap(r.their_angle, 62, 12)}")
            lines.append("")
            lines.append("      ---- copy and send ----")
            for para in r.pitch.split("\n"):
                lines.append(f"      {para}" if para else "")
            lines.append("      -----------------------")

    if activity:
        lines.append("")
        lines.append("RECENT LEAGUE ACTIVITY")
        for m in activity:
            mark = " <- you" if m.involves_me else ""
            lines.append(f"  {m.summary}{mark}")

    lines.append("")
    lines.append(f"Roster limit {roster_limit}. Sleeper's API is read-only: execute these by hand.")
    return "\n".join(lines)


def render_recap(recap) -> str:
    """Week in review, oriented around what to learn rather than the score."""
    r = recap
    header = f"Week {r.week} recap"
    if not r.played:
        return "\n".join([
            header,
            "=" * len(header),
            "",
            f"Week {r.week} has not been played yet - no results to recap.",
        ])

    verdict = "WON" if r.won else "LOST"
    lines = [
        header,
        "=" * len(header),
        "",
        f"{verdict}  {r.my_points:.1f} - {r.opponent_points:.1f}  vs {r.opponent_name}",
        f"Points left on the bench: {r.points_left_on_bench:.1f}",
        "",
    ]

    lines.append("STARTERS: projected vs actual")
    for res in sorted((x for x in r.results if x.started), key=lambda x: -x.actual):
        lines.append(
            f"  {res.player.name:<24} {res.projected:6.1f} -> {res.actual:6.1f}  {res.miss:+6.1f}"
        )

    if r.notable_bench:
        lines.append("")
        lines.append("BENCH players who beat a starter")
        for res in r.notable_bench:
            lines.append(
                f"  {res.player.name:<24} {res.projected:6.1f} -> {res.actual:6.1f}"
            )
        lines.append(
            "  A bench player outscoring a starter is only a mistake if it was"
        )
        lines.append(
            "  foreseeable. Check the projection, not the outcome."
        )

    lines.append("")
    lines.append(
        "Points left on the bench measures the perfect-hindsight lineup, which"
    )
    lines.append(
        "nobody can set. It is a variance gauge, not a scorecard."
    )
    return "\n".join(lines)
