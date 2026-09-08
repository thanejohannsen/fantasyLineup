"""Terminal rendering of an advisory."""

from __future__ import annotations

from ..engine.locks import LockState, format_countdown
from .advisory import Advisory

_UNKNOWN = LockState(locked=False, kickoff_utc=None)


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
    return f"  {slot:<5} {player.name:<{width}} {player.points:6.2f}  {opp:<7} {marker:>8}"


def render_advisory(advisory: Advisory, team_name: str = "") -> str:
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
            lines.append(f"        {p.name:<26} {p.points:6.2f}          {marker:>8}")

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

    locked = advisory.locked_players
    if locked:
        lines.append("")
        lines.append(f"{len(locked)} starter(s) already locked and excluded from changes.")

    return "\n".join(lines)
