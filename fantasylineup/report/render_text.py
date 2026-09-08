"""Terminal rendering of an advisory."""

from __future__ import annotations

from .advisory import Advisory


def _player_line(slot: str, player, width: int = 26) -> str:
    if player is None:
        return f"  {slot:<5} {'(empty)':<{width}}      --"
    opp = f"vs {player.opponent}" if player.opponent else ""
    return f"  {slot:<5} {player.name:<{width}} {player.points:6.2f}  {opp}"


def render_advisory(advisory: Advisory, team_name: str = "") -> str:
    lines: list[str] = []
    header = f"Week {advisory.week} lineup"
    if team_name:
        header += f" - {team_name}"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    lines.append(f"OPTIMAL  {advisory.optimal.total_points:.2f} projected")
    for slot, player in advisory.optimal.describe():
        lines.append(_player_line(slot, player))

    if advisory.optimal.bench:
        lines.append("")
        lines.append("  Bench")
        for p in advisory.optimal.bench:
            lines.append(f"        {p.name:<26} {p.points:6.2f}")

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
            sit = f"  instead of {c.sit.name} ({c.sit.points:.2f})" if c.sit else ""
            lines.append(f"  START {c.start.name} ({c.start.points:.2f}) at {c.slot}{sit}")

    return "\n".join(lines)
