"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .db import open_db
from .engine.lineup import starting_slots
from .engine.trades import best_trades_across_league
from .engine.waivers import explain_no_targets, rank_waiver_targets, shortlist_candidates
from .model.projections import REST_OF_SEASON, latest_projections, sync_projections
from .pipeline import (
    all_rosters,
    blended_projections,
    fetch_market_fits,
    find_opponent,
    roster_names,
    roster_players_for,
    standard_deviations,
)
from .report.advisory import build_advisory
from .report.render_text import render_advisory, render_moves
from .sources.kickoffs import sync_kickoffs
from .sources.sleeper import SleeperClient
from .sync import (
    compute_availability,
    load_league,
    sync_league,
    sync_players,
    sync_rosters,
    sync_schedule,
    sync_users,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def cmd_sync(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.paths.ensure()

    with open_db(cfg.paths.db) as conn, SleeperClient(
        cfg.sources.sleeper_base,
        cfg.paths.cache,
        max_calls_per_min=cfg.sources.max_calls_per_min,
    ) as client:
        league = sync_league(conn, client, cfg.league.league_id)
        sync_users(conn, client, cfg.league.league_id)
        n_players = sync_players(conn, client, ttl_hours=cfg.sources.players_dump_ttl_hours)
        snapshot_id = sync_rosters(conn, client, cfg.league.league_id)
        n_games = sync_schedule(conn, client, cfg.league.season)
        avail = compute_availability(conn, cfg.league.league_id, cfg.league.roster_id)

    n_teams = league.get("total_rosters")
    print(f"League      {league.get('name')} ({league.get('season')}, {n_teams} teams)")
    print(f"Players     {n_players} fantasy-relevant of the full dump")
    print(f"Games       {n_games} scheduled")
    print(f"Snapshot    #{snapshot_id}")
    print(f"Rostered    {avail.rostered_count} across the league")
    print(f"My roster   {len(avail.my_players)} players (roster_id {cfg.league.roster_id})")
    print(f"Free agents {len(avail.free_agents)} active players unrostered")
    return 0


def _resolve_week(client: SleeperClient, requested: int | None) -> int:
    if requested is not None:
        return requested
    state = client.state()
    return int(state.get("display_week") or state.get("week") or 1)


def cmd_advise(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.paths.ensure()

    with open_db(cfg.paths.db) as conn, SleeperClient(
        cfg.sources.sleeper_base,
        cfg.paths.cache,
        max_calls_per_min=cfg.sources.max_calls_per_min,
    ) as client:
        league = load_league(conn, cfg.league.league_id)
        scoring = league["scoring_settings"]
        week = _resolve_week(client, args.week)

        if not args.no_refresh:
            sync_rosters(conn, client, cfg.league.league_id)
            sync_users(conn, client, cfg.league.league_id)
            sync_schedule(conn, client, cfg.league.season)
            sync_kickoffs(conn, cfg.league.season, week)
            sync_projections(conn, client, cfg.league.season, week, scoring)
            if args.ros:
                sync_projections(conn, client, cfg.league.season, None, scoring)

        avail = compute_availability(conn, cfg.league.league_id, cfg.league.roster_id)
        fits = {} if args.no_kalshi else fetch_market_fits(cfg.sources.kalshi_base)

        players, blends = blended_projections(
            conn,
            avail.my_players,
            cfg.league.season,
            week,
            scoring,
            fits,
            cfg.model.kalshi_max_shift,
        )

        matchup = find_opponent(conn, client, cfg.league.league_id, cfg.league.roster_id, week)
        opponent_starters: list = []
        sds = standard_deviations(players, blends)
        if matchup.opponent_roster_id is not None:
            opponent_ids = roster_players_for(
                conn, cfg.league.league_id, matchup.opponent_roster_id
            )
            opponent_players, opponent_blends = blended_projections(
                conn,
                opponent_ids,
                cfg.league.season,
                week,
                scoring,
                fits,
                cfg.model.kalshi_max_shift,
            )
            # We cannot know what they will actually start, so assume they play
            # their best legal lineup. Assuming less would flatter our own odds.
            from .engine.lineup import optimize_lineup

            opponent_starters = optimize_lineup(
                opponent_players, starting_slots(league["roster_positions"])
            ).starters
            sds.update(standard_deviations(opponent_players, opponent_blends))

        advisory = build_advisory(
            conn,
            avail.snapshot_id,
            cfg.league.roster_id,
            players,
            league["roster_positions"],
            cfg.league.season,
            week,
            opponent_starters=opponent_starters,
            opponent_name=matchup.opponent_name,
            sds=sds,
            my_banked=matchup.my_banked,
            opponent_banked=matchup.opponent_banked,
            draws=cfg.model.sim_draws,
        )

    print(render_advisory(advisory, team_name=league.get("name", ""), blends=blends))
    return 0


def cmd_moves(args: argparse.Namespace) -> int:
    """Waiver targets and trade offers, valued on rest-of-season projections."""
    cfg = load_config(args.config)
    cfg.paths.ensure()

    with open_db(cfg.paths.db) as conn, SleeperClient(
        cfg.sources.sleeper_base,
        cfg.paths.cache,
        max_calls_per_min=cfg.sources.max_calls_per_min,
    ) as client:
        league = load_league(conn, cfg.league.league_id)
        scoring = league["scoring_settings"]
        slots = starting_slots(league["roster_positions"])
        roster_limit = len([s for s in league["roster_positions"] if s != "IR"])

        if not args.no_refresh:
            sync_rosters(conn, client, cfg.league.league_id)
            sync_users(conn, client, cfg.league.league_id)
            # Rest-of-season is the right horizon: a trade changes the roster
            # for the remainder of the season, not for one matchup.
            sync_projections(conn, client, cfg.league.season, None, scoring)

        fits = {} if args.no_kalshi else fetch_market_fits(cfg.sources.kalshi_base)
        avail = compute_availability(conn, cfg.league.league_id, cfg.league.roster_id)
        ros = REST_OF_SEASON

        def build(ids):
            return blended_projections(
                conn, ids, cfg.league.season, ros, scoring, fits, cfg.model.kalshi_max_shift
            )[0]

        my_players = build(avail.my_players)

        # --- waivers -------------------------------------------------------
        pool = shortlist_candidates(
            conn, avail.free_agents, latest_projections(conn, cfg.league.season, ros)
        )
        candidates = build(pool)
        targets = rank_waiver_targets(
            my_players, candidates, slots, roster_limit=roster_limit, limit=args.limit
        )
        near_miss = explain_no_targets(my_players, candidates, slots)

        # --- trades --------------------------------------------------------
        rosters = {rid: build(ids) for rid, ids in all_rosters(conn, cfg.league.league_id).items()}
        names = roster_names(conn, cfg.league.league_id)
        proposals = best_trades_across_league(
            my_players, rosters, names, slots, cfg.league.roster_id, limit=args.limit
        )

    print(render_moves(targets, proposals, roster_limit, near_miss))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fl", description="Lock-aware Sleeper lineup, waiver and trade advisor"
    )
    parser.add_argument("-c", "--config", default="config.toml", help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="pull league, players, rosters and schedule")
    p_sync.set_defaults(func=cmd_sync)

    p_advise = sub.add_parser("advise", help="recommend a starting lineup for a week")
    p_advise.add_argument("-w", "--week", type=int, default=None, help="defaults to current week")
    p_advise.add_argument(
        "--no-refresh", action="store_true", help="use cached data without refetching"
    )
    p_advise.add_argument(
        "--ros", action="store_true", help="also refresh rest-of-season projections"
    )
    p_advise.add_argument(
        "--no-kalshi", action="store_true", help="skip market data, use Sleeper projections only"
    )
    p_advise.set_defaults(func=cmd_advise)

    p_moves = sub.add_parser("moves", help="waiver targets and trade offers")
    p_moves.add_argument("--limit", type=int, default=5)
    p_moves.add_argument("--no-refresh", action="store_true")
    p_moves.add_argument("--no-kalshi", action="store_true")
    p_moves.set_defaults(func=cmd_moves)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except (LookupError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
