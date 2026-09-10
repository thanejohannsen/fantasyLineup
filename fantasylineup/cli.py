"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .db import open_db
from .engine.lineup import optimize_lineup, starting_slots
from .engine.confidence import assess_all
from .engine.trades import best_trades_across_league, explain_trade
from .engine.waivers import explain_no_targets, rank_waiver_targets, shortlist_candidates
from .model.calibration import (
    fit_variance,
    load_observations,
    save_params,
    score_source,
)
from .model.projections import REST_OF_SEASON, latest_projections, sync_projections
from .pipeline import (
    MatchupContext,
    all_matchups,
    all_rosters,
    current_starters,
    games_played,
    league_activity,
    live_states,
    blended_projections,
    fetch_market_fits,
    find_opponent,
    roster_names,
    roster_players_for,
    standard_deviations,
)
from .report.advisory import build_advisory
from .report.render_html import TeamView, render_dashboard, render_moves_panel
from .report.recap import build_recap, sync_actuals
from .report.render_text import render_advisory, render_moves, render_recap
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


def _resolve_roster(conn, league_id: str, requested, default: int) -> int:
    """Turn a --team value into a roster id.

    Accepts a roster id or a team name, matched case-insensitively on a prefix
    so `--team unc` finds "Unc Show". Ambiguity is an error rather than a guess:
    reporting on the wrong manager's roster would look like a bug in the model.
    """
    if requested is None:
        return default
    names = roster_names(conn, league_id)
    if str(requested).isdigit() and int(requested) in names:
        return int(requested)
    wanted = str(requested).strip().lower()
    hits = [rid for rid, name in names.items() if name.lower().startswith(wanted)]
    if len(hits) == 1:
        return hits[0]
    listing = ", ".join(sorted(names.values()))
    if not hits:
        raise LookupError(f"no team matching {requested!r}. Teams: {listing}")
    raise LookupError(
        f"{requested!r} matches {len(hits)} teams: "
        + ", ".join(sorted(names[r] for r in hits))
    )


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

        roster_id = _resolve_roster(
            conn, cfg.league.league_id, getattr(args, "team", None), cfg.league.roster_id
        )
        avail = compute_availability(conn, cfg.league.league_id, roster_id)
        fits = {} if args.no_kalshi else fetch_market_fits(cfg.sources.kalshi_base)

        players, blends = blended_projections(
            conn,
            avail.my_players,
            cfg.league.season,
            week,
            scoring,
            fits,
            cfg.model.kalshi_max_shift,
            availability=cfg.health.availability(),
        )

        matchup = find_opponent(conn, client, cfg.league.league_id, roster_id, week)
        opponent_starters: list = []
        opponent_players_for_live: list = []
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
                availability=cfg.health.availability(),
            )
            # We cannot know what they will actually start, so assume they play
            # their best legal lineup. Assuming less would flatter our own odds.
            opponent_players_for_live = opponent_players
            opponent_starters = optimize_lineup(
                opponent_players, starting_slots(league["roster_positions"])
            ).starters
            sds.update(standard_deviations(opponent_players, opponent_blends))

        advisory = build_advisory(
            conn,
            avail.snapshot_id,
            roster_id,
            players,
            league["roster_positions"],
            cfg.league.season,
            week,
            opponent_starters=opponent_starters,
            opponent_name=matchup.opponent_name,
            sds=sds,
            my_banked=matchup.my_banked,
            opponent_banked=matchup.opponent_banked,
            live=live_states(
                conn,
                client.matchups(cfg.league.league_id, week),
                cfg.league.season,
                week,
                [*players, *opponent_players_for_live],
            ),
            draws=cfg.model.sim_draws,
        )

    print(render_advisory(advisory, team_name=league.get("name", ""), blends=blends))

    if args.publish:
        # One team only. `fl refresh` is what writes the full league page; this
        # is the quick single-team publish, so the selector is omitted.
        target = cfg.paths.site / "index.html"
        target.write_text(
            render_dashboard(
                league.get("name", "Fantasy"),
                [
                    TeamView(
                        roster_id=roster_id,
                        name=league.get("name", "Fantasy"),
                        advisory=advisory,
                    )
                ],
                default_roster_id=roster_id,
            ),
            encoding="utf-8",
        )
        print(f"\nDashboard written to {target}")
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
            # The weekly slate is also needed, not for valuation but because
            # whether a player has a weekly projection is what tells the health
            # model he is playing rather than out for the season.
            sync_projections(
                conn, client, cfg.league.season, _resolve_week(client, None), scoring
            )

        fits = {} if args.no_kalshi else fetch_market_fits(cfg.sources.kalshi_base)
        roster_id = _resolve_roster(
            conn, cfg.league.league_id, getattr(args, "team", None), cfg.league.roster_id
        )
        avail = compute_availability(conn, cfg.league.league_id, roster_id)
        ros = REST_OF_SEASON

        # weekly_week tells the health model whether a player is actually
        # playing, which is what separates "back from surgery" from "done for
        # the year". Without it every injured player would look identical.
        current_week = _resolve_week(client, None)

        def build(ids):
            return blended_projections(
                conn,
                ids,
                cfg.league.season,
                ros,
                scoring,
                fits,
                cfg.model.kalshi_max_shift,
                health_multipliers=cfg.health.as_table(),
                weekly_week=current_week,
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
            my_players,
            rosters,
            names,
            slots,
            roster_id,
            limit=args.limit,
            games_played=games_played(conn, cfg.league.season),
            jitter=cfg.trades.jitter,
            draws=cfg.trades.draws,
            min_games=cfg.trades.min_games_for_contingency,
            require_stable=cfg.trades.require_stable,
        )
        confidences = assess_all(
            proposals,
            my_players,
            slots,
            games_played=games_played(conn, cfg.league.season),
            jitter=cfg.trades.jitter,
            draws=cfg.trades.draws,
            min_games=cfg.trades.min_games_for_contingency,
            require_stable=cfg.trades.require_stable,
        )
        my_set = current_starters(conn, cfg.league.league_id, roster_id, my_players)
        rationales = {
            i: explain_trade(
                p,
                my_players,
                rosters[p.partner_roster_id],
                slots,
                my_starters=my_set,
                their_starters=current_starters(
                    conn,
                    cfg.league.league_id,
                    p.partner_roster_id,
                    rosters[p.partner_roster_id],
                ),
            )
            for i, p in enumerate(proposals)
        }

        activity = league_activity(
            conn, client, cfg.league.league_id, current_week, roster_id
        )

    print(
        render_moves(
            targets, proposals, roster_limit, near_miss, rationales, activity, confidences
        )
    )
    return 0


def _next_kickoff_hours(conn, season: int, week: int) -> float | None:
    """Hours until the next game that has not started, or None if all have."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    upcoming = []
    for r in conn.execute(
        "SELECT kickoff_utc, status FROM games WHERE season = ? AND week = ?", (season, week)
    ):
        if not r["kickoff_utc"] or (r["status"] and r["status"] != "pre"):
            continue
        try:
            kickoff = datetime.fromisoformat(r["kickoff_utc"])
        except ValueError:
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=UTC)
        if kickoff > now:
            upcoming.append((kickoff - now).total_seconds() / 3600)
    return min(upcoming) if upcoming else None


def cmd_refresh(args: argparse.Namespace) -> int:
    """One scheduled pass: sync, re-advise, and rewrite the dashboard.

    Meant to be called hourly. The expensive part is the trade search, which
    only runs when it is worth running -- there is no point re-searching trades
    every hour when rosters move a few times a week.
    """
    from datetime import UTC, datetime

    cfg = load_config(args.config)
    cfg.paths.ensure()
    now = datetime.now(UTC)

    with open_db(cfg.paths.db) as conn, SleeperClient(
        cfg.sources.sleeper_base,
        cfg.paths.cache,
        max_calls_per_min=cfg.sources.max_calls_per_min,
    ) as client:
        league = sync_league(conn, client, cfg.league.league_id)
        scoring = league["scoring_settings"]
        week = _resolve_week(client, args.week)

        sync_users(conn, client, cfg.league.league_id)
        sync_players(conn, client, ttl_hours=cfg.sources.players_dump_ttl_hours)
        sync_rosters(conn, client, cfg.league.league_id)
        sync_schedule(conn, client, cfg.league.season)
        sync_kickoffs(conn, cfg.league.season, week)
        sync_projections(conn, client, cfg.league.season, week, scoring)

        hours = _next_kickoff_hours(conn, cfg.league.season, week)
        # Trades and waivers are rest-of-season decisions that do not change
        # hour to hour. Refresh them well before the slate, not during it.
        want_moves = args.with_moves or (hours is not None and hours > 24)
        # Rest-of-season numbers are shown beside the weekly ones whether or not
        # the trade search runs, so they are fetched every pass. One request,
        # and a stale season column is worse than none.
        sync_projections(conn, client, cfg.league.season, None, scoring)

        fits = {} if args.no_kalshi else fetch_market_fits(cfg.sources.kalshi_base)
        slots = starting_slots(league["roster_positions"])
        avail = compute_availability(conn, cfg.league.league_id, cfg.league.roster_id)
        league_rosters = all_rosters(conn, cfg.league.league_id)
        names = roster_names(conn, cfg.league.league_id)
        entries = client.matchups(cfg.league.league_id, week)
        matchups = all_matchups(conn, client, cfg.league.league_id, week, entries=entries)

        # Every team is also exactly one other team's opponent, so memoising the
        # weekly blend halves the work rather than merely tidying it.
        weekly: dict[int, tuple] = {}

        def weekly_for(rid: int):
            if rid not in weekly:
                weekly[rid] = blended_projections(
                    conn, league_rosters.get(rid, set()), cfg.league.season, week,
                    scoring, fits, cfg.model.kalshi_max_shift,
                    availability=cfg.health.availability(),
                )
            return weekly[rid]

        # Season totals for every rostered player, keyed by id so a renderer
        # can look one up without re-deriving the blend.
        season_points: dict[str, float] = {}
        for rid_ in sorted(league_rosters):
            for pl in blended_projections(
                conn, league_rosters[rid_], cfg.league.season, REST_OF_SEASON,
                scoring, fits, cfg.model.kalshi_max_shift,
                health_multipliers=cfg.health.as_table(),
                weekly_week=week,
                return_weeks=cfg.health.return_weeks(),
            )[0]:
                season_points[pl.sleeper_id] = pl.points

        moves_html = _league_moves_html(
            conn, client, cfg, league, scoring, fits, week, avail, names
        ) if want_moves else {}

        views: list[TeamView] = []
        for rid in sorted(league_rosters):
            players, blends = weekly_for(rid)
            matchup = matchups.get(rid) or MatchupContext(None, "unknown", 0.0, 0.0)
            sds = standard_deviations(players, blends)
            opponent_starters: list = []
            opp_players_for_live: list = []
            if matchup.opponent_roster_id is not None:
                opp_players, opp_blends = weekly_for(matchup.opponent_roster_id)
                opp_players_for_live = opp_players
                opponent_starters = optimize_lineup(opp_players, slots).starters
                sds.update(standard_deviations(opp_players, opp_blends))

            # Points already on the board are known, so they must not be
            # re-drawn alongside the projection that anticipated them.
            live = live_states(
                conn, entries, cfg.league.season, week, [*players, *opp_players_for_live]
            )
            views.append(
                TeamView(
                    roster_id=rid,
                    name=names.get(rid, f"roster {rid}"),
                    advisory=build_advisory(
                        conn, avail.snapshot_id, rid, players,
                        league["roster_positions"], cfg.league.season, week,
                        opponent_starters=opponent_starters,
                        opponent_name=matchup.opponent_name,
                        sds=sds, my_banked=matchup.my_banked,
                        opponent_banked=matchup.opponent_banked,
                        live=live,
                        draws=cfg.model.sim_draws, now=now,
                    ),
                    moves_html=moves_html.get(rid, ""),
                    season_points=season_points,
                )
            )

    # Thane's team first, then the rest alphabetically: the page opens on his
    # own team, and the others are a list to scan rather than a roster order
    # nobody knows by heart.
    views.sort(key=lambda v: (v.roster_id != cfg.league.roster_id, v.name.lower()))
    target = cfg.paths.site / "index.html"
    target.write_text(
        render_dashboard(
            league.get("name", "Fantasy"), views, default_roster_id=cfg.league.roster_id
        ),
        encoding="utf-8",
    )
    mine = matchups.get(cfg.league.roster_id)
    print(f"Dashboard written to {target}")
    print(
        f"Week {week} vs {mine.opponent_name if mine else 'unknown'}; "
        f"{len(views)} teams; moves refreshed: {want_moves}"
    )
    return 0


def _league_moves_html(
    conn, client, cfg, league, scoring, fits, week, avail, names
) -> dict[int, str]:
    """Waiver and trade panels for every team, keyed by roster id.

    Run once for the whole league rather than once per team. The projection
    blend for all twelve rosters, the free-agent pool and the games-played
    table are identical whichever seat you are sitting in; only the perspective
    passed to the trade search changes.
    """
    slots = starting_slots(league["roster_positions"])
    roster_limit = len([s for s in league["roster_positions"] if s != "IR"])
    ros = REST_OF_SEASON

    def build(ids):
        return blended_projections(
            conn, ids, cfg.league.season, ros, scoring, fits,
            cfg.model.kalshi_max_shift,
            health_multipliers=cfg.health.as_table(),
            weekly_week=week,
            return_weeks=cfg.health.return_weeks(),
        )[0]

    rosters = {rid: build(ids) for rid, ids in all_rosters(conn, cfg.league.league_id).items()}
    # Availability is derived by subtraction, so the free-agent pool is the same
    # for everyone; only who can use it differs.
    candidates = build(
        shortlist_candidates(
            conn, avail.free_agents, latest_projections(conn, cfg.league.season, ros)
        )
    )
    played = games_played(conn, cfg.league.season)
    tuning = dict(
        games_played=played,
        jitter=cfg.trades.jitter,
        draws=cfg.trades.draws,
        min_games=cfg.trades.min_games_for_contingency,
        require_stable=cfg.trades.require_stable,
    )

    out: dict[int, str] = {}
    for rid, mine in rosters.items():
        proposals = best_trades_across_league(
            mine, rosters, names, slots, rid, limit=4, **tuning
        )
        my_set = current_starters(conn, cfg.league.league_id, rid, mine)
        out[rid] = render_moves_panel(
            rank_waiver_targets(
                mine, candidates, slots, roster_limit=roster_limit, limit=5
            ),
            proposals,
            {
                i: explain_trade(
                    p,
                    mine,
                    rosters[p.partner_roster_id],
                    slots,
                    my_starters=my_set,
                    their_starters=current_starters(
                        conn,
                        cfg.league.league_id,
                        p.partner_roster_id,
                        rosters[p.partner_roster_id],
                    ),
                )
                for i, p in enumerate(proposals)
            },
            league_activity(conn, client, cfg.league.league_id, week, rid),
            assess_all(proposals, mine, slots, **tuning),
            # Element ids must be unique across the whole document, not just
            # within one team's section.
            prefix=f"-{rid}-",
        )
    return out


def cmd_recap(args: argparse.Namespace) -> int:
    """What happened last week, and which calls were genuinely wrong."""
    cfg = load_config(args.config)
    cfg.paths.ensure()

    with open_db(cfg.paths.db) as conn, SleeperClient(
        cfg.sources.sleeper_base,
        cfg.paths.cache,
        max_calls_per_min=cfg.sources.max_calls_per_min,
    ) as client:
        league = load_league(conn, cfg.league.league_id)
        scoring = league["scoring_settings"]
        week = args.week if args.week is not None else max(1, _resolve_week(client, None) - 1)

        sync_actuals(conn, client, cfg.league.season, week, scoring)
        avail = compute_availability(conn, cfg.league.league_id, cfg.league.roster_id)

        # Projections are read back at the timestamp they were stored, never
        # recomputed: grading a past week with today's data would leak hindsight
        # and make the measured accuracy meaningless.
        players, _ = blended_projections(
            conn, avail.my_players, cfg.league.season, week, scoring, {}, 0.0,
            apply_weekly_health=False,
        )
        matchup = find_opponent(conn, client, cfg.league.league_id, cfg.league.roster_id, week)
        recap = build_recap(
            conn, avail.snapshot_id, cfg.league.roster_id, players,
            league["roster_positions"], cfg.league.season, week,
            my_points=matchup.my_banked, opponent_points=matchup.opponent_banked,
            opponent_name=matchup.opponent_name,
        )

    print(render_recap(recap))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Refit the model on completed weeks and version the result.

    With --backtest, replays the 2025 season from scratch. That measures the
    Sleeper baseline only: Kalshi kept no usable pre-kickoff history and its NFL
    prop series did not exist in 2025, so whether the blend helps can only be
    answered prospectively as 2026 weeks accumulate.
    """
    cfg = load_config(args.config)
    cfg.paths.ensure()

    season = 2025 if args.backtest else cfg.league.season
    weeks = list(range(1, (args.through or 18) + 1))

    with open_db(cfg.paths.db) as conn, SleeperClient(
        cfg.sources.sleeper_base,
        cfg.paths.cache,
        max_calls_per_min=cfg.sources.max_calls_per_min,
    ) as client:
        league = load_league(conn, cfg.league.league_id)
        scoring = league["scoring_settings"]

        if args.backtest:
            for week in weeks:
                sync_projections(conn, client, season, week, scoring)
                sync_actuals(conn, client, season, week, scoring)

        observations = load_observations(conn, season, weeks)
        if not observations:
            print(f"No completed weeks with stored projections for {season}.")
            print("Projections are graded as stored at decision time, so they")
            print("cannot be reconstructed after the fact. Run `fl refresh`")
            print("before each slate and this fills in as the season goes.")
            return 0

        baseline = score_source(observations)
        blend = score_source(load_observations(conn, season, weeks, source="blend"))
        variance = fit_variance(observations)
        version = save_params(
            conn,
            variance,
            {"sleeper": baseline, **({"blend": blend} if blend.n else {})},
            through_week=max(weeks),
            notes="backtest 2025" if args.backtest else f"in-season {season}",
        )

    print(f"Calibration v{version} - {season}, {len(observations)} observations")
    print()
    print("ACCURACY")
    print(f"  {baseline.describe()}")
    if blend.n:
        print(f"  {blend.describe()}")
    else:
        print("  blend      no stored history yet - see note below")
    print()
    print("FITTED SPREAD  sd = intercept + slope * projection")
    for position in ("QB", "RB", "WR", "TE", "K", "DEF"):
        if position in variance:
            intercept, slope = variance[position]
            print(f"  {position:4} {intercept:6.2f} + {slope:6.3f} x proj")
    print()
    if not blend.n:
        print("The Kalshi blend has no measurable track record yet, and cannot")
        print("get one retroactively: settled markets return an empty book and")
        print("every NFL prop market in existence is from 2026. Expect a few")
        print("weeks of live data before the comparison means anything.")
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
    p_advise.add_argument(
        "--publish", action="store_true", help="also write the GitHub Pages dashboard"
    )
    p_advise.add_argument(
        "--team", default=None, help="another manager's team, by name or roster id"
    )
    p_advise.set_defaults(func=cmd_advise)

    p_moves = sub.add_parser("moves", help="waiver targets and trade offers")
    p_moves.add_argument("--limit", type=int, default=5)
    p_moves.add_argument("--no-refresh", action="store_true")
    p_moves.add_argument("--no-kalshi", action="store_true")
    p_moves.add_argument(
        "--team", default=None, help="another manager's team, by name or roster id"
    )
    p_moves.set_defaults(func=cmd_moves)

    p_refresh = sub.add_parser("refresh", help="scheduled pass: sync, advise, publish dashboard")
    p_refresh.add_argument("-w", "--week", type=int, default=None)
    p_refresh.add_argument("--with-moves", action="store_true", help="force the trade search")
    p_refresh.add_argument("--no-kalshi", action="store_true")
    p_refresh.set_defaults(func=cmd_refresh)

    p_recap = sub.add_parser("recap", help="last week's results and what to learn")
    p_recap.add_argument("-w", "--week", type=int, default=None)
    p_recap.set_defaults(func=cmd_recap)

    p_cal = sub.add_parser("calibrate", help="refit the model on completed weeks")
    p_cal.add_argument("--backtest", action="store_true", help="replay the 2025 season")
    p_cal.add_argument("--through", type=int, default=None, help="last week to include")
    p_cal.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except (LookupError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
