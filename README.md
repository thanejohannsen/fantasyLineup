# fantasyLineup

A lock-aware lineup, waiver and trade advisor for a Sleeper fantasy football
league. It reads Sleeper and Kalshi, and tells you what to change and by when.

**Sleeper's API is read-only** — their docs state it plainly: *"No API Token is
necessary, as you cannot modify contents via this API."* Nothing here can set a
lineup, claim a player, or send an offer. Every recommendation is executed by
hand in the app.

## What it does

- **Recommends a lineup** by solving the slot assignment exactly, rather than
  sorting by projected points. With two FLEX slots a greedy pass can strand a
  dedicated slot.
- **Respects per-player deadlines.** Each player locks at his own kickoff, so by
  Sunday evening most of a roster is frozen while a Monday player is still live.
  Locked starters are pinned and the rest optimised around them, so every
  proposed change is one you can still make.
- **Optimises win probability, not points**, when the opponent is known. A heavy
  favourite protects the floor; a heavy underdog chases the ceiling. Points
  already scored are treated as known, so after a Thursday bust it chases and
  after a Thursday smash it protects — no special-case logic.
- **Ranks waivers and trades by marginal lineup value.** A player is worth what
  he adds to *your* best legal lineup, which is zero for a fifth running back on
  a roster already starting four. Trades must improve both sides, or they get
  declined and the offer was wasted effort.
- **Explains each trade and drafts the message.** Every offer comes with why it
  helps you, why the other manager should say yes, and a sendable pitch. Each
  claim is derived from roster state both managers can already see — a message
  that misstates a roster the other person can check in ten seconds is worse
  than no message, and it poisons the next offer too.
- **Publishes a dashboard** to GitHub Pages, refreshed hourly by Actions.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
# Set league_id and roster_id in config.toml, then:
fl sync                 # league, players, rosters, schedule
fl advise               # lineup for the current week
fl moves                # waiver targets and trade offers
fl recap --week 3       # what happened, and what to learn
fl refresh              # one scheduled pass; writes docs/index.html
```

## Data sources

| Source | Used for | Auth |
| --- | --- | --- |
| Sleeper `/v1/...` | league, rosters, matchups, players | none |
| Sleeper `/projections`, `/stats`, `/schedule` | projections, actuals, game status | none |
| Kalshi `trade-api/v2` | player prop ladders | none |
| ESPN public scoreboard | exact kickoff times, live game state | none |

Sleeper's projection and stats endpoints are undocumented. They return
*component* stats (`rush_yd`, `rec`, `rec_td`), which is why they are used: this
league has 43 scoring keys, and only its own settings produce the number that
appears on the scoreboard. Being undocumented they may change without notice, so
they are wrapped behind one module with fixture-backed tests.

ESPN's *public sports* scoreboard is not the ESPN fantasy API, which needs
browser cookies that expire yearly and is deliberately not used. It supplies
kickoff times, which neither other source can: Sleeper's schedule has a live
status but only a date, and Kalshi's `occurrence_datetime` is approximately game
*end* — it reads 20:00Z for a 1pm ET kickoff.

## What the market can and cannot tell you

Kalshi quotes prop ladders — "250+ passing yards", "275+", "300+" — which read
together are a survival function, and so a distribution. Players join to Sleeper
exactly: `custom_strike.football_player` **is** Sleeper's `kalshi_id`. There is
no name matching anywhere in this project.

Coverage is uneven, and the report says so rather than implying otherwise:

- **Receivers** have receiving yards, receptions and touchdowns quoted — every
  component of their score.
- **Quarterbacks** have passing yards but not passing touchdowns. `KXNFLTD` asks
  whether a player *scores*, which for a quarterback is his rushing.
- **Running backs** have no rushing-yards market at all, so their largest
  component is untouched.

Sleeper is the anchor; the market adjusts it within a cap scaled by how much of a
player's scoring it actually prices.

## Learning from results

`fl calibrate` refits per-position spread and bias on completed weeks, grading
the projection **stored at decision time** rather than one recomputed later,
which would leak hindsight. `fl calibrate --backtest` replays 2025 and
reproduces the shipped coefficients exactly from raw data.

Ten starters a week is far too small a sample to learn from, but the model
projects the whole NFL, so each completed week yields around two thousand
projection-and-outcome pairs. That supports fitting a handful of parameters and
nothing more. Per-player predictive models from one season of a twelve-team
league would be overfitting, and are not attempted.

**Whether the Kalshi blend actually beats Sleeper alone is not yet known, and
cannot be established retroactively.** Kalshi retains no usable pre-kickoff
history: settled markets return an empty book, candlesticks come back empty for
them, and every settled NFL prop market in existence is from 2026 -- the series
did not exist during the 2025 season. The comparison needs several weeks of live
data. Until then the capped, Sleeper-anchored blend is a deliberately
conservative default, not a measured improvement.

Measured on 2025: Sleeper MAE 4.06 points, bias -0.02 over 6,809 observations.

## Known limitations

- **Players are simulated independently.** A quarterback and his receiver rise
  together; a defence falls as the opposing offence rises. Independent draws
  understate the spread of a stacked lineup and overstate how reliably a
  favourite wins. No correlation term is applied because none has been fitted,
  and an invented coefficient would be a confident guess in every
  recommendation.
- **Defensive points-allowed tiers are collapsed to their mode.** Sleeper marks
  one bucket as certain, so DST means run slightly high and DST variance is badly
  understated. Fixing it needs a distribution over points allowed, which the
  Kalshi total and spread markets imply.
- **Thin markets are discarded, not down-weighted.** A ladder whose liquid rungs
  all sit in one tail is rejected outright, even though it carries some
  information.

## Development

```bash
.venv/bin/python -m pytest        # ~80 tests, no network required
```

Tests run against captured real API responses in `fixtures/`, so they fail if an
undocumented endpoint changes shape — which is the point.
