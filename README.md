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
  after a Thursday smash it protects — no special-case logic. A player partway
  through his game keeps the fraction of his projection still to come, scaled by
  the game clock.
- **Ranks waivers and trades by marginal lineup value.** A player is worth what
  he adds to *your* best legal lineup, which is zero for a fifth running back on
  a roster already starting four. Trades must improve both sides, or they get
  declined and the offer was wasted effort.
- **Ranks trades by value to you and flags the ones that compete.** Several good
  offers usually route through the same surplus player, and the moment he is
  traded the rest are dead. A flat list invites sending all of them and honouring
  whichever is accepted first, which is how you take the worst of three offers you
  could have had.
- **Names who leaves your lineup.** "He starts for you immediately" is half an
  answer; the other half is who he replaces, read from the lineup you actually
  set rather than the one the optimiser would pick.
- **Flags bye-week collisions a trade would create.** Marginal lineup value
  prices a season total, not the shape of the schedule behind it, so two
  starters sharing a bye is invisible to the gain. It is reported rather than
  priced: what a collision costs depends on bench depth at that position in that
  week, which is a different calculation and worth doing properly rather than
  approximating inside a trade search.
- **Explains each trade and drafts the message.** Every offer comes with why it
  helps you, why the other manager should say yes, and a sendable pitch. Each
  claim is derived from roster state both managers can already see — a message
  that misstates a roster the other person can check in ten seconds is worse
  than no message, and it poisons the next offer too.
- **Tests each gain against being wrong.** A marginal-value gain is a difference
  of two optimisations, so it jumps discretely when a slot assignment flips and a
  bare figure reads far more settled than it is. Every offer is quoted as a range
  and withheld when that range straddles zero.
- **Prices and discloses injuries.** Season-ending injuries are zeroed and
  excluded from trades in both directions; players back from surgery are
  discounted rather than written off; and any injured player in a proposal is
  named in the message, because the other manager sees the designation in his own
  app regardless.
- **Reports on any team in the league, not just yours.** A dropdown switches the
  whole page to another manager's seat: their optimal lineup, their waiver
  targets, their best trades with any of the other eleven. Useful as scouting --
  the offers another manager should want are the ones he is most likely to
  accept -- and as a check on the model, since a claim about someone else's
  roster is one you can verify in the app in ten seconds.
- **Shows the lineup you have set beside the one it recommends**, marks the slots
  that differ, and lists the bench. A change is measured by who enters and leaves
  the lineup, not by slot index -- the optimiser's choice between two equivalent
  RB slots is arbitrary, and reporting that as a change trains you to ignore the
  highlight.
- **Says where each number came from.** A `K` badge marks a projection the Kalshi
  market moved and by how much; everything else is Sleeper's own figure.
- **Publishes a dashboard** to GitHub Pages, refreshed by Actions.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
# Set league_id and roster_id in config.toml, then:
fl sync                 # league, players, rosters, schedule
fl advise               # lineup for the current week
fl moves                # waiver targets and trade offers
fl recap --week 3       # what happened, and what to learn
fl refresh              # one scheduled pass; writes docs/index.html
fl moves --team unc     # the same, from another manager's seat
```

`--team` takes a roster id or a name matched on a case-insensitive prefix, and
refuses an ambiguous one rather than guessing: reporting on the wrong manager's
roster looks like a bug in the model, not a typo in the command.

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

## How often it refreshes

The workflow asks every fifteen minutes. GitHub does not oblige: it drops
scheduled runs under load rather than queuing them, and an hourly cron measured
over its first two days fired **eleven times in forty-three hours** -- a run
every 4.3 hours on average, with one gap over six. Asking four times as often
does not make the scheduler punctual, it makes a missed slot cost minutes rather
than hours.

Rosters and lineups are re-read in full on every run, so a lineup change you make
in the app appears on the next one. The page carries the time it was built, which
is the only honest answer to how current it is. To force one immediately, run
the **Refresh advisory** workflow from the Actions tab.

Trades and waivers are rest-of-season decisions and only recompute when the next
kickoff is more than 24 hours away, so the slate itself does not get slowed down
by a search whose answer will not have changed.

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

## Injuries

The discriminator is **availability, not severity**, and it comes from the data.
Sleeper's *weekly* projections already price injury -- a player who is not
playing simply has no weekly record -- but its *rest-of-season* projections do
not. Sampled live, Michael Penix carried 143.2 rest-of-season points while out
for the year with a reconstructed ACL, and Tyreek Hill 93.7, while George Kittle,
who is actually playing after Achilles surgery, carried 169.3 -- a number that
already reflects his reduced role.

So the split is on whether a current-week projection exists:

| Regime | Meaning | ROS multiplier |
| --- | --- | --- |
| `HEALTHY` | no designation | 1.00 |
| `PLAYING_DIMINISHED`, structural | back from surgery, on the field | 0.90 |
| `PLAYING_DIMINISHED`, soft tissue | knock, still playing | 1.00 |
| `OUT_SHORT` | not playing, week to week | 0.40 |
| `OUT_LONG` | not playing, recovery outruns the season | 0.00 |

`injury_status` is never used alone. `injury_start_date` is 0 of 782 populated
upstream, so it carries no timing signal, and "Questionable" skews heavily toward
stars -- median ROS 95.3 against 38.7 for players with no designation, because
good players get reported. Haircutting the tag itself would dock Puka Nacua and
Ja'Marr Chase for undisclosed knocks while missing the players who are actually
gone.

Timelines come from published NFL cohort studies, not guesswork: Achilles
ruptures return 66.2% of players at a mean of
[10.9 months](https://pmc.ncbi.nlm.nih.gov/articles/PMC11682601/), ACL
reconstructions 61.8% at
[13.6 months](https://pubmed.ncbi.nlm.nih.gov/33553449/). Both exceed any
remaining schedule, which is why `OUT_LONG` is zero rather than merely
discounted. Decline after return is real but
[concentrated in the first season back](https://pubmed.ncbi.nlm.nih.gov/33218267/),
which is where a currently-playing post-surgery player sits by construction.

**The multipliers are literature-informed judgment, not fitted values.**
`playing_diminished_structural` is the weakest: Sleeper may already price some of
that decline, so the number is deliberately small and lives in `config.toml`.
It is the first thing `fl calibrate` should measure once weeks accumulate.

## When a gain is not real

One live proposal was quoted at +23.9, then +8.3, then -7.8 across successive
refreshes of the same sources -- same trade, same rosters, no new information.
Printed to one decimal place it read like an edge. It was noise, and it was about
to be sent to another manager.

The cause is structural rather than a bug. Marginal lineup value is the
difference between two optimisations, and an optimisation result is a step
function of its inputs: a projection moving a point or two flips which bench
player occupies a FLEX slot, and the gain jumps. So three things are checked
before an offer is shown.

**Stability.** Every projection is perturbed by a plausible relative error and
the gain recomputed a few hundred times. Each draw perturbs a player once and
uses that same perturbation on both sides of the subtraction, so what is measured
is the sensitivity of the *slot assignment*, not added noise. The tenth and
ninetieth percentiles are reported as a range, and an offer whose tenth
percentile is negative is not shown. The draws are seeded: advice that flickers
between hourly refreshes is worse than advice that is merely uncertain.

**Contingency.** Which player does the gain actually rest on? When a trade sends
two starters for one, the lineup hole is backfilled by promoting someone off the
bench, and the whole advantage can belong to that promotion rather than to the
players being exchanged. Each promoted player is zeroed and the lineup re-solved,
so the optimiser backfills with the real next alternative rather than an assumed
replacement level. In the live case the entire +8.3 came from promoting a tight
end returning from Achilles surgery; without him it was -47.5.

**Evidence.** A trade may not *lean on* a player carrying an injury designation
who has under three games this season. Snap share stabilises inside that window
where fantasy points, being touchdown-driven, do not. The gate is deliberately
one-sided: a player being **sent** is never tested this way, because selling an
injured asset is best done before the market prices the injury, and requiring
evidence there would suppress exactly the trades worth making first.

Thresholds live in `config.toml` under `[trades]`, `require_stable` included --
they are policy, not physics.

The draws are seeded, and seeding here is subtler than it looks. Fixing the
generator fixes the *sequence* of draws but not which player each one lands on,
and that mapping originally came from iterating a set of string ids -- an order
CPython salts per process. Identical inputs gave a p90 four points apart across
five runs, so the hourly dashboard quoted a different range every hour from
unchanged data. The ids are sorted now, and the test that guards it compares
separate processes under different `PYTHONHASHSEED` values, because the earlier
one compared two calls inside a single process and passed while the property it
named was false.

## Does it notice completed trades?

Yes, and not because it watches for them. Availability and every roster figure
are recomputed from a complete roster snapshot on each run, so any change --
trade, waiver claim, drop, someone else's pickup -- is reflected on the next
sync whether or not its transaction record was ever read. That was the point of
deriving availability rather than accumulating it.

The transaction feed on the dashboard is narrative on top of that: it says
*why* something moved, so a vanished waiver target or a void trade proposal has
an explanation rather than just disappearing.

Worth knowing about this league specifically: across the whole 2025 season it
recorded 227 transactions and **not one trade**. Every proposal this tool
generates would be breaking new ground, which is also why the pitch text matters.

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
.venv/bin/python -m pytest        # 142 tests, no network required
```

Tests run against captured real API responses in `fixtures/`, so they fail if an
undocumented endpoint changes shape — which is the point.
