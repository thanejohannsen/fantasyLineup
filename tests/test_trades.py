"""Waiver and trade engine tests."""

from __future__ import annotations

import dataclasses

import pytest

from fantasylineup.engine.lineup import optimize_lineup
from fantasylineup.engine.trades import TradeProposal, explain_trade, find_trades
from fantasylineup.engine.waivers import rank_waiver_targets
from tests.test_lineup import p

SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]


def test_trade_exploits_complementary_surplus():
    """The case the whole engine exists for.

    One roster is deep at running back and thin at receiver; the other is the
    mirror image. Swapping makes both lineups better at once, which raw
    projected points cannot see -- the players exchanged are identical in value.

    Note how much surplus it takes. With two FLEX slots a fourth running back
    still starts, so imbalance only becomes costly at the *fifth*, when the
    surplus finally cannot be absorbed and weak receivers are forced into the
    dedicated WR slots. This is why genuine trades are scarcer in a
    two-FLEX league than intuition suggests.
    """
    rb_heavy = [
        p("qb", "QB", 18.0),
        p("rb1", "RB", 17.0),
        p("rb2", "RB", 16.0),
        p("rb3", "RB", 15.0),
        p("rb4", "RB", 14.0),
        p("rb5", "RB", 13.0),
        p("wr_bad1", "WR", 4.0),
        p("wr_bad2", "WR", 3.0),
        p("te", "TE", 8.0),
    ]
    wr_heavy = [
        p("qb2", "QB", 18.0),
        p("wr1", "WR", 17.0),
        p("wr2", "WR", 16.0),
        p("wr3", "WR", 15.0),
        p("wr4", "WR", 14.0),
        p("wr5", "WR", 13.0),
        p("rb_bad1", "RB", 4.0),
        p("rb_bad2", "RB", 3.0),
        p("te2", "TE", 8.0),
    ]

    proposals = find_trades(rb_heavy, wr_heavy, SLOTS, partner_roster_id=7, partner_name="them")

    assert proposals, "a mutually beneficial trade exists and must be found"
    best = proposals[0]
    assert best.my_gain > 0 and best.their_gain > 0
    # We should be sending a running back and receiving a receiver.
    assert any(pl.position == "RB" for pl in best.give)
    assert any(pl.position == "WR" for pl in best.get)


def test_no_trade_when_both_rosters_are_balanced():
    """Two similar teams have nothing to gain, and nothing should be invented."""
    roster_a = [
        p("qb", "QB", 18.0),
        p("rb1", "RB", 14.0),
        p("rb2", "RB", 12.0),
        p("wr1", "WR", 14.0),
        p("wr2", "WR", 12.0),
        p("te", "TE", 9.0),
        p("rb3", "RB", 8.0),
        p("wr3", "WR", 8.0),
    ]
    roster_b = [
        p(f"b_{x.sleeper_id}", x.position, x.points - 0.1) for x in roster_a
    ]
    assert find_trades(roster_a, roster_b, SLOTS, 7, "them") == []


def test_one_sided_trades_are_rejected():
    """A deal the other manager loses on will be declined, so it is not proposed."""
    mine = [
        p("qb", "QB", 18.0),
        p("rb1", "RB", 3.0),
        p("rb2", "RB", 3.0),
        p("wr1", "WR", 3.0),
        p("wr2", "WR", 3.0),
        p("te", "TE", 3.0),
        p("f1", "WR", 2.0),
        p("f2", "RB", 2.0),
    ]
    theirs = [
        p("qb2", "QB", 18.0),
        p("srb1", "RB", 20.0),
        p("srb2", "RB", 19.0),
        p("swr1", "WR", 19.0),
        p("swr2", "WR", 18.0),
        p("ste", "TE", 15.0),
        p("sf1", "WR", 14.0),
        p("sf2", "RB", 13.0),
    ]
    for proposal in find_trades(mine, theirs, SLOTS, 7, "them"):
        assert proposal.their_gain > 0


def test_top_player_is_protected():
    """Managers do not trade their anchor, whatever the maths says."""
    mine = [
        p("stud", "RB", 30.0),
        p("qb", "QB", 18.0),
        p("rb2", "RB", 12.0),
        p("rb3", "RB", 11.0),
        p("wr1", "WR", 5.0),
        p("wr2", "WR", 4.0),
        p("te", "TE", 8.0),
        p("sp", "WR", 3.0),
    ]
    theirs = [
        p("twr1", "WR", 18.0),
        p("twr2", "WR", 17.0),
        p("twr3", "WR", 16.0),
        p("tqb", "QB", 18.0),
        p("trb", "RB", 4.0),
        p("trb2", "RB", 3.0),
        p("tte", "TE", 8.0),
        p("tsp", "WR", 12.0),
    ]
    for proposal in find_trades(mine, theirs, SLOTS, 7, "them"):
        assert "stud" not in {pl.sleeper_id for pl in proposal.give}


def test_fairness_band_blocks_a_fleece():
    """An offer that looks lopsided on rankings will not be taken seriously."""
    mine = [p("junk", "RB", 1.0), p("qb", "QB", 18.0), p("rb", "RB", 10.0), p("wr", "WR", 10.0)]
    theirs = [p("gem", "WR", 25.0), p("tqb", "QB", 18.0), p("trb", "RB", 10.0)]
    proposals = find_trades(mine, theirs, ["QB", "RB", "WR"], 7, "them", fairness_band=0.2)
    for proposal in proposals:
        assert abs(proposal.raw_given - proposal.raw_received) / max(
            proposal.raw_given, proposal.raw_received
        ) <= 0.2


# ------------------------------------------------------------------- waivers


def test_waiver_ranks_by_marginal_not_raw_points():
    """A fifth running back adds nothing; a receiver fills a real hole.

    Both candidates project identically, so any ranking by projected points
    would rate them equal.
    """
    roster = [
        p("qb", "QB", 18.0),
        p("rb1", "RB", 17.0),
        p("rb2", "RB", 16.0),
        p("rb3", "RB", 15.0),
        p("rb4", "RB", 14.0),
        p("wr1", "WR", 12.0),
        p("wr2", "WR", 2.0),
        p("te", "TE", 8.0),
    ]
    targets = rank_waiver_targets(
        roster, [p("fa_rb", "RB", 13.0), p("fa_wr", "WR", 13.0)], SLOTS
    )
    assert targets[0].player.sleeper_id == "fa_wr"
    assert targets[0].gain > 0


def test_full_roster_pairs_a_pickup_with_a_drop():
    """On a full roster the gain must be net of what has to be cut."""
    roster = [
        p("qb", "QB", 18.0),
        p("rb1", "RB", 14.0),
        p("rb2", "RB", 13.0),
        p("wr1", "WR", 14.0),
        p("wr2", "WR", 13.0),
        p("te", "TE", 9.0),
        p("flex1", "WR", 10.0),
        p("deadweight", "TE", 0.5),
    ]
    targets = rank_waiver_targets(
        roster, [p("upgrade", "WR", 16.0)], SLOTS, roster_limit=len(roster)
    )
    assert targets
    assert targets[0].drop is not None
    assert targets[0].drop.sleeper_id == "deadweight"
    assert targets[0].gain < targets[0].rostered_gain or targets[0].drop.points == 0.5


def test_worthless_pickups_are_not_suggested():
    roster = [
        p("qb", "QB", 18.0),
        p("rb1", "RB", 17.0),
        p("rb2", "RB", 16.0),
        p("wr1", "WR", 15.0),
        p("wr2", "WR", 14.0),
        p("te", "TE", 12.0),
        p("f1", "RB", 11.0),
        p("f2", "WR", 10.0),
    ]
    assert rank_waiver_targets(roster, [p("scrub", "WR", 1.0)], SLOTS) == []


def test_empty_inputs_are_handled():
    assert find_trades([], [p("a", "RB", 5.0)], SLOTS, 7, "them") == []
    assert rank_waiver_targets([], [], SLOTS) == []


def test_trade_shape_labels():
    mine = [
        p("qb", "QB", 18.0),
        p("rb1", "RB", 16.0),
        p("rb2", "RB", 15.0),
        p("rb3", "RB", 14.0),
        p("wr1", "WR", 3.0),
        p("wr2", "WR", 2.0),
        p("te", "TE", 8.0),
        p("x", "RB", 12.0),
    ]
    theirs = [
        p("twr1", "WR", 16.0),
        p("twr2", "WR", 15.0),
        p("twr3", "WR", 14.0),
        p("tqb", "QB", 18.0),
        p("trb1", "RB", 3.0),
        p("trb2", "RB", 2.0),
        p("tte", "TE", 8.0),
        p("ty", "WR", 12.0),
    ]
    for proposal in find_trades(mine, theirs, SLOTS, 7, "them"):
        assert proposal.shape in {"1-for-1", "1-for-2", "2-for-1"}
        assert proposal.joint_gain == pytest.approx(proposal.my_gain + proposal.their_gain)


def test_near_miss_bar_is_the_displaceable_starter():
    """A tight end cannot take a kicker's slot, so the kicker is not the bar.

    This roster's lowest-scoring starter is a kicker. Reporting him as the bar a
    free-agent tight end failed to clear is misleading -- the number that
    matters is the weakest starter the newcomer could actually replace.
    """
    from fantasylineup.engine.waivers import explain_no_targets

    roster = [
        p("qb", "QB", 300.0),
        p("rb1", "RB", 250.0),
        p("rb2", "RB", 200.0),
        p("wr1", "WR", 240.0),
        p("wr2", "WR", 210.0),
        p("te", "TE", 230.0),
        p("flex1", "WR", 180.0),
        p("flex2", "RB", 169.0),
        p("k", "K", 76.0),
        p("dst", "DEF", 110.0),
    ]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"]
    weakest, near = explain_no_targets(roster, [p("fa_te", "TE", 138.0)], slots)

    assert weakest is not None
    assert weakest.position != "K"
    assert weakest.points == pytest.approx(169.0)
    assert [c.sleeper_id for c in near] == ["fa_te"]


def test_shortlist_is_position_aware(tmp_path):
    """A global top-N shortlist fills with quarterbacks who cannot help.

    Season-long quarterback totals dominate every other position, so ranking the
    whole free-agent pool by projected points and taking the top slice yields a
    list of backup quarterbacks in a league that starts one. Measured on the
    real league that produced a shortlist where every candidate added exactly
    zero and no running back was ever evaluated.
    """
    from fantasylineup.db import connect, init_db
    from fantasylineup.engine.waivers import shortlist_candidates

    conn = connect(":memory:")
    init_db(conn)
    rows, projections = [], {}
    for i in range(30):
        rows.append((f"qb{i}", "QB"))
        projections[f"qb{i}"] = {"mean": 300 - i}
    for i in range(30):
        rows.append((f"rb{i}", "RB"))
        projections[f"rb{i}"] = {"mean": 200 - i}
    conn.executemany(
        """INSERT INTO players (sleeper_id, full_name, position, fantasy_positions,
                                active, updated_at)
           VALUES (?,?,?,'[]',1,'now')""",
        [(pid, pid, pos) for pid, pos in rows],
    )
    conn.commit()

    shortlist = shortlist_candidates(conn, {pid for pid, _ in rows}, projections)
    assert any(pid.startswith("rb") for pid in shortlist), "running backs must survive"
    assert sum(pid.startswith("qb") for pid in shortlist) <= 8
    conn.close()


def _explained(mine, theirs, slots):
    from fantasylineup.engine.trades import explain_trade

    proposals = find_trades(mine, theirs, slots, 7, "Unc Show")
    assert proposals
    return proposals[0], explain_trade(proposals[0], mine, theirs, slots)


def test_rationale_uses_each_player_own_position():
    """A package can span two positions, and one count cannot describe both.

    Reporting "Kittle and Montgomery start for them (they roster 2 at TE)"
    attaches the tight end's depth to a running back. It is a claim the other
    manager can check in ten seconds, and getting it wrong costs the offer.
    """
    mine = [
        p("qb", "QB", 300.0),
        p("te1", "TE", 240.0),
        p("te2", "TE", 169.0),
        p("rb1", "RB", 206.0),
        p("rb2", "RB", 180.0),
        p("rb3", "RB", 175.0),
        p("wr1", "WR", 250.0),
        p("wr2", "WR", 210.0),
        p("wr3", "WR", 130.0),
    ]
    theirs = [
        p("tqb", "QB", 300.0),
        p("thenry", "RB", 247.0),
        p("trb2", "RB", 120.0),
        p("trb3", "RB", 110.0),
        p("trb4", "RB", 100.0),
        p("tte", "TE", 90.0),
        p("twr1", "WR", 240.0),
        p("twr2", "WR", 200.0),
    ]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    proposal, rationale = _explained(mine, theirs, slots)

    # Every position claim must match the player it is attached to.
    for player in proposal.give:
        if player.name in rationale.their_angle:
            segment = rationale.their_angle.split(player.name, 1)[1].split(";")[0]
            assert f"at {player.position}" in segment, (
                f"{player.name} ({player.position}) described with the wrong position"
            )


def test_pitch_has_no_filler():
    """The message must read like a league-mate, not a form letter.

    Greetings and offers to "adjust the pieces" are the tells that make an
    otherwise sound proposal look automated, and get it ignored.
    """
    mine = [
        p("Alder", "QB", 300.0),
        p("Bramble", "TE", 240.0),
        p("Cedar", "TE", 169.0),
        p("Dogwood", "RB", 206.0),
        p("Elm", "WR", 250.0),
        p("Fir", "WR", 210.0),
        p("Gorse", "WR", 130.0),
        p("Hazel", "WR", 125.0),
    ]
    theirs = [
        p("Juniper", "QB", 300.0),
        p("Kapok", "RB", 247.0),
        p("Larch", "RB", 120.0),
        p("Maple", "TE", 90.0),
        p("Nutmeg", "WR", 240.0),
        p("Olive", "WR", 200.0),
        p("Poplar", "WR", 190.0),
        p("Quince", "WR", 180.0),
    ]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    _, rationale = _explained(mine, theirs, slots)
    lowered = rationale.pitch.lower()

    for tell in ("hey ", "interested in a trade", "happy to adjust", "let me know",
                 "helps us both", "cheers", "thanks!"):
        assert tell not in lowered, f"filler phrase in pitch: {tell!r}"

    # It should open with the offer itself.
    assert rationale.pitch.splitlines()[0].endswith("?")


def test_pitch_quantifies_their_side():
    """The reason to reply is the number, so it has to be in the message."""
    mine = [
        p("Alder", "QB", 300.0),
        p("Bramble", "TE", 240.0),
        p("Cedar", "TE", 169.0),
        p("Dogwood", "RB", 206.0),
        p("Elm", "WR", 250.0),
        p("Fir", "WR", 210.0),
        p("Gorse", "WR", 130.0),
        p("Hazel", "WR", 125.0),
    ]
    theirs = [
        p("Juniper", "QB", 300.0),
        p("Kapok", "RB", 247.0),
        p("Larch", "RB", 120.0),
        p("Maple", "TE", 90.0),
        p("Nutmeg", "WR", 240.0),
        p("Olive", "WR", 200.0),
        p("Poplar", "WR", 190.0),
        p("Quince", "WR", 180.0),
    ]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    proposal, rationale = _explained(mine, theirs, slots)
    # The pitch is hard-wrapped, so phrases straddle line breaks.
    flat = " ".join(rationale.pitch.split())
    assert f"+{proposal.their_gain:.0f} points" in flat
    assert "rest of season" in flat


def test_pitch_is_wrapped_for_pasting():
    """Unwrapped prose pastes into messaging apps as one unreadable line."""
    mine = [
        p("qb", "QB", 300.0),
        p("te1", "TE", 240.0),
        p("te2", "TE", 169.0),
        p("rb1", "RB", 206.0),
        p("wr1", "WR", 250.0),
        p("wr2", "WR", 210.0),
        p("wr3", "WR", 130.0),
        p("wr4", "WR", 125.0),
    ]
    theirs = [
        p("tqb", "QB", 300.0),
        p("thenry", "RB", 247.0),
        p("trb2", "RB", 120.0),
        p("tte", "TE", 90.0),
        p("twr1", "WR", 240.0),
        p("twr2", "WR", 200.0),
        p("twr3", "WR", 190.0),
        p("twr4", "WR", 180.0),
    ]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    _, rationale = _explained(mine, theirs, slots)

    assert all(len(line) <= 72 for line in rationale.pitch.split("\n"))
    assert rationale.why and rationale.their_angle


def test_pitch_never_invents_a_player():
    """Every name in the message must exist on one of the two rosters.

    The message deliberately names people outside the deal -- the player the
    incoming man displaces, and the one blocking him on my bench -- because
    those are the checkable, persuasive details. What it must never do is
    invent someone, which would make an otherwise sound proposal look careless.

    Names here are deliberately distinct rather than sharing prefixes, so a
    substring cannot satisfy the check by accident.
    """
    mine = [
        p("Alder", "QB", 300.0),
        p("Bramble", "TE", 240.0),
        p("Cedar", "TE", 169.0),
        p("Dogwood", "RB", 206.0),
        p("Elm", "WR", 250.0),
        p("Fir", "WR", 210.0),
        p("Gorse", "WR", 130.0),
        p("Hazel", "WR", 125.0),
    ]
    theirs = [
        p("Juniper", "QB", 300.0),
        p("Kapok", "RB", 247.0),
        p("Larch", "RB", 120.0),
        p("Maple", "TE", 90.0),
        p("Nutmeg", "WR", 240.0),
        p("Olive", "WR", 200.0),
        p("Poplar", "WR", 190.0),
        p("Quince", "WR", 180.0),
    ]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    proposal, rationale = _explained(mine, theirs, slots)

    import re

    known = {x.name for x in mine} | {x.name for x in theirs}
    # Split on word boundaries so trailing punctuation does not turn a real
    # name into an unrecognised token.
    words = set(re.findall(r"[A-Za-z']+", rationale.pitch))
    sentence_starters = {"By", "I", "On", "Worth"}
    invented = {w for w in words if w[0].isupper() and w not in known} - sentence_starters
    assert not invented, f"invented names: {invented}"

    # The deal's own players must of course appear.
    for player in (*proposal.give, *proposal.get):
        assert player.name in rationale.pitch


def test_pitch_names_who_the_incoming_player_displaces():
    """The strongest argument is concrete: who he beats out in *their* lineup.

    That claim requires knowing the lineup he actually set, so the real starters
    are supplied here as the live commands do. Their lineup is full, so the
    incoming tight end genuinely pushes someone onto the bench.
    """
    from fantasylineup.engine.trades import explain_trade

    incoming = p("Bramble", "TE", 240.0)
    outgoing = p("Kapok", "RB", 247.0)
    weak_te = p("Maple", "TE", 90.0)

    their_set = [
        p("Juniper", "QB", 300.0),
        outgoing,
        p("Larch", "RB", 120.0),
        p("Nutmeg", "WR", 240.0),
        p("Olive", "WR", 200.0),
        weak_te,
        p("Poplar", "WR", 190.0),
        p("Quince", "WR", 180.0),
    ]
    theirs = list(their_set)
    mine = [incoming, p("Alder", "QB", 300.0), p("Elm", "WR", 250.0)]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]

    proposal = TradeProposal(7, "them", (incoming,), (outgoing,), 9.0, 3.0)
    rationale = explain_trade(proposal, mine, theirs, slots, their_starters=their_set)

    flat = " ".join(rationale.pitch.split())
    assert "Bramble would start for you" in flat
    assert "Maple moves to your bench" in flat


def test_displacement_claim_is_omitted_when_no_lineup_is_known():
    """With nothing set, decline to name a benched player rather than guess.

    Every bug in this area came from substituting our optimal lineup for a
    manager's real one. Where no real lineup exists there is nothing to read, so
    the specific claim is dropped instead of being invented.
    """
    from fantasylineup.engine.trades import explain_trade

    incoming = p("Bramble", "TE", 240.0)
    outgoing = p("Kapok", "RB", 247.0)
    mine = [incoming, p("Alder", "QB", 300.0), p("Elm", "WR", 250.0)]
    theirs = [p("Juniper", "QB", 300.0), outgoing, p("Maple", "TE", 90.0)]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]

    proposal = TradeProposal(7, "them", (incoming,), (outgoing,), 9.0, 3.0)
    rationale = explain_trade(proposal, mine, theirs, slots, their_starters=None)
    assert "moves to your bench" not in rationale.pitch


def test_no_claim_contradicts_their_real_lineup():
    """The invariant that both previous bugs violated.

    Testing the specific sentence that broke would pass while the next
    equivalent claim breaks -- which is exactly what happened: a fix to two
    facts left a third reading our optimal lineup for the other manager, and it
    shipped a note saying a player he starts was not in his lineup.

    So this asserts the rule itself, on a scenario where the two sources
    deliberately disagree: our optimal for him benches a player he actually
    starts. Every membership claim must side with his lineup, not ours.
    """
    from fantasylineup.engine.trades import explain_trade
    from fantasylineup.engine.lineup import optimize_lineup

    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    contrarian = p("Sutton", "WR", 159.0)  # he starts him; our optimal does not
    better_bench = p("Kincaid", "TE", 163.0)
    outgoing = p("Waddle", "WR", 221.0)

    theirs = [
        p("Hurts", "QB", 310.0),
        p("Jeanty", "RB", 233.0),
        p("Henderson", "RB", 171.0),
        p("Nacua", "WR", 312.0),
        p("Rice", "WR", 229.0),
        outgoing,
        better_bench,
        contrarian,
        p("Spare", "RB", 168.0),
    ]
    # His actual lineup starts Sutton over the spare back; ours would not.
    their_set = [
        p("Hurts", "QB", 310.0),
        p("Jeanty", "RB", 233.0),
        p("Henderson", "RB", 171.0),
        p("Nacua", "WR", 312.0),
        p("Rice", "WR", 229.0),
        better_bench,
        outgoing,
        contrarian,
    ]
    our_optimal = {
        pl.sleeper_id for pl in optimize_lineup(theirs, slots).assignments.values()
    }
    real = {pl.sleeper_id for pl in their_set}
    assert our_optimal != real, "fixture must make the two sources disagree"
    assert "Sutton" in real and "Sutton" not in our_optimal

    incoming = p("Kittle", "TE", 152.0)
    mine = [incoming, p("McBride", "TE", 235.0), p("Bijan", "RB", 325.0)]
    proposal = TradeProposal(9, "them", (incoming,), (contrarian,), 6.0, 3.0)
    rationale = explain_trade(proposal, mine, theirs, slots, their_starters=their_set)

    text = " ".join((rationale.their_angle + " " + rationale.pitch).split())

    after = {
        pl.sleeper_id
        for pl in optimize_lineup(
            [x for x in their_set if x.sleeper_id != contrarian.sleeper_id] + [incoming],
            slots,
        ).assignments.values()
    }

    for player in their_set:
        # A player he starts must never be described as absent from his lineup.
        assert f"{player.name} is not in their lineup" not in text, player.name
        # And anyone said to be benched must genuinely leave it after the trade.
        if f"{player.name} moves to your bench" in text:
            assert player.sleeper_id not in after, player.name


# ------------------------------------------------- who the trade displaces


def _my_side(slots):
    """A roster whose real lineup deliberately differs from our optimal.

    Two tight ends are rostered; the manager starts the weaker one in a FLEX and
    benches a receiver our optimal would start. Any claim read from our optimal
    instead of his lineup names the wrong player.
    """
    qb = p("Daniels", "QB", 275.0)
    te1 = p("McBride", "TE", 200.0)
    te2 = p("Kittle", "TE", 136.0)
    benched_wr = p("Golden", "WR", 150.0)
    roster = [
        qb,
        p("Bijan", "RB", 295.0),
        p("Montgomery", "RB", 206.0),
        p("London", "WR", 198.0),
        p("Washington", "WR", 161.0),
        te1,
        p("Dowdle", "RB", 146.0),
        te2,
        benched_wr,
    ]
    # He starts the second tight end in the FLEX and leaves the receiver out.
    real = [qb, roster[1], roster[2], roster[3], roster[4], te1, roster[6], te2]
    return roster, real, qb, te2, benched_wr


def test_the_rationale_names_who_leaves_my_lineup():
    """"He starts for you immediately" is half an answer without the other half.

    Reported live: an offer said the incoming players start immediately and
    never said who they replace, which is the first thing a manager needs in
    order to judge it.
    """
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    roster, real, qb, te2, _ = _my_side(slots)
    incoming = (p("Meyers", "WR", 169.0), p("Purdy", "QB", 264.0))
    proposal = TradeProposal(3, "them", (qb,), incoming, 22.0, 3.0)

    rationale = explain_trade(
        proposal, roster, [p("Filler", "RB", 100.0)], slots, my_starters=real
    )
    why = " ".join(rationale.why.split())

    assert "Kittle" in why, "the FLEX he actually starts is who gets pushed out"
    assert "out of your lineup" in why


def test_displacement_is_read_from_the_real_lineup_not_our_optimal():
    """The mistake this guards against was made by hand while investigating.

    Our optimal for this roster starts the benched receiver; the manager does
    not. Naming him is naming someone the manager can see is already benched --
    the same class of error as the two lineup-claim bugs before it.
    """
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    roster, real, qb, te2, benched_wr = _my_side(slots)

    our_optimal = {
        x.sleeper_id for x in optimize_lineup(roster, slots).assignments.values()
    }
    assert benched_wr.sleeper_id in our_optimal, "fixture must make the sources disagree"
    assert benched_wr.sleeper_id not in {x.sleeper_id for x in real}

    proposal = TradeProposal(
        3, "them", (qb,), (p("Meyers", "WR", 169.0), p("Purdy", "QB", 264.0)), 22.0, 3.0
    )
    rationale = explain_trade(
        proposal, roster, [p("Filler", "RB", 100.0)], slots, my_starters=real
    )
    assert benched_wr.name not in rationale.why


def test_verbs_agree_with_the_number_of_players():
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    roster, real, qb, _, _ = _my_side(slots)
    partner = [p("Filler", "RB", 100.0)]

    two = explain_trade(
        TradeProposal(
            3, "them", (qb,), (p("Meyers", "WR", 169.0), p("Purdy", "QB", 264.0)), 22.0, 3.0
        ),
        roster, partner, slots, my_starters=real,
    )
    one = explain_trade(
        TradeProposal(3, "them", (qb,), (p("Purdy", "QB", 300.0),), 22.0, 3.0),
        roster, partner, slots, my_starters=real,
    )
    assert "start for you" in two.why and "starts for you" not in two.why
    assert "starts for you" in one.why


# --------------------------------------------------------- bye collisions


def test_a_bye_clash_the_trade_creates_is_reported():
    """Marginal lineup value prices a season total, not the schedule shape.

    Two starters sharing a bye is a week that cannot be covered, and the gain
    cannot see it. Reported live: a receiver acquired alongside one already
    rostered on the same team.
    """
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    same_bye = p("Washington", "WR", 161.0)
    roster = [
        p("Daniels", "QB", 275.0),
        p("Bijan", "RB", 295.0),
        p("Montgomery", "RB", 206.0),
        p("London", "WR", 198.0),
        dataclasses.replace(same_bye, bye_week=7),
        p("McBride", "TE", 200.0),
        p("Dowdle", "RB", 146.0),
        p("Kittle", "TE", 136.0),
    ]
    incoming = dataclasses.replace(p("Meyers", "WR", 169.0), bye_week=7)
    proposal = TradeProposal(3, "them", (roster[7],), (incoming,), 20.0, 3.0)

    rationale = explain_trade(proposal, roster, [p("Filler", "RB", 100.0)], slots)
    text = " ".join(rationale.caveats)
    assert "week 7 bye" in text
    assert "Meyers" in text and "Washington" in text


def test_a_bye_clash_the_roster_already_had_is_not_reported():
    """Only a clash this trade creates is news; the rest would bury it."""
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
    roster = [
        p("Daniels", "QB", 275.0),
        p("Bijan", "RB", 295.0),
        p("Montgomery", "RB", 206.0),
        dataclasses.replace(p("London", "WR", 198.0), bye_week=11),
        dataclasses.replace(p("Washington", "WR", 161.0), bye_week=11),
        p("McBride", "TE", 200.0),
        p("Dowdle", "RB", 146.0),
        p("Kittle", "TE", 136.0),
    ]
    incoming = dataclasses.replace(p("Meyers", "WR", 169.0), bye_week=5)
    proposal = TradeProposal(3, "them", (roster[7],), (incoming,), 20.0, 3.0)

    rationale = explain_trade(proposal, roster, [p("Filler", "RB", 100.0)], slots)
    assert not any("week 11" in c for c in rationale.caveats)
