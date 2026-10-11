"""THE LEMMA EXPLORER: what the planner is made of (Kaveh 2026-10-11: "put all the rules and lemmas somehow organized in a
lemma explorer where I can see what they are connected to, what they are doing ... take stock of what we've built and
see if I can explain it").

CATALOG: every rule of the planner (rust/analyst_core: plan.rs, threat.rs, rules.rs, paths.rs, pns.rs, ledger.rs,
waiting.rs, lib.rs) by layer -- what it does in plain words, the case that forced it, where it lives, the Lean results
that back it (checked against docs/LEAN_INDEX.md: a name not in the index fails the build), the rules it uses, its
status and its evidence (numbers as printed by the runs, from JOURNAL.md). The page adds the Lean index and a
connections graph.

    python -m catspace.incubator.analyst.lemma_explorer        # -> artifacts/experiments/lemma_explorer.html
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from catspace.io import paths

LAYERS = [
    ("facts", "Board facts", "What the board says without search: who can attack what, and what an exchange on a square is worth."),
    ("threats", "Threats and jobs", "What each side wins if the other does nothing, and what each piece is holding."),
    ("skip", "Lemmas that skip work", "Rules that decide a question without proving it, or decide which questions not to ask."),
    ("proofs", "Proofs", "How a claim is certified: searching every reply, and how a finished proof is reused."),
    ("decide", "Decisions", "How the move is chosen from what was proved."),
    ("memory", "Memory across moves", "What the planner carries from one move to the next."),
    ("off", "Switched off or replaced", "Built, measured, and kept out -- with the reason."),
]

CATALOG = [
    # ---- board facts
    dict(id="path_lemma", layer="facts", name="The path lemma",
         plain="After a move, a new attack comes only from the piece that arrived, or through a square the move vacated (plus en passant). So the moves that can attack a target are read off the board: arrivals, discoveries, batteries, pins.",
         why="Kaveh 2026-10-09: 'if the line from the king is blocked, there is no piece that can attack from there -- we should see that we have accounted for all the threats.'",
         where="paths.rs attack_paths", lean=["Ideas.PathLemma.path_lemma", "Ideas.PathLemma.no_new_attack"], uses=[], status="on",
         evidence="Audited against brute force: 836,694 moves, 0 missed, 0 over-claimed (2026-10-09)."),
    dict(id="attack_locality", layer="facts", name="Attacks are local to the touched squares",
         plain="The other direction of the path lemma: an attack is lost only if its piece left or a piece arrived on the line. Together: if no touched square is the attacker or between it and the target, the attack is unchanged.",
         why="Written for the incremental static exchange (2026-10-11), which recomputes only what a move can affect.",
         where="Lean only (the rule that uses it: incremental exchange)", lean=["Ideas.PathLemma.lost_path_lemma", "Ideas.PathLemma.attack_unchanged"], uses=["path_lemma"], status="on",
         evidence="Proved in Lean (lake build, 2026-10-11)."),
    dict(id="checking_moves", layer="facts", name="Checking moves from the paths",
         plain="Our checks are read off the paths onto their king (arrivals and discoveries; castling, en passant and promotions tried directly). No other move is tried for check.",
         why="Kaveh 2026-10-10: 'I looked at the squares the king has ... I reasoned only through the king.'",
         where="threat.rs checking_moves", lean=["Ideas.PathLemma.path_lemma"], uses=["path_lemma"], status="on",
         evidence="Census: 20,339 checking moves, 0 missed, 0 extra (2026-10-10)."),
    dict(id="exchange_law", layer="facts", name="The static exchange (the exchange law)",
         plain="What a capture sequence on one square is worth: captures alternate, cheapest attacker first, pieces behind a slider join as it leaves, either side may stop, a king may not recapture onto a square still attacked.",
         why="2026-10-10: two rooks against a pawn the king alone defended read as 'no threat' -- the old rule counted a defender, never how many. (The exchange law existed in Python and Lean since 2026-10-08; the Rust version is its mirror.)",
         where="rules.rs see / threats_of (mirror of exchange_law.py see_x)", lean=["Ideas.ExchangeLaw.seeXB", "Ideas.Exchange.cheapestXF_nonneg"], uses=[], status="on",
         evidence="Equal to the verified see_x on 3,028 of 3,028 attacked pieces (2026-10-10)."),
    dict(id="incremental_exchange", layer="facts", name="The incremental exchange",
         plain="After a move, the exchange is recomputed only on pieces whose lines, knight squares or neighbouring squares the move touched; every other value is carried over.",
         why="Kaveh 2026-10-10: 'make everything incremental -- look at what the opponent's move changed.'",
         where="rules.rs exchange_affected / threats_arr_inc", lean=["Ideas.PathLemma.attack_unchanged", "Ideas.FrameLemma.frame_unchanged"], uses=["exchange_law", "attack_locality"], status="on",
         evidence="175,502 of 175,502 move-side threat maps equal to a full recount (2026-10-11)."),
    # ---- threats and jobs
    dict(id="threat", layer="threats", name="The threat",
         plain="A threat is what a side wins if the other side does nothing, with a value: a piece (by the exchange), a promotion, a trapped piece, or the king (a forced mate by checks).",
         why="Kaveh 2026-10-10: 'a threat should have a value -- losing a piece, or becoming mate; these generic concepts in place of the very specific ones.'",
         where="threat.rs threats", lean=[], uses=["exchange_law", "promotion_threat", "trapped", "mate_by_checks"], status="on",
         evidence="Into the planner 2026-10-10: census frames' tactics 659 -> 664, Stockfish only-move tactics 241 -> 243, proofs a position 14.8 -> 13.1."),
    dict(id="mate_by_checks", layer="threats", name="Mate by checks",
         plain="A forced mate where every one of our moves is a check and every reply is searched -- cornering the king. (Limited to 3 checks: a numbered limit still to be replaced.)",
         why="Kaveh's Rc1 (2026-10-10): 'the rook lands on the back rank and gives mate' -- Rc8+ Bf8 Rxf8#, through a spite block.",
         where="threat.rs mate_by_checks", lean=["Ideas.checkmate_of_unmeetable_check", "Ideas.StalemateLemmas.checking_move_never_stalemates"], uses=["checking_moves"], status="on",
         evidence="Part of the threat set; the mate-first exit uses it."),
    dict(id="trapped", layer="threats", name="A trapped piece",
         plain="A piece of theirs that we do not win where it stands, but whose every move lands where we win it, with a move of ours onto it next. Cornering a piece, as mate is cornering the king.",
         why="Kaveh 2026-10-10: 'the lemma that covers mate is basically covering trapping.'",
         where="rules.rs trapped_unattacked (in threat.rs threats)", lean=[], uses=["exchange_law"], status="on",
         evidence="Restored Rg4 and b2 wins lost when the threat set lacked it (2026-10-10)."),
    dict(id="promotion_threat", layer="threats", name="A promotion threat",
         plain="A pawn that queens next move is a threat worth what the promotion keeps after their captures.",
         why="2026-10-10: Kd1 attacking the e1 knight while e7-e8=Q waited was refused as a single threat.",
         where="threat.rs threats", lean=[], uses=["exchange_law"], status="on", evidence="Kd1 found again (2026-10-10)."),
    dict(id="jobs", layer="threats", name="Jobs",
         plain="A piece's job is a threat it parries: lift it off the board and a threat appears (a piece it guards, a mate square it covers, a line it blocks, the king it shields).",
         why="Kaveh 2026-10-10: 'protecting against the threat should be generic, not specifically a piece.'",
         where="threat.rs jobs", lean=[], uses=["threat"], status="on", evidence="Used by overload and by the piece ledger."),
    dict(id="forcing_search", layer="threats", name="The forcing search",
         plain="What a side can FORCE: our moves are checks, captures and quiet threats worth 3+, every reply of theirs is searched, to the natural end of the forcing. A proof-number search; it also gives a value (a ladder 1, 2, 3, 5, 9) and the first move.",
         why="Kaveh 2026-10-10: 'why does there need to be a numeric check?' -- the old 'material by checks within 2' read its limit as 'no threat' and refused Be1.",
         where="threat.rs forcing_gain / forcing_value, pns.rs", lean=[], uses=["pns", "composed_standing", "incremental_exchange", "check_tempo"], status="on",
         evidence="As their reasoning's forced wins: Stockfish only-move tactics 284 -> 298, census 406 s -> 343 s (2026-10-11)."),
    dict(id="composed_standing", layer="threats", name="The composed standing",
         plain="Where the forcing search stops, a position stands at: the material already won, plus our best capture, minus their best capture elsewhere after ours (the intermezzo) -- charged only when the standing rests on our capture, and not when that capture gives check.",
         why="2026-10-11: Bb5+ c6 'won 9' -- Nxd1 takes the queen, then cxb5 takes the bishop back.",
         where="threat.rs Forcing::judge", lean=["Ideas.ExchangeLaw.exchangeLawComposed"], uses=["exchange_law", "check_tempo"], status="on",
         evidence="On 367 Stockfish tactics: 292 immediate wins (298 without it -- a bound), Bb5's false 9 gone (2026-10-11)."),
    dict(id="check_tempo", layer="threats", name="A check is a tempo",
         plain="While a side is in check, only captures that answer the check count as its gains: its other captures wait a move.",
         why="Kaveh 2026-10-10: 'any time the opponent can give us a check, they gain a tempo.' 2026-10-11: Rd2+ 'netted -6' for a capture Black could not make while in check.",
         where="plan.rs net_after; threat.rs Forcing::judge", lean=[], uses=[], status="on", evidence="Rd2+ chosen as Stockfish did (2026-10-11)."),
    dict(id="pns", layer="proofs", name="Proof-number search",
         plain="A generic AND/OR search that always expands the node closest to settling the question; children judged on sight cost nothing; a line prior weights each move (1.0 today -- the hook for a learned policy). A budget is only a safety net: hitting it is 'unknown'.",
         why="Kaveh 2026-10-10: 'take all of this proofing stuff into Rust'; 'a prior to decide how far to follow each line -- unity for now.'",
         where="pns.rs search", lean=[], uses=[], status="on", evidence="Ported from goal_search.py (Python, 2026-09-06); same answers as the first Rust version, 64.3 s vs 74.7 s on 200 positions."),
    # ---- skip work
    dict(id="single_threat", layer="skip", name="The single-threat lemma",
         plain="A move that attacks a piece wins nothing if that piece has a safe escape after which we have no threat at all -- no new static threat, nothing forced. Such moves are refused without proof; if the forcing search is unsure, the move is proved instead.",
         why="Replaced the lone-attack lemma's hand-made exceptions (2026-10-10); Kaveh: 'these generic concepts in place of the very specific ones.'",
         where="plan.rs (the seed loop)", lean=[], uses=["threat", "forcing_search", "exchange_law"], status="on",
         evidence="Random sample: Stockfish disagrees on 23 of 13,006 refusals (0.2%) (2026-10-10)."),
    dict(id="stop_rule", layer="skip", name="The stop rule",
         plain="Targets are tried by value (the king first); stop when what is already won strictly beats the next target's bound -- the bound includes a promotion capture.",
         why="2026-10-10/11: 'r on g8 is worth 5' cut hxg8=Q+ (rook and a queen); an equal-value cut hid cxd4 and Kxg2.",
         where="plan.rs (targets loop)", lean=[], uses=[], status="on", evidence="hxg8=Q, cxd4, Kxg2 found (2026-10-11)."),
    dict(id="pinned_to_what", layer="skip", name="Pinned to what",
         plain="A piece of ours pinned against our king zone: lift it and run their reasoning; if they would win more, its moves off the line are refused without proof (a knight's every move leaves the line).",
         why="Kaveh 2026-10-09: 'the knight on d7 is pinned because the queen can give a mate on d8 if the knight hops.'",
         where="plan.rs (frozen)", lean=["Ideas.GeoLemmas.knight_leaves_every_line"], uses=["path_lemma"], status="on", evidence="Built 2026-10-09."),
    dict(id="mate_first", layer="skip", name="A mate ends the analysis",
         plain="A forced mate by checks, or a proven mate against the king, ends the analysis at once -- nothing else is computed.",
         why="Kaveh 2026-10-10: 'anytime we find a path to mate, we should play that ... no more search needed.'",
         where="plan.rs plan_inner", lean=[], uses=["mate_by_checks", "prove"], status="on", evidence="Built 2026-10-10."),
    # ---- proofs
    dict(id="prove", layer="proofs", name="A proof",
         plain="A focused alpha-beta search that a move wins at least its goal against EVERY reply; the best move at every node is stored with it (the proof can be replayed later).",
         why="The planner's certificate since 2026-10-09.", where="waiting.rs prove_n, lib.rs prove_pos", lean=["Ideas.localCheck_sound"], uses=[], status="on",
         evidence="Stalemate scored as a draw since 2026-10-10 (all 300 stalemating moves now fail their certificates)."),
    dict(id="frame_value", layer="proofs", name="A frame value",
         plain="When a proof fails at its goal, the value search says what the move certainly nets in its frame: a lower bound that cannot see outside the frame.",
         why="2026-10-10: proofs were all-or-nothing at one goal -- Rxa2 'fails' at 9 but still wins 4, so their wins were invisible.",
         where="lib.rs certified_value", lean=[], uses=[], status="on", evidence="Decisions losing 150+ cp 15.3% -> 12.8% with it (2026-10-10)."),
    dict(id="frame_independence", layer="proofs", name="Frame independence (the pass)",
         plain="Inside a proof, a reply that touches no square bearing on the frame (on a line through a frame square, a knight's move from one, or the square itself) is one 'pass': the frame's pieces and every attack onto them are unchanged. An attack on the king and an attack on the queen are the same kind of input, differing in value only.",
         why="Kaveh 2026-10-11: a proof should read its frame and the opponent's other frames from the index, never the whole board; this is the theorem that makes reusing such a proof sound. The search's leaf still reads the whole board, so today's certificates are exact in the frame and 'modulo the rest of the board' outside it.",
         where="lib.rs pursue_val_p (zone, pass), rules.rs exchange_affected", lean=["Ideas.FrameLemma.frame_unchanged", "Ideas.FrameLemma.attack_reach", "Ideas.FrameLemma.between_reach"],
         uses=["attack_locality"], status="on", evidence="Proved 2026-10-11 (153 theorems, 0 open). The reuse that rests on it is the next build."),
    dict(id="no_chase", layer="proofs", name="No chase without a point (the harbour)",
         plain="A move that attacks a piece fails to win it -- no search needed -- when the piece can step (without capture or check) to a harbour where nothing is collectable by captures, no line of our checks wins what is missing, and no move of ours renews the attack on it or makes threats that win the goal against one reply (the fork law, every kind of threat incl. mate). Never applied to checks, promotions, pawn moves (the pawn chase) or king targets.",
         why="Kaveh 2026-10-11: 'if the opponent can protect the piece, chasing it becomes pointless ... unless you're chasing it with a pawn'. Its stoppers were mined from the questions the search held: every one banked material, was forcing, made a second threat, chased with a pawn, or was a zugzwang.",
         where="lemma.rs chase_lemma (flag CHASE, off)", lean=[], uses=["exchange_law", "threat", "mate_by_checks"], status="off",
         evidence="Held-out 40 games: 11,807 questions decided, 0 held proofs skipped; planner decisions identical on 3,226 positions; nodes -8%, time neutral (2026-10-11)."),
    dict(id="two_proofs", layer="proofs", name="Full proofs and frame values are trusted differently",
         plain="A full proof (every reply searched) is a lower bound: a capture's net is the larger of it and the board's count. A frame value is capped: for a capture the smaller; for a quiet move, minus their best win after it.",
         why="2026-10-10/11: reading every capture by the board lost 75 real wins (Nxe5 uncovering Be2xg4); trusting frame values let fxe4 and Qc2 through.",
         where="plan.rs net_after", lean=[], uses=["prove", "frame_value", "net"], status="on", evidence="Frames' tactics 630 -> 671 when restored (2026-10-10)."),
    # ---- decisions
    dict(id="net", layer="decide", name="The net of a move",
         plain="Every candidate move gets one value: what it takes now minus the opponent's best proved or forced win after it (their gains on our pieces floored by the exchange law, a check's tempo respected) -- or its proof, per the two-proofs rule.",
         why="Kaveh 2026-10-10: 'with a logic-based play there should be no blunders ... whatever the blunder guard is doing should be part of the main planner's job.'",
         where="plan.rs net_after", lean=[], uses=["forcing_search", "exchange_law", "check_tempo", "two_proofs"], status="on",
         evidence="The blunder audit then found 2 'certain' blunders in 7,876 positions, neither confirmed by Stockfish (2026-10-10)."),
    dict(id="wins_ranking", layer="decide", name="Wins ranked by net, with siblings",
         plain="The immediate wins are ranked by their net; every other capture on the same square joins the ranking; a win whose net is not positive is no win.",
         why="2026-10-10: Raxa7 'won 1' while Nf4 forked the queen; the recapture Bxe5 was never generated.",
         where="plan.rs", lean=[], uses=["net"], status="on", evidence="Stockfish only-move tactics 246 -> 272 with the ranking (2026-10-10)."),
    dict(id="defence", layer="decide", name="Defence by net",
         plain="Against their threat: defences read off their winning move's geometry, ranked by net; if the best still loses, every other move is scanned (a cheap gate first, their reasoning on the best five, the rest when unsettled).",
         why="2026-10-10: the first defence found was played -- Nb4 stopped Nxc6 and lost the knight to cxb4 (337 of 613 warned decisions).",
         where="plan.rs (defence phase)", lean=[], uses=["net"], status="on", evidence="Decisions losing 150+ cp 15.3% -> 9.1% over the day (2026-10-10)."),
    dict(id="tie_break", layer="decide", name="The tie-break",
         plain="Between moves of equal net: material taken now first, then piece activity (the moved piece's safe squares plus twice the pieces it attacks), minus 2 for each move of theirs that would attack it with a cheaper piece.",
         why="Kaveh 2026-10-10: 'whenever there is a tie, prioritize piece activity, until a strategic planning layer steps in.'",
         where="plan.rs tie_score", lean=[], uses=[], status="on", evidence="Agrees with Stockfish on 12 of 15 tie cases from three walked games (2026-10-10)."),
    dict(id="arming", layer="decide", name="Arming",
         plain="When nothing wins now: a preparing move that leaves a target at most one escape, so that a seed wins next -- refused if the preparing move itself nets below zero.",
         why="Kaveh 2026-10-09: 'how do we cover all three?'; 2026-10-10: Bh6 armed Qxg7 and hung to Qxh6.", where="plan.rs (arming phase)", lean=[], uses=["net"], status="on",
         evidence="Armed decisions 576 -> 388 in the census when losing preparations were refused (2026-10-10)."),
    dict(id="promotion_lemma", layer="decide", name="The unstoppable passer",
         plain="The rule of the square, generalized: king distance, a knight-distance table, the bishop's colour; checked against races, material lost by the move, and a stalemate guard.",
         why="Kaveh 2026-10-10: 'cheaply find endgame solutions instead of a deep search or a table lookup.'",
         where="plan.rs unstoppable", lean=["Ideas.GeoLemmas.bishop_keeps_colour", "Ideas.Passer.unstoppablePasser_sound'"], uses=[], status="on",
         evidence="Stockfish audit until sound: 107 of 109 claims winning (2026-10-10)."),
    dict(id="overload", layer="decide", name="Overload",
         plain="A piece with two jobs: take on the first job's square; after its recapture there, the second job's threat stands -- checked by playing the two moves.",
         why="Kaveh 2026-10-10: 'a piece needed in one frame and in another has an opportunity cost.'",
         where="plan.rs overloads", lean=[], uses=["jobs"], status="on", evidence="31 new certified wins in the census (2026-10-10)."),
    # ---- memory
    dict(id="frame_memory", layer="memory", name="The frame memory",
         plain="Every proof question is kept with its board, answer and stored proof; asked again: the same board reuses it, a changed board replays a proof that held or re-proves one that failed.",
         why="Kaveh 2026-10-09: 'each out is a value -- memorize it, unless something changed.'", where="plan.rs Memo / mprove", lean=[], uses=["prove"], status="on",
         evidence="Reusing a failure by its support was audited unsafe and is off (2026-10-09)."),
    dict(id="ledger", layer="memory", name="The piece ledger and the nucleus",
         plain="Each piece's jobs and the pins, diffed after every move: their freed pieces are asked what they can do; ours are searched first for double threats.",
         why="Kaveh 2026-10-10: 'the pin is suddenly removed ... pony, where can you hop? It cannot do anything, so my queen is free.'",
         where="ledger.rs, plan.rs (nucleus)", lean=[], uses=["jobs"], status="on", evidence="Immediate wins 2,799 -> 2,806, none lost (2026-10-10)."),
    dict(id="mate_playout", layer="memory", name="Playing out a proven mate",
         plain="A proven mate is kept with its proof; on later moves the opponent's reply is read off the board and our answer off the proof -- no search.",
         why="Kaveh 2026-10-10: 'you've found it, you've proven it, so you can just use it.'", where="plan.rs PlanState / Solution", lean=[], uses=["prove", "mate_first"], status="on",
         evidence="6 of 6 mates delivered against Stockfish's defence, 11 moves from the stored proof with no search (2026-10-10)."),
    # ---- off / replaced
    dict(id="lone_attack", layer="off", name="The lone-attack lemma (replaced)",
         plain="Refused a move attacking a piece that can step away, unless check, a second target, a new path onto the king, or a capture.",
         why="The first lemma under review; its hand-made exceptions missed jobs and mates.", where="replaced by the single-threat lemma", lean=[], uses=[], status="replaced",
         evidence="Skipped 28 Stockfish only-move solutions; the single-threat lemma skips 13 (2026-10-10)."),
    dict(id="frame_pool", layer="off", name="The waiting-frames pool (off)",
         plain="All frames of both sides kept across moves with their missing conditions; arms, hands, defuses.",
         why="Wired in 2026-10-10 rather than rebuilt.", where="waiting.rs FramePool (set_pool)", lean=[], uses=[], status="off",
         evidence="6x the cost; its 1-pawn wins lose 27 cp median, its armed decisions lose 150+ in 35% (2026-10-10). Its seeds (2+) are kept."),
    dict(id="blunder_audit", layer="off", name="The blunder audit (off)",
         plain="After the decision, their reasoning once; a certain blunder (another move nets 2+ more) replaced.",
         why="Became a pass once decisions were made by net.", where="plan.rs AFTER_CHECK / AVOID", lean=[], uses=[], status="off",
         evidence="Found 2 'certain' blunders in 7,876 positions, neither confirmed by Stockfish (2026-10-10)."),
    dict(id="change_gate", layer="off", name="The change flag as a gate (off)",
         plain="Certify a move's net only if the move creates a new threat for them (the change flag).",
         why="Kaveh 2026-10-10: 'see how this change changes the state of another piece.'", where="plan.rs CHANGE_GATE", lean=[], uses=[], status="off",
         evidence="Catches 84% of their new wins; as a gate, decisions losing 150+ rose 15.8% -> 18.3% (2026-10-10)."),
]

EXPLAIN = [
    ("Read the board", "The path lemma says which moves can attack what; the exchange law says what each capture is worth; after a move both are updated only where the move touched.", ["path_lemma", "exchange_law", "incremental_exchange"]),
    ("Mate first", "If a forced mate exists (by checks, or proven against the king), play it and stop -- and keep playing it out from the proof.", ["mate_first", "mate_by_checks", "mate_playout"]),
    ("Their threat", "On the null move: what would they win if we passed -- by the same reasoning, including what they can force.", ["threat", "forcing_search"]),
    ("Our candidate wins", "Targets by value; only moves on a path onto a target; the single-threat lemma refuses lone attacks; each candidate is proved, or valued in its frame.", ["path_lemma", "single_threat", "stop_rule", "prove", "frame_value"]),
    ("Value every move by its net", "What it takes now minus their best proved or forced win after it -- full proofs as lower bounds, frame values capped.", ["net", "two_proofs", "check_tempo", "composed_standing"]),
    ("Decide", "The best net among the wins; else defend by net (scanning every move if still losing); else arm; ties by material taken, then piece activity.", ["wins_ranking", "defence", "arming", "tie_break"]),
    ("Remember", "Proofs, jobs and pins carry to the next move; what changed seeds the next search.", ["frame_memory", "ledger"]),
]


def build() -> Path:
    idx = {d["full"]: d for d in json.load(open(paths.derived("lean_index.json")))}
    ids = {c["id"] for c in CATALOG}
    for c in CATALOG:
        for n in c["lean"]:
            if n not in idx:
                raise SystemExit(f"{c['id']}: Lean name {n} is not in the index")
        for u in c["uses"]:
            if u not in ids:
                raise SystemExit(f"{c['id']}: uses unknown {u}")
    used_by = {c["id"]: [d["id"] for d in CATALOG if c["id"] in d["uses"]] for c in CATALOG}
    lean = [dict(name=d["full"], kind=d["kind"], file=d["file"].split("/")[-1], line=d["line"], doc=d["doc"][:300], used_by=d["used_by"])
            for d in idx.values() if d["kind"] in ("theorem", "lemma")]
    data = dict(layers=LAYERS, catalog=CATALOG, used_by=used_by, explain=EXPLAIN, lean=lean, built=time.strftime("%Y-%m-%d %H:%M"))
    # the connections graph (mermaid): layers as subgraphs, "uses" as edges
    lines = ["flowchart LR"]
    for lid, lname, _ in LAYERS:
        members = [c for c in CATALOG if c["layer"] == lid]
        if not members:
            continue
        lines.append(f'  subgraph {lid}["{lname}"]')
        for c in members:
            lines.append(f'    {c["id"]}["{c["name"].replace(chr(34), "")}"]')
        lines.append("  end")
    for c in CATALOG:
        for u in c["uses"]:
            lines.append(f"  {u} --> {c['id']}")
    data["mermaid"] = "\n".join(lines)
    html = (Path(__file__).with_name("lemma_explorer.html")).read_text().replace("/*__DATA__*/null", json.dumps(data))
    out = Path(paths.experiment("lemma_explorer.html"))
    out.write_text(html)
    print(f"OBS {len(CATALOG)} rules in {len(LAYERS)} layers, {sum(len(c['uses']) for c in CATALOG)} connections, "
          f"{sum(1 for c in CATALOG if c['lean'])} backed by Lean; {len(lean)} Lean theorems -> {out}")
    return out


if __name__ == "__main__":
    build()
