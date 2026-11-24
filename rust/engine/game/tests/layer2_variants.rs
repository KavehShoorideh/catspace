//! PLAN 79.35 PARTS E AND F: LAYER 2 SAYS FOUR MORE GAMES ON THE SAME LAYER 1 -- TWO WITH ANOTHER START (CHESS960, HORDE) AND TWO WITH OTHER RULES OF MOVING AND ENDING (ATOMIC, ANTICHESS)
//! (SYSTEM_SPEC 11 "Layer 2 says variants, with another start and with other rules, on the same layer 1"; ledger rows 41, 40; pieces: rules, representation).
//! This file names the games on purpose (a test file may). The reference is DATA: tests/fixtures/<game>.fixture, written by catspace/incubator/analyst/discovery/layer2_fixtures_print.py from the library `chess` (a reference that shares
//! no code with the engine): the positions, the allowed moves, the ends, the facts, and the counts of positions after a few plies. Here the kernel's mode of each game is read against it, the rules of each game by made positions whose
//! expected sets are WRITTEN BY HAND, and the relations layer 2 writes (bear, may_move) against sets a program of this file makes by walking the board by hand (never by the rules).
//! The description of each game and its certificate are tested in layer2_variants_described.rs and by `certify`.
use cs_game::graph::Static;
use cs_game::relations::PartRelations;
use cs_game::kernel::{chess960_fen, KernelGame};
use cs_game::placed::Placed;
use cs_game::rules::{Ruled, StateText};
use cs_game::starts::Starts;
use cs_game::{Fact, Game};
use std::collections::{BTreeMap, BTreeSet};
mod variants_common;
use variants_common::{fixture, kernel, rules_root, GAMES};

type S = <KernelGame as Game>::State;


fn names_of(g: &KernelGame, s: &S) -> Vec<String> {
    let mut v: Vec<String> = g.moves(s).iter().map(|m| g.move_name(s, m)).collect();
    v.sort();
    v
}

/// The units of a state as sets of text: "side kind place".
fn units_of(g: &KernelGame, s: &S) -> BTreeSet<String> {
    let (pn, kn) = (g.place_names(), g.kind_names());
    g.cells(s).into_iter().map(|(o, k, p)| format!("{} {} {}", g.roles()[o], kn[k as usize], pn[p as usize])).collect()
}

fn units_of_facts(f: &BTreeSet<Fact>) -> BTreeSet<String> {
    f.iter().filter(|x| x.relation == "cell").map(|x| format!("{} {} {}{}", x.args[2], x.args[3], x.args[0], x.args[1])).collect()
}

fn count<G: Game>(g: &G, s: &G::State, depth: usize, k: usize, counts: &mut Vec<u64>) {
    if depth == 0 { return; }
    let ms = g.moves(s);
    counts[k] += ms.len() as u64;
    for m in &ms { let t = g.apply(s, m); count(g, &t, depth - 1, k + 1, counts); }
}

fn at(g: &KernelGame, text: &str) -> S { g.read_state_text(text).unwrap_or_else(|e| panic!("{text}: {e}")) }

fn counts_of(g: &KernelGame, s: &S) -> BTreeMap<(String, String), usize> {
    let kn = g.kind_names();
    let mut m = BTreeMap::new();
    for (o, k, _p) in g.cells(s) { *m.entry((g.roles()[o].clone(), kn[k as usize].clone())).or_default() += 1; }
    m
}

// ------------------------------------------------------------------------------------------------------------------------------ the four fixtures exist, and their games differ as the file says

#[test]
fn the_four_fixtures_exist_say_how_their_game_differs_and_reach_every_way_the_game_ends() {
    let want: BTreeMap<&str, &str> = [("chess960", "start"), ("horde", "both"), ("atomic", "rules"), ("antichess", "rules")].into_iter().collect();
    for game in GAMES {
        let fx = fixture(game);
        assert_eq!(fx.differs, want[game], "{game}: how the game differs from the standard one");
        let g = kernel(game);
        let mut reached = BTreeSet::new();
        for p in &fx.pos {
            let s = at(&g, &p.text);
            assert_eq!(g.outcome(&s), p.end.as_ref().map(|e| e.1.clone()), "{game}: the end at {}", p.label);
            if let Some(e) = &p.end { assert!(g.moves(&s).is_empty() && g.mover(&s).is_none(), "{game}: an ended position has no move and no mover ({})", p.label); reached.insert(e.0.clone()); }
        }
        for st in &fx.steps {
            if let Some(e) = &st.end { reached.insert(e.0.clone()); }
        }
        assert_eq!(reached, fx.ends.iter().cloned().collect::<BTreeSet<_>>(), "{game}: every way the game ends is reached, and no other");
        println!("OBS 79.35 E ends {game}: {} positions and {} steps in the file; ends reached {:?}", fx.pos.len(), fx.steps.len(), reached);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ the start: each unit's place, the counts, the side to move, the count of allowed moves

fn standard_counts() -> BTreeMap<(String, String), usize> {
    let mut m = BTreeMap::new();
    for side in ["white", "black"] {
        for (k, n) in [("pawn", 8), ("rook", 2), ("knight", 2), ("bishop", 2), ("queen", 1), ("king", 1)] { m.insert((side.to_string(), k.to_string()), n); }
    }
    m
}

#[test]
fn the_start_of_each_game_is_layer_2s_and_equals_the_fixture_in_every_unit_the_counts_the_side_and_the_moves() {
    // the starts the game has: units' places equal the fixture's facts, counts a side written by hand, the side to move, the count of allowed moves
    let fx = fixture("chess960");
    let base = kernel("chess960");
    assert_eq!(base.start_count(), 960);
    let mut seen = 0;
    for (n, label) in &fx.starts {
        let g = base.with_start(*n).unwrap();
        let s = g.initial();
        let p = fx.pos.iter().find(|p| &p.label == label).unwrap();
        assert_eq!(units_of(&g, &s), units_of_facts(&p.facts), "chess960 start {n}: the units' places");
        assert_eq!(counts_of(&g, &s), standard_counts(), "chess960 start {n}: eight, two, two, two, one, one a side");
        assert_eq!((g.mover(&s), p.side.as_str()), (Some(0), "white"));
        // by hand: sixteen steps of the smallest kind, two steps of each unit of the kind that leaps (one when it stands in a corner column), nothing of the others (shut in); the two-unit move is not yet there
        let corner_leapers = units_of(&g, &s).iter().filter(|u| u.starts_with("white knight a1") || u.starts_with("white knight h1")).count();
        assert_eq!(g.moves(&s).len(), 20 - corner_leapers, "chess960 start {n}: first moves by hand");
        assert_eq!(names_of(&g, &s), p.moves, "chess960 start {n}: the allowed moves equal the file's");
        seen += 1;
    }
    assert!(seen >= 5 && base.with_start(960).is_err());
    // the starts are of the game: one game's start 0 and the standard start are two arrangements of one set of units, the leader next to a corner in start 0
    let s0 = base.with_start(0).unwrap().initial();
    assert!(units_of(&base, &s0).contains("white king g1") && units_of(&base, &s0).contains("white rook h1"), "start 0: the leader next to the unit in the corner");
    let s959 = base.with_start(959).unwrap().initial();
    assert!(units_of(&base, &s959).contains("white king b1") && units_of(&base, &s959).contains("white rook a1"), "start 959: the leader next to the unit in the other corner");
    // the other three games have one start
    for game in ["horde", "atomic", "antichess"] {
        let g = kernel(game);
        let fx = fixture(game);
        assert_eq!((g.start_count(), g.with_start(1).is_err()), (1, true));
        let p = fx.pos.iter().find(|p| p.label == "start").unwrap();
        let s = g.initial();
        assert_eq!(units_of(&g, &s), units_of_facts(&p.facts), "{game}: the units' places");
        assert_eq!(names_of(&g, &s), p.moves, "{game}: the allowed moves of the start");
        assert_eq!(g.mover(&s), Some(0));
    }
    let h = kernel("horde");
    let s = h.initial();
    let c = counts_of(&h, &s);
    let side = |o: &str, k: &str| c.get(&(o.to_string(), k.to_string())).copied().unwrap_or(0);
    assert_eq!((side("white", "pawn"), side("white", "king"), side("white", "queen"), side("white", "rook"), side("white", "knight"), side("white", "bishop")), (36, 0, 0, 0, 0, 0), "horde: 36 of the smallest kind and no leader");
    assert_eq!((side("black", "pawn"), side("black", "king"), side("black", "queen"), side("black", "rook"), side("black", "knight"), side("black", "bishop")), (8, 1, 1, 2, 2, 2));
    assert_eq!(h.moves(&s).len(), 8, "horde: eight first moves: only the fourth row can step");
    for game in ["atomic", "antichess"] {
        let g = kernel(game);
        let s = g.initial();
        assert_eq!(counts_of(&g, &s), standard_counts(), "{game}: the start of chess");
        assert_eq!(g.moves(&s).len(), 20);
        let castles = |g: &KernelGame, s: &S| { let t = g.state_text(s); t.split(' ').nth(2).unwrap().to_string() };
        assert_eq!(castles(&g, &s), if game == "atomic" { "KQkq" } else { "-" }, "{game}: the two-unit move of the leader is there at the start only in atomic");
    }
    println!("OBS 79.35 E start: chess960 {seen} starts of 960 equal the file's (units, counts, white to move, 20 moves; starts 0 and 959 have the leader next to a corner); horde 36 + 0 against 8 1 1 2 2 2, 8 first moves; atomic and antichess the start of chess");
}

#[test]
fn the_numbering_of_the_960_starts_equals_the_library_s_for_every_number() {
    let fx = fixture("chess960");
    assert_eq!(fx.scharnagl.len(), 960);
    for (n, row) in &fx.scharnagl {
        let ours = chess960_fen(*n as u32);
        assert_eq!(ours.split('/').next().unwrap(), row, "start {n}: the first row");
    }
    println!("OBS 79.35 E start: all 960 numbered starts equal the library's own numbering (their first rows)");
}

// ------------------------------------------------------------------------------------------------------------------------------ every position of the fixture against the kernel

#[test]
fn every_position_of_every_fixture_has_the_kernels_allowed_moves_side_and_end_and_every_step_changes_the_state_as_the_file_says() {
    for game in GAMES {
        let fx = fixture(game);
        let g = kernel(game);
        let (mut positions, mut moves_seen, mut steps) = (0usize, 0usize, 0usize);
        for p in &fx.pos {
            let s = at(&g, &p.text);
            assert_eq!(names_of(&g, &s), p.moves, "{game}: allowed moves at {}", p.label);
            for n in &p.moves { assert!(g.find_move(&s, n).is_some(), "{game}: {n} is found by its name at {}", p.label); }
            assert_eq!(g.mover(&s).map(|m| g.roles()[m].clone()), if p.end.is_none() { Some(p.side.clone()) } else { None }, "{game}: the side to move at {}", p.label);
            assert_eq!(units_of(&g, &s), units_of_facts(&p.facts), "{game}: the units at {}", p.label);
            positions += 1;
            moves_seen += p.moves.len();
        }
        for st in &fx.steps {
            let parent = fx.pos.iter().find(|p| p.label == st.parent).unwrap();
            let s = at(&g, &parent.text);
            let after = g.apply(&s, &g.find_move(&s, &st.name).unwrap());
            let mut want = parent.facts.clone();
            for f in &st.removed { assert!(want.remove(f), "{game}: a removed fact was there"); }
            for f in &st.added { want.insert(f.clone()); }
            assert_eq!(units_of(&g, &after), units_of_facts(&want), "{game}: the units after {} at {}", st.name, st.parent);
            assert_eq!(g.outcome(&after), st.end.as_ref().map(|e| e.1.clone()), "{game}: the end after {} at {}", st.name, st.parent);
            steps += 1;
        }
        println!("OBS 79.35 E positions {game}: {positions} positions ({moves_seen} allowed moves) equal the library's moves, side, end and units; {steps} moves change the units as the file says");
    }
}

#[test]
fn the_counts_of_positions_after_a_few_plies_equal_the_library_s() {
    for game in GAMES {
        let fx = fixture(game);
        let g = kernel(game);
        let mut nodes = 0u64;
        for (label, text, want) in &fx.perft {
            let s = at(&g, text);
            let mut got = vec![0u64; want.len() + 1];
            count(&g, &s, want.len(), 1, &mut got);
            assert_eq!(&got[1..], &want[..], "{game}: {label}: positions reached after each ply");
            nodes += want.iter().sum::<u64>();
        }
        println!("OBS 79.35 E counts {game}: equal the library's for {} starts to depths {:?} ({nodes} positions counted)", fx.perft.len(), fx.perft.iter().map(|p| p.2.len()).collect::<Vec<_>>());
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ the rules of each game, by made positions; the expected sets are written by hand

fn set(v: &[&str]) -> Vec<String> { let mut v: Vec<String> = v.iter().map(|s| s.to_string()).collect(); v.sort(); v }

fn play(g: &KernelGame, s: &S, name: &str) -> S { g.apply(s, &g.find_move(s, name).unwrap_or_else(|| panic!("{name} is not allowed"))) }

fn text_of(g: &KernelGame, s: &S) -> String { g.state_text(s) }

#[test]
fn chess960_the_two_unit_move_of_the_leader_from_unlike_starts_on_both_wings_and_where_it_is_refused() {
    let g = kernel("chess960");
    // the move is written as the leader onto its own unit of the corner side; the leader and that unit land on the third and fourth, or the seventh and sixth, file whatever they stood on
    let cases: [(&str, &str, &[&str]); 3] = [
        ("4k3/8/8/8/8/8/8/RK5R w KQ - 0 1", "leader b1, units a1 and h1", &["white king c1", "white rook d1", "white king g1", "white rook f1"]),
        ("4k3/8/8/8/8/8/8/R1K4R w KQ - 0 1", "leader c1: long side the leader stays", &["white king c1", "white rook d1", "white king g1", "white rook f1"]),
        ("4k3/8/8/8/8/8/8/R5KR w KQ - 0 1", "leader g1: short side the leader stays", &["white king c1", "white rook d1", "white king g1", "white rook f1"]),
    ];
    for (text, why, lands) in cases {
        let s = at(&g, text);
        let long = format!("{}{}", if text.contains("RK5R") { "b1" } else if text.contains("R1K4R") { "c1" } else { "g1" }, "a1");
        let short = format!("{}{}", &long[..2], "h1");
        let names = names_of(&g, &s);
        assert!(names.contains(&long) && names.contains(&short), "{why}: both two-unit moves are allowed: {names:?}");
        let after_long = units_of(&g, &play(&g, &s, &long));
        let after_short = units_of(&g, &play(&g, &s, &short));
        assert!(lands[..2].iter().all(|u| after_long.contains(*u)) && !after_long.contains("white rook a1"), "{why}: long side {after_long:?}");
        assert!(lands[2..].iter().all(|u| after_short.contains(*u)) && !after_short.contains("white rook h1"), "{why}: short side {after_short:?}");
        // after either the rights of that side are spent
        assert_eq!(text_of(&g, &play(&g, &s, &long)).split(' ').nth(2), Some("-"), "{why}: both rights are gone after one two-unit move");
    }
    // the unit next to the corner on the long side with the leader on b1: only one unit, one wing
    let s = at(&g, "4k3/8/8/8/8/8/8/RK6 w Q - 0 1");
    assert!(names_of(&g, &s).contains(&"b1a1".to_string()) && !names_of(&g, &s).iter().any(|n| n == "b1h1"));
    // refused: a path square struck; the leader struck; a unit in the way; a unit that shielded the leader on the row; no right kept
    let s = at(&g, "4k3/8/8/8/8/8/5r2/R3K2R w KQ - 0 1");
    let n = names_of(&g, &s);
    assert!(n.contains(&"e1a1".to_string()) && !n.contains(&"e1h1".to_string()), "the short side crosses a struck place: refused; the long side is allowed");
    let n = names_of(&g, &at(&g, "4k3/8/8/8/4r3/8/8/R3K2R w KQ - 0 1"));
    assert!(!n.contains(&"e1a1".to_string()) && !n.contains(&"e1h1".to_string()), "the leader is struck: neither side");
    let n = names_of(&g, &at(&g, "4k3/8/8/8/8/8/8/RKN4R w KQ - 0 1"));
    assert!(!n.contains(&"b1h1".to_string()) && !n.contains(&"b1a1".to_string()), "a unit stands where the leader must land (c1) and on its way (c1): both sides refused");
    let n = names_of(&g, &at(&g, "4k3/8/8/8/8/8/8/qRK5 w B - 0 1"));
    assert!(!n.contains(&"c1b1".to_string()), "the unit on b1 shields the leader from the attacker on a1: moving it away would leave the leader struck: refused");
    let n = names_of(&g, &at(&g, "4k3/8/8/8/8/8/8/R3K2R w - - 0 1"));
    assert!(!n.contains(&"e1a1".to_string()) && !n.contains(&"e1h1".to_string()), "no right kept: no two-unit move");
    println!("OBS 79.35 E rules chess960: the two-unit move from three unlike starts on both wings lands as written by hand; refused when a path place is struck, when the leader is struck, when a unit is in the way, when a unit shields the leader on the row, and with no right");
}

#[test]
fn horde_the_first_step_of_the_smallest_kind_from_the_first_and_second_rows_and_the_end_with_no_unit_left() {
    let g = kernel("horde");
    assert_eq!(names_of(&g, &at(&g, "4k3/8/8/8/8/8/8/P7 w - - 0 1")), set(&["a1a2", "a1a3"]), "the first row: one step or two");
    assert_eq!(names_of(&g, &at(&g, "4k3/8/8/8/8/8/P7/1P6 w - - 0 1")), set(&["a2a3", "a2a4", "b1b2", "b1b3"]), "second and first row side by side");
    assert_eq!(names_of(&g, &at(&g, "4k3/8/8/8/8/8/P7/P7 w - - 0 1")), set(&["a2a3", "a2a4"]), "the unit behind is shut in");
    assert_eq!(names_of(&g, &at(&g, "4k3/8/8/8/8/8/8/PP6 w - - 0 1")), set(&["a1a2", "a1a3", "b1b2", "b1b3"]));
    // the passing square is set only by a two-row step from the second row
    let s = at(&g, "4k3/8/8/8/8/8/8/P7 w - - 0 1");
    assert_eq!(text_of(&g, &play(&g, &s, "a1a3")).split(' ').nth(3), Some("-"), "a two-row step from the first row leaves no passing square");
    let s = at(&g, "4k3/8/8/8/8/8/P7/8 w - - 0 1");
    assert_eq!(text_of(&g, &play(&g, &s, "a2a4")).split(' ').nth(3), Some("a3"), "a two-row step from the second row leaves the square it passed");
    // the end: the side with no unit left loses; the other side's leader stays
    let s = at(&g, "4k3/8/8/8/8/8/8/8 w - - 0 1");
    assert_eq!((g.outcome(&s), g.moves(&s).len(), g.mover(&s)), (Some(vec![0, 100]), 0, None), "white has none: black wins, white to move");
    let s = at(&g, "4k3/8/8/8/8/8/8/8 b - - 0 1");
    assert_eq!(g.outcome(&s), Some(vec![0, 100]), "white has none: black wins, black to move");
    // a taking of the last one ends it
    let s = at(&g, "4k3/8/8/8/8/8/1p6/P7 b - - 0 1");
    assert_eq!(names_of(&g, &s), set(&["b2a1q", "b2a1r", "b2a1b", "b2a1n", "b2b1q", "b2b1r", "b2b1b", "b2b1n", "e8d7", "e8d8", "e8e7", "e8f7", "e8f8"]), "black may take the unit on the first row (and become another kind there)");
    let s = play(&g, &s, "b2a1q");
    assert_eq!(g.outcome(&s), Some(vec![0, 100]), "the last unit of the horde taken: the horde side has lost");
    // no leader on the horde's side: no refused move for it; a side with moves and none left is no end, and none at all is a draw
    let s = at(&g, "4k3/8/8/8/8/p7/P7/8 w - - 0 1");
    assert_eq!((g.outcome(&s), g.moves(&s).len()), (Some(vec![50, 50]), 0), "the horde blocked: no move left is a draw, as in the standard game");
    println!("OBS 79.35 E rules horde: the first step of the smallest kind from the first and second rows as written by hand, the passing square only from the second, the end with no unit left (both sides to move), the last unit taken, the blocked horde");
}

#[test]
fn atomic_a_taking_removes_the_taker_the_taken_and_the_units_around_except_the_smallest_kind() {
    let g = kernel("atomic");
    let s = at(&g, "4k3/8/3b1p2/4p3/3nP3/5N2/8/4K3 w - - 0 1");
    let after = play(&g, &s, "f3e5");
    assert_eq!(units_of(&g, &after), ["white king e1", "white pawn e4", "black king e8", "black pawn f6"].iter().map(|s| s.to_string()).collect::<BTreeSet<_>>(),
               "the taker, the taken, the bishop and the knight around are gone; the two units of the smallest kind around stay");
    assert_eq!(g.captured(&s, &g.find_move(&s, "f3e5").unwrap()).len(), 3, "the other side lost the unit taken and the two around");
    // a leader does not take; a leader may not stand beside another's taking; the leaders side by side void a threat
    assert_eq!(names_of(&g, &at(&g, "4k3/8/8/8/8/8/3p4/3K4 w - - 0 1")), set(&["d1c2", "d1e2"]), "the leader does not take the unit beside it, and the two places the unit bears on are shut");
    let s = at(&g, "8/p7/8/8/8/3k4/r2K4/8 w - - 0 1");
    assert_eq!(names_of(&g, &s), set(&["d2c1", "d2d1", "d2e1", "d2c2", "d2e2", "d2c3", "d2e3"]), "leaders side by side: the row the other side's unit strikes does not threaten the leader, so it may step along it");
    // taking beside the own leader is refused (the own leader would be removed)
    let s = at(&g, "4k3/8/8/8/8/2K5/3p4/1N6 w - - 0 1");
    let n = names_of(&g, &s);
    assert!(n.contains(&"b1a3".to_string()) && !n.contains(&"b1d2".to_string()), "the unit that would take beside the own leader may not: the leader would be removed with it");
    assert_eq!(n.iter().filter(|m| m.starts_with("c3")).cloned().collect::<Vec<_>>(), set(&["c3b2", "c3b3", "c3b4", "c3c2", "c3c4", "c3d3", "c3d4"]), "the leader steps to the seven places that are free; it does not take the unit beside it");
    // the end: a leader removed ends the game, for the side whose leader stays
    let s = at(&g, "3rk3/8/8/8/8/8/8/3QK3 w - - 0 1");
    assert!(names_of(&g, &s).contains(&"d1d8".to_string()));
    let after = play(&g, &s, "d1d8");
    assert_eq!((g.outcome(&after), g.moves(&after).len(), g.mover(&after)), (Some(vec![100, 0]), 0, None), "the other leader stood beside the taking: it is removed and the game is won");
    println!("OBS 79.35 E rules atomic: a taking removes the taker, the taken and the units around except the smallest kind (written by hand); the leader does not take; the leaders side by side void the threat along a row; the end by a leader removed");
}

#[test]
fn antichess_a_taking_is_compulsory_the_leader_is_an_ordinary_unit_the_smallest_kind_may_become_a_leader_and_both_ways_of_winning() {
    let g = kernel("antichess");
    assert_eq!(names_of(&g, &at(&g, "4k3/8/8/2p1p3/3P4/8/8/4K3 w - - 0 1")), set(&["d4c5", "d4e5"]), "a taking is compulsory: only the two takings");
    // the leader is an ordinary unit: it takes and is taken, no refusal for a struck leader
    let s = at(&g, "k7/7p/8/8/8/8/8/R6K w - - 0 1");
    assert_eq!(names_of(&g, &s), set(&["a1a8"]), "the only taking is of the leader");
    let after = play(&g, &s, "a1a8");
    assert_eq!((g.outcome(&after), g.mover(&after)), (None, Some(1)), "a leader taken and play goes on");
    let s = at(&g, "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
    assert!(names_of(&g, &s).contains(&"e1e2".to_string()) && names_of(&g, &s).len() == 1, "the leader struck is no check; the compulsory taking is its only move");
    // no two-unit move of the leader
    let n = names_of(&g, &at(&g, "4k3/8/8/8/8/8/8/R3K2R w - - 0 1"));
    assert!(!n.contains(&"e1g1".to_string()) && !n.contains(&"e1c1".to_string()) && !n.contains(&"e1a1".to_string()) && !n.contains(&"e1h1".to_string()));
    // the smallest kind becomes a leader as well as the four other kinds at the last row
    assert_eq!(names_of(&g, &at(&g, "8/P7/8/8/8/8/p7/8 w - - 0 1")), set(&["a7a8q", "a7a8r", "a7a8b", "a7a8n", "a7a8k"]), "five kinds to become, the leader among them");
    // both ways of winning: no unit left; no move left
    let s = at(&g, "8/8/8/8/8/8/8/4K3 b - - 0 1");
    assert_eq!((g.outcome(&s), g.moves(&s).len()), (Some(vec![0, 100]), 0), "no unit left to the side to move: it wins");
    let s = at(&g, "8/8/8/8/8/p7/P7/8 w - - 0 1");
    assert_eq!((g.outcome(&s), g.moves(&s).len()), (Some(vec![100, 0]), 0), "no move left to the side to move: it wins");
    let s = at(&g, "8/8/8/8/8/p7/P7/8 b - - 0 1");
    assert_eq!(g.outcome(&s), Some(vec![0, 100]), "the same for the other side");
    let s = at(&g, "4k3/8/8/8/8/8/8/R3K2k w - - 0 1");
    assert!(g.outcome(&s).is_none());
    println!("OBS 79.35 E rules antichess: the compulsory taking, the leader taken and play going on, no check, no two-unit move, five kinds to become at the last row, both ways of winning (as written by hand)");
}

// ------------------------------------------------------------------------------------------------------------------------------ bear and may_move on these games (part B's meaning), every fixture position

/// bear by a hand walk of the board: the places a unit of `kind` on `at` bears on, a place held by its own side counted (it guards it).
fn bear_by_hand(units: &BTreeMap<(i32, i32), (String, String)>, side: &str, kind: &str, at: (i32, i32)) -> BTreeSet<(i32, i32)> {
    let mut out = BTreeSet::new();
    let on = |p: (i32, i32)| (0..8).contains(&p.0) && (0..8).contains(&p.1);
    let ray = |out: &mut BTreeSet<(i32, i32)>, d: (i32, i32)| {
        let mut p = (at.0 + d.0, at.1 + d.1);
        while on(p) { out.insert(p); if units.contains_key(&p) { break; } p = (p.0 + d.0, p.1 + d.1); }
    };
    let orth = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    match kind {
        "rook" => for d in orth { ray(&mut out, d) },
        "bishop" => for d in diag { ray(&mut out, d) },
        "queen" => for d in orth.iter().chain(diag.iter()) { ray(&mut out, *d) },
        "knight" => for (a, b) in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)] { let p = (at.0 + a, at.1 + b); if on(p) { out.insert(p); } },
        "king" => for a in -1..=1 { for b in -1..=1 { let p = (at.0 + a, at.1 + b); if (a, b) != (0, 0) && on(p) { out.insert(p); } } },
        "pawn" => { let f = if side == "white" { 1 } else { -1 }; for a in [-1, 1] { let p = (at.0 + a, at.1 + f); if on(p) { out.insert(p); } } }
        other => panic!("{other}"),
    }
    out
}

fn square((f, r): (i32, i32)) -> String { format!("{}{}", (b'a' + f as u8) as char, r + 1) }

#[test]
fn bear_and_may_move_written_onto_layer_1s_graph_equal_a_hand_walk_of_the_board_at_every_fixture_position_and_the_allowed_moves_lie_in_may_move_but_for_the_listed_ones() {
    for game in GAMES {
        let fx = fixture(game);
        let g = kernel(game);
        let ruled = Ruled::open_variant(kernel(game), game, &rules_root(), "learned/chess/graph_vocabulary.json", None, None).unwrap();
        let st: Static = ruled.board().unwrap();
        let (leader_takable, leader_takes) = (matches!(game, "antichess" | "atomic"), game != "atomic");
        let (mut units_read, mut bears, mut moves, mut allowed, mut exceptions) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for p in &fx.pos {
            let s = at(&g, &p.text);
            let part = st.position(&ruled, &s);
            let mut board: BTreeMap<(i32, i32), (String, String)> = BTreeMap::new();
            for f in p.facts.iter().filter(|f| f.relation == "cell") {
                let file = (f.args[0].as_bytes()[0] - b'a') as i32;
                let rank = f.args[1].parse::<i32>().unwrap() - 1;
                board.insert((file, rank), (f.args[2].clone(), f.args[3].clone()));
            }
            let pn = g.place_names();
            for (ui, u) in part.units.iter().enumerate() {
                let name = &pn[u.at as usize];
                let pos = ((name.as_bytes()[0] - b'a') as i32, (name[1..].parse::<i32>().unwrap()) - 1);
                let (side, kind) = board[&pos].clone();
                let want_bear = bear_by_hand(&board, &side, &kind, pos);
                let got_bear: BTreeSet<String> = part.bears().iter().filter(|r| r.unit == ui).map(|r| pn[r.place as usize].clone()).collect();
                assert_eq!(got_bear, want_bear.iter().map(|q| square(*q)).collect::<BTreeSet<_>>(), "{game}: {} bears at {}: {side} {kind} {name}", p.label, p.text);
                // may_move: the places stepped onto by the kind's step and the game's rule of what may be taken
                let mut want_move = BTreeSet::new();
                if kind == "pawn" {
                    let f = if side == "white" { 1 } else { -1 };
                    let ahead = (pos.0, pos.1 + f);
                    if (0..8).contains(&ahead.1) && !board.contains_key(&ahead) {
                        want_move.insert(ahead);
                        // 2026-11-24 (plan 79.50): the first move of two places is in may_move as the description's `move_by` says it: from the second row of white, the seventh of black, and in the game with a horde the first row of white too, over two empty places
                        let start = if side == "white" { 1 } else { 6 };
                        let beyond = (pos.0, pos.1 + 2 * f);
                        if (pos.1 == start || (game == "horde" && side == "white" && pos.1 == 0)) && (0..8).contains(&beyond.1) && !board.contains_key(&beyond) { want_move.insert(beyond); }
                    }
                    for q in &want_bear { if let Some((o, k)) = board.get(q) { if *o != side && (leader_takable || k != "king") { want_move.insert(*q); } } }
                } else if kind == "king" && !leader_takes {
                    for q in &want_bear { if !board.contains_key(q) { want_move.insert(*q); } }
                } else {
                    for q in &want_bear {
                        match board.get(q) { None => { want_move.insert(*q); } Some((o, k)) => if *o != side && (leader_takable || k != "king") { want_move.insert(*q); } }
                    }
                }
                let got_move: BTreeSet<String> = part.moves().iter().filter(|r| r.unit == ui).map(|r| pn[r.place as usize].clone()).collect();
                assert_eq!(got_move, want_move.iter().map(|q| square(*q)).collect::<BTreeSet<_>>(), "{game}: {} may move at {}: {side} {kind} {name}", p.label, p.text);
                units_read += 1;
                bears += got_bear.len();
                moves += got_move.len();
            }
            // every allowed move is in may_move, but for the taking in passing and the two-unit move of the leader (the two-row step too before plan 79.50)
            if p.end.is_none() {
                for m in g.moves(&s) {
                    let (from, to) = g.move_ends(&s, &m);
                    let ui = part.units.iter().position(|u| u.at == from).unwrap();
                    let kind = &g.kind_names()[part.units[ui].kind as usize];
                    let in_may = part.moves().iter().any(|r| r.unit == ui && r.place == to);
                    allowed += 1;
                    if !in_may {
                        let (f, t) = (&pn[from as usize], &pn[to as usize]);
                        // 2026-11-24 (plan 79.50): the two-row step is not an exception any more (the pawn_two of the list is gone)
                        let passing = kind == "pawn" && f.as_bytes()[0] != t.as_bytes()[0] && g.kind_at(&s, to).is_none();
                        let two_unit = kind == "king" && (g.kind_at(&s, to).map(|k| g.kind_names()[k as usize] == "rook").unwrap_or(false) || (f.as_bytes()[0] as i32 - t.as_bytes()[0] as i32).abs() == 2);
                        assert!(passing || two_unit, "{game}: {} {f}{t} is allowed but not in may_move and is none of the two listed exceptions at {}", p.label, p.text);
                        exceptions += 1;
                    }
                }
            }
        }
        println!("OBS 79.35 E relations {game}: {units_read} units of {} positions: {bears} bear and {moves} may_move relations on layer 1's graph equal the hand walk; {allowed} allowed moves, {exceptions} of them outside may_move (the taking in passing, the two-unit move)", fx.pos.len());
    }
}

/// A SMOKE TEST, NOT A DEMAND OF MEANING: the other questions the layers above ask of a placed game (the hypothetical boards `without` and `relocate`, the reason a step is refused `why_not`, whether a side strikes a place `strikes_place`, the line a move takes
/// `find_step`) are asked at every position of every fixture, for every unit and place they take, and answer without a fault; the counts of the answers a game refuses (None) are printed for the report. Their meaning in these games is listed in the plan's report, not tested here.
#[test]
fn the_other_questions_of_a_placed_game_are_answered_at_every_fixture_position_without_a_fault() {
    for game in GAMES {
        let fx = fixture(game);
        let g = kernel(game);
        let (mut asked, mut without_none, mut relocate_none, mut steps_found, mut steps_refused) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for p in &fx.pos {
            let s = at(&g, &p.text);
            if p.end.is_some() { continue; }
            let cells = g.cells(&s);
            for &(owner, _kind, place) in &cells {
                asked += 1;
                if g.without(&s, place).is_none() { without_none += 1; }
                let _ = g.strikes_place(&s, owner, place);
            }
            if let Some(&(_, _, from)) = cells.iter().find(|c| Some(c.0) == g.mover(&s)) {
                for to in 0..64u16 {
                    if g.relocate(&s, from, to).is_none() { relocate_none += 1; }
                    match g.find_step(&s, from, to) { Some(_) => steps_found += 1, None => { steps_refused += 1; let _ = g.why_not(&s, from, to); } }
                }
            }
        }
        println!("OBS 79.35 E placed {game}: {asked} units asked for the board without them ({without_none} refused as no position of the game), {relocate_none} relocations refused, {steps_found} steps found and {steps_refused} refused (each refusal asked its reason)");
    }
}
