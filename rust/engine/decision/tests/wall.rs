//! Plan 79.15 PART B (SYSTEM_SPEC 8 the wall and the rule for choosing a move; piece: the wake): UNIT TESTS of `wall::candidates` first (numbers only), the first case being decision 18 of 79.14 (game bc_v_old_A:A9100:1, ply 21), then the INTEGRATION of the wall in the wake on that position
//! (the position as the engine logged it): the first move's probe is far below the search's value and other moves hold; the wake must not play it. No puzzle, no match.
use cs_decision::wall::{candidates, Held, Lowered};

/// Decision 18 of 79.14, as its log shows it: value 0 at depth 6; the first move (index 12) -1301; nine others 0 (probes in the order asked); the verification was cut (nothing held by it).
fn decision_18() -> Vec<(usize, i64)> { vec![(12, -1301), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, -390), (10, -475)] }

#[test]
fn b1_decision_18_the_first_move_is_not_a_candidate_and_the_unverified_moves_whose_probe_held_are() {
    let c = candidates(Some(12), 0, &decision_18(), &[], false);
    assert_eq!(c.lowered, Lowered::ByProbe);
    assert!(c.unverified);
    assert_eq!(c.held, Held::ProbeOnly);
    assert_eq!(c.moves, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(!c.moves.contains(&12));
}

#[test]
fn b2_a_verification_that_ran_gives_its_held_moves_and_not_the_lowered_first() {
    let c = candidates(Some(12), 0, &decision_18(), &[3, 5], true);
    assert_eq!((c.lowered, c.unverified, c.held), (Lowered::ByProbe, false, Held::Verified));
    assert_eq!(c.moves, vec![3, 5]);
}

#[test]
fn b3_a_first_the_probe_held_and_the_verification_dropped_is_lowered_too() {
    let probes = vec![(2, 0), (0, 5), (1, 0)];
    let c = candidates(Some(2), 0, &probes, &[0, 1], true);
    assert_eq!((c.lowered, c.held), (Lowered::ByVerification, Held::Verified));
    assert_eq!(c.moves, vec![0, 1]);
}

#[test]
fn b4_no_other_move_holds_then_the_highest_probe_alone_the_first_asked_on_a_tie() {
    let probes = vec![(4, -1301), (0, -300), (1, -300), (2, -800)];
    let c = candidates(Some(4), 0, &probes, &[], false);
    assert_eq!((c.lowered, c.held, c.unverified), (Lowered::ByProbe, Held::Highest, true));
    assert_eq!(c.moves, vec![0]);
    // the verification ran and held nothing: the same
    let c = candidates(Some(4), 0, &probes, &[], true);
    assert_eq!((c.held, c.unverified, c.moves.clone()), (Held::Highest, false, vec![0]));
    // the verification dropped every move whose probe held, the first among them: the highest probe, which here is the first's own
    let probes = vec![(1, 7), (0, 3), (2, -100)];
    let c = candidates(Some(0), 0, &probes, &[], true);
    assert_eq!((c.lowered, c.held, c.moves), (Lowered::ByVerification, Held::Highest, vec![1]));
}

#[test]
fn b5_no_lowered_first_move_the_set_is_mains_the_first_at_the_head() {
    // the probe held and the verification kept it
    let probes = vec![(3, 0), (1, 0), (2, 0)];
    let c = candidates(Some(3), 0, &probes, &[1, 3, 2], true);
    assert_eq!((c.lowered, c.held, c.unverified, c.moves), (Lowered::No, Held::AsBefore, false, vec![3, 1, 2]));
    // the clock cut the verification and the first's probe held: the first alone, as on main
    let c = candidates(Some(3), 0, &probes, &[], false);
    assert_eq!((c.lowered, c.moves), (Lowered::No, vec![3]));
    // the first move was not asked (the clock ended before it): as on main, the first alone
    let c = candidates(Some(3), 0, &[], &[], false);
    assert_eq!((c.lowered, c.moves), (Lowered::No, vec![3]));
    // no first move read: the moves held as they are
    let c = candidates(None, 0, &[], &[], false);
    assert_eq!((c.lowered, c.moves), (Lowered::No, vec![]));
    // the game decided (no probes asked, nothing held): the first alone
    let c = candidates(Some(0), 1_000_000, &[], &[], false);
    assert_eq!(c.moves, vec![0]);
}

#[test]
fn b6_a_first_move_above_the_value_is_never_lowered() {
    let probes = vec![(1, 40), (0, 0)];
    let c = candidates(Some(1), 0, &probes, &[0, 1], true);
    assert_eq!((c.lowered, c.moves), (Lowered::No, vec![1, 0]));
}

// ------------------------------------------------------------------------------------------------------------------------------ the whole wake on decision 18

use cs_decision::threat_verdict::Record;
use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::ledger::Worth;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    (g.clone(), Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap())
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }

/// The search alone at a clock too short for its verification: the old player's reading, then the wake with the old pieces (the wall is part of `decide` whatever pieces are on).
#[test]
fn b7_the_wake_does_not_play_a_first_move_the_probe_lowered_while_other_moves_hold() {
    let (g, st) = world();
    let s = KernelGame::from_fen("chess", "1n1qkbnr/2pbpppp/8/3p4/8/2PPPP2/1B4PP/QN2KBNR b Kk - 2 11").unwrap().initial();
    let judge = Worth { name: "record".into(), table: table(), ours_first: false };
    // many clocks: whenever the log says the first move was lowered, the move played is not the first
    let mut lowered_seen = 0;
    for seconds in [0.02, 0.05, 0.1, 0.2, 0.4] {
        let params = WakeParams { seconds, depth: 64, pieces: Pieces { reasons_pick: true, ..Pieces::default() }, places: vec![],
            inputs: WakeInputs { record: Record { worth: table(), half: HashMap::new() }, ..WakeInputs::default() } };
        let d = decide(&st, &g, &s, &[], &judge, &params);
        let first = d.first.clone();
        if d.log.first_lowered.is_some() {
            lowered_seen += 1;
            let probe_of = |m: &String| d.log.moves.iter().find(|x| &x.mv == m).and_then(|x| x.value);
            let value = d.value;
            let others_hold = d.log.moves.iter().any(|x| Some(&x.mv) != first.as_ref() && x.value.is_some_and(|v| v >= value));
            if others_hold { assert_ne!(d.mv, first, "the lowered first move was played at {seconds} s (its probe {:?}, the value {value})", first.as_ref().and_then(probe_of)); }
            assert!(!d.equal.is_empty());
        }
    }
    println!("lowered first moves seen at the five clocks: {lowered_seen}");
}

/// With no lowered first move the wake is move for move main's: at a depth the clock does not cut (a small depth, a long clock) the moves held equal are the search's first move at the head and the moves the verification held, as main built them.
#[test]
fn b8_with_no_lowered_first_move_the_moves_held_equal_are_mains() {
    use cs_tactics::ledger::{Search, WIN};
    let (g, st) = world();
    let judge = Worth { name: "record".into(), table: table(), ours_first: false };
    let (mut same, mut lowered) = (0, 0);
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1",
                "1n1qkbnr/2pbpppp/8/3p4/8/2PPPP2/1B4PP/QN2KBNR b Kk - 2 11", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPPBQPPP/R3K2R w KQkq - 0 9", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "8/8/4k3/8/3K4/8/3P4/8 w - - 0 1"] {
        let s = KernelGame::from_fen("chess", fen).unwrap().initial();
        let params = WakeParams { seconds: 30.0, depth: 4, pieces: Pieces { reasons_pick: true, ..Pieces::default() }, places: vec![], inputs: WakeInputs { record: Record { worth: table(), half: HashMap::new() }, ..WakeInputs::default() } };
        let d = decide(&st, &g, &s, &[], &judge, &params);
        // main's set, rebuilt from the search itself
        let me = g.mover(&s).unwrap_or(0);
        let mut ls = Search::new(&g, &judge, me, 30.0);
        ls.verify_ties = true;
        if !d.named_mine.is_empty() { ls.named.insert(g.key(&s), d.named_mine.clone()); }       // (the names the wake gave its search)
        if !d.named_theirs.is_empty() { ls.named_ply.insert(1, d.named_theirs.clone()); }
        let ln = ls.run(&s, 4);
        let moves = g.moves(&s);
        let mut main_equal: Vec<usize> = vec![];
        if ln.value.abs() < WIN - 1000 && ln.first_index.is_some() {
            ls.resume(10.0);
            let probes = ls.probes(&s, ln.depth, ln.value, None);
            let first: Vec<usize> = probes.iter().filter(|(_, v)| *v >= ln.value).map(|(i, _)| *i).collect();
            main_equal = if first.is_empty() { first } else { ls.verify(&s, ln.depth, ln.value, &first) };
            let o = ln.first_index.unwrap();
            main_equal.retain(|i| *i != o);
            main_equal.insert(0, o);
        }
        if d.log.first_lowered.is_some() { lowered += 1; continue; }
        let main_names: Vec<String> = main_equal.iter().map(|i| g.move_name(&s, &moves[*i])).collect();
        if ln.value.abs() < WIN - 1000 { assert_eq!(d.equal, main_names, "{fen}"); same += 1; }
    }
    println!("positions with the set of main's: {same}; with a lowered first move (not compared): {lowered}");
    assert!(same >= 3);
}
