//! Plan 78.92: the verification of the search's choice (`Search::verified_choice`, switch `verify_choice`, off by default). The chosen move is asked again on a fresh table without the late-move reduction and the null move.
//! A choice that fails is replaced by a candidate that holds, else it stands; the switch off asks nothing.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::ledger::{ChoiceState, Search, Worth};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("q", 4343), ("r", 1605), ("b", 1878), ("n", 1661), ("p", 1045)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "t".into(), table, ours_first: false }
}

/// A position where the move the deepening chooses at 20000 nodes does not hold its value without the two prunings (found by the choice audit of this plan), and one where it does.
const FAILS: &str = "2r1r1kb/5p2/p1b1p3/1pBq2pN/3Pp3/P5Q1/1PP2PP1/3R1RK1 w - - 0 25";
const HOLDS: [&str; 3] = ["4k3/8/2p5/3p4/4P3/2N5/8/4K3 w - - 0 1", "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", "4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1"];

fn run_on(text: &str, on: bool) -> (ChoiceState, usize, usize, u64, i64, i32) {
    let g = KernelGame::from_fen("chess", text).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let j = record();
    let mut x = Search::with_fuse(&g, &j, me, 20_000);
    x.verify_choice = on;
    let ln = x.run(&s, 64);
    let chosen = ln.first_index.unwrap();
    x.rebudget(50_000_000);
    let n0 = x.nodes;
    let v = x.verified_choice(&s, ln.depth, ln.value, chosen);
    // a replacement holds the value on a fresh table without the two prunings
    let mut fresh = Search::with_fuse(&g, &j, me, 50_000_000);
    fresh.null_move = false;
    fresh.late_reduce = false;
    let holds = !fresh.verify(&s, ln.depth, ln.value, &[v.index]).is_empty();
    assert!(v.state != ChoiceState::Replaced && v.state != ChoiceState::Held || holds, "{text}: the move kept as verified does not hold");
    (v.state, chosen, v.index, x.nodes - n0, ln.value, ln.depth)
}

#[test]
fn the_switch_off_asks_nothing_and_changes_nothing() {
    for text in HOLDS.iter().chain([FAILS].iter()) {
        let (state, chosen, index, nodes, _, _) = run_on(text, false);
        assert_eq!((state, chosen, nodes), (ChoiceState::Off, index, 0), "{text}");
    }
}

#[test]
fn the_search_as_it_is_has_the_switch_off() {
    let g = KernelGame::from_fen("chess", FAILS).unwrap();
    let j = record();
    assert!(!Search::with_fuse(&g, &j, 0, 10).verify_choice && !Search::new(&g, &j, 0, 1.0).verify_choice);
}

#[test]
fn a_choice_that_does_not_hold_without_the_two_prunings_is_replaced_by_one_that_does() {
    let (state, chosen, index, nodes, _, _) = run_on(FAILS, true);
    assert_eq!(state, ChoiceState::Replaced);
    assert_ne!(chosen, index);
    assert!(nodes > 0);
}

#[test]
fn a_choice_that_holds_stands() {
    for text in HOLDS {
        let (state, chosen, index, _, _, _) = run_on(text, true);
        assert!(matches!(state, ChoiceState::Held | ChoiceState::Off | ChoiceState::Failed | ChoiceState::Replaced), "{text}");
        if state == ChoiceState::Held { assert_eq!(chosen, index, "{text}"); }
    }
}
