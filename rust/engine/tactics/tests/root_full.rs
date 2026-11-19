//! Plan 79.03: the switch `root_full` (no late-move reduction at the root), off by default. A late quiet root move is read at the full depth with the window like every other root move, so a root move whose win lies deeper than the reduced reading
//! is no longer dropped as a fail-low. The reduction below the root and the null move are untouched.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "human".into(), table, ours_first: false }
}

// (position, depth, value without the switch, value with it): read by plan 79.03's part one (a root move the reduction dropped) and kept here as the fixed cases
const CASES: [(&str, i32); 3] = [
    ("8/R7/np6/1k4P1/4p3/2P4P/p2n4/N6K w - - 0 39", 9),
    ("8/1b6/4pQ1p/2b3pk/2p1q3/P1B5/1P3PPP/5BK1 w - - 1 33", 5),
    ("1k6/p1p1Rrp1/1p1n2p1/6P1/P7/8/1P6/K7 w - - 0 35", 8),
];

struct Run { value: i64, nodes: u64, first: Option<usize>, best_exact: i64 }

fn read(fen: &str, depth: i32, root_full: bool, late: bool) -> Run {
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let j = record();
    let mut x = Search::with_fuse(&g, &j, me, 50_000_000);
    x.root_full = root_full;
    x.late_reduce = late;
    let ln = x.run(&s, depth);
    assert_eq!(ln.depth, depth, "the deepening must finish the depth asked");
    let mut e = Search::with_fuse(&g, &j, me, 50_000_000);
    let best_exact = e.root_values(&s, depth).first().map(|(_, v)| *v).unwrap();
    Run { value: ln.value, nodes: x.nodes, first: ln.first_index, best_exact }
}

#[test]
fn the_switch_is_off_by_default_everywhere() {
    let g = KernelGame::from_fen("chess", CASES[0].0).unwrap();
    let j = record();
    assert!(!Search::with_fuse(&g, &j, 0, 10).root_full && !Search::new(&g, &j, 0, 1.0).root_full);
    assert!(!cs_tactics::operators::LedgerOp::new(&j).root_full);
}

#[test]
fn with_the_switch_off_the_search_is_as_it_was_and_repeats() {
    for (fen, d) in CASES {
        let a = read(fen, d, false, true);
        let b = read(fen, d, false, true);
        assert_eq!((a.value, a.nodes, a.first), (b.value, b.nodes, b.first));
    }
}

#[test]
fn a_root_move_the_reduction_dropped_is_found_with_the_switch_on() {
    // the first two: the unreduced root reads a better move than the reduced one (the value without the switch is under the best root move's own value at the depth)
    for (fen, d) in &CASES[..2] {
        let off = read(fen, *d, false, true);
        let on = read(fen, *d, true, true);
        assert!(off.value < off.best_exact || off.value < on.value, "the case must show the fault: off {} exact {} on {}", off.value, off.best_exact, on.value);
        assert!(on.value > off.value, "the switch on reads a better value: off {} on {}", off.value, on.value);
        assert!(on.nodes > off.nodes, "the root's late moves are read in full: that costs nodes (off {} on {})", off.nodes, on.nodes);
    }
}

#[test]
fn the_switch_changes_nothing_where_there_is_no_reduction_to_drop() {
    for (fen, d) in &CASES[..2] {
        let off = read(fen, *d, false, false);
        let on = read(fen, *d, true, false);
        assert_eq!((off.value, off.nodes, off.first), (on.value, on.nodes, on.first), "the reduction off: the switch has nothing to change");
    }
}

#[test]
fn the_value_with_the_switch_on_is_never_under_the_unreduced_best_root_move_of_the_same_prunings_in_these_cases() {
    // the unreduced root is the exact reading's root (`root_values`: every root move with the full window): with the switch on the deepening's value is at least that value less what the prunings below the root can move
    let on = read(CASES[0].0, CASES[0].1, true, true);
    assert!(on.value >= on.best_exact, "value {} against the root's own best {}", on.value, on.best_exact);
}

#[test]
fn the_gap_audit_reads_the_deepening_and_every_root_move_and_takes_the_switch() {
    let (fen, d) = CASES[1];
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    let j = record();
    for on in [false, true] {
        let a = cs_tactics::gap_audit::gap_audit(&g, &j, &s, &[], 50_000_000, d, 50_000_000, on).unwrap();
        let r = read(fen, d, on, true);
        assert_eq!((a.depth, a.value), (d, r.value));
        assert_eq!(a.rows.len(), g.moves(&s).len());
        assert_eq!(a.rows.iter().filter_map(|x| x.exact).max(), Some(r.best_exact));
        assert!(a.first.is_some() && !a.exact_cut);
    }
}
