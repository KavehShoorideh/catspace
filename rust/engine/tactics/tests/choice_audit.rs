//! Plan 78.92: the choice audit repeats (a node limit, fresh tables), its reference never ranks the played move above its own best, a reading the fuse cuts has no numbers, and the switches the audit turns
//! are on in the search as it is.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::choice_audit::choice_audit;
use cs_tactics::ledger::{Search, Worth};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("q", 4343), ("r", 1605), ("b", 1878), ("n", 1661), ("p", 1045)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "t".into(), table, ours_first: false }
}

const POSITIONS: [&str; 4] = ["8/8/3k4/7P/6K1/3N1P2/8/1r6 w - - 1 54", "4k3/8/2p5/3p4/4P3/2N5/8/4K3 w - - 0 1", "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", "4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1"];

#[test]
fn the_audit_repeats_and_the_reference_is_consistent() {
    for pos in POSITIONS {
        let g = KernelGame::from_fen("chess", pos).unwrap();
        let s = g.initial();
        let j = record();
        let a = choice_audit(&g, &j, &s, &[], 20_000, 64, 2_000_000, 2, None, false).unwrap();
        let b = choice_audit(&g, &j, &s, &[], 20_000, 64, 2_000_000, 2, None, false).unwrap();
        assert_eq!((a.depth, a.value, a.played, a.run_nodes), (b.depth, b.value, b.played, b.run_nodes), "{pos}");
        assert_eq!(a.readings.len(), 8);
        for (x, y) in a.readings.iter().zip(b.readings.iter()) {
            assert_eq!((x.best, x.played, x.nodes), (y.best, y.played, y.nodes), "{pos} {}", x.name);
            if let (Some(best), Some(pl)) = (x.best, x.played) { assert!(pl <= best, "{pos} {}: the played move is above the best", x.name); }
            if x.best.is_none() { assert!(x.played.is_none()); }
        }
    }
}

#[test]
fn a_cut_reading_has_no_numbers() {
    let g = KernelGame::from_fen("chess", POSITIONS[3]).unwrap();
    let s = g.initial();
    let a = choice_audit(&g, &record(), &s, &[], 5_000, 64, 50, 2, None, false).unwrap();
    assert!(a.readings.iter().all(|r| r.best.is_none() && r.played.is_none()));
}

#[test]
fn the_search_as_it_is_has_both_prunings_on() {
    let g = KernelGame::from_fen("chess", POSITIONS[1]).unwrap();
    let j = record();
    let x = Search::with_fuse(&g, &j, 0, 1000);
    assert!(x.prune && x.null_move && x.late_reduce && !x.verify_ties);
}

#[test]
fn the_repair_reads_the_move_the_verification_leaves() {
    let g = KernelGame::from_fen("chess", POSITIONS[0]).unwrap();
    let s = g.initial();
    let j = record();
    let off = choice_audit(&g, &j, &s, &[], 20_000, 64, 2_000_000, 2, None, false).unwrap();
    let on = choice_audit(&g, &j, &s, &[], 20_000, 64, 2_000_000, 2, None, true).unwrap();
    assert_eq!(off.repair_nodes, 0);
    assert_eq!((on.depth, on.value, on.run_nodes), (off.depth, off.value, off.run_nodes));
    assert!(on.repair_nodes > 0 || on.repair_state != cs_tactics::ledger::ChoiceState::Held);
}
