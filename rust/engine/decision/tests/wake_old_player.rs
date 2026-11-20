//! Plan 79.05 (SYSTEM_SPEC 8, the wake): the old player is the wake with every piece off, unchanged; a piece asked for and not ported is listed, never skipped silently; `Params` names no move by default.
use cs_decision::wake::{decide, Pieces, WakeParams};
use cs_decision::Params;
use cs_game::kernel::KernelGame;
use cs_game::placed::{Placed, Steps};
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "human".into(), table, ours_first: false }
}

/// A static part the old player never reads (the wake needs one to be called; a two-place chain is the smallest the graph accepts).
fn tiny() -> Static {
    let names = vec!["x1".to_string(), "x2".to_string()];
    let steps = Steps { directions: vec!["d".into()], rows: vec![(0, 0, 1)], ..Default::default() };
    Static::new(&steps, &Names::new(names, vec![], vec![])).unwrap()
}

#[test]
fn the_params_name_no_move_and_the_pieces_are_off_by_default() {
    assert!(Params::default().named.is_empty());
    assert_eq!(Pieces::default(), Pieces::old_player());
    assert!(Pieces::old_player().unported().is_empty());
    assert!(!Pieces::all().unported().is_empty(), "a piece asked for and not ported must be listed");
}

#[test]
fn the_wake_with_every_piece_off_is_the_old_player() {
    let st = tiny();
    for fen in ["8/R7/np6/1k4P1/4p3/2P4P/p2n4/N6K w - - 0 39", "8/1b6/4pQ1p/2b3pk/2p1q3/P1B5/1P3PPP/5BK1 w - - 1 33"] {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let j = record();
        let _ = g.place_names();
        let me = g.mover(&s).unwrap();
        let mut plain = Search::with_fuse(&g, &j, me, 50_000_000);
        let want = plain.run(&s, 5);
        // the wake runs with a node budget through its own clock-free path only in its tests: the clock of the decision is long, and the depth is capped at the same 5
        let p = WakeParams { seconds: 60.0, depth: 5, ..WakeParams::default() };
        let got = decide(&st, &g, &s, &[], &j, &p);
        assert_eq!(got.mv, want.first);
        assert_eq!(got.value, want.value);
        assert!(got.unported.is_empty() && got.contributions.len() == 1 && got.contributions[0].step == 5);
    }
}
