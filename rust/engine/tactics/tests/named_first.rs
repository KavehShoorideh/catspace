//! Plan 79.05 (the wake): `Search::named`, moves named by position, read first and unreduced. An ORDER and an exemption from the late-move reduction, never a verdict: with nothing named the search is exactly as it was (same value, same nodes, same first
//! move, and it repeats); with moves named at the root and below it the search still finishes the depth and the value is the exact value of the position at that depth.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "human".into(), table, ours_first: false }
}

const CASES: [(&str, i32); 2] = [
    ("8/R7/np6/1k4P1/4p3/2P4P/p2n4/N6K w - - 0 39", 6),
    ("8/1b6/4pQ1p/2b3pk/2p1q3/P1B5/1P3PPP/5BK1 w - - 1 33", 5),
];

#[test]
fn nothing_is_named_by_default_everywhere() {
    let g = KernelGame::from_fen("chess", CASES[0].0).unwrap();
    let j = record();
    assert!(Search::with_fuse(&g, &j, 0, 10).named.is_empty() && Search::new(&g, &j, 0, 1.0).named.is_empty());
    assert!(cs_tactics::operators::LedgerOp::new(&j).named.is_empty());
}

#[test]
fn with_nothing_named_the_search_is_as_it_was_and_repeats() {
    for (fen, d) in CASES {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let j = record();
        let mut a = Search::with_fuse(&g, &j, me, 50_000_000);
        let mut b = Search::with_fuse(&g, &j, me, 50_000_000);
        b.named.insert(0xdead_beef, vec!["no such move".to_string()]);          // (a key no position has, a name no move has: nothing may change)
        let (la, lb) = (a.run(&s, d), b.run(&s, d));
        assert_eq!((la.value, a.nodes, la.first_index), (lb.value, b.nodes, lb.first_index));
    }
}

#[test]
fn moves_named_at_the_root_and_below_change_the_order_and_never_the_value() {
    for (fen, d) in CASES {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let j = record();
        let mut plain = Search::with_fuse(&g, &j, me, 50_000_000);
        plain.prune = false;                                                // (no forward pruning: the exact value at the depth, whatever the order)
        let want = plain.run(&s, d);
        let moves = g.moves(&s);
        let mut x = Search::with_fuse(&g, &j, me, 50_000_000);
        x.prune = false;
        // name the last legal root move, and every move of the position after the first one (below the root)
        x.named.insert(g.key(&s), vec![g.move_name(&s, moves.last().unwrap())]);
        let q = g.apply(&s, &moves[0]);
        x.named.insert(g.key(&q), g.moves(&q).iter().map(|m| g.move_name(&q, m)).collect());
        let got = x.run(&s, d);
        assert_eq!(got.depth, d, "the deepening must finish the depth asked");
        assert_eq!(got.value, want.value, "an order is never a verdict: the value at the depth is the same");
    }
}
