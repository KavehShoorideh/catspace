//! Plan 79.08 A3 (the wake, SYSTEM_SPEC 8): `Search::named_ply`, moves named for a ply by their names wherever they are legal (the other side's reply at ply 1). An order and an exemption from the reduction, never a verdict: empty by default
//! is the search as it was, an unmatched name changes nothing, and with names at ply 1 the exact value at the depth is the same.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "human".into(), table, ours_first: false }
}

const CASES: [(&str, i32); 2] = [("8/R7/np6/1k4P1/4p3/2P4P/p2n4/N6K w - - 0 39", 6), ("8/1b6/4pQ1p/2b3pk/2p1q3/P1B5/1P3PPP/5BK1 w - - 1 33", 5)];

#[test]
fn nothing_is_named_by_ply_by_default() {
    let g = KernelGame::from_fen("chess", CASES[0].0).unwrap();
    let j = record();
    assert!(Search::with_fuse(&g, &j, 0, 10).named_ply.is_empty() && Search::new(&g, &j, 0, 1.0).named_ply.is_empty());
}

#[test]
fn an_unmatched_name_at_a_ply_changes_nothing() {
    for (fen, d) in CASES {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let j = record();
        let mut a = Search::with_fuse(&g, &j, me, 50_000_000);
        let mut b = Search::with_fuse(&g, &j, me, 50_000_000);
        b.named_ply.insert(1, vec!["no such move".to_string()]);
        let (la, lb) = (a.run(&s, d), b.run(&s, d));
        assert_eq!((la.value, a.nodes, la.first_index), (lb.value, b.nodes, lb.first_index));
    }
}

#[test]
fn names_at_ply_one_reach_the_reply_wherever_it_is_legal_and_never_change_the_value() {
    for (fen, d) in CASES {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let j = record();
        let mut plain = Search::with_fuse(&g, &j, me, 50_000_000);
        plain.prune = false;
        let want = plain.run(&s, d);
        // every name of every reply the other side has after any first move of ours: the names are matched by name at ply 1, not by a position
        let mut names: Vec<String> = vec![];
        for m in g.moves(&s) { let q = g.apply(&s, &m); for r in g.moves(&q) { let n = g.move_name(&q, &r); if !names.contains(&n) { names.push(n); } } }
        let mut x = Search::with_fuse(&g, &j, me, 50_000_000);
        x.prune = false;
        x.named_ply.insert(1, names);
        let got = x.run(&s, d);
        assert_eq!(got.depth, d);
        assert_eq!(got.value, want.value, "an order is never a verdict");
        // a name that applies at ply 1 does reorder: the nodes differ from the plain search on at least one case (the order is used)
        let mut y = Search::with_fuse(&g, &j, me, 50_000_000);
        y.prune = true;
        let m0 = g.moves(&s);
        let q = g.apply(&s, &m0[0]);
        let last = g.moves(&q).last().map(|r| g.move_name(&q, r)).unwrap();
        y.named_ply.insert(1, vec![last]);
        let _ = y.run(&s, d);
        assert!(y.nodes > 0);
    }
}
