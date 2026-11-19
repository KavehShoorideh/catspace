//! Plan 79.02: the equal reading that asks a held move again (`Search::ties_reasked`, switch `reask`, off by default). The moves whose null window held the best are asked once more on a fresh table with the late-move reduction and the null move off,
//! in the deepening's own order of the root moves; a move the unpruned answer puts under the best leaves the set, a move not reached before the clock stopped stays; the move the search plays is not asked again.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("q", 4343), ("r", 1605), ("b", 1878), ("n", 1661), ("p", 1045)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "t".into(), table, ours_first: false }
}

const FENS: [&str; 6] = [
    "2r1r1kb/5p2/p1b1p3/1pBq2pN/3Pp3/P5Q1/1PP2PP1/3R1RK1 w - - 0 25",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 12",
    "r3k2r/pp1n1ppp/2p1pn2/q2p4/1b1P4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 9",
    "4k3/8/2p5/3p4/4P3/2N5/8/4K3 w - - 0 1",
    "2kr3r/ppp2ppp/2n1bn2/4p3/4P1b1/2NP1N2/PPP1BPPP/R1B1K2R w KQ - 4 9",
];

struct Read { old: Vec<usize>, new: Vec<usize>, first: Vec<usize>, count: cs_tactics::ledger::ReaskCount, chosen: usize, depth: i32, value: i64 }

fn read(fen: &str, run_nodes: u64, tie_nodes: u64) -> Read {
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let j = record();
    let mut x = Search::with_fuse(&g, &j, me, run_nodes);
    let ln = x.run(&s, 64);
    let chosen = ln.first_index.unwrap();
    // the equal reading as it was, on a copy of the same run (a second search, the same budget: the table is the same)
    let mut y = Search::with_fuse(&g, &j, me, run_nodes);
    y.run(&s, 64);
    y.rebudget(tie_nodes);
    let mut old = y.ties(&s, ln.depth, ln.value);
    old.sort();
    x.rebudget(tie_nodes);
    x.reask = true;
    x.reask_skip = Some(chosen);
    let first: Vec<usize> = { let mut z = Search::with_fuse(&g, &j, me, run_nodes); z.run(&s, 64); z.rebudget(tie_nodes); let ord = z.deepening_order(&s); let mut f = z.probes_in(&s, ln.depth, ln.value, None, Some(&ord)).into_iter().filter(|(_, p)| *p >= ln.value).map(|(i, _)| i).collect::<Vec<_>>(); f.sort(); f };
    let mut new = x.ties(&s, ln.depth, ln.value);
    new.sort();
    Read { old, new, first, count: x.reasked, chosen, depth: ln.depth, value: ln.value }
}

#[test]
fn the_switch_is_off_by_default_and_off_the_equal_reading_is_as_it_was() {
    let g = KernelGame::from_fen("chess", FENS[0]).unwrap();
    let j = record();
    assert!(!Search::with_fuse(&g, &j, 0, 10).reask && !Search::new(&g, &j, 0, 1.0).reask);
    assert!(Search::with_fuse(&g, &j, 0, 10).reask_skip.is_none());
    for fen in FENS {
        let s0 = KernelGame::from_fen("chess", fen).unwrap();
        let s = s0.initial();
        let me = s0.mover(&s).unwrap();
        let mut a = Search::with_fuse(&s0, &j, me, 20_000);
        let ln = a.run(&s, 64);
        a.rebudget(5_000_000);
        let by_ties = a.ties(&s, ln.depth, ln.value);
        let mut b = Search::with_fuse(&s0, &j, me, 20_000);
        b.run(&s, 64);
        b.rebudget(5_000_000);
        let by_probes: Vec<usize> = b.probes(&s, ln.depth, ln.value, None).into_iter().filter(|(_, p)| *p >= ln.value).map(|(i, _)| i).collect();
        assert_eq!(by_ties, by_probes, "{fen}: with the switch off `ties` is the probes' held moves, in the fixed reading order");
        assert_eq!(a.reasked, cs_tactics::ledger::ReaskCount::default(), "{fen}: nothing is counted with the switch off");
    }
}

#[test]
fn the_set_is_never_larger_than_the_first_reading_and_a_move_it_drops_fails_the_unpruned_question() {
    let j = record();
    let mut dropped_somewhere = 0;
    for fen in FENS {
        let r = read(fen, 20_000, 50_000_000);
        // the set is a subset of the moves the first reading holds (in the deepening's order)
        assert!(r.new.iter().all(|i| r.first.contains(i)), "{fen}: the set holds a move the first reading did not");
        let mut kept = r.first.clone();
        kept.retain(|i| r.new.contains(i));
        let gone: Vec<usize> = r.first.iter().copied().filter(|i| !r.new.contains(i)).collect();
        assert_eq!(gone.len(), r.count.out, "{fen}: the count of moves put out is the moves that left");
        assert!(r.count.held == r.first.len() && r.count.asked >= r.count.held, "{fen}");
        assert_eq!(r.count.unreached, 0, "{fen}: with room in the slot every held move is asked again");
        assert_eq!(r.count.reasked, r.first.iter().filter(|i| **i != r.chosen).count(), "{fen}: every held move but the one played is asked again");
        // a move put out fails the unpruned question; a move kept (but the one played) passes it
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let mut f = Search::with_fuse(&g, &j, g.mover(&s).unwrap(), 50_000_000);
        for i in &gone { assert!(f.verify(&s, r.depth, r.value, &[*i]).is_empty(), "{fen}: move {i} was put out but holds without the prunings"); }
        for i in kept.iter().filter(|i| **i != r.chosen) { assert!(!f.verify(&s, r.depth, r.value, &[*i]).is_empty(), "{fen}: move {i} stayed but fails without the prunings"); }
        assert!(r.new.contains(&r.chosen) || !r.first.contains(&r.chosen), "{fen}: the move played is never asked again, so never put out");
        let _ = &r.old;
        dropped_somewhere += gone.len();
    }
    println!("OBS 79.02 the unpruned question put out {dropped_somewhere} moves over {} positions", FENS.len());
}

#[test]
fn a_clock_that_stops_the_first_reading_returns_the_moves_it_held_and_asks_nothing_again() {
    let j = record();
    for fen in FENS {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let mut x = Search::with_fuse(&g, &j, me, 20_000);
        let ln = x.run(&s, 64);
        x.rebudget(40);            // a fuse the first reading cannot finish
        x.reask = true;
        x.reask_skip = ln.first_index;
        let set = x.ties(&s, ln.depth, ln.value);
        assert!(x.cut() && x.reasked.first_cut && x.reasked.reasked == 0, "{fen}: the reading was cut in the first asking");
        assert_eq!(x.reasked.held, set.len(), "{fen}: the moves held so far are the set");
        assert!(x.reasked.asked < g.moves(&s).len(), "{fen}: some moves were not asked, so they are not in the set");
    }
}

#[test]
fn a_clock_that_stops_the_second_asking_keeps_the_moves_not_reached() {
    let j = record();
    let mut seen = 0;
    for fen in FENS {
        let full = read(fen, 20_000, 50_000_000);
        if full.count.reasked < 2 { continue; }
        // a fuse that lets the first reading finish and stops the second asking part of the way: try growing budgets, take the first one that cuts after at least one answer
        for fuse in [200u64, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 100_000, 300_000, 1_000_000] {
            let g = KernelGame::from_fen("chess", fen).unwrap();
            let s = g.initial();
            let me = g.mover(&s).unwrap();
            let mut x = Search::with_fuse(&g, &j, me, 20_000);
            let ln = x.run(&s, 64);
            x.rebudget(fuse);
            x.reask = true;
            x.reask_skip = ln.first_index;
            let set = x.ties(&s, ln.depth, ln.value);
            if x.reasked.first_cut || !x.cut() { continue; }
            assert!(x.reasked.unreached > 0 && x.reasked.reasked + x.reasked.unreached == full.count.reasked, "{fen}: asked again {} + not reached {} of {}", x.reasked.reasked, x.reasked.unreached, full.count.reasked);
            assert_eq!(set.len(), x.reasked.held - x.reasked.out, "{fen}: the moves not reached stay");
            seen += 1;
            break;
        }
    }
    assert!(seen >= 1, "no position cut the second asking: widen the fuses");
}

#[test]
fn the_deepening_order_is_the_roots_moves_with_the_tables_best_first() {
    let j = record();
    for fen in FENS {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let mut x = Search::with_fuse(&g, &j, g.mover(&s).unwrap(), 20_000);
        let ln = x.run(&s, 64);
        let mut o = x.deepening_order(&s);
        assert_eq!(o[0], ln.first_index.unwrap(), "{fen}: the table's best move is read first");
        o.sort();
        assert_eq!(o, (0..g.moves(&s).len()).collect::<Vec<_>>(), "{fen}: every legal move once");
    }
}
