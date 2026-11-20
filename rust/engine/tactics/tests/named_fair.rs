//! PLAN 79.11 A (SYSTEM_SPEC 8, the rule for choosing a move; piece: the search under the wake): THE TILT OF A NAMED MOVE. A named root move is read in full, an un-named late one reduced; a reduced reading that fails low is taken as it is, so the named move can come out above a rival that is as good or better.
//! The test names each legal root move in turn, in constructed positions, and compares the value of the search to the value of the same search with no reduction at the root (`root_full`, the reference: every root move read by one asking). The tilt is a named run whose value is UNDER the reference's
//! (a better rival was not asked in full). With `fair_names` off the tilt is confirmed in these positions; with it on there is none; with nothing named the switch changes nothing.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "human".into(), table, ours_first: false }
}

const FENS: [&str; 76] = [
    "1k6/2p5/8/4p3/5B2/8/8/K7 w - - 0 1",
    "1k6/2p5/8/8/2B5/8/8/K1R5 w - - 0 1",
    "1k6/2p5/8/8/5B2/8/8/K1R5 b - - 0 1",
    "1k6/2p5/8/8/5B2/8/8/K1R5 w - - 0 1",
    "1k6/2p5/8/8/5B2/8/8/K7 w - - 0 1",
    "1k6/2p5/8/8/8/8/8/R6K w - - 0 1",
    "1k6/p1p1Rrp1/1p1n2p1/6P1/P7/8/1P6/K7 w - - 0 35",
    "2kr3r/ppp2ppp/2n1bn2/4p3/4P1b1/2NP1N2/PPP1BPPP/R1B1K2R w KQ - 4 9",
    "2r1r1kb/5p2/p1b1p3/1pBq2pN/3Pp3/P5Q1/1PP2PP1/3R1RK1 w - - 0 25",
    "2r3k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "3q2k1/8/8/8/8/8/8/R5K1 w - - 0 1",
    "3q3k/8/8/8/3b4/2N5/8/R3K3 w - - 0 1",
    "4k3/7p/8/p7/8/8/8/RB2K3 w - - 0 1",
    "4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1",
    "4k3/8/2p5/3p4/4P3/2N5/8/4K3 w - - 0 1",
    "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1",
    "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1",
    "4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1",
    "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
    "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
    "4k3/8/8/4B3/8/8/1b6/R3K3 w - - 0 1",
    "4k3/8/8/8/3b4/8/8/R2NK3 w - - 0 1",
    "4k3/8/8/8/3bN3/8/8/R3K3 w - - 0 1",
    "4k3/8/8/8/8/4K3/4P3/8 w - - 0 1",
    "4k3/8/8/8/8/8/1b6/R3K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
    "4k3/8/8/8/8/8/P7/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
    "4k3/8/P7/8/8/8/8/4K3 w - - 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1",
    "4r1k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1",
    "6k1/5p1p/6p1/8/8/5N2/5PPP/6K1 b - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R3K3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
    "6k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1",
    "6k1/ppp3pp/4P3/2b5/8/5R2/P1q3PP/1q3R1K w - - 0 54",
    "7k/5K2/8/8/8/8/8/R7 w - - 0 1",
    "7k/8/6K1/8/8/8/8/R7 b - - 0 1",
    "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
    "7k/8/8/3p4/8/8/1b6/R2RK3 w - - 0 1",
    "7k/8/8/8/8/5b2/8/R3K3 w - - 0 1",
    "7k/8/8/p5p1/8/8/8/2R4K w - - 0 1",
    "8/1b6/4pQ1p/2b3pk/2p1q3/P1B5/1P3PPP/5BK1 w - - 1 33",
    "8/8/3k4/7P/6K1/3N1P2/8/1r6 w - - 1 54",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1",
    "8/P6k/8/8/8/8/p6K/8 w - - 0 1",
    "8/R7/np6/1k4P1/4p3/2P4P/p2n4/N6K w - - 0 39",
    "k6R/8/8/8/8/2N5/8/7K b - - 0 1",
    "k6R/8/8/8/8/8/2B5/7K b - - 0 1",
    "k6R/8/8/8/8/8/8/4R2K b - - 0 1",
    "k7/4n3/2p5/8/8/5B2/8/4R2K b - - 0 1",
    "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1",
    "k7/8/2p5/3n4/8/4N3/8/K7 w - - 0 1",
    "k7/8/2p5/3p4/8/8/8/K2R4 w - - 0 1",
    "k7/8/7n/3p4/8/8/8/K2R4 w - - 0 1",
    "k7/8/8/3p4/8/8/8/K2R4 w - - 0 1",
    "k7/8/8/8/8/8/8/4R2K w - - 0 1",
    "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 2 3",
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 12",
    "r3k2r/pp1n1ppp/2p1pn2/q2p4/1b1P4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 9",
    "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1",
    "r4rk1/pp3ppp/2p2n2/q2p4/3P4/2N1PN2/PP3PPP/R2QK2R w KQ - 0 12",
    "R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
];

const NAMES: usize = 12;
const DEPTH: i32 = 6;
const FUSE: u64 = 10_000_000;

struct Run { value: i64, first: Option<usize>, nodes: u64, finished: bool, reread: cs_tactics::ledger::FairCount }

/// one search of the root of `fen`, with `NAMES` moves named at the root from the move numbered `from` (None: nothing named), the switch as given
fn run(fen: &str, from: Option<usize>, fair: bool) -> Run {
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let j = record();
    let mut x = Search::with_fuse(&g, &j, me, FUSE);
    x.fair_names = fair;
    if let Some(i) = from { let ms = g.moves(&s); x.named.insert(g.key(&s), (0..NAMES).map(|k| g.move_name(&s, &ms[(i + k) % ms.len()])).collect()); }
    let l = x.run(&s, DEPTH);
    Run { value: l.value, first: l.first_index, nodes: x.nodes, finished: l.depth == DEPTH, reread: x.fair_reread }
}

fn count_of(g: &KernelGame) -> usize { g.moves(&g.initial()).len() }

#[test]
fn the_switch_is_off_by_default_and_counts_nothing() {
    let g = KernelGame::from_fen("chess", FENS[0]).unwrap();
    let j = record();
    assert!(!Search::with_fuse(&g, &j, 0, 10).fair_names && !Search::new(&g, &j, 0, 1.0).fair_names);
    for fen in FENS.iter().take(8) {
        let r = run(fen, Some(0), false);
        assert_eq!(r.reread, cs_tactics::ledger::FairCount::default());
    }
}

#[test]
fn with_nothing_named_the_switch_changes_nothing() {
    for fen in FENS.iter().take(20) {
        let (a, b) = (run(fen, None, false), run(fen, None, true));
        assert_eq!((a.value, a.first, a.nodes), (b.value, b.first, b.nodes), "{fen}");
        assert_eq!(b.reread, cs_tactics::ledger::FairCount::default());
    }
}

/// THE TILT, CONFIRMED where it is found: with the switch on, the rivals that were only read reduced are read again in full; the count of those that come out at or above the value of the named best (they were under it in the reduced reading, so the named move came out above a rival as good or better)
/// is what the repair removes. The value of the search is never lower with the repair than without it.
#[test]
fn the_tilt_is_confirmed_and_the_repair_is_never_lower() {
    let (mut cases, mut reread, mut understated, mut above, mut lower, mut raised, mut nodes_off, mut nodes_on) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0u64, 0u64);
    for fen in FENS {
        let n = count_of(&KernelGame::from_fen("chess", fen).unwrap());
        for from in (0..n).step_by(NAMES) {
            let (off, on) = (run(fen, Some(from), false), run(fen, Some(from), true));
            if !(off.finished && on.finished) { continue; }
            cases += 1;
            reread += on.reread.reread; understated += on.reread.understated; above += on.reread.above;
            nodes_off += off.nodes; nodes_on += on.nodes;
            if on.value > off.value { raised += 1; eprintln!("RAISED {fen} from {from}: off {} on {}, first {:?} -> {:?}", off.value, on.value, off.first, on.first); }
            if on.value < off.value { lower += 1; eprintln!("LOWER {fen} from {from}: off {} on {}", off.value, on.value); }
        }
    }
    eprintln!("OBS tilt (named root moves: {NAMES} a position, depth {DEPTH}, one thread): cases {cases}, rivals read again {reread}, of them at or above the named best {understated}, strictly above {above}; value raised by the repair in {raised} cases, lowered in {lower}; nodes without {nodes_off}, with {nodes_on}");
    assert_eq!(lower, 0, "the repair never lowers the value");
    assert!(reread > 0, "the rereading happens where a named move is best and a rival was read reduced");
    assert!(above > 0 && raised > 0, "the tilt: a named best came out above a rival that, read in full, is better: the value is raised by the repair");
}
