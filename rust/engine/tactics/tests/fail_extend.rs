//! PLAN 79.16 A (SYSTEM_SPEC 8 the rule for choosing a move, 7 the budget; piece: the wake): THE SEARCH DOES NOT STOP ON A FAILED FIRST MOVE. `Search::fail_extend` (seconds; 0.0 the default, the old player's path never sets it) takes the stopped depth up again when the move the deepening stood on failed low there.
//! The cases are the shape 79.15 part A found: the search finishes depth d, its first move fails low at d+1, the clock ends before another move is found.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth, WIN};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "human".into(), table, ours_first: false }
}

const FENS: [&str; 24] = [
    "1k6/2p5/8/4p3/5B2/8/8/K7 w - - 0 1", "2kr3r/ppp2ppp/2n1bn2/4p3/4P1b1/2NP1N2/PPP1BPPP/R1B1K2R w KQ - 4 9", "2r1r1kb/5p2/p1b1p3/1pBq2pN/3Pp3/P5Q1/1PP2PP1/3R1RK1 w - - 0 25",
    "3q3k/8/8/8/3b4/2N5/8/R3K3 w - - 0 1", "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "4k3/8/8/4B3/8/8/1b6/R3K3 w - - 0 1", "6k1/ppp3pp/4P3/2b5/8/5R2/P1q3PP/1q3R1K w - - 0 54",
    "7k/8/8/3p4/8/8/1b6/R2RK3 w - - 0 1", "8/1b6/4pQ1p/2b3pk/2p1q3/P1B5/1P3PPP/5BK1 w - - 1 33", "8/R7/np6/1k4P1/4p3/2P4P/p2n4/N6K w - - 0 39", "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/pp1n1ppp/2p1pn2/q2p4/1b1P4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 9", "r4rk1/pp3ppp/2p2n2/q2p4/3P4/2N1PN2/PP3PPP/R2QK2R w KQ - 0 12", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4r1k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1", "k7/8/2p5/3n4/8/4N3/8/K7 w - - 0 1",
    "7k/8/8/8/8/5b2/8/R3K3 w - - 0 1", "4k3/8/8/8/3bN3/8/8/R3K3 w - - 0 1", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
];

/// the extension is off by default and the search's answer without it does not depend on it: a reading no clock stops is the same with it asked for
#[test]
fn off_by_default_and_a_finished_reading_is_the_same() {
    let j = record();
    for fen in FENS.iter().take(10) {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let mut a = Search::with_fuse(&g, &j, me, 10_000_000);
        assert_eq!(a.fail_extend, 0.0);
        let la = a.run(&s, 4);
        let mut b = Search::with_fuse(&g, &j, me, 10_000_000);
        b.fail_extend = 1.0;
        let lb = b.run(&s, 4);
        assert_eq!((la.value, la.first_index, la.depth, la.nodes), (lb.value, lb.first_index, lb.depth, lb.nodes), "{fen}");
        assert!(a.failed_first.is_none() && b.failed_first.is_none());
    }
}

/// with the switch off a clock-stopped search answers as the old one did: the answer of the last finished depth, nothing read again (the old player's path)
#[test]
fn switch_off_the_stopped_search_answers_the_depth_before() {
    let j = record();
    for fen in FENS {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let mut a = Search::new(&g, &j, me, 0.03);
        let l = a.run(&s, 40);
        assert!(a.failed_first.is_none(), "{fen}");
        let last_complete = a.reads.iter().filter(|r| r.complete).map(|r| r.depth).max().unwrap_or(0);
        assert_eq!(l.depth, last_complete, "{fen}");
    }
}

/// THE CASES: a clock that stops the deepening, the extension given; wherever the first move failed low the outcome is one of three and the answer follows it
#[test]
fn a_failed_first_move_is_read_on_and_the_answer_follows() {
    let j = record();
    let (mut cases, mut other, mut all_failed, mut cut) = (0, 0, 0, 0);
    for clock in [0.01, 0.02, 0.04, 0.08, 0.16] {
        for fen in FENS {
            let g = KernelGame::from_fen("chess", fen).unwrap();
            let s = g.initial();
            let me = g.mover(&s).unwrap();
            let mut x = Search::new(&g, &j, me, clock);
            x.fail_extend = 1.0;
            let l = x.run(&s, 40);
            let Some(f) = x.failed_first.clone() else { continue };
            cases += 1;
            assert!(f.outcome == "first_held" || f.outcome == "cut" || f.first_value.is_some_and(|v| v < f.goal), "the first move failed low at the depth read again ({fen})");
            assert!(f.seconds <= 1.0 + 0.1, "the extension is bounded by its seconds");
            match f.outcome {
                "other_held" => { other += 1; assert_eq!(l.depth, f.depth); assert!(l.value >= f.goal, "{fen}: the answer holds the value of the depth before"); assert_eq!(l.first_index, f.found); }
                "all_failed" => { all_failed += 1; assert_eq!(l.depth, f.depth); assert!(l.value < f.goal || l.value.abs() >= WIN - 1000); }
                "cut" | "first_held" => { cut += 1; assert_eq!(l.depth, f.depth - 1, "a cut extension and a heard first move leave the depth before's answer"); assert_eq!(l.value, f.goal); }
                o => panic!("{o}"),
            }
        }
    }
    eprintln!("OBS fail_extend cases {cases}: other held {other}, all failed {all_failed}, cut {cut}");
    assert!(cases >= 1, "no position of the set shows a failed first move at these clocks");
}

/// the extension costs at most its seconds
#[test]
fn the_extension_costs_at_most_its_seconds() {
    let j = record();
    let g = KernelGame::from_fen("chess", FENS[1]).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let mut x = Search::new(&g, &j, me, 0.05);
    x.fail_extend = 0.02;
    let t = std::time::Instant::now();
    let _ = x.run(&s, 40);
    assert!(t.elapsed().as_secs_f64() < 0.05 + 0.02 + 0.1);
}

/// PLAN 79.16 A: the positions of 79.15 part A (the first move lowered by the equal reading at 0.2 s), 36 of the 79
const PARTA: [&str; 36] = [
    "r2k2n1/3n3r/4R2p/7P/3p1p2/BR3P2/P2K4/8 w - - 0 39",
    "r1bqkbnr/2pppp2/1pn4p/6N1/6P1/PP6/2QPPPBP/RNB1K2R w KQkq - 0 9",
    "1r5r/n2k4/3Nb1P1/8/P7/K1b4P/8/R1Bn2NR w - - 17 41",
    "r1bqkbnr/5pp1/1pn1p3/3p4/2P2P1p/PpNPPQ2/6PP/R1B1K1NR w KQkq - 0 12",
    "rnb2knr/2p2pp1/8/1p6/1q3p1p/3Pp3/PN2Q1PP/R2K1BNR w - - 0 19",
    "8/6k1/5qp1/8/1n5P/1r6/2N5/3K4 w - - 0 52",
    "8/1pk5/4P1R1/2p2P2/2P5/6p1/K7/4bB2 b - - 0 50",
    "rnb1kb1r/pp2p1pp/8/2p5/2Pq1n2/PP1P1P2/2K4P/RNBQ1BNR b kq - 4 17",
    "7r/3P4/8/1k6/4K3/8/3b3p/R7 w - - 0 57",
    "Bn2k1n1/2p2pp1/p3p3/7P/1q6/3Q4/1B2N3/RK2b2R b - - 0 30",
    "8/8/7p/2b4k/8/3K2p1/6N1/8 w - - 0 60",
    "8/8/8/5k1p/P6P/2b2PB1/K5n1/8 w - - 10 54",
    "r2q1knr/p4p1p/np4p1/2pb1N2/8/PP1Q4/2PP1PPP/R3KBNR w KQ - 0 13",
    "1r2k2r/n4N2/6P1/5b2/Pb6/2n4P/2K5/R1B3NR w k - 9 37",
    "Bn1k3r/7p/3P2p1/6P1/1B5P/1N1b1Nn1/8/b2KR3 b - - 4 34",
    "8/8/2p5/2k5/1r6/b3K3/1p4p1/1R6 w - - 0 53",
    "4k1nr/7p/r4pp1/n7/3P2P1/2N2PbP/Rp1B4/4K1NR w Kk - 2 29",
    "rn2kb1r/6pp/8/pp1p1P1P/4b3/1Pq1P3/R1P1K1N1/2BQ1B1R b kq - 0 24",
    "4k3/4n3/2P5/4N3/7P/4R3/r3n3/1K6 b - - 8 46",
    "r4b2/1p6/2k5/5q1P/2p2P2/p1P2K2/8/4B3 w - - 0 38",
    "8/8/5N2/8/3k1P1p/7P/8/K7 w - - 1 50",
    "2k4B/2p5/p2p4/7P/2P5/8/3NR3/4K1N1 b - - 0 42",
    "5bnr/4k3/P4p2/2p3p1/6P1/8/1R3N1P/4K2R b K - 0 29",
    "5R2/1k6/3p4/4p1b1/4P3/K4P1r/4N3/2n5 w - - 6 44",
    "8/2k2p2/P1p3p1/8/8/5R2/N4b2/7K b - - 2 46",
    "4k1nr/2pp1p1p/4p2P/4Pn2/r4P2/P7/1R2N3/1NK4R b k - 2 30",
    "2b5/5k2/8/5p2/2pKP2p/2P4P/r3q3/8 w - - 0 51",
    "3r4/6p1/2k4p/q6P/4n3/3p4/7R/1K6 w - - 0 57",
    "r3k1n1/1pp3b1/P1n5/1B3b2/4q3/4Q2P/4K2B/RN4NR b q - 1 24",
    "7r/4n2p/2k2p1b/4p3/4P3/2K2P1P/r7/6NR w - - 1 37",
    "1r6/3Pk3/4P2R/7P/8/5N2/2bK4/n7 b - - 2 51",
    "1n2k2r/6pp/8/3p1P2/2r5/8/4N3/R3K2R b k - 1 37",
    "1n1qkbnr/2pbpppp/8/3p4/8/2PPPP2/1B4PP/QN2KBNR b Kk - 2 11",
    "1n1q1bnr/r4ppp/1p3k2/pB6/P2pp1b1/BP4P1/2PPNP1P/RN1QK2R w KQ - 4 17",
    "2k5/5pr1/2n1pn1p/p2q4/P2P4/4PP2/6PP/1RQ1KB1R w K - 1 23",
    "rn2k1nr/ppp2pp1/1b6/P2ppbq1/1P1P3p/2P1PP1P/6P1/RNBQKBNR b KQkq - 0 11",
];

/// THE CASES OF PART A: each asked at the clock as played with the extension given; the outcome follows the rules above wherever the first move failed low
#[test]
fn the_cases_of_79_15_part_a() {
    let j = record();
    let (mut cases, mut other, mut all_failed, mut cut, mut stopped) = (0, 0, 0, 0, 0);
    for fen in PARTA {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let mut x = Search::new(&g, &j, me, 0.2);
        x.fail_extend = 0.1;
        let l = x.run(&s, 40);
        if x.reads.last().is_some_and(|r| !r.complete) { stopped += 1; }
        let Some(f) = x.failed_first.clone() else { continue };
        cases += 1;
        assert!(f.seconds <= 0.1 + 0.05, "{fen}");
        match f.outcome {
            "other_held" => { other += 1; assert_eq!(l.depth, f.depth); assert!(l.value >= f.goal); assert_eq!(l.first_index, f.found); }
            "all_failed" => { all_failed += 1; assert_eq!(l.depth, f.depth); }
            "cut" | "first_held" => { cut += 1; assert_eq!(l.depth, f.depth - 1); }
            o => panic!("{o}"),
        }
    }
    eprintln!("OBS fail_extend part A positions 36: reading stopped by the clock in {stopped}, failed first move read on in {cases} (other held {other}, all failed {all_failed}, first held or cut {cut})");
    assert!(cases >= 1);
}
