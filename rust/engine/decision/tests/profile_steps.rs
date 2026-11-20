//! PLAN 79.16 C (SYSTEM_SPEC 11 everything in the hot path fast; 7 the budget; piece: the wake): THE PROFILE OF THE STEPS BEFORE THE SEARCH, by question, and the proof that a mend changes no row. `profile_*` prints (never asserts a time): the work and the time of each round of the threat list on the positions of 79.15's part A (nodes, queries, seconds, ns a node); `rows_equal_*` compares the rows of the whole list with a stored digest.
use cs_decision::threat_verdict::Record;
use cs_decision::threats::{concrete_threats, ConcreteList, ThreatParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let mut st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 2 3", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "6k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1"] {
        let s = KernelGame::from_fen("chess", fen).unwrap().initial();
        let p = st.position(&g, &s);
        st.learn(&p);
    }
    (g, st)
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }

/// positions of a game's middle: developed, contested, with lines
const FENS: [&str; 16] = [
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "r1bqkbnr/2pppp2/1pn4p/6N1/6P1/PP6/2QPPPBP/RNB1K2R w KQkq - 0 9", "r1bqkbnr/5pp1/1pn1p3/3p4/2P2P1p/PpNPPQ2/6PP/R1B1K1NR w KQkq - 0 12",
    "rnb1kb1r/pp2p1pp/8/2p5/2Pq1n2/PP1P1P2/2K4P/RNBQ1BNR b kq - 4 17", "Bn2k1n1/2p2pp1/p3p3/7P/1q6/3Q4/1B2N3/RK2b2R b - - 0 30", "r2q1knr/p4p1p/np4p1/2pb1N2/8/PP1Q4/2PP1PPP/R3KBNR w KQ - 0 13",
    "rn2kb1r/6pp/8/pp1p1P1P/4b3/1Pq1P3/R1P1K1N1/2BQ1B1R b kq - 0 24", "r4b2/1p6/2k5/5q1P/2p2P2/p1P2K2/8/4B3 w - - 0 38", "r3k1n1/1pp3b1/P1n5/1B3b2/4q3/4Q2P/4K2B/RN4NR b q - 1 24",
    "1n1qkbnr/2pbpppp/8/3p4/8/2PPPP2/1B4PP/QN2KBNR b Kk - 2 11", "1n1q1bnr/r4ppp/1p3k2/pB6/P2pp1b1/BP4P1/2PPNP1P/RN1QK2R w KQ - 4 17", "2k5/5pr1/2n1pn1p/p2q4/P2P4/4PP2/6PP/1RQ1KB1R w K - 1 23",
    "rn2k1nr/ppp2pp1/1b6/P2ppbq1/1P1P3p/2P1PP1P/6P1/RNBQKBNR b KQkq - 0 11", "rnb2knr/2p2pp1/8/1p6/1q3p1p/3Pp3/PN2Q1PP/R2K1BNR w - - 0 19", "r2k2n1/3n3r/4R2p/7P/3p1p2/BR3P2/P2K4/8 w - - 0 39", "1r5r/n2k4/3Nb1P1/8/P7/K1b4P/8/R1Bn2NR w - - 17 41",
];

fn digest(l: &ConcreteList) -> String {
    let row = |r: &cs_decision::threats::ThreatRow| format!("{:?}|{:?}|{}|{:?}|{:?}|{:?}|{:?}|{}|{:?}", r.unit, r.verdict, r.class, r.net, r.line, r.ended, r.first, r.nodes, r.responses.iter().map(|x| (x.names.clone(), x.net, x.state.word(), x.nodes)).collect::<Vec<_>>());
    let mut out = vec![];
    for (n, v) in [("m", &l.mover), ("o", &l.other), ("x", &l.exchange), ("d", &l.dropped)] { for r in v { out.push(format!("{n}:{}", row(r))); } }
    out.push(format!("counts {:?} {} {} {} {}", l.dropped_counts, l.resp_read, l.resp_dropped, l.resp_sweeps, l.resp_reread));
    out.join("\n")
}

fn list_of(g: &KernelGame, st: &Static, fen: &str) -> (ConcreteList, f64) {
    let s = KernelGame::from_fen("chess", fen).unwrap().initial();
    let rec = Record { worth: table(), half: HashMap::new() };
    let king = g.kind_names().iter().position(|k| k == "king").map(|k| k as cs_game::placed::Kind);
    let tp = ThreatParams { goal: king, share_table: true, every_sides: Some(vec![0, 1]), ..Default::default() };
    let t = std::time::Instant::now();
    let l = concrete_threats(st, g, &s, &rec, &tp);
    (l, t.elapsed().as_secs_f64())
}

#[test]
fn profile_the_rounds_of_the_threat_list() {
    let (g, st) = world();
    let mut by: std::collections::BTreeMap<String, (f64, u64, u64)> = Default::default();
    let mut all = 0.0;
    for fen in FENS {
        let (l, t) = list_of(&g, &st, fen);
        all += t;
        for c in &l.cost { let e = by.entry(c.name.to_string()).or_default(); e.0 += c.seconds; e.1 += c.queries; e.2 += c.nodes; }
    }
    println!("PROFILE threat list over {} positions: {:.3} s, {:.4} s a position", FENS.len(), all, all / FENS.len() as f64);
    for (k, (s, q, n)) in &by { println!("PROFILE {k}: {:.3} s, {q} queries, {n} nodes, {:.0} ns a node, {:.5} s a query", s, if *n > 0 { s * 1e9 / *n as f64 } else { 0.0 }, if *q > 0 { s / *q as f64 } else { 0.0 }); }
}

/// the digest of every list is written once (`DIGEST_WRITE=path`) and compared afterwards (`DIGEST_READ=path`): rows equal before and after a mend
#[test]
fn rows_equal_to_the_stored_digest() {
    let (g, st) = world();
    let d: Vec<String> = FENS.iter().map(|f| digest(&list_of(&g, &st, f).0)).collect();
    let all = d.join("\n=====\n");
    if let Ok(p) = std::env::var("DIGEST_WRITE") { std::fs::write(p, &all).unwrap(); }
    if let Ok(p) = std::env::var("DIGEST_READ") { let old = std::fs::read_to_string(p).unwrap(); assert!(old == all, "the rows differ from the stored digest"); println!("OBS rows equal 79.16 C: {} positions, the digest of every list equal to the stored one", FENS.len()); }
}

/// the cost of a node of the plain search on the same positions (the floor the steps' per-node cost is read against), with the mover of a node asked once (`lean`) and as before: the same values, first moves and nodes
#[test]
fn profile_the_plain_search_a_node_and_lean_changes_no_answer() {
    let (g, _st) = world();
    let j = cs_tactics::ledger::Worth { name: "record".into(), table: table(), ours_first: false };
    let mut answers: Vec<Vec<(i64, Option<usize>, u64)>> = vec![];
    for lean in [false, true] {
        let (mut nodes, mut secs) = (0u64, 0.0);
        let mut a = vec![];
        for fen in FENS {
            let s = KernelGame::from_fen("chess", fen).unwrap().initial();
            let me = g.mover(&s).unwrap();
            let mut x = cs_tactics::ledger::Search::with_fuse(&g, &j, me, 300_000);
            x.lean = lean;
            let t = std::time::Instant::now();
            let l = x.run(&s, 6);
            secs += t.elapsed().as_secs_f64();
            nodes += x.nodes;
            a.push((l.value, l.first_index, x.nodes));
        }
        answers.push(a);
        println!("PROFILE plain search lean={lean}: {nodes} nodes, {:.3} s, {:.0} ns a node", secs, secs * 1e9 / nodes as f64);
    }
    assert_eq!(answers[0], answers[1], "lean changes no value, first move or node count");
}

/// the cost of one call of each thing a node asks of the game (nanoseconds a call)
#[test]
fn profile_the_game_calls_a_node_makes() {
    let (g, _st) = world();
    let s = KernelGame::from_fen("chess", FENS[0]).unwrap().initial();
    let n = 200_000u32;
    macro_rules! time { ($name:expr, $e:expr) => {{ let t = std::time::Instant::now(); let mut k = 0usize; for _ in 0..n { k += std::hint::black_box($e); } println!("PROFILE call {}: {:.0} ns ({k})", $name, t.elapsed().as_secs_f64() * 1e9 / n as f64); }}; }
    let ms = g.moves(&s);
    time!("moves", g.moves(&s).len());
    time!("outcome", g.outcome(&s).is_some() as usize);
    time!("mover", g.mover(&s).unwrap_or(0));
    time!("key", g.key(&s) as usize & 1);
    time!("apply", { let q = g.apply(&s, &ms[0]); g.key(&q) as usize & 1 } );
    time!("captured of every move", ms.iter().map(|m| g.captured(&s, m).len()).sum::<usize>());
    time!("changed of every move", ms.iter().map(|m| g.changed(&s, m).len()).sum::<usize>());
    time!("units", g.units(&s).len());
}
