//! Plan 79.10 PART C (the names go to the search directly, SYSTEM_SPEC 8 "the wake", 7 the stake and the budget): UNIT TESTS of the direct naming in `wake::decide`: the names are given as they are, a small count a position, the answers to their largest threat first; no line of a named move is read before the search; the
//! reading for the chain of claims comes after the search where the clock allows; the clock is held; the value is not touched. Constructed positions only; no puzzle, no match.
use cs_decision::frame_rows::FrameRead;
use cs_decision::threat_verdict::Record;
use cs_decision::memory::{Parts, WakeMemory};
use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::patterns::Board;
use cs_tactics::sheet::questions::Rules;
use cs_tactics::sheet::threat_frames::FrameReadParams;
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
fn record() -> Record { Record { worth: table(), half: HashMap::new() } }
fn judge() -> Worth { Worth { name: "human".into(), table: table(), ours_first: false } }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }

fn keys_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Vec<u64> {
    let part = st.position(g, s);
    let b = Board::new(st, &part, Some(kind(g, "king")));
    let mut keys: Vec<u64> = vec![];
    for k in 1..=2 { b.connected_sets(k, &mut |its| { for key in b.keys(its, &[0, 1, 2, 3]) { if !keys.contains(&key.hash) { keys.push(key.hash); } } }); }
    keys
}
fn frame_read(g: &KernelGame, st: &Static, s: &<KernelGame as Game>::State) -> FrameRead {
    let worth: HashMap<&str, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].into_iter().collect();
    let kv: Vec<i64> = g.kind_names().iter().map(|k| worth.get(k.as_str()).copied().unwrap_or(10_000)).collect();
    FrameRead { want: keys_of(st, g, s), params: FrameReadParams { max_size: 3, fuse: 1_500, row_fuse: 30_000, answer_fuse: 1_500, conditional: true, kv: Some(kv), bands: None, rules: Rules { settle: true, progress: true, cause: true, band: true, root_cap: 0 }, deadline: None } }
}

/// the whole with the pieces of 79.10 on (`direct`) or the 79.09 flow (names read before the search, their goals on a pass)
fn whole(g: &KernelGame, st: &Static, s: &<KernelGame as Game>::State, seconds: f64, depth: i32, direct: bool, with_frames: bool) -> WakeParams {
    WakeParams { seconds, depth, pieces: Pieces { carried: with_frames, frames: with_frames, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: false, heads: false, after_move: direct, direct_names: direct },
        inputs: WakeInputs { memory: WakeMemory::given(Parts { record: Some(record()), ..Default::default() }, None), goal: Some(kind(g, "king")), frame: if with_frames { Some(frame_read(g, st, s)) } else { None }, budget: 4, ..WakeInputs::default() }, ..WakeParams::default() }
}

const POSITIONS: [&str; 3] = ["r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1"];

#[test]
fn c_the_names_go_to_the_search_with_no_line_read_before_it_the_claims_are_read_after() {
    let (g, st) = world();
    let s = state(POSITIONS[1]);
    let d = decide(&st, &g, &s, &[], &judge(), &whole(&g, &st, &s, 60.0, 4, true, false));
    assert!(!d.contributions.iter().any(|c| c.piece.contains("their lines read")), "no line of a named move is read before the search");
    let steps: Vec<(u8, &str)> = d.contributions.iter().map(|c| (c.step, c.piece)).collect();
    let named_at = steps.iter().position(|x| x.0 == 4 && x.1.contains("given to the search directly")).expect("the naming step");
    let search_at = steps.iter().position(|x| x.0 == 5).expect("the search");
    let claims_at = steps.iter().position(|x| x.0 == 7).expect("the reading for the claims");
    assert!(named_at < search_at && search_at < claims_at, "named, then searched, then read for the claims: {steps:?}");
    assert!(!d.named_mine.is_empty());
    // the move played has its own line read after the search, where the clock allows
    let mv = d.mv.clone().unwrap();
    if d.named_mine.contains(&mv) { assert!(d.log.named_mine.first().is_some_and(|k| k.0 == mv), "the move played is read first: {:?}", d.log.named_mine); }
    assert!(d.log.moves.iter().filter(|m| m.named).count() == d.named_mine.len());
}

#[test]
fn c_the_names_are_a_small_count_the_answers_first_and_what_the_count_cut_is_listed() {
    let (g, st) = world();
    for fen in POSITIONS {
        let s = state(fen);
        let mut p = whole(&g, &st, &s, 60.0, 3, true, false);
        p.inputs.named_budget = 2;
        let d = decide(&st, &g, &s, &[], &judge(), &p);
        assert!(d.named_mine.len() <= 2 && d.named_theirs.len() <= 2, "{fen}: {:?} {:?}", d.named_mine, d.named_theirs);
        let all = decide(&st, &g, &s, &[], &judge(), &whole(&g, &st, &s, 60.0, 3, true, false));
        // what the count cut is a gap, listed with the moves of the root that were named for it
        let cut = all.named_mine.len().saturating_sub(2);
        assert!(d.log.unlisted.iter().filter(|u| !u.1).count() >= cut.min(1) || all.named_mine.len() <= 2, "{fen}");
        assert_eq!(&all.named_mine[..d.named_mine.len().min(all.named_mine.len())], &d.named_mine[..d.named_mine.len().min(all.named_mine.len())], "the same order of stake, cut at the count");
    }
}

#[test]
fn c_the_value_of_the_search_is_the_same_with_the_names_given_directly() {
    let (g, st) = world();
    for fen in POSITIONS {
        let s = state(fen);
        let j = judge();
        let new = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, 60.0, 4, true, false));
        let old = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, 60.0, 4, false, false));
        assert_eq!((new.value, new.depth), (old.value, old.depth), "{fen}: the names order what is read; the value is the judge's");
        // and exact at the depth, with no forward pruning, with the names at the root and below it
        let me = g.mover(&s).unwrap();
        let mut a = Search::with_fuse(&g, &j, me, 50_000_000);
        a.prune = false;
        let mut b = Search::with_fuse(&g, &j, me, 50_000_000);
        b.prune = false;
        b.named.insert(g.key(&s), new.named_mine.clone());
        for m in &new.log.moves { if let Some(x) = &m.their_after { let c = g.apply(&s, &g.find_move(&s, &m.mv).unwrap()); b.named.insert(g.key(&c), x.names.clone()); } }
        assert_eq!(a.run(&s, 4).value, b.run(&s, 4).value);
    }
}

#[test]
fn c_the_clock_is_held_with_the_names_direct_and_the_time_by_step_is_printed() {
    let (g, st) = world();
    for (fen, secs) in [(POSITIONS[0], 0.2), (POSITIONS[1], 0.2), (POSITIONS[0], 2.0)] {
        let s = state(fen);
        let j = judge();
        let params = whole(&g, &st, &s, secs, 64, true, true);
        let t0 = std::time::Instant::now();
        let d = decide(&st, &g, &s, &[], &j, &params);
        let wall = t0.elapsed().as_secs_f64();
        let steps: Vec<String> = d.contributions.iter().map(|c| format!("step {} {} {:.3} s ({})", c.step, c.piece, c.seconds, c.note)).collect();
        println!("OBS clock held 79.10 C: clock {secs} s, one thread, wall {:.3} s, move {:?}, depth {}, {} nodes; by step: {}", wall, d.mv, d.depth, d.nodes, steps.join("; "));
        assert!(d.mv.is_some());
        assert!(wall <= secs * 1.25 + 0.03, "the whole decision ran past its clock: {wall}");
        assert!(d.contributions.iter().any(|c| c.step == 3 && c.piece.contains("after each move")), "their goals after each move were read, within the clock");
    }
}
