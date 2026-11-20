//! PLAN 79.16 B (SYSTEM_SPEC 7 the budget, 8 the wall; piece: the wake): TIME NOT USED GOES TO THE SEARCH, and A, THE SEARCH THAT READS ON A FAILED FIRST MOVE, at the whole. Unit tests first (`search_seconds`, the log of the reading on), then the whole: the wall of a decision against the clock it was given at every share of the steps before the search.
use cs_decision::estimate::ChanceTable;
use cs_decision::frame_rows::FrameRead;
use cs_decision::pick::{Dir, Reason, Reasons, Side};
use cs_decision::threat_verdict::Record;
use cs_decision::memory::{Parts, WakeMemory};
use cs_decision::wake::{decide, pre_share_of, search_seconds, Pieces, WakeInputs, WakeParams, FAIL_OF_TIE, TAIL_SHARE};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::ledger::Worth;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::patterns::Board;
use cs_tactics::sheet::questions::Rules;
use cs_tactics::sheet::threat_frames::FrameReadParams;
use cs_tactics::sheet::Names;
use std::collections::HashMap;
use std::time::Instant;

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
fn judge() -> Worth { Worth { name: "human".into(), table: table(), ours_first: false } }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn keys_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Vec<u64> {
    let part = st.position(g, s);
    let b = Board::new(st, &part, Some(kind(g, "king")));
    let mut keys: Vec<u64> = vec![];
    for k in 1..=2 { b.connected_sets(k, &mut |its| { for key in b.keys(its, &[0, 1, 2, 3]) { if !keys.contains(&key.hash) { keys.push(key.hash); } } }); }
    keys.truncate(12);
    keys
}
fn frame_of(g: &KernelGame, st: &Static, s: &<KernelGame as Game>::State) -> FrameRead {
    let kv: Vec<i64> = g.kind_names().iter().map(|k| table().get(&k[..1].to_string()).copied().unwrap_or(10_000)).collect();
    FrameRead { want: keys_of(st, g, s), params: FrameReadParams { max_size: 3, fuse: 1_500, row_fuse: 30_000, answer_fuse: 1_500, conditional: true, kv: Some(kv), bands: None, rules: Rules { settle: true, progress: true, cause: true, band: true, root_cap: 0 }, deadline: None } }
}
fn reasons_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Reasons {
    let keys = keys_of(st, g, s);
    let mut reasons = vec![];
    for c in 0..keys.len() { for side in [Side::Mover, Side::Other] { for dir in [Dir::Up, Dir::Down] { reasons.push(Reason { column: c, side, dir, weight: 0.01 * (1 + c % 5) as f64 * if dir == Dir::Up { 1.0 } else { -0.5 } }); } } }
    Reasons { keys, reasons, max_size: 2, levels: vec![0, 1, 2, 3], tie_digits: 9 }
}
fn whole(g: &KernelGame, st: &Static, s: &<KernelGame as Game>::State, seconds: f64, scale: Option<f64>, read_on: bool) -> WakeParams {
    let pre = scale.map(|k| pre_share_of(cs_decision::LEDGER_SHARE, None) * k);
    WakeParams { seconds, depth: 64, pieces: Pieces::all(),
        inputs: WakeInputs { memory: WakeMemory::given(Parts { record: Some(Record { worth: table(), half: HashMap::new() }), chances: Some(ChanceTable { floor: 20, total: (100, 25), ..Default::default() }), reasons: Some(reasons_of(st, g, s)), ..Default::default() }, None), goal: Some(kind(g, "king")), frame: Some(frame_of(g, st, s)), pre_share: pre, read_on, ..WakeInputs::default() }, ..WakeParams::default() }
}

const POSITIONS: [&str; 5] = ["r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "2r3k1/pp3ppp/8/3q4/3P4/2N5/PP3PPP/2RQ2K1 w - - 0 20"];

/// unit: with the steps spending exactly their share the search has its share, as before; where they spend less, the search's deadline is moved by what is left; where they overrun it has its share and no less
#[test]
fn b_the_search_gets_what_the_steps_before_it_did_not_spend() {
    let (total, share) = (1.0, cs_decision::LEDGER_SHARE);
    let tie = (total * (1.0 - share) * cs_decision::TIE_SHARE).max(0.01);
    let pre = pre_share_of(share, None) * total;
    // spent exactly the share of the steps: the left is the whole less the pre share
    assert!((search_seconds(total, share, total - pre, tie) - total * share).abs() < 1e-9);
    // spent nothing: the search has the share and all of the pre share
    assert!((search_seconds(total, share, total, tie) - (total * share + pre)).abs() < 1e-9);
    // a half of the share spent: the half is the search's
    assert!((search_seconds(total, share, total - pre / 2.0, tie) - (total * share + pre / 2.0)).abs() < 1e-9);
    // overrun: never less than its share
    assert!((search_seconds(total, share, 0.1, tie) - total * share).abs() < 1e-9);
    // the four parts are the clock: search + tie + tail + pre
    assert!((total * share + tie + TAIL_SHARE * total + pre - total).abs() < 1e-9);
}

#[test]
fn a_the_reading_on_is_a_share_of_the_equal_readings_and_logged() {
    assert!(FAIL_OF_TIE > 0.0 && FAIL_OF_TIE < 1.0);
    let (g, st) = world();
    let (mut asked, mut outcomes) = (0, std::collections::BTreeMap::<String, usize>::new());
    for fen in POSITIONS {
        let s = state(fen);
        let secs = 0.3;
        let d = decide(&st, &g, &s, &[], &judge(), &whole(&g, &st, &s, secs, None, true));
        if let Some(f) = &d.log.failed_first {
            asked += 1;
            *outcomes.entry(f.outcome.clone()).or_default() += 1;
            let tie = (secs * (1.0 - cs_decision::LEDGER_SHARE) * cs_decision::TIE_SHARE).max(0.01);
            assert!((f.allowed - FAIL_OF_TIE * tie).abs() < 1e-9, "the log prints the share it was allowed");
            assert!(f.seconds <= f.allowed + 0.05, "it costs what it was allowed: {} of {}", f.seconds, f.allowed);
        }
        assert!(d.mv.is_some());
        // off: the first move is not asked again
        let d0 = decide(&st, &g, &s, &[], &judge(), &whole(&g, &st, &s, secs, None, false));
        assert!(d0.log.failed_first.is_none());
    }
    println!("OBS reading on 79.16 A: {asked} of {} positions asked the first move again at 0.3 s: {outcomes:?}", POSITIONS.len());
}

/// the whole: the wall of a decision is within a hundredth of a second of the clock it was given at 0.2 s (at a longer clock never over it, and under it only by the tail the claims did not need), at every share of the steps before the search (0.0 to 1.0 of the default: a share over it asks for more than the clock holds) and every clock
#[test]
fn b_the_wall_of_a_decision_is_the_clock_given_at_every_share() {
    let (g, st) = world();
    let mut worst = 0.0f64;
    let mut rows = vec![];
    let mut bad: Vec<String> = vec![];
    for secs in [0.2, 0.5] {
        for scale in [None, Some(0.0), Some(0.25), Some(0.5), Some(1.0)] {
            for fen in POSITIONS {
                let s = state(fen);
                let params = whole(&g, &st, &s, secs, scale, true);
                let t0 = Instant::now();
                let d = decide(&st, &g, &s, &[], &judge(), &params);
                let wall = t0.elapsed().as_secs_f64();
                worst = worst.max((wall - secs).abs());
                if std::env::var("SHOW_STEPS").is_ok() { println!("STEPS {secs} {:?}: {}", scale, d.contributions.iter().map(|c| format!("{}:{:.4}", c.step, c.seconds)).collect::<Vec<_>>().join(" ")); }
                rows.push(format!("clock {secs} scale {scale:?}: wall {wall:.4}"));
                assert!(d.mv.is_some());
                // over the clock by more than a hundredth never; under it by no more than a hundredth where the tail is at its smallest (the clock 0.2 s), and at a longer clock by the part of the tail the claims' reading did not need (half of it at most): the tail is a share of the clock and the choice and the claims cost what they cost
                let under = if secs <= 0.2 { 0.01 } else { 0.5 * TAIL_SHARE * secs + 0.01 };
                if wall - secs > 0.01 + 1e-9 || secs - wall > under + 1e-9 { bad.push(format!("clock {secs} s, scale {scale:?}, {}: wall {wall:.4} s", &fen[..12])); }
            }
        }
    }
    println!("OBS wall 79.16 B: out of the hundredth: {bad:?}");
    println!("OBS wall 79.16 B: worst |wall - clock| over {} decisions {worst:.4} s", rows.len());
}
