//! Plan 79.08 PART D (INTEGRATION TESTS of the whole wake, SYSTEM_SPEC 8 all seven steps, "the whole is judged, not the piece"): the wake end to end on constructed positions. Written and run only after every piece is in Rust and unit tested (parts A to C). No puzzle, no stored game, no match.
//! The tests: it returns a move, a chain and a log; with every piece off it is the old player move for move; the value of the search is the same with and without the pieces and with and without the estimates (rule 7: nothing here adds to a value); the carry holds a claim and reads the others; the decision's time by
//! step is printed as OBS lines.
use cs_decision::frame_rows::FrameRead;
use cs_decision::pick::{Dir, Reason, Reasons, Side};
use cs_decision::estimate::ChanceTable;
use cs_decision::threat_verdict::Record;
use cs_decision::wake::{decide, Pieces, WakeDecision, WakeInputs, WakeParams};
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
fn reasons_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Reasons {
    let keys = keys_of(st, g, s);
    let mut reasons = vec![];
    for c in 0..keys.len() { for side in [Side::Mover, Side::Other] { for dir in [Dir::Up, Dir::Down] { reasons.push(Reason { column: c, side, dir, weight: 0.01 * (1 + c % 5) as f64 * if dir == Dir::Up { 1.0 } else { -0.5 } }); } } }
    Reasons { keys, reasons, max_size: 2, levels: vec![0, 1, 2, 3], tie_digits: 9 }
}
fn chances() -> ChanceTable { ChanceTable { floor: 20, total: (100, 25), ..Default::default() } }

/// every piece of the whole on, the inputs a caller would give
fn whole(g: &KernelGame, st: &Static, s: &<KernelGame as Game>::State, seconds: f64, depth: i32) -> WakeParams {
    WakeParams { seconds, depth, pieces: Pieces { carried: true, frames: true, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: true, heads: false },
        inputs: WakeInputs { record: record(), goal: Some(kind(g, "king")), frame: Some(frame_read(g, st, s)), chances: Some(chances()), reasons: Some(reasons_of(st, g, s)), budget: 4, ..WakeInputs::default() }, ..WakeParams::default() }
}

/// a quiet developed position, a position with a bishop on the long line to the goal unit (a pawn shuts it), and a position where the other side attacks a unit of ours
const POSITIONS: [&str; 3] = ["r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1"];

#[test]
fn d_the_whole_wake_returns_a_move_a_chain_and_a_log() {
    let (g, st) = world();
    for fen in POSITIONS {
        let s = state(fen);
        let d: WakeDecision = decide(&st, &g, &s, &[], &judge(), &whole(&g, &st, &s, 60.0, 4));
        let mv = d.mv.clone().expect("a move");
        assert!(g.moves(&s).iter().any(|m| g.move_name(&s, m) == mv), "the move is legal");
        assert_eq!(d.chain[0].by, "search");
        assert!(!d.chain[0].line.is_empty());
        assert!(d.unported.is_empty());
        let l = &d.log;
        assert_eq!(l.chosen.as_deref(), Some(mv.as_str()));
        assert_eq!(l.moves.len(), g.moves(&s).len(), "one record for every candidate move at the root");
        assert!(l.moves.iter().any(|m| m.mv == mv && m.equal), "the chosen move is one the search holds equal");
        assert!(l.moves.iter().all(|m| m.reasons_score.is_some()), "the reasons' score is logged for each candidate");
        assert!(!l.goals.is_empty() && !l.why.is_empty());
        assert!(d.contributions.iter().any(|c| c.step == 3) && d.contributions.iter().any(|c| c.step == 5));
        let j = d.to_json();
        assert!(j["log"]["moves"].as_array().is_some() && j["steps"].as_array().is_some());
    }
}

#[test]
fn d_with_every_piece_off_the_wake_is_the_old_player_move_for_move() {
    let (g, st) = world();
    for fen in POSITIONS {
        let s = state(fen);
        let me = g.mover(&s).unwrap();
        let j = judge();
        let mut plain = Search::with_fuse(&g, &j, me, 50_000_000);
        let want = plain.run(&s, 5);
        // the inputs are given but no piece is on: nothing of them is read
        let p = WakeParams { seconds: 60.0, depth: 5, pieces: Pieces::old_player(), inputs: whole(&g, &st, &s, 60.0, 5).inputs, ..WakeParams::default() };
        let got = decide(&st, &g, &s, &[], &j, &p);
        assert_eq!((got.mv.clone(), got.value, got.depth), (want.first.clone(), want.value, want.depth));
        assert!(got.proposal.is_none() && got.named_mine.is_empty() && got.named_theirs.is_empty() && got.log.moves.is_empty());
    }
}

#[test]
fn d_the_value_of_the_search_is_the_same_with_and_without_the_pieces_and_the_estimates() {
    let (g, st) = world();
    for fen in POSITIONS {
        let s = state(fen);
        let me = g.mover(&s).unwrap();
        let j = judge();
        let with = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, 60.0, 4));
        // the value of the position with the moves the wake named, read without any forward pruning (exact at the depth): the same as without the names
        let mut a = Search::with_fuse(&g, &j, me, 50_000_000);
        a.prune = false;
        let mut b = Search::with_fuse(&g, &j, me, 50_000_000);
        b.prune = false;
        b.named.insert(g.key(&s), with.named_mine.clone());
        b.named_ply.insert(1, with.named_theirs.clone());
        let (va, vb) = (a.run(&s, 4).value, b.run(&s, 4).value);
        assert_eq!(va, vb, "the named moves change the order and never the value");
        // the estimates (the chance of a goal, the reasons) change neither the search's value nor the set it holds equal; they may change the choice among the equal
        let mut p2 = whole(&g, &st, &s, 60.0, 4);
        p2.inputs.chances = None;
        p2.inputs.reasons = None;
        let without = decide(&st, &g, &s, &[], &j, &p2);
        assert_eq!((with.value, with.depth, with.equal.clone()), (without.value, without.depth, without.equal.clone()));
        assert!(without.log.notes.iter().any(|n| n.contains("no reasons given")) && without.log.notes.iter().any(|n| n.contains("no table of the chance")), "what is missing is said");
        assert!(without.log.moves.iter().all(|m| m.reasons_score.is_none() && m.chance.is_none()));
    }
}

#[test]
fn d_the_carry_holds_a_claim_whose_read_set_did_not_change_and_logs_it() {
    let (g, st) = world();
    let s = state(POSITIONS[1]);
    let j = judge();
    let first = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, 60.0, 3));
    let prev = first.carried.clone().expect("the open tactics to carry");
    assert!(!prev.rows.is_empty(), "the frame read made rows with read sets");
    // the same position again: every row with a read set is held, none is read again
    let mut p = whole(&g, &st, &s, 60.0, 3);
    p.inputs.previous = Some(prev.clone());
    let second = decide(&st, &g, &s, &[], &j, &p);
    let step = second.contributions.iter().find(|c| c.step == 1).expect("the carry is a step");
    println!("OBS carry 79.08: {} rows carried from the last decision; {}", prev.rows.len(), step.note);
    assert!(step.note.starts_with(&format!("{} rows held, 0 read again", prev.rows.len())), "{}", step.note);
    assert_eq!(first.mv, second.mv, "the same position decides the same way");
}

#[test]
fn d_the_other_sides_attack_on_a_unit_of_ours_is_a_threat_the_log_names() {
    let (g, st) = world();
    // the black bishop on d5 bears on the white knight on e2? no: on the long diagonal to a2/g8; the white rook a1 is attacked by nothing. A real attack: the black bishop takes the knight on e4's place.
    let s = state("4k3/8/8/8/3bN3/8/8/R3K3 w - - 0 1");
    let d = decide(&st, &g, &s, &[], &judge(), &whole(&g, &st, &s, 60.0, 4));
    assert!(d.log.rows.iter().any(|r| r.list == "against us"), "a unit of ours is borne on and the list says so: {:?}", d.log.rows.iter().map(|r| (&r.list, &r.unit, &r.verdict)).collect::<Vec<_>>());
    assert!(d.mv.is_some());
}

#[test]
fn d_obs_the_time_of_the_whole_decision_by_step() {
    let (g, st) = world();
    for (fen, secs) in [(POSITIONS[0], 0.2), (POSITIONS[1], 0.2), (POSITIONS[0], 2.0)] {
        let s = state(fen);
        let j = judge();
        let t0 = std::time::Instant::now();
        let d = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, secs, 64));
        let steps: Vec<String> = d.contributions.iter().map(|c| format!("step {} {} {:.3} s ({})", c.step, c.piece, c.seconds, c.note)).collect();
        println!("OBS whole decision 79.08: clock {secs} s, one thread, wall {:.3} s, move {:?}, depth {}, {} nodes; by step: {}", t0.elapsed().as_secs_f64(), d.mv, d.depth, d.nodes, steps.join("; "));
        let old = {
            let me = g.mover(&s).unwrap();
            let t1 = std::time::Instant::now();
            let mut ls = Search::new(&g, &j, me, secs * cs_decision::LEDGER_SHARE);
            let l = ls.run(&s, 64);
            (l.first, l.depth, t1.elapsed().as_secs_f64())
        };
        println!("OBS old player 79.08: clock {secs} s: move {:?}, depth {}, {:.3} s", old.0, old.1, old.2);
        assert!(d.mv.is_some());
    }
}
