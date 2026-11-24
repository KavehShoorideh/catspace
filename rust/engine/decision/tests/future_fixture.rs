//! PLAN 79.51 (SYSTEM_SPEC 7 "Future prediction: one channel, two sub-layers, and the decision to stop"; piece: tactical reasoning, opportunity policy; ledger row 50): THE FIXTURE, Kaveh's position of 2026-11-24, read by the channel. The expected printout
//! is his reading: (1) the target whose loss ends the game first: one forcing move on it exists (the short-leaping unit goes to the third row with a threat on it), it gains nothing (the unit is taken), no more force can arrive: LEFT, the
//! cause said, the forcing move KEPT; (2) the next target by worth: a long-moving unit of ours is on its row with two units of ours between: a goal with two unmet facts; the foot unit's step forward meets one and cannot be taken (it is held),
//! the other is met by the kept forcing move: after the step the goal is a THREAT NOW, of the target's worth; (3) the step is the move the goal starts with. The test prints the statements as OBS lines and asserts that structure; the second
//! half is the proof that the channel adds nothing to a decision: the same decisions with the channel on and off, field by field, at a fixed depth.
use cs_decision::future::{self, FutureParams, Stop, Sub};
use cs_decision::memory::{Parts, WakeMemory};
use cs_decision::threat_verdict::Record;
use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::ledger::Worth;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

pub const FIXTURE: &str = "6k1/pr2q1pp/1p2pp2/1Qp1nr2/7B/6P1/PP2PP1P/R4RK1 b - - 0 1";

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let mut st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", FIXTURE] {
        let s = KernelGame::from_fen("chess", fen).unwrap().initial();
        let p = st.position(&g, &s);
        st.learn(&p);
    }
    (g, st)
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn record() -> Record { Record { worth: table(), half: HashMap::new() } }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn params(g: &KernelGame) -> FutureParams { FutureParams { goal: Some(kind(g, "king")), seconds: 20.0, ..FutureParams::default() } }

#[test]
fn f_the_fixture_prints_the_reading_of_the_owner() {
    let (g, st) = world();
    let s = state(FIXTURE);
    let r = future::read(&st, &g, &s, &record(), &params(&g));
    for l in r.said() { println!("OBS future_79_51 fixture: {l}"); }
    println!("OBS future_79_51 fixture: {} targets, {} goals, {} joins, {} forcing moves read ({} not read), {} line reads, {:.3} s", r.targets.len(), r.goals.len(), r.joins.len(), r.forcing_reads, r.forcing_unread, r.line_reads, r.seconds);
    // (1) the unit whose loss ends the game first: LEFT, with its cause, the forcing move kept
    let first = &r.targets[0];
    assert!(first.stake.is_infinite(), "the order of looking: the unit whose loss ends the game first");
    assert_eq!(first.sub, Sub::Both);
    match &first.stop {
        Stop::Left { near, far, kept } => {
            assert!(near.starts_with("near:") && far.starts_with("far:"), "the cause is said by each sub-layer: {near} / {far}");
            assert!(kept.iter().any(|m| m.mv == "e5f3" && m.forcing.forcing == Some(true)), "the forcing move on it is kept: {kept:?}");
        }
        other => panic!("the first target must be left: {other:?}"),
    }
    // (2) the next target by worth: a goal with two unmet facts, alive, a threat now of its worth, started by the step
    let q = r.targets.iter().find(|t| t.target.text.contains("queen")).expect("the queen is a target");
    assert_eq!(q.stake, 3583.0);
    let Stop::Kept { by, goals } = &q.stop else { panic!("the queen's stop must be kept: {:?}", q.stop) };
    assert_eq!(*by, Sub::Far);
    let goal = &r.goals[goals[0]];
    assert_eq!(goal.bearer.text, "black rook on f5");
    assert_eq!(goal.facts.len(), 2);
    assert!(goal.alive);
    let tn = goal.threat_now.as_ref().expect("a threat now");
    assert_eq!(tn.first.mv, "c5c4");
    assert!(!tn.first.lost_there(), "the step cannot be taken for nothing: it is held");
    assert!(tn.first.taken_there && tn.first.held);
    assert_eq!(tn.rest.len(), 1);
    assert_eq!(tn.rest[0].1.mv, "e5f3");
    assert!(tn.verified);
    assert_eq!(tn.worth, 3583.0);
    // the join: the forcing move kept from the left target is offered to the fact of the other unit between
    assert!(r.joins.iter().any(|j| j.mv == "e5f3" && j.from.text.contains("king")), "{:?}", r.joins);
    // the near read of the first move is made and a gain is counted only where it closes
    let n = goal.near.as_ref().expect("the near read of the first move");
    println!("OBS future_79_51 fixture: near read of {}: line {}, net {}, {}, closes {}", tn.first.mv, n.line.join(" "), n.net, n.verdict, n.closes);
}

fn whole(g: &KernelGame, future: Option<FutureParams>) -> WakeParams {
    WakeParams { seconds: 600.0, depth: 4, pieces: Pieces { carried: true, frames: true, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: true, heads: false, after_move: true, direct_names: true },
        inputs: WakeInputs { memory: WakeMemory::given(Parts { record: Some(record()), ..Default::default() }, None), goal: Some(kind(g, "king")), budget: 4, future, ..WakeInputs::default() }, ..WakeParams::default() }
}

/// THE CHANNEL ADDS NOTHING: at a fixed depth (the clock is far above what the depth needs, so the count of nodes is fixed) the decision with the channel asked and the decision without it agree in every field the decision has; only the log's
/// `future` differs, and it is there only when asked.
#[test]
fn f_the_decision_is_the_same_with_the_channel_on_and_off() {
    let (g, st) = world();
    let judge = Worth { name: "human".into(), table: table(), ours_first: false };
    let mut asked = 0;
    for fen in [FIXTURE, "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1"] {
        let s = state(fen);
        let off = decide(&st, &g, &s, &[], &judge, &whole(&g, None));
        let on = decide(&st, &g, &s, &[], &judge, &whole(&g, Some(params(&g))));
        assert!(off.log.future.is_none() && on.log.future.is_some());
        assert_eq!((off.mv.as_ref(), off.value, off.depth, off.nodes), (on.mv.as_ref(), on.value, on.depth, on.nodes), "move, value, depth and nodes at {fen}");
        assert_eq!(off.chain, on.chain, "the chain of claims at {fen}");
        assert_eq!((&off.named_mine, &off.named_theirs, &off.equal, &off.first), (&on.named_mine, &on.named_theirs, &on.equal, &on.first), "the moves named and held equal at {fen}");
        let (a, b) = (off.log.to_json(), on.log.to_json());
        let (mut a, mut b) = (a, b);
        // the seconds are the clock, not an answer; the future key is the one thing that differs
        for v in [&mut a, &mut b] { v.as_object_mut().unwrap().remove("step_seconds"); v.as_object_mut().unwrap().remove("depth_reads"); v.as_object_mut().unwrap().remove("future"); }
        assert_eq!(a, b, "every field of the decision log but the channel's at {fen}");
        asked += 1;
    }
    println!("OBS future_79_51 same: {asked} decisions at depth 4 with the channel on and off agree in the move, value, depth, nodes, chain, names and every log field but `future`");
}

/// THE LINE THAT OPENS FOR THEM TOO (the mend of the first counting, SYSTEM_SPEC 7: the other side then has no free move): our rook would bear on their queen if our knight left the file, and their queen bears on our rook just the same and nothing holds it; the
/// move that opens the line gives them the first move, so the goal is alive but is no threat now, and the channel says why.
#[test]
fn f_a_line_that_opens_for_them_too_is_no_threat_now() {
    let (g, st) = world();
    let s = state("4r1k1/3nqp2/3p1npQ/2pP2N1/4N3/5rP1/5P1P/4R1K1 w - - 0 26");
    let r = future::read(&st, &g, &s, &record(), &params(&g));
    let goal = r.goals.iter().find(|x| x.bearer.text == "white rook on e1" && x.target.text == "black queen on e7").expect("the goal on the file");
    println!("OBS future_79_51 opened: {}", goal.said);
    assert!(goal.alive && goal.threat_now.is_none());
    assert!(goal.opened_for_them.as_deref().is_some_and(|w| w.contains("they move first")), "{:?}", goal.opened_for_them);
}

/// THE CHANNEL SAYS WHAT IT READ AND WHICH SUB-LAYER MADE IT, and a gain is counted only where the near read closes: a near read exists only for a goal that is a threat now.
#[test]
fn f_every_statement_carries_its_sublayer_and_what_it_read() {
    let (g, st) = world();
    let s = state(FIXTURE);
    let r = future::read(&st, &g, &s, &record(), &params(&g));
    assert!(r.targets.iter().all(|t| t.sub == Sub::Both && !t.read.is_empty() && !t.near.read.is_empty()));
    assert!(r.goals.iter().all(|x| x.sub == Sub::Far && !x.read.is_empty()));
    let j = r.to_json();
    assert!(j["goals"].as_array().unwrap().iter().all(|x| x["sub"] == "far"));
    assert!(j["targets"].as_array().unwrap().iter().all(|x| x["sub"] == "near and far"));
    for x in &r.goals { if x.near.is_some() { assert!(x.threat_now.is_some()); } }
}
