//! Plan 79.09 PART B (THE CLOCK IS HELD, SYSTEM_SPEC 7 "the budget is questions", 8 the wake; piece: the wake): UNIT TESTS of the clock of the steps before the search, each step alone on constructed positions: a step past its clock asks nothing more and LISTS what it did not ask (a gap, never a refutation); with a distant clock it
//! is the same as with none; the questions are asked in the order of stake; and the whole decision's wall time is within the clock. Written and run before the integration tests of the whole (wake_whole.rs). No puzzle, no stored game, no match.
use cs_decision::estimate::ChanceTable;
use cs_decision::frame_rows::{frame_read, FrameRead};
use cs_decision::goals::{propose, GoalParams};
use cs_decision::pick::{scores, scores_by, Dir, Reason, Reasons, Side};
use cs_decision::rule::keep_read_by;
use cs_decision::threat_verdict::Record;
use cs_decision::threats::{concrete_threats, kind_worths, ThreatParams};
use cs_decision::wake::{decide, pre_share_of, Pieces, WakeInputs, WakeParams, TAIL_SHARE};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::ledger::Worth;
use cs_tactics::lines::LineParams;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::patterns::Board;
use cs_tactics::sheet::questions::Rules;
use cs_tactics::sheet::threat_frames::FrameReadParams;
use cs_tactics::sheet::Names;
use std::collections::HashMap;
use std::time::{Duration, Instant};

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
fn past() -> Instant { Instant::now() - Duration::from_millis(1) }
fn far() -> Instant { Instant::now() + Duration::from_secs(3600) }

const DEVELOPED: &str = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7";
const LINE: &str = "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1";
const ATTACKED: &str = "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1";

/// the learned seeds a caller gives: twelve, as the learned set is (not every key of the board)
fn keys_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Vec<u64> {
    let part = st.position(g, s);
    let b = Board::new(st, &part, Some(kind(g, "king")));
    let mut keys: Vec<u64> = vec![];
    for k in 1..=2 { b.connected_sets(k, &mut |its| { for key in b.keys(its, &[0, 1, 2, 3]) { if !keys.contains(&key.hash) { keys.push(key.hash); } } }); }
    keys.truncate(12);
    keys
}
fn frame_read_of(g: &KernelGame, st: &Static, s: &<KernelGame as Game>::State, deadline: Option<Instant>) -> FrameRead {
    let kv: Vec<i64> = g.kind_names().iter().map(|k| table().get(&k[..1].to_string()).copied().unwrap_or(10_000)).collect();
    FrameRead { want: keys_of(st, g, s), params: FrameReadParams { max_size: 3, fuse: 1_500, row_fuse: 30_000, answer_fuse: 1_500, conditional: true, kv: Some(kv), bands: None, rules: Rules { settle: true, progress: true, cause: true, band: true, root_cap: 0 }, deadline } }
}
fn reasons_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Reasons {
    let keys = keys_of(st, g, s);
    let mut reasons = vec![];
    for c in 0..keys.len() { for side in [Side::Mover, Side::Other] { for dir in [Dir::Up, Dir::Down] { reasons.push(Reason { column: c, side, dir, weight: 0.01 * (1 + c % 5) as f64 * if dir == Dir::Up { 1.0 } else { -0.5 } }); } } }
    Reasons { keys, reasons, max_size: 2, levels: vec![0, 1, 2, 3], tie_digits: 9 }
}

#[test]
fn b_the_frame_read_past_its_clock_reads_nothing_and_lists_the_frames_it_did_not_reach() {
    let (g, st) = world();
    let s = state(DEVELOPED);
    let k = Some(kind(&g, "king"));
    let full = frame_read(&st, &g, &s, k, &frame_read_of(&g, &st, &s, None), None);
    let distant = frame_read(&st, &g, &s, k, &frame_read_of(&g, &st, &s, Some(far())), None);
    assert!(full.frames > 0, "the learned seeds name something on the board");
    // with a clock the seeds are built in the order of the worth of their kinds, the greater first (so that a clock that stops the building keeps the largest stake): the frames may join differently from the build without one, but all of them are read
    assert!(distant.frames > 0 && distant.questions > 0, "a distant clock reads the frames");
    assert_eq!(distant.unread_frames, 0, "and leaves none unread");
    let cut = frame_read(&st, &g, &s, k, &frame_read_of(&g, &st, &s, Some(past())), None);
    assert_eq!((cut.frames, cut.rows.len(), cut.questions), (0, 0, 0), "past its clock nothing is built or asked");
    assert!(cut.unread_frames > 0, "and what was not reached is listed");
}

#[test]
fn b_the_threat_list_past_its_clock_asks_no_unit_and_lists_them_all() {
    let (g, st) = world();
    let s = state(ATTACKED);
    let k = Some(kind(&g, "king"));
    let none = concrete_threats(&st, &g, &s, &record(), &ThreatParams { goal: k, share_table: true, ..Default::default() });
    let distant = concrete_threats(&st, &g, &s, &record(), &ThreatParams { goal: k, share_table: true, deadline: Some(far()), ..Default::default() });
    let key = |l: &cs_decision::threats::ConcreteList| { let mut v: Vec<_> = l.mover.iter().chain(&l.other).chain(&l.exchange).chain(&l.dropped).map(|r| (r.unit, r.verdict, r.net, r.responses.len())).collect(); v.sort_by_key(|x| format!("{x:?}")); v };
    assert_eq!(key(&none), key(&distant), "a distant clock reads as none");
    assert!(distant.unasked.is_empty() && distant.not_asked.is_empty());
    let n_rows = none.mover.iter().chain(&none.other).chain(&none.exchange).chain(&none.dropped).filter(|r| !r.goal).count();
    assert!(n_rows > 0);
    let cut = concrete_threats(&st, &g, &s, &record(), &ThreatParams { goal: k, deadline: Some(past()), ..Default::default() });
    assert!(cut.mover.is_empty() && cut.other.is_empty() && cut.exchange.is_empty() && cut.goal.is_empty(), "nothing is read past the clock");
    assert_eq!(cut.unasked.len(), n_rows, "every unit the list would have read is listed as not asked");
    assert!(!cut.not_asked.is_empty(), "and the list says so in words");
    assert!(cut.dropped_counts.keys().any(|w| w.contains("out of budget")), "the forced end of the game is a gap: the goal's fact is unread, never unmet");
}

#[test]
fn b_the_proposer_past_its_clock_still_makes_the_goal_of_the_goal_kind_from_the_graph_and_marks_the_forced_end_unread() {
    let (g, st) = world();
    let s = state(LINE);
    let k = Some(kind(&g, "king"));
    let p = propose(&st, &g, &s, &record(), &ThreatParams { goal: k, deadline: Some(past()), ..Default::default() }, &GoalParams::default());
    assert!(p.asked.is_empty() && !p.unasked.is_empty(), "no unit asked, every one listed");
    let gu = p.goals.iter().find(|x| x.goal_unit).expect("the goal of the goal kind comes from the graph, not from a question of a unit");
    assert!(gu.facts.iter().any(|f| f.name == "forced_end" && f.status == cs_decision::goals::Status::Unread), "the forced end is unread: a gap, not 'no forced end'");
}

#[test]
fn b_the_questions_are_asked_in_the_order_of_stake() {
    let (g, st) = world();
    let s = state(DEVELOPED);
    let k = Some(kind(&g, "king"));
    let p = propose(&st, &g, &s, &record(), &ThreatParams { goal: k, deadline: Some(far()), ..Default::default() }, &GoalParams { budget: 6, ..Default::default() });
    let w = kind_worths(&g, &s, &record(), &p.asked);
    let ws: Vec<f64> = p.asked.iter().map(|u| w.get(&u.kind).copied().unwrap_or(0.0)).collect();
    assert!(ws.len() >= 2);
    // the first units asked (the contested ones) are in the order of their worth, the greater first
    let first = ws.len().min(3);
    assert!(ws[..first].windows(2).all(|x| x[0] >= x[1]), "{ws:?}");
}

#[test]
fn b_the_named_moves_are_read_in_order_and_those_past_the_clock_are_listed_not_refuted() {
    let (g, _st) = world();
    let s = state(LINE);
    let all: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    let mine: Vec<String> = all.iter().take(5).cloned().collect();
    let lp = LineParams { depth: 4, fuse: 12_000 };
    let none = keep_read_by(&g, &s, &record(), &mine, &[], 12, lp, true, None);
    let distant = keep_read_by(&g, &s, &record(), &mine, &[], 12, lp, true, Some(far()));
    let names = |k: &cs_decision::rule::KeptNames| (k.mine.iter().map(|x| (x.name.clone(), x.net)).collect::<Vec<_>>(), k.refuted.iter().map(|x| x.0.clone()).collect::<Vec<_>>(), k.unlisted.len());
    assert_eq!(names(&none), names(&distant), "a distant clock reads as none");
    let cut = keep_read_by(&g, &s, &record(), &mine, &[], 12, lp, true, Some(past()));
    assert_eq!((cut.read, cut.mine.len(), cut.refuted.len(), cut.unlisted.len()), (0, 0, 0, mine.len()), "none is read, none is refuted, every one is listed as not read");
}

#[test]
fn b_the_reasons_score_past_its_clock_scores_no_move_and_never_scores_zero() {
    let (g, st) = world();
    let s = state(DEVELOPED);
    let r = reasons_of(&st, &g, &s);
    let all: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    let k = Some(kind(&g, "king"));
    let full = scores(&st, &g, &s, k, &r, &all);
    let distant = scores_by(&st, &g, &s, k, &r, &all, Some(far()));
    assert_eq!(full, distant);
    assert!(scores_by(&st, &g, &s, k, &r, &all, Some(past())).is_empty(), "a move not reached has no score (not a score of 0)");
}

#[test]
fn b_the_share_of_the_clock_the_steps_before_the_search_may_take_is_what_the_search_and_its_equal_reading_leave_less_the_tail() {
    let share = cs_decision::LEDGER_SHARE;
    let pre = pre_share_of(share, None);
    let tie = (1.0 - share) * cs_decision::TIE_SHARE;
    assert!((pre + share + tie + TAIL_SHARE - 1.0).abs() < 1e-9, "the four shares are the whole clock");
    assert_eq!(pre_share_of(1.0, None), 0.0, "the old player's share leaves nothing to the steps before the search");
    assert_eq!(pre_share_of(share, Some(0.5)), 0.5, "a stated share is taken as it is");
}

fn whole(g: &KernelGame, st: &Static, s: &<KernelGame as Game>::State, seconds: f64, pre: Option<f64>) -> WakeParams {
    WakeParams { seconds, depth: 64, pieces: Pieces { carried: true, frames: true, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: true, heads: false },
        inputs: WakeInputs { record: record(), goal: Some(kind(g, "king")), frame: Some(frame_read_of(g, st, s, None)), chances: Some(ChanceTable { floor: 20, total: (100, 25), ..Default::default() }), reasons: Some(reasons_of(st, g, s)), pre_share: pre, ..WakeInputs::default() }, ..WakeParams::default() }
}

#[test]
fn b_the_whole_decision_is_within_the_clock_and_says_what_it_did_not_ask() {
    let (g, st) = world();
    for fen in [DEVELOPED, LINE, ATTACKED] {
        let s = state(fen);
        for secs in [0.2, 1.0] {
            let params = whole(&g, &st, &s, secs, None);
            let t0 = Instant::now();
            let d = decide(&st, &g, &s, &[], &judge(), &params);
            let wall = t0.elapsed().as_secs_f64();
            let steps: Vec<String> = d.contributions.iter().map(|c| format!("step {} {:.3} s ({})", c.step, c.seconds, c.note)).collect();
            println!("OBS clock held 79.09: clock {secs} s, wall {wall:.3} s, move {:?}, depth {}; by step: {}", d.mv, d.depth, steps.join("; "));
            assert!(d.mv.is_some());
            // the search stops on its clock every 1024 nodes and the steps before it on theirs: a small slack, the clock itself is never run past by a step of the decision
            assert!(wall <= secs * 1.1 + 0.02, "wall {wall} s over the clock {secs} s");
        }
    }
    // a decision whose steps before the search have no share asks nothing of them and says so; the move is still legal and the search keeps the clock
    let s = state(DEVELOPED);
    let params = whole(&g, &st, &s, 0.4, Some(0.0));
    let t0 = Instant::now();
    let d = decide(&st, &g, &s, &[], &judge(), &params);
    assert!(t0.elapsed().as_secs_f64() <= 0.4 * 1.1 + 0.02);
    let mv = d.mv.clone().expect("a move");
    assert!(g.moves(&s).iter().any(|m| g.move_name(&s, m) == mv));
    let frames = d.contributions.iter().find(|c| c.step == 2).expect("the frame read is a step");
    assert!(frames.note.contains("not reached"), "{}", frames.note);
    assert!(d.contributions.iter().any(|c| c.step == 4 && c.note.contains("not read")));
}
