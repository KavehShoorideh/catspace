//! PLAN 79.17 PART D, UNIT TESTS (SYSTEM_SPEC 7, the measure of progress held as a goal; 8 the rule for choosing a move and the wall; ledger rows 13, 24, 35; piece: the wake). The held goal on constructed positions: taken up at the first decision, carried, a subgoal reached logged as an event, the move toward it played only
//! where the calculation holds it equal, dropped only for a stated cause (the class changed, no line lowers it, the end of the game found), and with no goal given the wake logs nothing of it. The log is read as words.
use cs_decision::log::MeasureLog;
use cs_decision::threat_verdict::Record;
use cs_decision::wake::{decide, MeasureGoal, Pieces, WakeDecision, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::ledger::Worth;
use cs_tactics::measure::Term;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

type St = <KernelGame as Game>::State;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn state(fen: &str) -> St { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn kind(g: &KernelGame, name: &str) -> u8 { g.kind_names().iter().position(|k| k == name).unwrap() as u8 }
fn play(g: &KernelGame, s: &St, name: &str) -> St { let m = g.moves(s).into_iter().find(|m| g.move_name(s, m) == name).unwrap_or_else(|| panic!("no move {name}")); g.apply(s, &m) }

fn params(g: &KernelGame, seconds: f64, measure: Option<MeasureGoal>, previous: Option<cs_decision::carry::Carried>) -> WakeParams {
    let pieces = Pieces { goals: true, threats: true, direct_names: true, ..Pieces::old_player() };
    WakeParams { seconds, depth: 64, pieces, inputs: WakeInputs { record: Record { worth: table(), half: HashMap::new() }, goal: Some(kind(g, "king")), measure, previous, ..WakeInputs::default() }, ..WakeParams::default() }
}
fn goal(g: &KernelGame, text: &str, d: u32) -> MeasureGoal { let kk = kind(g, "king"); MeasureGoal { term: Term::parse(&text.replace("K", &kk.to_string())).unwrap(), d } }
fn ml(d: &WakeDecision) -> &MeasureLog { d.log.measure.as_ref().expect("the measure was given: the log has its entry") }

const START: &str = "7k/8/8/8/8/8/R7/K7 w - - 0 1";

#[test]
fn with_no_goal_given_the_wake_logs_nothing_of_it_and_carries_nothing() {
    let (g, st) = world();
    let s = state(START);
    let d = decide(&st, &g, &s, &[], &Worth { name: "r".into(), table: table(), ours_first: false }, &params(&g, 1.0, None, None));
    assert!(d.log.measure.is_none());
    assert!(d.carried.as_ref().is_none_or(|c| c.held.is_none()));
    assert!(d.log.to_json()["measure"].is_null());
    assert!(!d.contributions.iter().any(|c| c.piece.starts_with("the held goal")));
}

#[test]
fn the_goal_is_taken_up_named_and_carried_and_the_event_is_logged_when_the_value_falls() {
    let (g, st) = world();
    let judge = Worth { name: "r".into(), table: table(), ours_first: false };
    let mg = goal(&g, "region(1,K)", 2);
    let mut s = state(START);
    let mut prev = None;
    let mut reached = 0;
    for ply in 0..6 {
        let d = decide(&st, &g, &s, &[], &judge, &params(&g, 1.0, Some(mg.clone()), prev.clone()));
        let m = ml(&d);
        if ply == 0 {
            assert!(m.said.contains("taken up"), "{}", m.said);
            assert!(m.held_before.is_none());
        } else {
            assert!(m.held_before.is_some());
        }
        if m.reached { reached += 1; }
        println!("OBS wake_measure ply {ply}: value {:?} held {:?} reached {} (total {}), first {:?} in {:?} moves, played toward {}, move {:?}; {}", m.value_now, m.held_after, m.reached, m.reached_total, m.first, m.moves, m.played_toward, d.mv, m.said);
        assert!(d.carried.as_ref().is_some_and(|c| c.held.is_some()), "the goal is carried with the decision");
        if m.dropped.is_some() { break; }
        if m.played_toward { assert_eq!(d.mv, m.first); assert!(d.log.why.starts_with("toward the held goal")); }
        let mv = d.mv.clone().expect("a move");
        s = play(&g, &s, &mv);
        // the other side answers with its first legal move (a fixed defender: the test is of the goal, not of the defence)
        let Some(reply) = g.moves(&s).first().map(|m| g.move_name(&s, m)) else { break };
        s = play(&g, &s, &reply);
        prev = d.carried.clone();
    }
    println!("OBS wake_measure: subgoals reached over the line: {reached}");
}

#[test]
fn a_goal_is_dropped_when_the_class_changes_and_the_cause_is_logged_and_kept() {
    let (g, st) = world();
    let judge = Worth { name: "r".into(), table: table(), ours_first: false };
    let mg = goal(&g, "region(1,K)", 2);
    let d1 = decide(&st, &g, &state(START), &[], &judge, &params(&g, 1.0, Some(mg.clone()), None));
    let held = d1.carried.clone().unwrap();
    assert!(held.held.as_ref().unwrap().dropped.is_none());
    // a unit of ours is gone: the class is not the class the goal was taken up in
    let d2 = decide(&st, &g, &state("7k/8/8/8/8/8/8/K7 w - - 0 1"), &[], &judge, &params(&g, 1.0, Some(mg.clone()), Some(held.clone())));
    let m = ml(&d2);
    assert!(m.dropped.as_ref().is_some_and(|c| c.starts_with("the class changed")), "{:?}", m.dropped);
    assert!(m.first.is_none() && !m.played_toward);
    println!("OBS wake_measure: {}", m.said);
    // dropped for good: the next decision, back in the first class, says it was dropped earlier and names nothing
    let d3 = decide(&st, &g, &state(START), &[], &judge, &params(&g, 1.0, Some(mg), d2.carried.clone()));
    assert!(ml(&d3).said.contains("dropped earlier") && ml(&d3).first.is_none());
}

/// negative control: a term no line of ours can lower (the turn is 0 wherever it is our turn): the question reads to its end, finds nothing and the goal is dropped, with the cause
#[test]
fn control_a_term_nothing_can_lower_is_dropped_with_its_cause_after_a_question_read_to_its_end() {
    let (g, st) = world();
    let judge = Worth { name: "r".into(), table: table(), ours_first: false };
    let mg = MeasureGoal { term: Term::parse("turn").unwrap(), d: 2 };
    let d1 = decide(&st, &g, &state(START), &[], &judge, &params(&g, 2.0, Some(mg), None));
    let m = ml(&d1);
    assert!(m.first.is_none() && !m.cut, "the question is read to its end: {}", m.said);
    assert!(m.dropped.as_ref().is_some_and(|c| c.starts_with("no line of at most 2 of our moves lowers")), "{:?}", m.dropped);
    println!("OBS wake_measure control: {}", m.said);
}

/// the search finds the end of the game: it is played and the goal is dropped for that cause; the measure never overrides the calculation
#[test]
fn the_end_of_the_game_found_by_the_search_is_played_and_the_goal_dropped() {
    let (g, st) = world();
    let judge = Worth { name: "r".into(), table: table(), ours_first: false };
    let mg = goal(&g, "region(1,K)", 2);
    let d = decide(&st, &g, &state("7k/R7/6K1/8/8/8/8/8 w - - 0 1"), &[], &judge, &params(&g, 2.0, Some(mg), None));
    assert_eq!(d.mv.as_deref(), Some("a7a8"));
    let m = ml(&d);
    assert!(m.dropped.as_ref().is_some_and(|c| c.starts_with("the end of the game is found by the search")), "{:?} {}", m.dropped, m.said);
    assert!(!m.played_toward);
    println!("OBS wake_measure: {}", m.said);
}
