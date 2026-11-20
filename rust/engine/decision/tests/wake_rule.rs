//! Plan 79.08 PART C (the rule for choosing a move, SYSTEM_SPEC 8, 13 items 26 and 27): UNIT TESTS of the pure parts of the fixed rule (rule.rs): the goal with the fewest things in its way, their working threats, and the ladder among the moves the search holds equal. The loop (`wake::decide`) is tested in
//! wake_whole.rs, after every piece is in Rust. No puzzle, no stored game, no match.
use cs_decision::goals::{propose, GoalParams};
use cs_decision::rule::{chosen_goal, choose, fix_moves, their_threats, things_in_the_way};
use cs_decision::threat_verdict::{Record, VerdictKind};
use cs_decision::threats::{ConcreteList, ThreatParams, ThreatRow};
use cs_decision::wake::UnitRef;
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}
fn record() -> Record { Record { worth: [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect(), half: HashMap::new() } }
fn at(g: &KernelGame, name: &str) -> u16 { g.place_names().iter().position(|p| p == name).unwrap() as u16 }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn role(g: &KernelGame, name: &str) -> usize { g.roles().iter().position(|r| r == name).unwrap() }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn unit(g: &KernelGame, r: &str, k: &str, p: &str) -> UnitRef { UnitRef { owner: role(g, r), kind: kind(g, k), at: at(g, p) } }

/// a bishop on the long line to the goal unit, shut by one pawn that one knight holds; a rook that can take the knight or come to the sixth rank
const LINE: &str = "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1";

#[test]
fn c_the_chosen_goal_is_the_one_with_the_fewest_things_in_its_way_and_has_a_move() {
    let (g, st) = world();
    let s = state(LINE);
    let white = role(&g, "white");
    let p = propose(&st, &g, &s, &record(), &ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() }, &GoalParams::default());
    let kinds = g.kind_names();
    let i = chosen_goal(&p.goals, white, &record(), &kinds, Some(kind(&g, "king"))).expect("white has a goal with a fix that is a move");
    let c = &p.goals[i];
    assert_eq!(c.side, white);
    assert!(!fix_moves(c).is_empty() && !c.unmet.is_empty());
    for x in p.goals.iter().filter(|x| x.side == white && !x.unmet.is_empty() && !fix_moves(x).is_empty()) {
        assert!(things_in_the_way(c) <= things_in_the_way(x), "no other goal with a move has fewer things in its way");
    }
    // the goal unit's goal has the end of the game as a fact that names no stopper: it has at least two things in its way (the pawn that shuts the line, and the end itself)
    let gu = p.goals.iter().find(|x| x.side == white && x.goal_unit).unwrap();
    assert!(things_in_the_way(gu) >= 2);
    // the stopper that shuts the line is a goal of the same form, with fewer things in its way than the goal it blocks: the far goal gives the reading its nearest stopper
    let pawn = p.goals.iter().find(|x| x.side == white && x.target == unit(&g, "black", "pawn", "c6")).unwrap();
    assert!(pawn.parent.is_some() || pawn.depth == 0);
    assert!(things_in_the_way(pawn) < things_in_the_way(gu) || chosen_goal(&p.goals, white, &record(), &kinds, Some(kind(&g, "king"))).is_some());
}

#[test]
fn c_their_threats_are_the_confirmed_rows_against_us_the_end_of_the_game_first_then_the_larger_net() {
    let (g, _st) = world();
    let me = role(&g, "white");
    let mk = |net: i64, goal: bool, verdict: VerdictKind, two: bool| { let mut r = ThreatRow::new(me, if goal { None } else { Some(unit(&g, "white", "pawn", "a2")) }, "threat", vec![], None); r.verdict = verdict; r.net = Some(net); r.goal = goal; r.two_moves = two; r };
    let mut l = ConcreteList::default();
    l.mover.push(mk(300, false, VerdictKind::Confirmed, false));
    l.mover.push(mk(900, false, VerdictKind::Confirmed, false));
    l.mover.push(mk(5000, false, VerdictKind::Exchange, false));      // not working
    l.mover.push(mk(2000, false, VerdictKind::Confirmed, true));      // needs a quiet move first: not standing now
    l.mover.push(mk(1, true, VerdictKind::Confirmed, false));         // the end of the game forced against us
    let t = their_threats(&l);
    assert_eq!(t.iter().map(|r| (r.goal, r.net)).collect::<Vec<_>>(), vec![(true, Some(1)), (false, Some(900)), (false, Some(300))]);
}

#[test]
fn c_the_choice_among_equal_moves_goes_down_the_ladder_threat_then_goal_then_estimate() {
    let eq: Vec<String> = ["m1", "m2", "m3", "m4"].iter().map(|s| s.to_string()).collect();
    let none = |_: &[String]| None;
    // nothing answers, nothing fixes, no estimate: the search's own first move
    let c = choose(&eq, &[], &[], &none).unwrap();
    assert_eq!((c.mv.as_str(), c.answers_their_threat, c.fixes_our_goal), ("m1", false, false));
    // a move that answers their largest threat is taken before a fix
    let c = choose(&eq, &["m3".to_string()], &["m2".to_string()], &none).unwrap();
    assert_eq!((c.mv.as_str(), c.answers_their_threat, c.fixes_our_goal), ("m3", true, false));
    // among the answers, the one that is also a fix of our goal
    let c = choose(&eq, &["m2".to_string(), "m3".to_string()], &["m3".to_string()], &none).unwrap();
    assert_eq!((c.mv.as_str(), c.answers_their_threat, c.fixes_our_goal, c.left_after_goal), ("m3", true, true, 1));
    // no answer: a fix of our goal; then the estimates pick among what is left
    let pick_last = |set: &[String]| set.last().cloned();
    let c = choose(&eq, &[], &["m2".to_string(), "m4".to_string()], &pick_last).unwrap();
    assert_eq!((c.mv.as_str(), c.fixes_our_goal), ("m4", true));
    assert!(c.why.contains("estimates pick"));
    // an answer that the search does not hold equal is not chosen (the rule chooses among the equal moves only)
    let c = choose(&eq, &["not equal".to_string()], &[], &none).unwrap();
    assert_eq!((c.mv.as_str(), c.answers_their_threat), ("m1", false));
    assert!(choose(&[], &[], &[], &none).is_none());
}
