//! Plan 79.08 PART C (the rule for choosing a move, SYSTEM_SPEC 8, 13 items 26 and 27): UNIT TESTS of the pure parts of the fixed rule (rule.rs): the goal with the fewest things in its way, their working threats, and the ladder among the moves the search holds equal. The loop (`wake::decide`) is tested in
//! wake_whole.rs, after every piece is in Rust. No puzzle, no stored game, no match.
use cs_decision::goals::{propose, GoalParams};
use cs_decision::rule::{Stakes, chain_top, choose, fix_moves, fixes_among, named_fixes, names_of, stake_of, their_threats, things_in_the_way};
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
    let chains = named_fixes(&p.goals, white, &Stakes::read(&g, &s, &record(), &p.goals, Some(kind(&g, "king"))));
    assert!(!chains.is_empty(), "white has a goal with a fix that is a move");
    for c in &chains {
        let near = &p.goals[c.goal];
        assert_eq!(near.side, white);
        assert!(!fix_moves(near).is_empty() && !near.unmet.is_empty());
        assert_eq!(c.moves, fix_moves(near));
        // the goal named is the nearest of its chain: no other goal with a move under the same top has fewer things in its way
        for (j, x) in p.goals.iter().enumerate().filter(|(j, x)| x.side == white && !x.unmet.is_empty() && !fix_moves(x).is_empty() && chain_top(&p.goals, *j) == c.top) {
            assert!(things_in_the_way(near) <= things_in_the_way(x), "goal {j} has fewer things in the way");
        }
    }
    // the chains stand in order of stake, the greater first
    assert!(chains.windows(2).all(|w| w[0].stake >= w[1].stake));
    // the goal unit's goal has the end of the game as a fact that names no stopper: it has at least two things in its way (the pawn that shuts the line, and the end itself)
    let gu = p.goals.iter().position(|x| x.side == white && x.goal_unit).unwrap();
    assert!(things_in_the_way(&p.goals[gu]) >= 2);
    // the stopper that shuts the line is a goal of the same form below the goal it blocks: its chain is the goal unit's, and the goal unit's chain is asked first
    let pawn = p.goals.iter().position(|x| x.side == white && x.target == unit(&g, "black", "pawn", "c6")).unwrap();
    assert!(p.goals[pawn].parent.is_some() || p.goals[pawn].depth == 0);
    assert_eq!(chain_top(&p.goals, pawn), if p.goals[pawn].parent.is_some() { gu } else { pawn });
    assert!(chains[0].stake.is_infinite(), "the goal kind has the largest stake: its chain comes first");
    assert!(p.goals[chains[0].top].goal_unit || p.goals[chains[0].top].target.kind == kind(&g, "king"));
}

/// a goal of white to copy, with its facts and fixes: the stake tests change only its place in a chain and its target
fn base_goal(p: &cs_decision::goals::Proposal, white: usize) -> cs_decision::goals::Goal {
    p.goals.iter().find(|x| x.side == white && !x.unmet.is_empty() && !fix_moves(x).is_empty() && !x.goal_unit).expect("a goal with a fix").clone()
}

#[test]
fn a_the_stake_of_a_goal_is_that_of_the_top_of_its_chain_by_the_record_the_goal_kind_above_all() {
    let (g, st) = world();
    let s = state(LINE);
    let white = role(&g, "white");
    let p = propose(&st, &g, &s, &record(), &ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() }, &GoalParams::default());
    let kinds = g.kind_names();
    let b = base_goal(&p, white);
    let mk = |k: &str, at_: &str, parent: Option<UnitRef>, depth: u8| { let mut x = b.clone(); x.target = unit(&g, "black", k, at_); x.parent = parent; x.depth = depth; x };
    let queen = unit(&g, "black", "queen", "d8");
    let goals = vec![mk("pawn", "a7", None, 0), mk("queen", "d8", None, 0), mk("king", "e8", None, 0), mk("pawn", "b7", Some(queen), 1), mk("knight", "b8", Some(unit(&g, "black", "pawn", "b7")), 2)];
    // the worths are read from the record through the game's own ledger words (the targets must stand on the board), so this test reads them on a board where they do
    let board = state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    let goals: Vec<_> = goals.into_iter().enumerate().map(|(i, mut x)| { x.target = [unit(&g, "black", "pawn", "a7"), unit(&g, "black", "queen", "d8"), unit(&g, "black", "king", "e8"), unit(&g, "black", "pawn", "b7"), unit(&g, "black", "knight", "b8")][i]; x }).collect();
    let _ = &kinds;
    let sk = Stakes::read(&g, &board, &record(), &goals, Some(kind(&g, "king")));
    assert_eq!(stake_of(&goals, 0, &sk), 475.0);
    assert_eq!(stake_of(&goals, 1, &sk), 3583.0);
    assert!(stake_of(&goals, 2, &sk).is_infinite(), "the goal kind is above every worth");
    assert_eq!(stake_of(&goals, 3, &sk), 3583.0, "a subgoal carries the stake of the goal at the top of its chain, not of its own target");
    assert_eq!(stake_of(&goals, 4, &sk), 3583.0, "also two levels down");
    // a kind the record does not hold is worth nothing: no worth is assumed
    let none = Stakes::read(&g, &board, &Record { worth: HashMap::new(), half: HashMap::new() }, &goals, Some(kind(&g, "king")));
    assert_eq!(stake_of(&goals, 1, &none), 0.0);
}

#[test]
fn a_the_fixes_are_named_chain_by_chain_in_order_of_stake_the_nearest_goal_of_each() {
    let (g, st) = world();
    let s = state(LINE);
    let white = role(&g, "white");
    let p = propose(&st, &g, &s, &record(), &ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() }, &GoalParams::default());
    let kinds = g.kind_names();
    let b = base_goal(&p, white);
    let queen = unit(&g, "black", "queen", "d8");
    let mut mk = |k: &str, at_: &str, parent: Option<UnitRef>, depth: u8, mv: &str| { let mut x = b.clone(); x.target = unit(&g, "black", k, at_); x.parent = parent; x.depth = depth; x.fix_set.clear(); x.fixes.clear(); x.facts.clear(); x.fixes.push(cs_decision::goals::Fix { kind: cs_decision::goals::FixKind::Bring, against: vec![], to: None, steps: None, moves: vec![mv.to_string()], words: "" }); x };
    // a pawn goal alone (small stake); a far goal of the queen with a stopper below it (the pawn b7) and a stopper of that (the knight c8): the three are one chain, the queen's
    let goals = vec![mk("pawn", "a7", None, 0, "m_pawn"), mk("queen", "d8", None, 0, "m_queen"), mk("pawn", "b7", Some(queen), 1, "m_stop1"), mk("knight", "c8", Some(unit(&g, "black", "pawn", "b7")), 2, "m_stop2")];
    // all four have no facts: the same things in the way (none), so the nearest of the queen's chain is the DEEPEST member (the stopper before the goal it stops)
    let board = state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    let goals: Vec<_> = goals.into_iter().enumerate().map(|(i, mut x)| { x.target = [unit(&g, "black", "pawn", "a7"), unit(&g, "black", "queen", "d8"), unit(&g, "black", "pawn", "b7"), unit(&g, "black", "knight", "b8")][i]; x }).collect();
    let sk = Stakes::read(&g, &board, &record(), &goals, Some(kind(&g, "king")));
    let _ = &kinds;
    let chains = named_fixes(&goals, white, &sk);
    assert_eq!(chains.len(), 2, "two chains: the queen's and the pawn's");
    assert_eq!(chains[0].top, 1, "the chain of the larger stake first");
    assert_eq!(chains[0].moves, vec!["m_stop2".to_string()], "a far goal gives the reading its nearest stopper only");
    assert_eq!(chains[1].moves, vec!["m_pawn".to_string()], "the small chain comes after");
    // the other side has no goal here
    assert!(named_fixes(&goals, 1 - white, &sk).is_empty());
    // the ladder takes the fixes of the first chain that has an equal move
    let eq: Vec<String> = ["x", "m_pawn", "m_stop2"].iter().map(|s| s.to_string()).collect();
    assert_eq!(fixes_among(&chains, &eq), (vec!["m_stop2".to_string()], Some(0)));
    let eq2: Vec<String> = ["x", "m_pawn"].iter().map(|s| s.to_string()).collect();
    assert_eq!(fixes_among(&chains, &eq2), (vec!["m_pawn".to_string()], Some(1)));
    assert_eq!(fixes_among(&chains, &["x".to_string()]), (vec![], None));
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

/// PLAN 79.10 B: the queen's chain of four goals (the queen, the pawn that stops it, the knight that stops the pawn) with the moves given for each, and the pawn's own chain
fn chain_goals(g: &KernelGame, st: &Static, moves: [&[&str]; 4]) -> (Vec<cs_decision::goals::Goal>, Stakes, usize) {
    let s = state(LINE);
    let white = role(g, "white");
    let p = propose(st, g, &s, &record(), &ThreatParams { goal: Some(kind(g, "king")), ..Default::default() }, &GoalParams::default());
    let b = base_goal(&p, white);
    let board = state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    let spec = [("pawn", "a7", None, 0u8), ("queen", "d8", None, 0), ("pawn", "b7", Some(unit(g, "black", "queen", "d8")), 1), ("knight", "b8", Some(unit(g, "black", "pawn", "b7")), 2)];
    let goals: Vec<_> = spec.iter().enumerate().map(|(i, (k, at_, parent, depth))| {
        let mut x = b.clone(); x.target = unit(g, "black", k, at_); x.parent = *parent; x.depth = *depth; x.fix_set.clear(); x.fixes.clear(); x.facts.clear();
        x.fixes.push(cs_decision::goals::Fix { kind: cs_decision::goals::FixKind::Bring, against: vec![], to: None, steps: None, moves: moves[i].iter().map(|m| m.to_string()).collect(), words: "" });
        x }).collect();
    let sk = Stakes::read(g, &board, &record(), &goals, Some(kind(g, "king")));
    (goals, sk, white)
}
fn names(v: &[&str]) -> Vec<String> { v.iter().map(|s| s.to_string()).collect() }

#[test]
fn b_the_next_goal_of_a_chain_follows_the_nearest_in_the_order_of_nearness_each_move_once() {
    let (g, st) = world();
    let (goals, sk, white) = chain_goals(&g, &st, [&["m_pawn"], &["m_queen", "shared"], &["m_stop1", "shared"], &["m_stop2"]]);
    let chains = named_fixes(&goals, white, &sk);
    assert_eq!(chains[0].moves, names(&["m_stop2"]), "the nearest goal of the chain: the deepest stopper");
    assert_eq!(chains[0].later.iter().map(|x| (x.0, x.1.clone())).collect::<Vec<_>>(), vec![(2, names(&["m_stop1", "shared"])), (1, names(&["m_queen"]))], "then the goals after it, nearest first; a move a nearer goal holds is not listed again");
    assert_eq!(chains[0].all_moves(), names(&["m_stop2", "m_stop1", "shared", "m_queen"]));
    assert!(chains[1].later.is_empty());
}

#[test]
fn b_a_goal_with_no_move_among_its_fixes_is_passed_over_to_the_next_goal_of_the_chain() {
    let (g, st) = world();
    let (goals, sk, white) = chain_goals(&g, &st, [&["m_pawn"], &["m_queen"], &["m_stop1"], &[]]);
    let chains = named_fixes(&goals, white, &sk);
    assert_eq!(chains[0].goal, 2, "the knight has no move: the next goal of the chain is the nearest that is asked");
    assert_eq!(chains[0].moves, names(&["m_stop1"]));
    assert_eq!(chains[0].later.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1]);
}

#[test]
fn b_the_names_go_in_rounds_so_that_many_fixes_of_the_nearest_goal_do_not_keep_the_next_goal_out() {
    let (g, st) = world();
    let many: Vec<String> = (0..7).map(|i| format!("f{i}")).collect();
    let many: Vec<&str> = many.iter().map(|s| s.as_str()).collect();
    let (goals, sk, white) = chain_goals(&g, &st, [&["m_pawn"], &["m_queen"], &["m_stop1"], &many]);
    let chains = named_fixes(&goals, white, &sk);
    // round one: the nearest goal of the chain of the largest stake (seven fixes), then the small chain's; round two: the next goal of the first chain; round three: the next
    assert_eq!(names_of(&chains, 12), names(&["f0", "f1", "f2", "f3", "f4", "f5", "f6", "m_pawn", "m_stop1", "m_queen"]));
    assert_eq!(names_of(&chains, 8), names(&["f0", "f1", "f2", "f3", "f4", "f5", "f6", "m_pawn"]), "a count that ends inside a round keeps the order of stake");
    assert_eq!(names_of(&chains, 0), Vec::<String>::new());
    assert_eq!(names_of(&[], 5), Vec::<String>::new());
    // the ladder of the choice counts the moves of every goal of a chain
    assert_eq!(fixes_among(&chains, &names(&["x", "m_queen", "m_pawn"])), (names(&["m_queen"]), Some(0)));
}
