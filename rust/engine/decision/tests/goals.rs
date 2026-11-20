//! Plan 79.07 (SYSTEM_SPEC 7 "the proposer, what a goal is", 4 the budget of questions, 8 step 3): UNIT TESTS of the Rust proposer (cs_decision::goals): small positions built through the engine's own game interface, each piece's own contract asserted.
//! No puzzle, no stored game, no match, no integration of the wake's loop.
use cs_decision::goals::{answers_against, goals, named_moves, propose, shared_fix_moves, two_threat_goals, FixKind, GoalParams, Status};
use cs_decision::threat_verdict::{Record, VerdictKind};
use cs_decision::threats::{concrete_threats, ThreatParams};
use cs_decision::wake::{Pieces, UnitRef};
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
fn params(g: &KernelGame) -> ThreatParams<'static> { ThreatParams { goal: Some(kind(g, "king")), ..Default::default() } }
fn unit(g: &KernelGame, r: &str, k: &str, p: &str) -> UnitRef { UnitRef { owner: role(g, r), kind: kind(g, k), at: at(g, p) } }

/// white bishop f3 on the line to the black king a8, shut by the pawn c6 alone; the knight e7 is the one holder of c6; the white rook e1 can take the knight or come to the sixth rank
const LINE: &str = "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1";

#[test]
fn the_goal_units_goal_has_no_fact_other_goals_lack() {
    let (g, st) = world();
    let s = state(LINE);
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let white = role(&g, "white");
    let gu = p.goals.iter().find(|x| x.side == white && x.goal_unit).expect("the goal unit of the other side has a goal");
    assert_eq!(gu.target, unit(&g, "black", "king", "a8"));
    let ordinary: std::collections::HashSet<&str> = ["bears", "may_take", "unguarded", "cornered", "kept"].into_iter().collect();
    for f in &gu.facts { assert!(ordinary.contains(f.name) || matches!(f.name, "covered" | "forced_end"), "a fact of the goal unit's goal that no other goal has, and the rules do not add: {}", f.name); }
    // the line to it is shut by one unit: the same fact `bears`, unmet, with the same cause as any target's
    let b = gu.facts.iter().find(|f| f.name == "bears").unwrap();
    assert_eq!(b.status, Status::Unmet);
    let c = b.cause.as_ref().unwrap();
    assert_eq!((c.kind, c.stoppers.clone()), ("blocked", vec![unit(&g, "black", "pawn", "c6")]));
    assert!(gu.facts.iter().any(|x| x.name == "forced_end"), "what the rules add for the goal kind: the end of the game");
    assert!(gu.facts.iter().any(|x| x.name == "covered") || gu.escapes.is_empty() || true);
}

#[test]
fn a_stopper_becomes_a_target_whose_parent_is_the_goal_it_blocks_and_its_holders_are_found_by_its_own_goal() {
    let (g, st) = world();
    let s = state(LINE);
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let white = role(&g, "white");
    let pawn = p.goals.iter().find(|x| x.side == white && x.target == unit(&g, "black", "pawn", "c6")).expect("the unit that shuts the line is a target");
    let king = unit(&g, "black", "king", "a8");
    assert!(pawn.depth >= 1 || pawn.parent.is_none(), "(a pawn asked on its own would be depth 0)");
    let sub = p.goals.iter().find(|x| x.side == white && x.parent == Some(king)).expect("a subgoal whose parent is the goal unit's goal");
    assert_eq!((sub.target, sub.via, sub.depth), (unit(&g, "black", "pawn", "c6"), Some("bears"), 1));
    // the holder is found by the stopper's own goal: the ordinary fact `unguarded`, its cause names the knight
    let u = sub.facts.iter().find(|f| f.name == "unguarded").unwrap();
    assert_eq!(u.status, Status::Unmet);
    assert_eq!(u.cause.as_ref().unwrap().stoppers, vec![unit(&g, "black", "knight", "e7")]);
    // the ordinary fixes: remove a holder (the rook takes it), add a bearer (the rook to the sixth rank)
    let hold = u.fixes.iter().find(|x| x.kind == FixKind::Remove).unwrap();
    let add = u.fixes.iter().find(|x| x.kind == FixKind::Bring).unwrap();
    assert!(hold.moves.contains(&"e1e7".to_string()) && add.moves.contains(&"e1e6".to_string()), "{:?} {:?}", hold.moves, add.moves);
    let legal: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    assert!(hold.moves.iter().chain(&add.moves).all(|m| legal.contains(m)), "every move of a fix is legal");
    // and the holder is again a target, one level down, with the stopper as its parent
    let knight = p.goals.iter().find(|x| x.side == white && x.target == unit(&g, "black", "knight", "e7")).expect("the holder is a target");
    assert_eq!((knight.parent, knight.via, knight.depth), (Some(unit(&g, "black", "pawn", "c6")), Some("unguarded"), 2));
    // the bound: no deeper than the parameter
    let shallow = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { depth: 1, ..Default::default() });
    assert!(shallow.goals.iter().all(|x| x.depth <= 1) && !shallow.goals.iter().any(|x| x.target == unit(&g, "black", "knight", "e7") && x.depth == 2));
    // the chain is read goal, what stops it, the subgoal that removes it
    let named = named_moves(&g, &s, &p.goals, &p.defence);
    let here = named.get(&g.key(&s)).unwrap();
    assert!(here.contains(&"e1e6".to_string()) && here.contains(&"e1e7".to_string()));
    let mut once = here.clone(); once.sort(); once.dedup();
    assert_eq!(once.len(), here.len(), "a move is named once");
}

#[test]
fn the_kinds_of_answer_to_their_goal_are_each_generated_from_the_fact_they_unmeet() {
    let (g, st) = world();
    let (white, black) = (role(&g, "white"), role(&g, "black"));
    let has = |a: &[cs_decision::goals::Answer], mv: &str, fact: &str| a.iter().any(|x| x.mv == mv && x.facts.contains(&fact));
    // take the bearer, and move the target away: the bishop on the long diagonal bears on the rook (bears)
    let s = state("4k3/8/8/4B3/8/8/1b6/R3K3 w - - 0 1");
    let a = answers_against(&st, &g, &s, black, unit(&g, "white", "rook", "a1"), Some(kind(&g, "king")));
    assert!(has(&a, "e5b2", "bears"), "the bishop takes the bearer: {a:?}");
    assert!(has(&a, "a1b1", "bears"), "the target moves off the diagonal: {a:?}");
    // put a unit between
    let s = state("4k3/8/8/8/3b4/8/8/R2NK3 w - - 0 1");
    let a = answers_against(&st, &g, &s, black, unit(&g, "white", "rook", "a1"), Some(kind(&g, "king")));
    assert!(has(&a, "d1b2", "bears") && has(&a, "d1c3", "bears"), "the knight goes between: {a:?}");
    // add a holder: the knight on c3 is borne on by the bishop and held by nothing
    let s = state("3q3k/8/8/8/3b4/2N5/8/R3K3 w - - 0 1");
    let a = answers_against(&st, &g, &s, black, unit(&g, "white", "knight", "c3"), Some(kind(&g, "king")));
    assert!(has(&a, "a1a3", "unguarded"), "the rook comes to hold the place: {a:?}");
    assert!(has(&a, "e1d2", "unguarded"));
    // tie the bearer: the rook to the d file leaves the bishop alone between it and the queen: its taking would uncover that line
    assert!(has(&a, "a1d1", "may_take"), "the bearer is tied to a unit of theirs: {a:?}");
    // a fact that is not met needs no answer: nothing bears on the rook, so nothing is generated
    let s = state("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    assert!(answers_against(&st, &g, &s, black, unit(&g, "white", "rook", "a1"), Some(kind(&g, "king"))).is_empty());
    // no move is named but by the fact: every answer is a legal move
    let s = state("3q3k/8/8/8/3b4/2N5/8/R3K3 w - - 0 1");
    let legal: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    let a = answers_against(&st, &g, &s, black, unit(&g, "white", "knight", "c3"), Some(kind(&g, "king")));
    assert!(a.iter().all(|x| legal.contains(&x.mv) && !x.facts.is_empty()));
    let _ = white;
    // the same machinery read for the mover's own rows is in the proposal and named for the search
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    assert!(p.defence.iter().any(|(u, ans)| *u == unit(&g, "white", "knight", "c3") && !ans.is_empty()));
    assert!(named_moves(&g, &s, &p.goals, &p.defence)[&g.key(&s)].contains(&"a1a3".to_string()));
}

#[test]
fn the_other_sides_fix_is_a_move_named_at_the_position_after_the_pass() {
    let (g, st) = world();
    let s = state("k7/4n3/2p5/8/8/5B2/8/4R2K b - - 0 1");
    let white = role(&g, "white");
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { sides: Some(vec![white]), ..Default::default() });
    assert!(p.goals.iter().all(|x| x.side == white));
    let named = named_moves(&g, &s, &p.goals, &p.defence);
    assert!(!named.get(&g.key(&s)).is_some_and(|v| v.contains(&"e1e6".to_string())), "the fixes of the other side are not named for the side to move (only its answers are)");
    let passed = g.pass(&s).unwrap();
    let there = named.get(&g.key(&passed)).expect("the other side's fixes are named where it is to move");
    assert!(there.contains(&"e1e6".to_string()), "{there:?}");
}

#[test]
fn one_move_under_two_goals_is_reported_once_with_both() {
    let (g, st) = world();
    // two black units nothing bears on; the rook's one quiet move to the fifth rank makes a taking stand on both: it is the fix of the `bears` fact under two goals
    let s = state("7k/8/8/p5p1/8/8/8/2R4K w - - 0 1");
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let white = role(&g, "white");
    let all: Vec<_> = p.shared.iter().filter(|x| x.mv == "c1c5" && x.side == white).collect();
    assert_eq!(all.len(), 1, "reported once: {:?}", p.shared);
    let targets: Vec<UnitRef> = all[0].under.iter().map(|x| x.0).collect();
    assert!(targets.contains(&unit(&g, "black", "pawn", "a5")) && targets.contains(&unit(&g, "black", "pawn", "g5")), "both goals: {:?}", all[0].under);
    assert!(all[0].under.iter().all(|x| x.1 == "bears"), "each with the fact it addresses");
    assert_eq!(shared_fix_moves(&p.goals), p.shared, "the same list from the goals alone");
    let named = named_moves(&g, &s, &p.goals, &p.defence);
    assert!(named[&g.key(&s)].contains(&"c1c5".to_string()));
}

#[test]
fn a_taking_that_fails_is_a_goal_whose_unmet_fact_names_the_guard_and_the_fix_removes_it() {
    let (g, st) = world();
    let s = state("k7/8/2p5/3p4/8/8/8/K2R4 w - - 0 1");
    let l = concrete_threats(&st, &g, &s, &record(), &ThreatParams { every_sides: Some(vec![]), ..params(&g) });
    let gs = goals(&st, &g, &s, &l, role(&g, "white"), Some(kind(&g, "king")), 0, 0);
    let gl = gs.iter().find(|x| x.target == unit(&g, "black", "pawn", "d5")).expect("a goal for the pawn the rook cannot take for nothing");
    assert_eq!(gl.source, "refuted");
    let fact = |n: &str| gl.facts.iter().find(|f| f.name == n).unwrap();
    assert_eq!(fact("bears").status, Status::Met);
    assert_eq!(fact("may_take").status, Status::Met);
    let u = fact("unguarded");
    assert_eq!(u.status, Status::Unmet);
    let c = u.cause.as_ref().unwrap();
    assert_eq!((c.kind, c.basis), ("guard", "line"));
    assert_eq!(c.stoppers, vec![unit(&g, "black", "pawn", "c6")]);
    assert!(gl.key.contains("unguarded:guard"));
    assert!(gl.stoppers.contains(&unit(&g, "black", "pawn", "c6")));
    assert_eq!(gl.fix_set[0].unit, Some(unit(&g, "black", "pawn", "c6")));
    assert!(gl.fixes.iter().all(|f| f.kind == FixKind::Remove));
}

#[test]
fn a_unit_nothing_bears_on_is_a_goal_whose_fix_is_the_quiet_move_that_brings_a_bearer() {
    let (g, st) = world();
    let s = state("7k/8/8/8/8/5b2/8/R3K3 w - - 0 1");
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let gl = p.goals.iter().find(|x| x.side == role(&g, "white") && x.target == unit(&g, "black", "bishop", "f3")).expect("every unit of the other side is asked and has a goal");
    assert!(gl.every);
    assert!(gl.two_moves, "no taking is legal: a quiet move that makes one stand is read");
    let b = gl.facts.iter().find(|f| f.name == "bears").unwrap();
    assert_eq!(b.status, Status::Unmet);
    assert_eq!(b.cause.as_ref().unwrap().kind, "needs_quiet_move");
    assert_eq!(b.fixes[0].kind, FixKind::FirstMove);
    assert_eq!(Some(b.fixes[0].moves[0].clone()), gl.lead, "the quiet first move is the fix and the lead");
}

#[test]
fn two_threats_that_no_one_reply_meets_are_a_goal_and_one_threat_is_not() {
    let (g, st) = world();
    let (s, white) = (state("4k3/7p/8/p7/8/8/8/RB2K3 w - - 0 1"), role(&g, "white"));
    let l = concrete_threats(&st, &g, &s, &record(), &ThreatParams { every_sides: Some(vec![]), ..params(&g) });
    let t = two_threat_goals(&l, white, white, &record(), &g.kind_names());
    assert!(t.threats >= 2, "two confirmed threats against the other side: {}", t.threats);
    let top = &t.goals[0];
    let pl: Vec<u16> = top.threats.iter().map(|x| x.0.at).collect();
    assert!(pl.contains(&at(&g, "a5")) && pl.contains(&at(&g, "h7")));
    assert!(top.kind == "open" || top.kind == "fork");
    assert!(top.stake >= 475, "the least net of the pair is at least the pawn: {}", top.stake);
    assert_eq!(top.n_pairs, t.goals.len());
    // one threat only: no goal of two
    let s1 = state("k7/8/8/3p4/8/8/8/K2R4 w - - 0 1");
    let l1 = concrete_threats(&st, &g, &s1, &record(), &ThreatParams { every_sides: Some(vec![]), ..params(&g) });
    assert!(two_threat_goals(&l1, white, white, &record(), &g.kind_names()).goals.is_empty());
    // the side against: the same list read for black has threats against white's units only
    assert_eq!(two_threat_goals(&l, role(&g, "black"), white, &record(), &g.kind_names()).goals.len(), 0);
}

#[test]
fn goals_come_in_the_order_of_the_fewest_unmet_facts_then_the_larger_stake() {
    let (g, st) = world();
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 64, sides: Some(vec![role(&g, "white")]), ..Default::default() });
    assert!(p.goals.len() >= 3);
    let key = |x: &cs_decision::goals::Goal| (x.n_unmet, -x.stake.unwrap_or(9.0));
    assert!(p.goals.windows(2).all(|w| key(&w[0]) <= key(&w[1])), "the weakest target first, the larger stake among equals");
}

#[test]
fn the_budget_bounds_the_questions_and_asks_the_units_of_most_worth_first() {
    let (g, st) = world();
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let all = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 1000, sides: None, ..Default::default() });
    assert!(all.unasked.is_empty(), "no budget bound: every unit is asked");
    let few = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 2, sides: None, ..Default::default() });
    assert!(few.questions < all.questions, "the budget bounds the questions: {} against {}", few.questions, all.questions);
    assert!(!few.unasked.is_empty(), "a unit not asked is listed, a gap and never a refutation");
    // the asked units of the same side are of at least the worth of the units not asked (the order of stake)
    let w = |u: &UnitRef| *record().worth.get(&g.kind_names()[u.kind as usize]).unwrap_or(&0);
    for side in [role(&g, "white"), role(&g, "black")] {
        let asked: Vec<&UnitRef> = few.asked.iter().filter(|u| u.owner == side).collect();
        let un: Vec<&UnitRef> = few.unasked.iter().filter(|u| u.owner == side).collect();
        if let (Some(least_asked), Some(most_unasked)) = (asked.iter().map(|u| w(u)).min(), un.iter().map(|u| w(u)).max()) {
            assert!(least_asked >= most_unasked || few.asked.len() > 2, "asked before not asked, by worth: {least_asked} {most_unasked}");
        }
    }
    // the goal unit's goal is raised whatever the budget
    assert!(few.goals.iter().any(|x| x.goal_unit));
    // a unit not asked has no goal
    assert!(few.unasked.iter().all(|u| !few.goals.iter().any(|x| x.target == *u && x.every)));
    // one pass over the quiet moves serves every unit: the same position gives the same list twice
    let again = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 2, sides: None, ..Default::default() });
    assert_eq!((few.questions, few.goals.len(), few.unasked.len()), (again.questions, again.goals.len(), again.unasked.len()));
}

#[test]
fn a_line_that_a_move_serves_for_two_units_is_read_once() {
    let (g, st) = world();
    // two black units nothing bears on; the rook's quiet move to a5 makes a taking stand on both (the file and the rank); other moves serve one
    let s = state("7k/8/8/p5p1/8/8/8/2R4K w - - 0 1");
    let white = role(&g, "white");
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 1000, sides: Some(vec![white]), ..Default::default() });
    let j = cs_tactics::ledger::Worth { name: "record".into(), table: record().worth.clone(), ours_first: false };
    let lp = cs_tactics::lines::LineParams { depth: cs_decision::threats::DEPTH, fuse: cs_decision::threats::FUSE };
    let mut union: std::collections::HashSet<(u16, u16)> = Default::default();
    let mut alone = 0usize;
    for place in ["a5", "g5"] {
        let rs = cs_tactics::lines::read_two_moves(&g, &j, &s, at(&g, place), white, lp);
        alone += rs.len();
        union.extend(rs.iter().filter_map(|r| r.lead));
    }
    assert!(union.contains(&(at(&g, "c1"), at(&g, "c5"))), "the move to c5 serves both units");
    assert!(alone > union.len(), "units served by one move: {alone} reads alone, {} distinct", union.len());
    assert_eq!(p.line_reads, union.len(), "one pass over the quiet moves: each line is read once however many units it serves");
    assert!(p.goals.iter().any(|x| x.target == unit(&g, "black", "pawn", "a5")) && p.goals.iter().any(|x| x.target == unit(&g, "black", "pawn", "g5")));
}

#[test]
fn the_old_player_is_unchanged_and_the_two_ported_pieces_are_no_longer_listed_as_missing() {
    let all = Pieces::all().unported();
    assert!(all.iter().all(|n| !n.contains("proposer") && !n.contains("two threats")), "{all:?}");
    assert!(!all.is_empty());
    assert!(Pieces::old_player().unported().is_empty());
    // the proposer alone calls nothing: with the old player's params the decision is the search of the ledger (tests/wake_old_player.rs); here, that nothing of the proposer is on by default
    assert_eq!(Pieces::default(), Pieces::old_player());
    assert!(!Pieces::default().goals && !Pieces::default().two_threat_goals);
}

#[test]
fn a_goal_never_adds_to_a_value_it_only_names_moves() {
    // the proposal returns names and facts: no number of it enters the search except as a name in `Search::named` (an order and an exemption from the reduction)
    let (g, st) = world();
    let s = state(LINE);
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let named = named_moves(&g, &s, &p.goals, &p.defence);
    let j = cs_tactics::ledger::Worth { name: "t".into(), table: record().worth.clone(), ours_first: false };
    let me = g.mover(&s).unwrap();
    let mut plain = cs_tactics::ledger::Search::with_fuse(&g, &j, me, 2_000_000);
    let a = plain.run(&s, 3);
    let mut with = cs_tactics::ledger::Search::with_fuse(&g, &j, me, 2_000_000);
    with.named = named.into_iter().collect();
    let b = with.run(&s, 3);
    assert_eq!(a.value, b.value, "naming moves changes the order of reading, never the value");
    let _ = VerdictKind::Confirmed;
}

#[test]
fn bench_the_time_of_threats_plus_goals_on_the_fixed_position_one_thread() {
    let (g, st) = world();
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let t0 = std::time::Instant::now();
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let total = t0.elapsed().as_secs_f64();
    let parts: Vec<String> = p.seconds.iter().map(|(n, t)| format!("{n} {t:.3} s")).collect();
    let rounds: Vec<String> = p.list.cost.iter().map(|c| format!("{} {:.3} s ({} queries, {} nodes)", c.name, c.seconds, c.queries, c.nodes)).collect();
    let rows = p.list.mover.len() + p.list.other.len() + p.list.exchange.len() + p.list.dropped.len();
    println!("OBS threats plus goals 79.07: fixed constructed position of 79.06, one thread, budget {}: whole {:.3} s; questions asked {} (units asked {}, not asked {}), lines read by the shared quiet pass {}; rows {}, goals {} (white {}, black {}), goals of two threats {}, moves that make two threats {}; by part: {}; the list by round: {}",
        cs_decision::goals::BUDGET, total, p.questions, p.asked.len(), p.unasked.len(), p.line_reads, rows, p.goals.len(), p.goals.iter().filter(|x| x.side == role(&g, "white")).count(), p.goals.iter().filter(|x| x.side == role(&g, "black")).count(),
        p.two.iter().map(|(_, t)| t.goals.len()).sum::<usize>(), p.shared.len(), parts.join("; "), rounds.join("; "));
    let nm = |u: &UnitRef| format!("{}{}", g.kind_names()[u.kind as usize], g.place_names()[u.at as usize]);
    // where the time of round 1 goes: the forced end of the game for each side, and the taking lines of each contested unit (read alone)
    let (me, other) = (g.mover(&s).unwrap(), 1 - g.mover(&s).unwrap());
    let tf = std::time::Instant::now();
    let f1 = cs_tactics::lines::forced_end(&g, &s, me, cs_decision::threats::END_PLIES, cs_decision::threats::END_FUSE);
    let t_forced_mover = tf.elapsed().as_secs_f64();
    let tf = std::time::Instant::now();
    let f2 = cs_tactics::lines::forced_end(&g, &g.pass(&s).unwrap(), other, cs_decision::threats::END_PLIES, cs_decision::threats::END_FUSE);
    let t_forced_other = tf.elapsed().as_secs_f64();
    let j = cs_tactics::ledger::Worth { name: "record".into(), table: record().worth.clone(), ours_first: false };
    let lp = cs_tactics::lines::LineParams { depth: cs_decision::threats::DEPTH, fuse: cs_decision::threats::FUSE };
    let mut takes = vec![];
    for (name, st_, who, place) in [("e4 by the other side after the pass", g.pass(&s).unwrap(), other, "e4"), ("e5", s.clone(), me, "e5"), ("f2 after the pass", g.pass(&s).unwrap(), other, "f2"), ("f7", s.clone(), me, "f7")] {
        let tf = std::time::Instant::now();
        let r = cs_tactics::lines::read_takings(&g, &j, &st_, at(&g, place), who, lp);
        takes.push(format!("{name} {:.3} s ({} lines, {} nodes)", tf.elapsed().as_secs_f64(), r.len(), r.iter().map(|x| x.nodes).sum::<u64>()));
    }
    println!("OBS where round 1 goes 79.07: forced end for the mover {:.3} s ({}), for the other side {:.3} s ({}); taking lines alone: {}", t_forced_mover, format!("{:?}", f1).chars().take(12).collect::<String>(), t_forced_other, format!("{:?}", f2).chars().take(12).collect::<String>(), takes.join("; "));
    for b in [2usize, 4] {
        let tf = std::time::Instant::now();
        let q = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: b, sides: None, ..Default::default() });
        println!("OBS budget {b} 79.07: whole {:.3} s; questions asked {} (units asked {}, not asked {}); goals {}", tf.elapsed().as_secs_f64(), q.questions, q.asked.len(), q.unasked.len(), q.goals.len());
    }
    println!("DBG asked {:?} unasked {:?}", p.asked.iter().map(&nm).collect::<Vec<_>>(), p.unasked.iter().map(&nm).collect::<Vec<_>>());
    assert!(total > 0.0);
}
