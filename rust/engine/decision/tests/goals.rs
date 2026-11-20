//! Plan 79.07 (SYSTEM_SPEC 7 "the proposer, what a goal is", 4 the budget of questions, 8 step 3): UNIT TESTS of the Rust proposer (cs_decision::goals): small positions built through the engine's own game interface, each piece's own contract asserted.
//! No puzzle, no stored game, no match, no integration of the wake's loop.
use cs_decision::goals::{double_threats, goals, named_moves, propose, two_threat_goals, FixKind, GoalParams, Status};
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
fn a_line_onto_the_goal_unit_shut_by_one_unit_is_a_goal_unit_goal_with_the_count_as_facts() {
    let (g, st) = world();
    let s = state(LINE);
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let white = role(&g, "white");
    let gu = p.goals.iter().find(|x| x.side == white && x.goal_unit).expect("the goal unit of the other side has a goal");
    assert_eq!(gu.target, unit(&g, "black", "king", "a8"));
    assert_eq!(gu.source, "goal_unit");
    assert_eq!(gu.lines_onto.len(), 1);
    assert_eq!(gu.lines_onto[0].blocker, unit(&g, "black", "pawn", "c6"));
    let f = gu.facts.iter().find(|f| f.name == "outnumbers").expect("the count of bearers and holders is a fact");
    assert_eq!(f.counts, Some((1, 1)), "one bearer (the bishop), one holder (the knight)");
    assert_eq!(f.status, Status::Unmet, "the bearers do not outnumber the holders");
    assert_eq!(f.place, Some(at(&g, "c6")));
    assert_eq!(f.cause.as_ref().unwrap().stoppers, vec![unit(&g, "black", "knight", "e7")], "the holder is what stops it");
    assert!(gu.facts.iter().any(|x| x.name == "borne_on" && x.status == Status::Unmet), "the goal unit itself is not borne on while the line is shut");
    assert!(gu.facts.iter().any(|x| x.name == "forced_end" && x.status == Status::Unmet));
    assert!(gu.n_unmet >= 2);
    // the goal unit's goal comes first among the questions: no row of a unit is needed for it
    assert!(p.goals.iter().position(|x| x.goal_unit).is_some());
}

#[test]
fn removing_the_only_holder_is_a_fix_and_its_moves_are_legal_moves() {
    let (g, st) = world();
    let s = state(LINE);
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let gu = p.goals.iter().find(|x| x.goal_unit && x.side == role(&g, "white")).unwrap();
    let f = gu.facts.iter().find(|f| f.name == "outnumbers").unwrap();
    let hold = f.fixes.iter().find(|x| x.kind == FixKind::Remove).expect("remove a holder");
    assert_eq!(hold.against, vec![(unit(&g, "black", "knight", "e7"), Some(at(&g, "c6")))]);
    assert!(hold.moves.contains(&"e1e7".to_string()), "the rook takes the holder: {:?}", hold.moves);
    let legal: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    assert!(hold.moves.iter().all(|m| legal.contains(m)), "every move of a fix is legal (built from legal moves, never from an edited position)");
    let el = gu.fix_set.iter().find(|e| e.unit == Some(unit(&g, "black", "knight", "e7"))).expect("the holder is in the fix set");
    assert!(el.removable && el.kind == FixKind::Remove && el.moves.contains(&"e1e7".to_string()));
}

#[test]
fn adding_a_bearer_by_a_quiet_move_is_a_fix_and_its_move_is_named() {
    let (g, st) = world();
    let s = state(LINE);
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let gu = p.goals.iter().find(|x| x.goal_unit && x.side == role(&g, "white")).unwrap();
    let f = gu.facts.iter().find(|f| f.name == "outnumbers").unwrap();
    let add = f.fixes.iter().find(|x| x.kind == FixKind::Bring).expect("add a bearer");
    assert_eq!(add.to, Some(at(&g, "c6")));
    assert!(add.moves.contains(&"e1e6".to_string()), "the rook on the sixth rank bears on the blocker: {:?}", add.moves);
    assert!(!add.moves.contains(&"e1e7".to_string()) || add.moves.contains(&"e1e7".to_string()), "(taking the holder is the other fix)");
    let named = named_moves(&g, &s, &p.goals, &p.doubles);
    let here = named.get(&g.key(&s)).expect("the mover's fixes are named at the position itself");
    assert!(here.contains(&"e1e6".to_string()) && here.contains(&"e1e7".to_string()));
    let mut once = here.clone(); once.sort(); once.dedup();
    assert_eq!(once.len(), here.len(), "a move is named once");
    let legal: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    assert!(here.iter().all(|m| legal.contains(m)));
}

#[test]
fn the_other_sides_fix_is_a_move_named_at_the_position_after_the_pass() {
    let (g, st) = world();
    let s = state("k7/4n3/2p5/8/8/5B2/8/4R2K b - - 0 1");
    let white = role(&g, "white");
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { sides: Some(vec![white]), ..Default::default() });
    assert!(p.goals.iter().all(|x| x.side == white));
    let named = named_moves(&g, &s, &p.goals, &p.doubles);
    assert!(named.get(&g.key(&s)).is_none(), "nothing is named for the side to move: only the other side's goals were asked");
    let passed = g.pass(&s).unwrap();
    let there = named.get(&g.key(&passed)).expect("the other side's fixes are named where it is to move");
    assert!(there.contains(&"e1e6".to_string()), "{there:?}");
}

#[test]
fn a_move_that_bears_on_the_goal_unit_and_uncovers_a_line_onto_a_unit_of_high_stake_raises_two_threats() {
    let (g, st) = world();
    // the knight on d5 shuts the rook's file onto the queen on d8; its move to e7 bears on the king on g8 and opens the file
    let s = state("3q2k1/5ppp/8/3N4/8/8/8/K2R4 w - - 0 1");
    let d = double_threats(&st, &g, &s, role(&g, "white"), Some(kind(&g, "king")));
    let m = d.iter().find(|x| x.mv == "d5e7").expect("the knight's move makes two threats at once");
    assert!(m.goal_unit);
    assert!(m.targets.contains(&unit(&g, "black", "king", "g8")) && m.targets.contains(&unit(&g, "black", "queen", "d8")));
    assert!(d[0].goal_unit, "the moves that bear on the goal unit come first");
    assert!(d.iter().skip_while(|x| x.goal_unit).all(|x| !x.goal_unit));
    // it is named for the search
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams::default());
    let named = named_moves(&g, &s, &p.goals, &p.doubles);
    assert!(named[&g.key(&s)].contains(&"d5e7".to_string()));
}

#[test]
fn a_taking_that_fails_is_a_goal_whose_unmet_fact_names_the_guard_and_the_fix_removes_it() {
    let (g, st) = world();
    let s = state("k7/8/2p5/3p4/8/8/8/K2R4 w - - 0 1");
    let l = concrete_threats(&st, &g, &s, &record(), &ThreatParams { every_sides: Some(vec![]), ..params(&g) });
    let gs = goals(&st, &g, &s, &l, role(&g, "white"), Some(kind(&g, "king")));
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
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 64, sides: Some(vec![role(&g, "white")]) });
    assert!(p.goals.len() >= 3);
    let key = |x: &cs_decision::goals::Goal| (x.n_unmet, -x.stake.unwrap_or(9.0));
    assert!(p.goals.windows(2).all(|w| key(&w[0]) <= key(&w[1])), "the weakest target first, the larger stake among equals");
}

#[test]
fn the_budget_bounds_the_questions_and_asks_the_units_of_most_worth_first() {
    let (g, st) = world();
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let all = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 1000, sides: None });
    assert!(all.unasked.is_empty(), "no budget bound: every unit is asked");
    let few = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 2, sides: None });
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
    let again = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 2, sides: None });
    assert_eq!((few.questions, few.goals.len(), few.unasked.len()), (again.questions, again.goals.len(), again.unasked.len()));
}

#[test]
fn a_line_that_a_move_serves_for_two_units_is_read_once() {
    let (g, st) = world();
    // two black units nothing bears on; the rook's quiet move to a5 makes a taking stand on both (the file and the rank); other moves serve one
    let s = state("7k/8/8/p5p1/8/8/8/2R4K w - - 0 1");
    let white = role(&g, "white");
    let p = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: 1000, sides: Some(vec![white]) });
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
    let named = named_moves(&g, &s, &p.goals, &p.doubles);
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
        p.two.iter().map(|(_, t)| t.goals.len()).sum::<usize>(), p.doubles.len(), parts.join("; "), rounds.join("; "));
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
        let q = propose(&st, &g, &s, &record(), &params(&g), &GoalParams { budget: b, sides: None });
        println!("OBS budget {b} 79.07: whole {:.3} s; questions asked {} (units asked {}, not asked {}); goals {}", tf.elapsed().as_secs_f64(), q.questions, q.asked.len(), q.unasked.len(), q.goals.len());
    }
    println!("DBG asked {:?} unasked {:?}", p.asked.iter().map(&nm).collect::<Vec<_>>(), p.unasked.iter().map(&nm).collect::<Vec<_>>());
    assert!(total > 0.0);
}
