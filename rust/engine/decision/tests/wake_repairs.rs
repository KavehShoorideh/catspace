//! Plan 79.08 PART A (the repairs, SYSTEM_SPEC 8 steps 3 to 5, 4 the budget): UNIT TESTS on small constructed positions, one for each repair. No puzzle, no stored game, no match.
//! A1 may_take reads the game's legal moves; A2 the lines read from one position share a table; A3 (named by ply) is tested in cs_tactics (tests/named_ply.rs); A4 only fixes and answers whose own line was read and not refuted are named.
use cs_decision::goals::{goals, propose, GoalParams, Status};
use cs_decision::rule::keep_read;
use cs_decision::threat_verdict::{Record, VerdictKind};
use cs_decision::threats::{concrete_threats, ConcreteList, ThreatParams, ThreatRow};
use cs_decision::wake::UnitRef;
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::lines::LineParams;
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

/// A list with one row against the black knight: the unit the white bishop on e2 bears on.
fn list_with_row_on_the_knight(g: &KernelGame) -> ConcreteList {
    let mut r = ThreatRow::new(role(g, "black"), Some(unit(g, "black", "knight", "c4")), "contested", vec![], None);
    r.verdict = VerdictKind::Exchange;
    let mut l = ConcreteList::default();
    l.other.push(r);
    l
}

fn may_take_of(g: &KernelGame, st: &Static, fen: &str) -> (Status, Option<&'static str>, Vec<UnitRef>) {
    let s = state(fen);
    let l = list_with_row_on_the_knight(g);
    let gs = goals(st, g, &s, &l, role(g, "white"), Some(kind(g, "king")), 1, 8);
    let gl = gs.iter().find(|x| x.target == unit(g, "black", "knight", "c4")).expect("a goal on the knight");
    let f = gl.facts.iter().find(|f| f.name == "may_take").expect("the fact may_take");
    (f.status, f.cause.as_ref().map(|c| c.kind), f.cause.as_ref().map(|c| c.stoppers.clone()).unwrap_or_default())
}

#[test]
fn a1_a_tied_bearer_is_not_free_to_take_and_a_free_one_is() {
    let (g, st) = world();
    // the white bishop on e2 bears on the knight on c4 along the diagonal, and is pinned to its king on e1 by the rook on e8: the game's legal moves do not let it take
    let (status, cause, stoppers) = may_take_of(&g, &st, "4r1k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1");
    assert_eq!(status, Status::Unmet, "the graph's may-move relation says free; the game's legal moves say tied");
    assert_eq!(cause, Some("tied"));
    assert_eq!(stoppers, vec![unit(&g, "black", "rook", "e8")]);
    // the control: no rook on the file, the same bearer takes
    let (status, cause, _) = may_take_of(&g, &st, "6k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1");
    assert_eq!((status, cause), (Status::Met, None));
}

fn rows_key(l: &ConcreteList) -> Vec<(Option<UnitRef>, String, Option<i64>)> {
    let mut v: Vec<_> = l.mover.iter().chain(&l.other).chain(&l.exchange).chain(&l.dropped).map(|r| (r.unit, format!("{:?}", r.verdict), r.net)).collect();
    v.sort_by_key(|x| (x.0.map(|u| (u.owner, u.at)), x.1.clone()));
    v
}

#[test]
fn a2_lines_read_from_one_position_share_a_table_and_give_the_same_verdicts() {
    let (g, st) = world();
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let k = kind(&g, "king");
    let plain = ThreatParams { goal: Some(k), ..Default::default() };
    let shared = ThreatParams { goal: Some(k), share_table: true, ..Default::default() };
    let t0 = std::time::Instant::now();
    let a = concrete_threats(&st, &g, &s, &record(), &plain);
    let ta = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let b = concrete_threats(&st, &g, &s, &record(), &shared);
    let tb = t0.elapsed().as_secs_f64();
    let (na, nb): (u64, u64) = (a.cost.iter().map(|c| c.nodes).sum(), b.cost.iter().map(|c| c.nodes).sum());
    println!("OBS shared table 79.08: fixed constructed position of 79.06, one thread: own table per line {:.3} s, {} nodes; one table per position and side {:.3} s, {} nodes; rows {} and {}", ta, na, tb, nb, rows_key(&a).len(), rows_key(&b).len());
    assert_eq!(rows_key(&a).len(), rows_key(&b).len(), "the same rows");
    // the verdicts are of the same kind row for row (a shared table may read a deeper line from the table, never a different judgement of the row's kind)
    let ka: Vec<_> = rows_key(&a).into_iter().map(|x| (x.0, x.1)).collect();
    let kb: Vec<_> = rows_key(&b).into_iter().map(|x| (x.0, x.1)).collect();
    assert_eq!(ka, kb, "the verdict of every row is unchanged by the shared table");
    // (direction, one run: the lines read from one position do not share many sub-positions, so the table saves little; it must not cost more than a tenth)
    assert!((nb as f64) <= (na as f64) * 1.10, "a shared table must not read much more than a table for every line");
    // a list read twice the same way is the same list
    let c = concrete_threats(&st, &g, &s, &record(), &shared);
    assert_eq!(rows_key(&b), rows_key(&c));
}

#[test]
fn a4_a_named_move_is_one_whose_line_was_read_and_not_refuted_within_the_budget() {
    let (g, _st) = world();
    // white to move; the rook can go to the eighth rank where the queen takes it for nothing, or to the second rank (quiet). Black, after a pass, can put its queen on the first rank where the rook takes it, or to the seventh.
    let s = state("3q2k1/8/8/8/8/8/8/R5K1 w - - 0 1");
    let lp = LineParams { depth: 4, fuse: 12_000 };
    let mine = vec!["a1a8".to_string(), "a1a2".to_string()];
    let theirs = vec!["d8d1".to_string(), "d8d7".to_string()];
    let k = keep_read(&g, &s, &record(), &mine, &theirs, 12, lp, true);
    assert!(k.refuted.iter().any(|r| r.0 == "a1a8" && !r.1), "the rook given for nothing is refuted and not named: {:?}", k.refuted);
    assert!(k.mine.iter().any(|m| m.name == "a1a2"));
    assert!(k.refuted.iter().any(|r| r.0 == "d8d1" && r.1), "their queen given for nothing is refuted too");
    assert!(k.theirs.iter().any(|m| m.name == "d8d7"));
    assert_eq!(k.read, 4);
    // the budget: one read a side; the second move is not read (a gap, listed, never a refutation)
    let b = keep_read(&g, &s, &record(), &mine, &theirs, 1, lp, true);
    assert_eq!(b.read, 2);
    assert!(b.unlisted.iter().any(|u| u.0 == "a1a2" && !u.1) && b.unlisted.iter().any(|u| u.0 == "d8d7" && u.1));
    // a name that is not a legal move is not read and not listed
    let n = keep_read(&g, &s, &record(), &["zzzz".to_string()], &[], 12, lp, true);
    assert_eq!((n.read, n.mine.len(), n.refuted.len()), (0, 0, 0));
}

#[test]
fn a4_the_names_never_change_a_value_the_search_reads() {
    // the search with the named moves keeps the value it has without them (an order, never a verdict): see cs_tactics tests named_first and named_ply; here the proposer's own names are given to a search at the root
    let (g, st) = world();
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let p = propose(&st, &g, &s, &record(), &ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() }, &GoalParams { budget: 2, ..Default::default() });
    let names = cs_decision::goals::named_moves(&g, &s, &p.goals, &p.defence);
    let me = g.mover(&s).unwrap();
    let j = cs_tactics::ledger::Worth { name: "record".into(), table: record().worth.clone(), ours_first: false };
    let mut a = cs_tactics::ledger::Search::with_fuse(&g, &j, me, 50_000_000);
    a.prune = false;
    let mut b = cs_tactics::ledger::Search::with_fuse(&g, &j, me, 50_000_000);
    b.prune = false;
    b.named = names.into_iter().collect();
    assert_eq!(a.run(&s, 4).value, b.run(&s, 4).value);
}
