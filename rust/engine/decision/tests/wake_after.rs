//! Plan 79.10 PART A (their goals after each of our moves, SYSTEM_SPEC 8 "evaluate the threats of the opponent", 7 the stake): UNIT TESTS of `after.rs` and of the fact about our move that the decision logs. Constructed positions only; no puzzle, no match.
use cs_decision::after::{fact_of, their_goals, their_goals_after, ChainView};
use cs_decision::threat_verdict::Record;
use cs_decision::memory::{Parts, WakeMemory};
use cs_decision::wake::{decide, Pieces, UnitRef, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
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
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn record() -> Record { Record { worth: table(), half: HashMap::new() } }
fn judge() -> Worth { Worth { name: "human".into(), table: table(), ours_first: false } }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn at(g: &KernelGame, name: &str) -> u16 { g.place_names().iter().position(|p| p == name).unwrap() as u16 }
fn role(g: &KernelGame, name: &str) -> usize { g.roles().iter().position(|r| r == name).unwrap() }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn unit(g: &KernelGame, r: &str, k: &str, p: &str) -> UnitRef { UnitRef { owner: role(g, r), kind: kind(g, k), at: at(g, p) } }

/// black to move: a bishop of white's on the long line to black's king, shut by a pawn (and a knight on the file of white's rook): white's goal on the king has things in its way, which black's moves change
const LINE: &str = "k7/4n3/2p5/8/8/5B2/8/4R2K b - - 0 1";

#[test]
fn a_their_goals_come_from_the_graph_alone_in_order_of_stake_the_goal_kind_first() {
    let (g, st) = world();
    let s = state(LINE);
    let white = role(&g, "white");
    let t0 = std::time::Instant::now();
    let tg = their_goals(&st, &g, &s, &record(), Some(kind(&g, "king")), white);
    println!("OBS their goals 79.10 A: from the graph alone at one position {:.4} s, {} goals, {} chains", t0.elapsed().as_secs_f64(), tg.goals.len(), tg.views.len());
    assert!(!tg.goals.is_empty() && tg.goals.iter().all(|x| x.side == white), "the goals are white's");
    assert!(tg.goals.iter().all(|x| x.line.is_empty()), "no line was read: the goals are the graph's");
    assert!(tg.views[0].stake.is_infinite() && tg.views[0].top == unit(&g, "black", "king", "a8"), "the chain of the goal kind is first");
    assert!(tg.views.windows(2).all(|w| w[0].stake >= w[1].stake));
    assert!(tg.views[0].stoppers.contains(&unit(&g, "black", "pawn", "c6")), "what stands in its way is named: {:?}", tg.views[0].stoppers);
}

#[test]
fn a_each_move_of_ours_gets_the_position_after_it_read_and_its_names_are_legal_there() {
    let (g, st) = world();
    let s = state(LINE);
    let moves = g.moves(&s);
    let order: Vec<usize> = (0..moves.len()).collect();
    let r = their_goals_after(&st, &g, &s, &record(), Some(kind(&g, "king")), &order, 12, None);
    assert_eq!(r.rows.len(), moves.len(), "every move has a child read");
    assert!(r.not_asked.is_empty());
    for row in &r.rows {
        let m = g.find_move(&s, &row.mv).expect("a legal move of ours");
        let child = g.apply(&s, &m);
        assert_eq!(row.key, g.key(&child), "the names are given for that position");
        assert!(row.names.len() <= 12);
        for n in &row.names { assert!(g.find_move(&child, n).is_some(), "{n} is a legal move of theirs at the position after {}", row.mv); }
    }
    assert!(r.rows.iter().any(|x| !x.names.is_empty()));
}

#[test]
fn a_a_move_that_takes_a_stopper_out_of_their_way_says_so_and_one_that_puts_a_thing_in_it_says_so() {
    let (g, st) = world();
    let s = state(LINE);
    let order: Vec<usize> = (0..g.moves(&s).len()).collect();
    let r = their_goals_after(&st, &g, &s, &record(), Some(kind(&g, "king")), &order, 12, None);
    // the knight that stood in the way of the rook's line goes away: it is taken out of the way (it is followed to where it went, and where it went it no longer stops)
    let f = r.rows.iter().find(|x| x.mv == "e7f5").and_then(|x| x.fact.clone()).expect("a fact about the knight's move");
    assert!(f.took_out.iter().any(|u| u.kind == kind(&g, "knight") && u.at == at(&g, "f5")), "{f:?}");
    assert!(f.put_in.is_empty());
    // the king steps to the other side of the pawn's file: a thing is put in the way of their chain
    let f2 = r.rows.iter().find(|x| x.mv == "a8b7").and_then(|x| x.fact.clone()).expect("a fact about the king's move");
    assert!(!f2.put_in.is_empty() || f2.in_the_way_after > f2.in_the_way_before.unwrap_or(0), "{f2:?}");
    for x in r.rows.iter().filter_map(|x| x.fact.as_ref()) { assert!(x.took_out.iter().all(|u| !x.put_in.contains(u)), "a unit is not both taken out and put in: {x:?}"); }
}

#[test]
fn a_the_comparison_follows_a_unit_the_move_itself_moved_and_needs_no_game() {
    let a = UnitRef { owner: 1, kind: 2, at: 10 };
    let b = UnitRef { owner: 1, kind: 3, at: 11 };
    let top = UnitRef { owner: 1, kind: 1, at: 7 };
    let before = vec![ChainView { top, stake: f64::INFINITY, in_the_way: 2, stoppers: vec![a, b] }];
    // the unit on 10 moves to 20 and still stops the chain; the unit on 11 stays; a new one on 30 appears: nothing is taken out, one is put in
    let c = UnitRef { owner: 1, kind: 4, at: 30 };
    let after = vec![ChainView { top, stake: f64::INFINITY, in_the_way: 3, stoppers: vec![UnitRef { at: 20, ..a }, b, c] }];
    let f = fact_of(&before, &after, Some((10, 20))).unwrap();
    assert_eq!((f.in_the_way_before, f.in_the_way_after, f.took_out.clone(), f.put_in.clone()), (Some(2), 3, vec![], vec![c]));
    // the unit leaves and stops nothing now: it is taken out of the way
    let after2 = vec![ChainView { top, stake: f64::INFINITY, in_the_way: 1, stoppers: vec![b] }];
    let f2 = fact_of(&before, &after2, Some((10, 20))).unwrap();
    assert_eq!((f2.took_out.clone(), f2.put_in.clone()), (vec![UnitRef { at: 20, ..a }], vec![]));
    // no chain after: no fact; a chain with no counterpart before: nothing is said to be put in
    assert!(fact_of(&before, &[], None).is_none());
    let f3 = fact_of(&[], &after, None).unwrap();
    assert!(f3.in_the_way_before.is_none() && f3.put_in.is_empty() && f3.took_out.is_empty());
}

#[test]
fn a_the_clock_of_the_step_lists_the_moves_it_did_not_reach_a_gap_and_not_a_refutation() {
    let (g, st) = world();
    let s = state(LINE);
    let order: Vec<usize> = (0..g.moves(&s).len()).collect();
    let r = their_goals_after(&st, &g, &s, &record(), Some(kind(&g, "king")), &order, 12, Some(std::time::Instant::now()));
    assert!(r.rows.is_empty());
    assert_eq!(r.not_asked.len(), order.len());
}

fn params(after: bool) -> WakeParams {
    WakeParams { seconds: 60.0, depth: 4, pieces: Pieces { threats: true, goals: true, two_threat_goals: true, after_move: after, ..Pieces::old_player() },
        inputs: WakeInputs { memory: WakeMemory::given(Parts { record: Some(record()), ..Default::default() }, None), goal: Some(kind(&KernelGame::new("chess", 518).unwrap(), "king")), budget: 4, ..WakeInputs::default() }, ..WakeParams::default() }
}

#[test]
fn a_the_fact_about_our_move_is_logged_and_changes_no_value() {
    let (g, st) = world();
    let s = state(LINE);
    let j = judge();
    let on = decide(&st, &g, &s, &[], &j, &params(true));
    let off = decide(&st, &g, &s, &[], &j, &params(false));
    assert!(off.log.moves.iter().all(|m| m.their_after.is_none()), "off: nothing is read");
    assert!(on.log.moves.iter().any(|m| m.their_after.as_ref().is_some_and(|a| !a.said.is_empty())), "on: the fact about the move is in the log");
    let knight = on.log.moves.iter().find(|m| m.mv == "e7f5").and_then(|m| m.their_after.as_ref()).expect("the knight's move is logged");
    assert!(knight.said.contains("took out of the way"), "{}", knight.said);
    // the search's value is the same with and without the names it gave under each of our moves: read without any forward pruning, exact at the depth
    let me = g.mover(&s).unwrap();
    let mut a = Search::with_fuse(&g, &j, me, 50_000_000);
    a.prune = false;
    let mut b = Search::with_fuse(&g, &j, me, 50_000_000);
    b.prune = false;
    let r = their_goals_after(&st, &g, &s, &record(), Some(kind(&g, "king")), &(0..g.moves(&s).len()).collect::<Vec<_>>(), 12, None);
    for x in &r.rows { b.named.insert(x.key, x.names.clone()); }
    assert!(!b.named.is_empty());
    assert_eq!(a.run(&s, 4).value, b.run(&s, 4).value, "the names order and never add to a value");
    assert_eq!(on.value, off.value, "and the decision's value is the same with the piece on and off");
}
