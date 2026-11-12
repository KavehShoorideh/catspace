//! Plan 78.36: the search toward claims, on tic-tac-toe in GDL (a game the layer knows nothing about) with a stand-in oracle: what a claim of a store says at a leaf.
use cs_game::gdl_game::GdlGame;
use cs_game::{Fact, Game};
use cs_operator::Budget;
use cs_tactics::toward::{toward, Class, Verdict};
use std::collections::HashSet;

const TTT: &str = "
(role x) (role o)
(init (cell 1 1 b)) (init (cell 1 2 b)) (init (cell 1 3 b))
(init (cell 2 1 b)) (init (cell 2 2 b)) (init (cell 2 3 b))
(init (cell 3 1 b)) (init (cell 3 2 b)) (init (cell 3 3 b))
(init (control x))
(<= (legal ?p (mark ?m ?n)) (true (cell ?m ?n b)) (true (control ?p)))
(<= (legal x noop) (true (control o)))
(<= (legal o noop) (true (control x)))
(<= (next (cell ?m ?n x)) (does x (mark ?m ?n)))
(<= (next (cell ?m ?n o)) (does o (mark ?m ?n)))
(<= (next (cell ?m ?n ?w)) (true (cell ?m ?n ?w)) (distinct ?w b))
(<= (next (cell ?m ?n b)) (true (cell ?m ?n b)) (does ?w (mark ?j ?k)) (distinct ?m ?j))
(<= (next (cell ?m ?n b)) (true (cell ?m ?n b)) (does ?w (mark ?j ?k)) (distinct ?n ?k))
(<= (next (control x)) (true (control o)))
(<= (next (control o)) (true (control x)))
(<= (row ?m ?p) (true (cell ?m 1 ?p)) (true (cell ?m 2 ?p)) (true (cell ?m 3 ?p)))
(<= (column ?n ?p) (true (cell 1 ?n ?p)) (true (cell 2 ?n ?p)) (true (cell 3 ?n ?p)))
(<= (diagonal ?p) (true (cell 1 1 ?p)) (true (cell 2 2 ?p)) (true (cell 3 3 ?p)))
(<= (diagonal ?p) (true (cell 1 3 ?p)) (true (cell 2 2 ?p)) (true (cell 3 1 ?p)))
(<= (line ?p) (row ?m ?p))
(<= (line ?p) (column ?n ?p))
(<= (line ?p) (diagonal ?p))
(<= open (true (cell ?m ?n b)))
(<= terminal (line x))
(<= terminal (line o))
(<= terminal (not open))
(<= (goal x 100) (line x))
(<= (goal x 50) (not (line x)) (not (line o)))
(<= (goal x 0) (line o))
(<= (goal o 100) (line o))
(<= (goal o 50) (not (line x)) (not (line o)))
(<= (goal o 0) (line x))
";

type S = <GdlGame as Game>::State;
const X: usize = 0;

fn game() -> GdlGame { GdlGame::load("ttt", TTT).unwrap() }
fn after(g: &GdlGame, names: &[&str]) -> S {
    let mut s = g.initial();
    for n in names { let m = g.find_move(&s, n).unwrap_or_else(|| panic!("no move {n}")); s = g.apply(&s, &m); }
    s
}
fn holds(g: &GdlGame, s: &S, cell: (&str, &str), who: &str) -> bool { g.base_facts(s).contains(&Fact::new("cell", &[cell.0, cell.1, who])) }
fn name_of(g: &GdlGame, s: &S, i: usize) -> String { g.move_name(s, &g.moves(s)[i]) }
fn big() -> Budget { Budget::new(60.0, 50_000_000) }
fn none(_: &GdlGame, _: &S) -> Option<Verdict> { None }
fn run<O: cs_tactics::toward::LeafOracle<GdlGame>>(g: &GdlGame, s: &S, o: &O, depth: u32) -> cs_tactics::toward::Toward { toward(g, s, X, o, &HashSet::new(), depth, &mut big()) }

#[test]
fn with_no_claim_only_the_games_own_outcome_is_known() {
    let g = game();
    // x to move with two in a row and the third open: the move that completes the row is a win at once (the game's own outcome), every other move is unknown at depth 1
    let s = after(&g, &["(mark 1 1)", "(mark 2 1)", "(mark 1 2)", "(mark 2 2)"]);
    let r = run(&g, &s, &none, 1);
    assert_eq!(r.depth, 1);
    let wins = r.of_class(Class::Win);
    assert_eq!(wins.iter().map(|i| name_of(&g, &s, *i)).collect::<Vec<_>>(), vec!["(mark 1 3)".to_string()]);
    assert_eq!(r.count(Class::Unknown), r.root.len() - 1);
    assert!(r.evals > 0, "the oracle was asked at the leaves");
    // from the start with nothing claimed, nothing is known at any depth that cannot reach the end of the game
    let r0 = run(&g, &g.initial(), &none, 4);
    assert_eq!(r0.best_class(), Class::Unknown);
    assert_eq!(r0.count(Class::Unknown), 9);
}

#[test]
fn a_forced_line_of_depth_two_to_a_claimed_win_is_found() {
    let g = game();
    // the claim: the side to move holds the centre -> win. Reached after x takes the centre and o answers anything: a leaf at ply 2, whatever o plays
    let oracle = |g: &GdlGame, s: &S| if g.mover(s) == Some(X) && holds(g, s, ("2", "2"), "x") { Some(Verdict::Win) } else { None };
    let s = g.initial();
    let r = run(&g, &s, &oracle, 4);
    assert_eq!(r.depth, 2, "the first iteration that sees it is the second; the search stops there");
    let wins = r.of_class(Class::Win);
    assert_eq!(wins.iter().map(|i| name_of(&g, &s, *i)).collect::<Vec<_>>(), vec!["(mark 2 2)".to_string()]);
    assert!(r.reads[0].complete && r.reads[1].complete);
    // at depth 1 nothing was known yet
    assert_eq!(run(&g, &s, &oracle, 1).best_class(), Class::Unknown);
}

#[test]
fn a_move_the_opponent_can_turn_into_a_claimed_loss_is_refuted() {
    let g = game();
    // the claim: the side to move, with o holding the centre, loses. A move of x that leaves the centre free lets o take it
    let oracle = |g: &GdlGame, s: &S| if g.mover(s) == Some(X) && holds(g, s, ("2", "2"), "o") { Some(Verdict::Loss) } else { None };
    let s = g.initial();
    let r = run(&g, &s, &oracle, 2);
    let refuted: Vec<String> = r.of_class(Class::Loss).iter().map(|i| name_of(&g, &s, *i)).collect();
    assert_eq!(refuted.len(), 8, "every move but the centre: {refuted:?}");
    assert!(!refuted.contains(&"(mark 2 2)".to_string()));
    assert_eq!(r.of_class(Class::Unknown).iter().map(|i| name_of(&g, &s, *i)).collect::<Vec<_>>(), vec!["(mark 2 2)".to_string()]);
}

#[test]
fn unknown_stays_distinct_from_a_claimed_draw() {
    let g = game();
    // after x takes the centre, a claimed DRAW stands only where o answered in the corner (1 1); o's other answers are unknown. o, the minimiser, prefers the unknown to the claimed draw
    let oracle = |g: &GdlGame, s: &S| if g.mover(s) == Some(X) && holds(g, s, ("2", "2"), "x") && holds(g, s, ("1", "1"), "o") { Some(Verdict::Draw) } else { None };
    let s = g.initial();
    let r = run(&g, &s, &oracle, 2);
    let centre = r.root.iter().find(|x| name_of(&g, &s, x.0) == "(mark 2 2)").unwrap();
    assert_eq!(centre.1, Class::Unknown, "loss < unknown < draw: the opponent leaves us in the unknown");
    // and where every answer is a claimed draw, the draw is the class
    let all = |g: &GdlGame, s: &S| if g.mover(s) == Some(X) && holds(g, s, ("2", "2"), "x") { Some(Verdict::Draw) } else { None };
    let r2 = run(&g, &s, &all, 2);
    assert_eq!(r2.of_class(Class::Draw).iter().map(|i| name_of(&g, &s, *i)).collect::<Vec<_>>(), vec!["(mark 2 2)".to_string()]);
    assert_eq!(r2.best_class(), Class::Draw);
}

#[test]
fn a_position_met_again_is_a_draw() {
    let g = game();
    let s = g.initial();
    let m = g.find_move(&s, "(mark 2 2)").unwrap();
    let key = g.key(&g.apply(&s, &m));
    let mut hist = HashSet::new();
    hist.insert(key);
    let r = toward(&g, &s, X, &none, &hist, 1, &mut big());
    assert_eq!(r.of_class(Class::Draw).iter().map(|i| name_of(&g, &s, *i)).collect::<Vec<_>>(), vec!["(mark 2 2)".to_string()], "the position the record already holds is the rules' draw");
}

#[test]
fn the_budget_cut_gives_the_last_completed_depth() {
    let g = game();
    let oracle = |g: &GdlGame, s: &S| if g.mover(s) == Some(X) && holds(g, s, ("2", "2"), "o") { Some(Verdict::Loss) } else { None };
    let s = g.initial();
    let full = run(&g, &s, &oracle, 4);
    assert!(full.depth >= 3 && !full.cut);
    // a node fuse that lets two iterations finish and not a third
    let two: u64 = full.reads[..2].iter().map(|r| r.nodes).sum();
    let mut b = Budget::new(60.0, two + 3);
    let cutr = toward(&g, &s, X, &oracle, &HashSet::new(), 4, &mut b);
    assert!(cutr.cut);
    assert_eq!(cutr.depth, 2, "the last completed iteration");
    assert!(!cutr.reads.last().unwrap().complete);
    // its classes are those a search asked for exactly that depth gives
    let exact = run(&g, &s, &oracle, 2);
    assert_eq!(cutr.root.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>(), exact.root.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>());
    // a fuse that cuts the first iteration: nothing is known
    let mut tiny = Budget::new(60.0, 3);
    let none_done = toward(&g, &s, X, &oracle, &HashSet::new(), 4, &mut tiny);
    assert_eq!((none_done.depth, none_done.best_class()), (0, Class::Unknown));
    // the clock cuts too: a deadline already past
    let mut late = Budget::new(0.0, 50_000_000);
    let r = toward(&g, &s, X, &oracle, &HashSet::new(), 4, &mut late);
    assert!(r.cut && r.depth <= 1);
}

#[test]
fn the_shortest_win_is_preferred() {
    let g = game();
    // two ways to a claimed win: at ply 1 (x takes the corner) and at ply 2 (the centre); the first depth that shows one stops the deepening, so the nearer is found
    let oracle = |g: &GdlGame, s: &S| if g.mover(s) != Some(X) && holds(g, s, ("1", "1"), "x") { Some(Verdict::Loss) } else if g.mover(s) == Some(X) && holds(g, s, ("2", "2"), "x") { Some(Verdict::Win) } else { None };
    let s = g.initial();
    let r = run(&g, &s, &oracle, 4);
    assert_eq!(r.depth, 1, "a win is found at the first depth that shows one");
    assert_eq!(r.of_class(Class::Win).iter().map(|i| name_of(&g, &s, *i)).collect::<Vec<_>>(), vec!["(mark 1 1)".to_string()]);
}

#[test]
fn a_win_that_rests_on_the_games_outcome_alone_is_kept_apart_from_a_claimed_one() {
    let g = game();
    // x has two in a row, third open: completing it is a game win; a claim that the side with the other cell as o's... says another move is a claimed win
    let s = after(&g, &["(mark 1 1)", "(mark 2 1)", "(mark 1 2)", "(mark 2 2)"]);
    let oracle = |g: &GdlGame, s: &S| if g.mover(s) != Some(X) && holds(g, s, ("3", "3"), "x") { Some(Verdict::Loss) } else { None };
    let r = run(&g, &s, &oracle, 1);
    let wins = r.of_class(Class::Win);
    assert_eq!(wins.len(), 2, "the row, and the claimed win at (3 3): {wins:?}");
    assert_eq!(r.game_wins().iter().map(|i| name_of(&g, &s, *i)).collect::<Vec<_>>(), vec!["(mark 1 3)".to_string()]);
    assert_eq!(name_of(&g, &s, wins[0]), "(mark 1 3)", "a game win is above a claimed one");
    // only a claim: no game win
    let only = run(&g, &g.initial(), &|g: &GdlGame, s: &S| if g.mover(s) != Some(X) && holds(g, s, ("2", "2"), "x") { Some(Verdict::Loss) } else { None }, 1);
    assert_eq!(only.of_class(Class::Win).len(), 1);
    assert!(only.game_wins().is_empty());
}

#[test]
fn the_search_stops_at_two_plies_when_nothing_speaks() {
    let g = game();
    let r = run(&g, &g.initial(), &none, 8);
    assert!(r.silent && r.depth == cs_tactics::toward::SILENT_DEPTH && !r.cut);
    assert_eq!(r.best_class(), Class::Unknown);
    // a claim that speaks only at three plies is not heard: that is the cost of the stop, by design
    let late = |g: &GdlGame, s: &S| if g.mover(s) != Some(X) && holds(g, s, ("2", "2"), "x") && holds(g, s, ("1", "1"), "x") { Some(Verdict::Loss) } else { None };
    assert!(run(&g, &g.initial(), &late, 8).silent);
    // a claim that speaks within two plies keeps the search going
    let early = |g: &GdlGame, s: &S| if g.mover(s) == Some(X) && holds(g, s, ("2", "2"), "o") { Some(Verdict::Draw) } else { None };
    let r2 = run(&g, &g.initial(), &early, 4);
    assert!(!r2.silent && r2.depth >= 3);
}
