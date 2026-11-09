//! The MCTS operator on tic-tac-toe in GDL (a game the layer knows nothing about): it finds a win in one, avoids a loss, obeys the fuse, repeats under a seed and
//! refuses goals it does not read.
use cs_game::gdl_game::GdlGame;
use cs_game::Game;
use cs_operator::{Budget, Goal, Mode, Operator, Status};
use cs_planning::mcts::MctsOp;

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

fn game() -> GdlGame { GdlGame::load("ttt", TTT).unwrap() }

/// Play the named moves in turn (x and o alternate; the other side's noop is not a move of the game's trait, which lists only the mover's).
fn after(g: &GdlGame, names: &[&str]) -> <GdlGame as Game>::State {
    let mut s = g.initial();
    for n in names { let m = g.find_move(&s, n).unwrap_or_else(|| panic!("no move {n}")); s = g.apply(&s, &m); }
    s
}

fn estimate(g: &GdlGame, s: &<GdlGame as Game>::State, role: usize, nodes: u64, seed: u64) -> cs_operator::Outcome<<GdlGame as Game>::Move> {
    MctsOp::new(1.41, 50, seed).run(g, s, &Goal::Estimate { role }, Budget::new(30.0, nodes))
}

/// x holds (1 1) and (1 2), o holds (2 1) and (2 2); x to move: (1 3) wins, and o threatens (2 3).
fn visits(o: &cs_operator::Outcome<<GdlGame as Game>::Move>) -> Vec<u64> {
    o.facts.iter().filter(|f| f.relation == "visits").map(|f| f.args[1].parse().unwrap()).collect()
}

#[test]
fn the_win_in_one_is_the_most_visited_move() {
    let g = game();
    let s = after(&g, &["(mark 1 1)", "(mark 2 1)", "(mark 1 2)", "(mark 2 2)"]);
    let o = estimate(&g, &s, 0, 20_000, 7);
    assert_eq!(o.status, Status::Read);
    let win = g.find_move(&s, "(mark 1 3)").unwrap();
    assert!(o.mv.as_ref() == Some(&win), "most visited should be the win; visits {:?}", visits(&o));
    assert_eq!(o.candidates.len(), g.moves(&s).len());
    assert!(o.value.unwrap() > 5000);
    assert!(o.depth.unwrap() >= 1);
}

#[test]
fn the_losing_move_is_the_least_visited() {
    let g = game();
    // x: (1 1), (3 3); o: (2 2); x to move would lose nothing here, so take o's turn: o holds (2 1),(2 2) and x must block (2 3)
    let s = after(&g, &["(mark 1 1)", "(mark 2 1)", "(mark 3 3)", "(mark 2 2)"]);
    let o = estimate(&g, &s, 0, 30_000, 3);
    let block = g.find_move(&s, "(mark 2 3)").unwrap();
    assert!(o.mv.as_ref() == Some(&block), "x must block; visits {:?}", visits(&o));
    let v = visits(&o);
    let ms = g.moves(&s);
    let bi = ms.iter().position(|m| *m == block).unwrap();
    // every other move loses at once against a good reply; none is visited more than the block
    assert!(v.iter().enumerate().all(|(i, n)| i == bi || *n < v[bi]));
}

#[test]
fn the_fuse_cuts_and_the_parts_sum_to_the_total() {
    let g = game();
    let s = g.initial();
    let o = estimate(&g, &s, 0, 300, 1);
    let playouts: u64 = o.facts.iter().find(|f| f.relation == "playouts").unwrap().args[0].parse().unwrap();
    assert_eq!(o.status, Status::Read);
    assert!(playouts >= 1 && playouts < 100, "{playouts} playouts");
    assert!(o.cost.nodes <= 300);
    assert_eq!(o.parts.iter().map(|p| p.cost.nodes).sum::<u64>(), o.cost.nodes);
    // a longer run records several parts, still summing to the total
    let o = estimate(&g, &s, 0, 20_000, 1);
    assert!(o.parts.len() > 1);
    assert_eq!(o.parts.iter().map(|p| p.cost.nodes).sum::<u64>(), o.cost.nodes);
}

#[test]
fn the_same_seed_gives_the_same_visits() {
    let g = game();
    let s = g.initial();
    let a = estimate(&g, &s, 0, 5_000, 42);
    let b = estimate(&g, &s, 0, 5_000, 42);
    assert_eq!(visits(&a), visits(&b));
    assert_eq!(a.value, b.value);
}

#[test]
fn other_goals_are_unsupported() {
    let g = game();
    let op = MctsOp::new(1.41, 50, 1);
    for goal in [Goal::Win { role: 0, mode: Mode::Forced }, Goal::Survey { max_defects: 0 }] {
        assert!(!Operator::<GdlGame>::accepts(&op, &goal));
        let o = op.run(&g, &g.initial(), &goal, Budget::new(1.0, 100));
        assert_eq!(o.status, Status::Unsupported);
    }
    assert!(Operator::<GdlGame>::accepts(&op, &Goal::Estimate { role: 0 }));
}
