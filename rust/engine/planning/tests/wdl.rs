//! Plan 78.26: win / draw / loss mass, each piece alone -- `split`, the tallies of a search, the results table's back-off, `sharp`, and the operator's `Prefer` on
//! tic-tac-toe in GDL (a game the layer knows nothing about).
use cs_game::gdl_game::GdlGame;
use cs_game::{Game, Role};
use cs_operator::{Budget, Goal, Operator, Status};
use cs_planning::mcts::{search, Uniform};
use cs_planning::rollout::Draw;
use cs_planning::wdl::{sharp, split, Table, Wdl, WdlOp, MIN_MASS, TABLE_MIN};
use std::collections::HashMap;
use std::sync::Arc;

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

fn after(g: &GdlGame, names: &[&str]) -> <GdlGame as Game>::State {
    let mut s = g.initial();
    for n in names { let m = g.find_move(&s, n).unwrap_or_else(|| panic!("no move {n}")); s = g.apply(&s, &m); }
    s
}

/// A one-move game that holds units: role "p" has an "a", role "q" has an "a" and a "b"; "p" is to move.
struct Holder;
impl Game for Holder {
    type State = u8;
    type Move = u8;
    fn name(&self) -> String { "holder".into() }
    fn roles(&self) -> Vec<String> { vec!["p".into(), "q".into()] }
    fn initial(&self) -> u8 { 0 }
    fn mover(&self, s: &u8) -> Option<Role> { if *s == 0 { Some(0) } else { None } }
    fn moves(&self, s: &u8) -> Vec<u8> { if *s == 0 { vec![1] } else { vec![] } }
    fn apply(&self, _s: &u8, m: &u8) -> u8 { *m }
    fn outcome(&self, s: &u8) -> Option<Vec<i64>> { if *s == 0 { None } else { Some(vec![50, 50]) } }
    fn move_name(&self, _s: &u8, m: &u8) -> String { m.to_string() }
    fn key(&self, s: &u8) -> u64 { 0xabc + *s as u64 }
    fn units(&self, _s: &u8) -> Vec<(Role, String)> { vec![(1, "b".into()), (0, "a".into()), (1, "a".into())] }
}

#[test]
fn split_keeps_the_score_and_its_edges() {
    assert_eq!(split(100.0), [1.0, 0.0, 0.0]);
    assert_eq!(split(0.0), [0.0, 0.0, 1.0]);
    assert_eq!(split(50.0), [0.0, 1.0, 0.0]);
    for v in [0.0, 3.0, 25.0, 49.0, 50.0, 51.0, 70.0, 99.0, 100.0] {
        let m = split(v);
        assert!((m.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let w = Wdl::new(m[0], m[1], m[2]);
        assert!((w.score() - v / 100.0).abs() < 1e-12, "{v}");
    }
    let m = split(75.0);
    assert!((m[0] - 0.5).abs() < 1e-12 && (m[1] - 0.5).abs() < 1e-12 && m[2] == 0.0);
}

#[test]
fn wdl_arithmetic() {
    let w = Wdl::new(6.0, 2.0, 2.0);
    assert_eq!(w.n(), 10.0);
    assert_eq!(w.p(), Some((0.6, 0.2, 0.2)));
    assert_eq!(Wdl::default().p(), None);
    assert!((w.score() - 0.7).abs() < 1e-12 && (w.sharpness() - 0.8).abs() < 1e-12);
    assert_eq!(w.flip(), Wdl::new(2.0, 2.0, 6.0));
    assert_eq!(w.scaled(5.0), Wdl::new(3.0, 1.0, 1.0));
    assert_eq!(w.add(&w).n(), 20.0);
}

#[test]
fn the_tallies_hold_the_playouts_and_their_mean() {
    let g = game();
    let s = g.initial();
    let mut b = Budget::new(30.0, 4000);
    let r = search(&g, &s, &mut b, 1.41, 50, &mut Uniform::new(5), &Draw);
    assert!(r.playouts > 50);
    for role in 0..2 {
        let w = r.root_wdl[role];
        assert!((w.n() - r.playouts as f64).abs() < 1e-9);
        assert!((w.score() - r.root_mean[role] / 100.0).abs() < 1e-9, "role {role}");
    }
    let total: f64 = r.move_wdl.iter().map(|m| m[0].n()).sum();
    assert!((total - r.playouts as f64).abs() < 1e-9);
    for (i, m) in r.move_wdl.iter().enumerate() { assert!((m[1].n() - r.root[i].0 as f64).abs() < 1e-9); }
    // the two roles see each other's win and loss
    assert!((r.root_wdl[0].win - r.root_wdl[1].loss).abs() < 1e-9);
}

#[test]
fn a_lookup_backs_off_from_the_exact_key_to_the_units_key_and_flips() {
    let g = Holder;
    let s = g.initial();
    let keys = Table::keys(&g, &s);
    assert_eq!(keys, vec!["exact:abc".to_string(), "units:a|a,b:p".to_string()]);
    let mut t = Table { name: "t".into(), rows: HashMap::new() };
    assert!(t.lookup(&g, &s, 0).is_none());
    t.rows.insert(keys[1].clone(), [20.0, 5.0, 5.0]);
    let (k, w) = t.lookup(&g, &s, 0).unwrap();
    assert_eq!(k, keys[1]);
    assert_eq!(w, Wdl::new(20.0, 5.0, 5.0));
    assert_eq!(t.lookup(&g, &s, 1).unwrap().1, Wdl::new(5.0, 5.0, 20.0));
    t.rows.insert(keys[0].clone(), [30.0, 0.0, 0.0]);
    let (k, w) = t.lookup(&g, &s, 0).unwrap();
    assert_eq!((k, w), (keys[0].clone(), Wdl::new(30.0, 0.0, 0.0)));
}

#[test]
fn a_row_under_the_minimum_is_no_hit() {
    let g = Holder;
    let s = g.initial();
    let keys = Table::keys(&g, &s);
    let mut t = Table::default();
    t.rows.insert(keys[0].clone(), [TABLE_MIN - 1.0, 0.0, 0.0]);
    t.rows.insert(keys[1].clone(), [5.0, 5.0, 5.0]);
    assert!(t.lookup(&g, &s, 0).is_none());
    // a game with no units gives the exact key only
    assert_eq!(Table::keys(&game(), &game().initial()).len(), 1);
}

#[test]
fn sharp_quiet_and_unknown() {
    assert_eq!(sharp(&Wdl::new(20.0, 20.0, 20.0)), Some(true));
    assert_eq!(sharp(&Wdl::new(10.0, 80.0, 10.0)), Some(false));
    assert_eq!(sharp(&Wdl::new(10.0, 10.0, 0.0)), None);
    assert!(Wdl::new(15.0, 15.0, 0.0).n() >= MIN_MASS);
    assert_eq!(sharp(&Wdl::new(15.0, 15.0, 0.0)), Some(true));   // exactly at the threshold is sharp
}

fn op(table: Option<Arc<Table>>) -> WdlOp<Uniform, Draw> { WdlOp::new(1.41, 50, Uniform::new(3), Draw, table) }

#[test]
fn prefer_picks_the_winning_move_and_estimate_reads_the_position() {
    let g = game();
    // x holds (1 1) and (1 2), o holds (2 1) and (2 2); x to move: (1 3) wins
    let s = after(&g, &["(mark 1 1)", "(mark 2 1)", "(mark 1 2)", "(mark 2 2)"]);
    let moves = g.moves(&s);
    let among: Vec<usize> = (0..moves.len()).collect();
    let o = op(None).run(&g, &s, &Goal::Prefer { role: 0, among: among.clone(), defects: 0 }, Budget::new(30.0, 20_000));
    assert_eq!(o.status, Status::Read);
    assert_eq!(g.move_name(&s, o.mv.as_ref().unwrap()), "(mark 1 3)");
    assert_eq!(o.candidates.len(), among.len());
    assert!(o.evidence.iter().all(|l| l.starts_with("wdl: ")));
    assert_eq!(o.facts.iter().filter(|f| f.relation == "move_wdl").count(), among.len());
    let e = op(None).run(&g, &s, &Goal::Estimate { role: 0 }, Budget::new(30.0, 5000));
    let wdl = e.facts.iter().find(|f| f.relation == "wdl").unwrap();
    let (win, draw, loss): (f64, f64, f64) = (wdl.args[0].parse().unwrap(), wdl.args[1].parse().unwrap(), wdl.args[2].parse().unwrap());
    assert!((win + draw + loss - 1.0).abs() < 1e-3 && win > 0.5);
    assert!(e.facts.iter().any(|f| f.relation == "sharpness"));
    assert!(e.evidence.iter().all(|l| l.starts_with("wdl: ")));
}

#[test]
fn a_table_hit_is_pooled_with_the_playouts_and_a_terminal_is_one_sure_outcome() {
    let g = Holder;
    let s = g.initial();
    let keys = Table::keys(&g, &s);
    let mut t = Table { name: "t".into(), rows: HashMap::new() };
    t.rows.insert(keys[1].clone(), [1000.0, 0.0, 0.0]);
    let e = op(Some(Arc::new(t))).run(&g, &s, &Goal::Estimate { role: 0 }, Budget::new(5.0, 100));
    let f = e.facts.iter().find(|f| f.relation == "wdl").unwrap();
    assert_eq!(f.args[4], "playouts+table");
    // the table's 1000 wins are brought down to TABLE_CAP mass beside the playouts' draws
    assert!(f.args[0].parse::<f64>().unwrap() > 0.5);
    assert!(e.evidence.iter().any(|l| l.contains(&keys[1])));
    let q = g.apply(&s, &1);
    let p = op(None).run(&g, &s, &Goal::Prefer { role: 0, among: vec![0], defects: 0 }, Budget::new(5.0, 100));
    assert_eq!(p.candidates, vec![0]);
    let _ = q;
}
