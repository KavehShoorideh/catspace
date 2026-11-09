//! The lemma prior on a small GDL game (three cells in a row): a snapshot whose one lemma names a response gives that move the boost and the others 1.0; an empty
//! snapshot gives uniform weights; the prior plugs into the weighted playout policy.
use cs_decision::LemmaPrior;
use cs_game::gdl_game::GdlGame;
use cs_game::{Game, Vocabulary};
use cs_game::gdl_game::FactView;
use cs_memory::snapshot::Snapshot;
use cs_planning::rollout::{Prior, Weighted};
use cs_planning::mcts::Rollout;

const TTT: &str = "
(role x) (role o)
(init (cell 1 b)) (init (cell 2 b)) (init (cell 3 b))
(init (control x))
(<= (legal ?p (mark ?m)) (true (cell ?m b)) (true (control ?p)))
(<= (legal x noop) (true (control o)))
(<= (legal o noop) (true (control x)))
(<= (next (cell ?m x)) (does x (mark ?m)))
(<= (next (cell ?m o)) (does o (mark ?m)))
(<= (next (cell ?m ?w)) (true (cell ?m ?w)) (distinct ?w b))
(<= (next (cell ?m b)) (true (cell ?m b)) (does ?w (mark ?j)) (distinct ?m ?j))
(<= (next (control x)) (true (control o)))
(<= (next (control o)) (true (control x)))
(<= (line ?p) (true (cell 1 ?p)) (true (cell 2 ?p)) (true (cell 3 ?p)))
(<= open (true (cell ?m b)))
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

fn snapshot(lemmas: &str) -> Snapshot {
    let text = format!(r#"{{"format":1,"game":"ttt","parent":null,"vocabulary":{{"derived":[],"delta":["cell"],"units":null}},"judge":null,"teacher_used":false,"lemmas":[{lemmas}]}}"#);
    Snapshot::parse(text.as_bytes()).unwrap()
}

fn view() -> FactView { FactView::new(TTT, Vocabulary { derived: vec![], delta: vec!["cell".into()], units: None }).unwrap() }

/// When cell 2 is open, mark it.
const MARK_TWO: &str = r#"{"name":"mark-two","body":[{"relation":"cell","args":["2","b"],"neg":false}],
  "response":{"removed":[{"relation":"cell","args":["2","b"],"neg":false}],"added":[{"relation":"cell","args":["2","us"],"neg":false}]},"stats":{},"meta":{}}"#;

#[test]
fn a_bound_lemma_boosts_the_move_it_names() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let s = g.initial();
    let ms = g.moves(&s);
    let (snap, v) = (snapshot(MARK_TWO), view());
    let p = LemmaPrior::new(&snap, &v, 4.0);
    let w = p.weights(&g, &s, &ms);
    assert_eq!(w.len(), ms.len());
    for (m, wi) in ms.iter().zip(&w) {
        let want = if g.move_name(&s, m) == "(mark 2)" { 4.0 } else { 1.0 };
        assert_eq!(*wi, want, "{}", g.move_name(&s, m));
    }
    // it plugs into the weighted policy, and the boosted move is the commonest pick
    let mut roll = Weighted::new(p, 0.05, 3);
    let mut counts = vec![0usize; ms.len()];
    for _ in 0..1500 { counts[roll.pick(&g, &s, &ms)] += 1; }
    let two = ms.iter().position(|m| g.move_name(&s, m) == "(mark 2)").unwrap();
    assert!(counts.iter().enumerate().all(|(i, c)| i == two || *c < counts[two]), "{counts:?}");
}

#[test]
fn an_empty_snapshot_gives_uniform_weights() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let s = g.initial();
    let ms = g.moves(&s);
    let (snap, v) = (snapshot(""), view());
    assert_eq!(LemmaPrior::new(&snap, &v, 4.0).weights(&g, &s, &ms), vec![1.0; ms.len()]);
}
