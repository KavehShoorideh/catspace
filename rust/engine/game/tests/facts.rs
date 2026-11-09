//! PLAN 78.13 THE FACT LAYER: the chess kernels' facts against the descriptions' fluents by co-play (rules ingestion, certified), and the fact view's derived
//! relations. The kernels give facts for the variants that have a written description; a wrong conversion is a disagreement the co-play finds.
use cs_game::certify::coplay;
use cs_game::gdl_game::{FactView, GdlGame};
use cs_game::kernel::{uci_to_spec_term, KernelGame};
use cs_game::{Fact, Game, Vocabulary};
use std::path::PathBuf;

fn rules(rel: &str) -> String { std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules").join(rel)).unwrap() }

fn text_of(variant: &str) -> String {
    match variant {
        "chess" => rules("chess.gdl"),
        v => cs_gdl::compose(&rules("chess.gdl"), &rules(&format!("variants/{v}.gdl"))).unwrap(),
    }
}

#[test]
fn the_kernels_facts_equal_the_descriptions_fluents_at_every_state_of_a_co_play() {
    for v in ["chess", "kingofthehill", "3check"] {
        let k = KernelGame::new(v, 518).unwrap();
        let g = GdlGame::load(v, &text_of(v)).unwrap();
        let r = coplay(&k, &g, &|n| uci_to_spec_term(n), 6, 11);
        println!("OBS 78.13 facts co-play {v}: games {}, states {}, with facts compared {}, disagreements {}", r.games, r.states, r.fact_states, r.disagreements.len());
        assert!(r.disagreements.is_empty(), "{v}: {:?}", r.disagreements);
        assert!(r.fact_states > 100, "the facts were really compared at {} states", r.fact_states);
    }
}

#[test]
fn a_wrong_conversion_is_a_disagreement() {
    // the control: compare the three-check kernel against plain chess's description -- its facts carry `checks`, which plain chess has no fluent for
    let k = KernelGame::new("3check", 518).unwrap();
    let g = GdlGame::load("chess", &text_of("chess")).unwrap();
    let r = coplay(&k, &g, &|n| uci_to_spec_term(n), 2, 3);
    assert!(!r.disagreements.is_empty());
}

#[test]
fn the_fact_view_derives_the_relations_the_vocabulary_names() {
    let vocab = Vocabulary { derived: vec!["attack".into(), "check".into(), "opponent".into()], delta: vec!["cell".into()], units: None };
    let view = FactView::new(&text_of("chess"), vocab).unwrap();
    let k = KernelGame::from_fen("chess", "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1").unwrap();
    let s = k.initial();
    let facts = view.facts(&k.base_facts(&s)).unwrap();
    let has = |rel: &str, args: &[&str]| facts.contains(&Fact::new(rel, args));
    assert!(has("cell", &["e", "2", "black", "queen"]));
    assert!(has("opponent", &["white", "black"]), "a static relation of the description is in the view");
    assert!(has("attack", &["black", "e", "2", "e", "1"]), "the queen attacks the king");
    assert!(has("check", &["white"]), "white is in check");
    assert!(!has("check", &["black"]));
    // a relation the description does not have is refused when the view is built
    assert!(FactView::new(&text_of("chess"), Vocabulary { derived: vec!["no_such_relation".into()], ..Default::default() }).is_err());
}

#[test]
fn a_game_declares_its_units_and_the_ledger_layers_count_them() {
    // a tiny description of another game (not chess): units on a row, a unit steps to a free cell or takes the neighbour in front
    let text = "
(role red) (role blue)
(init (piece 1 red)) (init (piece 2 blue)) (init (piece 4 red)) (init (control red))
(<= (cellnum 1) ) (<= (cellnum 2)) (<= (cellnum 3)) (<= (cellnum 4))
(succ 1 2) (succ 2 3) (succ 3 4)
(<= (legal ?r (step ?x ?y)) (true (control ?r)) (true (piece ?x ?r)) (succ ?x ?y) (not (occupied ?y)))
(<= (legal ?r (take ?x ?y)) (true (control ?r)) (true (piece ?x ?r)) (succ ?x ?y) (true (piece ?y ?o)) (distinct ?r ?o))
(<= (legal ?r noop) (true (control ?c)) (distinct ?r ?c) (role ?r))
(<= (occupied ?y) (true (piece ?y ?r)))
(<= (next (piece ?y ?r)) (does ?r (step ?x ?y)))
(<= (next (piece ?y ?r)) (does ?r (take ?x ?y)))
(<= (next (piece ?z ?q)) (true (piece ?z ?q)) (does ?r (step ?x ?y)) (distinct ?z ?x))
(<= (next (piece ?z ?q)) (true (piece ?z ?q)) (does ?r (take ?x ?y)) (distinct ?z ?x) (distinct ?z ?y))
(<= (next (control blue)) (true (control red)))
(<= (next (control red)) (true (control blue)))
(<= (threat ?r ?x ?y) (true (piece ?x ?r)) (succ ?x ?y) (true (piece ?y ?o)) (distinct ?r ?o))
(<= terminal (not (has_moves)))
(<= has_moves (true (control ?r)) (legal ?r ?m) (distinct ?m noop))
(<= (goal ?r 50) (role ?r))
";
    let vocab = Vocabulary { derived: vec!["threat".into()], delta: vec!["piece".into()],
                             units: Some(cs_game::UnitDecl { relation: "piece".into(), owner: 1, kind: None }) };
    let g = GdlGame::load("row", text).unwrap().with_vocabulary(vocab.clone()).unwrap();
    let s0 = g.initial();
    assert_eq!(g.units(&s0).len(), 3);
    assert!(g.units(&s0).iter().all(|(_, k)| k == "unit"));
    // red's piece at 1 steps? no: cell 2 is blue's -> red takes it
    let take = g.moves(&s0).into_iter().find(|m| g.move_name(&s0, m) == "(take 1 2)").expect("red can take the neighbour");
    let cap = g.captured(&s0, &take);
    assert_eq!(cap.len(), 1);
    assert_eq!(g.taken_at(&s0, &take), vec![(1, "unit".to_string(), "2".to_string())]);
    let view = FactView::new(text, vocab).unwrap();
    let facts = view.facts(&g.base_facts(&s0)).unwrap();
    assert!(facts.contains(&Fact::new("threat", &["red", "1", "2"])));
    assert!(facts.contains(&Fact::new("piece", &["1", "red"])));
}
