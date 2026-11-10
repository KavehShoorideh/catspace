//! PLAN 78.14: the lemma mechanism end to end in Rust -- a snapshot's lemma bound to a position by the operator, the moves it proposes read first by the ledger
//! search WITHOUT changing the value found (an order, never a verdict), the ablation switch, and the card builder whose table a lemma is verified on.
use cs_decision::cards::card_of;
use cs_decision::{decide_with_lemmas, Params};
use cs_game::gdl_game::FactView;
use cs_game::kernel::KernelGame;
use cs_game::rng::Rng;
use cs_game::{Game, Vocabulary};
use cs_memory::cards::CardSet;
use cs_memory::lemma::Lemma;
use cs_memory::op::LemmaOp;
use cs_memory::snapshot::Snapshot;
use cs_operator::{Budget, Goal, Operator, Status};
use cs_tactics::ledger::{Count, Search};
use std::path::PathBuf;

fn spec() -> String { std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules/chess.gdl")).unwrap() }

const SNAPSHOT: &str = r#"{"format":1,"game":"chess","parent":null,
 "vocabulary":{"derived":["attack"],"delta":["cell"],"units":null},"judge":null,"teacher_used":false,"lemmas":[
 {"name":"take-the-attacker",
  "body":[{"relation":"cell","args":["?x","?y","us","?k"],"neg":false},{"relation":"attack","args":["them","?a","?b","?x","?y"],"neg":false},{"relation":"cell","args":["?a","?b","them","?j"],"neg":false}],
  "response":{"removed":[{"relation":"cell","args":["?a","?b","them","?j"],"neg":false}],"added":[{"relation":"cell","args":["?a","?b","us","?m"],"neg":false}]},
  "stats":{},"meta":{}}]}"#;

fn vocab() -> Vocabulary { Vocabulary { derived: vec!["attack".into()], delta: vec!["cell".into()], units: None } }

#[test]
fn the_operator_proposes_the_capture_of_the_attacker() {
    let snap = Snapshot::parse(SNAPSHOT.as_bytes()).unwrap();
    let view = FactView::new(&spec(), vocab()).unwrap();
    let g = KernelGame::from_fen("chess", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let s = g.initial();
    let op = LemmaOp::new(&snap, &view);
    let o = op.run(&g, &s, &Goal::Respond { role: 0 }, Budget::new(5.0, 1_000_000));
    assert_eq!(o.status, Status::Read);
    let names: Vec<String> = o.candidates.iter().map(|i| g.move_name(&s, &g.moves(&s)[*i])).collect();
    assert_eq!(names, vec!["e4d5".to_string()], "the lemma proposes the one move that takes the attacker");
    assert!(o.facts.iter().any(|f| f.relation == "fired" && f.args[0] == "take-the-attacker"));
    assert!(o.cost.nodes > 0 && !o.parts.is_empty());
    // the same lemma from the other seat: black to move, the white pawn attacked by nothing of black's kind here -> fires for the side to move only where it holds
    let gb = KernelGame::from_fen("chess", "4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1").unwrap();
    let sb = gb.initial();
    let ob = op.run(&gb, &sb, &Goal::Respond { role: 1 }, Budget::new(5.0, 1_000_000));
    let nb: Vec<String> = ob.candidates.iter().map(|i| gb.move_name(&sb, &gb.moves(&sb)[*i])).collect();
    assert_eq!(nb, vec!["d5e4".to_string()], "the lemma is side-relative: black's pawn takes the pawn that attacks it");
    // ablation: left out, nothing is proposed
    let mut ablated = LemmaOp::new(&snap, &view);
    ablated.exclude.insert("take-the-attacker".into());
    assert!(ablated.run(&g, &s, &Goal::Respond { role: 0 }, Budget::new(5.0, 1_000_000)).candidates.is_empty());
}

#[test]
fn proposed_moves_change_the_order_of_reading_and_never_the_value() {
    let g = KernelGame::new("chess", 518).unwrap();
    let mut rng = Rng::new(5);
    let mut compared = 0;
    for game in 0..6u64 {
        let mut s = g.initial();
        for _ in 0..(10 + 4 * game) {
            let ms = g.moves(&s);
            if ms.is_empty() { break; }
            s = g.apply(&s, &ms[rng.below(ms.len())]);
        }
        if g.outcome(&s).is_some() { continue; }
        let me = g.mover(&s).unwrap();
        let n = g.moves(&s).len();
        for prefer in [vec![], vec![n - 1], vec![n / 2, 0]] {
            let mut ls = Search::with_fuse(&g, &Count, me, 400_000_000);
            ls.prune = false;
            ls.extend = false;
            ls.root_first = prefer.clone();
            let line = ls.run(&s, 3);
            let mut base = Search::with_fuse(&g, &Count, me, 400_000_000);
            base.prune = false;
            base.extend = false;
            let b = base.run(&s, 3);
            assert_eq!(line.value, b.value, "the value is the same whatever is read first (prefer {prefer:?})");
            compared += 1;
        }
    }
    println!("OBS 78.14 prefer leaves the value alone: {compared} comparisons identical");
    assert!(compared >= 12);
}

#[test]
fn a_decision_with_the_operator_plays_and_reports_the_lemma_step() {
    let snap = Snapshot::parse(SNAPSHOT.as_bytes()).unwrap();
    let view = FactView::new(&spec(), vocab()).unwrap();
    let g = KernelGame::from_fen("chess", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let s = g.initial();
    let op = LemmaOp::new(&snap, &view);
    let p = Params { max_depth: 2, ledger_depth: 4, ..Params::default() };
    let d = decide_with_lemmas(&g, &s, 30.0, &Count, &p, Some(&op as &dyn Operator<KernelGame>)).unwrap();
    let ops: Vec<&str> = d.steps.iter().map(|st| st.operator.as_str()).collect();
    assert_eq!(ops, vec!["lemma", "forcing", "ledger"]);
    assert_eq!(g.move_name(&s, d.mv.as_ref().unwrap()), "e4d5");
    assert!(d.steps[2].goal.describe().contains("prefer 1"));
}

#[test]
fn cards_hold_exact_values_and_a_lemma_is_judged_on_them() {
    let view = FactView::new(&spec(), vocab()).unwrap();
    let snap = Snapshot::parse(SNAPSHOT.as_bytes()).unwrap();
    let lemma: Lemma = snap.lemmas[0].clone();
    let mut cards = vec![];
    for (i, fen) in ["4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1", "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/4K2R w - - 0 1"].iter().enumerate() {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        cards.push(card_of(&g, &view, &Count, &s, 3, 50_000_000, &format!("c{i}"), &format!("g{i}")).unwrap().expect("a card"));
    }
    let c0 = &cards[0];
    assert!(!c0.cut);
    let best = c0.moves.iter().map(|m| m.value).max().unwrap();
    let take = c0.moves.iter().find(|m| m.name == "e4d5").unwrap();
    assert_eq!(take.value, best, "taking the pawn that attacks is optimal");
    assert_eq!(take.gained, 1);
    let set = CardSet::new(cards);
    let ev = set.eval(&lemma, None, 64);
    println!("OBS 78.14 the hand lemma on 4 cards: fires {} holds {} refuted {} precision {:.2}", ev.fires, ev.holds, ev.refuted, ev.precision);
    assert_eq!(ev.fires, 3, "it fires on the first three (an attacked pawn), not on the bare rook endgame");
    assert!(ev.holds >= 2);
}

// ---------------------------------------------------------------------------------------------------------------------------------- another game

const ROW: &str = "
(role red) (role blue)
(init (piece 1 red)) (init (piece 2 blue)) (init (piece 4 red)) (init (piece 5 blue)) (init (control red))
(succ 1 2) (succ 2 3) (succ 3 4) (succ 4 5)
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
(<= terminal (not has_moves))
(<= has_moves (true (control ?r)) (legal ?r ?m) (distinct ?m noop))
(<= (goal ?r 50) (role ?r))
";

#[test]
fn the_whole_lemma_stack_runs_on_another_game_that_declares_its_units() {
    use cs_game::gdl_game::GdlGame;
    use cs_game::UnitDecl;
    let vocab = Vocabulary { derived: vec!["threat".into()], delta: vec!["piece".into()], units: Some(UnitDecl { relation: "piece".into(), owner: 1, kind: None }) };
    let g = GdlGame::load("row", ROW).unwrap().with_vocabulary(vocab.clone()).unwrap();
    let view = FactView::new(ROW, vocab).unwrap();
    // the states within six plies of the start
    let mut states = vec![g.initial()];
    let mut frontier = states.clone();
    for _ in 0..6 {
        let mut next = vec![];
        for s in &frontier { for m in g.moves(s) { next.push(g.apply(s, &m)); } }
        states.extend(next.clone());
        frontier = next;
    }
    let mut cards = vec![];
    for (i, s) in states.iter().enumerate() {
        if let Some(c) = card_of(&g, &view, &Count, s, 3, 5_000_000, &format!("s{i}"), &format!("g{i}")).unwrap() { cards.push(c); }
    }
    assert!(cards.len() > 5, "{} cards of the row game", cards.len());
    // the ledger layers see the game's units: taking a piece is a gain in the exact values
    let c0 = &cards[0];
    let take = c0.moves.iter().find(|m| m.name == "(take 1 2)").expect("red can take at the start");
    assert_eq!(take.gained, 1);
    // a lemma in the game's own vocabulary: "if a unit of ours threatens one of theirs, taking it is among the best" -- judged on the cards by the same code as chess
    let snap_text = r#"{"format":1,"game":"row","parent":null,"vocabulary":{"derived":["threat"],"delta":["piece"],"units":{"relation":"piece","owner":1,"kind":null}},"judge":null,"teacher_used":false,"lemmas":[
      {"name":"take-the-threatened","body":[{"relation":"threat","args":["us","?x","?y"],"neg":false}],
       "response":{"removed":[{"relation":"piece","args":["?x","us"],"neg":false},{"relation":"piece","args":["?y","them"],"neg":false}],"added":[{"relation":"piece","args":["?y","us"],"neg":false}]},"stats":{},"meta":{}}]}"#;
    let snap = Snapshot::parse(snap_text.as_bytes()).unwrap();
    let ev = CardSet::new(cards).eval(&snap.lemmas[0], None, 64);
    println!("OBS 78.14 another game, the row: lemma on {} cards fires {} holds {} refuted {} precision {:.2} excess {:.2}", ev.fires + ev.unknown + (0), ev.fires, ev.holds, ev.refuted, ev.precision, ev.excess);
    assert!(ev.fires >= 1 && ev.holds >= 1);
    // and the operator proposes it at the start
    let op = LemmaOp::new(&snap, &view);
    let o = op.run(&g, &g.initial(), &Goal::Respond { role: 0 }, Budget::new(5.0, 1_000_000));
    let names: Vec<String> = o.candidates.iter().map(|i| g.move_name(&g.initial(), &g.moves(&g.initial())[*i])).collect();
    assert_eq!(names, vec!["(take 1 2)".to_string(), "(take 4 5)".to_string()], "both of red's pieces threaten one of blue's");
}

// ---------------------------------------------------------------------------------------------------------------------------------- plan 78.24: a move's effect on a DERIVED relation

#[test]
fn a_derived_delta_relation_shows_in_the_cards_and_a_lemma_over_it_fires_on_the_moves_whose_effect_matches() {
    use cs_game::gdl_game::GdlGame;
    use cs_game::Fact;
    let derived = Vocabulary { derived: vec!["threat".into()], delta: vec!["threat".into()], units: None };
    let g = GdlGame::load("row", ROW).unwrap().with_vocabulary(Vocabulary { derived: vec!["threat".into()], delta: vec!["piece".into()], units: None }).unwrap();
    let view = FactView::new(ROW, derived.clone()).unwrap();
    let card = card_of(&g, &view, &Count, &g.initial(), 2, 1_000_000, "c", "g").unwrap().unwrap();
    let t = |a: &str, b: &str| Fact::new("threat", &["us", a, b]);
    let m12 = card.moves.iter().find(|m| m.name == "(take 1 2)").unwrap();
    let m45 = card.moves.iter().find(|m| m.name == "(take 4 5)").unwrap();
    assert_eq!((m12.removed.clone(), m12.added.clone()), (vec![t("1", "2")], vec![]), "taking 1 2 removes the threat 1->2 (4->5 stays)");
    assert_eq!((m45.removed.clone(), m45.added.clone()), (vec![t("4", "5")], vec![]));
    // the same position with a base-only delta: the cards are the ones of old (piece facts, no derived relation)
    let base_view = FactView::new(ROW, Vocabulary { derived: vec!["threat".into()], delta: vec!["piece".into()], units: None }).unwrap();
    let bc = card_of(&g, &base_view, &Count, &g.initial(), 2, 1_000_000, "c", "g").unwrap().unwrap();
    let b12 = bc.moves.iter().find(|m| m.name == "(take 1 2)").unwrap();
    assert!(b12.removed.iter().all(|f| f.relation == "piece") && b12.removed.len() == 2 && b12.added == vec![Fact::new("piece", &["2", "us"])]);
    // a lemma whose response is over the derived relation proposes exactly the moves whose effect matches
    let text = r#"{"format":1,"game":"row","parent":null,"vocabulary":{"derived":["threat"],"delta":["threat"],"units":null},"judge":null,"teacher_used":false,"lemmas":[
      {"name":"lift-threat-12","body":[{"relation":"threat","args":["us","?x","?y"],"neg":false}],
       "response":{"removed":[{"relation":"threat","args":["us","1","2"],"neg":false}],"added":[]},"stats":{},"meta":{}}]}"#;
    let snap = Snapshot::parse(text.as_bytes()).unwrap();
    let op = LemmaOp::new(&snap, &view);
    let s = g.initial();
    let o = op.run(&g, &s, &Goal::Respond { role: 0 }, Budget::new(5.0, 1_000_000));
    let names: Vec<String> = o.candidates.iter().map(|i| g.move_name(&s, &g.moves(&s)[*i])).collect();
    assert_eq!(names, vec!["(take 1 2)".to_string()]);
}

#[test]
fn the_effects_reader_reads_base_facts_only_unless_a_delta_relation_is_derived() {
    use cs_memory::op::Effects;
    let g = cs_game::gdl_game::GdlGame::load("row", ROW).unwrap();
    let view = FactView::new(ROW, Vocabulary { derived: vec!["threat".into()], delta: vec![], units: None }).unwrap();
    let (base_rel, derived_rel) = (vec!["piece".to_string()], vec!["piece".to_string(), "threat".to_string()]);
    let s = g.initial();
    let base = g.base_facts(&s);
    let (a, b) = (Effects::at(&view, &base_rel, &base).unwrap(), Effects::at(&view, &derived_rel, &base).unwrap());
    assert!(!a.on_full_views() && b.on_full_views());
    let m = g.moves(&s).into_iter().find(|m| g.move_name(&s, m) == "(take 1 2)").unwrap();
    let after = g.base_facts(&g.apply(&s, &m));
    let (ra, aa) = a.of(&after).unwrap();
    let (rb, ab) = b.of(&after).unwrap();
    assert_eq!((ra.len(), aa.len()), (2, 1), "base only: two pieces leave their squares, one arrives");
    assert!(rb.iter().any(|f| f.relation == "threat") && rb.iter().filter(|f| f.relation == "piece").count() == 2 && ab.len() == 1);
}
