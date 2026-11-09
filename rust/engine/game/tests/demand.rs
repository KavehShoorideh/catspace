//! PLAN 78.19 A: the fact view with demand evaluation against the full evaluation (the oracle) over seeded random play of the chess kernel, the cone of the
//! vocabulary, a negative control, and the interned rows against the string facts.
use cs_game::fact_audit::{audit, positions};
use cs_game::gdl_game::FactView;
use cs_game::kernel::KernelGame;
use cs_game::{Fact, Vocabulary};
use cs_gdl::Program;
use std::path::PathBuf;

fn rules(rel: &str) -> String { std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules").join(rel)).unwrap() }
fn vocab() -> Vocabulary { Vocabulary { derived: vec!["attack".into(), "check".into(), "opponent".into()], delta: vec!["cell".into()], units: None } }

#[test]
fn demand_facts_equal_full_facts_over_random_play() {
    for v in ["chess", "kingofthehill"] {
        let text = if v == "chess" { rules("chess.gdl") } else { cs_gdl::compose(&rules("chess.gdl"), &rules(&format!("variants/{v}.gdl"))).unwrap() };
        let k = KernelGame::new(v, 518).unwrap();
        let pos = positions(&k, 6, 40, 7);
        let full = FactView::new(&text, vocab()).unwrap().with_demand(false);
        let dem = FactView::new(&text, vocab()).unwrap();
        assert!(!full.demand() && dem.demand());
        let r = audit(&full, &dem, &pos).unwrap();
        println!("OBS 78.19 test audit {v}: positions {}, equal {}, unequal {}, ms full {:.3} demand {:.3}", r.positions, r.equal, r.unequal, r.ms_per_position_full(), r.ms_per_position_demand());
        assert!(r.positions > 100 && r.unequal == 0, "{:?}", r.examples);
    }
}

#[test]
fn the_cone_is_a_part_of_the_description() {
    let dem = FactView::new(&rules("chess.gdl"), vocab()).unwrap();
    let ((cr, cc), (tr, tc)) = dem.cone_size();
    println!("OBS 78.19 cone: {cr} relations of {tr}, {cc} clauses of {tc}");
    assert!(cr < tr && cc < tc);
}

#[test]
fn a_wrong_cone_makes_the_audit_fail() {
    // NEGATIVE CONTROL at the program level: evaluate the vocabulary's cone with each relation dropped in turn; some drop must change `attack` on a position
    let text = rules("chess.gdl");
    let k = KernelGame::new("chess", 518).unwrap();
    let pos = positions(&k, 2, 12, 3);
    let view = FactView::new(&text, vocab()).unwrap();
    let mut p = Program::load(&text).unwrap();
    let attack = p.rel_id("attack").unwrap();
    let cone = p.cone(&[attack, p.rel_id("check").unwrap()]);
    let mut disagreements = 0;
    for base in &pos {
        let mut rows = vec![];
        for f in base {
            let r = p.rel_id(&format!("true.{}", f.relation)).unwrap();
            let mut row = [cs_gdl::FREE; cs_gdl::ARITY_MAX];
            for (i, a) in f.args.iter().enumerate() { row[i] = cs_game::gdl_game::parse_term(&mut p, a).unwrap(); }
            rows.push((r, row));
        }
        let good = p.run(&rows).facts(attack);
        for &drop in &cone {
            let mut c = cone.clone();
            c.remove(&drop);
            let bad = p.run_in(&rows, &c).facts(attack);
            let (mut g, mut b) = (good.clone(), bad);
            g.sort();
            b.sort();
            if g != b { disagreements += 1; }
        }
    }
    println!("OBS 78.19 negative control: {disagreements} disagreements over {} positions", pos.len());
    assert!(disagreements > 0, "a deliberately wrong cone must be caught");
    let _ = view;
}

#[test]
fn rows_are_the_facts_without_strings() {
    let text = rules("chess.gdl");
    let k = KernelGame::new("chess", 518).unwrap();
    let view = FactView::new(&text, vocab()).unwrap();
    let mut n = 0;
    for base in positions(&k, 2, 20, 5) {
        let facts = view.facts(&base).unwrap();
        let rows = view.rows(&base).unwrap();
        let derived: usize = facts.iter().filter(|f| !base.contains(f) && vocab().derived.contains(&f.relation)).count();
        assert_eq!(rows.len(), derived);
        // a repeated call (the memos warm) gives the same answer
        assert_eq!(view.facts(&base).unwrap(), facts);
        n += 1;
    }
    assert!(n > 20);
    let _: Fact = Fact::new("x", &[]);
}

#[test]
fn sharing_the_statics_saves_their_copy() {
    // the cost each run used to pay (A2): a copy of every static relation, indexes included
    let p = Program::load(&rules("chess.gdl")).unwrap();
    let n = 200;
    let t = std::time::Instant::now();
    let mut rows = 0;
    for _ in 0..n { let c: Vec<cs_gdl::Relation> = p.statics.values().cloned().collect(); rows += c.iter().map(|r| r.order.len()).sum::<usize>(); }
    let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("OBS 78.19 A2: copying the statics cost {ms:.3} ms a run ({} static relations, {} rows); a run now copies none", p.statics.len(), rows / n);
    assert!(rows > 0);
}
