//! PLAN 78.19 A1/A2/A4: `run_for` equals `run` on every relation of the cone, the statics are shared, the indexes are kept (not cleared). Small descriptions
//! written here; the chess-sized check is game/tests/demand.rs.
use cs_gdl::{Program, Rel, Row, ARITY_MAX, FREE};

const TEXT: &str = "
(role a) (role b)
(edge 1 2) (edge 2 3) (edge 3 4)
(<= (reach ?x ?y) (edge ?x ?y))
(<= (reach ?x ?z) (reach ?x ?y) (edge ?y ?z))
(<= (lit ?x) (true (on ?x)))
(<= (dark ?x) (reach 1 ?x) (not (lit ?x)))
(<= (unrelated ?x) (true (on ?x)) (distinct ?x 9))
(<= (legal a go) (true (on 1)))
";

fn row(p: &mut Program, xs: &[&str]) -> Row { let mut r = [FREE; ARITY_MAX]; for (i, x) in xs.iter().enumerate() { r[i] = p.terms.sym(x); } r }

fn state(p: &mut Program, on: &[&str]) -> Vec<(Rel, Row)> {
    let t = p.rel_id("true.on").unwrap();
    on.iter().map(|x| (t, row(p, &[x]))).collect()
}

fn rows_of(db: &cs_gdl::Db, r: Rel) -> Vec<Row> { let mut v = db.facts(r); v.sort(); v }

#[test]
fn run_for_equals_run_on_every_relation_of_the_cone() {
    let mut p = Program::load(TEXT).unwrap();
    let wanted = [p.rel_id("dark").unwrap()];
    let cone = p.cone(&wanted);
    assert!(cone.contains(&p.rel_id("reach").unwrap()) && cone.contains(&p.rel_id("lit").unwrap()), "a negated dependency is in the cone");
    assert!(!cone.contains(&p.rel_id("unrelated").unwrap()) && !cone.contains(&p.rel_id("legal").unwrap()));
    let mut checked = 0;
    for on in [vec![], vec!["2"], vec!["2", "3"], vec!["1", "4"]] {
        let facts = state(&mut p, &on);
        let full = p.run(&facts);
        let dem = p.run_for(&facts, &wanted);
        for &r in &cone { assert_eq!(rows_of(&full, r), rows_of(&dem, r), "{} at {on:?}", p.rel_name(r)); checked += 1; }
        // demand really skipped the strata outside the cone
        let u = p.rel_id("unrelated").unwrap();
        assert!(dem.facts(u).is_empty());
        if !on.is_empty() { assert!(!full.facts(u).is_empty()); }
    }
    assert!(checked > 10);
}

#[test]
fn a_wrong_cone_disagrees() {
    // NEGATIVE CONTROL: the cone of `dark` without `lit` (which `dark` negates) must give a different `dark`
    let mut p = Program::load(TEXT).unwrap();
    let dark = p.rel_id("dark").unwrap();
    let mut cone = p.cone(&[dark]);
    assert!(cone.remove(&p.rel_id("lit").unwrap()));
    let facts = state(&mut p, &["2"]);
    let full = p.run(&facts);
    let bad = p.run_in(&facts, &cone);
    assert_ne!(rows_of(&full, dark), rows_of(&bad, dark), "dropping a needed relation must be visible");
}

#[test]
fn statics_are_shared_and_runs_do_not_leak() {
    let mut p = Program::load(TEXT).unwrap();
    let edge = p.rel_id("edge").unwrap();
    let before = p.statics.get(&edge).unwrap().order.len();
    let f1 = state(&mut p, &["2"]);
    let d1 = p.run(&f1);
    let f2 = state(&mut p, &[]);
    let d2 = p.run(&f2);
    let lit = p.rel_id("lit").unwrap();
    assert_eq!(d1.facts(lit).len(), 1);
    assert!(d2.facts(lit).is_empty(), "a run's relations are its own");
    assert_eq!(p.statics.get(&edge).unwrap().order.len(), before);
    assert_eq!(rows_of(&d1, edge).len(), before, "a static relation reads through the layer");
    assert!(std::rc::Rc::strong_count(&p.statics) >= 1);
}

#[test]
fn the_recursive_closure_is_complete() {
    let mut p = Program::load(TEXT).unwrap();
    let reach = p.rel_id("reach").unwrap();
    let db = p.run(&[]);
    assert_eq!(db.facts(reach).len(), 6);
}
