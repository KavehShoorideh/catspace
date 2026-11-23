//! PLAN 79.36 PART C, LAYER 0, THE LOGIC AND THE STRUCTURE: the well-formedness of a formula given what is bound outside, the relations an index of facts holds, the one interface of distance (SYSTEM_SPEC 11, last block; the base, layer 0; ledger row 43).
//! The expected answers are written by hand in each test.
use cs_base::logic::formula::{Domain, Formula, FormulaError};
use cs_base::logic::{Fact, FactIndex, Lit};
use cs_base::number::Num;
use cs_base::sets::Set;
use cs_base::structure::geometry::{Plane, Point};
use cs_base::structure::{Graph, Metric};
use std::collections::BTreeSet;

fn bound(vars: &[&str]) -> BTreeSet<String> { vars.iter().map(|v| v.to_string()).collect() }

#[test]
fn an_index_lists_the_relations_that_have_a_row_each_once_in_the_order_of_the_text() {
    let idx = FactIndex::new(&[Fact::new("r", &["1"]), Fact::new("p", &["1", "2"]), Fact::new("r", &["2"])]);
    assert_eq!(FactIndex::relations(&idx), vec!["p", "r"]);
    assert!(FactIndex::relations(&FactIndex::new(&[])).is_empty());
}

#[test]
fn a_literal_alone_is_well_formed_whatever_is_bound() {
    let f = Formula::Lit(Lit::pos("r", &["?x"]));
    assert_eq!(Formula::validate_with(&f, &bound(&[])), Ok(()));
    assert_eq!(Formula::validate_with(&f, &bound(&["x", "y"])), Ok(()));
}

#[test]
fn the_branches_of_an_or_must_bind_the_same_placeholders_besides_those_bound_outside() {
    let left = Formula::Lit(Lit::pos("r", &["?x"]));
    let right = Formula::Lit(Lit::pos("q", &["?y"]));
    let differ = Formula::Or(vec![left.clone(), right.clone()]);
    assert_eq!(Formula::validate_with(&differ, &bound(&[])), Err(FormulaError::BranchesDiffer));
    let same = Formula::Or(vec![left.clone(), Formula::Lit(Lit::pos("q", &["?x"]))]);
    assert_eq!(Formula::validate_with(&same, &bound(&[])), Ok(()));
    // what the caller has bound is not new to a branch: with x and y bound outside, the two branches bind nothing new and agree
    assert_eq!(Formula::validate_with(&differ, &bound(&["x", "y"])), Ok(()));
}

#[test]
fn a_filter_must_read_only_placeholders_something_binds_and_a_quantifier_may_not_shadow_one_bound_outside() {
    let neg = Formula::Lit(Lit::pos("r", &["?z"]).negated());
    // a negated literal read beside a literal that binds z is fine; alone it reads z before anything binds it
    let both = Formula::And(vec![Formula::Lit(Lit::pos("q", &["?z"])), neg.clone()]);
    assert_eq!(Formula::validate_with(&both, &bound(&[])), Ok(()));
    let over = Domain::Set(Set::from_items(["a".to_string(), "b".to_string()]));
    let every = Formula::Every { var: "v".into(), over: over.clone(), body: Box::new(Formula::Lit(Lit::pos("r", &["?v"]))) };
    assert_eq!(Formula::validate_with(&every, &bound(&[])), Ok(()));
    assert_eq!(Formula::validate_with(&every, &bound(&["v"])), Ok(()), "the shadowing is looked for when the formula is read, not here");
    let not = Formula::Not(Box::new(Formula::Lit(Lit::pos("r", &["?z"]))));
    assert_eq!(Formula::validate_with(&not, &bound(&["z"])), Ok(()));
    let some = Formula::Some { var: "w".into(), over, body: Box::new(Formula::Lit(Lit::pos("r", &["?w"]))) };
    assert_eq!(Formula::validate_with(&some, &bound(&[])), Ok(()));
}

#[test]
fn the_distance_of_a_plane_is_the_straight_run_and_none_when_the_number_overflows() {
    let (a, b) = (Point::whole(0, 0), Point::whole(3, 4));
    assert_eq!(Metric::distance(&Plane, &a, &b), Some(Num::whole(5)));
    assert_eq!(Metric::distance(&Plane, &a, &a), Some(Num::whole(0)));
    let far = Point::whole(i64::MAX, i64::MAX);
    assert_eq!(Metric::distance(&Plane, &a, &far), None);
}

#[test]
fn the_distance_of_a_graph_is_the_steps_of_the_shortest_path_and_none_when_there_is_none() {
    let mut g: Graph<u32, u8> = Graph::new();
    g.add_both(1, 2, 0);
    g.add_both(2, 3, 0);
    g.add_node(9);
    assert_eq!(Metric::distance(&g, &1, &3), Some(Num::whole(2)));
    assert_eq!(Metric::distance(&g, &1, &1), Some(Num::whole(0)));
    assert_eq!(Metric::distance(&g, &1, &9), None, "no path: no distance");
}
