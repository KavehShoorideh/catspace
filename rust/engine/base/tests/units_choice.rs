//! PLAN 79.36 PART C, LAYER 0, THE CHOICE: the choosers, the tree and what it holds, the orders, the wants, the forcing by chance, the guarantee (SYSTEM_SPEC 11, last block; the base, layer 0; ledger row 43).
//! Layer 0 names no thing of a world: the trees here are trees of numbers. The expected values are worked out by hand in the comments.
use cs_base::chance::SUM_TOLERANCE;
use cs_base::choice::{force_chance, force_chance_set, guarantee, party, Chooser, Order, Tree, Wants};
use cs_base::number::{Num, NumError};
use cs_base::sets::Set;
use cs_base::structure::Sequence;

const A: Chooser = Chooser::A;
const B: Chooser = Chooser::B;

fn leaf(v: i64) -> Tree<i64> { Tree::Result(v) }
fn score() -> Order<i64> { Order::by(|r: &i64| Num::whole(*r)) }

#[test]
fn the_tolerance_for_a_real_number_in_a_chance_is_a_billionth() {
    assert_eq!(SUM_TOLERANCE, 1e-9);
}

#[test]
fn a_chooser_is_its_number_and_the_first_two_are_nought_and_one() {
    assert_eq!(Chooser::id(Chooser::new(7)), 7);
    assert_eq!((Chooser::id(A), Chooser::id(B)), (0, 1));
    assert_eq!(Chooser::new(0), A);
}

#[test]
fn a_set_made_empty_has_no_member_and_takes_the_ones_put_in() {
    let mut s = Set::<u32>::new();
    assert!(s.is_empty() && s.len() == 0);
    assert!(s.insert(3) && !s.insert(3));
    assert!(s.contains(&3) && !s.contains(&4));
}

#[test]
fn the_results_of_a_tree_are_listed_left_to_right_a_result_met_twice_twice_and_a_lone_result_is_its_own_list() {
    assert_eq!(Tree::results(&leaf(5)), vec![&5]);
    let t = Tree::turn(A, vec![leaf(1), Tree::turn(B, vec![leaf(2), leaf(1)]).unwrap()]).unwrap();
    assert_eq!(Tree::results(&t), vec![&1, &2, &1]);
    let table = Tree::table(2, 2, vec![leaf(1), leaf(2), leaf(3), leaf(4)]).unwrap();
    assert_eq!(Tree::results(&table), vec![&1, &2, &3, &4], "cells in row-major order");
    let chance = Tree::chance(vec![(leaf(8), Num::frac(1, 4).unwrap()), (leaf(9), Num::frac(3, 4).unwrap())]).unwrap();
    assert_eq!(Tree::results(&chance), vec![&8, &9]);
}

#[test]
fn a_tree_says_whether_a_choice_by_chance_or_a_choice_at_once_is_anywhere_in_it() {
    let plain = Tree::turn(A, vec![leaf(1), leaf(2)]).unwrap();
    assert!(!Tree::has_chance(&plain) && !Tree::has_table(&plain));
    let table = Tree::table(1, 1, vec![leaf(1)]).unwrap();
    let chance = Tree::chance(vec![(leaf(1), Num::whole(1))]).unwrap();
    assert!(Tree::has_table(&table) && !Tree::has_chance(&table));
    assert!(Tree::has_chance(&chance) && !Tree::has_table(&chance));
    let deep = Tree::turn(B, vec![leaf(0), Tree::chance(vec![(table.clone(), Num::whole(1))]).unwrap()]).unwrap();
    assert!(Tree::has_chance(&deep) && Tree::has_table(&deep), "found below other nodes");
    let deeper = Tree::turn(A, vec![Tree::table(1, 1, vec![chance]).unwrap()]).unwrap();
    assert!(Tree::has_chance(&deeper) && Tree::has_table(&deeper));
    assert!(!Tree::has_chance(&leaf(1)) && !Tree::has_table(&leaf(1)));
}

#[test]
fn an_order_scores_a_result_by_its_score_and_the_reversed_order_by_minus_it_and_an_overflow_is_said() {
    let o = score();
    assert_eq!(Order::score(&o, &3), Ok(Num::whole(3)));
    assert_eq!(Order::score(&o.clone().reversed(), &3), Ok(Num::whole(-3)));
    let worst = Order::by(|_: &i64| Num::whole(i64::MIN)).reversed();
    assert_eq!(Order::score(&worst, &0), Err(NumError::Overflow), "minus the least whole number is not a whole number");
}

#[test]
fn what_each_chooser_wants_is_the_order_given_to_it_and_a_chooser_given_none_has_none() {
    let w = Wants::new().with(A, score());
    assert!(Wants::order(&w, A).is_some());
    assert!(Wants::order(&w, B).is_none());
    assert_eq!(Wants::order(&w, A).unwrap().score(&4), Ok(Num::whole(4)));
}

#[test]
fn the_least_chance_a_set_can_make_a_property_hold_weighs_the_chance_nodes_and_lets_the_rest_pick_against_it() {
    // a half: the result 1; a half: B picks between 2 and 0. The property: the result is at least 1.
    let t = Tree::chance(vec![(leaf(1), Num::frac(1, 2).unwrap()), (Tree::turn(B, vec![leaf(2), leaf(0)]).unwrap(), Num::frac(1, 2).unwrap())]).unwrap();
    let holds = |r: &i64| *r >= 1;
    let against = force_chance_set(&t, &party(&[A]), &holds).unwrap();
    assert_eq!(against.as_ratio(), Some((1, 2)), "B picks 0: only the first half holds");
    let together = force_chance_set(&t, &party(&[A, B]), &holds).unwrap();
    assert_eq!(together.as_ratio(), Some((1, 1)), "with B in the set it picks 2: both halves hold");
    assert_eq!(force_chance_set(&t, &party(&[A]), &|_: &i64| false).unwrap(), Num::whole(0));
    assert_eq!(force_chance_set(&leaf(4), &party(&[A]), &|r: &i64| *r == 4).unwrap(), Num::whole(1));
}

#[test]
fn the_least_chance_for_one_chooser_is_the_case_of_a_set_of_one() {
    let t = Tree::turn(A, vec![Tree::chance(vec![(leaf(1), Num::frac(1, 3).unwrap()), (leaf(0), Num::frac(2, 3).unwrap())]).unwrap(), leaf(0)]).unwrap();
    let holds = |r: &i64| *r == 1;
    assert_eq!(force_chance(&t, A, &holds), force_chance_set(&t, &party(&[A]), &holds));
    assert_eq!(force_chance(&t, A, &holds).unwrap().as_ratio(), Some((1, 3)), "A takes the chance node: a third");
    assert_eq!(force_chance(&t, B, &holds).unwrap().as_ratio(), Some((0, 1)), "B picks the option where it cannot hold");
}

#[test]
fn what_a_set_can_make_sure_of_is_the_best_at_its_own_turn_and_the_worst_at_the_turn_of_the_rest() {
    // A picks between 3 and a turn of B between 5 and 4
    let t = Tree::turn(A, vec![leaf(3), Tree::turn(B, vec![leaf(5), leaf(4)]).unwrap()]).unwrap();
    assert_eq!(guarantee(&t, &party(&[A]), &score()), Ok(Num::whole(4)), "A: the better of 3 and what B leaves it (the worse of 5 and 4)");
    assert_eq!(guarantee(&t, &party(&[B]), &score()), Ok(Num::whole(3)), "B in the set picks its best (5), the rest pick the worst of 3 and 5");
    assert_eq!(guarantee(&leaf(9), &party(&[A]), &score()), Ok(Num::whole(9)));
}

#[test]
fn a_guarantee_by_chance_is_the_expectation_and_the_first_overflow_is_said() {
    let t = Tree::chance(vec![(leaf(2), Num::frac(1, 2).unwrap()), (leaf(6), Num::frac(1, 2).unwrap())]).unwrap();
    assert_eq!(guarantee(&t, &party(&[A]), &score()).unwrap().as_ratio(), Some((4, 1)));
    let big = Tree::turn(A, vec![leaf(i64::MAX), leaf(1)]).unwrap();
    assert_eq!(guarantee(&big, &party(&[A]), &score()), Ok(Num::whole(i64::MAX)));
    let neg = Order::by(|_: &i64| Num::whole(i64::MIN)).reversed();
    assert_eq!(guarantee(&big, &party(&[A]), &neg), Err(NumError::Overflow));
}

#[test]
fn the_items_of_a_sequence_are_the_ones_it_was_made_of_in_order() {
    let s = Sequence::new(vec![3, 1, 2]);
    assert_eq!(Sequence::items(&s), &[3, 1, 2]);
    assert!(Sequence::<u8>::items(&Sequence::new(vec![])).is_empty());
}
