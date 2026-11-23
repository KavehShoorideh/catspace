//! PLAN 79.36 PART C, LAYER 0, THE WALK AND THE EXPLICIT TREE AS A NODE: the numbers of the hot path, what a walk answers, what a cut is, the line kept by hand, the tree as a node that gives its options (SYSTEM_SPEC 11, last block; the base, layer 0; ledger row 43).
//! The nodes are nodes of small graphs of numbers written here; the expected steps are worked out by hand in the comments.
use cs_base::choice::tree_space::{Free, TreeSpace};
use cs_base::choice::walk::{Ask, Cut, Look, Node, Out, Space, Steer, Walk, Whole, MOST};
use cs_base::choice::{party, Chooser, Tree};
use cs_base::number::Num;

const A: Chooser = Chooser::A;
const B: Chooser = Chooser::B;

fn leaf(v: i64) -> Tree<i64> { Tree::Result(v) }

// ------------------------------------------------------------------------------------------------------------------------------ the numbers of the hot path

#[test]
fn a_walk_reads_at_most_sixty_four_properties_together() {
    assert_eq!(MOST, 64);
}

#[test]
fn a_whole_number_of_the_hot_path_has_its_largest_as_none_nought_as_zero_and_goes_up_and_down_by_one() {
    assert_eq!(<u8 as Whole>::NONE, 255);
    assert_eq!(<u16 as Whole>::NONE, u16::MAX);
    assert_eq!(<usize as Whole>::NONE, usize::MAX);
    assert_eq!((<u8 as Whole>::ZERO, <u32 as Whole>::ZERO, <u64 as Whole>::ZERO), (0, 0, 0));
    assert_eq!(Whole::up(3u8), 4);
    assert_eq!(Whole::down(3u8), 2);
    assert_eq!(Whole::up(0u16), 1);
    assert_eq!(Whole::down(1usize), 0);
}

#[test]
fn going_down_from_nought_stays_at_nought_and_never_becomes_the_mark_of_no_answer() {
    // mended 2026-11-23 (found by plan 79.36): before, this wrapped to the largest number, which is the mark of no answer
    assert_eq!(Whole::down(0u8), 0);
    assert_eq!(Whole::down(0u16), 0);
    assert_eq!(Whole::down(0u32), 0);
    assert_eq!(Whole::down(0u64), 0);
    assert_eq!(Whole::down(0usize), 0);
    assert_ne!(Whole::down(0u8), <u8 as Whole>::NONE);
}

#[test]
fn an_answer_sheet_of_nothing_holds_none_for_every_property_in_both_readings() {
    let out = Out::<u8>::none();
    assert_eq!(out.strict.len(), MOST);
    assert!(out.strict.iter().all(|n| *n == 255) && out.loose.iter().all(|n| *n == 255));
    let big = Out::<u16>::none();
    assert!(big.strict.iter().all(|n| *n == u16::MAX));
}

// ------------------------------------------------------------------------------------------------------------------------------ a small graph as a node

/// A graph of nodes by number. `own[n]` says whether the asker picks at n; `held[n]` the properties that hold at n; `adj[n]` the options of n (the nodes they lead to); a node with no option ends a line.
struct Graph { adj: Vec<Vec<usize>>, own: Vec<bool>, held: Vec<u64>, at: Vec<usize>, record: Vec<u64> }

impl Graph {
    fn new(adj: Vec<Vec<usize>>, own: Vec<bool>, held: Vec<u64>) -> Graph { Graph { adj, own, held, at: vec![0], record: vec![] } }
}

impl Node for Graph {
    type Opt = usize;
    fn own_turn(&self) -> bool { self.own[*self.at.last().unwrap()] }
    fn options(&mut self, out: &mut Vec<usize>) { out.extend(self.adj[*self.at.last().unwrap()].iter().copied()); }
    fn enter(&mut self, o: &usize) { self.at.push(*o); }
    fn leave(&mut self) { self.at.pop(); }
}

impl Space for Graph {
    const CYCLES: bool = true;
    fn key(&self) -> u64 { *self.at.last().unwrap() as u64 }
    fn look(&mut self) -> Look { let n = *self.at.last().unwrap(); Look { bound: self.held[n], held: self.held[n], ended: self.adj[n].is_empty() } }
    fn met_before(&self, key: u64) -> bool { self.record.contains(&key) }
    fn breaks(&mut self, _cand: u64, _opts: &[usize]) -> u64 { 0 }
}

/// What steers: a budget of nodes (`None`: no limit), whether answers are kept, and a record of what it was asked.
struct Steering { budget: Option<u32>, keeps: bool, spent: u32, ordered: Vec<usize>, reverse: bool }

impl Steering {
    fn new() -> Steering { Steering { budget: None, keeps: false, spent: 0, ordered: vec![], reverse: false } }
}

impl Steer<Graph> for Steering {
    fn spend(&mut self, _asked: Ask) -> Result<(), Cut> {
        self.spent += 1;
        match self.budget { Some(b) if self.spent > b => Err(Cut), _ => Ok(()) }
    }
    fn order(&mut self, _space: &Graph, opts: &mut Vec<usize>, first: usize) {
        self.ordered.push(first);
        if self.reverse { opts.reverse(); }
    }
    fn keeps(&self) -> bool { self.keeps }
}

/// 0 (own) -> 1, 2;  1 (rest) -> 3, 4;  2 (rest) -> 5;  3 holds property 0 only;  4 holds property 0 and 1;  5 holds nothing and ends
fn graph() -> Graph {
    Graph::new(vec![vec![1, 2], vec![3, 4], vec![5], vec![], vec![], vec![]], vec![true, false, false, false, false, false], vec![0, 0, 0, 0b01, 0b11, 0])
}

fn ask(strict: u64) -> Ask { Ask { strict, loose: 0 } }

#[test]
fn a_property_held_at_the_node_asked_about_is_met_in_nought_steps() {
    let mut g = graph();
    g.at = vec![3];
    let mut w = Walk::<_, _, u8>::new(g, Steering::new());
    let mut out = Out::<u8>::none();
    assert_eq!(w.go(5, ask(0b01), false, false, &mut out), Ok(()));
    assert_eq!(out.strict[0], 0);
    assert_eq!(out.strict[1], 255, "a property not asked is not answered");
}

#[test]
fn at_the_asker_s_turn_some_option_must_meet_the_property_and_at_the_rest_s_every_option_so_the_steps_are_the_longest_needed() {
    // from 0 the asker goes to 1, where the rest pick 3 or 4: property 0 holds at both, in two steps in all; property 1 holds only at 4: the rest can pick 3, so it is not met
    let mut w = Walk::<_, _, u8>::new(graph(), Steering::new());
    let mut out = Out::<u8>::none();
    assert_eq!(w.go(4, ask(0b11), false, false, &mut out), Ok(()));
    assert_eq!(out.strict[0], 2);
    assert_eq!(out.strict[1], 255, "the rest can always pick the option where it does not hold");
}

#[test]
fn a_property_not_met_within_the_depth_leaves_the_answer_as_it_was_and_marks_that_a_line_reached_the_limit() {
    let mut w = Walk::<_, _, u8>::new(graph(), Steering::new());
    let mut out = Out::<u8>::none();
    assert_eq!(w.go(1, ask(0b01), false, false, &mut out), Ok(()));
    assert_eq!(out.strict[0], 255, "two steps are needed and one is allowed");
    assert!(w.horizon, "a line met the limit with a strict property still open");
    let mut w2 = Walk::<_, _, u8>::new(graph(), Steering::new());
    let mut out2 = Out::<u8>::none();
    w2.go(2, ask(0b01), false, false, &mut out2).unwrap();
    assert_eq!(out2.strict[0], 2);
    assert!(!w2.horizon);
}

#[test]
fn a_node_that_ends_a_line_meets_only_what_it_holds_itself() {
    let mut g = graph();
    g.at = vec![5];
    let mut w = Walk::<_, _, u8>::new(g, Steering::new());
    let mut out = Out::<u8>::none();
    w.go(9, ask(0b01), false, false, &mut out).unwrap();
    assert_eq!(out.strict[0], 255);
    assert!(!w.horizon, "the line ended: it did not meet the limit");
}

#[test]
fn a_walk_stopped_by_what_steers_it_is_a_cut_and_never_an_answer() {
    let mut s = Steering::new();
    s.budget = Some(2);
    let mut w = Walk::<_, _, u8>::new(graph(), s);
    let mut out = Out::<u8>::none();
    assert_eq!(w.go(4, ask(0b01), false, false, &mut out), Err(Cut));
    assert_eq!(out.strict[0], 255, "no answer is given by a stopped walk");
    assert_eq!(w.steer.spent, 3, "the third node entered was refused");
}

#[test]
fn the_answers_do_not_depend_on_the_order_the_options_are_tried_in_nor_on_keeping_answers() {
    let mut answers = vec![];
    for (reverse, keeps) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut s = Steering::new();
        s.reverse = reverse;
        s.keeps = keeps;
        let mut w = Walk::<_, _, u8>::new(graph(), s);
        let mut out = Out::<u8>::none();
        w.go(4, ask(0b11), false, false, &mut out).unwrap();
        answers.push((out.strict[0], out.strict[1]));
        assert!(!w.steer.ordered.is_empty() && w.steer.ordered.iter().all(|f| *f <= 1), "the order is asked with the first open property");
    }
    assert!(answers.iter().all(|a| *a == (2, 255)), "{answers:?}");
}

#[test]
fn a_node_met_again_on_the_line_walked_by_hand_ends_the_line_and_taints_the_answer_and_taking_the_key_off_lets_it_go_on() {
    let mut g = graph();
    g.at = vec![1];
    let mut w = Walk::<_, _, u8>::new(g, Steering::new());
    w.line_push(1);
    let mut out = Out::<u8>::none();
    w.go(4, ask(0b01), true, false, &mut out).unwrap();
    assert_eq!(out.strict[0], 255, "the node of the line is met again: no answer");
    assert_eq!(w.taint, 0b01, "the strict reading rests on the line");
    w.line_pop();
    w.taint = 0;
    w.go(4, ask(0b01), true, false, &mut out).unwrap();
    assert_eq!(out.strict[0], 1, "from 1 the rest pick 3 or 4: one step to a node that holds it");
    assert_eq!(w.taint, 0);
}

/// 0 (own) -> 1;  1 (own) -> 0, 2;  2 holds property 0 and ends.   The line 0, 1, 0 meets node 0 again.
fn cycle() -> Graph {
    Graph::new(vec![vec![1], vec![0, 2], vec![]], vec![true, true, false], vec![0, 0, 0b01])
}

#[test]
fn an_answer_that_rests_on_the_line_is_not_kept_so_a_later_question_at_another_line_is_not_poisoned_by_it() {
    // from 0 the asker goes to 1 and from 1 to 2 (two steps); the option back to 0 meets the line again and is no answer. With answers kept, the same question asked again, and one asked from node 1, give the true steps.
    let mut s = Steering::new();
    s.keeps = true;
    let mut w = Walk::<_, _, u8>::new(cycle(), s);
    let mut first = Out::<u8>::none();
    w.go(6, ask(0b01), false, false, &mut first).unwrap();
    assert_eq!(first.strict[0], 2);
    assert_eq!(w.taint & 0b01, 0b01, "the walk met a node of its own line, so some answer rested on the line");
    w.taint = 0;
    let mut again = Out::<u8>::none();
    w.go(6, ask(0b01), false, false, &mut again).unwrap();
    assert_eq!(again.strict[0], 2, "the same again from what was kept");
    // asked from node 1 (the line is now empty): one step, not what the first walk found for 1 under the line
    w.space.at = vec![1];
    let mut from_one = Out::<u8>::none();
    w.go(6, ask(0b01), false, false, &mut from_one).unwrap();
    assert_eq!(from_one.strict[0], 1);
}

#[test]
fn a_node_in_the_record_kept_before_the_walk_began_ends_a_line_with_no_taint() {
    let mut g = graph();
    g.at = vec![1];
    g.record = vec![1];
    let mut w = Walk::<_, _, u8>::new(g, Steering::new());
    let mut out = Out::<u8>::none();
    w.go(4, ask(0b01), true, false, &mut out).unwrap();
    assert_eq!((out.strict[0], w.taint), (255, 0));
}

// ------------------------------------------------------------------------------------------------------------------------------ an explicit tree as a node

type TS<'t> = TreeSpace<'t, i64, fn(&i64) -> bool>;

fn positive(r: &i64) -> bool { *r > 0 }
fn whole_score(r: &i64) -> Result<Num, cs_base::number::NumError> { Ok(Num::whole(*r)) }

#[test]
fn at_the_turn_of_a_member_of_the_set_the_node_is_the_askers_and_at_the_turn_of_another_it_is_not() {
    let t = Tree::turn(A, vec![leaf(1), leaf(2)]).unwrap();
    let (mine, theirs) = (party(&[A]), party(&[B]));
    let space = TreeSpace::forcing(&t, &mine, positive as fn(&i64) -> bool);
    assert!(Node::own_turn(&space));
    let other = TreeSpace::forcing(&t, &theirs, positive as fn(&i64) -> bool);
    assert!(!Node::own_turn(&other));
    assert!(!Node::chance(&space), "a node of a tree is not a choice by chance, whatever the tree says: chance is read as every outcome");
}

#[test]
fn the_options_of_a_node_are_numbered_in_the_trees_order_and_stepping_in_and_out_changes_the_node_and_the_key() {
    let t = Tree::turn(A, vec![leaf(1), Tree::turn(B, vec![leaf(2), leaf(3), leaf(4)]).unwrap()]).unwrap();
    let set = party(&[A]);
    let mut space = TreeSpace::forcing(&t, &set, positive as fn(&i64) -> bool);
    let mut out = vec![];
    Node::options(&mut space, &mut out);
    assert_eq!(out, vec![0, 1]);
    let root_key = Space::key(&space);
    Node::enter(&mut space, &1);
    assert_ne!(Space::key(&space), root_key, "another node, another key");
    assert!(!Node::own_turn(&space), "B's turn");
    let mut inner = vec![];
    Node::options(&mut space, &mut inner);
    assert_eq!(inner, vec![0, 1, 2]);
    Node::leave(&mut space);
    assert_eq!(Space::key(&space), root_key, "back at the root");
}

#[test]
fn a_result_says_whether_the_property_holds_of_it_and_that_it_ends_a_line_and_any_other_node_says_nothing() {
    let t = Tree::turn(A, vec![leaf(5), leaf(-5)]).unwrap();
    let set = party(&[A]);
    let mut space = TreeSpace::forcing(&t, &set, positive as fn(&i64) -> bool);
    assert_eq!(Space::look(&mut space), Look { bound: 0, held: 0, ended: false });
    Node::enter(&mut space, &0);
    assert_eq!(Space::look(&mut space), Look { bound: 1, held: 1, ended: true });
    Node::leave(&mut space);
    Node::enter(&mut space, &1);
    assert_eq!(Space::look(&mut space), Look { bound: 0, held: 0, ended: true });
}

#[test]
fn a_tree_is_never_met_before_and_breaks_no_property() {
    let t = leaf(1);
    let set = party(&[A]);
    let mut space = TreeSpace::forcing(&t, &set, positive as fn(&i64) -> bool);
    assert!(!Space::met_before(&space, 0) && !Space::met_before(&space, Space::key(&space)));
    assert_eq!(Space::breaks(&mut space, 0b1, &[0]), 0);
    assert!(!<TS<'static> as Space>::CYCLES);
}

#[test]
fn a_choice_at_once_is_read_as_the_parties_picks_in_turn_by_whether_each_party_is_the_sets_own() {
    let t = Tree::table(2, 3, (1..=6).map(leaf).collect()).unwrap();
    let count = |who: &[Chooser]| { let set = party(who); let mut s = TreeSpace::forcing(&t, &set, positive as fn(&i64) -> bool); let mut o = vec![]; Node::options(&mut s, &mut o); (o.len(), Node::own_turn(&s)) };
    assert_eq!(count(&[A, B]), (6, true), "both parties the set's own: some cell");
    assert_eq!(count(&[]), (6, false), "neither: every cell");
    assert_eq!(count(&[A]), (2, true), "the row party only: some row, then every column");
    assert_eq!(count(&[B]), (3, true), "the column party only: some column, then every row");
    // the row party's pick is the asker's, the column party's is the rest's
    let set = party(&[A]);
    let mut s = TreeSpace::forcing(&t, &set, positive as fn(&i64) -> bool);
    Node::enter(&mut s, &1);
    let mut cols = vec![];
    Node::options(&mut s, &mut cols);
    assert_eq!((cols.len(), Node::own_turn(&s)), (3, false));
    Node::enter(&mut s, &2);
    assert_eq!(Space::look(&mut s).held, 1, "the cell of row 1, column 2 holds the sixth result");
}

#[test]
fn forcing_leaves_out_a_choice_by_chance_of_no_chance_and_valuing_keeps_every_outcome() {
    let t = Tree::chance(vec![(leaf(1), Num::whole(1)), (leaf(2), Num::whole(0))]).unwrap();
    let set = party(&[A]);
    let mut forcing = TreeSpace::forcing(&t, &set, positive as fn(&i64) -> bool);
    let mut o = vec![];
    Node::options(&mut forcing, &mut o);
    assert_eq!(o, vec![0], "an outcome of no chance is not an outcome");
    let mut valuing = TreeSpace::valuing(&t, &set);
    let mut all = vec![];
    Node::options(&mut valuing, &mut all);
    assert_eq!(all, vec![0, 1]);
}

#[test]
fn the_value_of_a_tree_for_a_set_is_the_best_at_its_turn_the_worst_at_the_rest_s_and_the_expectation_at_chance() {
    let t = Tree::turn(A, vec![leaf(2), Tree::turn(B, vec![leaf(9), leaf(1)]).unwrap(), Tree::chance(vec![(leaf(4), Num::frac(1, 2).unwrap()), (leaf(8), Num::frac(1, 2).unwrap())]).unwrap()]).unwrap();
    let (mine, theirs) = (party(&[A]), party(&[B]));
    let mut score = whole_score;
    // A: the best of 2, what B leaves it (1) and the expectation 6 is 6
    assert_eq!(TreeSpace::valuing(&t, &mine).value(&mut score), Ok(Num::whole(6)));
    // B holds the set: the rest (A) pick the worst of 2, B's best (9) and 6: 2
    assert_eq!(TreeSpace::valuing(&t, &theirs).value(&mut score), Ok(Num::whole(2)));
    let mut under = TreeSpace::forcing(&t, &mine, positive as fn(&i64) -> bool);
    assert_eq!(under.value(&mut score), Ok(Num::whole(6)), "the same for a space made by forcing");
}

#[test]
fn the_value_of_a_result_is_its_score_and_an_overflow_in_a_score_is_said() {
    let t = leaf(7);
    let set = party(&[A]);
    let mut ok = whole_score;
    assert_eq!(TreeSpace::valuing(&t, &set).value(&mut ok), Ok(Num::whole(7)));
    let mut over = |_: &i64| Num::whole(i64::MAX).add(&Num::whole(1));
    assert!(TreeSpace::valuing(&t, &set).value(&mut over).is_err());
}

#[test]
fn a_walk_of_an_explicit_tree_asks_nothing_of_what_steers_it_and_keeps_no_answers() {
    type Sp = TreeSpace<'static, i64, fn(&i64) -> bool>;
    let mut free = Free;
    assert_eq!(Steer::<Sp>::spend(&mut free, Ask { strict: 1, loose: 0 }), Ok(()));
    assert!(!Steer::<Sp>::keeps(&free));
    let t: &'static Tree<i64> = Box::leak(Box::new(Tree::turn(A, vec![leaf(1), leaf(2)]).unwrap()));
    let set: &'static _ = Box::leak(Box::new(party(&[A])));
    let space: Sp = TreeSpace::forcing(t, set, positive as fn(&i64) -> bool);
    let mut opts = vec![1, 0];
    Steer::<Sp>::order(&mut free, &space, &mut opts, 0);
    assert_eq!(opts, vec![1, 0], "no order is put on the options");
}
