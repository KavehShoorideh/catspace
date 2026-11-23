//! PLAN 79.36 PART C, LAYER 0, THE WALK OF VALUES: what a node gives, what a kind hands in, the order of the hooks, the window that cuts, the expectation at a choice by chance (SYSTEM_SPEC 11, last block; the base, layer 0; ledger row 43).
//! The nodes are nodes of small trees of numbers written here; the expected values and the counts of nodes read are worked out by hand in the comments.
use cs_base::choice::fold::{Bounds, Fold, Kind, Node, Pre};

/// A tree by node number: kind 0 own turn, 1 the rest's turn, 2 chance, 3 a leaf of the value `value[n]`.
struct Tree { adj: Vec<Vec<usize>>, kind: Vec<u8>, value: Vec<f64>, at: Vec<usize> }

impl Node for Tree {
    type Opt = usize;
    fn own_turn(&self) -> bool { self.kind[*self.at.last().unwrap()] == 0 }
    fn chance(&self) -> bool { self.kind[*self.at.last().unwrap()] == 2 }
    fn options(&mut self, out: &mut Vec<usize>) { out.extend(self.adj[*self.at.last().unwrap()].iter().copied()); }
    fn enter(&mut self, o: &usize) { self.at.push(*o); }
    fn leave(&mut self) { self.at.pop(); }
}

/// the worked tree: 0 own -> 1, 2;  1 rest -> 3, 4;  2 rest -> 5, 6;  leaves 3: 3, 4: 5, 5: 2, 6: 9.   Played to the end: node 1 is the least of 3 and 5, 3; node 2 the least of 2 and 9, 2; the root the greatest, 3.
fn worked() -> Tree {
    Tree { adj: vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![], vec![], vec![], vec![]], kind: vec![0, 1, 1, 3, 3, 3, 3], value: vec![0.0, 0.0, 0.0, 3.0, 5.0, 2.0, 9.0], at: vec![0] }
}

/// what the kind records of the walk
#[derive(Default)]
struct Trace { before: Vec<usize>, horizon: Vec<usize>, improved: Vec<(u32, usize)>, opened: u32, closed: u32, finished: u32 }

struct Minimax { reverse: bool, stand: Option<f64>, trace: Trace }

fn minimax(_windowed: bool) -> Minimax { Minimax { reverse: false, stand: None, trace: Trace::default() } }

impl Kind<Tree> for Minimax {
    type Val = f64;
    type Err = String;
    type Edge = ();
    type Token = usize;
    const WINDOWED: bool = true;
    const CHANCE: bool = false;
    fn before(&mut self, space: &Tree, _left: u32, _ply: u32, _win: &mut Bounds<f64>) -> Result<Pre<f64, usize>, String> {
        let n = *space.at.last().unwrap();
        self.trace.before.push(n);
        if space.kind[n] == 3 { Ok(Pre::Done(space.value[n])) } else { Ok(Pre::Go(n)) }
    }
    fn horizon(&mut self, _space: &Tree, token: usize, _ply: u32) -> Result<f64, String> { self.trace.horizon.push(token); Ok(0.0) }
    fn base(&mut self, space: &Tree, _token: &usize) -> Option<f64> { if space.at.len() == 1 { self.stand } else { None } }
    fn order(&mut self, _space: &Tree, opts: &mut Vec<usize>) { if self.reverse { opts.reverse(); } }
    fn edge(&mut self, _space: &Tree, _o: &usize) -> Option<()> { Some(()) }
    fn improved(&mut self, _space: &Tree, ply: u32, o: &usize) { self.trace.improved.push((ply, *o)); }
    fn open(&mut self, _token: &usize) { self.trace.opened += 1; }
    fn close(&mut self) { self.trace.closed += 1; }
    fn finish(&mut self, _space: &Tree, _token: usize, got: Option<f64>, _entry: &Bounds<f64>) -> Result<f64, String> { self.trace.finished += 1; Ok(got.unwrap_or(f64::NAN)) }
}

fn everything() -> Bounds<f64> { Bounds { lo: f64::NEG_INFINITY, hi: f64::INFINITY } }

#[test]
fn the_value_is_the_best_at_the_askers_node_and_the_worst_at_the_others_and_every_option_is_read_when_the_window_is_wide_and_nothing_is_cut() {
    // the window is the whole line, but a window cuts as soon as what the asker has meets what the rest hold it to: node 2 is cut after its first leaf (2 is already below the 3 the root has)
    let mut f = Fold::new(worked(), minimax(true));
    assert_eq!(f.go(9, 0, everything()), Ok(3.0));
    assert_eq!(f.kind.trace.before, vec![0, 1, 3, 4, 2, 5], "the leaf of 9 is never read: node 2 is worth at most 2, below what the root already holds");
    assert_eq!((f.kind.trace.opened, f.kind.trace.closed, f.kind.trace.finished), (3, 3, 3), "three nodes were opened and left, and each answered");
}

#[test]
fn the_hook_that_says_a_new_option_is_best_is_called_for_each_option_strictly_better_than_the_rest_in_the_order_the_walk_leaves_them() {
    let mut f = Fold::new(worked(), minimax(true));
    f.go(9, 0, everything()).unwrap();
    // in node 1 (the rest pick the least) leaf 3 is the first and so the best; leaf 4 (5) is not better; the root then holds node 1; in node 2 leaf 5 (2) is the first
    assert_eq!(f.kind.trace.improved, vec![(1, 3), (0, 1), (1, 5)]);
}

#[test]
fn the_order_the_options_are_tried_in_changes_what_is_read_and_never_the_value() {
    let mut k = minimax(true);
    k.reverse = true;
    let mut f = Fold::new(worked(), k);
    assert_eq!(f.go(9, 0, everything()), Ok(3.0));
    // node 2 first: its leaves in the order 9, then 2; then node 1: leaf 5, then leaf 3: the root holds 2 and node 1 is read in full
    assert_eq!(f.kind.trace.before, vec![0, 2, 6, 5, 1, 4, 3]);
}

#[test]
fn a_narrow_window_around_the_true_value_gives_it_and_one_that_excludes_it_gives_a_bound_on_the_right_side() {
    let mut inside = Fold::new(worked(), minimax(true));
    assert_eq!(inside.go(9, 0, Bounds { lo: 0.0, hi: 10.0 }), Ok(3.0));
    let mut above = Fold::new(worked(), minimax(true));
    let v = above.go(9, 0, Bounds { lo: 4.0, hi: 10.0 }).unwrap();
    assert!(v <= 4.0, "the true value is 3, below the window: a bound at or below its floor, got {v}");
    let mut below = Fold::new(worked(), minimax(true));
    let w = below.go(9, 0, Bounds { lo: 0.0, hi: 2.0 }).unwrap();
    assert!(w >= 2.0, "the true value is 3, above the window's roof: a bound at or above it, got {w}");
}

#[test]
fn with_no_step_left_the_value_is_the_one_the_kind_gives_for_a_node_it_cannot_open() {
    let mut f = Fold::new(worked(), minimax(true));
    assert_eq!(f.go(0, 0, everything()), Ok(0.0), "the root is not a leaf: its horizon value");
    assert_eq!(f.kind.trace.horizon, vec![0]);
    let mut one = Fold::new(worked(), minimax(true));
    assert_eq!(one.go(1, 0, everything()), Ok(0.0), "one step: the options are at the horizon, worth nought each");
    assert_eq!(one.kind.trace.horizon, vec![1, 2]);
}

#[test]
fn what_a_kind_hands_in_is_asked_one_node_at_a_time_before_horizon_edge_and_finish() {
    let mut k = minimax(true);
    let mut space = worked();
    let mut win = everything();
    // at the root, which is no leaf, the kind gives the token of the node to open; at a leaf, the value
    match Kind::<Tree>::before(&mut k, &space, 9, 0, &mut win) { Ok(Pre::Go(t)) => assert_eq!(t, 0), _ => panic!("the root is to be opened") }
    space.at.push(4);
    match Kind::<Tree>::before(&mut k, &space, 9, 1, &mut win) { Ok(Pre::Done(v)) => assert_eq!(v, 5.0), _ => panic!("a leaf is worth its value") }
    assert_eq!(k.trace.before, vec![0, 4]);
    assert_eq!(Kind::<Tree>::horizon(&mut k, &space, 7, 2), Ok(0.0));
    assert_eq!(k.trace.horizon, vec![7]);
    assert_eq!(Kind::<Tree>::edge(&mut k, &space, &3), Some(()));
    assert_eq!(Kind::<Tree>::finish(&mut k, &space, 0, Some(2.5), &win), Ok(2.5));
    assert!(Kind::<Tree>::finish(&mut k, &space, 0, None, &win).unwrap().is_nan(), "no option: the kind says what that is worth");
    assert_eq!(k.trace.finished, 2);
}

#[test]
fn standing_where_one_is_is_an_option_of_the_asker_and_the_best_so_far() {
    let mut k = minimax(true);
    k.stand = Some(10.0);
    let mut f = Fold::new(worked(), k);
    assert_eq!(f.go(9, 0, everything()), Ok(10.0), "the asker may stop at the root, where it is worth 10");
    let mut low = minimax(true);
    low.stand = Some(1.0);
    assert_eq!(Fold::new(worked(), low).go(9, 0, everything()), Ok(3.0), "standing worth 1 is not better than playing on");
}

// ------------------------------------------------------------------------------------------------------------------------------ the hooks that have a provided answer

#[test]
fn what_a_kind_provides_unless_it_says_otherwise_is_no_stand_the_order_given_the_same_window_and_value_and_nothing_at_improved_open_close() {
    struct Bare;
    impl Kind<Tree> for Bare {
        type Val = f64;
        type Err = ();
        type Edge = f64;
        type Token = ();
        const WINDOWED: bool = true;
        const CHANCE: bool = false;
        fn before(&mut self, _s: &Tree, _l: u32, _p: u32, _w: &mut Bounds<f64>) -> Result<Pre<f64, ()>, ()> { Ok(Pre::Go(())) }
        fn horizon(&mut self, _s: &Tree, _t: (), _p: u32) -> Result<f64, ()> { Ok(0.0) }
        fn edge(&mut self, _s: &Tree, _o: &usize) -> Option<f64> { Some(1.0) }
        fn finish(&mut self, _s: &Tree, _t: (), got: Option<f64>, _e: &Bounds<f64>) -> Result<f64, ()> { Ok(got.unwrap_or(0.0)) }
    }
    let space = worked();
    let mut k = Bare;
    assert_eq!(Kind::<Tree>::base(&mut k, &space, &()), None);
    let mut opts = vec![2, 1, 0];
    Kind::<Tree>::order(&mut k, &space, &mut opts);
    assert_eq!(opts, vec![2, 1, 0], "no order is put on the options");
    let win = Bounds { lo: 1.0, hi: 5.0 };
    assert_eq!(Kind::<Tree>::child(&k, 7.0, &win), win, "the option's node is read in the same window");
    assert_eq!(Kind::<Tree>::lift(&k, 7.0, 3.5), Ok(3.5), "the value of an option is the value of its node");
    Kind::<Tree>::improved(&mut k, &space, 0, &1);
    Kind::<Tree>::open(&mut k, &());
    Kind::<Tree>::close(&mut k);
}

#[test]
#[should_panic(expected = "a kind without chance is never asked to mix")]
fn a_kind_that_does_not_weigh_chance_is_never_asked_to_mix_and_is_stopped_if_it_is() {
    struct Bare;
    impl Kind<Tree> for Bare {
        type Val = f64;
        type Err = ();
        type Edge = f64;
        type Token = ();
        const WINDOWED: bool = false;
        const CHANCE: bool = false;
        fn before(&mut self, _s: &Tree, _l: u32, _p: u32, _w: &mut Bounds<f64>) -> Result<Pre<f64, ()>, ()> { Ok(Pre::Go(())) }
        fn horizon(&mut self, _s: &Tree, _t: (), _p: u32) -> Result<f64, ()> { Ok(0.0) }
        fn edge(&mut self, _s: &Tree, _o: &usize) -> Option<f64> { Some(1.0) }
        fn finish(&mut self, _s: &Tree, _t: (), got: Option<f64>, _e: &Bounds<f64>) -> Result<f64, ()> { Ok(got.unwrap_or(0.0)) }
    }
    let _ = Kind::<Tree>::mix(&mut Bare, None, 0.5, 1.0);
}

#[test]
fn a_node_says_whether_it_is_a_choice_by_chance_and_what_it_gives_when_asked_and_a_node_not_overriding_it_is_none() {
    let mut t = Tree { adj: vec![vec![1, 2], vec![], vec![]], kind: vec![2, 3, 3], value: vec![0.0, 4.0, 8.0], at: vec![0] };
    assert!(Node::chance(&t));
    assert!(!Node::own_turn(&t));
    let mut o = vec![];
    Node::options(&mut t, &mut o);
    assert_eq!(o, vec![1, 2]);
    Node::enter(&mut t, &2);
    assert!(!Node::chance(&t));
    Node::leave(&mut t);
    assert!(Node::chance(&t));
    // a node that does not override `chance` is not a choice by chance
    struct Plain;
    impl Node for Plain { type Opt = u8; fn own_turn(&self) -> bool { true } fn options(&mut self, _out: &mut Vec<u8>) {} fn enter(&mut self, _o: &u8) {} fn leave(&mut self) {} }
    assert!(!Node::chance(&Plain));
}

// ------------------------------------------------------------------------------------------------------------------------------ a choice by chance

struct Expect { trace: Trace }

impl Kind<Tree> for Expect {
    type Val = f64;
    type Err = String;
    type Edge = f64;
    type Token = usize;
    const WINDOWED: bool = false;
    const CHANCE: bool = true;
    fn before(&mut self, space: &Tree, _l: u32, _p: u32, _w: &mut Bounds<f64>) -> Result<Pre<f64, usize>, String> {
        let n = *space.at.last().unwrap();
        self.trace.before.push(n);
        if space.kind[n] == 3 { Ok(Pre::Done(space.value[n])) } else { Ok(Pre::Go(n)) }
    }
    fn horizon(&mut self, _s: &Tree, _t: usize, _p: u32) -> Result<f64, String> { Ok(0.0) }
    fn edge(&mut self, _s: &Tree, o: &usize) -> Option<f64> { if *o == 99 { None } else { Some(if *o % 2 == 1 { 0.25 } else { 0.75 }) } }
    fn mix(&mut self, acc: Option<f64>, e: f64, v: f64) -> Result<f64, String> { Ok(acc.unwrap_or(0.0) + e * v) }
    fn finish(&mut self, _s: &Tree, _t: usize, got: Option<f64>, _e: &Bounds<f64>) -> Result<f64, String> { Ok(got.ok_or("no option")?) }
}

#[test]
fn at_a_choice_by_chance_the_value_is_the_weighted_sum_of_its_options_and_the_asker_picks_between_that_and_a_sure_one() {
    // the root (own) picks between node 1, a chance of 0.25 for the leaf 3 (value 8) and 0.75 for the leaf 4 (value 4): 0.25 * 8 + 0.75 * 4 = 5, and node 2, a leaf worth 4
    let t = Tree { adj: vec![vec![1, 2], vec![3, 4], vec![]], kind: vec![0, 2, 3, 3, 3], value: vec![0.0, 0.0, 4.0, 8.0, 4.0], at: vec![0] };
    // 1's options are 3 and 4: weights by option number: 3 is odd (0.25), 4 is even (0.75)
    let mut f = Fold::new(t, Expect { trace: Trace::default() });
    assert_eq!(f.go(9, 0, everything()), Ok(5.0));
    assert_eq!(f.kind.trace.before, vec![0, 1, 3, 4, 2]);
}

#[test]
fn an_option_the_kind_leaves_out_by_giving_it_no_edge_is_not_read_and_a_choice_with_no_option_left_is_the_kinds_to_answer() {
    let t = Tree { adj: vec![vec![99], vec![]], kind: vec![0, 3], value: vec![0.0, 1.0], at: vec![0] };
    let mut f = Fold::new(t, Expect { trace: Trace::default() });
    assert_eq!(f.go(9, 0, everything()), Err("no option".to_string()));
    assert_eq!(f.kind.trace.before, vec![0], "the option left out was never entered");
}
