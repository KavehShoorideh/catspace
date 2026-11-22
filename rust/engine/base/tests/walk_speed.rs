//! PLAN 79.32 PART F (LAYER 0'S TEST OF SPEED; SYSTEM_SPEC 11 "The pure search is mathematics and lives in layer 0", 13 item 50). A made implicit tree of a fixed size is walked by `choice::walk::Walk`: no tree is built, a node is a number and its options are computed when asked, stepped into and back out of in place.
//! It prints OBS lines (nodes read, seconds, nodes a second, whole numbers of two sizes as the depth and the steps) and asserts the answer against a plain recursion written apart in this file over the same implicit tree. The speed is printed, never asserted: a number of nodes a second is a measure, not a law.
use cs_base::choice::walk::{Ask, Cut, Look, Out, Space, Steer, Walk, Whole};
use std::time::Instant;

const BRANCH: u64 = 4;
const DEPTH: u32 = 21;

/// The implicit tree: a node is a 64-bit number; its options are `BRANCH` numbers made from it; the asker picks at even depths. Property 0 holds at a node whose number has the low five bits clear; property 1 at a node whose number is a multiple of 7 (`sparse`: the low twenty bits clear, a multiple of 1000003: so rare that every own choice reads all its options). Nothing is stored.
struct Implicit { stack: Vec<(u64, u32)>, sparse: bool }

fn child(n: u64, i: u64) -> u64 { (n ^ (i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(23) ^ n }
fn props(n: u64, sparse: bool) -> u64 { if sparse { (n & 0xFFFFF == 0) as u64 | (((n % 1_000_003 == 0) as u64) << 1) } else { (n & 31 == 0) as u64 | (((n % 7 == 0) as u64) << 1) } }

impl Space for Implicit {
    type Opt = u64;
    const CYCLES: bool = false;
    fn key(&self) -> u64 { self.stack.last().unwrap().0 }
    fn look(&mut self) -> Look { let (n, d) = *self.stack.last().unwrap(); let p = props(n, self.sparse); Look { bound: p, held: p, ended: d == DEPTH } }
    fn own_turn(&self) -> bool { self.stack.last().unwrap().1 % 2 == 0 }
    fn met_before(&self, _key: u64) -> bool { false }
    fn options(&mut self, out: &mut Vec<u64>) { let n = self.stack.last().unwrap().0; out.extend((0..BRANCH).map(|i| child(n, i))); }
    fn breaks(&mut self, _cand: u64, _opts: &[u64]) -> u64 { 0 }
    fn enter(&mut self, o: &u64) { let d = self.stack.last().unwrap().1; self.stack.push((*o, d + 1)); }
    fn leave(&mut self) { self.stack.pop(); }
}

struct Count { nodes: u64 }
impl Steer<Implicit> for Count {
    fn spend(&mut self, _a: Ask) -> Result<(), Cut> { self.nodes += 1; Ok(()) }
    fn order(&mut self, _s: &Implicit, _o: &mut Vec<u64>, _first: usize) {}
    fn keeps(&self) -> bool { false }
}

/// The reference: the same reading by plain recursion over the same implicit tree (steps to meet each of the two properties, or None), written here with no part of the walk.
fn reference(n: u64, d: u32, left: u32, want: u64, sparse: bool) -> [Option<u32>; 2] {
    let p = props(n, sparse);
    let mut res = [None, None];
    for i in 0..2 { if want >> i & 1 == 1 && p >> i & 1 == 1 { res[i] = Some(0); } }
    let todo: Vec<usize> = (0..2).filter(|i| want >> i & 1 == 1 && res[*i].is_none()).collect();
    if todo.is_empty() || d == DEPTH || left == 0 { return res; }
    let own = d % 2 == 0;
    for &i in &todo {
        let kids: Vec<Option<u32>> = (0..BRANCH).map(|k| reference(child(n, k), d + 1, left - 1, 1 << i, sparse)[i]).collect();
        res[i] = if own { kids.iter().flatten().min().map(|x| x + 1) } else if kids.iter().all(|k| k.is_some()) { kids.iter().flatten().max().map(|x| x + 1) } else { None };
    }
    res
}

fn walk_with<N: Whole>(depth: N, root: u64, sparse: bool) -> (Out<N>, u64, f64) {
    let mut w = Walk::<_, _, N>::new(Implicit { stack: vec![(root, 0)], sparse }, Count { nodes: 0 });
    let mut out = Out::<N>::none();
    let t0 = Instant::now();
    w.go(depth, Ask { strict: 0b11, loose: 0 }, false, false, &mut out).unwrap();
    (out, w.steer.nodes, t0.elapsed().as_secs_f64())
}

/// The answers are the reference's wherever the walk's first-found steps equal the fewest (an own node takes the first option that meets it, so the walk's steps are an upper bound of the fewest: whether a property is met agrees exactly, and the steps are never fewer).
#[test]
fn the_walk_of_an_implicit_tree_answers_as_the_plain_recursion_and_its_speed_is_printed() {
    let mut met = [0usize; 2];
    let mut roots = 0;
    for r in 0..40u64 {
        let root = child(r, 0);
        let (out, _, _) = walk_with::<u8>(6, root, false);
        let want = reference(root, 0, 6, 0b11, false);
        for i in 0..2 {
            let got = (out.strict[i] != u8::NONE).then(|| out.strict[i] as u32);
            assert_eq!(got.is_some(), want[i].is_some(), "root {r} property {i}");
            if let (Some(g), Some(w)) = (got, want[i]) { assert!(g >= w, "steps are an upper bound of the fewest"); met[i] += 1; }
        }
        roots += 1;
    }
    assert!(met[0] > 0 && met[1] > 0 && met[0] < roots * 2, "the roots are of both kinds ({met:?} of {roots})");
    // the speed, on one fixed size (BRANCH ^ DEPTH leaves at most), the numbers as u8 and as u64
    let (mut nodes8, mut secs8) = (0u64, 0f64);
    let (mut nodes64, mut secs64) = (0u64, 0f64);
    for r in 0..6u64 {
        let root = child(1000 + r, 1);
        let (_, n, s) = walk_with::<u8>(DEPTH as u8, root, true);
        nodes8 += n; secs8 += s;
        let (_, n, s) = walk_with::<u64>(DEPTH as u64, root, true);
        nodes64 += n; secs64 += s;
    }
    println!("OBS 79.32 cs_base walk, an implicit tree of branching {BRANCH} to depth {DEPTH}, six roots, the depth as u8: {nodes8} nodes in {secs8:.3} s, {:.0} nodes a second", nodes8 as f64 / secs8.max(1e-9));
    println!("OBS 79.32 cs_base walk, the same with the depth as u64: {nodes64} nodes in {secs64:.3} s, {:.0} nodes a second", nodes64 as f64 / secs64.max(1e-9));
    assert_eq!(nodes8, nodes64, "the same nodes whatever the whole number");
}
