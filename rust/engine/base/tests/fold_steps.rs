//! PLAN 79.33 PART C (THE STEPS WALK AS A CASE OF THE GENERAL WALK: THE ONE ATTEMPT; SYSTEM_SPEC 11, 13 item 51). The walk of `walk::Walk` answers, for up to 64 properties in two readings, within how many steps each is met, asking at a node only the properties still open and keeping answers by key.
//! The general walk (`fold::Fold`) answers one number. The attempt: the same implicit tree as `walk_speed.rs`, each property asked as a number by the general walk (1 where it is met within the depth, 0 where it is not: best at the asker's turn, worst at any other), nothing kept, the two properties as two walks.
//! The answers (met or not) are held against `Walk`'s; the nodes and the seconds of both are printed. The steps are not asked: the general walk's best is the greater of the options' values, the steps walk's own answer is the steps of the FIRST option that meets (an upper bound of the fewest), which is not a number's best, so the counts and the steps of the one cannot equal the other's.
use cs_base::choice::fold::{Bounds, Fold, Kind, Node, Pre};
use cs_base::choice::walk::{Ask, Cut, Look, Out, Space, Steer, Walk, Whole};
use std::time::Instant;

const BRANCH: u64 = 4;
const DEPTH: u32 = 21;

fn child(n: u64, i: u64) -> u64 { (n ^ (i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(23) ^ n }
fn props(n: u64, dense: bool) -> u64 { if dense { (n & 31 == 0) as u64 | (((n % 7 == 0) as u64) << 1) } else { (n & 0xFFFFF == 0) as u64 | (((n % 1_000_003 == 0) as u64) << 1) } }

struct Implicit { stack: Vec<(u64, u32)>, dense: bool }

impl Node for Implicit {
    type Opt = u64;
    fn own_turn(&self) -> bool { self.stack.last().unwrap().1 % 2 == 0 }
    fn options(&mut self, out: &mut Vec<u64>) { let n = self.stack.last().unwrap().0; out.extend((0..BRANCH).map(|i| child(n, i))); }
    fn enter(&mut self, o: &u64) { let d = self.stack.last().unwrap().1; self.stack.push((*o, d + 1)); }
    fn leave(&mut self) { self.stack.pop(); }
}

impl Space for Implicit {
    const CYCLES: bool = false;
    fn key(&self) -> u64 { self.stack.last().unwrap().0 }
    fn look(&mut self) -> Look { let (n, d) = *self.stack.last().unwrap(); let p = props(n, self.dense); Look { bound: p, held: p, ended: d == DEPTH } }
    fn met_before(&self, _key: u64) -> bool { false }
    fn breaks(&mut self, _cand: u64, _opts: &[u64]) -> u64 { 0 }
}

struct Count { nodes: u64 }
impl Steer<Implicit> for Count {
    fn spend(&mut self, _a: Ask) -> Result<(), Cut> { self.nodes += 1; Ok(()) }
    fn order(&mut self, _s: &Implicit, _o: &mut Vec<u64>, _first: usize) {}
    fn keeps(&self) -> bool { false }
}

/// One property as a number: 1 where it is met within the depth, else 0.
struct Met { bit: u64, nodes: u64 }

impl Kind<Implicit> for Met {
    type Val = u8;
    type Err = ();
    type Edge = ();
    type Token = ();
    const WINDOWED: bool = true;
    const CHANCE: bool = false;
    fn before(&mut self, space: &Implicit, _left: u32, _ply: u32, _win: &mut Bounds<u8>) -> Result<Pre<u8, ()>, ()> {
        self.nodes += 1;
        let (n, d) = *space.stack.last().unwrap();
        if props(n, space.dense) & self.bit != 0 { return Ok(Pre::Done(1)); }
        if d == DEPTH { return Ok(Pre::Done(0)); }
        Ok(Pre::Go(()))
    }
    fn horizon(&mut self, _space: &Implicit, _token: (), _ply: u32) -> Result<u8, ()> { Ok(0) }
    fn edge(&mut self, _space: &Implicit, _o: &u64) -> Option<()> { Some(()) }
    fn finish(&mut self, _space: &Implicit, _token: (), got: Option<u8>, _entry: &Bounds<u8>) -> Result<u8, ()> { Ok(got.unwrap_or(0)) }
}

#[test]
fn the_steps_walk_as_a_case_of_the_general_walk_is_tried_and_its_cost_is_printed() {
    answers_and_cost(false, DEPTH, "sparse, depth 21");
    answers_and_cost(true, 6, "dense, depth 6");
}

fn answers_and_cost(dense: bool, depth: u32, what: &str) {
    let (mut walk_nodes, mut walk_secs, mut fold_nodes, mut fold_secs) = (0u64, 0f64, 0u64, 0f64);
    let mut met = [0usize; 2];
    for r in 0..6u64 {
        let root = child(1000 + r, 1);
        let mut w = Walk::<_, _, u8>::new(Implicit { stack: vec![(root, 0)], dense }, Count { nodes: 0 });
        let mut out = Out::<u8>::none();
        let t0 = Instant::now();
        w.go(depth as u8, Ask { strict: 0b11, loose: 0 }, false, false, &mut out).unwrap();
        walk_secs += t0.elapsed().as_secs_f64();
        walk_nodes += w.steer.nodes;
        for i in 0..2 {
            let mut f = Fold::new(Implicit { stack: vec![(root, 0)], dense }, Met { bit: 1 << i, nodes: 0 });
            let t0 = Instant::now();
            let v = f.go(depth, 0, Bounds { lo: 0, hi: 1 }).unwrap();
            fold_secs += t0.elapsed().as_secs_f64();
            fold_nodes += f.kind.nodes;
            assert_eq!(v == 1, out.strict[i] != <u8 as Whole>::NONE, "root {r} property {i}: met or not agrees");
            met[i] += (v == 1) as usize;
        }
    }
    println!("OBS 79.33 steps as a case ({what}): the steps walk, two properties in one walk: {walk_nodes} nodes in {walk_secs:.3} s; the general walk, one property a walk, two walks: {fold_nodes} nodes in {fold_secs:.3} s (met {met:?} of 6 roots)");
}
