//! PLAN 79.33 PART D (SYSTEM_SPEC 11 "What plan 79.32 left of one algorithm", 13 item 51; piece: the tactical reasoner as cases of the base): the answers and counts of the walks that weigh numbers, written to a file behind an env variable, so that the answers before and after a walk became a case of layer 0's general walk
//! (`cs_base::choice::fold`) can be compared to the last number (scripts/whole_79_33_compare.py). Three walks: the search toward claims (`toward`, `toward_with`, `toward_memo`), the tail of a line (`settle_tail`), the search inside a frame (`frame::solve`).
//! Each test writes its file when its variable is set (WALK_TOWARD, WALK_TAIL, WALK_FRAME) and prints an OBS line with the number of questions; the time of the whole file is what scripts/whole_79_33_times.sh measures, one test thread.
use cs_base::maker::Maker;
use cs_game::gdl_game::GdlGame;
use cs_game::kernel::KernelGame;
use cs_game::{Fact, Game, Role};
use cs_operator::Budget;
use cs_rep::history::Seen;
use cs_tactics::ledger::Worth;
use cs_tactics::lines::settle_tail;
use cs_tactics::toward::{toward, toward_memo, toward_with, Toward, Verdict};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Instant;

const TTT: &str = "
(role x) (role o)
(init (cell 1 1 b)) (init (cell 1 2 b)) (init (cell 1 3 b))
(init (cell 2 1 b)) (init (cell 2 2 b)) (init (cell 2 3 b))
(init (cell 3 1 b)) (init (cell 3 2 b)) (init (cell 3 3 b))
(init (control x))
(<= (legal ?p (mark ?m ?n)) (true (cell ?m ?n b)) (true (control ?p)))
(<= (legal x noop) (true (control o)))
(<= (legal o noop) (true (control x)))
(<= (next (cell ?m ?n x)) (does x (mark ?m ?n)))
(<= (next (cell ?m ?n o)) (does o (mark ?m ?n)))
(<= (next (cell ?m ?n ?w)) (true (cell ?m ?n ?w)) (distinct ?w b))
(<= (next (cell ?m ?n b)) (true (cell ?m ?n b)) (does ?w (mark ?j ?k)) (distinct ?m ?j))
(<= (next (cell ?m ?n b)) (true (cell ?m ?n b)) (does ?w (mark ?j ?k)) (distinct ?n ?k))
(<= (next (control x)) (true (control o)))
(<= (next (control o)) (true (control x)))
(<= (row ?m ?p) (true (cell ?m 1 ?p)) (true (cell ?m 2 ?p)) (true (cell ?m 3 ?p)))
(<= (column ?n ?p) (true (cell 1 ?n ?p)) (true (cell 2 ?n ?p)) (true (cell 3 ?n ?p)))
(<= (diagonal ?p) (true (cell 1 1 ?p)) (true (cell 2 2 ?p)) (true (cell 3 3 ?p)))
(<= (diagonal ?p) (true (cell 1 3 ?p)) (true (cell 2 2 ?p)) (true (cell 3 1 ?p)))
(<= (line ?p) (row ?m ?p))
(<= (line ?p) (column ?n ?p))
(<= (line ?p) (diagonal ?p))
(<= open (true (cell ?m ?n b)))
(<= terminal (line x))
(<= terminal (line o))
(<= terminal (not open))
(<= (goal x 100) (line x))
(<= (goal x 50) (not (line x)) (not (line o)))
(<= (goal x 0) (line o))
(<= (goal o 100) (line o))
(<= (goal o 50) (not (line x)) (not (line o)))
(<= (goal o 0) (line x))
";

type S = <GdlGame as Game>::State;

fn holds(g: &GdlGame, s: &S, cell: (&str, &str), who: &str) -> bool { g.base_facts(s).contains(&Fact::new("cell", &[cell.0, cell.1, who])) }

fn show(r: &Toward) -> String {
    let reads: Vec<(u32, u64, u64, bool)> = r.reads.iter().map(|x| (x.depth, x.nodes, x.evals, x.complete)).collect();
    format!("depth {} root {:?} nodes {} evals {} cut {} exact {} silent {} reads {:?}", r.depth, r.root, r.nodes, r.evals, r.cut, r.exact, r.silent, reads)
}

/// The search toward claims on made positions of a small game, with four stand-in oracles, three entries, several depths and several budgets (a budget that cuts is a case).
#[test]
fn the_search_toward_claims_answers_and_counts_are_written() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let mut m = Maker::new(79_33);
    let mut text = String::new();
    let mut asked = 0usize;
    let mut spent = 0f64;
    let none = |_: &GdlGame, _: &S| None::<Verdict>;
    let centre = |g: &GdlGame, s: &S| if g.mover(s) == Some(0) && holds(g, s, ("2", "2"), "x") { Some(Verdict::Win) } else { None };
    let corner = |g: &GdlGame, s: &S| if holds(g, s, ("1", "1"), "o") && holds(g, s, ("3", "3"), "o") { Some(if g.mover(s) == Some(1) { Verdict::Win } else { Verdict::Loss }) } else if holds(g, s, ("2", "2"), "o") && g.mover(s) == Some(0) { Some(Verdict::Draw) } else { None };
    let mixed = |g: &GdlGame, s: &S| if holds(g, s, ("1", "2"), "x") && g.mover(s) == Some(0) { Some(Verdict::Loss) } else if holds(g, s, ("3", "2"), "o") { Some(Verdict::Draw) } else { None };
    for case in 0..150u64 {
        let mut s = g.initial();
        let plies = m.below(7);
        for _ in 0..plies {
            if g.outcome(&s).is_some() { break; }
            let ms = g.moves(&s);
            if ms.is_empty() { break; }
            let mv = ms[m.below(ms.len() as u64) as usize].clone();
            s = g.apply(&s, &mv);
        }
        if g.outcome(&s).is_some() { continue; }
        let role = g.mover(&s).unwrap_or(0);
        let depth = 1 + m.below(6) as u32;
        let fuse = [300u64, 4_000, 50_000_000][m.below(3) as usize];
        let seen = Seen::new();
        for (oi, which) in ["none", "centre", "corner", "mixed"].iter().enumerate() {
            let mut run = |entry: usize| -> Toward {
                let mut b = Budget::new(60.0, fuse);
                match (oi, entry) {
                    (0, 0) => toward(&g, &s, role, &none, &seen, depth, &mut b),
                    (0, 1) => toward_with(&g, &s, role, &none, &seen, depth, None, &mut b),
                    (0, _) => toward_memo(&g, &s, role, &none, &seen, depth, None, &mut b),
                    (1, 0) => toward(&g, &s, role, &centre, &seen, depth, &mut b),
                    (1, 1) => toward_with(&g, &s, role, &centre, &seen, depth, None, &mut b),
                    (1, _) => toward_memo(&g, &s, role, &centre, &seen, depth, Some(3), &mut b),
                    (2, 0) => toward(&g, &s, role, &corner, &seen, depth, &mut b),
                    (2, 1) => toward_with(&g, &s, role, &corner, &seen, depth, None, &mut b),
                    (2, _) => toward_memo(&g, &s, role, &corner, &seen, depth, None, &mut b),
                    (_, 0) => toward(&g, &s, role, &mixed, &seen, depth, &mut b),
                    (_, 1) => toward_with(&g, &s, role, &mixed, &seen, depth, None, &mut b),
                    (_, _) => toward_memo(&g, &s, role, &mixed, &seen, depth, None, &mut b),
                }
            };
            for entry in 0..3 {
                let t0 = Instant::now();
                let r = run(entry);
                spent += t0.elapsed().as_secs_f64();
                let _ = writeln!(text, "case {case} oracle {which} entry {entry} depth {depth} fuse {fuse}\t{}", show(&r));
                asked += 1;
            }
        }
    }
    if let Ok(path) = std::env::var("WALK_TOWARD") { std::fs::write(path, &text).unwrap(); }
    println!("OBS 79.33 walk toward: {asked} questions, the walks took {spent:.3} s");
    assert!(asked > 1000);
}

fn record() -> Worth {
    let table: HashMap<String, i64> = [("q", 4343), ("r", 1605), ("b", 1878), ("n", 1661), ("p", 1045)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "t".into(), table, ours_first: false }
}

/// The tail of a line on made positions of a game of captures (the end of the game counted as the search counts it).
#[test]
fn the_tail_of_a_line_answers_and_counts_are_written() {
    let starts = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "6k1/ppp3pp/4P3/2b5/8/5R2/P1q3PP/1q3R1K w - - 0 54",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    let j = record();
    let mut m = Maker::new(79_34);
    let mut text = String::new();
    let mut asked = 0usize;
    let mut spent = 0f64;
    for (si, fen) in starts.iter().enumerate() {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        for case in 0..60u64 {
            let mut s = g.initial();
            let plies = m.below(24);
            for _ in 0..plies {
                if g.outcome(&s).is_some() { break; }
                let ms = g.moves(&s);
                if ms.is_empty() { break; }
                let mv = ms[m.below(ms.len() as u64) as usize].clone();
                s = g.apply(&s, &mv);
            }
            if g.outcome(&s).is_some() { continue; }
            for me in [0 as Role, 1] {
                let t0 = Instant::now();
                let (tail, hit) = settle_tail(&g, &j, me, &s);
                spent += t0.elapsed().as_secs_f64();
                let names: Vec<String> = tail.iter().map(|mv| g.move_name(&s, mv)).collect();
                let _ = writeln!(text, "start {si} case {case} me {me}\thit {hit} tail {names:?}");
                asked += 1;
            }
        }
    }
    if let Ok(path) = std::env::var("WALK_TAIL") { std::fs::write(path, &text).unwrap(); }
    println!("OBS 79.33 walk tail: {asked} questions, the walks took {spent:.3} s");
    assert!(asked > 300);
}

/// A frame made of numbers: a state is a number and a count of moves made; its moves are numbers made from it, a move gains a small whole or half number, the side to move may always stop and the exit is a small number.
struct Implicit { branch: u64, depth: u32 }

impl cs_tactics::sheet::frame::Frame for Implicit {
    type State = (u64, u32);
    type Move = u64;
    type Key = (u64, u32);
    fn moves(&self, s: &(u64, u32)) -> Vec<u64> { if s.1 >= self.depth { vec![] } else { (0..1 + (s.0 >> 3) % self.branch).collect() } }
    fn gain(&self, s: &(u64, u32), m: &u64) -> f64 { (((s.0 ^ (m + 1).wrapping_mul(0x9E37_79B9)) >> 4) % 9) as f64 * 0.5 }
    fn next(&self, s: &(u64, u32), m: &u64) -> (u64, u32) { (s.0.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(17) ^ ((m + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 2), s.1 + 1) }
    fn exit(&self, s: &(u64, u32)) -> f64 { ((s.0 >> 7) % 3) as f64 * 0.5 }
    fn key(&self, s: &(u64, u32)) -> (u64, u32) { (s.0 % 400_001, s.1) }
    fn label(&self, _s: &(u64, u32), m: &u64) -> String { format!("m{m}") }
}

/// The search inside a frame, on frames made of numbers (several branchings and depths, with and without a budget that cuts) and on the swap of strikers on a place.
#[test]
fn the_search_inside_a_frame_answers_and_counts_are_written() {
    use cs_tactics::sheet::frame::{solve, Static, StaticState, Striker};
    use cs_tactics::sheet::{Names, Side, Unit};
    let mut m = Maker::new(79_35);
    let mut text = String::new();
    let mut asked = 0usize;
    let mut spent = 0f64;
    for case in 0..20000u64 {
        let f = Implicit { branch: 3 + m.below(3), depth: 6 + m.below(5) as u32 };
        let start = (m.next_u64() >> 3, 0u32);
        for budget in [10u64, 2000, 5_000_000] {
            let t0 = Instant::now();
            let r = solve(&f, &start, budget, "made");
            spent += t0.elapsed().as_secs_f64();
            let _ = writeln!(text, "made {case} branch {} depth {} budget {budget}\tvalue {} line {:?} open {} nodes {}", f.branch, f.depth, r.value, r.line, r.open, r.nodes);
            asked += 1;
        }
    }
    let names = Names::new(vec!["a".into(), "b".into(), "c".into()], vec!["k".into()], vec!["x".into(), "y".into()]);
    let frame = Static { names: &names };
    for case in 0..20000u64 {
        let mk = |m: &mut Maker, n: u64| -> Vec<Striker> { (0..n).map(|_| Striker { worth: m.range(1, 12) as f64 * 0.5, only_last: m.below(5) == 0, unit: Unit { side: Side::Us, kind: 0, at: 0 } }).collect() };
        let (n1, n2) = (m.below(5), m.below(5));
        let mut mine = mk(&mut m, n1);
        mine.sort_by(|a, b| a.worth.partial_cmp(&b.worth).unwrap());
        let mut theirs = mk(&mut m, n2);
        theirs.sort_by(|a, b| a.worth.partial_cmp(&b.worth).unwrap());
        let state: StaticState = (mine, theirs, m.range(1, 12) as f64 * 0.5);
        for budget in [1u64, 3, 1_000_000] {
            let t0 = Instant::now();
            let r = solve(&frame, &state, budget, "static");
            spent += t0.elapsed().as_secs_f64();
            let _ = writeln!(text, "static {case} budget {budget}\tvalue {} line {:?} open {} nodes {}", r.value, r.line, r.open, r.nodes);
            asked += 1;
        }
    }
    if let Ok(path) = std::env::var("WALK_FRAME") { std::fs::write(path, &text).unwrap(); }
    println!("OBS 79.33 walk frame: {asked} questions, the walks took {spent:.3} s");
    assert!(asked > 1000);
}
