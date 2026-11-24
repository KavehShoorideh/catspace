//! PLAN 79.41 PART E (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ..."; ledger row 45; piece: rules): THE TIMES OF THE PASSES OF 79.41 (this file began as the clock of 79.40, translated_speed_bits.rs, and keeps its name and its list of positions). The same two questions at the same list of
//! positions for each game: the kernel, the plain translation, the module of every pass of 79.40 (`prev`), bits alone, each new pass alone on top of bits, and every pass together (`fast`). The modules of the single passes are not in the repository:
//! tests/test_plan79_41.py makes them under generated/scratch (scripts/gdl_to_rust_make.py --scratch) and this file is compiled with `--cfg singles` (see that test); without the flag only the plain module and `fast` are compared. Ignored by default
//! (`-- --ignored --nocapture --test-threads=1`); a short run is always made.
#![allow(unexpected_cfgs)]
mod variants_common;
#[macro_use]
mod translated_common;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::rng::Rng;
use cs_game::Game;
use std::time::Instant;
use translated_common::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

/// Counts the allocations of the process (this binary only), so that a question's making of memory is a number: the clock tests print allocations a position.
struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { ALLOCS.fetch_add(1, Ordering::Relaxed); BYTES.fetch_add(l.size() as u64, Ordering::Relaxed); System.alloc(l) }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l) }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 { ALLOCS.fetch_add(1, Ordering::Relaxed); BYTES.fetch_add(n as u64, Ordering::Relaxed); System.realloc(p, l, n) }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

include!("generated/chess.rs");
include!("generated/chess_fast.rs");
include!("generated/atomic.rs");
include!("generated/atomic_fast.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits_scan.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits_cut.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits_shift.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits_base.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev41_orient.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev41_ends.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev41_start.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev_noworth.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev_worth.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev_42.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev_project.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev_asked.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits_scan.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits_cut.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits_shift.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits_base.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev41_orient.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev41_ends.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev41_start.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev_noworth.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev_worth.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev_42.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev_project.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev_asked.rs");
impl_tr!(chess);
impl_tr!(chess_fast, passes);
impl_tr!(atomic);
impl_tr!(atomic_fast, passes);
#[cfg(singles)]
impl_tr!(chess_prev, passes);
#[cfg(singles)]
impl_tr!(chess_bits, passes);
#[cfg(singles)]
impl_tr!(chess_bits_scan, passes);
#[cfg(singles)]
impl_tr!(chess_bits_cut, passes);
#[cfg(singles)]
impl_tr!(chess_bits_shift, passes);
#[cfg(singles)]
impl_tr!(chess_bits_base, passes);
#[cfg(singles)]
impl_tr!(chess_prev41_orient, passes);
#[cfg(singles)]
impl_tr!(chess_prev41_ends, passes);
#[cfg(singles)]
impl_tr!(chess_prev41_start, passes);
#[cfg(singles)]
impl_tr!(chess_prev_noworth, passes);
#[cfg(singles)]
impl_tr!(chess_prev_worth, passes);
#[cfg(singles)]
impl_tr!(chess_prev_42, passes);
#[cfg(singles)]
impl_tr!(chess_prev_project, passes);
#[cfg(singles)]
impl_tr!(chess_prev_asked, passes);
#[cfg(singles)]
impl_tr!(atomic_prev, passes);
#[cfg(singles)]
impl_tr!(atomic_bits, passes);
#[cfg(singles)]
impl_tr!(atomic_bits_scan, passes);
#[cfg(singles)]
impl_tr!(atomic_bits_cut, passes);
#[cfg(singles)]
impl_tr!(atomic_bits_shift, passes);
#[cfg(singles)]
impl_tr!(atomic_bits_base, passes);
#[cfg(singles)]
impl_tr!(atomic_prev41_orient, passes);
#[cfg(singles)]
impl_tr!(atomic_prev41_ends, passes);
#[cfg(singles)]
impl_tr!(atomic_prev41_start, passes);
#[cfg(singles)]
impl_tr!(atomic_prev_noworth, passes);
#[cfg(singles)]
impl_tr!(atomic_prev_worth, passes);
#[cfg(singles)]
impl_tr!(atomic_prev_42, passes);
#[cfg(singles)]
impl_tr!(atomic_prev_project, passes);
#[cfg(singles)]
impl_tr!(atomic_prev_asked, passes);

const SEED: u64 = 77;

/// The kernel's states with the move each makes, and the same states of one module (the kernel's walk of the certificate; the module follows it move by move).
struct Positions<T: Tr> { kernel: Vec<(<KernelGame as Game>::State, usize)>, trans: Vec<(T::S, usize)>, moves: usize }

fn positions<T: Tr>(kernel: &KernelGame, t: &mut T, games: usize, plies: usize, every: usize) -> Positions<T> {
    let mut rng = Rng::new(SEED);
    let mut p = Positions::<T> { kernel: vec![], trans: vec![], moves: 0 };
    for _ in 0..games {
        let (mut ks, mut ts) = (kernel.initial(), t.initial_t());
        let mut ply = 0usize;
        loop {
            let km = kernel.moves(&ks);
            if kernel.outcome(&ks).is_some() || km.is_empty() { break; }
            let mut sorted: Vec<(String, usize)> = km.iter().enumerate().map(|(i, m)| (kernel.move_name(&ks, m), i)).collect();
            sorted.sort();
            let follow = ply < plies;
            if follow && ply % every == 0 {
                let want = sorted.iter().map(|x| spec_term(&x.0)).min().unwrap();
                let kmin = km.iter().position(|m| spec_term(&kernel.move_name(&ks, m)) == want).unwrap();
                let tmv = t.moves_t(&ts);
                let tmin = tmv.iter().position(|m| t.render_t(*m) == want).unwrap();
                assert_eq!(km.len(), tmv.len());
                p.moves += km.len();
                p.kernel.push((ks.clone(), kmin));
                p.trans.push((ts.clone(), tmin));
            }
            let (nm, i) = sorted[rng.below(sorted.len())].clone();
            if follow {
                let tm = t.term_t(&spec_term(&nm)).unwrap();
                ts = t.next_t(&ts, tm);
            }
            ks = kernel.apply(&ks, &km[i]);
            ply += 1;
        }
    }
    p
}

#[derive(Clone, Copy, PartialEq)]
enum Work { Moves, MovesAndNext }

fn pass_kernel(k: &KernelGame, p: &[(<KernelGame as Game>::State, usize)], n: usize, w: Work) -> usize {
    let mut sum = 0;
    for (s, i) in p.iter().take(n) {
        let ms = k.moves(s);
        sum += ms.len();
        if w == Work::MovesAndNext { let c = k.apply(s, &ms[*i]); std::hint::black_box(&c); sum += 1; }
    }
    sum
}
fn pass_trans<T: Tr>(t: &mut T, p: &Positions<T>, n: usize, w: Work) -> usize {
    let mut sum = 0;
    for (s, i) in p.trans.iter().take(n) {
        let ms = t.moves_t(s);
        sum += ms.len();
        if w == Work::MovesAndNext { let c = t.next_t(s, ms[*i]); std::hint::black_box(&c); sum += 1; }
    }
    sum
}

/// Positions a second: the passes are repeated until `min_secs` have gone (at least one pass).
fn rate(min_secs: f64, n: usize, mut one: impl FnMut() -> usize) -> (f64, usize) {
    let t0 = Instant::now();
    let (mut done, mut sink) = (0usize, 0usize);
    loop {
        sink += one();
        done += n;
        if t0.elapsed().as_secs_f64() >= min_secs { break; }
    }
    (done as f64 / t0.elapsed().as_secs_f64(), sink)
}

struct Row { name: &'static str, rate: Box<dyn FnMut(f64, usize, Work) -> (f64, usize)> }

type Maker = Box<dyn Fn(&KernelGame) -> (Row, usize, usize)>;

fn module_row<T: Tr + 'static>(name: &'static str, kernel: &KernelGame) -> (Row, usize, usize) {
    let mut t = T::make();
    let p = positions(kernel, &mut t, 20, 80, 4);
    let (n, moves) = (p.trans.len(), p.moves);
    (Row { name, rate: Box::new(move |secs, n, w| rate(secs, n, || pass_trans(&mut t, &p, n, w))) }, n, moves)
}

macro_rules! makers { ($( ($name:literal, $m:ty) ),* $(,)?) => {{ let mut v: Vec<Maker> = vec![]; $( v.push(Box::new(|k: &KernelGame| module_row::<$m>($name, k))); )* v }} }

/// `runs` full runs of one second each (three in the table), after a short run; every module must answer as many moves as the kernel.
fn run_game<P: Tr + 'static>(game: &str, full: bool, makers: Vec<Maker>, runs: usize) {
    let kernel = KernelGame::new(game, 518).unwrap();
    let kp = positions(&kernel, &mut P::make(), 20, 80, 4);
    let n = kp.trans.len();
    println!("OBS 79.40 E list {game}: {n} positions (every 4th state of the first 80 plies of the certificate's 20 random games, seed {SEED}), {:.1} allowed moves a position on average", kp.moves as f64 / n as f64);
    let mut rows: Vec<Row> = vec![];
    let mut kern = Row { name: "kernel", rate: Box::new({ let k = KernelGame::new(game, 518).unwrap(); let pos = kp.kernel.clone(); move |secs, n, w| rate(secs, n, || pass_kernel(&k, &pos, n, w)) }) };
    for mk in makers.iter() {
        let (r, nn, mv) = mk(&kernel);
        assert_eq!((nn, mv), (n, kp.moves), "{}: the module's positions are the kernel's", r.name);
        rows.push(r);
    }
    let short = 20.min(n);
    for (w, wn) in [(Work::Moves, "allowed moves"), (Work::MovesAndNext, "allowed moves and the next state of one move")] {
        let load = load_average();
        let mut line = format!("kernel {:.0}", (kern.rate)(0.0, short, w).0);
        for r in rows.iter_mut() { line += &format!(" {} {:.0}", r.name, (r.rate)(0.0, short, w).0); }
        println!("OBS 79.40 E {game} short ({short} positions, {wn}) load {load}: {line} positions a second");
        if !full { continue; }
        for run in 1..=runs {
            // a run is five rounds in which every module is clocked for a fifth of a second in turn, so that a load that comes and goes falls on all of them alike; the rate is positions over seconds of the five slices
            let load = load_average();
            let mut tot = vec![(0.0f64, 0.0f64); rows.len() + 1];
            for _round in 0..5 {
                let t0 = Instant::now();
                let r0 = (kern.rate)(0.2, n, w).0;
                tot[0].0 += r0 * t0.elapsed().as_secs_f64(); tot[0].1 += t0.elapsed().as_secs_f64();
                for (i, r) in rows.iter_mut().enumerate() {
                    let t0 = Instant::now();
                    let x = (r.rate)(0.2, n, w).0;
                    let dt = t0.elapsed().as_secs_f64();
                    tot[i + 1].0 += x * dt; tot[i + 1].1 += dt;
                }
            }
            let mut line = format!("kernel {:.0}", tot[0].0 / tot[0].1);
            for (i, r) in rows.iter().enumerate() { line += &format!(" {} {:.0}", r.name, tot[i + 1].0 / tot[i + 1].1); }
            println!("OBS 79.40 E {game} run {run} ({wn}) load before {load}, after {}: {line} positions a second", load_average());
        }
    }
}

/// The time by piece of the module with every pass (the pieces that run: calls, time, rows made, rows visited, indexes asked), over the list a few times.
fn pieces<F: Tr + 'static>(game: &str, label: &str) {
    let kernel = KernelGame::new(game, 518).unwrap();
    let mut t = F::make();
    let p = positions(&kernel, &mut t, 20, 80, 4);
    let n = p.trans.len();
    t.profiling_t(true);
    t.reset_profile_t();
    let t0 = Instant::now();
    let (a0, b0) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    let mut sink = 0;
    for _ in 0..5 { sink += pass_trans(&mut t, &p, n, Work::MovesAndNext); }
    let wall = t0.elapsed().as_secs_f64();
    let (a1, b1) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    println!("OBS 79.40 E allocations {game} {label}: {:.1} allocations and {:.0} bytes allocated a position (moves and one next each)", (a1 - a0) as f64 / (5 * n) as f64, (b1 - b0) as f64 / (5 * n) as f64);
    t.profiling_t(false);
    let pcs = t.pieces_t();
    let total: u64 = pcs.iter().map(|x| x.3).sum();
    println!("OBS 79.40 E pieces {game} {label}: {} positions answered (moves and one next each, profile on, load {}), wall {wall:.2} s, the pieces together {:.2} s ({:.0} per cent of the wall), checksum {sink}", 5 * n, load_average(), total as f64 / 1e9, 100.0 * total as f64 / 1e9 / wall);
    let mut order: Vec<usize> = (0..pcs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pcs[i].3));
    for (rank, &i) in order.iter().take(10).enumerate() {
        let (name, st, calls, nanos, rows, visits, probes) = pcs[i];
        println!("OBS 79.40 E piece {game} {label} #{}: {name}{} share {:.1}% time {:.3} s calls {calls} rows made {rows} rows visited {visits} indexes asked {probes}", rank + 1, if st { " (static)" } else { "" }, 100.0 * nanos as f64 / total as f64, nanos as f64 / 1e9);
    }
    println!("OBS 79.40 E pieces {game} {label}: {} of {} pieces ran; the ten above hold {:.1}% of the pieces' time", pcs.iter().filter(|x| x.2 > 0).count(), pcs.len(), 100.0 * order.iter().take(10).map(|&i| pcs[i].3).sum::<u64>() as f64 / total as f64);
}


#[cfg(singles)]
fn chess_makers() -> Vec<Maker> { makers![("plain", chess::Game), ("prev", chess_prev::Game), ("bits", chess_bits::Game), ("b+scan", chess_bits_scan::Game), ("b+cut", chess_bits_cut::Game), ("b+shift", chess_bits_shift::Game), ("b+base", chess_bits_base::Game), ("p+start", chess_prev41_start::Game), ("p+ends", chess_prev41_ends::Game), ("p+orient", chess_prev41_orient::Game), ("p+asked", chess_prev_asked::Game), ("p+project", chess_prev_project::Game), ("p+prev42", chess_prev_42::Game), ("p+worth", chess_prev_worth::Game), ("p+noworth", chess_prev_noworth::Game), ("fast", chess_fast::Game)] }
#[cfg(not(singles))]
fn chess_makers() -> Vec<Maker> { makers![("plain", chess::Game), ("fast", chess_fast::Game)] }
#[cfg(singles)]
fn atomic_makers() -> Vec<Maker> { makers![("plain", atomic::Game), ("prev", atomic_prev::Game), ("bits", atomic_bits::Game), ("b+scan", atomic_bits_scan::Game), ("b+cut", atomic_bits_cut::Game), ("b+shift", atomic_bits_shift::Game), ("b+base", atomic_bits_base::Game), ("p+start", atomic_prev41_start::Game), ("p+ends", atomic_prev41_ends::Game), ("p+orient", atomic_prev41_orient::Game), ("p+asked", atomic_prev_asked::Game), ("p+project", atomic_prev_project::Game), ("p+prev42", atomic_prev_42::Game), ("p+worth", atomic_prev_worth::Game), ("p+noworth", atomic_prev_noworth::Game), ("fast", atomic_fast::Game)] }
#[cfg(not(singles))]
fn atomic_makers() -> Vec<Maker> { makers![("plain", atomic::Game), ("fast", atomic_fast::Game)] }

#[test]
fn chess_a_short_run_of_the_kernel_and_the_modules_gives_the_same_number_of_moves() { run_game::<chess::Game>("chess", false, chess_makers(), 0); }
#[test]
fn atomic_a_short_run_of_the_kernel_and_the_modules_gives_the_same_number_of_moves() { run_game::<atomic::Game>("atomic", false, atomic_makers(), 0); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_times_of_the_passes() { run_game::<chess::Game>("chess", true, chess_makers(), 3); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_times_of_the_passes() { run_game::<atomic::Game>("atomic", true, atomic_makers(), 3); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_time_by_piece_of_each_module() { pieces::<chess::Game>("chess", "plain"); pieces::<chess_fast::Game>("chess", "fast"); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_time_by_piece_of_each_module() { pieces::<atomic::Game>("atomic", "plain"); pieces::<atomic_fast::Game>("atomic", "fast"); }

#[cfg(singles)]
fn chess_combos() -> Vec<Maker> { makers![("plain", chess::Game), ("prev", chess_prev::Game), ("bits", chess_bits::Game), ("b+scan", chess_bits_scan::Game), ("b+cut", chess_bits_cut::Game), ("b+shift", chess_bits_shift::Game), ("b+base", chess_bits_base::Game), ("p+start", chess_prev41_start::Game), ("p+ends", chess_prev41_ends::Game), ("p+orient", chess_prev41_orient::Game), ("p+asked", chess_prev_asked::Game), ("p+project", chess_prev_project::Game), ("p+prev42", chess_prev_42::Game), ("p+worth", chess_prev_worth::Game), ("p+noworth", chess_prev_noworth::Game), ("fast", chess_fast::Game)] }
#[cfg(singles)]
fn atomic_combos() -> Vec<Maker> { makers![("plain", atomic::Game), ("prev", atomic_prev::Game), ("bits", atomic_bits::Game), ("b+scan", atomic_bits_scan::Game), ("b+cut", atomic_bits_cut::Game), ("b+shift", atomic_bits_shift::Game), ("b+base", atomic_bits_base::Game), ("p+start", atomic_prev41_start::Game), ("p+ends", atomic_prev41_ends::Game), ("p+orient", atomic_prev41_orient::Game), ("p+asked", atomic_prev_asked::Game), ("p+project", atomic_prev_project::Game), ("p+prev42", atomic_prev_42::Game), ("p+worth", atomic_prev_worth::Game), ("p+noworth", atomic_prev_noworth::Game), ("fast", atomic_fast::Game)] }
#[cfg(singles)]
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_times_of_the_combinations() { run_game::<chess::Game>("chess", true, chess_combos(), 3); }
#[cfg(singles)]
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_times_of_the_combinations() { run_game::<atomic::Game>("atomic", true, atomic_combos(), 3); }
