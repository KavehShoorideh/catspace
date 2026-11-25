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
// 2026-11-24 (plan 79.45): the first game's module with every pass and its lemma (kept), and under `--cfg singles` the module of every pass of 79.43 (`p43`: no gated, no lemma) and the lemma with every pass but gated (`lm_nogate`); the clocks of the lemma are at the end of the file
include!("generated/chess_lemma.rs");
#[cfg(singles)]
include!("generated/scratch/chess_p43.rs");
#[cfg(singles)]
include!("generated/scratch/chess_lm_nogate.rs");
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
impl_tr!(chess_fast, topdown);
impl_tr!(chess_lemma, topdown);
// 2026-11-24 (plan 79.52 part B): the kept modules of the commit before the plan (renamed, under generated/scratch, made by hand from git: the clocks "before"), compiled with `--cfg before52`
#[cfg(before52)]
include!("generated/scratch/chess_before.rs");
#[cfg(before52)]
include!("generated/scratch/chess_lemma_before.rs");
#[cfg(before52)]
include!("generated/scratch/atomic_before.rs");
#[cfg(before52)]
include!("generated/scratch/chess_p52_cones.rs");
#[cfg(before52)]
include!("generated/scratch/atomic_p52_cones.rs");
#[cfg(before52)]
impl_tr!(chess_p52_cones, fresh);
#[cfg(before52)]
impl_tr!(atomic_p52_cones, fresh);
#[cfg(before52)]
include!("generated/scratch/chess_b52.rs");
#[cfg(before52)]
include!("generated/scratch/chess_lemma_b52.rs");
#[cfg(before52)]
include!("generated/scratch/atomic_b52.rs");
#[cfg(before52)]
impl_tr!(chess_b52, fresh);
#[cfg(before52)]
impl_tr!(chess_lemma_b52, fresh);
#[cfg(before52)]
impl_tr!(atomic_b52, fresh);
#[cfg(before52)]
impl_tr!(chess_before, passes);
#[cfg(before52)]
impl_tr!(chess_lemma_before, passes);
#[cfg(before52)]
impl_tr!(atomic_before, passes);
#[cfg(singles)]
impl_tr!(chess_p43, passes);
#[cfg(singles)]
impl_tr!(chess_lm_nogate, passes);
impl_tr!(atomic);
impl_tr!(atomic_fast, topdown);
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
enum Work { Moves, MovesAndNext, MovesInto, MovesIntoAndNext, LegalInto, LegalIntoAndNext, End, EndTd, HasMoveTd, TdLoad }

fn pass_kernel(k: &KernelGame, p: &[(<KernelGame as Game>::State, usize)], n: usize, w: Work) -> usize {
    let mut sum = 0;
    for (s, i) in p.iter().take(n) {
        if matches!(w, Work::End | Work::EndTd) { sum += k.outcome(s).is_some() as usize; continue; }
        if w == Work::TdLoad { sum += 1; continue; }
        if w == Work::HasMoveTd { sum += !k.moves(s).is_empty() as usize; continue; }
        let ms = k.moves(s);
        sum += ms.len();
        if matches!(w, Work::MovesAndNext | Work::MovesIntoAndNext | Work::LegalIntoAndNext) { let c = k.apply(s, &ms[*i]); std::hint::black_box(&c); sum += 1; }
    }
    sum
}
fn pass_trans<T: Tr>(t: &mut T, p: &Positions<T>, n: usize, w: Work) -> usize {
    let mut sum = 0;
    let mut buf: Vec<u32> = Vec::with_capacity(256);                       // 2026-11-24 (plan 79.52): the buffer of the questions that make no memory (`moves_into_t`, `legal_into_t`)
    for (s, i) in p.trans.iter().take(n) {
        match w {
            Work::End => { sum += t.terminal_t(s) as usize; }
            Work::EndTd => { sum += t.terminal_td_t(s) as usize; }
            Work::TdLoad => { t.td_load_only_t(s); sum += 1; }
            Work::HasMoveTd => { sum += t.has_move_td_t(s) as usize; }
            Work::Moves | Work::MovesAndNext => {
                let ms = t.moves_t(s);
                sum += ms.len();
                if w == Work::MovesAndNext { let c = t.next_t(s, ms[*i]); std::hint::black_box(&c); sum += 1; }
            }
            Work::MovesInto | Work::MovesIntoAndNext => {
                t.moves_into_t(s, &mut buf);
                sum += buf.len();
                if w == Work::MovesIntoAndNext { let c = t.next_t(s, buf[*i]); std::hint::black_box(&c); sum += 1; }
            }
            Work::LegalInto | Work::LegalIntoAndNext => {
                t.legal_into_t(s, &mut buf);
                sum += buf.len();
                if w == Work::LegalIntoAndNext { let c = t.next_t(s, buf[*i]); std::hint::black_box(&c); sum += 1; }
            }
        }
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
    run_game_works::<P>(game, full, makers, runs, &[(Work::Moves, "allowed moves"), (Work::MovesAndNext, "allowed moves and the next state of one move")]);
}

/// The same with the questions to clock named (plan 79.52: the questions that make no memory, with the end and without it).
fn run_game_works<P: Tr + 'static>(game: &str, full: bool, makers: Vec<Maker>, runs: usize, works: &[(Work, &str)]) {
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
    for &(w, wn) in works {
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
fn pieces<F: Tr + 'static>(game: &str, label: &str) { pieces_work::<F>(game, label, Work::MovesAndNext, "moves and one next each") }

fn pieces_work<F: Tr + 'static>(game: &str, label: &str, work: Work, what: &str) {
    let kernel = KernelGame::new(game, 518).unwrap();
    let mut t = F::make();
    let p = positions(&kernel, &mut t, 20, 80, 4);
    let n = p.trans.len();
    t.profiling_t(true);
    t.reset_profile_t();
    let t0 = Instant::now();
    let (a0, b0) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    let mut sink = 0;
    for _ in 0..5 { sink += pass_trans(&mut t, &p, n, work); }
    let wall = t0.elapsed().as_secs_f64();
    let (a1, b1) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    println!("OBS 79.40 E allocations {game} {label} ({what}): {:.1} allocations and {:.0} bytes allocated a position", (a1 - a0) as f64 / (5 * n) as f64, (b1 - b0) as f64 / (5 * n) as f64);
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

// 2026-11-24 (plan 79.45): THE TIMES OF THE LEMMA. The kernel, the plain translation, every pass of 79.43 (`prev43`), every pass but gated with the lemma (`lemma`) and every pass with the lemma (`lemma+gated`, the kept module); the same list, the same two questions. Without the flag `fast` stands for prev43 plus the pass gated, which does nothing without a lemma, and `lemma+gated` is the only module with the lemma.
#[cfg(singles)]
fn chess_lemma_makers() -> Vec<Maker> { makers![("plain", chess::Game), ("prev43", chess_p43::Game), ("lemma", chess_lm_nogate::Game), ("lemma+gated", chess_lemma::Game)] }
#[cfg(not(singles))]
fn chess_lemma_makers() -> Vec<Maker> { makers![("plain", chess::Game), ("fast", chess_fast::Game), ("lemma+gated", chess_lemma::Game)] }

#[test]
fn chess_a_short_run_of_the_modules_with_the_lemma_gives_the_same_number_of_moves() { run_game::<chess::Game>("chess", false, chess_lemma_makers(), 0); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_times_of_the_lemma() { run_game::<chess::Game>("chess", true, chess_lemma_makers(), 3); }
#[cfg(singles)]
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_time_by_piece_of_the_lemma_modules() { pieces::<chess_p43::Game>("chess", "prev43"); pieces::<chess_lm_nogate::Game>("chess", "lemma"); pieces::<chess_lemma::Game>("chess", "lemma+gated"); }
#[cfg(not(singles))]
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_time_by_piece_of_the_lemma_modules() { pieces::<chess_fast::Game>("chess", "fast"); pieces::<chess_lemma::Game>("chess", "lemma+gated"); }

// ------------------------------------------------------------------------------------------------------------------------------------------ PLAN 79.47 PART B: the clocks of the five more games (the kernel, the plain translation, every pass)
// 2026-11-24 (plan 79.47): the five more games of the seven certified kernels (3check, kingofthehill, antichess, chess960, horde), each a delta on the first game's description, the modules made by the same translator.
include!("generated/3check.rs");
include!("generated/3check_fast.rs");
impl_tr!(threecheck);
impl_tr!(threecheck_fast, passes);
include!("generated/kingofthehill.rs");
include!("generated/kingofthehill_fast.rs");
impl_tr!(kingofthehill);
impl_tr!(kingofthehill_fast, passes);
include!("generated/antichess.rs");
include!("generated/antichess_fast.rs");
impl_tr!(antichess);
impl_tr!(antichess_fast, passes);
include!("generated/chess960.rs");
include!("generated/chess960_fast.rs");
impl_tr!(chess960);
impl_tr!(chess960_fast, passes);
include!("generated/horde.rs");
include!("generated/horde_fast.rs");
impl_tr!(horde);
impl_tr!(horde_fast, passes);
#[test]
fn threecheck_a_short_run_of_the_kernel_and_the_modules_gives_the_same_number_of_moves() { run_game::<threecheck::Game>("3check", false, makers![("plain", threecheck::Game), ("fast", threecheck_fast::Game)], 0); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn threecheck_the_times_of_the_passes() { run_game::<threecheck::Game>("3check", true, makers![("plain", threecheck::Game), ("fast", threecheck_fast::Game)], 3); }
#[test]
fn kingofthehill_a_short_run_of_the_kernel_and_the_modules_gives_the_same_number_of_moves() { run_game::<kingofthehill::Game>("kingofthehill", false, makers![("plain", kingofthehill::Game), ("fast", kingofthehill_fast::Game)], 0); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn kingofthehill_the_times_of_the_passes() { run_game::<kingofthehill::Game>("kingofthehill", true, makers![("plain", kingofthehill::Game), ("fast", kingofthehill_fast::Game)], 3); }
#[test]
fn antichess_a_short_run_of_the_kernel_and_the_modules_gives_the_same_number_of_moves() { run_game::<antichess::Game>("antichess", false, makers![("plain", antichess::Game), ("fast", antichess_fast::Game)], 0); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn antichess_the_times_of_the_passes() { run_game::<antichess::Game>("antichess", true, makers![("plain", antichess::Game), ("fast", antichess_fast::Game)], 3); }
#[test]
fn chess960_a_short_run_of_the_kernel_and_the_modules_gives_the_same_number_of_moves() { run_game::<chess960::Game>("chess960", false, makers![("plain", chess960::Game), ("fast", chess960_fast::Game)], 0); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess960_the_times_of_the_passes() { run_game::<chess960::Game>("chess960", true, makers![("plain", chess960::Game), ("fast", chess960_fast::Game)], 3); }
#[test]
fn horde_a_short_run_of_the_kernel_and_the_modules_gives_the_same_number_of_moves() { run_game::<horde::Game>("horde", false, makers![("plain", horde::Game), ("fast", horde_fast::Game)], 0); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn horde_the_times_of_the_passes() { run_game::<horde::Game>("horde", true, makers![("plain", horde::Game), ("fast", horde_fast::Game)], 3); }

// ------------------------------------------------------------------------------------------------------------------------------------------ PLAN 79.52 PART B: the clocks of the passes `cones` and `words`
// The kernel, the kept modules of the commit before the plan (`before`, `lemma-before`; only with `--cfg before52`), and the kept modules now (`fast`, `lemma`): the allowed moves with the end and without it, into a buffer the caller owns, with and without the next
// state of one move, and the way the old entry point (`moves_t`) stands. The allocation count and the profile by piece after each: `chess_the_pieces_after_part_b`.
const WORKS_52: &[(Work, &str)] = &[(Work::Moves, "allowed moves (moves_t)"), (Work::MovesInto, "allowed moves into a buffer"), (Work::LegalInto, "allowed moves into a buffer, without the end"),
    (Work::MovesAndNext, "allowed moves (moves_t) and the next state of one move"), (Work::MovesIntoAndNext, "allowed moves into a buffer and the next state of one move"), (Work::LegalIntoAndNext, "allowed moves without the end and the next state of one move")];
#[cfg(before52)]
fn chess_52() -> Vec<Maker> { makers![("before", chess_before::Game), ("lemma-before", chess_lemma_before::Game), ("step1", chess_p52_cones::Game), ("fast", chess_fast::Game), ("lemma", chess_lemma::Game)] }
#[cfg(not(before52))]
fn chess_52() -> Vec<Maker> { makers![("fast", chess_fast::Game), ("lemma", chess_lemma::Game)] }
#[cfg(before52)]
fn atomic_52() -> Vec<Maker> { makers![("before", atomic_before::Game), ("step1", atomic_p52_cones::Game), ("fast", atomic_fast::Game)] }
#[cfg(not(before52))]
fn atomic_52() -> Vec<Maker> { makers![("fast", atomic_fast::Game)] }
#[test]
fn chess_a_short_run_of_part_b_gives_the_same_number_of_moves() { run_game_works::<chess_fast::Game>("chess", false, chess_52(), 0, WORKS_52); }
#[test]
fn atomic_a_short_run_of_part_b_gives_the_same_number_of_moves() { run_game_works::<atomic_fast::Game>("atomic", false, atomic_52(), 0, WORKS_52); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_times_of_part_b() { run_game_works::<chess_fast::Game>("chess", true, chess_52(), 3, WORKS_52); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_times_of_part_b() { run_game_works::<atomic_fast::Game>("atomic", true, atomic_52(), 3, WORKS_52); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_pieces_after_part_b() {
    for (w, what) in [(Work::MovesAndNext, "moves and one next each"), (Work::MovesIntoAndNext, "moves into a buffer and one next each"), (Work::LegalIntoAndNext, "allowed moves without the end and one next each")] {
        pieces_work::<chess_fast::Game>("chess", "fast", w, what);
        pieces_work::<chess_lemma::Game>("chess", "lemma", w, what);
    }
}
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_pieces_after_part_b() {
    for (w, what) in [(Work::MovesAndNext, "moves and one next each"), (Work::LegalIntoAndNext, "allowed moves without the end and one next each")] { pieces_work::<atomic_fast::Game>("atomic", "fast", w, what); }
}
#[cfg(before52)]
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_pieces_before_part_b() { pieces::<chess_before::Game>("chess", "fast-before"); pieces::<chess_lemma_before::Game>("chess", "lemma-before"); pieces::<atomic_before::Game>("atomic", "fast-before"); }

// ------------------------------------------------------------------------------------------------------------------------------------------ PLAN 79.53: the clocks of the tables for walks and of the top-down way
// The kernel, the kept modules of the commit before the plan (`b52`: every pass of 79.52; only with `--cfg before52`) and the kept modules now (`fast`, `lemma`): the end, the allowed moves, the allowed moves and one next state. The column of the top-down end is in the same table (Work::EndTd) once the pass exists.
const WORKS_53: &[(Work, &str)] = &[(Work::TdLoad, "only the load of the top-down tables"), (Work::End, "the end (terminal_t), bottom-up"), (Work::EndTd, "the end, top-down"), (Work::HasMoveTd, "is there a legal turn, top-down"), (Work::MovesInto, "allowed moves into a buffer"), (Work::LegalInto, "allowed moves into a buffer, without the end"), (Work::MovesIntoAndNext, "allowed moves into a buffer and the next state of one move")];
#[cfg(before52)]
fn chess_53() -> Vec<Maker> { makers![("b52", chess_b52::Game), ("lemma-b52", chess_lemma_b52::Game), ("fast", chess_fast::Game), ("lemma", chess_lemma::Game)] }
#[cfg(not(before52))]
fn chess_53() -> Vec<Maker> { makers![("fast", chess_fast::Game), ("lemma", chess_lemma::Game)] }
#[cfg(before52)]
fn atomic_53() -> Vec<Maker> { makers![("b52", atomic_b52::Game), ("fast", atomic_fast::Game)] }
#[cfg(not(before52))]
fn atomic_53() -> Vec<Maker> { makers![("fast", atomic_fast::Game)] }
#[test]
fn chess_a_short_run_of_plan_53_gives_the_same_number_of_moves() { run_game_works::<chess_fast::Game>("chess", false, chess_53(), 0, WORKS_53); }
#[test]
fn atomic_a_short_run_of_plan_53_gives_the_same_number_of_moves() { run_game_works::<atomic_fast::Game>("atomic", false, atomic_53(), 0, WORKS_53); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_times_of_plan_53() { run_game_works::<chess_fast::Game>("chess", true, chess_53(), 3, WORKS_53); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_times_of_plan_53() { run_game_works::<atomic_fast::Game>("atomic", true, atomic_53(), 3, WORKS_53); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_pieces_in_plan_53() {
    for (w, what) in [(Work::MovesIntoAndNext, "moves into a buffer and one next each")] {
        pieces_work::<chess_fast::Game>("chess", "fast", w, what);
        pieces_work::<chess_lemma::Game>("chess", "lemma", w, what);
    }
}
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_pieces_in_plan_53() { pieces_work::<atomic_fast::Game>("atomic", "fast", Work::MovesIntoAndNext, "moves into a buffer and one next each"); }
#[cfg(before52)]
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_pieces_before_plan_53() { pieces_work::<chess_b52::Game>("chess", "b52", Work::MovesIntoAndNext, "moves into a buffer and one next each"); pieces_work::<chess_lemma_b52::Game>("chess", "lemma-b52", Work::MovesIntoAndNext, "moves into a buffer and one next each"); pieces_work::<atomic_b52::Game>("atomic", "b52", Work::MovesIntoAndNext, "moves into a buffer and one next each"); }
