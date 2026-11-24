//! PLAN 79.38 PART D (SYSTEM_SPEC 11, "The crux of layer 2 is a translator from the description to Rust, made fast piece by piece under tests"; ledger row 45; piece: rules): HOW FAST ARE THE PASSES. One thread; the same list of positions as
//! translated_speed.rs (every 4th state of the first 80 plies of the certificate's 20 random games, seed 77, the kernel's walk, the ends left out); for each game the same two questions about a position (its allowed moves; the state after one
//! of them, the smallest by its text) asked of the kernel over the outside library, the plain translation, each pass alone (dense, demand, copy) and all three together (fast). Three runs of each after a short run, the machine's load printed beside
//! each; then the time by piece of the module with every pass. This is a clock, not a law: the numbers are OBS lines, read as directions. The full run is ignored by default
//! (`cargo test --release --offline -p cs_game --test translated_speed_passes -- --ignored --nocapture --test-threads=1`); a short run is always made and checks that every module answers as many moves as the kernel.
mod variants_common;
#[macro_use]
mod translated_common;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::rng::Rng;
use cs_game::Game;
use std::time::Instant;
use translated_common::*;

include!("generated/chess.rs");
include!("generated/chess_dense.rs");
include!("generated/chess_demand.rs");
include!("generated/chess_copy.rs");
include!("generated/chess_delta.rs");
include!("generated/chess_fast.rs");
include!("generated/atomic.rs");
include!("generated/atomic_dense.rs");
include!("generated/atomic_demand.rs");
include!("generated/atomic_copy.rs");
include!("generated/atomic_delta.rs");
include!("generated/atomic_fast.rs");
impl_tr!(chess);
impl_tr!(chess_dense, passes);
impl_tr!(chess_demand, passes);
impl_tr!(chess_copy, passes);
impl_tr!(chess_delta, passes);
impl_tr!(chess_fast, passes);
impl_tr!(atomic);
impl_tr!(atomic_dense, passes);
impl_tr!(atomic_demand, passes);
impl_tr!(atomic_copy, passes);
impl_tr!(atomic_delta, passes);
impl_tr!(atomic_fast, passes);

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

fn module_row<T: Tr + 'static>(name: &'static str, kernel: &KernelGame) -> (Row, usize, usize) {
    let mut t = T::make();
    let p = positions(kernel, &mut t, 20, 80, 4);
    let (n, moves) = (p.trans.len(), p.moves);
    (Row { name, rate: Box::new(move |secs, n, w| rate(secs, n, || pass_trans(&mut t, &p, n, w))) }, n, moves)
}

fn run_game<P, D, M, C, L, F>(game: &str, full: bool) where P: Tr + 'static, D: Tr + 'static, M: Tr + 'static, C: Tr + 'static, L: Tr + 'static, F: Tr + 'static {
    let kernel = KernelGame::new(game, 518).unwrap();
    let kp = positions(&kernel, &mut P::make(), 20, 80, 4);
    let n = kp.trans.len();
    println!("OBS 79.38 D list {game}: {n} positions (every 4th state of the first 80 plies of the certificate's 20 random games, seed {SEED}), {:.1} allowed moves a position on average", kp.moves as f64 / n as f64);
    let mut rows: Vec<Row> = vec![];
    let mut kern = Row { name: "kernel", rate: Box::new({ let k = KernelGame::new(game, 518).unwrap(); let pos = kp.kernel.clone(); move |secs, n, w| rate(secs, n, || pass_kernel(&k, &pos, n, w)) }) };
    for (r, nn, mv) in [module_row::<P>("plain", &kernel), module_row::<D>("dense", &kernel), module_row::<M>("demand", &kernel), module_row::<C>("copy", &kernel), module_row::<L>("delta", &kernel), module_row::<F>("fast", &kernel)] {
        assert_eq!((nn, mv), (n, kp.moves), "{}: the module's positions are the kernel's", r.name);
        rows.push(r);
    }
    let short = 20.min(n);
    for (w, wn) in [(Work::Moves, "allowed moves"), (Work::MovesAndNext, "allowed moves and the next state of one move")] {
        let load = load_average();
        let mut line = format!("kernel {:.0}", (kern.rate)(0.0, short, w).0);
        for r in rows.iter_mut() { line += &format!(" {} {:.0}", r.name, (r.rate)(0.0, short, w).0); }
        println!("OBS 79.38 D {game} short ({short} positions, {wn}) load {load}: {line} positions a second");
        if !full { continue; }
        for run in 1..=3 {
            let load = load_average();
            let mut line = format!("kernel {:.0}", (kern.rate)(1.0, n, w).0);
            for r in rows.iter_mut() { line += &format!(" {} {:.0}", r.name, (r.rate)(1.0, n, w).0); }
            println!("OBS 79.38 D {game} run {run} ({wn}) load before {load}, after {}: {line} positions a second", load_average());
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
    let mut sink = 0;
    for _ in 0..5 { sink += pass_trans(&mut t, &p, n, Work::MovesAndNext); }
    let wall = t0.elapsed().as_secs_f64();
    t.profiling_t(false);
    let pcs = t.pieces_t();
    let total: u64 = pcs.iter().map(|x| x.3).sum();
    println!("OBS 79.38 D pieces {game} {label}: {} positions answered (moves and one next each, profile on, load {}), wall {wall:.2} s, the pieces together {:.2} s ({:.0} per cent of the wall), checksum {sink}", 5 * n, load_average(), total as f64 / 1e9, 100.0 * total as f64 / 1e9 / wall);
    let mut order: Vec<usize> = (0..pcs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pcs[i].3));
    for (rank, &i) in order.iter().take(10).enumerate() {
        let (name, st, calls, nanos, rows, visits, probes) = pcs[i];
        println!("OBS 79.38 D piece {game} {label} #{}: {name}{} share {:.1}% time {:.3} s calls {calls} rows made {rows} rows visited {visits} indexes asked {probes}", rank + 1, if st { " (static)" } else { "" }, 100.0 * nanos as f64 / total as f64, nanos as f64 / 1e9);
    }
    println!("OBS 79.38 D pieces {game} {label}: {} of {} pieces ran; the ten above hold {:.1}% of the pieces' time", pcs.iter().filter(|x| x.2 > 0).count(), pcs.len(), 100.0 * order.iter().take(10).map(|&i| pcs[i].3).sum::<u64>() as f64 / total as f64);
}

#[test]
fn chess_a_short_run_of_the_kernel_the_plain_translation_and_every_pass_gives_the_same_number_of_moves() { run_game::<chess::Game, chess_dense::Game, chess_demand::Game, chess_copy::Game, chess_delta::Game, chess_fast::Game>("chess", false); }
#[test]
fn atomic_a_short_run_of_the_kernel_the_plain_translation_and_every_pass_gives_the_same_number_of_moves() { run_game::<atomic::Game, atomic_dense::Game, atomic_demand::Game, atomic_copy::Game, atomic_delta::Game, atomic_fast::Game>("atomic", false); }

#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_times_of_the_passes() { run_game::<chess::Game, chess_dense::Game, chess_demand::Game, chess_copy::Game, chess_delta::Game, chess_fast::Game>("chess", true); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_times_of_the_passes() { run_game::<atomic::Game, atomic_dense::Game, atomic_demand::Game, atomic_copy::Game, atomic_delta::Game, atomic_fast::Game>("atomic", true); }

#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_time_by_piece_of_each_module() {
    pieces::<chess::Game>("chess", "plain"); pieces::<chess_dense::Game>("chess", "dense"); pieces::<chess_demand::Game>("chess", "demand"); pieces::<chess_copy::Game>("chess", "copy"); pieces::<chess_delta::Game>("chess", "delta"); pieces::<chess_fast::Game>("chess", "fast");
}
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_time_by_piece_of_each_module() {
    pieces::<atomic::Game>("atomic", "plain"); pieces::<atomic_dense::Game>("atomic", "dense"); pieces::<atomic_demand::Game>("atomic", "demand"); pieces::<atomic_copy::Game>("atomic", "copy"); pieces::<atomic_delta::Game>("atomic", "delta"); pieces::<atomic_fast::Game>("atomic", "fast");
}
