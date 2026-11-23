//! PLAN 79.37 PART E AND PART F (SYSTEM_SPEC 11, "The crux of layer 2 ..."; ledger rows 45, 44; piece: rules): HOW FAST IS A PLAIN TRANSLATION. One thread; the same list of positions for the three ways of answering the same two questions
//! about a position -- its allowed moves, and the state after one of them: the evaluator of the description (cs_game's `GdlGame`), the code the translator wrote (tests/generated/<game>.rs), and the kernel over the outside library
//! (`KernelGame`, shakmaty). Three runs of each, a short run first, the machine's load printed beside each run; then the time by piece of the translated code (the ten pieces that take most, with their calls and the work they did).
//! The list: every 4th state of the first 80 plies of the certificate's random games (seed 77, kernel's walk), the states that are not ends: the corpus is the certificate's, so the positions are those the certificate saw. This is a clock, not a law:
//! the numbers are OBS lines, read as directions. The full run is ignored by default (`cargo test --release --offline -p cs_game --test translated_speed -- --ignored --nocapture --test-threads=1`); a short run is always made.
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
include!("generated/atomic.rs");
impl_tr!(chess);
impl_tr!(atomic);

const SEED: u64 = 77;

/// The three kinds of state of one position, and the move each makes (the smallest by its text, found outside the timing).
struct Positions<T: Tr> { kernel: Vec<(<KernelGame as Game>::State, usize)>, eval: Vec<(cs_game::gdl_game::GdlState, usize)>, trans: Vec<(T::S, usize)>, moves: usize }

/// The corpus: the kernel's walk of the certificate (moves sorted by name, the seeded generator), every `every`-th state of the first `plies` plies of each of `games` games, the ends left out; the evaluator and the translated code
/// follow the walk move by move, so each position is made once in each of the three.
fn positions<T: Tr>(kernel: &KernelGame, t: &mut T, ev: &cs_game::gdl_game::GdlGame, games: usize, plies: usize, every: usize) -> Positions<T> {
    let mut rng = Rng::new(SEED);
    let mut p = Positions::<T> { kernel: vec![], eval: vec![], trans: vec![], moves: 0 };
    for _ in 0..games {
        let (mut ks, mut es, mut ts) = (kernel.initial(), ev.initial(), t.initial_t());
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
                let em = ev.moves(&es);
                let emin = em.iter().position(|m| ev.move_name(&es, m) == want).unwrap();
                let tmv = t.moves_t(&ts);
                let tmin = tmv.iter().position(|m| t.render_t(*m) == want).unwrap();
                assert_eq!(km.len(), em.len());
                assert_eq!(km.len(), tmv.len());
                p.moves += km.len();
                p.kernel.push((ks.clone(), kmin));
                p.eval.push((es.clone(), emin));
                p.trans.push((ts.clone(), tmin));
            }
            let (nm, i) = sorted[rng.below(sorted.len())].clone();
            if follow {
                let text = spec_term(&nm);
                let em = ev.find_move(&es, &text).unwrap();
                let tm = t.term_t(&text).unwrap();
                es = ev.apply(&es, &em);
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

/// One pass over the first `n` positions: positions answered, and a checksum so no answer is thrown away.
fn pass_kernel<T: Tr>(k: &KernelGame, p: &Positions<T>, n: usize, w: Work) -> usize {
    let mut sum = 0;
    for (s, i) in p.kernel.iter().take(n) {
        let ms = k.moves(s);
        sum += ms.len();
        if w == Work::MovesAndNext { let c = k.apply(s, &ms[*i]); std::hint::black_box(&c); sum += 1; }
    }
    sum
}
fn pass_eval<T: Tr>(g: &cs_game::gdl_game::GdlGame, p: &Positions<T>, n: usize, w: Work) -> usize {
    let mut sum = 0;
    for (s, i) in p.eval.iter().take(n) {
        let ms = g.moves(s);
        sum += ms.len();
        if w == Work::MovesAndNext { let c = g.apply(s, &ms[*i]); std::hint::black_box(&c); sum += 1; }
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
fn rate(min_secs: f64, n: usize, mut one: impl FnMut() -> usize) -> (f64, usize, f64, usize) {
    let t0 = Instant::now();
    let (mut done, mut sink) = (0usize, 0usize);
    loop {
        sink += one();
        done += n;
        if t0.elapsed().as_secs_f64() >= min_secs { break; }
    }
    let secs = t0.elapsed().as_secs_f64();
    (done as f64 / secs, done, secs, sink)
}

fn run_game<T: Tr>(game: &str, text: &str, full: bool) {
    let kernel = KernelGame::new(game, 518).unwrap();
    let mut t = T::make();
    let ev = evaluator(text);
    let p = positions(&kernel, &mut t, &ev, 20, 80, 4);
    let n = p.trans.len();
    println!("OBS 79.37 E list {game}: {n} positions (every 4th state of the first 80 plies of the certificate's 20 random games, seed {SEED}), {:.1} allowed moves a position on average", p.moves as f64 / n as f64);
    let short = 20.min(n);
    let (secs, label) = if full { (1.0, "run") } else { (0.0, "short") };
    for (w, wn) in [(Work::Moves, "allowed moves"), (Work::MovesAndNext, "allowed moves and the next state of one move")] {
        // a short run first
        let load = load_average();
        let a = rate(0.0, short, || pass_kernel(&kernel, &p, short, w));
        let b = rate(0.0, short, || pass_trans(&mut t, &p, short, w));
        let c = rate(0.0, short, || pass_eval(&ev, &p, short, w));
        println!("OBS 79.37 E {game} short ({short} positions, {wn}) load {load}: kernel {:.0} translated {:.0} evaluator {:.0} positions a second", a.0, b.0, c.0);
        if !full { continue; }
        for r in 1..=3 {
            let load = load_average();
            let a = rate(secs, n, || pass_kernel(&kernel, &p, n, w));
            let b = rate(secs, n, || pass_trans(&mut t, &p, n, w));
            let c = rate(secs, n, || pass_eval(&ev, &p, n, w));
            println!("OBS 79.37 E {game} {label} {r} ({wn}) load before {load}, after {}: kernel {:.0} ({} positions, {:.2} s), translated {:.0} ({} positions, {:.2} s), evaluator {:.0} ({} positions, {:.2} s) positions a second",
                     load_average(), a.0, a.1, a.2, b.0, b.1, b.2, c.0, c.1, c.2);
        }
    }
    if !full { return; }
    // the time by piece: one thread, the profile on, the combined work over the list a few times
    t.profiling_t(true);
    t.reset_profile_t();
    let t0 = Instant::now();
    let mut sink = 0;
    for _ in 0..5 { sink += pass_trans(&mut t, &p, n, Work::MovesAndNext); }
    let wall = t0.elapsed().as_secs_f64();
    t.profiling_t(false);
    let pieces = t.pieces_t();
    let total: u64 = pieces.iter().map(|x| x.3).sum();
    let reads = (5 * n) as u64;
    println!("OBS 79.37 E pieces {game}: {} positions answered (moves and one next each, profile on, load {}), wall {wall:.2} s, the pieces together {:.2} s ({:.0} per cent of the wall), checksum {sink}", reads, load_average(), total as f64 / 1e9, 100.0 * total as f64 / 1e9 / wall);
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pieces[i].3));
    for (rank, &i) in order.iter().take(10).enumerate() {
        let (name, st, calls, nanos, rows, visits, probes) = pieces[i];
        println!("OBS 79.37 E piece {game} #{}: {name}{} share {:.1}% time {:.3} s calls {calls} rows made {rows} rows visited {visits} indexes asked {probes}", rank + 1, if st { " (static)" } else { "" }, 100.0 * nanos as f64 / total as f64, nanos as f64 / 1e9);
    }
    let live = pieces.iter().filter(|x| x.2 > 0).count();
    println!("OBS 79.37 E pieces {game}: {live} of {} pieces ran; the ten above hold {:.1}% of the pieces' time", pieces.len(), 100.0 * order.iter().take(10).map(|&i| pieces[i].3).sum::<u64>() as f64 / total as f64);
}

#[test]
fn chess_a_short_run_of_the_three_ways_of_answering_gives_the_same_number_of_moves() { run_game::<chess::Game>("chess", &description(None), false); }
#[test]
fn atomic_a_short_run_of_the_three_ways_of_answering_gives_the_same_number_of_moves() { run_game::<atomic::Game>("atomic", &description(Some("variants/atomic.gdl")), false); }

#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn chess_the_times() { run_game::<chess::Game>("chess", &description(None), true); }
#[test]
#[ignore = "a clock: run by hand, alone on the machine, with --nocapture"]
fn atomic_the_times() { run_game::<atomic::Game>("atomic", &description(Some("variants/atomic.gdl")), true); }
