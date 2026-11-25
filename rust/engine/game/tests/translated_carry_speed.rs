//! PLAN 79.54 PART B, THE CLOCKS (SYSTEM_SPEC 11, last block, (ix); ledger rows 55, 45; piece: rules). THE INTERFACE: none added; it reads the modules `<game>_fast.rs` (every pass but `carry`) and `<game>_carry.rs` (every pass) through `Tr` and the outside library through `KernelGame`.
//! WHAT IT CLOCKS: positions a second of the question "the allowed moves with the end" (`moves_into_t`; the library's `moves`) over THREE WALKS of the same positions, each position asked once in the order of the walk, the walk repeated until `CS_CARRY_SECS` (default 1.0) have gone:
//!   ALONG a seeded game (each state after the last: successor pairs), a DEPTH-FIRST WALK of a small tree (a state, its first three children in the order of their names, and under each of them three more, three levels: each child follows its parent or the last
//!   leaf of the sibling before it: successor, sibling and cousin pairs; and back up), and JUMPS (the same states in a seeded random order: far pairs). Each row is clocked in turn, three runs of each, and the one-minute load is printed at the start of every run.
//! `cargo test --release --offline -p cs_game --test translated_carry_speed -- --ignored --nocapture --test-threads=1`.
#[macro_use]
mod translated_common;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::rng::Rng;
use cs_game::Game;
use std::time::Instant;
use translated_common::*;

include!("generated/chess_fast.rs");
include!("generated/chess_carry.rs");
include!("generated/atomic_fast.rs");
include!("generated/atomic_carry.rs");
impl_tr!(chess_fast, carry);
impl_tr!(chess_carry, carry);
impl_tr!(atomic_fast, carry);
impl_tr!(atomic_carry, carry);

const SEED: u64 = 5454;
fn env_f64(k: &str, d: f64) -> f64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

struct Walk<T: Tr> { name: &'static str, kernel: Vec<<KernelGame as Game>::State>, trans: Vec<T::S> }

/// The states of three walks, the kernel's and the module's in step (a state is made on both sides by the same move, by its name).
fn walks<T: Tr>(kernel: &KernelGame, t: &mut T, games: usize, plies: usize, trees: usize) -> Vec<Walk<T>> {
    let mut rng = Rng::new(SEED);
    let mut along = Walk::<T> { name: "along", kernel: vec![], trans: vec![] };
    let mut tree = Walk::<T> { name: "depth-first", kernel: vec![], trans: vec![] };
    let mut roots: Vec<(<KernelGame as Game>::State, T::S)> = vec![];
    for g in 0..games {
        let (mut ks, mut ts) = (kernel.initial(), t.initial_t());
        for ply in 0..plies {
            let km = kernel.moves(&ks);
            if kernel.outcome(&ks).is_some() || km.is_empty() { break; }
            along.kernel.push(ks.clone());
            along.trans.push(ts.clone());
            if ply % 7 == 3 && roots.len() < trees && g % 2 == 0 { roots.push((ks.clone(), ts.clone())); }
            let mut sorted: Vec<(String, usize)> = km.iter().enumerate().map(|(i, m)| (spec_term(&kernel.move_name(&ks, m)), i)).collect();
            sorted.sort();
            let (nm, i) = sorted[rng.below(sorted.len())].clone();
            let tm = t.term_t(&nm).unwrap();
            ts = t.next_t(&ts, tm);
            ks = kernel.apply(&ks, &km[i]);
        }
    }
    // the depth-first walk: a state, then for each of its first three children (by name) the walk of that child, three levels deep
    fn dfs<T: Tr>(kernel: &KernelGame, t: &mut T, ks: &<KernelGame as Game>::State, ts: &T::S, depth: usize, out: &mut Walk<T>) {
        out.kernel.push(ks.clone());
        out.trans.push(ts.clone());
        if depth == 0 { return; }
        let km = kernel.moves(ks);
        if kernel.outcome(ks).is_some() || km.is_empty() { return; }
        let mut sorted: Vec<(String, usize)> = km.iter().enumerate().map(|(i, m)| (spec_term(&kernel.move_name(ks, m)), i)).collect();
        sorted.sort();
        for (nm, i) in sorted.into_iter().take(3) {
            let tm = t.term_t(&nm).unwrap();
            let nts = t.next_t(ts, tm);
            let nks = kernel.apply(ks, &km[i]);
            dfs(kernel, t, &nks, &nts, depth - 1, out);
        }
    }
    for (ks, ts) in &roots { dfs(kernel, t, ks, ts, 3, &mut tree); }
    // jumps: the states of the depth-first walks and of the game, in a seeded random order
    let mut jumps = Walk::<T> { name: "jumps", kernel: vec![], trans: vec![] };
    let mut order: Vec<usize> = (0..tree.kernel.len() + along.kernel.len()).collect();
    for i in (1..order.len()).rev() { order.swap(i, rng.below(i + 1)); }
    for i in order {
        if i < tree.kernel.len() { jumps.kernel.push(tree.kernel[i].clone()); jumps.trans.push(tree.trans[i].clone()); } else { let j = i - tree.kernel.len(); jumps.kernel.push(along.kernel[j].clone()); jumps.trans.push(along.trans[j].clone()); }
    }
    // the pass walk: a state, the state passed (the next role to move, nothing played; made here from the facts, `translated_common::passed`), the state again, ... (the library has no such state: its row is not clocked on this walk)
    let mut pass = Walk::<T> { name: "pass", kernel: vec![], trans: vec![] };
    for s in along.trans.iter().take(300) {
        if let Some(ps) = passed(t, s, "control") { pass.trans.push(s.clone()); pass.trans.push(ps); }
    }
    vec![along, tree, jumps, pass]
}

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

fn clock<T: Tr, U: Tr>(game: &str, secs: f64, runs: usize) {
    let kernel = KernelGame::new(game, 518).unwrap();
    let (mut a, mut b) = (T::make(), U::make());
    // the walks are made by the module without the pass; the module with the pass reads the same states (made from the facts)
    let ws = walks::<T>(&kernel, &mut a, 12, 70, 6);
    let wb: Vec<Vec<U::S>> = ws.iter().map(|w| w.trans.iter().map(|s| b.state_of_facts_t(&a.facts_of_t(s)).unwrap()).collect()).collect();
    for (w, sb) in ws.iter().zip(wb.iter()) {
        println!("OBS 79.54 B clock {game} walk {}: {} positions", w.name, w.trans.len());
        let n = w.trans.len();
        for run in 0..runs {
            println!("OBS 79.54 B clock {game} walk {} run {run}: one-minute load {}", w.name, load_average());
            let k = if w.kernel.is_empty() { (0.0, 0) } else { rate(secs, n, || { let mut sum = 0; for s in &w.kernel { sum += kernel.moves(s).len(); } sum }) };
            let mut buf = Vec::with_capacity(256);
            let r_a = rate(secs, n, || { let mut sum = 0; for s in &w.trans { a.moves_into_t(s, &mut buf); sum += buf.len(); } sum });
            let r_b = rate(secs, n, || { let mut sum = 0; for s in sb { b.moves_into_t(s, &mut buf); sum += buf.len(); } sum });
            println!("OBS 79.54 B clock {game} walk {} run {run}: positions a second, allowed moves with the end: kernel {:.0}, module without the pass {:.0}, module with the pass {:.0} (checksums {} {} {})", w.name, k.0, r_a.0, r_b.0, k.1, r_a.1, r_b.1);
        }
    }
}

#[test]
fn the_clocks_run_briefly() {
    clock::<chess_fast::Game, chess_carry::Game>("chess", 0.05, 1);
}
#[test]
#[ignore = "the clock: about a minute a game; run alone"]
fn chess_the_times_of_the_carry() { clock::<chess_fast::Game, chess_carry::Game>("chess", env_f64("CS_CARRY_SECS", 1.0), 3); }
#[test]
#[ignore = "the clock: about a minute a game; run alone"]
fn atomic_the_times_of_the_carry() { clock::<atomic_fast::Game, atomic_carry::Game>("atomic", env_f64("CS_CARRY_SECS", 1.0), 3); }
