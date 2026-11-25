//! A MEASURE ASKED BY KAVEH 2026-11-25 (SYSTEM_SPEC 11, last block, (ix); ledger row 55; piece: rules; LAYER 2). The lower layers are locked; this adds no pass and changes no module: it only clocks.
//! WHAT IT TAKES: `CS_PLY_GAMES`, a file with one game a line, the moves of the game by the names of the outside library, separated by spaces; `CS_PLY_OUT`, the file the rows are written to;
//! `CS_PLY_REPLAYS` (default 20), how often every game is walked for the clock; `CS_PLY_MAX` (default 120), the last ply kept.
//! WHAT IT DOES: every game is followed from its start; at every ply the one question "the allowed moves with the end" is asked once, of the outside library (`KernelGame::moves`) and of three kept
//! modules of the first game (`fast`: every pass but the carry; `carry`: every pass; `lemma`: with the lemma of the cases), in the order of the game, so each state is asked right after the state before it.
//! WHAT IT RETURNS (rows of `CS_PLY_OUT`, comma separated):
//!   T,<label>,<ply>,<asks>,<nanoseconds in all>                        the clock of the question by ply
//!   P,<label>,<ply>,<piece>,<nanoseconds in all>,<rows made>,<calls>   the same question with the profile on (two walks), by piece
//!   S,<ply>,<states>,<allowed moves in all>,<units in all>             what stands at that ply
//! WHO CALLS IT: `scripts/ply_profile_print.py`, by hand. `cargo test --release --offline -p cs_game --test translated_ply_profile -- --ignored --nocapture`.
#[macro_use]
mod translated_common;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::Game;
use std::fmt::Write as _;
use std::time::Instant;
use translated_common::*;

include!("generated/chess_fast.rs");
include!("generated/chess_carry.rs");
include!("generated/chess_lemma.rs");
impl_tr!(chess_fast, carry);
impl_tr!(chess_carry, carry);
impl_tr!(chess_lemma, topdown);

type KS = <KernelGame as Game>::State;
fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

/// The states of every game for the library, a game a list; a game stops at a name the library does not hold or at the end.
fn kernel_games(kernel: &KernelGame, lines: &[Vec<String>], max: usize) -> (Vec<Vec<KS>>, Vec<Vec<String>>) {
    let (mut all, mut names) = (vec![], vec![]);
    for g in lines {
        let (mut s, mut states, mut played) = (kernel.initial(), vec![], vec![]);
        for nm in g.iter().take(max) {
            if kernel.outcome(&s).is_some() { break; }
            states.push(s.clone());
            let Some(m) = kernel.find_move(&s, nm) else { break };
            played.push(spec_term(&kernel.move_name(&s, &m)));
            s = kernel.apply(&s, &m);
        }
        all.push(states);
        names.push(played);
    }
    (all, names)
}

/// The same states for a module, made by the same moves.
fn module_games<T: Tr>(t: &mut T, kstates: &[Vec<KS>], names: &[Vec<String>]) -> Vec<Vec<T::S>> {
    let mut all = vec![];
    for (ks, nm) in kstates.iter().zip(names.iter()) {
        let (mut s, mut states) = (t.initial_t(), vec![]);
        for (i, _) in ks.iter().enumerate() {
            states.push(s.clone());
            if i < nm.len() { let m = t.term_t(&nm[i]).unwrap(); s = t.next_t(&s, m); }
        }
        all.push(states);
    }
    all
}

fn module_rows<T: Tr>(label: &str, kstates: &[Vec<KS>], names: &[Vec<String>], replays: usize, max: usize, out: &mut String) -> Vec<u64> {
    let mut t = T::make();
    let games = module_games(&mut t, kstates, names);
    let (mut ns, mut asks, mut legal) = (vec![0u64; max], vec![0u64; max], vec![0u64; max]);
    let mut buf = Vec::with_capacity(256);
    for r in 0..replays {
        for g in &games {
            for (ply, s) in g.iter().enumerate() {
                let t0 = Instant::now();
                t.moves_into_t(s, &mut buf);
                ns[ply] += t0.elapsed().as_nanos() as u64;
                asks[ply] += 1;
                if r == 0 { legal[ply] += buf.len() as u64; }
            }
        }
    }
    for ply in 0..max { if asks[ply] > 0 { writeln!(out, "T,{label},{ply},{},{}", asks[ply], ns[ply]).unwrap(); } }
    // the profile by piece, two walks
    t.profiling_t(true);
    let np = t.pieces_t().len();
    let mut acc = vec![(0u64, 0u64, 0u64); max * np];
    for _ in 0..2 {
        for g in &games {
            for (ply, s) in g.iter().enumerate() {
                t.reset_profile_t();
                t.moves_into_t(s, &mut buf);
                for (i, p) in t.pieces_t().iter().enumerate() { let a = &mut acc[ply * np + i]; a.0 += p.3; a.1 += p.4; a.2 += p.2; }
            }
        }
    }
    t.profiling_t(false);
    let pcs = t.pieces_t();
    for ply in 0..max { for i in 0..np { let a = acc[ply * np + i]; if a.2 > 0 { writeln!(out, "P,{label},{ply},{},{},{},{}", pcs[i].0.replace(',', ";"), a.0, a.1, a.2).unwrap(); } } }
    let (tn, ta): (u64, u64) = (ns.iter().sum(), asks.iter().sum());
    println!("OBS ply profile {label}: {ta} asks, {:.0} ns an ask on average, load {}", tn as f64 / ta.max(1) as f64, load_average());
    legal
}

fn run(path: &str, out_path: &str, replays: usize, max: usize) {
    let kernel = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<Vec<String>> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| l.split_whitespace().map(|x| x.to_string()).collect()).collect();
    let (kstates, names) = kernel_games(&kernel, &lines, max);
    let mut out = String::new();
    let (mut ns, mut asks, mut units, mut legal) = (vec![0u64; max], vec![0u64; max], vec![0u64; max], vec![0u64; max]);
    for r in 0..replays {
        for g in &kstates {
            for (ply, s) in g.iter().enumerate() {
                let t0 = Instant::now();
                let n = kernel.moves(s).len();
                ns[ply] += t0.elapsed().as_nanos() as u64;
                asks[ply] += 1;
                if r == 0 { units[ply] += kernel.units(s).len() as u64; legal[ply] += n as u64; }
            }
        }
    }
    for ply in 0..max { if asks[ply] > 0 { writeln!(out, "T,library,{ply},{},{}", asks[ply], ns[ply]).unwrap(); } }
    println!("OBS ply profile library: {} asks, {:.0} ns an ask on average, load {}", asks.iter().sum::<u64>(), ns.iter().sum::<u64>() as f64 / asks.iter().sum::<u64>().max(1) as f64, load_average());
    let lf = module_rows::<chess_fast::Game>("fast", &kstates, &names, replays, max, &mut out);
    let lc = module_rows::<chess_carry::Game>("carry", &kstates, &names, replays, max, &mut out);
    let ll = module_rows::<chess_lemma::Game>("lemma", &kstates, &names, replays, max, &mut out);
    assert!(lf == legal && lc == legal && ll == legal, "a module's count of allowed moves differs from the library's at some ply");
    for ply in 0..max { let n = asks[ply] / replays as u64; if n > 0 { writeln!(out, "S,{ply},{n},{},{}", legal[ply], units[ply]).unwrap(); } }
    std::fs::write(out_path, out).unwrap();
    println!("OBS ply profile: {} games, {} states, rows written to {out_path}; every module's count of allowed moves equals the library's at every ply", kstates.len(), kstates.iter().map(|g| g.len()).sum::<usize>());
}

#[test]
#[ignore = "a clock, run by hand through scripts/ply_profile_print.py on a quiet machine"]
fn the_time_of_the_allowed_moves_by_ply() {
    let path = std::env::var("CS_PLY_GAMES").expect("CS_PLY_GAMES names the file of games");
    let out = std::env::var("CS_PLY_OUT").expect("CS_PLY_OUT names the file of rows");
    run(&path, &out, env_usize("CS_PLY_REPLAYS", 20), env_usize("CS_PLY_MAX", 120));
}
