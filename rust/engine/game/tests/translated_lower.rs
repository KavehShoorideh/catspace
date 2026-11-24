//! PLAN 79.47 PART D (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy", "The crux of layer 2 is a translator"; piece: rules; ledger rows 44, 45): THE LOWER LINK RAISED. THIS TEXT IS THE INTERFACE, written before the body.
//! The chain is description -> evaluator -> translated module (every pass) -> kernel. The kernel is held to the module on millions of states (translated_kernel.rs); the module is held to the evaluator on thousands (translated_agree.rs, the certificate's 20 seeded games and the fixtures).
//! This file holds the module of each of the seven games to the evaluator on about two thousand seeded games, so that the link under the kernel's is counted in the same order of size.
//! WHAT IT TAKES. A game (its kept every-pass module, tests/generated/<game>_fast.rs, and the description text) and a corpus: seeds `LOW_SEED0 + k`, `LOW_GAMES_A_SEED` games each, from the kernel's start (the kernel only chooses the walk: its moves sorted by name, one picked by the seeded generator, as the
//! certificate walks). WHAT IT COMPARES, at every state, by `translated_common::compare_at`: whose turn, the allowed moves as a set of terms, whether ended, the result, the facts; and the next state of the move played (the facts of the next state, compared at the next state). Counted: seeds, games, states, allowed moves, next states.
//! A disagreement is counted and kept whole (the state's line of moves and the message of the comparison); the seed's corpus goes on from the next seed. Cores: threads, each with its own module and evaluator; the corpus is a function of the seed list, never of the cores.
//! WHO CALLS IT. `python -m catspace.incubator.analyst.discovery.kernel_translated lower <game>` runs `<game>_the_lower_link` (ignored: it is long and uses every core) and keeps the counts in the game's record, in its chain. ENVIRONMENT: CS_LOW_SEEDS (default 100), CS_LOW_GAMES (games a seed, default 20), CS_LOW_CORES, CS_LOW_OUT (the counts file, JSON).
//! The evaluator answers about 330 positions a second on one core; the first run of a game is made small (CS_LOW_SEEDS=2) to measure that and the corpus is sized to about five minutes a game.
mod translated_common;
use cs_game::kernel::KernelGame;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use translated_common::*;

pub const LOW_SEED0: u64 = 5000;

include!("generated/chess_fast.rs");
impl_tr!(chess_fast, passes);
include!("generated/atomic_fast.rs");
impl_tr!(atomic_fast, passes);
include!("generated/3check_fast.rs");
impl_tr!(threecheck_fast, passes);
include!("generated/kingofthehill_fast.rs");
impl_tr!(kingofthehill_fast, passes);
include!("generated/antichess_fast.rs");
impl_tr!(antichess_fast, passes);
include!("generated/chess960_fast.rs");
impl_tr!(chess960_fast, passes);
include!("generated/horde_fast.rs");
impl_tr!(horde_fast, passes);

#[derive(Default, Clone)]
struct Low { seeds: usize, games: usize, states: usize, moves: usize, nexts: usize, ended: usize, disagreements: usize, first: Vec<String> }

fn env_usize(name: &str, default: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default) }

fn esc(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() { match ch { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), c if (c as u32) < 32 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o.push('"');
    o
}

fn lower<T: Tr + 'static>(game: &str, text: &str) {
    let cores = env_usize("CS_LOW_CORES", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let (seeds, games) = (env_usize("CS_LOW_SEEDS", 100), env_usize("CS_LOW_GAMES", 20));
    let kernel = KernelGame::new(game, 518).expect("the kernel");
    let next = AtomicUsize::new(0);
    let progress = AtomicU64::new(0);
    let done = std::sync::atomic::AtomicBool::new(false);
    let results: Mutex<Vec<(usize, Low)>> = Mutex::new(vec![]);
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        let mut hs = vec![];
        for _ in 0..cores {
            hs.push(std::thread::Builder::new().stack_size(64 << 20).spawn_scoped(s, || {
                let mut t = T::make();
                let ev = evaluator(text);
                loop {
                    let k = next.fetch_add(1, Ordering::SeqCst);
                    if k >= seeds { break; }
                    let mut tally = Tally::default();
                    let mut rels = BTreeMap::new();
                    let r = coplay(&kernel, &mut t, &ev, None, games, LOW_SEED0 + k as u64, 0, 0, &mut tally, &mut rels);
                    // the next state of the move played is compared at the state it leads to (its facts), so the next states compared are the states of the games but their first ones
                    let mut c = Low { seeds: 1, games, states: tally.positions, moves: tally.moves, nexts: tally.positions.saturating_sub(games), ended: tally.terminal, ..Default::default() };
                    if let Err(e) = r { c.disagreements = 1; c.first.push(format!("seed {}: {e}", LOW_SEED0 + k as u64)); }
                    progress.fetch_add(tally.positions as u64, Ordering::Relaxed);
                    results.lock().unwrap().push((k, c));
                }
            }).unwrap());
        }
        s.spawn(|| {
            let mut tick = 0;
            while !done.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(500));
                tick += 1;
                if tick % 60 == 0 { println!("OBS 79.47 D progress {game}: {} s, {} of {seeds} seeds begun, {} states, load {}", t0.elapsed().as_secs(), next.load(Ordering::SeqCst).min(seeds), progress.load(Ordering::Relaxed), load_average()); }
            }
        });
        for h in hs { h.join().unwrap(); }
        done.store(true, Ordering::SeqCst);
    });
    let mut all = results.into_inner().unwrap();
    all.sort_by_key(|(k, _)| *k);
    let mut c = Low::default();
    for (_, x) in &all {
        c.seeds += x.seeds; c.games += x.games; c.states += x.states; c.moves += x.moves; c.nexts += x.nexts; c.ended += x.ended; c.disagreements += x.disagreements;
        if c.first.len() < 3 { c.first.extend(x.first.iter().cloned()); }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("OBS 79.47 D lower {game}: {} seeds, {} games, {} states, {} allowed moves compared, {} next states compared, {} ended, disagreements {} ({secs:.1} s, {cores} cores, {:.0} states a second)", c.seeds, c.games, c.states, c.moves, c.nexts, c.ended, c.disagreements, c.states as f64 / secs.max(1e-9));
    for f in &c.first { println!("FINDING 79.47 D {game}: {f}"); }
    if let Ok(p) = std::env::var("CS_LOW_OUT") {
        let first = c.first.iter().map(|f| esc(f)).collect::<Vec<_>>().join(", ");
        std::fs::write(&p, format!("{{\"game\": {}, \"seeds\": {}, \"seed0\": {LOW_SEED0}, \"games_a_seed\": {games}, \"games\": {}, \"states\": {}, \"moves\": {}, \"nexts\": {}, \"ended\": {}, \"disagreements\": {}, \"first\": [{first}], \"seconds\": {secs:.1}, \"cores\": {cores}}}\n",
            esc(game), c.seeds, c.games, c.states, c.moves, c.nexts, c.ended, c.disagreements)).expect("write the counts");
    }
    assert_eq!(c.disagreements, 0, "{game}: the module and the evaluator disagree: {:?}", c.first);
}

#[test]
#[ignore = "the lower link: all cores, a few minutes, run by hand (discovery/kernel_translated.py lower chess); the machine is shared"]
fn chess_the_lower_link_the_module_and_the_evaluator_on_seeded_games() { lower::<chess_fast::Game>("chess", &description(None)); }
#[test]
#[ignore = "the lower link: all cores, a few minutes, run by hand (discovery/kernel_translated.py lower atomic); the machine is shared"]
fn atomic_the_lower_link_the_module_and_the_evaluator_on_seeded_games() { lower::<atomic_fast::Game>("atomic", &description(Some("variants/atomic.gdl"))); }
#[test]
#[ignore = "the lower link: all cores, a few minutes, run by hand (discovery/kernel_translated.py lower 3check); the machine is shared"]
fn threecheck_the_lower_link_the_module_and_the_evaluator_on_seeded_games() { lower::<threecheck_fast::Game>("3check", &description(Some("variants/3check.gdl"))); }
#[test]
#[ignore = "the lower link: all cores, a few minutes, run by hand (discovery/kernel_translated.py lower kingofthehill); the machine is shared"]
fn kingofthehill_the_lower_link_the_module_and_the_evaluator_on_seeded_games() { lower::<kingofthehill_fast::Game>("kingofthehill", &description(Some("variants/kingofthehill.gdl"))); }
#[test]
#[ignore = "the lower link: all cores, a few minutes, run by hand (discovery/kernel_translated.py lower antichess); the machine is shared"]
fn antichess_the_lower_link_the_module_and_the_evaluator_on_seeded_games() { lower::<antichess_fast::Game>("antichess", &description(Some("variants/antichess.gdl"))); }
#[test]
#[ignore = "the lower link: all cores, a few minutes, run by hand (discovery/kernel_translated.py lower chess960); the machine is shared"]
fn chess960_the_lower_link_the_module_and_the_evaluator_on_seeded_games() { lower::<chess960_fast::Game>("chess960", &description(Some("variants/chess960.gdl"))); }
#[test]
#[ignore = "the lower link: all cores, a few minutes, run by hand (discovery/kernel_translated.py lower horde); the machine is shared"]
fn horde_the_lower_link_the_module_and_the_evaluator_on_seeded_games() { lower::<horde_fast::Game>("horde", &description(Some("variants/horde.gdl"))); }
