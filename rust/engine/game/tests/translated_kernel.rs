//! PLAN 79.46 PART A (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy", "The crux of layer 2 is a translator"; piece: rules; ledger rows 44, 45):
//! THE KERNEL HELD TO THE TRANSLATED MODULE. THIS TEXT IS THE INTERFACE, written before the body.
//!
//! THE CHAIN. description -> evaluator (the reader of the description) -> translated module (held to the evaluator piece by piece, translated_agree.rs, translated_pieces.rs) -> kernel (this file holds it to the translated
//! module). The evaluator answers about 330 positions a second, the translated module with every pass about 96,000 (plan 79.43 READ), so the kernel can be held to the description on a corpus hundreds of times larger than the
//! existing certificate's (cs_game::certify, 20 seeded games), by way of the module that is itself held to the evaluator.
//!
//! WHAT IT TAKES. A game (a kernel of cs_game::kernel::KernelGame by its name, and the kept every-pass translated module of the same description, tests/generated/<game>_fast.rs, written by discovery/gdl_to_rust.py and only READ here),
//! and a corpus: seeded random games. THE WALK is the certificate's (cs_game::certify::coplay): at each state the kernel's moves sorted by its move names, one picked by a seeded `Rng` (one generator for all games of a seed). A
//! corpus is `games` games from the kernel's start with `seed`, and for the small run also `starts_games` games from each position of the game's list rules/variants/<game>.starts (the certificate's corpus exactly).
//! WHAT IT COMPARES, at EVERY state, the two sides side by side (a state of the kernel and a state of the module are different types; they meet as the description's fluents, the bridge the certificate's own start uses,
//! `KernelGame::state_facts` of outside.rs, against the module's `facts_of`; moves meet as the description's terms, `spec_term(kernel name)` against the module's `render`):
//!   (1) whose turn (the role the mover is, or none), (2) the allowed moves as a SET of terms, (3) the next state after EVERY allowed move (both sides' fluents after the move, not only the move played), (4) whether the game has ended,
//!   (5) the result (the goal of each role) where it has. Counted: games, states, allowed moves compared, next states compared, ended states, results compared.
//! A DISAGREEMENT is a finding, never hidden and never a stop: it is counted under its CAUSE (the answer, the fluents or the move class that differ, which side has the extra), the first few of each are kept WHOLE (the state as
//! fluents and as the kernel's text, the line played to it, the answer of each side) and, for those kept, the evaluator is asked at the same state (`translated_common::evaluator_at`) and its agreeing side named. After a state
//! disagreement the module's state is begun again from the kernel's fluents so one cause is not counted over a whole game. A state the module cannot read from the kernel's fluents is LEFT OUT and counted under its class.
//! NOTHING IS WEAKENED to reach zero: no state is skipped, no answer is left out of the comparison.
//!
//! WHO CALLS IT. `cargo test --release --offline -p cs_game --test translated_kernel` (the small tests, a minute at most) and, by hand, `-- --ignored` (the large run, all cores by threads, a fixed list of seeds, about five minutes a
//! game; ignored for that cause: it is long, the machine is shared, the record it writes is read by the gate). The Python side (discovery/kernel_translated.py: `run <game>`, `status`) runs the large test, reads what it wrote
//! and keeps the record; tests/test_plan79_46.py is the gate.
//! WHAT IT WRITES. When the environment names `CS_KT_OUT`, the run's counts as a JSON object at that path (the format is in discovery/kernel_translated.py: games, states, moves, nexts, ended, results, plies, left_out, causes,
//! examples, seconds, cores, seeds); the Python side adds the hashes of what the answers depend on and the date, and writes the record beside the certificates.
//! ENVIRONMENT of the large run: CS_KT_CORES (threads; default all), CS_KT_SEEDS (how many seeds of the fixed list `LARGE_SEED0 + 0..`), CS_KT_GAMES (games a seed), CS_KT_OUT (the counts file).
//! This file names the games on purpose (a test file may); the layer crates do not.
mod translated_common;
use cs_game::kernel::KernelGame;
use std::collections::BTreeMap;
use translated_common::*;

include!("generated/chess_fast.rs");
include!("generated/atomic_fast.rs");
impl_tr!(chess_fast, passes);
impl_tr!(atomic_fast, passes);

/// The seed of the certificate (discovery/certify.py SEED) and its games, as many as the certificate plays from the kernel's own start and from each listed position.
pub const SEED: u64 = 77;
pub const GAMES: usize = 20;
pub const STARTS_GAMES: usize = 4;
/// The fixed list of seeds of the large run: `LARGE_SEED0 + k` for k below CS_KT_SEEDS; the corpus is a function of the list and of the games a seed, never of the cores.
pub const LARGE_SEED0: u64 = 1000;

/// One disagreement kept whole.
#[derive(Clone, Debug, Default)]
pub struct Finding { pub cause: String, pub seed: u64, pub game: usize, pub line: Vec<String>, pub state: String, pub kernel: String, pub translated: String, pub evaluator: String }

/// What a run counted. `causes`: disagreements by cause; `left_out`: states not compared, by class; `findings`: the first few of each cause.
#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub games: usize, pub plies: usize, pub states: usize, pub moves: usize, pub nexts: usize, pub ended: usize, pub results: usize,
    pub left_out: BTreeMap<String, usize>, pub causes: BTreeMap<String, usize>, pub findings: Vec<Finding>,
}

/// The corpus of one seed: `games` games (the kernel's own start, or `start` fluents-of-a-position text when given), compared at every state. Takes the translated module `t`, the description's text (for the evaluator's verdict), the kernel,
/// the seed. Adds to `counts`. `progress` is bumped by the states done (the large run's reporter reads it).
pub fn walk<T: Tr>(_t: &mut T, _text: &str, _kernel: &KernelGame, _games: usize, _seed: u64, _counts: &mut Counts, _progress: Option<&std::sync::atomic::AtomicU64>) { todo!("body: plan 79.46 part A") }

/// The certificate's corpus: GAMES from the kernel's start and STARTS_GAMES from each position listed for the game; the count of states is what the certificate's `positions` is.
pub fn small_run<T: Tr>(_variant: &str, _text: &str) -> Counts { todo!("body: plan 79.46 part A") }

/// The large corpus on `cores` threads: seeds LARGE_SEED0.., `games` games each, drawn off one counter; counts merged in seed order; progress printed at least once a minute.
pub fn large_run<T: Tr>(_variant: &str, _text: &str, _seeds: usize, _games: usize, _cores: usize) -> Counts { todo!("body: plan 79.46 part A") }

/// The counts as the JSON text the Python side reads.
pub fn counts_json(_variant: &str, _kind: &str, _c: &Counts, _seconds: f64, _cores: usize, _seeds: usize, _games: usize) -> String { todo!("body: plan 79.46 part B") }
