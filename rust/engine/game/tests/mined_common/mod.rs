//! PLAN 79.59 (SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules): WHAT THE TWO TEST FILES OF THE MINER SHARE (translated_mined.rs: the explanations; translated_judge.rs: the judge). Segments: runs of consecutive states of a game, from the games of a file (human games, the moves by the outside library's names; no player is read)
//! or from seeded random walks. A test helper, not part of the crate's sources; a test file may name the games.
#![allow(dead_code)]
use cs_game::kernel::KernelGame;
use cs_game::rng::Rng;
use cs_game::{Fact, Game};
use crate::variants_common::kernel;

pub const SEED: u64 = 5757;
pub const FREE: u32 = u32::MAX;
pub type State = <KernelGame as Game>::State;

pub fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

pub fn lines_of(path: &str, n: usize) -> Vec<Vec<String>> { std::fs::read_to_string(path).unwrap().lines().take(n).map(|l| l.split_whitespace().map(|w| w.to_string()).collect()).collect() }

/// The games of a file as states, from their start (`max` plies at most).
pub fn states_of_lines(game: &str, lines: &[Vec<String>], max: usize) -> Vec<Vec<State>> {
    let k = kernel(game);
    let mut out = vec![];
    for l in lines {
        let mut s = k.initial();
        let mut v = vec![s.clone()];
        for nm in l.iter().take(max) {
            if k.outcome(&s).is_some() { break; }
            let Some(m) = k.find_move(&s, nm) else { break };
            s = k.apply(&s, &m);
            v.push(s.clone());
        }
        out.push(v);
    }
    out
}

/// Seeded random walks of a game (the moves taken in the order of their names, as the other measures do).
pub fn seeded_states(k: &KernelGame, n: usize, plies: usize) -> Vec<Vec<State>> {
    let mut rng = Rng::new(SEED);
    let mut out = vec![];
    for _ in 0..n {
        let mut s = k.initial();
        let mut states = vec![s.clone()];
        for _ in 0..plies {
            let mv = k.moves(&s);
            if mv.is_empty() || k.outcome(&s).is_some() { break; }
            let mut sorted: Vec<(String, usize)> = mv.iter().enumerate().map(|(i, m)| (k.move_name(&s, m), i)).collect();
            sorted.sort();
            let (_, i) = sorted[rng.below(sorted.len())].clone();
            s = k.apply(&s, &mv[i]);
            states.push(s.clone());
        }
        out.push(states);
    }
    out
}

pub fn pairs(facts: &[Fact]) -> Vec<(String, Vec<String>)> { facts.iter().map(|x| (x.relation.clone(), x.args.clone())).collect() }

