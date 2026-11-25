//! PLAN 79.54 PART B (SYSTEM_SPEC 11, last block, (ix) "answers carried from the last state"; ledger rows 55, 45; piece: rules): THE PASS `carry` PINNED. Interface under test (written before the body: the head of gdl_to_rust.py, pass (25), and translated_common/mod.rs):
//! `carried_t(s)` / `carried_valid_t()` of the modules `<game>_carry.rs` (every pass, `carry` among them) beside the modules `<game>_fast.rs` (every pass but `carry`), which answer every question as a state not read before.
//! WHAT IT HOLDS, for each of the seven games, over seeded random games: the module with the pass gives the answers of the module without it (the end, the mover, the outcome, the allowed moves as terms, the facts of the state) at every state, whatever the order the states are asked in:
//!   (1) ALONG A GAME (each state after the last), (2) ACROSS SIBLINGS (a parent, its children one after another, the parent again), (3) JUMPS to an unrelated state (the states of all the games in a seeded random order), (4) the same state asked again by questions that need other
//!   parts of the cones (the order of the questions turns with the state), and EVERY RELATION of the kept evaluation (`carried_t`) against the rows the other module makes of the same state as a state not read before: equal, relation by relation, as text.
//! It prints how many pieces ran for a question with and without the pass (the profile's calls), so that the pass is seen to carry. This file names the games on purpose (a test file may).
#[macro_use]
mod translated_common;
use cs_game::rng::Rng;
use std::collections::{BTreeMap, BTreeSet};
use translated_common::*;

include!("generated/chess_fast.rs");
include!("generated/chess_carry.rs");
include!("generated/atomic_fast.rs");
include!("generated/atomic_carry.rs");
include!("generated/3check_fast.rs");
include!("generated/3check_carry.rs");
include!("generated/kingofthehill_fast.rs");
include!("generated/kingofthehill_carry.rs");
include!("generated/antichess_fast.rs");
include!("generated/antichess_carry.rs");
include!("generated/chess960_fast.rs");
include!("generated/chess960_carry.rs");
include!("generated/horde_fast.rs");
include!("generated/horde_carry.rs");
impl_tr!(chess_fast, carry);
impl_tr!(chess_carry, carry);
impl_tr!(atomic_fast, carry);
impl_tr!(atomic_carry, carry);
impl_tr!(threecheck_fast, carry);
impl_tr!(threecheck_carry, carry);
impl_tr!(kingofthehill_fast, carry);
impl_tr!(kingofthehill_carry, carry);
impl_tr!(antichess_fast, carry);
impl_tr!(antichess_carry, carry);
impl_tr!(chess960_fast, carry);
impl_tr!(chess960_carry, carry);
impl_tr!(horde_fast, carry);
impl_tr!(horde_carry, carry);

const SEED: u64 = 5454;
fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

type Rels = BTreeMap<&'static str, BTreeSet<Vec<String>>>;
fn rels<T: Tr>(t: &T, v: Vec<(&'static str, Vec<Vec<u32>>)>) -> Rels { v.into_iter().map(|(n, r)| (n, r.into_iter().map(|row| row.iter().map(|x| t.render_t(*x)).collect()).collect())).collect() }
fn calls<T: Tr>(t: &T) -> u64 { t.pieces_t().iter().filter(|p| !p.1).map(|p| p.2).sum() }

#[derive(Default)]
struct Tally { states: usize, relations: usize, rows: usize, calls_carry: u64, calls_plain: u64, questions: usize }

/// One state: the same questions asked of both modules, the order of the questions turned by `turn`; every answer and every relation compared.
fn check<C: Tr, F: Tr>(c: &mut C, f: &mut F, sc: &C::S, label: &str, turn: usize, tally: &mut Tally) {
    let facts = c.facts_of_t(sc);
    let sf = f.state_of_facts_t(&facts).unwrap();
    let names = |t: &dyn Fn(u32) -> String, v: &[u32]| v.iter().map(|m| t(*m)).collect::<BTreeSet<_>>();
    let mut buf = Vec::with_capacity(256);
    let before_c = calls(c);
    c.moves_into_t(sc, &mut buf);                     // the first question of a state the module has not read: its pieces run are counted
    let after_c = calls(c);
    let before_f = calls(f);
    f.moves_into_t(&sf, &mut buf);
    let after_f = calls(f);
    let (term_c, mover_c, out_c, mv_c);
    match turn % 3 {
        0 => { mv_c = c.moves_t(sc); term_c = c.terminal_t(sc); mover_c = c.mover_t(sc); out_c = c.outcome_t(sc); }
        1 => { out_c = c.outcome_t(sc); term_c = c.terminal_t(sc); mover_c = c.mover_t(sc); mv_c = c.moves_t(sc); }
        _ => { term_c = c.terminal_t(sc); mv_c = c.moves_t(sc); out_c = c.outcome_t(sc); mover_c = c.mover_t(sc); }
    }
    let (term_f, mover_f, out_f, mv_f) = (f.terminal_t(&sf), f.mover_t(&sf), f.outcome_t(&sf), f.moves_t(&sf));
    assert_eq!(term_c, term_f, "{label}: the end");
    assert_eq!(mover_c, mover_f, "{label}: the mover");
    assert_eq!(out_c, out_f, "{label}: the outcome");
    assert_eq!(names(&|m| c.render_t(m), &mv_c), names(&|m| f.render_t(m), &mv_f), "{label}: the allowed moves");
    // every relation of the kept evaluation against the other module's rows of the same state
    let (vc, vf) = (c.carried_t(sc), f.carried_t(&sf));
    let (rc, rf) = (rels(c, vc), rels(f, vf));
    for (n, rows) in &rc {
        let other = rf.get(n).unwrap_or_else(|| panic!("{label}: the relation {n} is in one module and not in the other"));
        assert_eq!(rows, other, "{label}: the relation {n} (kept {} rows, made again {})", rows.len(), other.len());
        tally.rows += rows.len();
    }
    for (n, rows) in &rf { if !rc.contains_key(n) { assert!(rows.is_empty(), "{label}: the relation {n} has rows and the piece that makes it is not valid"); } }
    tally.relations += rc.len();
    tally.states += 1;
    tally.questions += 1;
    tally.calls_carry += after_c - before_c;
    tally.calls_plain += after_f - before_f;
}

fn run<C: Tr, F: Tr>(game: &str) {
    let (games, plies) = (env_usize("CS_CARRY_GAMES", 8), env_usize("CS_CARRY_PLIES", 70));
    let mut c = C::make();
    let mut f = F::make();
    c.profiling_t(true);
    f.profiling_t(true);
    let mut rng = Rng::new(SEED);
    let mut tally = Tally::default();
    let mut all: Vec<C::S> = vec![];
    // (1) along games, and (2) across siblings
    let (mut along, mut sib, mut pass) = (Tally::default(), Tally::default(), Tally::default());
    for g in 0..games {
        let mut s = c.initial_t();
        for ply in 0..plies {
            check(&mut c, &mut f, &s, &format!("{game} game {g} ply {ply} along"), ply + g, &mut along);
            if let Some(ps) = passed(&mut c, &s, "control") {          // a state, the state passed (made here from the facts), the state again
                check(&mut c, &mut f, &ps, &format!("{game} game {g} ply {ply} passed"), ply, &mut pass);
                check(&mut c, &mut f, &s, &format!("{game} game {g} ply {ply} back from the pass"), ply + 1, &mut pass);
            }
            let ms = c.moves_t(&s);
            if ms.is_empty() { break; }
            all.push(s.clone());
            if ply % 3 == 0 {
                for (i, m) in ms.iter().take(4).enumerate() {
                    let ch = c.next_t(&s, *m);
                    check(&mut c, &mut f, &ch, &format!("{game} game {g} ply {ply} child {i}"), i + ply, &mut sib);
                }
                check(&mut c, &mut f, &s, &format!("{game} game {g} ply {ply} parent again"), ply + 1, &mut sib);
            }
            let m = ms[rng.below(ms.len())];
            s = c.next_t(&s, m);
        }
    }
    // (3) jumps: every state met, in a seeded random order
    let mut order: Vec<usize> = (0..all.len()).collect();
    for i in (1..order.len()).rev() { order.swap(i, rng.below(i + 1)); }
    let mut jump = Tally::default();
    for (k, i) in order.iter().enumerate() { check(&mut c, &mut f, &all[*i], &format!("{game} jump {k}"), k, &mut jump); }
    for t in [&along, &sib, &jump, &pass] { tally.states += t.states; tally.relations += t.relations; tally.rows += t.rows; }
    println!("OBS 79.54 B {game} carry equals no-carry: {} states ({} along a game, {} across siblings, {} jumps, {} passed and back), {} relations of valid pieces compared ({} rows) with the rows made again, 0 differ", tally.states, along.states, sib.states, jump.states, pass.states, tally.relations, tally.rows);
    for (kind, t) in [("along", &along), ("siblings", &sib), ("jumps", &jump), ("pass and back", &pass)] {
        println!("OBS 79.54 B {game} {kind}: pieces run by the first question of a state (the allowed moves with the end; profile calls) with the pass {:.1}, without {:.1}", t.calls_carry as f64 / t.questions.max(1) as f64, t.calls_plain as f64 / t.questions.max(1) as f64);
    }
}

#[test]
fn chess_carry_equals_no_carry() { run::<chess_carry::Game, chess_fast::Game>("chess"); }
#[test]
fn atomic_carry_equals_no_carry() { run::<atomic_carry::Game, atomic_fast::Game>("atomic"); }
#[test]
fn threecheck_carry_equals_no_carry() { run::<threecheck_carry::Game, threecheck_fast::Game>("3check"); }
#[test]
fn kingofthehill_carry_equals_no_carry() { run::<kingofthehill_carry::Game, kingofthehill_fast::Game>("kingofthehill"); }
#[test]
fn antichess_carry_equals_no_carry() { run::<antichess_carry::Game, antichess_fast::Game>("antichess"); }
#[test]
fn chess960_carry_equals_no_carry() { run::<chess960_carry::Game, chess960_fast::Game>("chess960"); }
#[test]
fn horde_carry_equals_no_carry() { run::<horde_carry::Game, horde_fast::Game>("horde"); }

#[test]
fn a_module_started_again_reads_in_full() {
    // `restart` drops the kept evaluation: the next state read is read in full (every piece of its cone run), and a state asked after a restart answers as the module without the pass
    let mut c = chess_carry::Game::make();
    let mut f = chess_fast::Game::make();
    let s = c.initial_t();
    let _ = c.moves_t(&s);
    assert!(c.carried_valid_t() > 0, "a state read leaves valid pieces");
    let facts = c.facts_of_t(&s);
    c.restart_t(&facts).unwrap();
    let s2 = c.initial_t();
    let sf = f.initial_t();
    let (a, b) = (c.moves_t(&s2), f.moves_t(&sf));
    assert_eq!(a.iter().map(|m| c.render_t(*m)).collect::<BTreeSet<_>>(), b.iter().map(|m| f.render_t(*m)).collect::<BTreeSet<_>>());
}
