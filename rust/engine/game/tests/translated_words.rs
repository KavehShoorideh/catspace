//! PLAN 79.52 PART B (SYSTEM_SPEC 11, last block "The lower layers are made as fast as the outside library first; the facts stay ..."; ledger rows 55, 45; piece: rules): THE TWO PASSES `cones` AND `words` OF THE TRANSLATOR, PINNED.
//! Interface under test (written before the bodies, translated_common/mod.rs and the head of gdl_to_rust.py): `moves_into_t(s, out)` and `legal_into_t(s, out)` of every module, a state of a module made with the pass `words` held as the bits of words.
//! (1) STEP 1, ONLY WHAT WAS ASKED: after the profile is reset, the allowed moves without the end (`legal_into_t`) run no piece of the end or of the outcome, the allowed moves with the end (`moves_into_t`) run the end and not the outcome, the next state runs neither,
//!     the outcome runs both; a planted state (a position that ends by the lack of material while a move exists) is told apart by the two questions. (2) STEP 2, THE STATE AS WORDS: a state is made from facts and handed out as the same facts (a fact outside the sorts of
//!     its places is an error), two states with the same facts are equal, a state is `Copy`-sized words with no list. (3) NO ALLOCATION: after a first pass, `legal_into_t`, `moves_into_t` and `next_t` over a list of states make no allocation (printed as OBS lines).
//! (4) AGREEMENT: on seeded random games the module with the two passes gives the plain module's mover, allowed moves, end, outcome, facts and the next facts of the moves tried. This file names the games on purpose (a test file may).
#[macro_use]
mod translated_common;
use cs_game::rng::Rng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use translated_common::*;

struct Counting;
thread_local! { static ALLOCS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }     // the allocations of this thread (the tests run on several)
fn allocs() -> u64 { ALLOCS.with(|c| c.get()) }
thread_local! { static TRACE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }              // 1: print where this thread allocates (a debugging aid of the ignored test below)
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        if TRACE.with(|c| c.get()) == 1 {
            TRACE.with(|c| c.set(2));
            let bt = std::backtrace::Backtrace::force_capture().to_string();
            let frames: Vec<&str> = bt.lines().filter(|x| x.contains("generated") || x.contains("chess_fast") || x.trim_start().starts_with(|c: char| c.is_ascii_digit()) && x.contains("::")).filter(|x| !x.contains("alloc::") && !x.contains("std::") && !x.contains("core::")).take(4).collect();
            eprintln!("ALLOC {} bytes: {}", l.size(), frames.join(" <- "));
            TRACE.with(|c| c.set(1));
        }
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l) }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 { ALLOCS.with(|c| c.set(c.get() + 1)); System.realloc(p, l, n) }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

include!("generated/chess.rs");
include!("generated/chess_fast.rs");
include!("generated/atomic.rs");
include!("generated/atomic_fast.rs");
impl_tr!(chess);
impl_tr!(chess_fast, fresh);
impl_tr!(atomic);
impl_tr!(atomic_fast, fresh);

const SEED: u64 = 5252;

fn facts(v: &[(&str, &[&str])]) -> Vec<(String, Vec<String>)> { v.iter().map(|(r, a)| (r.to_string(), a.iter().map(|x| x.to_string()).collect())).collect() }

/// The calls of each piece since the profile was reset, as (name, calls) of the pieces that ran.
fn ran<T: Tr>(t: &T) -> Vec<&'static str> { t.pieces_t().iter().filter(|p| p.2 > 0).map(|p| p.0).collect() }

/// Random games of the module itself: the states reached (every `every`-th), by the seeded generator, as long as `plies`.
fn states<T: Tr>(t: &mut T, games: usize, plies: usize, every: usize) -> Vec<T::S> {
    let mut rng = Rng::new(SEED);
    let mut out = vec![];
    for _ in 0..games {
        let mut s = t.initial_t();
        for ply in 0..plies {
            let ms = t.moves_t(&s);
            if ms.is_empty() { break; }
            if ply % every == 0 { out.push(s.clone()); }
            let m = ms[rng.below(ms.len())];
            s = t.next_t(&s, m);
        }
    }
    out
}

/// A state that the module has not read yet: the next state of the k-th of the allowed moves of `s` (the last when k is too large).
fn fresh_of<T: Tr>(t: &mut T, s: &T::S, k: usize) -> T::S {
    let mut b = vec![];
    t.moves_into_t(s, &mut b);
    let m = b[k.min(b.len() - 1)];
    t.next_t(s, m)
}

fn only_what_was_asked<T: Tr>(game: &str) {
    let mut t = T::make();
    t.profiling_t(true);
    let list = states(&mut t, 4, 40, 3);
    let (end_pieces, outcome_pieces) = (["terminal", "insufficient", "two_units", "mating_material", "has_kind", "selfmate_material", "record"], ["goal"]);
    let mut buf = vec![];
    // the allowed moves without the end: none of the end's or the outcome's pieces, state after state (each state asked as a new one)
    for s in &list {
        let fresh = fresh_of(&mut t, s, 0);
        t.reset_profile_t();
        t.legal_into_t(&fresh, &mut buf);
        let r = ran(&t);
        for p in end_pieces.iter().chain(outcome_pieces.iter()) { assert!(!r.contains(p), "{game}: the allowed moves without the end ran the piece {p}"); }
        assert!(r.contains(&"legal"), "{game}: the allowed moves ran no `legal`");
        // the next state of a move: no piece of the end or of the outcome either
        t.reset_profile_t();
        let again = t.next_t(&fresh, buf[0]);
        let r = ran(&t);
        for p in end_pieces.iter().chain(outcome_pieces.iter()) { assert!(!r.contains(p), "{game}: the next state ran the piece {p}"); }
        let _ = again;
        // the allowed moves with the end: the end's pieces, not the outcome's
        let fresh2 = fresh_of(&mut t, s, usize::MAX);
        t.reset_profile_t();
        t.moves_into_t(&fresh2, &mut buf);
        let r = ran(&t);
        assert!(r.contains(&"terminal"), "{game}: the allowed moves with the end did not ask the end");
        assert!(!r.contains(&"goal"), "{game}: the allowed moves with the end ran the outcome's piece");
        // the outcome: both
        let fresh3 = fresh_of(&mut t, s, 1);
        t.reset_profile_t();
        let _ = t.outcome_t(&fresh3);
        let r = ran(&t);
        assert!(r.contains(&"terminal"), "{game}: the outcome did not ask the end");
    }
    println!("OBS 79.52 B only-what-was-asked {game}: {} states, the allowed moves without the end ran none of the pieces {:?} nor {:?}; the next state ran none of them; the allowed moves with the end ran the end and not the outcome", list.len(), end_pieces, outcome_pieces);
}

#[test]
fn chess_the_questions_run_only_the_pieces_they_need() { only_what_was_asked::<chess_fast::Game>("chess"); }
#[test]
fn atomic_the_questions_run_only_the_pieces_they_need() { only_what_was_asked::<atomic_fast::Game>("atomic"); }

#[test]
fn chess_a_planted_dead_position_has_no_allowed_move_with_the_end_and_has_them_without() {
    // two kings and one knight each side cannot be mated by force in the description's own criterion: the end holds by the lack of material while moves exist
    let mut t = chess_fast::Game::make();
    let f = facts(&[("cell", &["e", "1", "white", "king"]), ("cell", &["e", "8", "black", "king"]), ("control", &["white"]), ("halfmove", &["0"])]);
    let s = t.state_of_facts_t(&f).unwrap();
    let mut with_end = vec![];
    let mut without = vec![];
    t.moves_into_t(&s, &mut with_end);
    t.legal_into_t(&s, &mut without);
    assert!(t.terminal_t(&s), "two bare kings end the game");
    assert!(with_end.is_empty(), "the allowed moves with the end at an end: none");
    assert_eq!(without.len(), 5, "the white king on e1 has five squares and the `legal` rows are asked without the end");
    // and the plain module answers the first question as the module with the passes
    let mut p = chess::Game::make();
    let ps = p.state_of_facts_t(&f).unwrap();
    assert!(p.moves_t(&ps).is_empty());
    println!("OBS 79.52 B planted dead position: with the end {} moves, without {} moves, the plain module {} moves", with_end.len(), without.len(), p.moves_t(&ps).len());
}

fn words<T: Tr>(game: &str) {
    let mut t = T::make();
    let list = states(&mut t, 6, 60, 2);
    let mut seen = BTreeSet::new();
    for s in &list {
        // the facts handed out and made again give the same state
        let f = t.facts_of_t(s);
        let back = t.state_of_facts_t(&f).unwrap();
        assert!(&back == s, "{game}: a state made from its facts is the state");
        assert_eq!(t.facts_of_t(&back), f);
        // the same facts in another order, and a repeated fact, are the same state
        let mut g = f.clone();
        g.reverse();
        g.push(f[0].clone());
        assert!(&t.state_of_facts_t(&g).unwrap() == s, "{game}: the order of the facts or a repeated fact changes the state");
        seen.insert(format!("{f:?}"));
    }
    // a fact outside the sorts of its places is an error, never dropped
    let bad = facts(&[("cell", &["z", "1", "white", "king"])]);
    assert!(t.state_of_facts_t(&bad).is_err(), "{game}: a fact outside its sorts must be an error");
    let bad = facts(&[("nothing", &["e"])]);
    assert!(t.state_of_facts_t(&bad).is_err());
    // a state is a value of a few words: its size is printed
    println!("OBS 79.52 B words {game}: {} states, {} distinct, every one handed out as facts and made again equal; a state is {} bytes", list.len(), seen.len(), std::mem::size_of::<T::S>());
    assert!(std::mem::size_of::<T::S>() <= 512);
}

#[test]
fn chess_the_state_is_words_and_the_facts_stay() { words::<chess_fast::Game>("chess"); }
#[test]
fn atomic_the_state_is_words_and_the_facts_stay() { words::<atomic_fast::Game>("atomic"); }

fn no_allocation<T: Tr>(game: &str) {
    let mut t = T::make();
    let list = states(&mut t, 8, 60, 2);
    let mut buf = Vec::with_capacity(256);
    let mut sink = 0usize;
    // a first pass makes what the module keeps (the tables of compounds, the buffers)
    for _ in 0..2 { for s in &list { t.legal_into_t(s, &mut buf); sink += buf.len(); let c = t.next_t(s, buf[0]); sink += std::mem::size_of_val(&c); t.moves_into_t(s, &mut buf); sink += buf.len(); } }
    let a0 = allocs();
    for s in &list { t.legal_into_t(s, &mut buf); sink += buf.len(); }
    let a1 = allocs();
    for s in &list { t.moves_into_t(s, &mut buf); sink += buf.len(); }
    let a2 = allocs();
    for s in &list { t.legal_into_t(s, &mut buf); let c = t.next_t(s, buf[0]); sink += std::mem::size_of_val(&c); }
    let a3 = allocs();
    println!("OBS 79.52 B allocations {game}: over {} states, allowed moves without the end {} allocations, with the end {}, allowed moves and one next state {} (checksum {sink})", list.len(), a1 - a0, a2 - a1, a3 - a2);
    assert_eq!(a1 - a0, 0, "{game}: the allowed moves without the end allocate");
    assert_eq!(a2 - a1, 0, "{game}: the allowed moves with the end allocate");
    assert_eq!(a3 - a2, 0, "{game}: the allowed moves and the next state allocate");
}

#[test]
fn chess_the_allowed_moves_and_the_next_state_allocate_nothing() { no_allocation::<chess_fast::Game>("chess"); }
#[test]
fn atomic_the_allowed_moves_and_the_next_state_allocate_nothing() { no_allocation::<atomic_fast::Game>("atomic"); }

fn agree<A: Tr, B: Tr>(game: &str, games: usize) {
    let (mut a, mut b) = (A::make(), B::make());
    let mut rng = Rng::new(SEED + 1);
    let (mut states, mut moves, mut nexts) = (0usize, 0usize, 0usize);
    let mut buf = vec![];
    for _ in 0..games {
        let (mut sa, mut sb) = (a.initial_t(), b.initial_t());
        for _ply in 0..120 {
            states += 1;
            assert_eq!(a.facts_of_t(&sa), b.facts_of_t(&sb), "{game}: the facts");
            assert_eq!(a.outcome_t(&sa), b.outcome_t(&sb), "{game}: the outcome");
            assert_eq!(a.terminal_t(&sa), b.terminal_t(&sb));
            assert_eq!(a.mover_t(&sa), b.mover_t(&sb), "{game}: the mover");
            let names = |t: &A, v: &[u32]| v.iter().map(|m| t.render_t(*m)).collect::<BTreeSet<_>>();
            let ma = a.moves_t(&sa);
            let mb: BTreeSet<String> = b.moves_t(&sb).iter().map(|m| b.render_t(*m)).collect();
            assert_eq!(names(&a, &ma), mb, "{game}: the allowed moves");
            a.moves_into_t(&sa, &mut buf);
            assert_eq!(names(&a, &buf), mb, "{game}: moves_into");
            if ma.is_empty() { break; }
            a.legal_into_t(&sa, &mut buf);
            assert_eq!(names(&a, &buf), mb, "{game}: legal_into at a state that does not end");
            moves += ma.len();
            // the next facts of a few moves, then the move played
            let pick = rng.below(ma.len());
            for k in [0usize, ma.len() - 1, pick] {
                let name = a.render_t(ma[k]);
                let mbm = b.term_t(&name).unwrap();
                let (na, nb) = (a.next_t(&sa, ma[k]), b.next_t(&sb, mbm));
                assert_eq!(a.facts_of_t(&na), b.facts_of_t(&nb), "{game}: the next facts after {name}");
                nexts += 1;
            }
            let name = a.render_t(ma[pick]);
            let mbm = b.term_t(&name).unwrap();
            sa = a.next_t(&sa, ma[pick]);
            sb = b.next_t(&sb, mbm);
        }
    }
    println!("OBS 79.52 B agree {game}: {games} seeded games, {states} states, {moves} allowed moves, {nexts} next states compared with the plain module, 0 disagreements");
}

#[test]
fn chess_the_module_with_the_passes_agrees_with_the_plain_module() { agree::<chess_fast::Game, chess::Game>("chess", 12); }
#[test]
fn atomic_the_module_with_the_passes_agrees_with_the_plain_module() { agree::<atomic_fast::Game, atomic::Game>("atomic", 12); }

#[test]
#[ignore = "a debugging aid: where the allowed moves with the end allocate (RUST_BACKTRACE not needed)"]
fn where_the_allocations_are() {
    let mut t = chess_fast::Game::make();
    let list = states(&mut t, 2, 30, 3);
    let mut buf = Vec::with_capacity(256);
    for _ in 0..2 { for s in &list { t.legal_into_t(s, &mut buf); t.moves_into_t(s, &mut buf); } }
    for s in list.iter().take(3) {
        t.legal_into_t(s, &mut buf);
        TRACE.with(|c| c.set(1));
        t.moves_into_t(s, &mut buf);
        TRACE.with(|c| c.set(0));
        eprintln!("-----");
    }
}
