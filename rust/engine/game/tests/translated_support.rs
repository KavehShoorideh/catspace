//! PLAN 79.56 PART C (SYSTEM_SPEC 11, blocks "Layer 2 unlocked for lemmas" and "The support filter"; ledger rows 55, 56, 57; piece: rules; LAYER 2): THE PASS `support` PINNED. Interface under test: pass (26) in the head of
//! catspace/incubator/analyst/discovery/gdl_to_rust.py; the modules `<game>_support.rs` (every pass but `carry`, and `support`) beside the modules `<game>_fast.rs` (every pass but `carry`).
//! WHAT IT HOLDS, for each game where the shape is found: at every state of the seeded random games used for the pass carry (the same seed and sizes: CS_CARRY_GAMES 8, CS_CARRY_PLIES 70, along a game, across siblings, jumps to an unrelated
//! state) the module with the pass gives the answers of the module without it (the end, the mover, the outcome, the allowed moves as terms) and EVERY RELATION of its reading of the state (`all_relations`) is the same set of rows. One test
//! PLANTS a wrong reference (`sp_plant`: the answer a reference keeps is turned round) and shows the same comparison catches it. Counters (on under the profile switch) are printed as OBS lines: keys, tests run, keys answered from a reference.
//! This file names the games on purpose (a test file may).
#[macro_use]
mod translated_common;
use cs_game::rng::Rng;
use std::collections::{BTreeMap, BTreeSet};
use translated_common::*;

include!("generated/chess_fast.rs");
include!("generated/chess_support.rs");
include!("generated/atomic_fast.rs");
include!("generated/atomic_support.rs");
include!("generated/3check_fast.rs");
include!("generated/3check_support.rs");
include!("generated/kingofthehill_fast.rs");
include!("generated/kingofthehill_support.rs");
include!("generated/chess960_fast.rs");
include!("generated/chess960_support.rs");
include!("generated/horde_fast.rs");
include!("generated/horde_support.rs");
impl_tr!(chess_fast, topdown);
impl_tr!(chess_support, topdown);
impl_tr!(atomic_fast, topdown);
impl_tr!(atomic_support, topdown);
impl_tr!(threecheck_fast, topdown);
impl_tr!(threecheck_support, topdown);
impl_tr!(kingofthehill_fast, topdown);
impl_tr!(kingofthehill_support, topdown);
impl_tr!(chess960_fast, topdown);
impl_tr!(chess960_support, topdown);
impl_tr!(horde_fast, topdown);
impl_tr!(horde_support, topdown);

/// What the tests need of a module with the pass besides the questions every module has.
type Lg = Vec<(Vec<u32>, Vec<Vec<u32>>, Vec<Vec<u32>>, bool, u8)>;
trait Sp { fn counts(&self) -> (u64, u64, u64, u64, u64); fn reset(&mut self); fn plant(&mut self, on: bool); fn found(&self) -> Vec<(&'static str, &'static str, &'static str)>; fn log_on(&mut self); fn log_take(&mut self) -> Lg; }
macro_rules! impl_sp {
    ($m:ident) => {
        impl Sp for $m::Game {
            fn counts(&self) -> (u64, u64, u64, u64, u64) { self.support_counts() }
            fn reset(&mut self) { self.reset_support_counts() }
            fn plant(&mut self, on: bool) { self.sp_plant = on; }
            fn found(&self) -> Vec<(&'static str, &'static str, &'static str)> { $m::SUPPORT.to_vec() }
            fn log_on(&mut self) { self.sp_log = Some(vec![]); }
            fn log_take(&mut self) -> Lg { self.sp_log.take().unwrap_or_default().into_iter().map(|l| (l.key, l.d, l.s, l.ans, l.how)).collect() }
        }
    };
}
impl_sp!(chess_support);
impl_sp!(atomic_support);
impl_sp!(threecheck_support);
impl_sp!(kingofthehill_support);
impl_sp!(chess960_support);
impl_sp!(horde_support);

const SEED: u64 = 5454;
fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

type Rels = BTreeMap<&'static str, BTreeSet<Vec<String>>>;
fn rels<T: Tr>(t: &T, v: Vec<(&'static str, Vec<Vec<u32>>)>) -> Rels { v.into_iter().map(|(n, r)| (n, r.into_iter().map(|row| row.iter().map(|x| t.render_t(*x)).collect()).collect())).collect() }

#[derive(Default)]
struct Tally { states: usize, relations: usize, rows: usize, differs: usize }

/// One state: the same questions of both modules; every answer and every relation compared. -> whether they differ (with `must` the first difference panics).
fn check<C: Tr, F: Tr>(c: &mut C, f: &mut F, sc: &C::S, label: &str, must: bool, tally: &mut Tally) -> bool {
    let facts = c.facts_of_t(sc);
    let sf = f.state_of_facts_t(&facts).unwrap();
    let names = |t: &dyn Fn(u32) -> String, v: &[u32]| v.iter().map(|m| t(*m)).collect::<BTreeSet<_>>();
    let mut buf = Vec::with_capacity(256);
    c.moves_into_t(sc, &mut buf);
    let mv_c = buf.clone();
    let (term_c, mover_c, out_c) = (c.terminal_t(sc), c.mover_t(sc), c.outcome_t(sc));
    let (term_f, mover_f, out_f, mv_f) = (f.terminal_t(&sf), f.mover_t(&sf), f.outcome_t(&sf), f.moves_t(&sf));
    let mut bad = false;
    let mut note = |ok: bool, what: String| { if !ok { bad = true; if must { panic!("{label}: {what}"); } } };
    note(term_c == term_f, "the end".into());
    note(mover_c == mover_f, "the mover".into());
    note(out_c == out_f, "the outcome".into());
    note(names(&|m| c.render_t(m), &mv_c) == names(&|m| f.render_t(m), &mv_f), "the allowed moves".into());
    let (vc, vf) = (c.all_t(sc, &[]), f.all_t(&sf, &[]));
    let (rc, rf) = (rels(c, vc), rels(f, vf));
    for (n, rows) in &rc {
        let other = rf.get(n).unwrap_or_else(|| panic!("{label}: the relation {n} is in one module and not in the other"));
        note(rows == other, format!("the relation {n} ({} rows with the pass, {} without)", rows.len(), other.len()));
        tally.rows += rows.len();
    }
    tally.relations += rc.len();
    tally.states += 1;
    if bad { tally.differs += 1; }
    bad
}

/// The walks of the pass carry: along games, across siblings, jumps. `must`: panic at a difference; else the number of states that differ is returned.
fn run<C: Tr + Sp, F: Tr>(game: &str, must: bool, plant: bool) -> usize {
    let (games, plies) = (env_usize("CS_CARRY_GAMES", 8), env_usize("CS_CARRY_PLIES", 70));
    let mut c = C::make();
    let mut f = F::make();
    assert!(!c.found().is_empty(), "{game}: the module found no test");
    c.plant(plant);
    let mut rng = Rng::new(SEED);
    let mut tally = Tally::default();
    let mut all: Vec<C::S> = vec![];
    for g in 0..games {
        let mut s = c.initial_t();
        for ply in 0..plies {
            check(&mut c, &mut f, &s, &format!("{game} game {g} ply {ply} along"), must, &mut tally);
            let ms = c.moves_t(&s);
            if ms.is_empty() { break; }
            all.push(s.clone());
            if ply % 3 == 0 {
                for (i, m) in ms.iter().take(4).enumerate() {
                    let ch = c.next_t(&s, *m);
                    check(&mut c, &mut f, &ch, &format!("{game} game {g} ply {ply} child {i}"), must, &mut tally);
                }
            }
            let m = ms[rng.below(ms.len())];
            s = c.next_t(&s, m);
        }
    }
    let mut order: Vec<usize> = (0..all.len()).collect();
    for i in (1..order.len()).rev() { order.swap(i, rng.below(i + 1)); }
    for (k, i) in order.iter().enumerate() { check(&mut c, &mut f, &all[*i], &format!("{game} jump {k}"), must, &mut tally); }
    if must {
        println!("OBS 79.56 C {game} support equals no-support: {} states, {} relations compared ({} rows) with the rows of the module without the pass, {} differ; tests found: {:?}", tally.states, tally.relations, tally.rows, tally.differs, c.found());
    }
    tally.differs
}

macro_rules! game_tests {
    ($name:ident, $planted:ident, $sup:ident, $fast:ident, $g:expr) => {
        #[test]
        fn $name() { run::<$sup::Game, $fast::Game>($g, true, false); }
        #[test]
        fn $planted() {
            let n = run::<$sup::Game, $fast::Game>($g, false, true);
            println!("OBS 79.56 C {} a wrong reference planted: {} states differ from the module without the pass", $g, n);
            assert!(n > 0, "{}: a planted wrong reference was not caught by the equality test", $g);
        }
    };
}
game_tests!(chess_support_equals_no_support, chess_planted_wrong_reference_is_caught, chess_support, chess_fast, "chess");
game_tests!(atomic_support_equals_no_support, atomic_planted_wrong_reference_is_caught, atomic_support, atomic_fast, "atomic");
game_tests!(threecheck_support_equals_no_support, threecheck_planted_wrong_reference_is_caught, threecheck_support, threecheck_fast, "3check");
game_tests!(kingofthehill_support_equals_no_support, kingofthehill_planted_wrong_reference_is_caught, kingofthehill_support, kingofthehill_fast, "kingofthehill");
game_tests!(chess960_support_equals_no_support, chess960_planted_wrong_reference_is_caught, chess960_support, chess960_fast, "chess960");
game_tests!(horde_support_equals_no_support, horde_planted_wrong_reference_is_caught, horde_support, horde_fast, "horde");

/// The counters, with the profile on, along the games of the walk: keys, tests run, keys answered from a reference (OBS lines).
fn counts<C: Tr + Sp>(game: &str) {
    let (games, plies) = (env_usize("CS_CARRY_GAMES", 8), env_usize("CS_CARRY_PLIES", 70));
    let mut c = C::make();
    c.profiling_t(true);
    let mut rng = Rng::new(SEED);
    let mut moves = 0usize;
    for _ in 0..games {
        let mut s = c.initial_t();
        for _ in 0..plies {
            let ms = c.moves_t(&s);
            if ms.is_empty() { break; }
            moves += ms.len();
            let m = ms[rng.below(ms.len())];
            s = c.next_t(&s, m);
        }
    }
    let (st, keys, tests, ans, noref) = c.counts();
    println!("OBS 79.56 C {game} counters along {games} games: {st} states with a test run, {keys} keys, {tests} tests run, {ans} keys answered from a reference, {noref} guard classes with no reference; {moves} allowed moves");
    assert_eq!(keys, tests + ans, "{game}: every key is tested or answered");
    assert!(ans > 0, "{game}: no key was answered from a reference");
}
#[test]
fn chess_counters() { counts::<chess_support::Game>("chess"); }
#[test]
fn atomic_counters() { counts::<atomic_support::Game>("atomic"); }

/// The STOP RULE's material (plan 79.56): the support words of states, beside the keys. `CS_SUPPORT_WORDS` states are read from the walk (default 0: the test prints nothing). A key is written as its terms; a pattern as its terms with `*` for FREE.
fn words<C: Tr + Sp>(game: &str) {
    let n = env_usize("CS_SUPPORT_WORDS", 0);
    if n == 0 { return; }
    let mut c = C::make();
    let mut rng = Rng::new(SEED);
    let mut s = c.initial_t();
    let mut shown = 0;
    for ply in 0..400 {
        let ms = c.moves_t(&s);
        if ms.is_empty() { break; }
        if ply % 5 == 2 && shown < n {
            let mut buf = vec![];
            let other = c.next_t(&s, ms[0]);
            let _ = c.moves_t(&other);                       // the module keeps its last reading: ask another state first
            c.log_on();
            c.moves_into_t(&s, &mut buf);
            let lg = c.log_take();
            let pat = |c: &C, p: &Vec<u32>| p.iter().map(|x| if *x == u32::MAX { "*".to_string() } else { c.render_t(*x) }).collect::<Vec<_>>().join(" ");
            println!("OBS 79.56 words {game} state {shown} (ply {ply}): {} keys, {} allowed moves", lg.len(), buf.len());
            for (key, d, sw, ans, how) in lg.iter() {
                let ks = key.iter().map(|x| c.render_t(*x)).collect::<Vec<_>>().join(" ");
                println!("   key {ks} | answer {ans} | how {} | D {} patterns: {} | S {} patterns: {}", ["tested", "tested, kept as reference", "from reference"][*how as usize], d.len(), d.iter().map(|p| format!("({})", pat(&c, p))).collect::<Vec<_>>().join(" "), sw.len(), sw.iter().map(|p| format!("({})", pat(&c, p))).collect::<std::collections::BTreeSet<_>>().into_iter().collect::<Vec<_>>().join(" "));
            }
            shown += 1;
        }
        let m = ms[rng.below(ms.len())];
        s = c.next_t(&s, m);
    }
}
#[test]
#[ignore = "prints the support words by hand: CS_SUPPORT_WORDS=20 cargo test ... the_support_words -- --ignored --nocapture"]
fn the_support_words_chess() { words::<chess_support::Game>("chess"); }
