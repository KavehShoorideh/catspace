//! PLAN 79.53 PART B1 (SYSTEM_SPEC 11, last block, (vii); ledger rows 55, 45; piece: rules): THE TOP-DOWN END AND THE TOP-DOWN "IS THERE A LEGAL TURN", HELD TO THE BOTTOM-UP WAY.
//! Interface under test (written before the bodies; the head of the pass (24) in gdl_to_rust.py and `Tr::terminal_td_t`, `Tr::has_move_td_t` in translated_common/mod.rs): `terminal_td(&State) -> bool` and `has_move_td(&State) -> bool`, made by the pass `topdown`
//! over the same state of words as the bottom-up way. Held, for the seven games: (1) at every position of the game's fixture (the outside library's file: ended positions among them) and at the states of seeded random games of the module itself (every state),
//! `terminal_td_t` equals `terminal_t`, and `has_move_td_t` equals "the legal rows of the first role that has a turn are not none" (`legal_into_t`); (2) the counts of what was compared are printed. This file names the games on purpose (a test file may).
mod variants_common;
#[macro_use]
mod translated_common;
use cs_game::rng::Rng;
use translated_common::*;
use variants_common::fixture;

include!("generated/chess_fast.rs");
include!("generated/chess_lemma.rs");
include!("generated/atomic_fast.rs");
include!("generated/3check_fast.rs");
include!("generated/kingofthehill_fast.rs");
include!("generated/antichess_fast.rs");
include!("generated/chess960_fast.rs");
include!("generated/horde_fast.rs");
impl_tr!(chess_fast, topdown);
impl_tr!(chess_lemma, topdown);
impl_tr!(atomic_fast, topdown);
impl_tr!(threecheck_fast, topdown);
impl_tr!(kingofthehill_fast, topdown);
impl_tr!(antichess_fast, topdown);
impl_tr!(chess960_fast, topdown);
impl_tr!(horde_fast, topdown);

const SEED: u64 = 5353;

/// Every relation of the cone of the questions, by the top-down way and by the bottom-up way (the rows the module makes, or, for a relation it answers on demand, the rows by an ask with no place bound), at one state: the relations that differ, named.
fn relations_differing<T: Tr>(t: &mut T, s: &T::S) -> Vec<String> {
    let made: Vec<(&'static str, Vec<Vec<u32>>)> = t.all_t(s, &[]);
    let mut names: Vec<String> = made.iter().map(|x| x.0.to_string()).collect();
    names.extend(t.demanded_t().iter().map(|x| x.0.to_string()));
    names.sort();
    names.dedup();
    let mut bad = vec![];
    for name in names {
        let Some(td) = t.ask_td_t(s, &name) else { continue };
        let bu: Vec<Vec<u32>> = match made.iter().find(|x| x.0 == name) {
            Some(x) => x.1.clone(),
            None => { let (rows, _) = t.ask_t(s, &[], &[(name.clone(), vec![])]).remove(0); rows }
        };
        let mut bu: Vec<Vec<u32>> = bu;
        bu.sort();
        bu.dedup();
        // a place where a clause gives a compound term holds FREE - 1 top-down: compared without it
        let hole: Vec<usize> = (0..td.first().map(|r| r.len()).unwrap_or(0)).filter(|&j| td.iter().any(|r| r[j] == u32::MAX - 1)).collect();
        let cut = |v: &Vec<Vec<u32>>| -> std::collections::BTreeSet<Vec<u32>> { v.iter().map(|r| r.iter().enumerate().filter(|(j, _)| !hole.contains(j)).map(|(_, x)| *x).collect()).collect() };
        if cut(&td) != cut(&bu) { bad.push(format!("{name}: top-down {} rows, bottom-up {} rows", cut(&td).len(), cut(&bu).len())); }
    }
    bad
}

fn same<T: Tr>(t: &mut T, s: &T::S, label: &str, c: &mut (usize, usize, usize)) {
    let (bu, td) = (t.terminal_t(s), t.terminal_td_t(s));
    if bu != td { let bad = relations_differing(t, s); assert_eq!(bu, td, "{label}: the end: bottom-up {bu}, top-down {td}; the relations that differ: {bad:?}"); }
    let mut b = vec![];
    t.legal_into_t(s, &mut b);
    let (hm, hd) = (!b.is_empty(), t.has_move_td_t(s));
    if hm != hd { let bad = relations_differing(t, s); assert_eq!(hm, hd, "{label}: a legal turn: bottom-up {hm}, top-down {hd}; the relations that differ: {bad:?}"); }
    c.0 += 1;
    c.1 += bu as usize;
    c.2 += (!hm) as usize;
}

fn held<T: Tr>(game: &str, walks: usize) {
    // the fixture's positions (ended ones among them)
    let fx = fixture(game);
    let mut t = T::make();
    let mut c = (0usize, 0usize, 0usize);
    for p in &fx.pos {
        t.restart_t(&translated_common::facts_pairs(&p.facts)).unwrap();
        let s = t.state_of_facts_t(&translated_common::facts_pairs(&p.facts)).unwrap();
        same(&mut t, &s, &format!("{game} fixture {} ({})", p.label, p.text), &mut c);
    }
    let (fp, fe, fn_) = c;
    // seeded random games of the module itself, every state
    let mut t = T::make();
    let mut rng = Rng::new(SEED);
    let mut c2 = (0usize, 0usize, 0usize);
    for g in 0..walks {
        let mut s = t.initial_t();
        for ply in 0..160 {
            same(&mut t, &s, &format!("{game} game {g} ply {ply}"), &mut c2);
            let ms = t.moves_t(&s);
            if ms.is_empty() { break; }
            s = t.next_t(&s, ms[rng.below(ms.len())]);
        }
    }
    let over = t.td_overflows_t();
    println!("OBS 79.53 B1 overflows {game}: {over} questions of the walks' bound given to the bottom-up way");
    println!("OBS 79.53 B1 held {game}: {fp} fixture positions ({fe} ended, {fn_} with no legal turn) and {} states of {walks} seeded random games ({} ended, {} with no legal turn): the end and the legal turn the same top-down as bottom-up", c2.0, c2.1, c2.2);
    assert!(fp >= 100 && fe > 0);
}

#[test]
fn chess_top_down_equals_bottom_up() { held::<chess_fast::Game>("chess", 30); }
#[test]
fn chess_with_the_pin_lemma_top_down_equals_bottom_up() { held::<chess_lemma::Game>("chess", 30); }
#[test]
fn atomic_top_down_equals_bottom_up() { held::<atomic_fast::Game>("atomic", 30); }
#[test]
fn threecheck_top_down_equals_bottom_up() { held::<threecheck_fast::Game>("3check", 30); }
#[test]
fn kingofthehill_top_down_equals_bottom_up() { held::<kingofthehill_fast::Game>("kingofthehill", 30); }
#[test]
fn antichess_top_down_equals_bottom_up() { held::<antichess_fast::Game>("antichess", 30); }
#[test]
fn chess960_top_down_equals_bottom_up() { held::<chess960_fast::Game>("chess960", 30); }
#[test]
fn horde_top_down_equals_bottom_up() { held::<horde_fast::Game>("horde", 30); }
