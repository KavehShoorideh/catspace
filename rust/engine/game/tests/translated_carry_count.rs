//! PLAN 79.54 PART A (SYSTEM_SPEC 11, last block, (ix) "answers carried from the last state"; piece: rules; ledger rows 55, 45): THE COUNT, before any carry is built.
//! INTERFACE. No item of a module is added. It reads, for each game, the kept every-pass module (`<game>_fast.rs`) through `Tr` (`all_t`, `moves_into_t`, `next_t`, `pieces_t`, `profiling_t`) and the description through the reader (`cs_gdl::Program`, its `deps`).
//! WHAT IT COUNTS. Over seeded random games of the module itself: for each state `p` and the state `c` after the move played (SUCCESSOR pairs), and for `c` and `c2`, the state after another allowed move of `p` (SIBLING pairs):
//!   for every derived relation (every relation of the module that reads the state; the facts of the state, `true.*`, and the relations that read the turn, `does`, are left out of the totals): the rows made at the first state, at the second, the rows in both ("same"),
//!   in how many pairs the relation was equal as a whole; whether it reads the side to move (the dependency graph reaches the fluent the test names `SIDE`); and the share of the time of the question "the allowed moves with the end" (`moves_into_t`, bottom-up)
//!   that the pieces whose rows were ALL the same at the pair took (piece time from the module's own profile, wall time of the whole question around it).
//! B1'S CEILING, in the same pass: the share of that time in pieces whose cone reads no fluent that changed between the two states (a piece is skipped when no fluent word its cone reads differs). `CS_CARRY_GAMES` and `CS_CARRY_PLIES` set the corpus.
//! This file names the games on purpose (a test file may). The rows are compared as numbers of one module's tables (one `Game` serves both states of a pair).
#[macro_use]
mod translated_common;
use cs_game::rng::Rng;
use cs_gdl::Program;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;
use translated_common::*;

include!("generated/chess_fast.rs");
include!("generated/atomic_fast.rs");
include!("generated/3check_fast.rs");
include!("generated/kingofthehill_fast.rs");
include!("generated/antichess_fast.rs");
include!("generated/chess960_fast.rs");
include!("generated/horde_fast.rs");
impl_tr!(chess_fast, passes);
impl_tr!(atomic_fast, passes);
impl_tr!(threecheck_fast, passes);
impl_tr!(kingofthehill_fast, passes);
impl_tr!(antichess_fast, passes);
impl_tr!(chess960_fast, passes);
impl_tr!(horde_fast, passes);

const SEED: u64 = 5454;
/// The fluent that says whose turn it is, in these descriptions.
const SIDE: &str = "true.control";

fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

/// The relations that reach `target` through the dependency graph (the relation itself included).
fn reaches(prog: &Program, target: &str) -> BTreeSet<String> {
    let Some(t) = prog.rel_id(target) else { return BTreeSet::new() };
    let mut out = BTreeSet::new();
    for r in 0..prog.nrels() {
        let mut seen = BTreeSet::new();
        let mut todo = vec![r as u16];
        while let Some(x) = todo.pop() {
            if !seen.insert(x) { continue; }
            if x == t { out.insert(prog.rel_name(r as u16).to_string()); break; }
            todo.extend(prog.deps[x as usize].iter().copied());
        }
    }
    out
}

/// The fluents (`true.*`) a relation's cone reads.
fn fluents_read(prog: &Program, rel: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(r) = prog.rel_id(rel) else { return out };
    let mut seen = BTreeSet::new();
    let mut todo = vec![r];
    while let Some(x) = todo.pop() {
        if !seen.insert(x) { continue; }
        if prog.rel_name(x).starts_with("true.") { out.insert(prog.rel_name(x).to_string()); }
        todo.extend(prog.deps[x as usize].iter().copied());
    }
    out
}

type Rows = BTreeMap<&'static str, BTreeSet<Vec<u32>>>;
fn rows_at<T: Tr>(t: &mut T, s: &T::S) -> Rows { t.all_t(s, &[]).into_iter().map(|(n, r)| (n, r.into_iter().collect())).collect() }

#[derive(Default, Clone)]
struct Acc { pairs: usize, first: usize, second: usize, same: usize, equal_pairs: usize }
#[derive(Default)]
struct Kind {
    rel: BTreeMap<String, Acc>,
    nanos: BTreeMap<String, u64>, nanos_same: BTreeMap<String, u64>,
    wall: u64, piece_total: u64, piece_same: u64, piece_skippable: u64, piece_rows_same: f64, nonside_total: u64, nonside_rows_same: f64, piece_groups_same: f64, groups: BTreeMap<String, (usize, usize, usize)>,
    pairs: usize, bits_changed: usize, fluents_changed: BTreeMap<String, usize>,
}

/// The rows grouped by their value at the first place (a relation of no places is one group).
fn group_by(v: &BTreeSet<Vec<u32>>) -> BTreeMap<u32, BTreeSet<&Vec<u32>>> {
    let mut m: BTreeMap<u32, BTreeSet<&Vec<u32>>> = BTreeMap::new();
    for r in v { m.entry(r.first().copied().unwrap_or(0)).or_default().insert(r); }
    m
}

fn measure<T: Tr>(t: &mut T, prog: &Program, a: &T::S, b: &T::S, k: &mut Kind, reads_does: &BTreeSet<String>, side: &BTreeSet<String>) {
    // the question first (the module has not read `b`), then the rows of both
    let mut buf = Vec::with_capacity(256);
    t.moves_into_t(a, &mut buf);       // the module's last state is the first state of the pair (a no-op when it is already)
    t.reset_profile_t();
    let w = Instant::now();
    t.moves_into_t(b, &mut buf);
    k.wall += w.elapsed().as_nanos() as u64;
    let prof: Vec<(&'static str, bool, u64, u64)> = t.pieces_t().iter().map(|p| (p.0, p.1, p.2, p.3)).collect();
    let (ra, rb) = (rows_at(t, a), rows_at(t, b));
    k.pairs += 1;
    let changed: BTreeSet<String> = ra.keys().filter(|n| n.starts_with("true.") && ra[*n] != rb[*n]).map(|n| n.to_string()).collect();
    for f in &changed { *k.fluents_changed.entry(f.clone()).or_insert(0) += 1; }
    k.bits_changed += ra.iter().filter(|(n, _)| n.starts_with("true.")).map(|(n, x)| x.symmetric_difference(&rb[n]).count()).sum::<usize>();
    let mut same_rel: BTreeSet<&str> = BTreeSet::new();
    for (n, x) in &ra {
        let y = &rb[n];
        let e = k.rel.entry(n.to_string()).or_default();
        e.pairs += 1; e.first += x.len(); e.second += y.len(); e.same += x.intersection(y).count();
        if x == y { e.equal_pairs += 1; same_rel.insert(n); }
    }
    for (n, st, calls, nanos) in prof {
        if st || calls == 0 || n.starts_with("true.") || reads_does.contains(n) { continue; }
        *k.nanos.entry(n.to_string()).or_insert(0) += nanos;
        k.piece_total += nanos;
        if same_rel.contains(n) { *k.nanos_same.entry(n.to_string()).or_insert(0) += nanos; k.piece_same += nanos; }
        let reads = fluents_read(prog, n);
        if !reads.iter().any(|f| changed.contains(f)) { k.piece_skippable += nanos; }
        // the ceiling at the level of a row: the piece's time times the share of the rows made at the second state that the first state had too (the cost taken as proportional to the rows)
        let (x, y) = (&ra[n], &rb[n]);
        let ratio = if y.is_empty() { if x.is_empty() { 1.0 } else { 0.0 } } else { x.intersection(y).count() as f64 / y.len() as f64 };
        k.piece_rows_same += nanos as f64 * ratio;
        // the same at the level of a group of rows (the rows with one value at the first place), the group carried only when the whole group is the same: the best any read set of a group could do
        let (gx, gy) = (group_by(x), group_by(y));
        let same_rows: usize = gy.iter().filter(|(key, rows)| gx.get(*key) == Some(*rows)).map(|(_, rows)| rows.len()).sum();
        let gratio = if y.is_empty() { if x.is_empty() { 1.0 } else { 0.0 } } else { same_rows as f64 / y.len() as f64 };
        k.piece_groups_same += nanos as f64 * gratio;
        let e = k.groups.entry(n.to_string()).or_insert((0, 0, 0));
        e.0 += gy.len(); e.1 += 1; e.2 += gy.iter().filter(|(key, rows)| gx.get(*key) != Some(*rows)).count();
        if !side.contains(n) { k.nonside_total += nanos; k.nonside_rows_same += nanos as f64 * ratio; }
    }
}

fn report(game: &str, kind: &str, k: &Kind, side: &BTreeSet<String>, reads_does: &BTreeSet<String>) {
    println!("OBS 79.54 A {game} {kind}: {} pairs; mean facts changed a pair {:.2}; fluents changed (pairs): {:?}", k.pairs, k.bits_changed as f64 / k.pairs.max(1) as f64, k.fluents_changed);
    let (mut s1, mut s2, mut ss) = (0usize, 0usize, 0usize);
    let (mut d1, mut d2, mut ds) = (0usize, 0usize, 0usize);   // the relations that do not read the side to move
    for (n, a) in &k.rel {
        if n.starts_with("true.") || reads_does.contains(n.as_str()) { continue; }
        if a.second == 0 && a.first == 0 { continue; }
        let reads = side.contains(n);
        println!("OBS 79.54 A {game} {kind} relation {n}: rows first {:.1} second {:.1} same {:.1} share_same {:.1}% equal_in {}/{} pairs reads_side_to_move {} time_share_of_pieces {:.1}% time_share_when_all_same {:.1}%", a.first as f64 / a.pairs as f64, a.second as f64 / a.pairs as f64, a.same as f64 / a.pairs as f64,
            100.0 * a.same as f64 / a.second.max(1) as f64, a.equal_pairs, a.pairs, if reads { "yes" } else { "no" }, 100.0 * *k.nanos.get(n).unwrap_or(&0) as f64 / k.piece_total.max(1) as f64,
            100.0 * *k.nanos_same.get(n).unwrap_or(&0) as f64 / (*k.nanos.get(n).unwrap_or(&0)).max(1) as f64);
        s1 += a.first; s2 += a.second; ss += a.same;
        if !reads { d1 += a.first; d2 += a.second; ds += a.same; }
    }
    println!("OBS 79.54 A {game} {kind} total: derived rows first {s1} second {s2} same {ss} ({:.1}% of the second); of the relations that do not read the side to move: first {d1} second {d2} same {ds} ({:.1}%)", 100.0 * ss as f64 / s2.max(1) as f64, 100.0 * ds as f64 / d2.max(1) as f64);
    println!("OBS 79.54 A {game} {kind} ceiling: if every row that the first state had cost nothing (a piece's time taken as proportional to its rows) the pieces' time falls by {:.1}% ({:.1}% of the wall); the pieces that do not read the side to move are {:.1}% of the pieces' time and their unchanged rows {:.1}% of it",
        100.0 * k.piece_rows_same / k.piece_total.max(1) as f64, 100.0 * k.piece_rows_same / k.wall.max(1) as f64, 100.0 * k.nonside_total as f64 / k.piece_total.max(1) as f64, 100.0 * k.nonside_rows_same / k.piece_total.max(1) as f64);
    println!("OBS 79.54 A {game} {kind} group ceiling: if a group of rows (the rows with one value at the first place) were carried whenever the whole group was the same at both states (the best any read set of a group could do), the pieces' time falls by {:.1}% ({:.1}% of the wall) against {:.1}% at the level of a row",
        100.0 * k.piece_groups_same / k.piece_total.max(1) as f64, 100.0 * k.piece_groups_same / k.wall.max(1) as f64, 100.0 * k.piece_rows_same / k.piece_total.max(1) as f64);
    for (n, (g, c, ch)) in &k.groups {
        let per_state = *g as f64 / (*c).max(1) as f64;
        let ns = *k.nanos.get(n).unwrap_or(&0) as f64 / (*c).max(1) as f64;
        if k.nanos.get(n).copied().unwrap_or(0) as f64 / k.piece_total.max(1) as f64 > 0.02 { println!("OBS 79.54 A {game} {kind} groups of {n}: {:.1} a state, {:.0} ns of piece time a state, {:.1} ns a group; groups whose rows changed (the groups a read set could not carry) {:.2} a state", per_state, ns, ns / per_state.max(1.0), *ch as f64 / (*c).max(1) as f64); }
    }
    println!("OBS 79.54 A {game} {kind} time: the pieces whose rows were all the same took {:.1}% of the pieces' time ({:.1}% of the question's wall, the wall {} ns, the pieces {} ns); the pieces whose cone reads no changed fluent took {:.1}% of the pieces' time ({:.1}% of the wall)",
        100.0 * k.piece_same as f64 / k.piece_total.max(1) as f64, 100.0 * k.piece_same as f64 / k.wall.max(1) as f64, k.wall, k.piece_total,
        100.0 * k.piece_skippable as f64 / k.piece_total.max(1) as f64, 100.0 * k.piece_skippable as f64 / k.wall.max(1) as f64);
}

fn count<T: Tr>(game: &str, delta: Option<&str>) {
    let (games, plies) = (env_usize("CS_CARRY_GAMES", 6), env_usize("CS_CARRY_PLIES", 60));
    let prog = Program::load(&description(delta)).unwrap();
    let side = reaches(&prog, SIDE);
    let reads_does = reaches(&prog, "does");
    let mut t = T::make();
    t.profiling_t(true);
    let (mut succ, mut sib, mut grand, mut pass) = (Kind::default(), Kind::default(), Kind::default(), Kind::default());
    let mut rng = Rng::new(SEED);
    for _ in 0..games {
        let mut s = t.initial_t();
        let mut par: Option<T::S> = None;
        for _ply in 0..plies {
            let ms = t.moves_t(&s);
            if ms.is_empty() { break; }
            if let Some(ps) = passed(&mut t, &s, "control") { measure(&mut t, &prog, &s, &ps, &mut pass, &reads_does, &side); }
            let i = rng.below(ms.len());
            let c = t.next_t(&s, ms[i]);
            measure(&mut t, &prog, &s, &c, &mut succ, &reads_does, &side);
            if let Some(pp) = &par { measure(&mut t, &prog, pp, &c, &mut grand, &reads_does, &side); }
            if ms.len() > 1 {
                let mut j = rng.below(ms.len());
                if j == i { j = (j + 1) % ms.len(); }
                let c2 = t.next_t(&s, ms[j]);
                measure(&mut t, &prog, &c, &c2, &mut sib, &reads_does, &side);
            }
            par = Some(s);
            s = c;
        }
    }
    println!("OBS 79.54 A {game} load {}", load_average());
    report(game, "successor", &succ, &side, &reads_does);
    report(game, "sibling", &sib, &side, &reads_does);
    report(game, "second successor", &grand, &side, &reads_does);
    report(game, "pass", &pass, &side, &reads_does);
    // EXACT: after a pass the only fact that differs is the one that says whose turn it is, so every relation that does not read it is equal as a whole at every pair, and only relations that read it differ
    let mut differ_side = 0;
    for (n, a) in &pass.rel {
        if n.starts_with("true.") || reads_does.contains(n.as_str()) { continue; }
        if !side.contains(n) { assert_eq!(a.equal_pairs, a.pairs, "{game}: the relation {n} does not read the side to move and differs between a state and its passed state"); } else if a.equal_pairs < a.pairs { differ_side += 1; }
    }
    println!("OBS 79.54 A {game} pass exact: {} pairs; every relation that does not read the side to move is equal in every pair; {differ_side} relations that read it differ in some", pass.pairs);
}

#[test]
fn chess_the_count() { count::<chess_fast::Game>("chess", None); }
#[test]
fn atomic_the_count() { count::<atomic_fast::Game>("atomic", Some("variants/atomic.gdl")); }
#[test]
fn threecheck_the_count() { count::<threecheck_fast::Game>("3check", Some("variants/3check.gdl")); }
#[test]
fn kingofthehill_the_count() { count::<kingofthehill_fast::Game>("kingofthehill", Some("variants/kingofthehill.gdl")); }
#[test]
fn antichess_the_count() { count::<antichess_fast::Game>("antichess", Some("variants/antichess.gdl")); }
#[test]
fn chess960_the_count() { count::<chess960_fast::Game>("chess960", Some("variants/chess960.gdl")); }
#[test]
fn horde_the_count() { count::<horde_fast::Game>("horde", Some("variants/horde.gdl")); }
