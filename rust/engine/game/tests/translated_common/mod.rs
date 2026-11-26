//! PLAN 79.37 (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ..."; ledger rows 45, 44; piece: rules): WHAT THE TESTS AND THE TIMING OF THE TRANSLATED CODE SHARE. The generated modules (tests/generated/<game>.rs, written by
//! discovery/gdl_to_rust.py) all have the same public items; `Tr` names the ones the tests use, `impl_tr!` forwards them, and everything below is written once for every game. The evaluator it is held against is cs_game's `GdlGame`
//! (the five questions) and cs_gdl's `Program` (every relation). Nothing here is part of the crate's sources.
#![allow(dead_code)]
pub mod pieces;
use cs_game::gdl_game::{parse_term, render_term, row_args, GdlGame, GdlState};
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::rng::Rng;
use cs_game::rules::restart_delta;
use cs_game::{Fact, Game};
use cs_gdl::{Program, Rel, Row, SortMember, ARITY_MAX, FREE};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// The public items of a generated module that the tests use (each module has them under these names as inherent items).
pub trait Tr {
    type S: Clone + PartialEq + std::fmt::Debug;
    fn make() -> Self;
    fn initial_t(&self) -> Self::S;
    fn roles_t(&self) -> Vec<u32>;
    fn mover_t(&mut self, s: &Self::S) -> Option<usize>;
    fn moves_t(&mut self, s: &Self::S) -> Vec<u32>;
    fn next_t(&mut self, s: &Self::S, m: u32) -> Self::S;
    fn terminal_t(&mut self, s: &Self::S) -> bool;
    fn outcome_t(&mut self, s: &Self::S) -> Option<Vec<i64>>;
    fn render_t(&self, t: u32) -> String;
    fn term_t(&mut self, text: &str) -> Result<u32, String>;
    fn state_of_facts_t(&mut self, f: &[(String, Vec<String>)]) -> Result<Self::S, String>;
    /// 2026-11-24 (plan 79.47): the module begun again at other initial facts (a position of a fixture or of a game's list of starts), as `evaluator_at` begins the evaluator: the relations that read an `init` relation are made again.
    fn restart_t(&mut self, f: &[(String, Vec<String>)]) -> Result<(), String>;
    fn facts_of_t(&self, s: &Self::S) -> Vec<(String, Vec<String>)>;
    fn statics_t(&self) -> Vec<(&'static str, Vec<Vec<u32>>)>;
    fn all_t(&mut self, s: &Self::S, does: &[(u32, u32)]) -> Vec<(&'static str, Vec<Vec<u32>>)>;
    fn noop_t(&self) -> u32;
    fn sym_t(&self, name: &str) -> Option<u32>;
    fn profiling_t(&mut self, on: bool);
    fn reset_profile_t(&mut self);
    /// (relations of the piece, static, calls, nanoseconds, rows made, rows visited, indexes asked)
    fn pieces_t(&self) -> Vec<(&'static str, bool, u64, u64, u64, u64, u64)>;
    /// PLAN 79.38: the passes the module was made with (none: the plain translation), the relations it answers on demand with how each is asked, the next relations whose copy it keeps, and its answers to keys.
    fn passes_t(&self) -> Vec<&'static str>;
    fn demanded_t(&self) -> Vec<(&'static str, Vec<usize>, bool)>;
    fn copied_t(&self) -> Vec<&'static str>;
    fn ask_t(&mut self, s: &Self::S, does: &[(u32, u32)], queries: &[(String, Vec<u32>)]) -> Vec<(Vec<Vec<u32>>, Option<bool>)>;
    /// PLAN 79.52 PART B (SYSTEM_SPEC 11, last block; ledger rows 55, 45). THE INTERFACE OF THE QUESTIONS THAT MAKE NO MEMORY, written before the bodies. WHO CALLS: the clocks and the tests of the modules made with the passes `cones` and `words`.
    /// `moves_into_t(s, out)`: TAKES a state and a buffer the caller owns; RETURNS (in the buffer, which is cleared first) the allowed moves of the mover as numbers (none at an end): what `moves_t` returns, with no vector made.
    /// `legal_into_t(s, out)`: the same without the test of the end and without any work that only the end or the outcome needs: the `legal` rows of the first role that has a turn that is not the turn that passes.
    /// The defaults answer by `moves_t`, so a module made without the passes has both and the old entry points (`moves_t`, `next_t`, `facts_of_t`, `state_of_facts_t`) are unchanged on top of them.
    fn moves_into_t(&mut self, s: &Self::S, out: &mut Vec<u32>) { out.clear(); out.extend(self.moves_t(s)); }
    fn legal_into_t(&mut self, s: &Self::S, out: &mut Vec<u32>) { self.moves_into_t(s, out) }
    /// PLAN 79.53 PART B (SYSTEM_SPEC 11, last block, (vii); ledger rows 55, 45). THE INTERFACE OF THE TOP-DOWN WAY, written before its body. WHO CALLS: the tests and the clocks of the modules made with the pass `topdown`; the engine's bridge later.
    /// `terminal_td_t(s)`: TAKES a state; RETURNS whether the end of the game holds, answered by a depth-first walk of the rules that stops at the first rule that holds: what `terminal_t` returns, by the other way.
    /// `has_move_td_t(s)`: TAKES a state; RETURNS whether a turn that is not the turn that passes is legal (the mover exists; the end is not asked), by the walk that stops at the first legal row.
    /// The defaults answer by the bottom-up way (`terminal_t`, `legal_into_t`), so a module made without the pass has both; the macro kind `topdown` overrides them.
    fn terminal_td_t(&mut self, s: &Self::S) -> bool { self.terminal_t(s) }
    fn has_move_td_t(&mut self, s: &Self::S) -> bool { let mut b = vec![]; self.legal_into_t(s, &mut b); !b.is_empty() }
    /// `ask_td_t(s, relation)`: the rows of a relation of the cone of the questions, no place bound, made by the top-down way (None: not such a relation, or the walk overflowed); the tests hold each to the bottom-up rows.
    fn ask_td_t(&mut self, _s: &Self::S, _rel: &str) -> Option<Vec<Vec<u32>>> { None }
    /// The number of questions the top-down way gave to the bottom-up way because a walk kept more rows than its bound (module text, (24)).
    fn td_overflows_t(&self) -> u64 { 0 }
    /// Only the load and the clearing of the tables of the top-down way (what they cost before any rule is walked).
    fn td_load_only_t(&mut self, _s: &Self::S) {}
    /// PLAN 79.54 PART B (SYSTEM_SPEC 11, last block, (ix); ledger rows 55, 45). THE INTERFACE OF THE ANSWERS CARRIED FROM THE LAST STATE, written before its body. WHO CALLS: the tests and the clocks of the modules made with the pass `carry` (macro kind `carry`).
    /// `carried_t(s)`: TAKES a state; reads it for every question of the cones (the evaluation kept from the last state is brought to it) and RETURNS the rows the kept evaluation holds of each relation of a valid piece of those cones, to be held to `all_t` (the relations made in full). The default has none.
    /// `carried_valid_t()`: how many pieces of the kept evaluation hold the rows of the last state read (the others are made again when asked). The default is 0.
    fn carried_t(&mut self, _s: &Self::S) -> Vec<(&'static str, Vec<Vec<u32>>)> { vec![] }
    fn carried_valid_t(&self) -> usize { 0 }
    /// Plan 79.56: the counters of the support filter (states, keys, tests run, keys answered from a reference, classes with no reference) and their reset; nought in a module without the pass.
    fn support_counts_t(&self) -> (u64, u64, u64, u64, u64) { (0, 0, 0, 0, 0) }
    fn support_reset_t(&mut self) {}
}

#[macro_export]
macro_rules! impl_tr {
    ($m:ident) => { $crate::impl_tr!($m, plain); };
    ($m:ident, $kind:ident) => {
        impl $crate::translated_common::Tr for $m::Game {
            type S = $m::State;
            fn make() -> Self { $m::Game::new() }
            fn initial_t(&self) -> Self::S { self.initial() }
            fn roles_t(&self) -> Vec<u32> { self.roles().to_vec() }
            fn mover_t(&mut self, s: &Self::S) -> Option<usize> { self.mover(s) }
            fn moves_t(&mut self, s: &Self::S) -> Vec<u32> { self.moves(s) }
            fn next_t(&mut self, s: &Self::S, m: u32) -> Self::S { self.next(s, m) }
            fn terminal_t(&mut self, s: &Self::S) -> bool { self.terminal(s) }
            fn outcome_t(&mut self, s: &Self::S) -> Option<Vec<i64>> { self.outcome(s) }
            fn render_t(&self, t: u32) -> String { self.render(t) }
            fn term_t(&mut self, text: &str) -> Result<u32, String> { self.term(text) }
            fn state_of_facts_t(&mut self, f: &[(String, Vec<String>)]) -> Result<Self::S, String> { self.state_of_facts(f) }
            fn restart_t(&mut self, f: &[(String, Vec<String>)]) -> Result<(), String> { self.restart(f) }
            fn facts_of_t(&self, s: &Self::S) -> Vec<(String, Vec<String>)> { self.facts_of(s) }
            fn statics_t(&self) -> Vec<(&'static str, Vec<Vec<u32>>)> { self.static_relations() }
            fn all_t(&mut self, s: &Self::S, does: &[(u32, u32)]) -> Vec<(&'static str, Vec<Vec<u32>>)> { self.all_relations(s, does) }
            fn noop_t(&self) -> u32 { self.noop() }
            fn sym_t(&self, name: &str) -> Option<u32> { self.sym(name) }
            fn profiling_t(&mut self, on: bool) { self.profiling = on; }
            fn reset_profile_t(&mut self) { self.reset_profile(); }
            fn pieces_t(&self) -> Vec<(&'static str, bool, u64, u64, u64, u64, u64)> {
                $m::PIECES.iter().zip(self.prof.iter()).map(|((n, st), p)| (*n, *st, p.calls, p.nanos, p.rows, p.visits, p.probes)).collect()
            }
            $crate::impl_tr_passes!($m, $kind);
        }
    };
}

#[macro_export]
macro_rules! impl_tr_passes {
    ($m:ident, plain) => {
        fn passes_t(&self) -> Vec<&'static str> { vec![] }
        fn demanded_t(&self) -> Vec<(&'static str, Vec<usize>, bool)> { vec![] }
        fn copied_t(&self) -> Vec<&'static str> { vec![] }
        fn ask_t(&mut self, _s: &Self::S, _does: &[(u32, u32)], _q: &[(String, Vec<u32>)]) -> Vec<(Vec<Vec<u32>>, Option<bool>)> { panic!("the plain module has no demanded relation") }
    };
    ($m:ident, topdown) => {
        $crate::impl_tr_passes!($m, fresh);
        fn terminal_td_t(&mut self, s: &Self::S) -> bool { self.terminal_td(s) }
        fn has_move_td_t(&mut self, s: &Self::S) -> bool { self.has_turn_td(s) }
        fn ask_td_t(&mut self, s: &Self::S, rel: &str) -> Option<Vec<Vec<u32>>> { self.ask_td(s, rel) }
        fn td_overflows_t(&self) -> u64 { self.td_over }
        fn td_load_only_t(&mut self, s: &Self::S) { self.td_load_only(s) }
    };
    ($m:ident, carry) => {
        $crate::impl_tr_passes!($m, topdown);
        fn carried_t(&mut self, s: &Self::S) -> Vec<(&'static str, Vec<Vec<u32>>)> { self.carried_relations(s) }
        fn carried_valid_t(&self) -> usize { self.carried_valid() }
    };
    ($m:ident, support) => {
        $crate::impl_tr_passes!($m, topdown);
        fn support_counts_t(&self) -> (u64, u64, u64, u64, u64) { self.support_counts() }
        fn support_reset_t(&mut self) { self.reset_support_counts() }
    };
    ($m:ident, fresh) => {
        $crate::impl_tr_passes!($m, passes);
        fn moves_into_t(&mut self, s: &Self::S, out: &mut Vec<u32>) { self.moves_into(s, out) }
        fn legal_into_t(&mut self, s: &Self::S, out: &mut Vec<u32>) { self.legal_moves_into(s, out) }
    };
    ($m:ident, passes) => {
        fn passes_t(&self) -> Vec<&'static str> { $m::PASSES.to_vec() }
        fn demanded_t(&self) -> Vec<(&'static str, Vec<usize>, bool)> { $m::Game::demanded() }
        fn copied_t(&self) -> Vec<&'static str> { $m::COPIED.to_vec() }
        fn ask_t(&mut self, s: &Self::S, does: &[(u32, u32)], q: &[(String, Vec<u32>)]) -> Vec<(Vec<Vec<u32>>, Option<bool>)> { self.ask_many(s, does, q) }
    };
}

pub fn rules_path(name: &str) -> std::path::PathBuf { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules").join(name) }

/// The description of a game as the translator reads it: the base text, composed with the delta where there is one.
pub fn description(delta: Option<&str>) -> String {
    let base = std::fs::read_to_string(rules_path("chess.gdl")).unwrap();
    match delta { None => base, Some(d) => cs_gdl::compose(&base, &std::fs::read_to_string(rules_path(d)).unwrap()).unwrap() }
}

pub fn evaluator(text: &str) -> GdlGame { GdlGame::load("evaluator", text).unwrap() }

/// The evaluator begun at the facts of a state shown (the same door the descriptions are held against the fixtures by).
pub fn evaluator_at(text: &str, facts: &BTreeSet<Fact>) -> GdlGame {
    let f: Vec<Fact> = facts.iter().cloned().collect();
    let delta = restart_delta(text, &f).unwrap();
    GdlGame::load("shown", &cs_gdl::compose(text, &delta).unwrap()).unwrap()
}

/// 2026-11-24 (plan 79.47): the reader's program of a description begun at the facts of a state shown (the same restarted text `evaluator_at` loads), for the comparison of every relation at a position of a game whose rules read its start.
pub fn program_at(text: &str, facts: &BTreeSet<Fact>) -> Program {
    let f: Vec<Fact> = facts.iter().cloned().collect();
    let delta = restart_delta(text, &f).unwrap();
    Program::load(&cs_gdl::compose(text, &delta).unwrap()).unwrap()
}

pub fn facts_pairs(f: &BTreeSet<Fact>) -> Vec<(String, Vec<String>)> { f.iter().map(|x| (x.relation.clone(), x.args.clone())).collect() }

pub fn fact_set(v: Vec<(String, Vec<String>)>) -> BTreeSet<Fact> { v.into_iter().map(|(relation, args)| Fact { relation, args }).collect() }

pub fn load_average() -> String {
    match std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "unknown".into(),
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the five questions at a state

#[derive(Default, Debug, Clone)]
pub struct Tally { pub positions: usize, pub moves: usize, pub nexts: usize, pub terminal: usize, pub facts: usize }

/// The translated code and the evaluator at one state: the side to move, the allowed moves, the end, the outcome, the facts, and the next state of each move in `which` (indexes of the moves sorted by name; all when `all`).
/// -> Err(the first disagreement: what, and the smallest case).
pub fn compare_at<T: Tr>(t: &mut T, ev: &GdlGame, ts: &T::S, es: &GdlState, label: &str, all: bool, tally: &mut Tally) -> Result<(), String> {
    let roles = t.roles_t();
    let tm = t.mover_t(ts).map(|i| t.render_t(roles[i]));
    let em = ev.mover(es).map(|i| ev.roles()[i].clone());
    if tm != em { return Err(format!("{label}: the side to move: translated {tm:?}, evaluator {em:?}")); }
    let (to, eo) = (t.outcome_t(ts), ev.outcome(es));
    if to != eo { return Err(format!("{label}: the outcome: translated {to:?}, evaluator {eo:?}")); }
    if t.terminal_t(ts) != eo.is_some() { return Err(format!("{label}: the end: translated {}, evaluator {}", t.terminal_t(ts), eo.is_some())); }
    let (tf, ef) = (fact_set(t.facts_of_t(ts)), ev.base_facts(es).into_iter().collect::<BTreeSet<_>>());
    if tf != ef { return Err(format!("{label}: the facts: only translated {:?}, only evaluator {:?}", tf.difference(&ef).collect::<Vec<_>>(), ef.difference(&tf).collect::<Vec<_>>())); }
    let tmv = t.moves_t(ts);
    let emv = ev.moves(es);
    let tn: BTreeMap<String, u32> = tmv.iter().map(|m| (t.render_t(*m), *m)).collect();
    let en: BTreeMap<String, _> = emv.iter().map(|m| (ev.move_name(es, m), *m)).collect();
    let (a, b): (BTreeSet<_>, BTreeSet<_>) = (tn.keys().cloned().collect(), en.keys().cloned().collect());
    if a != b { return Err(format!("{label}: the allowed moves: only translated {:?}, only evaluator {:?}", a.difference(&b).collect::<Vec<_>>(), b.difference(&a).collect::<Vec<_>>())); }
    if tn.len() != tmv.len() { return Err(format!("{label}: a move given twice by the translated code")); }
    tally.positions += 1;
    tally.moves += tn.len();
    tally.terminal += eo.is_some() as usize;
    tally.facts += tf.len();
    let names: Vec<&String> = tn.keys().collect();
    let which: Vec<usize> = if all { (0..names.len()).collect() } else { vec![] };
    for i in which {
        let name = names[i];
        let (tchild, echild) = (t.next_t(ts, tn[name]), ev.apply(es, &en[name]));
        let (a, b) = (fact_set(t.facts_of_t(&tchild)), ev.base_facts(&echild).into_iter().collect::<BTreeSet<_>>());
        if a != b { return Err(format!("{label}: the next state after {name}: only translated {:?}, only evaluator {:?}", a.difference(&b).collect::<Vec<_>>(), b.difference(&a).collect::<Vec<_>>())); }
        tally.nexts += 1;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------------------------------ every relation

/// The evaluator's rows for a state (and a joint move) as text: relation -> set of tuples.
fn evaluator_relations<T: Tr>(t: &T, prog: &mut Program, ts: &T::S, does: &[(u32, u32)]) -> Result<BTreeMap<String, BTreeSet<Vec<String>>>, String> {
    let mut facts: Vec<(Rel, Row)> = vec![];
    for (rel, args) in t.facts_of_t(ts) {
        let r = prog.rel_id(&format!("true.{rel}")).ok_or(format!("the evaluator has no fluent {rel}"))?;
        let mut row = [FREE; ARITY_MAX];
        for (i, a) in args.iter().enumerate() { row[i] = parse_term(prog, a)?; }
        facts.push((r, row));
    }
    if let Some(dr) = prog.rel_id("does") {
        for (r, m) in does {
            let mut row = [FREE; ARITY_MAX];
            row[0] = parse_term(prog, &t.render_t(*r))?;
            row[1] = parse_term(prog, &t.render_t(*m))?;
            facts.push((dr, row));
        }
    }
    let db = prog.run(&facts);
    let mut out = BTreeMap::new();
    for r in 0..prog.nrels() {
        let rows = db.facts(r as Rel);
        out.insert(prog.rel_name(r as Rel).to_string(), rows.iter().map(|row| row_args(row).iter().map(|x| render_term(prog, *x)).collect::<Vec<_>>()).collect::<BTreeSet<_>>());
    }
    Ok(out)
}

/// EVERY RELATION THAT READS THE STATE, translated against the evaluator on one state: equal rows. `count` gets, for each relation, the states checked and the rows compared. -> Err(relation and the smallest difference).
pub fn relations_at<T: Tr>(t: &mut T, prog: &mut Program, ts: &T::S, mv: Option<u32>, label: &str, count: &mut BTreeMap<String, (usize, usize)>) -> Result<(), String> {
    let roles = t.roles_t();
    let mover = t.mover_t(ts);
    let noop = t.noop_t();
    let does: Vec<(u32, u32)> = match mv { Some(m) => roles.iter().enumerate().map(|(i, r)| (*r, if Some(i) == mover { m } else { noop })).collect(), None => vec![] };
    let theirs = evaluator_relations(t, prog, ts, &does)?;
    for (name, rows) in t.all_t(ts, &does) {
        let mine: BTreeSet<Vec<String>> = rows.iter().map(|r| r.iter().map(|x| t.render_t(*x)).collect()).collect();
        let empty = BTreeSet::new();
        let ev = theirs.get(name).unwrap_or(&empty);
        if &mine != ev {
            return Err(format!("{label}: the relation {name}: only translated {:?}, only evaluator {:?}", mine.difference(ev).take(3).collect::<Vec<_>>(), ev.difference(&mine).take(3).collect::<Vec<_>>()));
        }
        let e = count.entry(name.to_string()).or_insert((0, 0));
        e.0 += 1;
        e.1 += mine.len();
    }
    Ok(())
}

pub struct StaticReport { pub name: String, pub rows: usize, pub domain: Option<u64> }

/// EVERY RELATION THAT DOES NOT READ THE STATE, translated against the evaluator on its WHOLE DOMAIN: the same rows, and then every tuple of the signature's sorts asked of both (`domain` counts them; None where a sort is a pattern of
/// compound terms or the relation has no signature). -> Err(relation and the first tuple they differ on).
pub fn statics_whole_domain<T: Tr>(t: &T, prog: &mut Program) -> Result<Vec<StaticReport>, String> {
    let mut out = vec![];
    for (name, rows) in t.statics_t() {
        let rid = prog.rel_id(name).ok_or(format!("the evaluator has no relation {name}"))?;
        let theirs: Vec<Row> = prog.statics.get(&rid).map(|r| r.order.clone()).unwrap_or_default();
        let a: BTreeSet<Vec<String>> = rows.iter().map(|r| r.iter().map(|x| t.render_t(*x)).collect()).collect();
        let b: BTreeSet<Vec<String>> = theirs.iter().map(|r| row_args(r).iter().map(|x| render_term(prog, *x)).collect()).collect();
        if a != b { return Err(format!("{name}: the rows: only translated {:?}, only evaluator {:?}", a.difference(&b).take(3).collect::<Vec<_>>(), b.difference(&a).take(3).collect::<Vec<_>>())); }
        if rows.len() != theirs.len() { return Err(format!("{name}: {} rows translated, {} by the evaluator", rows.len(), theirs.len())); }
        // the whole domain
        let sig = prog.signature(name).cloned();
        let mut domain = None;
        if let Some(sig) = sig {
            let mut members: Vec<Vec<String>> = vec![];
            let mut ok = true;
            for s in &sig {
                match prog.sorts.get(s) {
                    Some(ms) if ms.iter().all(|m| matches!(m, SortMember::Const(_))) => members.push(ms.iter().map(|m| match m { SortMember::Const(c) => c.clone(), _ => unreachable!() }).collect()),
                    _ => { ok = false; break; }
                }
            }
            let total: u64 = members.iter().map(|m| m.len() as u64).product();
            if ok && total <= 40_000_000 {
                let mine: HashSet<Vec<u32>> = rows.iter().cloned().collect();
                let tids: Vec<Vec<Option<u32>>> = members.iter().map(|ms| ms.iter().map(|c| t.sym_t(c)).collect()).collect();
                let eids: Vec<Vec<u32>> = members.iter().map(|ms| ms.iter().map(|c| prog.terms.sym(c)).collect()).collect();
                let stat = prog.statics.get(&rid).cloned();
                let mut idx = vec![0usize; members.len()];
                let mut count = 0u64;
                'dom: loop {
                    if members.iter().any(|m| m.is_empty()) { break; }
                    let tt: Option<Vec<u32>> = idx.iter().enumerate().map(|(k, &i)| tids[k][i]).collect();
                    let in_t = tt.map(|v| mine.contains(&v)).unwrap_or(false);
                    let mut row = [FREE; ARITY_MAX];
                    for (k, &i) in idx.iter().enumerate() { row[k] = eids[k][i]; }
                    let in_e = stat.as_ref().map(|r| r.facts.contains(&row)).unwrap_or(false);
                    if in_t != in_e {
                        let tuple: Vec<&String> = idx.iter().enumerate().map(|(k, &i)| &members[k][i]).collect();
                        return Err(format!("{name}: the tuple {tuple:?}: translated {in_t}, evaluator {in_e}"));
                    }
                    count += 1;
                    let mut k = members.len();
                    loop {
                        if k == 0 { break 'dom; }
                        k -= 1;
                        idx[k] += 1;
                        if idx[k] < members[k].len() { break; }
                        idx[k] = 0;
                    }
                }
                if members.is_empty() { count = 1; }
                domain = Some(count);
            }
        }
        out.push(StaticReport { name: name.to_string(), rows: rows.len(), domain });
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------------------------------------------------ co-play, the corpus of the certificate

/// The seeded random games of the certificate (cs_game::certify's walk: the kernel's moves sorted by its names, one picked by the seeded generator), the translated code and the evaluator played side by side and compared at every state:
/// the five questions, the next state of the move played (of every move at every `all_every`-th state), and, when `prog` is given, every relation that reads the state at the first `rel_games` games. Returns the paths played (the kernel's
/// move names), which are the corpus the speed test replays.
pub fn coplay<T: Tr>(kernel: &KernelGame, t: &mut T, ev: &GdlGame, mut prog: Option<&mut Program>, games: usize, seed: u64, all_every: usize, rel_games: usize,
                     tally: &mut Tally, rels: &mut BTreeMap<String, (usize, usize)>) -> Result<Vec<Vec<String>>, String> {
    let mut rng = Rng::new(seed);
    let mut corpus = vec![];
    let mut states = 0usize;
    for g in 0..games {
        let (mut ks, mut es, mut ts) = (kernel.initial(), ev.initial(), t.initial_t());
        let mut line: Vec<String> = vec![];
        loop {
            states += 1;
            let label = format!("game {g} after {line:?}");
            compare_at(t, ev, &ts, &es, &label, all_every > 0 && states % all_every == 0, tally)?;
            let km = kernel.moves(&ks);
            if kernel.outcome(&ks).is_some() || km.is_empty() { break; }
            let mut sorted: Vec<(String, usize)> = km.iter().enumerate().map(|(i, m)| (kernel.move_name(&ks, m), i)).collect();
            sorted.sort();
            let (nm, i) = sorted[rng.below(sorted.len())].clone();
            let text = spec_term(&nm);
            let tmove = t.term_t(&text)?;
            if let (Some(p), true) = (prog.as_deref_mut(), g < rel_games) {
                relations_at(t, p, &ts, None, &label, rels)?;
                relations_at(t, p, &ts, Some(tmove), &label, rels)?;
            }
            let em = ev.find_move(&es, &text).ok_or(format!("{label}: the evaluator does not allow {text}"))?;
            let tn = t.next_t(&ts, tmove);
            let en = ev.apply(&es, &em);
            ks = kernel.apply(&ks, &km[i]);
            ts = tn;
            es = en;
            line.push(nm);
        }
        corpus.push(line);
    }
    Ok(corpus)
}

/// The number of positions after `depth` plies from `s`, by the translated code.
pub fn perft<T: Tr>(t: &mut T, s: &T::S, depth: usize) -> u64 {
    if depth == 0 { return 1; }
    let ms = t.moves_t(s);
    if depth == 1 { return ms.len() as u64; }
    ms.iter().map(|m| { let c = t.next_t(s, *m); perft(t, &c, depth - 1) }).sum()
}

/// PLAN 79.54 (the pass case; Kaveh 2026-11-25): THE PASSED STATE, made here in the harness from the facts and not by any entry of a module: the same state with the next role to move and nothing played (the fact that says whose turn it is is the one fact changed). `control_fluent` names that fluent
/// (a test file may name it). None when the state has no such fact or the roles are not two.
pub fn passed<T: Tr>(t: &mut T, s: &T::S, control_fluent: &str) -> Option<T::S> {
    let roles: Vec<String> = t.roles_t().iter().map(|r| t.render_t(*r)).collect();
    if roles.len() != 2 { return None; }
    let mut facts = t.facts_of_t(s);
    let mut changed = false;
    for f in facts.iter_mut() {
        if f.0 == control_fluent && f.1.len() == 1 {
            let i = roles.iter().position(|r| *r == f.1[0])?;
            f.1[0] = roles[(i + 1) % 2].clone();
            changed = true;
        }
    }
    if !changed { return None; }
    t.state_of_facts_t(&facts).ok()
}
