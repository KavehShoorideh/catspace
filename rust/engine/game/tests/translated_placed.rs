//! PLAN 79.48 PART D (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 42): THE PLACED QUESTIONS OF THE KERNEL HELD TO THE
//! TRANSLATED MODULE OF THEIR DEFINITIONS. THIS TEXT IS THE INTERFACE, written before the body.
//!
//! WHAT IT TAKES. A game (the kernel by its name; the description composed with the additive text of the placed questions, rules/placed/base.json and, where a game has one, rules/placed/<game>.json, which the
//! translator made into the module tests/generated_placed/<game>.rs, every pass, only READ here), and a corpus: seeded random games (the walk of translated_kernel.rs and of the certificate, so the small corpus's count of
//! states is the certificate's). The questions are those of rust/engine/game/src/placed.rs that the description can answer (the list is catspace/incubator/analyst/discovery/placed_questions.py).
//! WHAT IT COMPARES, at every state of the walk, the kernel's answer with the module's rows of the definition, both as sets of numbers (a place is `file * 8 + rank`, a kind its index in the kernel's list of kinds):
//!   cells, kind_at          what stands where                       placed_cell      the board of the state
//!   bearing_from            what a unit bears on                    placed_bears     every unit that stands (one question), and PROBES: a unit of a kind and an owner GIVEN on a place (standing there or
//!                                                                                    not): the state of the probe holds the unit as a `cell` fluent on the place (what was there is replaced)
//!   reach_from              where a unit may move to                placed_reaches   every unit that stands, the owner to move and no passed-over place (one question for each side), and the probes
//!   strikes_place           what a side strikes                     placed_strikes   every place for each side
//!   move_ends, find_step    where an allowed move starts and ends   placed_mover     every allowed move (the ends are the projection); find_step: every pair of places the module's ends or the kernel's
//!                                                                                    moves have and a few drawn pairs; at a state the game calls over the questions are LEFT OUT and counted (the layers above ask the end first)
//!   move_info               what a move plays and takes             placed_mover, placed_takes
//!   place_names, kind_names the game's names of places and kinds    the description's sorts `file`, `rank`, `kind` (once a game)
//! and, on the small corpus, THE LOWER LINK: the module's rows of each definition against the evaluator of the composed description (cs_gdl's Program), the same queries (the states, the sides to move, the first probes).
//! A DISAGREEMENT is a finding, never hidden and never a stop: it is counted under its CAUSE (the question, the side that has the extra row, the class of the row), the first few of each are kept WHOLE (the state, the line
//! played to it, the rows of each side, the evaluator's answer to the same row). The checker is never weakened; what the questions leave out of a comparison (a state the game calls over) is counted.
//! WHO CALLS IT. `cargo test --release --offline -p cs_game --test translated_placed` (the small tests) and, by hand, `-- --ignored` (the large run, all cores by threads, a fixed list of seeds, a few minutes a game);
//! discovery/placed_translated.py runs them, reads the counts they write (CS_PL_OUT) and keeps the record. ENVIRONMENT of the large run: CS_PL_CORES, CS_PL_SEEDS, CS_PL_GAMES, CS_PL_PROBES, CS_PL_OUT.
//! This file names the games on purpose (a test file may); the layer crates do not.
mod translated_common;
use cs_game::gdl_game::{parse_term, render_term, row_args};
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::placed::Placed;
use cs_game::rng::Rng;
use cs_game::{Fact, Game as _};
use cs_gdl::{Program, Rel, Row, ARITY_MAX, FREE};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use translated_common::*;

include!("generated_placed/chess.rs");
include!("generated_placed/atomic.rs");
include!("generated_placed/3check.rs");
include!("generated_placed/kingofthehill.rs");
include!("generated_placed/antichess.rs");
include!("generated_placed/chess960.rs");
include!("generated_placed/horde.rs");
impl_tr!(placed_chess, passes);
impl_tr!(placed_atomic, passes);
impl_tr!(placed_threecheck, passes);
impl_tr!(placed_kingofthehill, passes);
impl_tr!(placed_antichess, passes);
impl_tr!(placed_chess960, passes);
impl_tr!(placed_horde, passes);

type KS = <KernelGame as cs_game::Game>::State;
type Pairs = Vec<(String, Vec<String>)>;
/// A row of an answer as numbers (a place is `file * 8 + rank`), the places not used `NONE`.
type R4 = [u16; 4];
const NONE: u16 = u16::MAX;

pub const SEED: u64 = 77;
pub const GAMES: usize = 20;
pub const STARTS_GAMES: usize = 4;
pub const LARGE_SEED0: u64 = 1000;
/// Probes (a unit given on a place) a state in the small run and in the large run; the first `EV_PROBES` of the small run are also put to the evaluator.
pub const PROBES_SMALL: usize = 48;
pub const PROBES_LARGE: usize = 8;
pub const EV_PROBES: usize = 1;
pub const KEEP: usize = 3;

// ------------------------------------------------------------------------------------------------------------------------------------------ the description and its additive text

/// The clauses of the `delta` of a document of rules/placed/ (one clause a line, in double quotes).
fn delta_of(name: &str) -> Option<String> {
    let text = std::fs::read_to_string(rules_path(&format!("placed/{name}.json"))).ok()?;
    let mut out = vec![];
    let mut inside = false;
    for ln in text.lines() {
        let t = ln.trim();
        if t.starts_with("\"delta\": [") { inside = true; continue; }
        if inside {
            if t == "]" || t == "]," { break; }
            let t = t.trim_end_matches(',');
            out.push(t.trim_matches('"').replace("\\\"", "\"").replace("\\\\", "\\"));
        }
    }
    Some(out.join("\n"))
}

/// The description of a game composed with the additive text of the placed questions (the base document, then the game's own where there is one).
pub fn text_of(delta: Option<&str>, game: &str) -> String {
    let mut text = cs_gdl::compose(&description(delta), &delta_of("base").expect("rules/placed/base.json")).unwrap();
    if let Some(d) = delta_of(game) { text = cs_gdl::compose(&text, &d).unwrap(); }
    text
}

fn start_text(game: &str) -> String {
    match game {
        "chess" => text_of(None, game),
        g => text_of(Some(&format!("variants/{g}.gdl")), g),
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ rows as numbers

/// Term numbers of the constants as the numbers of the answers: a file, a rank, a kind, a side.
struct Dec { file: Vec<u16>, rank: Vec<u16>, kind: Vec<u16>, side: Vec<u16> }

fn dec_make<T: Tr>(t: &T, kinds: &[String], roles: &[String]) -> Dec {
    const N: usize = 4096;
    let mut d = Dec { file: vec![NONE; N], rank: vec![NONE; N], kind: vec![NONE; N], side: vec![NONE; N] };
    for (i, c) in "abcdefgh".chars().enumerate() { if let Some(n) = t.sym_t(&c.to_string()) { d.file[n as usize] = i as u16; } }
    for (i, c) in "12345678".chars().enumerate() { if let Some(n) = t.sym_t(&c.to_string()) { d.rank[n as usize] = i as u16; } }
    for (i, k) in kinds.iter().enumerate() { if let Some(n) = t.sym_t(k) { d.kind[n as usize] = i as u16; } }
    for (i, r) in roles.iter().enumerate() { if let Some(n) = t.sym_t(r) { d.side[n as usize] = i as u16; } }
    d
}

fn at(v: &[u16], t: u32) -> u16 { v.get(t as usize).copied().unwrap_or(NONE) }

impl Dec {
    fn place(&self, f: u32, r: u32) -> u16 { let (a, b) = (at(&self.file, f), at(&self.rank, r)); if a == NONE || b == NONE { NONE } else { a * 8 + b } }
    /// The rows of a relation of the definitions as numbers: None where a term is not what its place says (counted as a row the harness cannot read).
    fn row(&self, rel: &str, x: &[u32]) -> Option<R4> {
        let r = match rel {
            "placed_cell" => [self.place(x[0], x[1]), at(&self.side, x[2]), at(&self.kind, x[3]), NONE],
            "placed_bears" | "placed_reaches" => [at(&self.kind, x[0]), at(&self.side, x[1]), self.place(x[2], x[3]), self.place(x[4], x[5])],
            "placed_strikes" => [at(&self.side, x[0]), self.place(x[1], x[2]), NONE, NONE],
            "placed_mover" | "placed_takes" => [at(&self.side, x[0]), self.place(x[1], x[2]), self.place(x[3], x[4]), at(&self.kind, x[5])],
            _ => return None,
        };
        if r.iter().take(match rel { "placed_cell" => 3, "placed_strikes" => 2, _ => 4 }).any(|v| *v == NONE) { None } else { Some(r) }
    }
}

/// The same rows from the text of an answer of the evaluator.
struct Names { files: BTreeMap<String, u16>, kinds: BTreeMap<String, u16>, sides: BTreeMap<String, u16> }

impl Names {
    fn new(kinds: &[String], roles: &[String]) -> Names {
        Names { files: "abcdefgh".chars().enumerate().map(|(i, c)| (c.to_string(), i as u16)).collect(), kinds: kinds.iter().enumerate().map(|(i, k)| (k.clone(), i as u16)).collect(), sides: roles.iter().enumerate().map(|(i, k)| (k.clone(), i as u16)).collect() }
    }
    fn place(&self, f: &str, r: &str) -> u16 { match (self.files.get(f), r.parse::<u16>()) { (Some(a), Ok(b)) if (1..=8).contains(&b) => a * 8 + b - 1, _ => NONE } }
    fn row(&self, rel: &str, x: &[String]) -> Option<R4> {
        let kind = |i: usize| self.kinds.get(&x[i]).copied().unwrap_or(NONE);
        let side = |i: usize| self.sides.get(&x[i]).copied().unwrap_or(NONE);
        let r = match rel {
            "placed_cell" => [self.place(&x[0], &x[1]), side(2), kind(3), NONE],
            "placed_bears" | "placed_reaches" => [kind(0), side(1), self.place(&x[2], &x[3]), self.place(&x[4], &x[5])],
            "placed_strikes" => [side(0), self.place(&x[1], &x[2]), NONE, NONE],
            "placed_mover" | "placed_takes" => [side(0), self.place(&x[1], &x[2]), self.place(&x[3], &x[4]), kind(5)],
            _ => return None,
        };
        Some(r)
    }
}

fn arity(rel: &str) -> usize { match rel { "placed_cell" => 4, "placed_bears" | "placed_reaches" => 6, "placed_strikes" => 3, _ => 6 } }

fn rows_set(d: &Dec, rel: &str, rows: &[Vec<u32>], bad: &mut usize) -> BTreeSet<R4> {
    let mut s = BTreeSet::new();
    for x in rows { match d.row(rel, x) { Some(r) => { s.insert(r); } None => *bad += 1 } }
    s
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the states the questions are asked at

fn pairs_of(facts: &[Fact]) -> Pairs { facts.iter().map(|x| (x.relation.clone(), x.args.clone())).collect() }

fn place_chars(p: u16) -> (String, String) { (((b'a' + (p / 8) as u8) as char).to_string(), (p % 8 + 1).to_string()) }

/// The state of the question of a side's units: the side to move, no passed-over place.
fn with_control(base: &Pairs, side: &str) -> Pairs {
    let mut v: Pairs = base.iter().filter(|(r, _)| r != "control" && r != "ep_square").cloned().collect();
    v.push(("control".into(), vec![side.to_string()]));
    v
}

/// The state of a probe: a unit of the kind and the side GIVEN on the place (what stood there replaced), the side to move, no passed-over place.
fn with_unit(base: &Pairs, p: u16, side: &str, kind: &str) -> Pairs {
    let (f, r) = place_chars(p);
    let mut v: Pairs = with_control(base, side).into_iter().filter(|(rel, a)| !(rel == "cell" && a[0] == f && a[1] == r)).collect();
    v.push(("cell".into(), vec![f, r, side.to_string(), kind.to_string()]));
    v
}

fn ask<T: Tr>(t: &mut T, s: &T::S, rels: &[&str]) -> Vec<Vec<Vec<u32>>> {
    let q: Vec<(String, Vec<u32>)> = rels.iter().map(|n| (n.to_string(), vec![u32::MAX; arity(n)])).collect();
    t.ask_t(s, &[], &q).into_iter().map(|(rows, _)| rows).collect()
}

fn evaluator_rows(prog: &mut Program, pairs: &Pairs, rels: &[&str]) -> Result<BTreeMap<String, BTreeSet<Vec<String>>>, String> {
    let mut facts: Vec<(Rel, Row)> = vec![];
    for (rel, args) in pairs {
        let r = prog.rel_id(&format!("true.{rel}")).ok_or(format!("the evaluator has no fluent {rel}"))?;
        let mut row = [FREE; ARITY_MAX];
        for (i, a) in args.iter().enumerate() { row[i] = parse_term(prog, a)?; }
        facts.push((r, row));
    }
    let db = prog.run(&facts);
    let mut out = BTreeMap::new();
    for name in rels {
        if let Some(r) = prog.rel_id(name) {
            out.insert(name.to_string(), db.facts(r).iter().map(|row| row_args(row).iter().map(|x| render_term(prog, *x)).collect::<Vec<_>>()).collect());
        }
    }
    Ok(out)
}

fn evaluator_set(prog: &mut Program, nm: &Names, pairs: &Pairs, rel: &str) -> BTreeSet<R4> {
    let rows = evaluator_rows(prog, pairs, &[rel]).expect("the evaluator runs");
    rows.get(rel).map(|s| s.iter().filter_map(|x| nm.row(rel, x)).collect()).unwrap_or_default()
}

// ------------------------------------------------------------------------------------------------------------------------------------------ what a run counts

#[derive(Clone, Debug, Default)]
pub struct Finding { pub cause: String, pub seed: u64, pub game: usize, pub line: Vec<String>, pub state: String, pub kernel: String, pub module: String, pub evaluator: String }

#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub games: usize, pub plies: usize, pub states: usize, pub probes: usize, pub ended: usize,
    /// question -> (comparisons made, rows compared)
    pub compared: BTreeMap<String, (usize, usize)>,
    /// the lower link: relation -> (comparisons made, rows compared), and the states it was made at
    pub lower: BTreeMap<String, (usize, usize)>, pub lower_states: usize,
    pub left_out: BTreeMap<String, usize>, pub causes: BTreeMap<String, usize>, pub findings: Vec<Finding>,
}

impl Counts {
    pub fn merge(&mut self, o: &Counts) {
        self.games += o.games; self.plies += o.plies; self.states += o.states; self.probes += o.probes; self.ended += o.ended; self.lower_states += o.lower_states;
        for (k, v) in &o.compared { let e = self.compared.entry(k.clone()).or_insert((0, 0)); e.0 += v.0; e.1 += v.1; }
        for (k, v) in &o.lower { let e = self.lower.entry(k.clone()).or_insert((0, 0)); e.0 += v.0; e.1 += v.1; }
        for (k, v) in &o.left_out { *self.left_out.entry(k.clone()).or_insert(0) += v; }
        for (k, v) in &o.causes { *self.causes.entry(k.clone()).or_insert(0) += v; }
        for f in &o.findings { if self.findings.iter().filter(|x| x.cause == f.cause).count() < KEEP { self.findings.push(f.clone()); } }
    }
    pub fn disagreements(&self) -> usize { self.causes.values().sum() }
}

fn text_of_rows(rel: &str, rows: &BTreeSet<R4>, names: &GameNames) -> String {
    rows.iter().map(|r| names.show(rel, r)).collect::<Vec<_>>().join("; ")
}

/// The kernel's names of its places, kinds and sides, and the way a row is said.
pub struct GameNames { pub places: Vec<String>, pub kinds: Vec<String>, pub roles: Vec<String> }

impl GameNames {
    fn show(&self, rel: &str, r: &R4) -> String {
        let (p, k, s) = (|i: u16| self.places.get(i as usize).cloned().unwrap_or("?".into()), |i: u16| self.kinds.get(i as usize).cloned().unwrap_or("?".into()), |i: u16| self.roles.get(i as usize).cloned().unwrap_or("?".into()));
        match rel {
            "placed_cell" => format!("{} {} on {}", s(r[1]), k(r[2]), p(r[0])),
            "placed_bears" | "placed_reaches" => format!("{} {} on {} -> {}", s(r[1]), k(r[0]), p(r[2]), p(r[3])),
            "placed_strikes" => format!("{} strikes {}", s(r[0]), p(r[1])),
            "kind_at" => format!("{} on {}", k(r[1]), p(r[0])),
            "ends" => format!("{} -> {}", p(r[0]), p(r[1])),
            _ => format!("{} {} -> {} ({})", s(r[0]), p(r[1]), p(r[2]), k(r[3])),
        }
    }
}

/// One question's comparison: its name, the relation of the definitions that answers it, and how a row is classed for naming a cause.
struct Qd { name: &'static str, rel: &'static str }

/// The class of a row, for naming a cause (the shape, not a rule).
fn class_of(q: &Qd, r: &R4, names: &GameNames, kernel: &KernelGame, ks: &KS) -> String {
    let kind = |i: u16| names.kinds.get(i as usize).cloned().unwrap_or("?".into());
    let shape = |a: u16, b: u16| {
        let (fa, ra, fb, rb) = ((a / 8) as i32, (a % 8) as i32, (b / 8) as i32, (b % 8) as i32);
        if fa == fb && (ra - rb).abs() == 2 { "a two-place step along a file" } else if ra == rb && (fa - fb).abs() == 2 { "a two-file move" } else { "other" }
    };
    match q.name {
        "cells" => kind(r[2]),
        "kind_at" => kind(r[1]),
        "bearing_from" | "bearing_from (given)" => kind(r[0]),
        "reach_from" | "reach_from (given)" => format!("{}, {}", kind(r[0]), shape(r[2], r[3])),
        "strikes_place" => {
            // whether the owner's own leader stands beside the place (the shape the description's `struck` leaves out)
            let leader = names.kinds.iter().position(|k| k == "king");
            let beside = leader.map(|l| kernel.cells(ks).iter().any(|(o, k, p)| *o as u16 == r[0] && *k as usize == l && *p != r[1] && ((*p / 8) as i32 - (r[1] / 8) as i32).abs() <= 1 && ((*p % 8) as i32 - (r[1] % 8) as i32).abs() <= 1)).unwrap_or(false);
            if beside { "the owner's own king beside the place".to_string() } else { "no king of the owner beside the place".to_string() }
        }
        "move_ends" | "find_step" => shape(r[0], r[1]).to_string(),
        "move_info (kinds played)" | "move_info (kinds taken)" => kind(r[3]),
        _ => String::new(),
    }
}

struct Acc<'a> {
    counts: &'a mut Counts, text: &'a str, start: &'a BTreeSet<Fact>, listed: bool, names: &'a GameNames, nm: &'a Names, kernel: &'a KernelGame,
    seed: u64, game: usize, line: &'a [String], fen: String, ks: &'a KS,
}

impl Acc<'_> {
    fn evaluator_has(&self, pairs: &Pairs, q: &Qd, row: &R4) -> String {
        let mut prog = if self.listed { program_at(self.text, self.start) } else { Program::load(self.text).unwrap() };
        let rows: BTreeSet<R4> = if q.name == "move_ends" || q.name == "find_step" {
            evaluator_set(&mut prog, self.nm, pairs, "placed_mover").iter().map(|r| [r[1], r[2], NONE, NONE]).collect()
        } else if q.name == "kind_at" {
            evaluator_set(&mut prog, self.nm, pairs, "placed_cell").iter().map(|r| [r[0], r[2], NONE, NONE]).collect()
        } else { evaluator_set(&mut prog, self.nm, pairs, q.rel) };
        format!("the evaluator {} this row", if rows.contains(row) { "HAS" } else { "does NOT have" })
    }

    /// Compare the kernel's rows with the module's rows for one question at one state; `pairs` is the state the module was asked at.
    fn compare(&mut self, q: &Qd, kernel_rows: &BTreeSet<R4>, module_rows: &BTreeSet<R4>, pairs: &Pairs, context: &str) {
        let e = self.counts.compared.entry(q.name.to_string()).or_insert((0, 0));
        e.0 += 1;
        e.1 += kernel_rows.len().max(module_rows.len());
        for (side, only, other) in [("kernel", kernel_rows.difference(module_rows), "module"), ("module", module_rows.difference(kernel_rows), "kernel")] {
            let _ = other;
            for r in only {
                let cause = format!("{}: only {side}, {}", q.name, class_of(q, r, self.names, self.kernel, self.ks));
                *self.counts.causes.entry(cause.clone()).or_insert(0) += 1;
                if self.counts.findings.iter().filter(|f| f.cause == cause).count() >= KEEP { continue; }
                let ev = self.evaluator_has(pairs, q, r);
                let rel = if q.name == "move_ends" || q.name == "find_step" { "ends" } else if q.name == "kind_at" { "kind_at" } else { q.rel };
                let (kt, mt) = (text_of_rows(rel, kernel_rows, self.names), text_of_rows(rel, module_rows, self.names));
                let cap = |s: String| if s.len() > 700 { format!("{}...", &s[..700]) } else { s };
                self.counts.findings.push(Finding { cause, seed: self.seed, game: self.game, line: self.line.to_vec(), state: format!("{} | {context} | the row {}", self.fen, self.names.show(rel, r)), kernel: cap(kt), module: cap(mt), evaluator: ev });
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the walk

const Q_CELLS: Qd = Qd { name: "cells", rel: "placed_cell" };
const Q_KIND_AT: Qd = Qd { name: "kind_at", rel: "placed_cell" };
const Q_BEARS: Qd = Qd { name: "bearing_from", rel: "placed_bears" };
const Q_BEARS_G: Qd = Qd { name: "bearing_from (given)", rel: "placed_bears" };
const Q_REACH: Qd = Qd { name: "reach_from", rel: "placed_reaches" };
const Q_REACH_G: Qd = Qd { name: "reach_from (given)", rel: "placed_reaches" };
const Q_STRIKES: Qd = Qd { name: "strikes_place", rel: "placed_strikes" };
const Q_ENDS: Qd = Qd { name: "move_ends", rel: "placed_mover" };
const Q_STEP: Qd = Qd { name: "find_step", rel: "placed_mover" };
const Q_PLAYED: Qd = Qd { name: "move_info (kinds played)", rel: "placed_mover" };
const Q_TAKEN: Qd = Qd { name: "move_info (kinds taken)", rel: "placed_takes" };

/// Ends of a move term as places: `(move f r f r)` and `(promote f r f r kind)`.
fn term_ends(nm: &Names, term: &str) -> Option<(u16, u16)> {
    let w: Vec<&str> = term.trim_matches(|c| c == '(' || c == ')').split(' ').collect();
    if w.len() >= 5 && (w[0] == "move" || w[0] == "promote") { Some((nm.place(w[1], w[2]), nm.place(w[3], w[4]))) } else { None }
}

pub struct Env<'a> { pub text: &'a str, pub names: &'a GameNames, pub nm: &'a Names, pub probes: usize, pub lower: bool }

/// The corpus of one seed: `games` games from the kernel's start (`listed`: the start is a position of the game's list, the module begun from the kernel's fluents) compared at every state.
fn walk<T: Tr>(t: &mut T, env: &Env, kernel: &KernelGame, listed: bool, games: usize, seed: u64, counts: &mut Counts, progress: Option<&AtomicU64>) {
    let mut rng = Rng::new(seed);
    let mut prng = Rng::new(seed ^ 0x5DEECE66D);
    let dec = dec_make(t, &env.names.kinds, &env.names.roles);
    let (roles, kinds) = (&env.names.roles, &env.names.kinds);
    let start_facts: BTreeSet<Fact> = kernel.state_facts(&kernel.initial()).into_iter().collect();
    if listed {
        if let Err(e) = t.restart_t(&facts_pairs(&start_facts)) { *counts.left_out.entry(format!("a listed start the module cannot restart at: {e}")).or_insert(0) += 1; return; }
    }
    let mut ev_prog: Option<Program> = if env.lower { Some(if listed { program_at(env.text, &start_facts) } else { Program::load(env.text).unwrap() }) } else { None };
    for g in 0..games {
        counts.games += 1;
        let mut ks = kernel.initial();
        let mut line: Vec<String> = vec![];
        loop {
            counts.states += 1;
            if let Some(p) = progress { p.fetch_add(1, Ordering::Relaxed); }
            let kf: Vec<Fact> = kernel.state_facts(&ks);
            let kfs: BTreeSet<Fact> = kf.iter().cloned().collect();
            let base = pairs_of(&kf);
            let ended = kernel.outcome(&ks).is_some();
            if ended { counts.ended += 1; }
            let cells = kernel.cells(&ks);
            let fen = kernel.fen(&ks);
            let turn = kernel.turn(&ks);
            let mut bad = 0usize;
            let mut acc = Acc { counts: &mut *counts, text: env.text, start: &start_facts, listed, names: env.names, nm: env.nm, kernel, seed, game: g, line: &line, fen: fen.clone(), ks: &ks };
            let lower_here = env.lower && ev_prog.is_some();
            // ------------------------------------------------ the state as it stands: what stands where, what bears, what strikes, the allowed moves' ends, what they play and take
            match t.state_of_facts_t(&base) {
                Err(e) => { *acc.counts.left_out.entry(format!("a state the module cannot read: {e}")).or_insert(0) += 1; }
                Ok(s0) => {
                    let rels = ["placed_cell", "placed_bears", "placed_strikes", "placed_mover", "placed_takes"];
                    let got = ask(t, &s0, &rels);
                    let m: BTreeMap<&str, BTreeSet<R4>> = rels.iter().zip(got.iter()).map(|(r, rows)| (*r, rows_set(&dec, r, rows, &mut bad))).collect();
                    if lower_here { lower_check(acc.counts, ev_prog.as_mut().unwrap(), env.nm, &base, &m, &rels, &dec, &got); }
                    // what stands where
                    let kc: BTreeSet<R4> = cells.iter().map(|(o, k, p)| [*p, *o as u16, *k as u16, NONE]).collect();
                    acc.compare(&Q_CELLS, &kc, &m["placed_cell"], &base, "the state as it stands");
                    let ka: BTreeSet<R4> = (0..64u16).filter_map(|p| kernel.kind_at(&ks, p).map(|k| [p, k as u16, NONE, NONE])).collect();
                    let ma: BTreeSet<R4> = m["placed_cell"].iter().map(|r| [r[0], r[2], NONE, NONE]).collect();
                    acc.compare(&Q_KIND_AT, &ka, &ma, &base, "the state as it stands");
                    // what every unit that stands bears on
                    let kb: BTreeSet<R4> = cells.iter().flat_map(|(o, k, p)| kernel.bearing_from(&ks, *o, *k, *p).into_iter().map(move |to| [*k as u16, *o as u16, *p, to])).collect();
                    acc.compare(&Q_BEARS, &kb, &m["placed_bears"], &base, "every unit that stands");
                    // what each side strikes
                    let mut kst = BTreeSet::new();
                    for o in 0..roles.len() { for p in 0..64u16 { if kernel.strikes_place(&ks, o, p) { kst.insert([o as u16, p, NONE, NONE]); } } }
                    acc.compare(&Q_STRIKES, &kst, &m["placed_strikes"], &base, "every place for each side");
                    // the moves
                    if ended {
                        *acc.counts.left_out.entry("a state the game calls over (move_ends, find_step, move_info)".into()).or_insert(0) += 1;
                    } else {
                        let mut kends = BTreeSet::new();
                        let mut kplay = BTreeSet::new();
                        let mut ktake = BTreeSet::new();
                        let mut kterms: Vec<(String, (u16, u16))> = vec![];
                        let mv = kernel.moves(&ks);
                        for m1 in &mv {
                            let (f, to) = kernel.move_ends(&ks, m1);
                            let (kind, cap) = kernel.move_info(&ks, m1);
                            kends.insert([f, to, NONE, NONE]);
                            kplay.insert([turn as u16, f, to, kind as u16]);
                            if let Some(j) = cap { ktake.insert([turn as u16, f, to, j as u16]); }
                            kterms.push((spec_term(&kernel.move_name(&ks, m1)), (f, to)));
                        }
                        let mends: BTreeSet<R4> = m["placed_mover"].iter().map(|r| [r[1], r[2], NONE, NONE]).collect();
                        acc.compare(&Q_ENDS, &kends, &mends, &base, "the allowed moves");
                        acc.compare(&Q_PLAYED, &kplay, &m["placed_mover"], &base, "the allowed moves");
                        acc.compare(&Q_TAKEN, &ktake, &m["placed_takes"], &base, "the allowed moves");
                        // the ends each allowed move's own term names, against the kernel's reading of its ends
                        let mut term_diff = BTreeSet::new();
                        for (term, e) in &kterms { if let Some(te) = term_ends(env.nm, term) { if te != *e { term_diff.insert([e.0, e.1, NONE, NONE]); } } }
                        acc.compare(&Q_ENDS, &term_diff, &BTreeSet::new(), &base, "the ends each allowed move's own term names");
                        // find_step: every pair the module's ends or the kernel's moves have, and a few drawn pairs
                        let mut pairs_asked: BTreeSet<(u16, u16)> = mends.iter().map(|r| (r[0], r[1])).collect();
                        pairs_asked.extend(kends.iter().map(|r| (r[0], r[1])));
                        for _ in 0..8 { pairs_asked.insert((prng.below(64) as u16, prng.below(64) as u16)); }
                        let allowed_terms: BTreeSet<String> = kterms.iter().map(|(x, _)| x.clone()).collect();
                        let (mut kfound, mut mfound) = (BTreeSet::new(), BTreeSet::new());
                        let mut not_allowed = BTreeSet::new();
                        for (f, to) in &pairs_asked {
                            if mends.contains(&[*f, *to, NONE, NONE]) { mfound.insert([*f, *to, NONE, NONE]); }
                            if let Some(m1) = kernel.find_step(&ks, *f, *to) {
                                kfound.insert([*f, *to, NONE, NONE]);
                                let term = spec_term(&kernel.move_name(&ks, &m1));
                                if !allowed_terms.contains(&term) { not_allowed.insert([*f, *to, NONE, NONE]); }
                                // the preferred one among several: a unit becoming the first kind of the list of those it may become (the kernel's preference), read as the module's term for the same ends
                                let queen = kterms.iter().find(|(x, e)| e == &(*f, *to) && x.starts_with("(promote") && x.ends_with(" queen)"));
                                if let Some((qt, _)) = queen { if &term != qt { not_allowed.insert([*f, *to, NONE, NONE]); } }
                            }
                        }
                        acc.compare(&Q_STEP, &kfound, &mfound, &base, "every pair of places the moves or a few drawn pairs");
                        acc.compare(&Q_STEP, &not_allowed, &BTreeSet::new(), &base, "the move found is an allowed move, the queen one among several");
                    }
                }
            }
            // ------------------------------------------------ the units of each side as the side to move, no passed-over place: where they may move to
            for (oi, oname) in roles.iter().enumerate() {
                let pairs = with_control(&base, oname);
                if let Ok(s1) = t.state_of_facts_t(&pairs) {
                    let rows = ask(t, &s1, &["placed_reaches"]);
                    let ms = rows_set(&dec, "placed_reaches", &rows[0], &mut bad);
                    if lower_here { lower_check(acc.counts, ev_prog.as_mut().unwrap(), env.nm, &pairs, &BTreeMap::from([("placed_reaches", ms.clone())]), &["placed_reaches"], &dec, &rows); }
                    let kr: BTreeSet<R4> = cells.iter().filter(|(o, _, _)| *o == oi).flat_map(|(o, k, p)| kernel.reach_from(&ks, *o, *k, *p).into_iter().map(move |to| [*k as u16, *o as u16, *p, to])).collect();
                    let mine: BTreeSet<R4> = ms.iter().filter(|r| r[1] as usize == oi).cloned().collect();
                    acc.compare(&Q_REACH, &kr, &mine, &pairs, &format!("every unit of {oname}, {oname} to move, no passed-over place"));
                }
            }
            // ------------------------------------------------ the units GIVEN on a place (standing there or not)
            for pi in 0..env.probes {
                let (o, k, p) = (prng.below(roles.len()), prng.below(kinds.len()), prng.below(64) as u16);
                let pairs = with_unit(&base, p, &roles[o], &kinds[k]);
                let Ok(s2) = t.state_of_facts_t(&pairs) else { *acc.counts.left_out.entry("a probe state the module cannot read".into()).or_insert(0) += 1; continue };
                let rows = ask(t, &s2, &["placed_bears", "placed_reaches"]);
                acc.counts.probes += 1;
                let (mb, mr) = (rows_set(&dec, "placed_bears", &rows[0], &mut bad), rows_set(&dec, "placed_reaches", &rows[1], &mut bad));
                if lower_here && pi < EV_PROBES { lower_check(acc.counts, ev_prog.as_mut().unwrap(), env.nm, &pairs, &BTreeMap::from([("placed_bears", mb.clone()), ("placed_reaches", mr.clone())]), &["placed_bears", "placed_reaches"], &dec, &rows); }
                let from = |s: &BTreeSet<R4>| -> BTreeSet<R4> { s.iter().filter(|r| r[0] == k as u16 && r[1] == o as u16 && r[2] == p).cloned().collect() };
                let kb: BTreeSet<R4> = kernel.bearing_from(&ks, o, k as u8, p).into_iter().map(|to| [k as u16, o as u16, p, to]).collect();
                let kr: BTreeSet<R4> = kernel.reach_from(&ks, o, k as u8, p).into_iter().map(|to| [k as u16, o as u16, p, to]).collect();
                let ctx = format!("given: {} {} on {}", roles[o], kinds[k], env.names.places[p as usize]);
                acc.compare(&Q_BEARS_G, &kb, &from(&mb), &pairs, &ctx);
                acc.compare(&Q_REACH_G, &kr, &from(&mr), &pairs, &ctx);
            }
            if bad > 0 { *acc.counts.left_out.entry("a row of the module the harness cannot read as numbers".into()).or_insert(0) += bad; }
            // the walk: the kernel's moves sorted by name, one picked by the seeded generator (the certificate's and translated_kernel's)
            let kmv = kernel.moves(&ks);
            if ended || kmv.is_empty() { break; }
            let mut sorted: Vec<(String, usize)> = kmv.iter().enumerate().map(|(i, m)| (kernel.move_name(&ks, m), i)).collect();
            sorted.sort();
            let (name, i) = sorted[rng.below(sorted.len())].clone();
            counts.plies += 1;
            line.push(name);
            ks = kernel.apply(&ks, &kmv[i]);
            let _ = &kfs;
        }
    }
}

/// THE LOWER LINK at one state: the module's rows of each definition asked against the evaluator's rows of the same relation at the same state.
fn lower_check(counts: &mut Counts, prog: &mut Program, nm: &Names, pairs: &Pairs, module: &BTreeMap<&str, BTreeSet<R4>>, rels: &[&str], _dec: &Dec, _raw: &[Vec<Vec<u32>>]) {
    counts.lower_states += 1;
    let theirs = evaluator_rows(prog, pairs, rels).expect("the evaluator runs");
    for rel in rels {
        let ev: BTreeSet<R4> = theirs.get(*rel).map(|s| s.iter().filter_map(|x| nm.row(rel, x)).collect()).unwrap_or_default();
        let m = &module[rel];
        let e = counts.lower.entry(rel.to_string()).or_insert((0, 0));
        e.0 += 1;
        e.1 += ev.len().max(m.len());
        let diff = ev.symmetric_difference(m).count();
        if diff > 0 { *counts.causes.entry(format!("lower link: {rel}: the module and the evaluator differ")).or_insert(0) += diff; }
    }
}

fn starts_of(variant: &str) -> Vec<String> {
    let p = rules_path(&format!("variants/{variant}.starts"));
    match std::fs::read_to_string(p) { Ok(s) => s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty() && !l.starts_with('#')).collect(), Err(_) => vec![] }
}

fn game_names(kernel: &KernelGame) -> GameNames { GameNames { places: kernel.place_names(), kinds: kernel.kind_names(), roles: kernel.roles() } }

/// The names the kernel gives, against the description's own sorts (once a game): the places are the squares of the sorts `file` and `rank`, file by file; the kinds are the members of the sort `kind`, sorted by text.
fn names_checked(text: &str, names: &GameNames, counts: &mut Counts) {
    let prog = Program::load(text).unwrap();
    let members = |sort: &str| -> Vec<String> {
        let mut v: Vec<String> = prog.sorts.get(sort).map(|ms| ms.iter().filter_map(|m| match m { cs_gdl::SortMember::Const(c) => Some(c.clone()), _ => None }).collect()).unwrap_or_default();
        v.sort();
        v
    };
    let (files, ranks, kinds) = (members("file"), members("rank"), members("kind"));
    let mut want = vec![];
    for f in &files { for r in &ranks { want.push(format!("{f}{r}")); } }
    let e = counts.compared.entry("place_names".into()).or_insert((0, 0));
    e.0 += 1;
    e.1 += want.len();
    if want != names.places { *counts.causes.entry("place_names: the kernel's names and the description's squares differ".into()).or_insert(0) += 1; }
    let e = counts.compared.entry("kind_names".into()).or_insert((0, 0));
    e.0 += 1;
    e.1 += kinds.len();
    if kinds != names.kinds { *counts.causes.entry(format!("kind_names: the kernel's {:?} and the description's {:?} differ", names.kinds, kinds)).or_insert(0) += 1; }
}

pub fn small_run<T: Tr>(variant: &str, text: &str) -> Counts {
    let mut t = T::make();
    let mut c = Counts::default();
    let kernel = KernelGame::new(variant, 518).expect("the kernel");
    let names = game_names(&kernel);
    let nm = Names::new(&names.kinds, &names.roles);
    names_checked(text, &names, &mut c);
    let env = Env { text, names: &names, nm: &nm, probes: PROBES_SMALL, lower: true };
    walk(&mut t, &env, &kernel, false, GAMES, SEED, &mut c, None);
    for fen in starts_of(variant) {
        let k = KernelGame::from_fen(variant, &fen).expect("a listed start");
        walk(&mut t, &env, &k, true, STARTS_GAMES, SEED, &mut c, None);
    }
    c
}

pub fn large_run<T: Tr>(variant: &str, text: &str, seeds: usize, games: usize, cores: usize, probes: usize) -> Counts {
    let kernel = KernelGame::new(variant, 518).expect("the kernel");
    let names = game_names(&kernel);
    let nm = Names::new(&names.kinds, &names.roles);
    let next = AtomicUsize::new(0);
    let progress = AtomicU64::new(0);
    let done = AtomicBool::new(false);
    let results: Mutex<Vec<(usize, Counts)>> = Mutex::new(vec![]);
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        let mut hs = vec![];
        for _ in 0..cores {
            hs.push(std::thread::Builder::new().stack_size(64 << 20).spawn_scoped(s, || {
                let mut t = T::make();
                let env = Env { text, names: &names, nm: &nm, probes, lower: false };
                loop {
                    let k = next.fetch_add(1, Ordering::SeqCst);
                    if k >= seeds { break; }
                    let mut c = Counts::default();
                    walk(&mut t, &env, &kernel, false, games, LARGE_SEED0 + k as u64, &mut c, Some(&progress));
                    results.lock().unwrap().push((k, c));
                }
            }).unwrap());
        }
        s.spawn(|| {
            let mut last = 0u64;
            let mut tick = 0;
            while !done.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(500));
                tick += 1;
                if tick % 60 == 0 {
                    let now = progress.load(Ordering::Relaxed);
                    println!("OBS 79.48 progress {variant}: {} s, {} of {seeds} seeds begun, {now} states ({:.0} a second since the last line), load {}", t0.elapsed().as_secs(), next.load(Ordering::SeqCst).min(seeds), (now - last) as f64 / 30.0, load_average());
                    last = now;
                }
            }
        });
        for h in hs { h.join().unwrap(); }
        done.store(true, Ordering::SeqCst);
    });
    let mut all = results.into_inner().unwrap();
    all.sort_by_key(|(k, _)| *k);
    let mut c = Counts::default();
    names_checked(text, &names, &mut c);
    for (_, x) in &all { c.merge(x); }
    c
}

fn esc(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() { match ch { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), c if (c as u32) < 32 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o.push('"');
    o
}

pub fn counts_json(variant: &str, kind: &str, c: &Counts, seconds: f64, cores: usize, seeds: usize, games: usize, probes: usize) -> String {
    let map = |m: &BTreeMap<String, usize>| format!("{{{}}}", m.iter().map(|(k, v)| format!("{}: {v}", esc(k))).collect::<Vec<_>>().join(", "));
    let pair = |m: &BTreeMap<String, (usize, usize)>| format!("{{{}}}", m.iter().map(|(k, v)| format!("{}: [{}, {}]", esc(k), v.0, v.1)).collect::<Vec<_>>().join(", "));
    let ex: Vec<String> = c.findings.iter().map(|f| format!("{{\"cause\": {}, \"seed\": {}, \"game\": {}, \"line\": {}, \"state\": {}, \"kernel\": {}, \"module\": {}, \"evaluator\": {}}}",
        esc(&f.cause), f.seed, f.game, esc(&f.line.join(" ")), esc(&f.state), esc(&f.kernel), esc(&f.module), esc(&f.evaluator))).collect();
    format!("{{\"game\": {}, \"kind\": {}, \"games\": {}, \"plies\": {}, \"states\": {}, \"probes\": {}, \"ended\": {}, \"compared\": {}, \"lower\": {}, \"lower_states\": {}, \"left_out\": {}, \"causes\": {}, \"examples\": [{}], \"seconds\": {:.1}, \"cores\": {cores}, \"seeds\": {seeds}, \"seed0\": {}, \"games_a_seed\": {games}, \"probes_a_state\": {probes}}}\n",
        esc(variant), esc(kind), c.games, c.plies, c.states, c.probes, c.ended, pair(&c.compared), pair(&c.lower), c.lower_states, map(&c.left_out), map(&c.causes), ex.join(", "), seconds, LARGE_SEED0)
}

fn emit(variant: &str, kind: &str, c: &Counts, seconds: f64, cores: usize, seeds: usize, games: usize, probes: usize) {
    let rows: usize = c.compared.values().map(|v| v.1).sum();
    let cmp: usize = c.compared.values().map(|v| v.0).sum();
    println!("OBS 79.48 {kind} {variant}: {} games, {} plies, {} states ({} ended), {} probes, {cmp} comparisons of {rows} rows over {} questions, lower link {} states {:?}, left out {:?}, disagreements {} {:?} ({seconds:.1} s, {cores} cores, {:.0} states a second)",
             c.games, c.plies, c.states, c.ended, c.probes, c.compared.len(), c.lower_states, c.lower, c.left_out, c.disagreements(), c.causes, c.states as f64 / seconds.max(1e-9));
    for (q, v) in &c.compared { println!("OBS 79.48 {kind} {variant} question {q}: {} comparisons, {} rows", v.0, v.1); }
    for f in &c.findings { println!("FINDING 79.48 {variant} [{}] seed {} game {} line {:?}\n   state {}\n   kernel    {}\n   module    {}\n   evaluator {}", f.cause, f.seed, f.game, f.line, f.state, f.kernel, f.module, f.evaluator); }
    if let Ok(p) = std::env::var("CS_PL_OUT") { std::fs::write(&p, counts_json(variant, kind, c, seconds, cores, seeds, games, probes)).expect("write the counts"); }
}

fn env_usize(name: &str, default: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default) }

fn small<T: Tr>(variant: &str) {
    let text = start_text(variant);
    let t0 = std::time::Instant::now();
    let c = small_run::<T>(variant, &text);
    emit(variant, "small", &c, t0.elapsed().as_secs_f64(), 1, 1, GAMES, PROBES_SMALL);
    assert!(c.states > 0 && c.lower_states > 0, "{variant}: nothing was compared");
}

fn large<T: Tr>(variant: &str) {
    let text = start_text(variant);
    let cores = env_usize("CS_PL_CORES", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let (seeds, games, probes) = (env_usize("CS_PL_SEEDS", 128), env_usize("CS_PL_GAMES", 100), env_usize("CS_PL_PROBES", PROBES_LARGE));
    let t0 = std::time::Instant::now();
    let c = large_run::<T>(variant, &text, seeds, games, cores, probes);
    emit(variant, "large", &c, t0.elapsed().as_secs_f64(), cores, seeds, games, probes);
}

// the small tests print their findings and never fail on a disagreement (a disagreement is a finding; the record and the gate read it); they fail only where nothing was compared
#[test]
fn chess_the_kernel_and_the_placed_module_on_the_certificates_games() { small::<placed_chess::Game>("chess"); }
#[test]
fn atomic_the_kernel_and_the_placed_module_on_the_certificates_games() { small::<placed_atomic::Game>("atomic"); }
#[test]
fn threecheck_the_kernel_and_the_placed_module_on_the_certificates_games() { small::<placed_threecheck::Game>("3check"); }
#[test]
fn kingofthehill_the_kernel_and_the_placed_module_on_the_certificates_games() { small::<placed_kingofthehill::Game>("kingofthehill"); }
#[test]
fn antichess_the_kernel_and_the_placed_module_on_the_certificates_games() { small::<placed_antichess::Game>("antichess"); }
#[test]
fn chess960_the_kernel_and_the_placed_module_on_the_certificates_games() { small::<placed_chess960::Game>("chess960"); }
#[test]
fn horde_the_kernel_and_the_placed_module_on_the_certificates_games() { small::<placed_horde::Game>("horde"); }

#[test]
#[ignore = "the large run: all cores, a few minutes, run by hand (discovery/placed_translated.py run chess); the machine is shared"]
fn chess_large_the_kernel_and_the_placed_module_on_the_fixed_seed_list() { large::<placed_chess::Game>("chess"); }
#[test]
#[ignore = "the large run: all cores, a few minutes, run by hand (discovery/placed_translated.py run atomic); the machine is shared"]
fn atomic_large_the_kernel_and_the_placed_module_on_the_fixed_seed_list() { large::<placed_atomic::Game>("atomic"); }
#[test]
#[ignore = "the large run: all cores, a few minutes, run by hand (discovery/placed_translated.py run 3check); the machine is shared"]
fn threecheck_large_the_kernel_and_the_placed_module_on_the_fixed_seed_list() { large::<placed_threecheck::Game>("3check"); }
#[test]
#[ignore = "the large run: all cores, a few minutes, run by hand (discovery/placed_translated.py run kingofthehill); the machine is shared"]
fn kingofthehill_large_the_kernel_and_the_placed_module_on_the_fixed_seed_list() { large::<placed_kingofthehill::Game>("kingofthehill"); }
#[test]
#[ignore = "the large run: all cores, a few minutes, run by hand (discovery/placed_translated.py run antichess); the machine is shared"]
fn antichess_large_the_kernel_and_the_placed_module_on_the_fixed_seed_list() { large::<placed_antichess::Game>("antichess"); }
#[test]
#[ignore = "the large run: all cores, a few minutes, run by hand (discovery/placed_translated.py run chess960); the machine is shared"]
fn chess960_large_the_kernel_and_the_placed_module_on_the_fixed_seed_list() { large::<placed_chess960::Game>("chess960"); }
#[test]
#[ignore = "the large run: all cores, a few minutes, run by hand (discovery/placed_translated.py run horde); the machine is shared"]
fn horde_large_the_kernel_and_the_placed_module_on_the_fixed_seed_list() { large::<placed_horde::Game>("horde"); }

/// THE NEGATIVE CONTROL: the kernel of another game's rules held to the chess module is caught, with the evaluator siding with the module; so the comparison can fail.
#[test]
fn a_kernel_of_other_rules_is_caught_by_the_placed_comparison() {
    let text = start_text("chess");
    let mut t = placed_chess::Game::make();
    let kernel = KernelGame::new("antichess", 518).unwrap();
    let names = game_names(&kernel);
    let nm = Names::new(&names.kinds, &names.roles);
    let env = Env { text: &text, names: &names, nm: &nm, probes: 8, lower: false };
    let mut c = Counts::default();
    walk(&mut t, &env, &kernel, false, 3, SEED, &mut c, None);
    println!("OBS 79.48 control: the antichess kernel against the chess module, {} states, disagreements {} {:?}", c.states, c.disagreements(), c.causes);
    assert!(c.disagreements() > 0, "a kernel of other rules was not caught");
    let f = c.findings.iter().find(|f| f.evaluator.contains("does NOT have") || f.evaluator.contains("HAS")).expect("a finding with the evaluator's answer");
    println!("FINDING 79.48 control [{}] evaluator: {}", f.cause, f.evaluator);
}

/// THE SECOND NEGATIVE CONTROL (the lower link): the module of one description held to the evaluator of another's (the module of chess, the evaluator of atomic's text) is caught by the comparison of rows, so the lower link is one that can fail.
#[test]
fn a_module_held_to_the_evaluator_of_another_description_is_caught_by_the_lower_link() {
    let text = start_text("atomic");
    let mut t = placed_chess::Game::make();
    let kernel = KernelGame::new("chess", 518).unwrap();
    let names = game_names(&kernel);
    let nm = Names::new(&names.kinds, &names.roles);
    let env = Env { text: &text, names: &names, nm: &nm, probes: 2, lower: true };
    let mut c = Counts::default();
    walk(&mut t, &env, &kernel, false, 2, SEED, &mut c, None);
    let low: usize = c.causes.iter().filter(|(k, _)| k.starts_with("lower link")).map(|(_, v)| v).sum();
    println!("OBS 79.48 control: the chess module against the evaluator of atomic's text, {} states, lower-link differences {low} {:?}", c.states, c.causes.keys().filter(|k| k.starts_with("lower link")).collect::<Vec<_>>());
    assert!(low > 0, "a module held to another description's evaluator was not caught");
}
