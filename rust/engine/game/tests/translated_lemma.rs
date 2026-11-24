//! PLAN 79.45 PARTS C AND D (SYSTEM_SPEC 11, "One lemma machinery, used twice" and "The crux of layer 2 is a translator ..."; piece: rules; ledger rows 49, 45): A LEMMA READ AS A REWRITE IS HELD TO THE PLAIN MODULE OF THE BASE DESCRIPTION, AND A LEMMA IS CERTIFIED AGAIN BY TRANSLATED CODE.
//! PART C, the harness. The module with a lemma (generated/chess_lemma.rs: every pass, the pass gated, the lemma exposed_by_cases; and, compiled with `--cfg singles`, generated/scratch/chess_lm_plain.rs: the rewrite and no pass, chess_lm_nogate.rs: every pass but gated) is held,
//! piece by piece, to the plain module of the SAME base description (generated/chess.rs: the description without the lemma's delta) by the harness of translated_common/pieces.rs, through a VIEW of the module: the view hands the harness the relations the plain module has and, in place of the left relation,
//! the replacement the rewrite made (so the harness holds the replacement to the plain left relation at every state it visits), and keeps back the relations the delta and the rewrite added (the plain module has no such relation). Then, at the same states, `lemma_at` holds what the view does not show:
//! the flag has a row exactly where the condition, read from the PLAIN module's rows, holds; the module's own left relation (made whole by `all_relations`) has the plain rows; the replacement has the plain rows; and, where the condition holds, the right relation has the plain left rows (the lemma, read again at every state the harness visits).
//! The questions (the mover, the allowed moves, the end, the outcome, the next state) and every relation the module makes are compared as before. Any disagreement fails the test with the state and the relation.
//! PART D, the certificate of the lemma by translated code. `certificate_*` plays states with the module and, where the condition holds (read from the module's own rows of the relations the condition names, not from the flag), compares the left relation with the right one as the certificate of discovery/rule_lemma_certify.py does with the
//! evaluator: the states, the states where it held, the rows of each side, the rows compared, the disagreements, printed as OBS lines. Small: the 208 fixtures of the lemma and, when LEMMA_CORPUS names a file, the states of the 20 seeded games of the 79.44 certificate (scripts/rule_lemma_translated_record.py writes them with the certificate's own player,
//! so the counts can be held to its counts). Large: `certificate_large` (ignored; LEMMA_GAMES games over LEMMA_THREADS threads of the kernel's seeded random games, 300 plies at most as in the certificate, and the record is written to LEMMA_RECORD when it names a file).
//! This file names the first game on purpose (a test file may).
#![allow(unexpected_cfgs)]
mod variants_common;
#[macro_use]
mod translated_common;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::rng::Rng;
use cs_game::Game;
use std::collections::{BTreeMap, BTreeSet};
use translated_common::pieces::*;
use translated_common::*;
use variants_common::fixture;

include!("generated/chess.rs");
include!("generated/chess_lemma.rs");
#[cfg(singles)]
include!("generated/scratch/chess_lm_plain.rs");
#[cfg(singles)]
include!("generated/scratch/chess_lm_nogate.rs");
impl_tr!(chess);
impl_tr!(chess_lemma, passes);
#[cfg(singles)]
impl_tr!(chess_lm_plain);
#[cfg(singles)]
impl_tr!(chess_lm_nogate, passes);

const SEED: u64 = 77; // the certificate's (discovery/certify.py)
const FUSE: usize = 300; // the certificate's fuse of a game, in plies (rule_lemma_certify.PLY_FUSE)

// ------------------------------------------------------------------------------------------------------------------------------------------ what a module says of its lemmas

#[derive(Clone, Debug)]
pub struct LemmaData { pub name: String, pub left: String, pub right: String, pub rep: String, pub flag: String, pub evidence: String, pub condition: Vec<(String, Vec<String>, bool)> }

pub trait LemmaTr { fn lemmas_t(&self) -> Vec<LemmaData>; }

macro_rules! impl_lm {
    ($m:ident) => {
        impl LemmaTr for $m::Game {
            fn lemmas_t(&self) -> Vec<LemmaData> {
                $m::LEMMAS.iter().map(|l| LemmaData { name: l.name.to_string(), left: l.left.to_string(), right: l.right.to_string(), rep: l.replacement.to_string(), flag: l.flag.to_string(), evidence: l.evidence.to_string(),
                    condition: l.condition.iter().map(|(r, a, n)| (r.to_string(), a.iter().map(|x| x.to_string()).collect(), *n)).collect() }).collect()
            }
        }
    };
}
impl_lm!(chess_lemma);
#[cfg(singles)]
impl_lm!(chess_lm_plain);
#[cfg(singles)]
impl_lm!(chess_lm_nogate);

type Rows = BTreeMap<String, BTreeSet<Vec<String>>>;

fn rows_by_name<T: Tr>(t: &T, all: &[(&'static str, Vec<Vec<u32>>)]) -> Rows {
    all.iter().map(|(n, rs)| (n.to_string(), rs.iter().map(|r| r.iter().map(|x| t.render_t(*x)).collect()).collect())).collect()
}

/// Whether a condition (literals: relation, arguments, negated) holds over rows by relation name: a join of the positive literals, then the negated ones and `differ` on each binding.
pub fn cond_holds(rows: &Rows, cond: &[(String, Vec<String>, bool)]) -> bool {
    let mut bindings: Vec<BTreeMap<String, String>> = vec![BTreeMap::new()];
    for (rel, args, neg) in cond {
        if *neg || rel == "differ" { continue; }
        let mut next = vec![];
        if let Some(rs) = rows.get(rel) {
            for b in &bindings {
                for r in rs {
                    if r.len() != args.len() { continue; }
                    let mut nb = b.clone();
                    let mut ok = true;
                    for (a, v) in args.iter().zip(r.iter()) {
                        if a.starts_with('?') {
                            match nb.get(a) { Some(x) if x != v => { ok = false; break; } Some(_) => {} None => { nb.insert(a.clone(), v.clone()); } }
                        } else if a != v { ok = false; break; }
                    }
                    if ok { next.push(nb); }
                }
            }
        }
        bindings = next;
    }
    bindings.iter().any(|b| {
        let sub = |a: &String| b.get(a).cloned().unwrap_or_else(|| a.clone());
        cond.iter().all(|(rel, args, neg)| {
            if rel == "differ" { sub(&args[0]) != sub(&args[1]) }
            else if *neg { let row: Vec<String> = args.iter().map(&sub).collect(); !rows.get(rel).map_or(false, |rs| rs.contains(&row)) }
            else { true }
        })
    })
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the view the piece-by-piece harness is given

/// A module with lemmas as the harness of the plain module sees it: the relations of the plain module, the left relation of a lemma read from its replacement, nothing the rewrite or the delta added.
pub struct View<T: Tr + LemmaTr> { pub m: T, pub lemmas: Vec<LemmaData>, pub plain_names: BTreeSet<String> }

impl<T: Tr + LemmaTr> View<T> {
    pub fn new(m: T, plain: &mut chess::Game) -> View<T> {
        let st = plain.initial_t();
        let plain_names = plain.all_t(&st, &[]).iter().map(|(n, _)| n.to_string()).collect();
        let lemmas = m.lemmas_t();
        View { m, lemmas, plain_names }
    }
}

impl<T: Tr + LemmaTr> Tr for View<T> {
    type S = T::S;
    fn make() -> Self { unimplemented!("a view is made with the plain module's names") }
    fn initial_t(&self) -> Self::S { self.m.initial_t() }
    fn roles_t(&self) -> Vec<u32> { self.m.roles_t() }
    fn mover_t(&mut self, s: &Self::S) -> Option<usize> { self.m.mover_t(s) }
    fn moves_t(&mut self, s: &Self::S) -> Vec<u32> { self.m.moves_t(s) }
    fn next_t(&mut self, s: &Self::S, m: u32) -> Self::S { self.m.next_t(s, m) }
    fn terminal_t(&mut self, s: &Self::S) -> bool { self.m.terminal_t(s) }
    fn outcome_t(&mut self, s: &Self::S) -> Option<Vec<i64>> { self.m.outcome_t(s) }
    fn render_t(&self, t: u32) -> String { self.m.render_t(t) }
    fn term_t(&mut self, text: &str) -> Result<u32, String> { self.m.term_t(text) }
    fn state_of_facts_t(&mut self, f: &[(String, Vec<String>)]) -> Result<Self::S, String> { self.m.state_of_facts_t(f) }
    fn facts_of_t(&self, s: &Self::S) -> Vec<(String, Vec<String>)> { self.m.facts_of_t(s) }
    fn statics_t(&self) -> Vec<(&'static str, Vec<Vec<u32>>)> { self.m.statics_t() }
    fn all_t(&mut self, s: &Self::S, does: &[(u32, u32)]) -> Vec<(&'static str, Vec<Vec<u32>>)> {
        let rows = self.m.all_t(s, does);
        let mut out = vec![];
        for (n, r) in rows.iter() {
            if let Some(l) = self.lemmas.iter().find(|l| l.left == *n) {
                let rep = rows.iter().find(|(m, _)| *m == l.rep).map(|x| x.1.clone()).unwrap_or_default();
                out.push((*n, rep));
            } else if self.plain_names.contains(*n) {
                out.push((*n, r.clone()));
            }
        }
        out
    }
    fn noop_t(&self) -> u32 { self.m.noop_t() }
    fn sym_t(&self, name: &str) -> Option<u32> { self.m.sym_t(name) }
    fn profiling_t(&mut self, on: bool) { self.m.profiling_t(on) }
    fn reset_profile_t(&mut self) { self.m.reset_profile_t() }
    fn pieces_t(&self) -> Vec<(&'static str, bool, u64, u64, u64, u64, u64)> { self.m.pieces_t() }
    fn passes_t(&self) -> Vec<&'static str> { self.m.passes_t() }
    fn demanded_t(&self) -> Vec<(&'static str, Vec<usize>, bool)> { self.m.demanded_t() }
    fn copied_t(&self) -> Vec<&'static str> { self.m.copied_t() }
    fn ask_t(&mut self, s: &Self::S, does: &[(u32, u32)], q: &[(String, Vec<u32>)]) -> Vec<(Vec<Vec<u32>>, Option<bool>)> { self.m.ask_t(s, does, q) }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ what the lemma adds to the comparison

#[derive(Default, Debug, Clone)]
pub struct LemmaCounts {
    pub states: usize,
    /// states where the flag had a row, and the condition read from the plain rows held (they must be the same states)
    pub flag_agrees: usize,
    pub held: usize,
    /// rows of the plain left relation, over all states: compared with the module's own left relation and with the replacement
    pub rows_left: usize,
    /// rows of the left relation where the condition held: compared with the rows of the right relation
    pub rows_compared: usize,
    pub bad: Vec<String>,
}

fn diff(a: &BTreeSet<Vec<String>>, b: &BTreeSet<Vec<String>>) -> String { format!("only first {:?}, only second {:?}", a.difference(b).take(3).collect::<Vec<_>>(), b.difference(a).take(3).collect::<Vec<_>>()) }

/// At one state (as facts): the lemma's claims about the module read against the plain module (see the head of this file).
pub fn lemma_at<B: Tr + LemmaTr>(a: &mut chess::Game, b: &mut View<B>, facts: &[(String, Vec<String>)], label: &str, c: &mut LemmaCounts) {
    let (sa, sb) = (a.state_of_facts_t(facts).unwrap(), b.state_of_facts_t(facts).unwrap());
    let all_a = a.all_t(&sa, &[]);
    let plain = rows_by_name(a, &all_a);
    let all_b = b.m.all_t(&sb, &[]);
    let module = rows_by_name(&b.m, &all_b);
    c.states += 1;
    for l in b.lemmas.clone() {
        let empty = BTreeSet::new();
        let (pl, ml, mr, mp) = (plain.get(&l.left).unwrap_or(&empty), module.get(&l.left).unwrap_or(&empty), module.get(&l.right).unwrap_or(&empty), module.get(&l.rep).unwrap_or(&empty));
        let held = cond_holds(&plain, &l.condition);
        let flag = module.get(&l.flag).map_or(false, |r| !r.is_empty());
        if held != flag { c.bad.push(format!("{label}: the flag {}: the condition on the plain rows is {held}, the flag has {} rows", l.flag, module.get(&l.flag).map_or(0, |r| r.len()))); } else { c.flag_agrees += 1; }
        if pl != ml { c.bad.push(format!("{label}: the module's own {}: {}", l.left, diff(pl, ml))); }
        if pl != mp { c.bad.push(format!("{label}: the replacement {} of {}: only plain {:?}, only replacement {:?}", l.rep, l.left, pl.difference(mp).take(3).collect::<Vec<_>>(), mp.difference(pl).take(3).collect::<Vec<_>>())); }
        c.rows_left += pl.len();
        if held {
            c.held += 1;
            c.rows_compared += pl.len();
            if pl != mr { c.bad.push(format!("{label}: the lemma {}: the condition holds and {} has other rows than the plain {}: only plain {:?}, only {} {:?}", l.name, l.right, l.left, pl.difference(mr).take(3).collect::<Vec<_>>(), l.right, mr.difference(pl).take(3).collect::<Vec<_>>())); }
        }
    }
}

fn report_c(game: &str, variant: &str, what: &str, p: &PieceCounts, c: &LemmaCounts) {
    println!("OBS 79.45 C {game} {variant} {what}: {} piece-by-piece comparisons ({} states, {} question answers, {} next states, {} relations in full, {} key probes, {} static relations whole), {} states with the lemma read again ({} with the flag as the condition says, {} where it held, {} rows of the left relation compared with the module's own and the replacement, {} rows compared with the right relation), {} disagreements",
             p.total(), p.states, p.questions, p.nexts, p.relations, p.keys, p.statics, c.states, c.flag_agrees, c.held, c.rows_left, c.rows_compared, c.bad.len());
}

fn fail(c: &LemmaCounts) { if !c.bad.is_empty() { panic!("{} disagreements, the first: {}", c.bad.len(), c.bad[0]); } }

fn statics<B: Tr + LemmaTr>(variant: &str) {
    let plain = chess::Game::make();
    let b = B::make();
    let mut c = PieceCounts::default();
    statics_equal(&plain, &b, &mut c).unwrap_or_else(|e| panic!("chess {variant}: {e}"));
    println!("OBS 79.45 C chess {variant} static: {} static relations whole equal the plain ones, 0 disagreements; passes {:?}, lemmas {:?}", c.statics, b.passes_t(), b.lemmas_t().iter().map(|l| (l.name.clone(), l.evidence.clone())).collect::<Vec<_>>());
    assert!(c.statics > 20);
}

fn fixtures<B: Tr + LemmaTr>(variant: &str) {
    let fx = fixture("chess");
    let mut a = chess::Game::make();
    let mut b = View::new(B::make(), &mut a);
    let (mut c, mut l) = (PieceCounts::default(), LemmaCounts::default());
    for p in &fx.pos {
        let facts = facts_pairs(&p.facts);
        let label = format!("chess {variant} fixture {} ({})", p.label, p.text);
        pieces_at(&mut a, &mut b, &facts, &None, &label, true, &mut c).unwrap_or_else(|e| panic!("{e}"));
        lemma_at(&mut a, &mut b, &facts, &label, &mut l);
        for st in fx.steps.iter().filter(|st| st.parent == p.label) {
            let mv = Some(spec_term(&st.name));
            pieces_at(&mut a, &mut b, &facts, &mv, &format!("{label} {}", st.name), true, &mut c).unwrap_or_else(|e| panic!("{e}"));
        }
    }
    report_c("chess", variant, "fixture", &c, &l);
    fail(&l);
    assert!(c.states > 100);
}

fn random_games<B: Tr + LemmaTr>(variant: &str, games: usize, rel_games: usize) {
    let kernel = KernelGame::new("chess", 518).unwrap();
    let mut a = chess::Game::make();
    let mut b = View::new(B::make(), &mut a);
    let walk = corpus(&kernel, &mut a, games, SEED);
    let (mut c, mut l) = (PieceCounts::default(), LemmaCounts::default());
    let t0 = std::time::Instant::now();
    for (g, line) in walk.iter().enumerate() {
        for (i, (facts, mv)) in line.iter().enumerate() {
            let label = format!("chess {variant} game {g} ply {i}");
            pieces_at(&mut a, &mut b, facts, mv, &label, g < rel_games, &mut c).unwrap_or_else(|e| panic!("{e}"));
            if g < rel_games { lemma_at(&mut a, &mut b, facts, &label, &mut l); }
        }
    }
    report_c("chess", variant, &format!("seeded games (seed {SEED}, {games} games, relations in the first {rel_games}; {:.1} s)", t0.elapsed().as_secs_f64()), &c, &l);
    fail(&l);
    assert!(l.held > 0 && l.rows_compared > 0, "the lemma was never read where it held");
}

macro_rules! lemma_tests {
    ($m:ident, $variant:literal, $s:ident, $f:ident, $r:ident) => {
        #[test]
        fn $s() { statics::<$m::Game>($variant); }
        #[test]
        fn $f() { fixtures::<$m::Game>($variant); }
        #[test]
        fn $r() { random_games::<$m::Game>($variant, 20, 20); }
    };
}

lemma_tests!(chess_lemma, "lemma", chess_lemma_static_relations_are_the_plain_ones, chess_lemma_every_piece_at_every_fixture_position_is_the_plain_one, chess_lemma_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
lemma_tests!(chess_lm_plain, "lm_plain", chess_lm_plain_static_relations_are_the_plain_ones, chess_lm_plain_every_piece_at_every_fixture_position_is_the_plain_one, chess_lm_plain_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
lemma_tests!(chess_lm_nogate, "lm_nogate", chess_lm_nogate_static_relations_are_the_plain_ones, chess_lm_nogate_every_piece_at_every_fixture_position_is_the_plain_one, chess_lm_nogate_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
