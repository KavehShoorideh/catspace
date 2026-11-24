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
use cs_game::outside::spec_term;
use cs_game::rng::Rng;
use cs_game::{Fact, Game as _};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
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

/// Findings kept whole for each cause (the count of a cause is kept whole).
pub const KEEP: usize = 3;

impl Counts {
    pub fn merge(&mut self, o: &Counts) {
        self.games += o.games; self.plies += o.plies; self.states += o.states; self.moves += o.moves; self.nexts += o.nexts; self.ended += o.ended; self.results += o.results;
        for (k, v) in &o.left_out { *self.left_out.entry(k.clone()).or_insert(0) += v; }
        for (k, v) in &o.causes { *self.causes.entry(k.clone()).or_insert(0) += v; }
        for f in &o.findings {
            if self.findings.iter().filter(|x| x.cause == f.cause).count() < KEEP { self.findings.push(f.clone()); }
        }
    }
    pub fn disagreements(&self) -> usize { self.causes.values().sum() }
}

/// What the evaluator is asked at the state of a disagreement.
enum Probe { Turn, Ended, Result, Moves, Next(String) }

fn facts_text(f: &BTreeSet<Fact>) -> String { f.iter().map(|x| format!("({} {})", x.relation, x.args.join(" "))).collect::<Vec<_>>().join(" ") }
fn set_text(s: &BTreeSet<String>) -> String { s.iter().cloned().collect::<Vec<_>>().join(" ") }

/// The class of a move term, for naming a cause (not a rule: the shape of the term and the number of files it crosses).
fn move_class(term: &str) -> String {
    let w: Vec<&str> = term.trim_matches(|c| c == '(' || c == ')').split(' ').collect();
    let two_files = w.len() >= 5 && w[1].len() == 1 && w[3].len() == 1 && (w[1].as_bytes()[0] as i32 - w[3].as_bytes()[0] as i32).abs() == 2;
    if two_files && w[0] == "move" { "two-file move".into() } else { w[0].to_string() }
}

fn relations_differing(a: &BTreeSet<Fact>, b: &BTreeSet<Fact>) -> String {
    let (ka, kb): (BTreeSet<&str>, BTreeSet<&str>) = (a.difference(b).map(|f| f.relation.as_str()).collect(), b.difference(a).map(|f| f.relation.as_str()).collect());
    format!("kernel+ {:?} translated+ {:?}", ka, kb)
}

fn ask_evaluator(text: &str, facts: &BTreeSet<Fact>, probe: &Probe, kernel: &str, translated: &str) -> String {
    let ev = evaluator_at(text, facts);
    let es = ev.initial();
    let answer = match probe {
        Probe::Turn => format!("{:?}", ev.mover(&es).map(|i| ev.roles()[i].clone())),
        Probe::Ended => format!("{}", ev.outcome(&es).is_some()),
        Probe::Result => format!("{:?}", ev.outcome(&es)),
        Probe::Moves => set_text(&ev.moves(&es).iter().map(|m| ev.move_name(&es, m)).collect()),
        Probe::Next(name) => match ev.find_move(&es, name) {
            None => return format!("the evaluator does not allow {name} here (kernel and translated both did)"),
            Some(m) => facts_text(&ev.base_facts(&ev.apply(&es, &m)).into_iter().collect()),
        },
    };
    let side = match (answer == kernel, answer == translated) { (true, false) => "agrees with the KERNEL", (false, true) => "agrees with the TRANSLATED module", (true, true) => "agrees with both", (false, false) => "agrees with NEITHER" };
    format!("{side}; its answer: {}", if answer.len() > 600 { format!("{}...", &answer[..600]) } else { answer })
}

struct Ctx<'a> { text: &'a str, seed: u64, game: usize, line: &'a [String], fen: String, facts: &'a BTreeSet<Fact> }

fn note(counts: &mut Counts, ctx: &Ctx, cause: String, probe: Probe, answers: &dyn Fn() -> (String, String)) {
    let n = counts.causes.entry(cause.clone()).or_insert(0);
    *n += 1;
    if counts.findings.iter().filter(|f| f.cause == cause).count() >= KEEP { return; }
    let (k, t) = answers();
    let ev = ask_evaluator(ctx.text, ctx.facts, &probe, &k, &t);
    counts.findings.push(Finding { cause, seed: ctx.seed, game: ctx.game, line: ctx.line.to_vec(), state: format!("{} | fluents {}", ctx.fen, facts_text(ctx.facts)), kernel: k, translated: t, evaluator: ev });
}

/// The corpus of one seed: `games` games from the kernel's start (`listed`: the start is a position of the game's list, the module begun from the kernel's fluents; otherwise the module's own start, itself compared), compared at every state. Takes the
/// translated module `t`, the description's text (for the evaluator's verdict), the kernel (it carries its start), the seed. Adds to `counts`. `progress` is bumped by the states done (the large run's reporter reads it).
pub fn walk<T: Tr>(t: &mut T, text: &str, kernel: &KernelGame, listed: bool, games: usize, seed: u64, counts: &mut Counts, progress: Option<&AtomicU64>) {
    let mut rng = Rng::new(seed);
    let troles: Vec<String> = t.roles_t().iter().map(|r| t.render_t(*r)).collect();
    let kroles = kernel.roles();
    for g in 0..games {
        counts.games += 1;
        let mut ks = kernel.initial();
        let mut kf: BTreeSet<Fact> = kernel.state_facts(&ks).into_iter().collect();
        let mut ts = if listed {
            match t.state_of_facts_t(&facts_pairs(&kf)) { Ok(s) => s, Err(_) => { *counts.left_out.entry("a listed start the module cannot read".into()).or_insert(0) += 1; continue; } }
        } else { t.initial_t() };
        let mut line: Vec<String> = vec![];
        let mut first = true;
        loop {
            counts.states += 1;
            if let Some(p) = progress { p.fetch_add(1, Ordering::Relaxed); }
            let ctx = Ctx { text, seed, game: g, line: &line, fen: kernel.fen(&ks), facts: &kf };
            // the state itself, at the first state of a game with the module's own start (every later state is compared as the next state of its parent)
            if first && !listed {
                let tf = fact_set(t.facts_of_t(&ts));
                if tf != kf { note(counts, &ctx, format!("start: {}", relations_differing(&kf, &tf)), Probe::Turn, &|| (facts_text(&kf), facts_text(&tf))); ts = t.state_of_facts_t(&facts_pairs(&kf)).expect("the module reads the kernel's start"); }
            }
            first = false;
            // (1) whose turn
            let km = kernel.mover(&ks).map(|r| kroles[r].clone());
            let tm = t.mover_t(&ts).map(|i| troles[i].clone());
            if km != tm { note(counts, &ctx, "turn".into(), Probe::Turn, &|| (format!("{km:?}"), format!("{tm:?}"))); }
            // (4) whether ended, (5) the result
            let (ko, to, tt) = (kernel.outcome(&ks), t.outcome_t(&ts), t.terminal_t(&ts));
            if ko.is_some() { counts.ended += 1; }
            if ko.is_some() != tt { note(counts, &ctx, "ended".into(), Probe::Ended, &|| (format!("{}", ko.is_some()), format!("{tt}"))); }
            if ko.is_some() || to.is_some() {
                counts.results += 1;
                if ko != to { note(counts, &ctx, "result".into(), Probe::Result, &|| (format!("{ko:?}"), format!("{to:?}"))); }
            }
            // (2) the allowed moves as a set of terms
            let kmv = kernel.moves(&ks);
            let kn: BTreeMap<String, usize> = kmv.iter().enumerate().map(|(i, m)| (spec_term(&kernel.move_name(&ks, m)), i)).collect();
            let tmv = t.moves_t(&ts);
            let tn: BTreeMap<String, u32> = tmv.iter().map(|m| (t.render_t(*m), *m)).collect();
            if tn.len() != tmv.len() { note(counts, &ctx, "moves: a move given twice by the translated module".into(), Probe::Moves, &|| (set_text(&kn.keys().cloned().collect()), format!("{} terms for {} moves", tn.len(), tmv.len()))); }
            let (ka, ta): (BTreeSet<String>, BTreeSet<String>) = (kn.keys().cloned().collect(), tn.keys().cloned().collect());
            if ka != ta {
                let (only_k, only_t): (BTreeSet<String>, BTreeSet<String>) = (ka.difference(&ta).cloned().collect(), ta.difference(&ka).cloned().collect());
                let cl = |s: &BTreeSet<String>| s.iter().map(|m| move_class(m)).collect::<BTreeSet<_>>();
                note(counts, &ctx, format!("moves: only kernel {:?}, only translated {:?}", cl(&only_k), cl(&only_t)), Probe::Moves, &|| (set_text(&ka), set_text(&ta)));
            }
            // which move is played (the certificate's walk: the kernel's names sorted, one picked by the seeded generator); none at an ended state
            let mut pick: Option<(String, usize)> = None;
            if ko.is_none() && !kmv.is_empty() {
                let mut sorted: Vec<(String, usize)> = kmv.iter().enumerate().map(|(i, m)| (kernel.move_name(&ks, m), i)).collect();
                sorted.sort();
                pick = Some(sorted[rng.below(sorted.len())].clone());
            }
            // (3) the next state after EVERY allowed move, in both: the fluents compared
            let mut chosen: Option<(BTreeSet<Fact>, _, _)> = None;
            for (term, i) in &kn {
                let Some(m) = tn.get(term) else { continue };
                let kchild = kernel.apply(&ks, &kmv[*i]);
                let kcf: BTreeSet<Fact> = kernel.state_facts(&kchild).into_iter().collect();
                let tchild = t.next_t(&ts, *m);
                let tcf = fact_set(t.facts_of_t(&tchild));
                counts.moves += 1;
                counts.nexts += 1;
                if kcf != tcf {
                    note(counts, &ctx, format!("next: {} after a {}", relations_differing(&kcf, &tcf), move_class(term)), Probe::Next(term.clone()), &|| (facts_text(&kcf), facts_text(&tcf)));
                }
                if pick.as_ref().map(|(n, _)| spec_term(n) == *term).unwrap_or(false) { chosen = Some((kcf, kchild, tchild)); }
            }
            let Some((name, i)) = pick else { break };
            counts.plies += 1;
            let (kcf, kchild, tchild) = match chosen {
                Some((kcf, kchild, tchild)) => (kcf, kchild, Some(tchild)),
                None => { let c = kernel.apply(&ks, &kmv[i]); (kernel.state_facts(&c).into_iter().collect(), c, None) } // (a move the module does not allow: already counted as a disagreement of the moves)
            };
            // the module's state goes on from its own next state when that has the kernel's fluents, else it is begun again from them (one cause is then not counted over the rest of the game)
            let tchild = match tchild {
                Some(s) if fact_set(t.facts_of_t(&s)) == kcf => s,
                _ => match t.state_of_facts_t(&facts_pairs(&kcf)) {
                    Ok(s) => s,
                    Err(_) => { *counts.left_out.entry("the rest of a game after a state the module cannot read from the kernel's fluents".into()).or_insert(0) += 1; break; }
                },
            };
            line.push(name);
            ks = kchild;
            kf = kcf;
            ts = tchild;
        }
    }
}

fn starts_of(variant: &str) -> Vec<String> {
    let p = rules_path(&format!("variants/{variant}.starts"));
    match std::fs::read_to_string(p) { Ok(s) => s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty() && !l.starts_with('#')).collect(), Err(_) => vec![] }
}

/// The certificate's corpus: GAMES from the kernel's start and STARTS_GAMES from each position listed for the game; the count of states is what the certificate's `positions` is.
pub fn small_run<T: Tr>(variant: &str, text: &str) -> Counts {
    let mut t = T::make();
    let mut c = Counts::default();
    let kernel = KernelGame::new(variant, 518).expect("the kernel");
    walk(&mut t, text, &kernel, false, GAMES, SEED, &mut c, None);
    for fen in starts_of(variant) {
        let k = KernelGame::from_fen(variant, &fen).expect("a listed start");
        walk(&mut t, text, &k, true, STARTS_GAMES, SEED, &mut c, None);
    }
    c
}

/// The large corpus on `cores` threads: seeds LARGE_SEED0.., `games` games each, drawn off one counter; counts merged in seed order; progress printed every 30 seconds.
pub fn large_run<T: Tr>(variant: &str, text: &str, seeds: usize, games: usize, cores: usize) -> Counts {
    let kernel = KernelGame::new(variant, 518).expect("the kernel");
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
                loop {
                    let k = next.fetch_add(1, Ordering::SeqCst);
                    if k >= seeds { break; }
                    let mut c = Counts::default();
                    walk(&mut t, text, &kernel, false, games, LARGE_SEED0 + k as u64, &mut c, Some(&progress));
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
                    println!("OBS 79.46 progress {variant}: {} s, {} of {seeds} seeds begun, {now} states ({:.0} a second since the last line), load {}", t0.elapsed().as_secs(), next.load(Ordering::SeqCst).min(seeds), (now - last) as f64 / 30.0, load_average());
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
    for (_, x) in &all { c.merge(x); }
    c
}

fn esc(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() { match ch { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), c if (c as u32) < 32 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o.push('"');
    o
}

/// The counts as the JSON text the Python side reads.
pub fn counts_json(variant: &str, kind: &str, c: &Counts, seconds: f64, cores: usize, seeds: usize, games: usize) -> String {
    let map = |m: &BTreeMap<String, usize>| format!("{{{}}}", m.iter().map(|(k, v)| format!("{}: {v}", esc(k))).collect::<Vec<_>>().join(", "));
    let ex: Vec<String> = c.findings.iter().map(|f| format!("{{\"cause\": {}, \"seed\": {}, \"game\": {}, \"line\": {}, \"state\": {}, \"kernel\": {}, \"translated\": {}, \"evaluator\": {}}}",
        esc(&f.cause), f.seed, f.game, esc(&f.line.join(" ")), esc(&f.state), esc(&f.kernel), esc(&f.translated), esc(&f.evaluator))).collect();
    format!("{{\"game\": {}, \"kind\": {}, \"games\": {}, \"plies\": {}, \"states\": {}, \"moves\": {}, \"nexts\": {}, \"ended\": {}, \"results\": {}, \"left_out\": {}, \"causes\": {}, \"examples\": [{}], \"seconds\": {:.1}, \"cores\": {cores}, \"seeds\": {seeds}, \"seed0\": {}, \"games_a_seed\": {games}}}\n",
        esc(variant), esc(kind), c.games, c.plies, c.states, c.moves, c.nexts, c.ended, c.results, map(&c.left_out), map(&c.causes), ex.join(", "), seconds, LARGE_SEED0)
}

fn emit(variant: &str, kind: &str, c: &Counts, seconds: f64, cores: usize, seeds: usize, games: usize) {
    println!("OBS 79.46 {kind} {variant}: {} games, {} plies, {} states, {} allowed moves compared, {} next states compared, {} ended, {} results compared, left out {:?}, disagreements {} {:?} ({seconds:.1} s, {cores} cores, {:.0} states a second)",
             c.games, c.plies, c.states, c.moves, c.nexts, c.ended, c.results, c.left_out, c.disagreements(), c.causes, c.states as f64 / seconds.max(1e-9));
    for f in &c.findings { println!("FINDING 79.46 {variant} [{}] seed {} game {} line {:?}\n   state {}\n   kernel     {}\n   translated {}\n   evaluator  {}", f.cause, f.seed, f.game, f.line, f.state, f.kernel, f.translated, f.evaluator); }
    if let Ok(p) = std::env::var("CS_KT_OUT") { std::fs::write(&p, counts_json(variant, kind, c, seconds, cores, seeds, games)).expect("write the counts"); }
}

fn env_usize(name: &str, default: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default) }

fn small<T: Tr>(variant: &str, text: &str) {
    let t0 = std::time::Instant::now();
    let c = small_run::<T>(variant, text);
    emit(variant, "small", &c, t0.elapsed().as_secs_f64(), 1, 1, GAMES);
    assert!(c.states > 0 && c.disagreements() == 0, "{variant}: the kernel and the translated module disagree: {:?}", c.causes);
}

fn large<T: Tr>(variant: &str, text: &str) {
    let cores = env_usize("CS_KT_CORES", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let (seeds, games) = (env_usize("CS_KT_SEEDS", 128), env_usize("CS_KT_GAMES", 280));
    let t0 = std::time::Instant::now();
    let c = large_run::<T>(variant, text, seeds, games, cores);
    emit(variant, "large", &c, t0.elapsed().as_secs_f64(), cores, seeds, games);
}

#[test]
fn chess_the_kernel_and_the_translated_module_agree_on_the_certificates_games() { small::<chess_fast::Game>("chess", &description(None)); }
#[test]
fn atomic_the_kernel_and_the_translated_module_agree_on_the_certificates_games() { small::<atomic_fast::Game>("atomic", &description(Some("variants/atomic.gdl"))); }

#[test]
#[ignore = "the large run: all cores, about five minutes, run by hand (discovery/kernel_translated.py run chess); the machine is shared"]
fn chess_large_the_kernel_and_the_translated_module_on_the_fixed_seed_list() { large::<chess_fast::Game>("chess", &description(None)); }
#[test]
#[ignore = "the large run: all cores, about five minutes, run by hand (discovery/kernel_translated.py run atomic); the machine is shared"]
fn atomic_large_the_kernel_and_the_translated_module_on_the_fixed_seed_list() { large::<atomic_fast::Game>("atomic", &description(Some("variants/atomic.gdl"))); }

/// THE NEGATIVE CONTROL: a kernel of other rules (atomic: a capture explodes) held to the module of chess is caught, with the evaluator of the chess description agreeing with the module; so the comparison above is one that can fail.
#[test]
fn a_kernel_of_other_rules_is_caught_and_the_evaluator_sides_with_the_module() {
    let text = description(None);
    let mut t = chess_fast::Game::make();
    let kernel = KernelGame::new("atomic", 518).unwrap();
    let mut c = Counts::default();
    walk(&mut t, &text, &kernel, false, 3, SEED, &mut c, None);
    println!("OBS 79.46 control: the atomic kernel against the chess module, {} states, disagreements {} {:?}", c.states, c.disagreements(), c.causes);
    assert!(c.disagreements() > 0, "a kernel of other rules was not caught");
    let f = c.findings.iter().find(|f| f.cause.starts_with("next")).expect("a next-state finding");
    println!("FINDING 79.46 control [{}] evaluator: {}", f.cause, f.evaluator);
    assert!(f.evaluator.contains("TRANSLATED"), "{}", f.evaluator);
}
