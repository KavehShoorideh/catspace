//! PLAN 79.59 PARTS D AND E (THE JUDGE OF THE MINED LEMMAS; SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules; ledger rows 59, 57, 58). THIS TEXT IS THE INTERFACE, written before the body.
//! A test file: it may name the games and the relations of the descriptions; the layer crates and the translator do not.
//!
//! THE JUDGE. TAKES a spec file (CS_MINED_SPEC; written by scripts/lemma_miner_run.py: for each game with a mined lemma, the test relation, its answer, its hits relation, the domain relation, the hit relation of each read and the text of each read) and the SEGMENTS: runs of consecutive states of
//! each game (the games of a file for the first game, CS_FACTS_GAMES, CS_FACTS_N games of at most CS_FACTS_MAX plies; seeded walks for the others). The modules are `generated/<game>_fast.rs` (every pass but carry: the module WITHOUT the lemma) and `generated_mined/<game>_mined.rs` (the same passes and the lemma read as a rewrite by pass (29): the module WITH it).
//!   THE GATE (zero tolerance). At every state of every segment the module with the lemma gives the module without it: the side to move, the end, the outcome, the allowed moves, and EVERY relation the module without it makes (`all_relations`, as sets of rows). A difference is counted, the segment it is in is named (game, index, ply), the run goes on.
//!   THE METRIC. (1) Tests for each allowed row: the module without the lemma tests the test relation at every key of its domain (the rows of the domain relation); with it only at the keys of the hits relation (the others take the answer of the null test). Printed: the domain, the hits and the allowed keys a state, and the two ratios to the allowed keys.
//!      (2) The clock, on the machine as it is (the load is printed in the line): nanoseconds a state of the allowed moves (`moves_into_t`), each module on the same states, best of CS_MINED_REPS passes, the two interleaved.
//!   THE STATEMENTS. The null test's class is the side to act; in the states where the null test holds for it (the goal unit of the side to act is borne on) the keys that are allowed are printed with the reads whose hit relation holds them (CS_MINED_SHOW states at most): which literal of which clause each allowed key changes the row of.
//! RETURNS OBS lines (`OBS 79.59 D ...`) and `MINED ...` lines for the printing scripts; the run FAILS on a difference unless CS_MINED_KEEP_GOING is set (the miner reads the lines either way).
//! WHO CALLS: scripts/lemma_miner_run.py `judge`. NOT on the engine's path.
mod mined_common;
#[macro_use]
mod translated_common;
mod variants_common;
use cs_game::kernel::KernelGame;
use cs_game::Game;
use mined_common::*;
use std::collections::{BTreeMap, BTreeSet};
use translated_common::*;
use variants_common::kernel;

include!("generated/chess_fast.rs");
include!("generated_mined/chess_mined.rs");
impl_tr!(chess_fast, topdown);
impl_tr!(chess_mined, topdown);
#[cfg(judge_all)]
include!("generated/chess960_fast.rs");
#[cfg(judge_all)]
include!("generated_mined/chess960_mined.rs");
#[cfg(judge_all)]
impl_tr!(chess960_fast, topdown);
#[cfg(judge_all)]
impl_tr!(chess960_mined, topdown);
#[cfg(judge_all)]
include!("generated/3check_fast.rs");
#[cfg(judge_all)]
include!("generated_mined/3check_mined.rs");
#[cfg(judge_all)]
impl_tr!(threecheck_fast, topdown);
#[cfg(judge_all)]
impl_tr!(threecheck_mined, topdown);
#[cfg(judge_all)]
include!("generated/kingofthehill_fast.rs");
#[cfg(judge_all)]
include!("generated_mined/kingofthehill_mined.rs");
#[cfg(judge_all)]
impl_tr!(kingofthehill_fast, topdown);
#[cfg(judge_all)]
impl_tr!(kingofthehill_mined, topdown);
#[cfg(judge_all)]
include!("generated/horde_fast.rs");
#[cfg(judge_all)]
include!("generated_mined/horde_mined.rs");
#[cfg(judge_all)]
impl_tr!(horde_fast, topdown);
#[cfg(judge_all)]
impl_tr!(horde_mined, topdown);

/// What the spec says of one game's lemma.
#[derive(Default, Clone, Debug)]
struct Spec { game: String, test: String, answer: String, hits: String, domain: String, null: String, hit: Vec<String>, read: Vec<String>, cone: Vec<String>, dviews: Vec<String> }

fn specs() -> Vec<Spec> {
    let path = std::env::var("CS_MINED_SPEC").expect("CS_MINED_SPEC names the spec file");
    let mut out: Vec<Spec> = vec![];
    for l in std::fs::read_to_string(path).unwrap().lines() {
        let w: Vec<&str> = l.splitn(2, ' ').collect();
        match w[0] {
            "game" => { let f: Vec<&str> = w[1].split(' ').collect(); out.push(Spec { game: f[0].into(), test: f[1].into(), answer: f[2].into(), hits: f[3].into(), domain: f[4].into(), null: f[5].into(), ..Default::default() }); }
            "hit" => out.last_mut().unwrap().hit.push(w[1].into()),
            "read" => out.last_mut().unwrap().read.push(w[1].into()),
            "dviews" => out.last_mut().unwrap().dviews = w[1].split(' ').map(|x| x.to_string()).collect(),
            "cone" => out.last_mut().unwrap().cone = w[1].split(' ').map(|x| x.to_string()).collect(),
            _ => {}
        }
    }
    out
}

type Rels = BTreeMap<&'static str, BTreeSet<Vec<String>>>;
fn rels<T: Tr>(t: &T, v: Vec<(&'static str, Vec<Vec<u32>>)>) -> Rels { v.into_iter().map(|(n, r)| (n, r.into_iter().map(|row| row.iter().map(|x| t.render_t(*x)).collect()).collect())).collect() }

#[derive(Default)]
struct Tally {
    states: u64, segments: u64, relations: u64, differs: u64, failing: Vec<String>,
    domain: u64, hits: u64, allowed: u64, hits_allowed: u64, refused_hits: u64,
    per_read: Vec<u64>,
    shown: u64, borne: u64,
}

fn place_name(f: &str, r: &str) -> String { format!("{f}{r}") }

fn judge<F: Tr, M: Tr>(game: &str, spec: &Spec, rep: &str, segs: &[Vec<State>], fast: &mut F, mined: &mut M, show: usize) -> (Tally, Vec<Vec<u32>>) {
    let k: KernelGame = kernel(game);
    let mut t = Tally { per_read: vec![0; spec.hit.len()], ..Default::default() };
    let mut order: Vec<Vec<u32>> = vec![];
    for (si, seg) in segs.iter().enumerate() {
        t.segments += 1;
        for (ply, s) in seg.iter().enumerate() {
            let facts = pairs(&k.state_facts(s));
            let sf = fast.state_of_facts_t(&facts).expect("the module reads the state");
            let sm = mined.state_of_facts_t(&facts).expect("the module with the lemma reads the state");
            let (af, am) = (fast.all_t(&sf, &[]), mined.all_t(&sm, &[]));
            let rf = rels(fast, af);
            let rm = rels(mined, am);
            let mut bad = false;
            let mut named: Vec<String> = vec![];
            for (name, rows) in &rf {
                t.relations += 1;
                match rm.get(name) {
                    Some(r) if r == rows => {}
                    None if spec.cone.iter().any(|c| c == name) => {}          // a relation made only to answer the test is not made where nothing reads the test: the replacement is held to the test below
                    other => { bad = true; if named.len() < 4 { named.push(format!("{name} ({} rows against {})", rows.len(), other.map_or(-1, |r| r.len() as i64))); } }
                }
            }
            // the replacement of the test (what the questions read in its place) has the rows of the test
            t.relations += 1;
            if rm.get(rep) != rf.get(spec.test.as_str()) { bad = true; named.push(format!("the replacement {rep} against {}", spec.test)); }
            let mv_f: BTreeSet<String> = fast.moves_t(&sf).iter().map(|m| fast.render_t(*m)).collect();
            let mv_m: BTreeSet<String> = mined.moves_t(&sm).iter().map(|m| mined.render_t(*m)).collect();
            if mv_f != mv_m || fast.mover_t(&sf).map(|i| fast.render_t(fast.roles_t()[i])) != mined.mover_t(&sm).map(|i| mined.render_t(mined.roles_t()[i])) || fast.outcome_t(&sf) != mined.outcome_t(&sm) || fast.terminal_t(&sf) != mined.terminal_t(&sm) { bad = true; }
            t.states += 1;
            if bad {
                t.differs += 1;
                if t.failing.len() < 8 { t.failing.push(format!("{game} segment {si} ply {ply}: {}", named.join("; "))); }
            }
            // the metric: the keys of the domain, the hits and the allowed keys
            let empty = BTreeSet::new();
            let domain = rm.get(spec.domain.as_str()).unwrap_or(&empty);
            let hits = rm.get(spec.hits.as_str()).unwrap_or(&empty);
            let test = rf.get(spec.test.as_str()).unwrap_or(&empty);
            let keys_of = |d: &BTreeSet<Vec<String>>| -> BTreeSet<Vec<String>> { d.iter().map(|r| r[r.len() - 4..].to_vec()).collect() };
            let dom_keys = keys_of(domain);
            let allowed: BTreeSet<Vec<String>> = dom_keys.difference(test).cloned().collect();
            t.domain += dom_keys.len() as u64;
            t.hits += hits.len() as u64;
            t.allowed += allowed.len() as u64;
            t.hits_allowed += hits.iter().filter(|h| allowed.contains(*h)).count() as u64;
            t.refused_hits += hits.iter().filter(|h| !allowed.contains(*h)).count() as u64;
            for (i, hn) in spec.hit.iter().enumerate() { t.per_read[i] += rm.get(hn.as_str()).map_or(0, |x| x.len() as u64); }
            // the statements: the states where the null test holds for the side to act
            if let Some(mi) = mined.mover_t(&sm) {
                let side = mined.render_t(mined.roles_t()[mi]);
                let borne = rm.get(spec.null.as_str()).map_or(false, |x| x.iter().any(|r| r[0] == side));
                if borne {
                    t.borne += 1;
                    if (t.shown as usize) < show {
                        t.shown += 1;
                        println!("MINED STATE {game} segment {si} ply {ply}: {} ; side to act {side}; the null test holds for it; keys of the domain {}, hits {}, allowed {}", k.fen(s), dom_keys.len(), hits.len(), allowed.len());
                        for key in &allowed {
                            let by: Vec<usize> = spec.hit.iter().enumerate().filter(|(_, hn)| rm.get(hn.as_str()).map_or(false, |x| x.contains(key))).map(|(i, _)| i).collect();
                            // the differing rows of the key, by the relation they are read off: the places (the file and the rank at the fifth and sixth place of a row of the view)
                            // a row of a view of 8 places puts a unit on a place (its side and kind are in the row); a row of a view of 6 places names a place whose unit the move takes off (what stands there is read from the state's cell facts)
                            let mut rows: Vec<String> = vec![];
                            for v in &spec.dviews { if let Some(rs) = rf.get(v.as_str()) { for r in rs.iter().filter(|r| r[..4] == key[..]) {
                                if r.len() == 8 { rows.push(format!("{v}: puts {} {} on {}", r[6], r[7], place_name(&r[4], &r[5]))); }
                                else if r.len() >= 6 {
                                    let held = facts.iter().find(|(rel, a)| rel == "cell" && a[0] == r[4] && a[1] == r[5]).map(|(_, a)| format!("{} {}", a[2], a[3]));
                                    rows.push(match held { Some(u) => format!("{v}: takes {u} off {}", place_name(&r[4], &r[5])), None => format!("{v}: nothing stands on {}", place_name(&r[4], &r[5])) });
                                }
                            } } }
                            println!("MINED ALLOWED {game}: from {} to {} meets read(s) {:?}; its differing rows: {}", place_name(&key[0], &key[1]), place_name(&key[2], &key[3]), by, rows.join(", "));
                        }
                        for key in dom_keys.iter().filter(|x| !allowed.contains(*x) && hits.contains(*x)).take(3) {
                            let by: Vec<usize> = spec.hit.iter().enumerate().filter(|(_, hn)| rm.get(hn.as_str()).map_or(false, |x| x.contains(key))).map(|(i, _)| i).collect();
                            println!("MINED REFUSED-AFTER-TEST {game}: from {} to {} meets read(s) {:?} and the test refuses it", place_name(&key[0], &key[1]), place_name(&key[2], &key[3]), by);
                        }
                    }
                }
            }
            order.push(vec![si as u32, ply as u32]);
        }
    }
    (t, order)
}

fn clock<F: Tr, M: Tr>(game: &str, segs: &[Vec<State>], fast: &mut F, mined: &mut M, reps: usize) -> (f64, f64, usize) {
    let k = kernel(game);
    let mut fs: Vec<F::S> = vec![];
    let mut ms: Vec<M::S> = vec![];
    for seg in segs {
        for s in seg {
            let facts = pairs(&k.state_facts(s));
            fs.push(fast.state_of_facts_t(&facts).unwrap());
            ms.push(mined.state_of_facts_t(&facts).unwrap());
        }
    }
    let (mut bf, mut bm) = (f64::MAX, f64::MAX);
    let mut buf = Vec::with_capacity(256);
    for _ in 0..reps.max(1) {
        let t0 = std::time::Instant::now();
        for s in &fs { fast.moves_into_t(s, &mut buf); }
        bf = bf.min(t0.elapsed().as_nanos() as f64 / fs.len().max(1) as f64);
        let t0 = std::time::Instant::now();
        for s in &ms { mined.moves_into_t(s, &mut buf); }
        bm = bm.min(t0.elapsed().as_nanos() as f64 / ms.len().max(1) as f64);
    }
    if std::env::var("CS_MINED_PROFILE").is_ok() {
        // where the time of the allowed moves goes, by piece (the relations a piece makes), for each module: the dearest pieces
        for (label, top) in [("without", { fast.profiling_t(true); fast.reset_profile_t(); for s in &fs { fast.moves_into_t(s, &mut buf); } let mut v = fast.pieces_t(); fast.profiling_t(false); v.sort_by(|a, b| b.3.cmp(&a.3)); v.into_iter().take(8).map(|(n, _s, c, ns, _r, _v, _p)| format!("{n} {c} calls {ns} ns")).collect::<Vec<_>>() }),
                             ("with", { mined.profiling_t(true); mined.reset_profile_t(); for s in &ms { mined.moves_into_t(s, &mut buf); } let mut v = mined.pieces_t(); mined.profiling_t(false); v.sort_by(|a, b| b.3.cmp(&a.3)); v.into_iter().take(8).map(|(n, _s, c, ns, _r, _v, _p)| format!("{n} {c} calls {ns} ns")).collect::<Vec<_>>() })] {
            println!("OBS 79.59 D {game} profile {label} the lemma, the dearest pieces of {} states: {}", fs.len(), top.join("; "));
        }
    }
    (bf, bm, fs.len())
}

fn report(game: &str, spec: &Spec, t: &Tally, c: (f64, f64, usize), secs: f64) {
    let per = |x: u64| x as f64 / t.states.max(1) as f64;
    println!("OBS 79.59 D {game} gate: {} states in {} segments, {} relations compared, {} states differ; the first failing segments: {:?}", t.states, t.segments, t.relations, t.differs, t.failing);
    println!("OBS 79.59 D {game} narrowness: a state holds {:.2} keys of the domain, {:.2} allowed keys, {:.2} hits (keys tested with the lemma; {:.2} of them allowed, {:.2} refused); tests for each allowed row: without the lemma {:.3}, with it {:.3}",
        per(t.domain), per(t.allowed), per(t.hits), per(t.hits_allowed), per(t.refused_hits), t.domain as f64 / t.allowed.max(1) as f64, t.hits as f64 / t.allowed.max(1) as f64);
    for (i, n) in t.per_read.iter().enumerate() { println!("MINED READ {game} {i} {} {:.4} {:.4}", spec.hit[i], per(*n), *n as f64 / t.hits.max(1) as f64); }
    println!("OBS 79.59 D {game} clock: the allowed moves of a state {:.0} ns without the lemma, {:.0} ns with it (ratio {:.2}) over {} states; load {}; {:.1} s in all", c.0, c.1, c.1 / c.0.max(1.0), c.2, load_average(), secs);
    println!("MINED RESULT {game} differs {} states {} tests_without {:.4} tests_with {:.4} ns_without {:.0} ns_with {:.0} borne {}", t.differs, t.states, t.domain as f64 / t.allowed.max(1) as f64, t.hits as f64 / t.allowed.max(1) as f64, c.0, c.1, t.borne);
}

macro_rules! one {
    ($f:ident, $m:ident, $name:expr, $spec:expr, $segs:expr, $show:expr, $reps:expr, $keep:expr) => {{
        let t0 = std::time::Instant::now();
        let (mut fast, mut mined) = (<$f::Game as Tr>::make(), <$m::Game as Tr>::make());
        let (t, _o) = judge($name, $spec, $m::LEMMAS[0].replacement, $segs, &mut fast, &mut mined, $show);
        let c = clock($name, $segs, &mut fast, &mut mined, $reps);
        report($name, $spec, &t, c, t0.elapsed().as_secs_f64());
        if t.differs > 0 && !$keep { panic!("the module with the lemma differs from the module without it at {} states of {}", t.differs, $name); }
    }};
}

#[test]
#[ignore = "the judge, by hand (plan 79.59 parts D, E): CS_MINED_SPEC names the spec file, CS_FACTS_GAMES the games file; scripts/lemma_miner_run.py judge"]
fn judge_all() {
    let path = std::env::var("CS_FACTS_GAMES").expect("CS_FACTS_GAMES names the file of games");
    let (n, max) = (env_usize("CS_FACTS_N", 30), env_usize("CS_FACTS_MAX", 40));
    let (show, reps, keep) = (env_usize("CS_MINED_SHOW", 12), env_usize("CS_MINED_REPS", 3), std::env::var("CS_MINED_KEEP_GOING").is_ok());
    for spec in specs() {
        let game = spec.game.as_str();
        let segs = if game == "chess" { states_of_lines("chess", &lines_of(&path, n), max) } else { seeded_states(&kernel(game), n, max) };
        println!("OBS 79.59 D {game}: {} segments of at most {max} plies ({})", segs.len(), if game == "chess" { "games of the file" } else { "seeded walks" });
        match game {
            "chess" => one!(chess_fast, chess_mined, "chess", &spec, &segs, show, reps, keep),
            #[cfg(judge_all)]
            "chess960" => one!(chess960_fast, chess960_mined, "chess960", &spec, &segs, show, reps, keep),
            #[cfg(judge_all)]
            "3check" => one!(threecheck_fast, threecheck_mined, "3check", &spec, &segs, show, reps, keep),
            #[cfg(judge_all)]
            "kingofthehill" => one!(kingofthehill_fast, kingofthehill_mined, "kingofthehill", &spec, &segs, show, reps, keep),
            #[cfg(judge_all)]
            "horde" => one!(horde_fast, horde_mined, "horde", &spec, &segs, show, reps, keep),
            other => println!("OBS 79.59 D {other}: no module with the lemma is compiled into this test (cfg judge_all), not judged"),
        }
    }
}
