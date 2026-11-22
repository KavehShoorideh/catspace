//! PLAN 79.24 PART C: THE TEST OF BEING DONE OF LAYER 3, HOW GOOD A POSITION IS (SYSTEM_SPEC 11 "a layer's test uses only that layer and what is below it", "layer 3 holds two implementations: value and the measure of progress"; ledger rows 35 and 36; piece: value and measure of progress).
//! It uses layer 1 (cs_rep), layer 2 (cs_game) and this layer (cs_value), no crate above, no search and no outside program. The references are DATA: tests/fixtures/value.fixture and ending.fixture, made once by
//! catspace/incubator/analyst/discovery/layer3_fixtures_print.py (human games of the test split in four rating bands, generated games that fill the thin endgames, the endgame table's own best play in the small endings; the format is described there).
//! This file reads the fixtures as opaque text: it names no object of any game; the names of the game, its units and its positions come from the files.
//! THE FIVE THINGS, each printing its OBS lines: (1) THE VALUE AGAINST RESULTS in ten bins, and against the plain count of material it was fitted beside, on the same positions; (2) THE MEASURE AGAINST THE TABLE along the table's best play in the small endings, exceptions listed;
//! (3) DO THE TWO SAY THE SAME: the rank agreement of the value and the measure with each other and with the truth, with the positions the two order oppositely; (4) WRITTEN AND READ BACK through the long-term memory of one game, and another game's reader does not see it; (5) THE RATE.
//! The fixture of the value holds positions of quiet rows only (nothing stands to be lost by inclusion for either side, as the table was fitted): the value of a position that is not quiet is what a search settles first, above this layer.
use cs_game::kernel::KernelGame;
use cs_game::rules::{Ruled, Rules, StateText};
use cs_game::Game;
use cs_rep::serde_json::Value as Json;
use cs_rep::statement::{Holder, Says, Standing, Statement};
use cs_rep::store::{GameKey, LongTerm, Partition, Reader};
use cs_value::progress::{Geometry, Progress};
use cs_value::worth::{decoders, Count, CountModel, KindModel, Value, COUNT_MODEL, KIND_MODEL};
use cs_value::{kendall_tau, Answer, Assess, Question};
use std::cmp::Ordering;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

type R = Ruled<KernelGame>;

// ------------------------------------------------------------------------------------------------------------------------------ the fixtures, read as opaque text

struct Pos { sample: String, game: u32, units: u32, y: usize, text: String }
struct SampleRow { name: String, kind: String, games: u32, positions: u32, end: u32, share: f64 }
struct ValueFx { name: String, decl: String, kind: KindModel, count: CountModel, samples: Vec<SampleRow>, pos: Vec<Pos>, bytes: usize }
struct Line { set: String, class: String, id: String, role: usize, won: bool, steps: Vec<(String, i64)> }
struct EndingFx { name: String, decl: String, measures: Vec<(String, String, u32, String)>, lines: Vec<Line>, bytes: usize }

fn fixture(name: &str) -> (String, usize) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let n = text.len();
    (text, n)
}

fn nums(v: &Json, k: &str) -> Vec<f64> { v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() }

fn value_fixture() -> ValueFx {
    let (text, bytes) = fixture("value.fixture");
    let mut fx = ValueFx { name: String::new(), decl: String::new(), kind: KindModel { kinds: vec![], intercept: [0.0; 3], coef: [vec![], vec![], vec![]], count: 0, source: String::new() },
                           count: CountModel { intercept: [0.0; 3], coef: [0.0; 3], count: 0, source: String::new() }, samples: vec![], pos: vec![], bytes };
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "game" => { fx.name = f[1].into(); fx.decl = f[2].into(); }
            "model" => {
                let v: Json = cs_rep::serde_json::from_str(f[2]).unwrap();
                let three = |k: &str| -> [f64; 3] { <[f64; 3]>::try_from(nums(&v, k)).unwrap() };
                match f[1] {
                    "kind" => fx.kind = KindModel { kinds: v["kinds"].as_array().unwrap().iter().map(|k| k.as_str().unwrap().to_string()).collect(), intercept: three("intercept"),
                                                    coef: [0, 1, 2].map(|c| v["coef"][c].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()), count: v["count"].as_u64().unwrap(), source: v["source"].as_str().unwrap().into() },
                    "count" => fx.count = CountModel { intercept: three("intercept"), coef: three("coef"), count: v["count"].as_u64().unwrap(), source: v["source"].as_str().unwrap().into() },
                    other => panic!("unknown model {other}"),
                }
            }
            "sample" => fx.samples.push(SampleRow { name: f[1].into(), kind: f[2].into(), games: f[3].parse().unwrap(), positions: f[4].parse().unwrap(), end: f[5].parse().unwrap(), share: f[6].parse().unwrap() }),
            "pos" => fx.pos.push(Pos { sample: f[1].into(), game: f[2].parse().unwrap(), units: f[4].parse().unwrap(), y: f[5].parse().unwrap(), text: f[6].into() }),
            other => panic!("unknown tag {other}"),
        }
    }
    fx
}

fn ending_fixture() -> EndingFx {
    let (text, bytes) = fixture("ending.fixture");
    let mut fx = EndingFx { name: String::new(), decl: String::new(), measures: vec![], lines: vec![], bytes };
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "game" => { fx.name = f[1].into(); fx.decl = f[2].into(); }
            "measure" => fx.measures.push((f[1].into(), f[2].into(), f[3].parse().unwrap(), f[4].into())),
            "line" => fx.lines.push(Line { set: f[1].into(), class: f[2].into(), id: f[3].into(), role: f[4].parse().unwrap(), won: f[5] == "1",
                                           steps: f[6].split(';').map(|s| { let (t, d) = s.rsplit_once('|').unwrap(); (t.to_string(), d.parse().unwrap()) }).collect() }),
            other => panic!("unknown tag {other}"),
        }
    }
    fx
}

fn rules_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules") }

fn open(name: &str, decl: &str) -> R { Ruled::open(KernelGame::new(name, 518).unwrap(), name, &rules_root(), decl).unwrap() }

fn state(g: &R, text: &str) -> <R as Game>::State { g.read_state_text(text).unwrap_or_else(|e| panic!("{text}: {e}")) }

fn with_measures(mut p: Partition, measures: &[(String, String, u32, String)]) -> Partition {
    for (class, term, d, sample) in measures {
        p = p.with_statement(Statement { id: format!("measure:{class}"), pattern: vec![], says: Says::Lower { term: term.clone(), within: *d }, standing: Standing::Conjecture(sample.clone()), holder: Holder::Ours, also: vec![] });
    }
    p
}

fn memory_of(g: &R, kind: &KindModel, count: &CountModel, measures: &[(String, String, u32, String)]) -> Reader {
    Reader::new(with_measures(Partition::new(g.rules_key()).with_entry(KIND_MODEL, Arc::new(kind.clone())).with_entry(COUNT_MODEL, Arc::new(count.clone())), measures))
}

/// the rule that makes a position an endgame position: at most this many units on the board in all (the convention of the earlier endgame coverage)
const ENDGAME_MEN: u32 = 8;

fn score_of(y: usize) -> f64 { [1.0, 0.5, 0.0][y] }

/// The ranks of tuples in their own order, ties equal (the number of distinct smaller tuples): a tuple becomes one number a rank correlation can read.
fn dense_rank(rows: &[Vec<f64>]) -> Vec<f64> {
    let mut ix: Vec<usize> = (0..rows.len()).collect();
    let cmp = |a: &Vec<f64>, b: &Vec<f64>| -> Ordering { for (x, y) in a.iter().zip(b) { let c = x.total_cmp(y); if c != Ordering::Equal { return c; } } a.len().cmp(&b.len()) };
    ix.sort_by(|&a, &b| cmp(&rows[a], &rows[b]));
    let mut out = vec![0.0; rows.len()];
    let mut r = 0.0;
    for k in 0..ix.len() {
        if k > 0 && cmp(&rows[ix[k - 1]], &rows[ix[k]]) != Ordering::Equal { r += 1.0; }
        out[ix[k]] = r;
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.total_cmp(b)); v[v.len() / 2] }

fn logloss(c: &cs_value::Chance, y: usize) -> f64 { -([c.win, c.draw, c.loss][y].max(1e-12)).ln() }

fn brier(c: &cs_value::Chance, y: usize) -> f64 { let p = [c.win, c.draw, c.loss]; (0..3).map(|k| (p[k] - if k == y { 1.0 } else { 0.0 }).powi(2)).sum() }

// ------------------------------------------------------------------------------------------------------------------------------ (1) the value against results

#[test]
fn layer3_the_value_against_held_out_results_in_ten_bins_and_against_the_plain_count_of_material() {
    let fx = value_fixture();
    let g = open(&fx.name, &fx.decl);
    let mem = memory_of(&g, &fx.kind, &fx.count, &[]);
    let value = Value::from_memory(&mem, KIND_MODEL).unwrap();
    let count = Count::from_memory(&mem, COUNT_MODEL).unwrap();
    // the samples and the share of endgame positions of every one (a sample under the thin share is filled by the generated games)
    for s in &fx.samples {
        let mine: Vec<&Pos> = fx.pos.iter().filter(|p| p.sample == s.name).collect();
        assert_eq!(mine.len() as u32, s.positions, "the sample line counts its positions");
        assert_eq!(mine.iter().filter(|p| p.units <= ENDGAME_MEN).count() as u32, s.end, "the sample line counts its endgame positions as the rule says: at most {ENDGAME_MEN} units on the board");
        println!("OBS 79.24 value sample {} ({}): {} games, {} positions, endgame positions {} = {:.4} of the sample", s.name, s.kind, s.games, s.positions, s.end, s.share);
    }
    assert!(fx.samples.iter().any(|s| s.kind == "human") && fx.samples.iter().any(|s| s.kind == "generated"), "human samples and the generated fill are both there");
    let humans: Vec<&SampleRow> = fx.samples.iter().filter(|s| s.kind == "human").collect();
    assert!(humans.len() >= 4, "human samples in four rating bands");
    // each position asked of both implementations, for the side to move
    let (mut rows, mut games_seen) = (vec![], std::collections::BTreeSet::new());
    for p in &fx.pos {
        let s = state(&g, &p.text);
        let me = g.mover(&s).expect("a fixture position is not over");
        let q = Question { game: &g, state: &s, side: me };
        let (a, b): (Answer, Answer) = (value.assess(&q).unwrap(), count.assess(&q).unwrap());
        let (ca, cb) = (a.chance.unwrap(), b.chance.unwrap());
        assert!((ca.win + ca.draw + ca.loss - 1.0).abs() < 1e-9 && (cb.win + cb.draw + cb.loss - 1.0).abs() < 1e-9);
        assert_eq!(a.rank, vec![ca.score()], "the rank of a value is the expected score");
        // asked for the other side, the same table gives the chances turned round
        let other = value.assess(&Question { game: &g, state: &s, side: g.other(me) }).unwrap().chance.unwrap();
        assert!((other.win - ca.loss).abs() < 1e-12 && (other.loss - ca.win).abs() < 1e-12 && (other.draw - ca.draw).abs() < 1e-12);
        assert!(matches!(a.sure, Some(Standing::Conjecture(_))), "a table fitted on a sample says so");
        rows.push((p, ca, cb));
        games_seen.insert((p.sample.clone(), p.game));
    }
    let n = rows.len() as f64;
    let (lv, lc) = (rows.iter().map(|(p, v, _)| logloss(v, p.y)).sum::<f64>() / n, rows.iter().map(|(p, _, c)| logloss(c, p.y)).sum::<f64>() / n);
    let (bv, bc) = (rows.iter().map(|(p, v, _)| brier(v, p.y)).sum::<f64>() / n, rows.iter().map(|(p, _, c)| brier(c, p.y)).sum::<f64>() / n);
    println!("OBS 79.24 value fixture {} bytes: {} positions of {} games; the value's table fitted on {} rows ({}), the plain count's on {} rows", fx.bytes, rows.len(), games_seen.len(), fx.kind.count, fx.kind.source, fx.count.count);
    println!("OBS 79.24 value log loss {lv:.4}, plain count log loss {lc:.4}; value Brier {bv:.4}, plain count Brier {bc:.4}, on the same {} positions", rows.len());
    for s in &fx.samples {
        let mine: Vec<&(&Pos, cs_value::Chance, cs_value::Chance)> = rows.iter().filter(|r| r.0.sample == s.name).collect();
        let m = mine.len() as f64;
        println!("OBS 79.24 value sample {}: log loss value {:.4}, plain count {:.4}", s.name, mine.iter().map(|(p, v, _)| logloss(v, p.y)).sum::<f64>() / m, mine.iter().map(|(p, _, c)| logloss(c, p.y)).sum::<f64>() / m);
    }
    // ten bins of equal count by the value's expected score
    let mut ix: Vec<usize> = (0..rows.len()).collect();
    ix.sort_by(|&a, &b| rows[a].1.score().total_cmp(&rows[b].1.score()).then(a.cmp(&b)));
    let mut seen_score = vec![];
    println!("OBS 79.24 value in ten bins (equal count, by the expected score the value gives the side to move): bin n | mean chance of win | share of wins seen | mean expected score | share of score seen");
    for b in 0..10 {
        let part = &ix[b * ix.len() / 10..(b + 1) * ix.len() / 10];
        let m = part.len() as f64;
        let (pw, ow) = (part.iter().map(|&i| rows[i].1.win).sum::<f64>() / m, part.iter().filter(|&&i| rows[i].0.y == 0).count() as f64 / m);
        let (ps, os) = (part.iter().map(|&i| rows[i].1.score()).sum::<f64>() / m, part.iter().map(|&i| score_of(rows[i].0.y)).sum::<f64>() / m);
        println!("OBS 79.24 bin {} {} | {pw:.3} | {ow:.3} | {ps:.3} | {os:.3}", b + 1, part.len());
        seen_score.push(os);
    }
    // the value orders the bins: the lowest bin scores below the highest, and the share seen rises from the lower half to the upper half
    let (lo, hi) = (seen_score[..5].iter().sum::<f64>() / 5.0, seen_score[5..].iter().sum::<f64>() / 5.0);
    assert!(seen_score[0] < seen_score[9] && lo < hi, "the share of score seen rises with the value: {seen_score:?}");
    // the value is not worse than the plain count of material it was fitted beside, on the same positions
    assert!(lv <= lc, "the value's log loss {lv} is above the plain count's {lc}");
}

// ------------------------------------------------------------------------------------------------------------------------------ (2) the measure against the table

struct Read { tuple: Vec<i64>, dist: i64, text: String }

/// The positions of each line that stand in the class the line began in (the same units of the two sides), read by the measure, and whether the line left the class (the table's best play may give a unit up: the class changes and a goal held is dropped
/// for that cause, as the wake drops it; the measure is not asked of a position of another class).
fn read_lines(g: &R, geo: &Geometry, fx: &EndingFx) -> Vec<(Vec<Read>, bool)> {
    let mem = Reader::new(with_measures(Partition::new(g.rules_key()), &fx.measures));
    let units_of = |s: &<R as Game>::State| { let mut u = g.units(s); u.sort(); u };
    let mut out = vec![];
    for l in fx.lines.iter() {
        let progress = Progress::from_memory(&mem, &format!("measure:{}", l.class), geo).unwrap();
        let mut v = vec![];
        let mut first_units = None;
        let mut left = false;
        for (text, dist) in l.steps.iter() {
            let s = state(g, text);
            assert_eq!(g.mover(&s), Some(l.role), "the attacker is to move at every position of a line");
            let u = units_of(&s);
            if first_units.get_or_insert_with(|| u.clone()) != &u { left = true; break; }
            let a = Assess::<R>::assess(&progress, &Question { game: g, state: &s, side: l.role }).unwrap();
            assert_eq!(a.rank, a.raw.clone().unwrap().iter().map(|x| -(*x as f64)).collect::<Vec<f64>>(), "the rank of a measure is its tuple negated");
            v.push(Read { tuple: a.raw.unwrap(), dist: *dist, text: text.clone() });
        }
        out.push((v, left));
    }
    out
}

fn tuple_text(t: &[i64]) -> String { t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(".") }

#[test]
fn layer3_the_measure_along_the_tables_best_play_never_rises_and_falls_within_its_stated_moves_exceptions_listed() {
    let fx = ending_fixture();
    let g = open(&fx.name, &fx.decl);
    let geo = Geometry::new(&g.board().unwrap());
    let all = read_lines(&g, &geo, &fx);
    println!("OBS 79.24 ending fixture {} bytes: {} lines of the table's best play, measures {:?}", fx.bytes, fx.lines.len(), fx.measures.iter().map(|m| (&m.0, &m.2)).collect::<Vec<_>>());
    let mut sets: Vec<String> = fx.lines.iter().map(|l| l.set.clone()).collect();
    sets.dedup();
    let mut exceptions: Vec<String> = vec![];
    let (mut total_pos, mut total_rises, mut total_slow, mut total_left) = (0usize, 0usize, 0usize, 0usize);
    for set in &sets {
        let (mut n_lines, mut n_pos, mut rises, mut slow, mut won, mut left_lines, mut dropped) = (0, 0, 0, 0, 0, 0, 0);
        for (li, l) in fx.lines.iter().enumerate().filter(|(_, l)| &l.set == set) {
            let d = fx.measures.iter().find(|m| m.0 == l.class).unwrap().2 as usize;
            let (v, left) = &all[li];
            n_lines += 1;
            won += l.won as usize;
            left_lines += *left as usize;
            n_pos += v.len();
            println!("OBS 79.24 ending {set} {} (attacker {}, ends {}{}): distance {} | measure {}", l.id, l.role, if l.won { "in the win" } else { "not in the win" }, if *left { ", the class changed after these positions" } else { "" },
                     v.iter().map(|r| r.dist.to_string()).collect::<Vec<_>>().join(" "), v.iter().map(|r| tuple_text(&r.tuple)).collect::<Vec<_>>().join(" "));
            for i in 0..v.len() {
                if i + 1 < v.len() && v[i + 1].tuple > v[i].tuple {
                    rises += 1;
                    exceptions.push(format!("{set} {} step {}: the measure ROSE {} -> {} (distance {} -> {}) at {}", l.id, i, tuple_text(&v[i].tuple), tuple_text(&v[i + 1].tuple), v[i].dist, v[i + 1].dist, v[i].text));
                }
                let moves_left = v.len() - i;                                     // our moves from this position to the end of the line or to the class's change
                let fell = (i + 1..v.len().min(i + 1 + d)).any(|j| v[j].tuple < v[i].tuple);
                if fell || (l.won && !left && moves_left <= d) { continue; }
                if *left && moves_left <= d { dropped += 1; continue; }            // the goal is dropped for a stated cause (the class changed) before it could fall: not a failure of the measure
                slow += 1;
                exceptions.push(format!("{set} {} step {}: no fall within {d} of our moves from {} (distance {}) at {}", l.id, i, tuple_text(&v[i].tuple), v[i].dist, v[i].text));
            }
        }
        println!("OBS 79.24 ending {set}: {n_lines} lines ({won} end in the win, {left_lines} leave the class), {n_pos} positions of the class with the attacker to move; the measure rose after {rises} of them; no fall within the stated moves at {slow}; the goal dropped by the class changing before it could fall at {dropped}");
        total_pos += n_pos; total_rises += rises; total_slow += slow; total_left += left_lines;
    }
    println!("OBS 79.24 ending in all: {total_pos} positions of the class, {total_left} lines leave the class, rises {total_rises}, no fall within the stated moves {total_slow}; the exceptions:");
    for e in &exceptions { println!("OBS 79.24 ending exception {e}"); }
    assert!(total_pos > 200, "the lines are read");
    // the ratchet of the first run: the numbers of the fixture as they were read at plan 79.24, so that a change that makes the measure worse is seen (a ratchet, not a claim that the measure is right)
    assert!(total_rises <= RISES_AT_FIRST_RUN && total_slow <= SLOW_AT_FIRST_RUN, "rises {total_rises} (was {RISES_AT_FIRST_RUN}), no fall in time {total_slow} (was {SLOW_AT_FIRST_RUN})");
}

const RISES_AT_FIRST_RUN: usize = 116;
const SLOW_AT_FIRST_RUN: usize = 32;

// ------------------------------------------------------------------------------------------------------------------------------ (3) do the two say the same

#[test]
fn layer3_value_and_measure_on_the_same_positions_rank_agreement_with_each_other_and_with_the_truth_and_where_they_order_two_positions_oppositely() {
    let vfx = value_fixture();
    let efx = ending_fixture();
    let g = open(&vfx.name, &vfx.decl);
    let geo = Geometry::new(&g.board().unwrap());
    let mem = memory_of(&g, &vfx.kind, &vfx.count, &efx.measures);
    let value = Value::from_memory(&mem, KIND_MODEL).unwrap();
    // (a) positions of whole games where the measure is defined (every group its term reads is present for the side to move)
    let states: Vec<_> = vfx.pos.iter().map(|p| state(&g, &p.text)).collect();
    for (class, _, _, _) in &efx.measures {
        let progress = Progress::from_memory(&mem, &format!("measure:{class}"), &geo).unwrap();
        let mut idx = vec![];
        for (i, s) in states.iter().enumerate() {
            let me = g.mover(s).unwrap();
            if progress.in_class(&g, s, me) { idx.push(i); }
        }
        let v: Vec<f64> = idx.iter().map(|&i| { let s = &states[i]; value.assess(&Question { game: &g, state: s, side: g.mover(s).unwrap() }).unwrap().rank[0] }).collect();
        let m_rank: Vec<Vec<f64>> = idx.iter().map(|&i| { let s = &states[i]; Assess::<R>::assess(&progress, &Question { game: &g, state: s, side: g.mover(s).unwrap() }).unwrap().rank }).collect();
        let m = dense_rank(&m_rank);
        let truth: Vec<f64> = idx.iter().map(|&i| score_of(vfx.pos[i].y)).collect();
        let distinct = { let mut s = m.clone(); s.sort_by(|a, b| a.total_cmp(b)); s.dedup(); s.len() };
        let t = |a: &[f64], b: &[f64]| kendall_tau(a, b).map_or("undefined".to_string(), |x| format!("{x:+.4}"));
        println!("OBS 79.24 agree whole games, measure {class}: {} of {} positions have the measure defined ({} distinct values of it); rank agreement (Kendall tau-b): value with measure {}, value with the result {}, measure with the result {}", idx.len(), states.len(), distinct, t(&v, &m), t(&v, &truth), t(&m, &truth));
        // the pairs the two order oppositely: the value says i is the better, the measure says i is the worse (a measure falls as a goal comes nearer: its rank is the tuple negated, higher is better)
        let (mut concord, mut opposite) = (0u64, 0u64);
        let mut worst: Vec<(f64, usize, usize)> = vec![];
        let top = m.iter().cloned().fold(0.0, f64::max) + 1.0;
        for a in 0..idx.len() {
            for b in 0..idx.len() {
                if a == b || (v[a] - v[b]).abs() < 0.05 || m[a] == m[b] { continue; }
                if v[a] > v[b] {
                    if m[a] > m[b] { concord += 1; } else { opposite += 1; worst.push(((v[a] - v[b]) * (m[b] - m[a]) / top, a, b)); }
                }
            }
        }
        worst.sort_by(|x, y| y.0.total_cmp(&x.0));
        println!("OBS 79.24 agree whole games, measure {class}: pairs the value separates by 0.05 of score or more and the measure separates: {} in all, {opposite} ordered oppositely ({:.4} of them)", concord + opposite, opposite as f64 / (concord + opposite).max(1) as f64);
        for (w, a, b) in worst.iter().take(5) {
            let (ia, ib) = (idx[*a], idx[*b]);
            println!("OBS 79.24 agree opposite pair (strength {w:.4}), measure {class}: value {:.3} measure rank {} result {} at {} | against value {:.3} measure rank {} result {} at {}", v[*a], m[*a], vfx.pos[ia].y, vfx.pos[ia].text, v[*b], m[*b], vfx.pos[ib].y, vfx.pos[ib].text);
        }
        assert!(!idx.is_empty());
    }
    // (b) the positions of the table's lines in the small endings: the truth is the table's distance (nearer is better)
    let all: Vec<Vec<Read>> = read_lines(&g, &geo, &efx).into_iter().map(|(v, _)| v).collect();
    for set in ["one", "two", "twofar"] {
        let mut idx: Vec<(usize, usize)> = vec![];
        for (li, _) in efx.lines.iter().enumerate().filter(|(_, l)| l.set == set) { for k in 0..all[li].len() { idx.push((li, k)); } }
        let truth: Vec<f64> = idx.iter().map(|&(li, k)| -(all[li][k].dist as f64)).collect();
        let m: Vec<f64> = dense_rank(&idx.iter().map(|&(li, k)| all[li][k].tuple.iter().map(|x| -(*x as f64)).collect()).collect::<Vec<Vec<f64>>>());
        let v: Vec<f64> = idx.iter().map(|&(li, k)| { let s = state(&g, &all[li][k].text); value.assess(&Question { game: &g, state: &s, side: efx.lines[li].role }).unwrap().rank[0] }).collect();
        let distinct_v = { let mut s = v.clone(); s.sort_by(|a, b| a.total_cmp(b)); s.dedup(); s.len() };
        let t = |a: &[f64], b: &[f64]| kendall_tau(a, b).map_or("undefined".to_string(), |x| format!("{x:+.4}"));
        println!("OBS 79.24 agree ending {set}: {} positions; the value takes {distinct_v} distinct values there; rank agreement: value with the table's distance {}, measure with the table's distance {}, value with measure {}", idx.len(), t(&v, &truth), t(&m, &truth), t(&v, &m));
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (4) written and read back

#[test]
fn layer3_a_table_written_to_the_long_term_memory_of_one_game_and_read_back_gives_the_same_answers_and_another_games_reader_does_not_see_it() {
    let vfx = value_fixture();
    let efx = ending_fixture();
    let g = open(&vfx.name, &vfx.decl);
    let geo = Geometry::new(&g.board().unwrap());
    let key = g.rules_key();
    let mut store = LongTerm::new();
    store.sleep().add_entry(&key, KIND_MODEL, Arc::new(vfx.kind.clone())).unwrap();
    store.sleep().add_entry(&key, COUNT_MODEL, Arc::new(vfx.count.clone())).unwrap();
    for (class, term, d, sample) in &efx.measures {
        store.sleep().add(&key, Statement { id: format!("measure:{class}"), pattern: vec![], says: Says::Lower { term: term.clone(), within: *d }, standing: Standing::Conjecture(sample.clone()), holder: Holder::Ours, also: vec![] }).unwrap();
    }
    let other = GameKey::of("another game", "its own rules");
    store.sleep().add_entry(&other, KIND_MODEL, Arc::new(KindModel { kinds: vec!["x".into()], intercept: [0.0; 3], coef: [vec![9.0], vec![0.0], vec![-9.0]], count: 1, source: "another game's table".into() })).unwrap();
    let dir = std::env::temp_dir().join(format!("cs_value_layer3_{}", std::process::id()));
    let path = store.write_head(&key, &dir).unwrap();
    // read back through the decoders of this layer
    let back = Reader::new(Partition::load(&path, &decoders()).unwrap());
    let made = store.reader(&key).unwrap();
    assert_eq!(back.key(), &key);
    assert_eq!(back.id(), made.id(), "the partition is named by its content: written and read back it is the same");
    let (v0, v1) = (Value::from_memory(&made, KIND_MODEL).unwrap(), Value::from_memory(&back, KIND_MODEL).unwrap());
    let (count_made, count_back) = (Count::from_memory(&made, COUNT_MODEL).unwrap(), Count::from_memory(&back, COUNT_MODEL).unwrap());
    assert_eq!(v0.model, v1.model, "the table is the same after the memory");
    assert_eq!(count_made.model, count_back.model);
    let mut asked = 0;
    for p in &vfx.pos {
        let s = state(&g, &p.text);
        let q = Question { game: &g, state: &s, side: g.mover(&s).unwrap() };
        assert_eq!(v0.assess(&q).unwrap(), v1.assess(&q).unwrap(), "the value answers the same after the memory (bitwise: the numbers are read back whole)");
        assert_eq!(count_made.assess(&q).unwrap(), count_back.assess(&q).unwrap());
        asked += 1;
    }
    for (class, _, _, _) in &efx.measures {
        let id = format!("measure:{class}");
        let (p0, p1) = (Progress::from_memory(&made, &id, &geo).unwrap(), Progress::from_memory(&back, &id, &geo).unwrap());
        assert_eq!(p0.term, p1.term);
        for l in efx.lines.iter().filter(|l| &l.class == class).take(5) {
            for (text, _) in l.steps.iter().take(6) {
                let s = state(&g, text);
                let q = Question { game: &g, state: &s, side: l.role };
                assert_eq!(Assess::<R>::assess(&p0, &q).unwrap(), Assess::<R>::assess(&p1, &q).unwrap());
                asked += 1;
            }
        }
    }
    // another game's reader does not see this game's tables, and this game's reader does not see another's
    let theirs = store.reader(&other).unwrap();
    assert!(Count::from_memory(&theirs, COUNT_MODEL).is_err(), "another game's reader holds no plain count of this game");
    assert!(Progress::from_memory(&theirs, "measure:one", &geo).is_err(), "another game's reader holds no measure of this game");
    assert_eq!(Value::from_memory(&theirs, KIND_MODEL).unwrap().model.kinds, vec!["x".to_string()], "it sees its own table and not this game's");
    assert!(store.reader(&GameKey::of("a third game", "rules")).is_none(), "a game the store knows nothing of has no reader");
    assert!(Value::from_memory(&Reader::new(Partition::new(key.clone())), KIND_MODEL).is_err(), "an empty partition answers nothing: a missing table is Unanswered, never a number");
    println!("OBS 79.24 memory: partition {} written to {} bytes and read back; {asked} answers equal; another game's reader sees its own table only", &back.id()[..12], std::fs::metadata(&path).unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------------------------------------------------------------------------ (5) the rate

#[test]
fn layer3_the_rate_of_each_implementation_on_the_fixture_positions_three_runs_medians() {
    let vfx = value_fixture();
    let efx = ending_fixture();
    let g = open(&vfx.name, &vfx.decl);
    let geo = Geometry::new(&g.board().unwrap());
    let mem = memory_of(&g, &vfx.kind, &vfx.count, &efx.measures);
    let value = Value::from_memory(&mem, KIND_MODEL).unwrap();
    let count = Count::from_memory(&mem, COUNT_MODEL).unwrap();
    let states: Vec<_> = vfx.pos.iter().map(|p| state(&g, &p.text)).collect();
    let sides: Vec<usize> = states.iter().map(|s| g.mover(s).unwrap()).collect();
    let mut runs = std::collections::BTreeMap::<&str, Vec<f64>>::new();
    for _ in 0..3 {
        let t = Instant::now();
        let mut sink = 0.0;
        for _ in 0..20 { for (s, &me) in states.iter().zip(&sides) { sink += value.assess(&Question { game: &g, state: s, side: me }).unwrap().rank[0]; } }
        runs.entry("value").or_default().push((20 * states.len()) as f64 / t.elapsed().as_secs_f64());
        let t = Instant::now();
        for _ in 0..20 { for (s, &me) in states.iter().zip(&sides) { sink += count.assess(&Question { game: &g, state: s, side: me }).unwrap().rank[0]; } }
        runs.entry("plain count").or_default().push((20 * states.len()) as f64 / t.elapsed().as_secs_f64());
        for (class, _, _, _) in &efx.measures {
            let progress = Progress::from_memory(&mem, &format!("measure:{class}"), &geo).unwrap();
            let t = Instant::now();
            let mut n = 0;
            for _ in 0..3 { for (s, &me) in states.iter().zip(&sides) { sink += Assess::<R>::assess(&progress, &Question { game: &g, state: s, side: me }).unwrap().rank[0]; n += 1; } }
            runs.entry(if class == "one" { "measure one" } else { "measure two" }).or_default().push(n as f64 / t.elapsed().as_secs_f64());
        }
        assert!(sink.is_finite());
    }
    for (name, v) in &runs { println!("OBS 79.24 rate {name}: {:.0} answers a second (median of {} runs over the {} fixture positions; runs {:?})", median(v.clone()), v.len(), states.len(), v.iter().map(|x| *x as u64).collect::<Vec<_>>()); }
    assert!(runs.values().all(|v| median(v.clone()) > 1000.0));
}

// ------------------------------------------------------------------------------------------------------------------------------ the interface itself

#[test]
fn layer3_the_interface_orders_answers_and_a_missing_table_is_not_a_number() {
    let a = Answer { rank: vec![0.5, 1.0], raw: None, chance: None, sure: None, read: vec![] };
    let b = Answer { rank: vec![0.5, 2.0], raw: None, chance: None, sure: None, read: vec![] };
    assert_eq!(a.order(&b), Ordering::Less);
    assert_eq!(b.order(&a), Ordering::Greater);
    assert_eq!(a.order(&a.clone()), Ordering::Equal);
    assert_eq!(kendall_tau(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), Some(1.0));
    assert_eq!(kendall_tau(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), Some(-1.0));
    assert_eq!(kendall_tau(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), None, "a constant list has no rank order to agree with");
    let c = cs_value::Chance { win: 0.5, draw: 0.25, loss: 0.25 };
    assert_eq!(c.score(), 0.625);
    assert_eq!(c.flip(), cs_value::Chance { win: 0.25, draw: 0.25, loss: 0.5 });
}
