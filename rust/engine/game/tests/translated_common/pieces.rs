//! PLAN 79.38 PART A (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; ledger row 45; piece: rules): THE HARNESS THAT HOLDS EACH PIECE OF A MODULE WITH PASSES TO THE SAME PIECE OF THE PLAIN MODULE.
//! Kaveh: "build unit tests around every little piece and making sure the pieces each provide the same answer, but at a better speed"; "the test cases are no longer chess positions, but they're little bits of the chess logic".
//! The plain module (no pass) is the reference. At each state, and a move made at it, the harness asks both modules the same things and compares what they answer, as text (the numbers of compound terms differ from module to module):
//!   (1) the five questions: the side to move, the allowed moves, the end, the outcome, and the next state of the move;
//!   (2) every relation the module with passes makes in full: the rows equal the plain module's rows for that relation (a relation the plain module has and the other does not make in full must be one the other answers on demand, or the harness fails: a piece may not be lost);
//!   (3) every relation answered on demand: for every way it is asked (the places its key binds, rows or only existence), the answer to keys made from the rows of the plain relation (members) and from rows with one place taken from another row (mostly non-members) equals what the plain relation's rows give for that key, and the whole relation asked with no key equals the plain rows;
//!   (4) the relations that do not read the state, whole: the rows of each equal the plain module's.
//! `PieceCounts` counts each comparison; a disagreement is returned with the relation and the smallest case.
#![allow(dead_code)]
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default, Debug, Clone)]
pub struct PieceCounts {
    pub states: usize,
    pub questions: usize,
    pub nexts: usize,
    pub relations: usize,
    pub keys: usize,
    pub statics: usize,
    /// relation -> (comparisons made in full, key probes)
    pub per_relation: BTreeMap<String, (usize, usize)>,
    /// relations the module with passes does not make in full (answered on demand)
    pub on_demand: BTreeSet<String>,
}

impl PieceCounts {
    pub fn total(&self) -> usize { self.questions + self.nexts + self.relations + self.keys + self.statics }
}

fn rows_text<T: Tr>(t: &T, rows: &[Vec<u32>]) -> BTreeSet<Vec<String>> { rows.iter().map(|r| r.iter().map(|x| t.render_t(*x)).collect()).collect() }

/// The relations that do not read the state, whole: the rows of each equal.
pub fn statics_equal<A: Tr, B: Tr>(a: &A, b: &B, c: &mut PieceCounts) -> Result<(), String> {
    let (sa, sb) = (a.statics_t(), b.statics_t());
    let mb: BTreeMap<&str, &Vec<Vec<u32>>> = sb.iter().map(|(n, r)| (*n, r)).collect();
    for (name, ra) in &sa {
        let rb = mb.get(name).ok_or(format!("{name}: a relation that does not read the state is missing from the module with passes"))?;
        let (ta, tb) = (rows_text(a, ra), rows_text(b, rb));
        if ta != tb { return Err(format!("{name} (static): only plain {:?}, only with passes {:?}", ta.difference(&tb).take(3).collect::<Vec<_>>(), tb.difference(&ta).take(3).collect::<Vec<_>>())); }
        c.statics += 1;
    }
    if sa.len() != sb.len() { return Err(format!("the number of static relations: plain {}, with passes {}", sa.len(), sb.len())); }
    Ok(())
}

fn the_does<T: Tr>(t: &mut T, s: &T::S, mv: &Option<String>) -> Result<Vec<(u32, u32)>, String> {
    match mv {
        None => Ok(vec![]),
        Some(text) => {
            let roles = t.roles_t();
            let mover = t.mover_t(s);
            let noop = t.noop_t();
            let m = t.term_t(text)?;
            Ok(roles.iter().enumerate().map(|(i, r)| (*r, if Some(i) == mover { m } else { noop })).collect())
        }
    }
}

/// The module with passes against the plain one at one state (given as facts) and, when `mv` is, the move made at it (as text). `with_relations`: also every relation and every way a demanded relation is asked.
pub fn pieces_at<A: Tr, B: Tr>(a: &mut A, b: &mut B, facts: &[(String, Vec<String>)], mv: &Option<String>, label: &str, with_relations: bool, c: &mut PieceCounts) -> Result<(), String> {
    let (sa, sb) = (a.state_of_facts_t(facts)?, b.state_of_facts_t(facts)?);
    // (1) the questions
    let (ra, rb) = (a.roles_t().iter().map(|r| a.render_t(*r)).collect::<Vec<_>>(), b.roles_t().iter().map(|r| b.render_t(*r)).collect::<Vec<_>>());
    if ra != rb { return Err(format!("{label}: the roles")); }
    let (ma, mb) = (a.mover_t(&sa).map(|i| ra[i].clone()), b.mover_t(&sb).map(|i| rb[i].clone()));
    if ma != mb { return Err(format!("{label}: the side to move: plain {ma:?}, with passes {mb:?}")); }
    let (oa, ob) = (a.outcome_t(&sa), b.outcome_t(&sb));
    if oa != ob { return Err(format!("{label}: the outcome: plain {oa:?}, with passes {ob:?}")); }
    if a.terminal_t(&sa) != b.terminal_t(&sb) { return Err(format!("{label}: the end")); }
    let (mva, mvb): (BTreeSet<String>, BTreeSet<String>) = (a.moves_t(&sa).iter().map(|m| a.render_t(*m)).collect(), b.moves_t(&sb).iter().map(|m| b.render_t(*m)).collect());
    if mva != mvb { return Err(format!("{label}: the allowed moves: only plain {:?}, only with passes {:?}", mva.difference(&mvb).take(3).collect::<Vec<_>>(), mvb.difference(&mva).take(3).collect::<Vec<_>>())); }
    if b.moves_t(&sb).len() != mvb.len() { return Err(format!("{label}: a move given twice by the module with passes")); }
    c.questions += 4;
    c.states += 1;
    if let Some(text) = mv {
        let (ta, tb) = (a.term_t(text)?, b.term_t(text)?);
        let (na, nb) = (a.next_t(&sa, ta), b.next_t(&sb, tb));
        let (fa, fb): (BTreeSet<_>, BTreeSet<_>) = (a.facts_of_t(&na).into_iter().collect(), b.facts_of_t(&nb).into_iter().collect());
        if fa != fb { return Err(format!("{label}: the next state after {text}: only plain {:?}, only with passes {:?}", fa.difference(&fb).take(3).collect::<Vec<_>>(), fb.difference(&fa).take(3).collect::<Vec<_>>())); }
        c.nexts += 1;
    }
    if !with_relations { return Ok(()); }
    // (2) every relation made in full
    let (does_a, does_b) = (the_does(a, &sa, mv)?, the_does(b, &sb, mv)?);
    let rels_a: BTreeMap<&str, Vec<Vec<u32>>> = a.all_t(&sa, &does_a).into_iter().collect();
    let rels_b: BTreeMap<&str, Vec<Vec<u32>>> = b.all_t(&sb, &does_b).into_iter().collect();
    let demanded: Vec<(&'static str, Vec<usize>, bool)> = b.demanded_t();
    let demanded_names: BTreeSet<&str> = demanded.iter().map(|d| d.0).collect();
    for (name, rows) in &rels_b {
        let theirs = rels_a.get(name).ok_or(format!("{label}: {name} is made by the module with passes and not by the plain module"))?;
        let (tb, ta) = (rows_text(b, rows), rows_text(a, theirs));
        if ta != tb { return Err(format!("{label}: the relation {name}: only plain {:?}, only with passes {:?}", ta.difference(&tb).take(3).collect::<Vec<_>>(), tb.difference(&ta).take(3).collect::<Vec<_>>())); }
        c.relations += 1;
        c.per_relation.entry(name.to_string()).or_default().0 += 1;
    }
    for name in rels_a.keys() {
        if !rels_b.contains_key(name) {
            if !demanded_names.contains(name) { return Err(format!("{label}: the relation {name} is lost: the module with passes neither makes it nor answers it on demand")); }
            c.on_demand.insert(name.to_string());
        }
    }
    // (3) every relation answered on demand, in every way it is asked
    let mut queries: Vec<(String, Vec<u32>)> = vec![];
    let mut expect: Vec<(String, Vec<Option<String>>, bool, bool)> = vec![];       // relation, key as text (None: free), whether the mode asks existence, whether the exact mode exists
    for (rel, places, exists) in &demanded {
        let rows: BTreeSet<Vec<String>> = rows_text(a, rels_a.get(rel).map(|x| x.as_slice()).unwrap_or(&[]));
        let arity = rels_a.get(rel).and_then(|r| r.first().map(|x| x.len())).unwrap_or(0).max(places.iter().copied().max().map(|m| m + 1).unwrap_or(0));
        let mut keys: BTreeSet<Vec<Option<String>>> = BTreeSet::new();
        let all: Vec<&Vec<String>> = rows.iter().collect();
        for (i, r) in all.iter().enumerate() {
            let mut k: Vec<Option<String>> = vec![None; arity];
            for &p in places { k[p] = Some(r[p].clone()); }
            keys.insert(k.clone());
            if !all.is_empty() && !places.is_empty() {
                let other = all[(i * 7 + 3) % all.len()];
                let p = places[i % places.len()];
                k[p] = Some(other[p].clone());
                keys.insert(k);
            }
            if keys.len() >= 48 { break; }
        }
        if rows.is_empty() || keys.len() < 4 {
            // too few members to make keys from: values of the state itself, at the places of the key
            let pool: Vec<String> = facts.iter().flat_map(|f| f.1.iter().cloned()).collect::<BTreeSet<_>>().into_iter().collect();
            for j in 0..6 {
                let mut k: Vec<Option<String>> = vec![None; arity];
                if !pool.is_empty() { for (n, &p) in places.iter().enumerate() { k[p] = Some(pool[(j * 5 + n * 3) % pool.len()].clone()); } }
                keys.insert(k);
            }
        }
        for k in keys {
            let mut key = vec![u32::MAX; arity];
            let mut ok = true;
            for (p, v) in k.iter().enumerate() { if let Some(t) = v { match b.term_t(t) { Ok(x) => key[p] = x, Err(_) => ok = false } } }
            if !ok { continue; }
            queries.push((rel.to_string(), key));
            expect.push((rel.to_string(), k, *exists, true));
        }
        // the relation whole, asked with no key
        queries.push((rel.to_string(), vec![u32::MAX; arity]));
        expect.push((rel.to_string(), vec![None; arity], false, false));
    }
    if !queries.is_empty() {
        let answers = b.ask_t(&sb, &does_b, &queries);
        for ((rel, k, exists, _), (rows, ex)) in expect.iter().zip(answers.iter()) {
            let plain = rows_text(a, rels_a.get(rel.as_str()).map(|x| x.as_slice()).unwrap_or(&[]));
            let want: BTreeSet<Vec<String>> = plain.into_iter().filter(|r| k.iter().enumerate().all(|(p, v)| v.as_ref().map(|t| r.get(p) == Some(t)).unwrap_or(true))).collect();
            let got = rows_text(b, rows);
            if want != got { return Err(format!("{label}: {rel} asked for the key {k:?}: plain rows {:?}, with passes {:?}", want.iter().take(3).collect::<Vec<_>>(), got.iter().take(3).collect::<Vec<_>>())); }
            if *exists {
                match ex {
                    Some(e) if *e == !want.is_empty() => {}
                    other => return Err(format!("{label}: {rel} asked whether a row exists for the key {k:?}: plain {}, with passes {other:?}", !want.is_empty())),
                }
            }
            c.keys += 1;
            c.per_relation.entry(rel.clone()).or_default().1 += 1;
        }
    }
    Ok(())
}

/// The states of the certificate's seeded games, from the kernel's walk (the kernel's moves sorted by its names, one picked by the seeded generator), followed by `a`: for each game a list of (the facts of the state, the move played from it as text; None at the end).
pub fn corpus<A: Tr>(kernel: &KernelGame, a: &mut A, games: usize, seed: u64) -> Vec<Vec<(Vec<(String, Vec<String>)>, Option<String>)>> {
    let mut rng = Rng::new(seed);
    let mut out = vec![];
    for _ in 0..games {
        let (mut ks, mut ts) = (kernel.initial(), a.initial_t());
        let mut line = vec![];
        loop {
            let km = kernel.moves(&ks);
            if kernel.outcome(&ks).is_some() || km.is_empty() { line.push((a.facts_of_t(&ts), None)); break; }
            let mut sorted: Vec<(String, usize)> = km.iter().enumerate().map(|(i, m)| (kernel.move_name(&ks, m), i)).collect();
            sorted.sort();
            let (nm, i) = sorted[rng.below(sorted.len())].clone();
            let text = spec_term(&nm);
            line.push((a.facts_of_t(&ts), Some(text.clone())));
            let m = a.term_t(&text).unwrap();
            ts = a.next_t(&ts, m);
            ks = kernel.apply(&ks, &km[i]);
        }
        out.push(line);
    }
    out
}
