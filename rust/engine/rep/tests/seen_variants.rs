//! PLAN 79.35 PART F: LAYER 1 IS UNCHANGED AND HOLDS A STATE OF EACH OF THE FOUR GAMES OF PLAN 79.35 (SYSTEM_SPEC 11 "Layer 2 says variants, with another start and with other rules, on the same layer 1"; ledger rows 41, 40; piece: representation).
//! Layer 1 says what is seen in a state it is shown: which unit stands where, how many of each kind. It is shown states of games that start differently and have other counts (a side with no leader and many of the smallest kind among
//! them), read from the fixtures of the rules' crate as opaque DATA (the files are text; this crate depends on no crate of the rules). No state is a START here: each is shown as a state, held, counted and written back identical.
//! The code and the strings of this file name no object of a game (tests/test_plan78.py scans for it); the files are found by what they say (`differs`), not by their names.
mod common;
use common::fixtures;
use cs_rep::{Fact, UnitDecl};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Every position of every fixture file of the rules' crate that says it differs from the standard game: (file, label, facts).
fn shown_states() -> Vec<(String, String, Vec<Fact>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("game").join("tests").join("fixtures");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "fixture")).collect();
    files.sort();
    let mut out = vec![];
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        if !text.lines().any(|l| l.starts_with("differs\t")) { continue; }
        let file = path.file_name().unwrap().to_string_lossy().to_string();
        for line in text.lines().filter(|l| l.starts_with("pos\t")) {
            let f: Vec<&str> = line.split('\t').collect();
            let facts: Vec<Fact> = f[6].split(' ').filter(|t| !t.is_empty() && *t != "-").map(|t| { let mut p = t.split(','); Fact { relation: p.next().unwrap().to_string(), args: p.map(|a| a.to_string()).collect() } }).collect();
            out.push((file.clone(), f[1].to_string(), facts));
        }
    }
    out
}

#[test]
fn a_state_of_each_of_the_four_games_is_held_counted_by_kind_and_written_back_identical_and_one_of_them_holds_thirty_six_of_a_kind() {
    let b = fixtures().remove(0).board;
    let d: UnitDecl = b.decl.clone().unwrap().units;
    let states = shown_states();
    let files: BTreeSet<&String> = states.iter().map(|s| &s.0).collect();
    assert_eq!(files.len(), 4, "four games that differ from the standard one: {files:?}");
    let (mut held, mut most, mut start_most) = (0usize, 0usize, BTreeMap::new());
    for (file, label, facts) in &states {
        let part = b.position_from_facts(facts).unwrap_or_else(|e| panic!("{file} {label}: {e}"));
        // the count of each kind a side holds, in layer 1's reading, equals a plain count of the facts that name it
        let mut plain: BTreeMap<(String, String), usize> = BTreeMap::new();
        for f in facts.iter().filter(|f| f.relation == d.relation) { *plain.entry((f.args[d.owner].clone(), f.args[d.kind.unwrap()].clone())).or_default() += 1; }
        for (ri, r) in b.names.roles.iter().enumerate() {
            for (ki, k) in b.names.kinds.iter().enumerate() {
                let n = part.count(ri, ki as u8);
                assert_eq!(n, plain.get(&(r.clone(), k.clone())).copied().unwrap_or(0), "{file} {label}: {r} {k}");
                most = most.max(n);
                if label == "start" { let e = start_most.entry(file.clone()).or_insert(0usize); *e = (*e).max(n); }
            }
        }
        assert_eq!(part.units.len(), plain.values().sum::<usize>());
        for (i, u) in part.units.iter().enumerate() { assert_eq!(part.occupant[u.at as usize] as usize, i + 1, "{file} {label}: the occupant of each place"); }
        // written back as text and read again: the same state
        let (marked, text) = b.state_text(&part);
        let back = b.part_from_text(marked.as_deref(), &text).unwrap();
        assert_eq!(back.definitions(&b), part.definitions(&b), "{file} {label}");
        assert_eq!(b.state_text(&back), (marked, text));
        held += 1;
    }
    // one of the four starts holds thirty-six of one kind on one side (the others at most eight)
    let big: Vec<(&String, &usize)> = start_most.iter().filter(|(_, n)| **n == 36).collect();
    assert_eq!(big.len(), 1, "exactly one start holds thirty-six of a kind: {start_most:?}");
    assert!(start_most.iter().filter(|(_, n)| **n != 36).all(|(_, n)| *n == 8));
    println!("OBS 79.35 F layer 1: {held} states of 4 games that differ from the standard one are held, counted by kind as a plain count of their facts says, and written back identical; the most of one kind a side holds in them is {most} (the start of one holds {})", big[0].1);
}
