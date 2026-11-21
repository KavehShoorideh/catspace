//! PLAN 79.23 PART B: THE TEST OF BEING DONE OF LAYER 2, THE RULES (SYSTEM_SPEC 11 "a layer's test uses only that layer and what is below it"; ledger row 36; piece: rules).
//! It uses layer 1 (cs_rep) and layer 2 (cs_game) and no crate above, and no outside program. The reference is DATA: tests/fixtures/<game>.fixture, written once by
//! catspace/incubator/analyst/discovery/layer2_fixtures_print.py from a library that shares no code with the engine (the format is described there). This file reads the fixtures as opaque
//! text: it names no object of any game; the names of the games, of their units and of their moves come from the files.
//! THE SIX THINGS: (1) the allowed moves and the counts of positions reached; (2) every way a game ends; (3) the position after a move as a change to layer 1's graph, with undo and keys;
//! (4) the text door; (5) the two readers agree; (6) the rate. Each prints its OBS lines.
use cs_game::certify::coplay;
use cs_game::gdl_game::GdlGame;
use cs_game::graph::Static;
use cs_game::keeper::GraphKeeper;
use cs_game::kernel::KernelGame;
use cs_game::rules::{DescriptionNames, Ruled, Rules, StateText};
use cs_game::tags::TagKeeper;
use cs_game::{Fact, Game};
use cs_rep::history::History;
use cs_rep::statement::Change;
use cs_rep::store::{GameKey, GameMemory};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::Instant;

type R = Ruled<KernelGame>;

// ------------------------------------------------------------------------------------------------------------------------------ the fixtures, read as opaque text

struct Pos { label: String, text: String, legal_text: String, end: Option<(String, Vec<i64>)>, moves: Vec<String>, facts: BTreeSet<Fact>, identity: String, side: String }
struct Step { parent: String, name: String, end: Option<(String, Vec<i64>)>, added: Vec<Fact>, removed: Vec<Fact>, identity: String, side: String }
struct Fx {
    name: String, decl: String, ends: Vec<String>, perft: Vec<(String, String, Vec<u64>)>, pos: Vec<Pos>, steps: Vec<Step>,
    lines: Vec<(String, Vec<(String, String)>)>, records: Vec<(String, Vec<String>, Vec<usize>, usize)>, bytes: usize,
}

fn facts(s: &str) -> Vec<Fact> {
    s.split(' ').filter(|t| !t.is_empty() && *t != "-").map(|t| { let mut p = t.split(','); Fact { relation: p.next().unwrap().to_string(), args: p.map(|a| a.to_string()).collect() } }).collect()
}

fn end_of(s: &str) -> Option<(String, Vec<i64>)> {
    if s == "-" { return None; }
    let (label, values) = s.split_once('/').unwrap();
    Some((label.to_string(), values.split(',').map(|v| v.parse().unwrap()).collect()))
}

fn rules_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules") }

fn fixtures() -> Vec<Fx> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "fixture")).collect();
    paths.sort();
    assert_eq!(paths.len(), 3, "the standard game and both variants of the repo are expected in {}", dir.display());
    paths.iter().map(|path| {
        let text = std::fs::read_to_string(path).unwrap();
        let mut fx = Fx { name: String::new(), decl: String::new(), ends: vec![], perft: vec![], pos: vec![], steps: vec![], lines: vec![], records: vec![], bytes: text.len() };
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            let words = |s: &str| -> Vec<String> { s.split(' ').filter(|w| !w.is_empty() && *w != "-").map(|w| w.to_string()).collect() };
            match f[0] {
                "game" => { fx.name = f[1].to_string(); fx.decl = f[2].to_string(); }
                "ends" => fx.ends = words(f[1]),
                "perft" => fx.perft.push((f[1].to_string(), f[2].to_string(), f[3].split(',').map(|n| n.parse().unwrap()).collect())),
                "pos" => fx.pos.push(Pos { label: f[1].into(), text: f[2].into(), legal_text: f[3].into(), end: end_of(f[4]), moves: words(f[5]), facts: facts(f[6]).into_iter().collect(), identity: f[7].into(), side: f[8].into() }),
                "step" => fx.steps.push(Step { parent: f[1].into(), name: f[2].into(), end: end_of(f[3]), added: facts(f[4]), removed: facts(f[5]), identity: f[6].into(), side: f[7].into() }),
                "line" => fx.lines.push((f[1].into(), f[2].split(';').filter(|w| !w.is_empty()).map(|w| { let (a, b) = w.split_once('|').unwrap(); (a.to_string(), b.to_string()) }).collect())),
                "record" => fx.records.push((f[1].into(), words(f[2]), f[3].split(',').map(|n| n.parse().unwrap()).collect(), f[4].parse().unwrap())),
                other => panic!("unknown tag {other}"),
            }
        }
        fx
    }).collect()
}

/// The game of a fixture, opened from its own declaration (the one input is where the declaration lies).
fn open(fx: &Fx) -> R {
    Ruled::open(KernelGame::new(&fx.name, 518).unwrap(), &fx.name, &rules_root(), &fx.decl).unwrap()
}

fn names_of(g: &R, s: &<R as Game>::State) -> Vec<String> {
    let mut v: Vec<String> = g.moves(s).iter().map(|m| g.move_name(s, m)).collect();
    v.sort();
    v
}

fn sorted_facts(g: &R, s: &<R as Game>::State) -> BTreeSet<Fact> { g.base_facts(s).into_iter().collect() }

fn end_text(g: &R, s: &<R as Game>::State) -> Option<Vec<i64>> { g.outcome(s) }

/// The count of positions reached after exactly 1, 2, .. plies, every state made.
fn count<G: Game>(g: &G, s: &G::State, depth: usize, k: usize, counts: &mut Vec<u64>) {
    if depth == 0 { return; }
    let ms = g.moves(s);
    counts[k] += ms.len() as u64;
    for m in &ms { let t = g.apply(s, m); count(g, &t, depth - 1, k + 1, counts); }
}

// ------------------------------------------------------------------------------------------------------------------------------ the interface: the game reports its own rules

#[test]
fn layer2_the_game_reports_the_address_of_its_own_rules_and_the_declaration_of_its_units_and_tempo() {
    let mut keys = vec![];
    for fx in fixtures() {
        let g = open(&fx);
        let key = g.rules_key();
        assert_eq!(key, GameKey::of(&fx.name, g.description()), "the address is the content address of the rules' text the game holds");
        assert_eq!(key.name, fx.name);
        assert!(g.fact_decl().tempo.is_some() && !g.fact_decl().units.relation.is_empty());
        assert!(g.declaration().static_relations.len() >= 2 && !g.declaration().objects.is_empty());
        assert_eq!(g.repetition_limit(), Some(2), "the repeat limit is read from the description");
        // the declaration is the game's own: layer 1 reads the reference facts of every fixture position through it
        let board = g.board().unwrap();
        for p in fx.pos.iter().take(40) {
            let v: Vec<Fact> = p.facts.iter().cloned().collect();
            let (mover, cells) = board.read_facts(&v).unwrap();
            assert_eq!(mover.map(|m| g.roles()[m].clone()), Some(p.side.clone()));
            assert_eq!(cells.len(), p.facts.iter().filter(|f| f.relation == g.fact_decl().units.relation).count());
        }
        println!("OBS 79.23 A {}: rules address {} (the rules' text {} bytes), units held in `{}` (owner argument {}, kind argument {:?}), tempo in `{}`, {} board relations, repeat limit {:?}, fixture {} bytes", fx.name, key.rules, g.description().len(), g.fact_decl().units.relation, g.fact_decl().units.owner, g.fact_decl().units.kind, g.fact_decl().tempo.unwrap(), g.declaration().static_relations.len(), g.repetition_limit(), fx.bytes);
        keys.push(key);
    }
    let addresses: BTreeSet<&String> = keys.iter().map(|k| &k.rules).collect();
    assert_eq!(addresses.len(), keys.len(), "three games, three addresses");
    // a name the declaration does not know is refused
    let fx = &fixtures()[0];
    assert!(Ruled::open(KernelGame::new(&fx.name, 518).unwrap(), "no_such_game", &rules_root(), &fx.decl).is_err());
}

// ------------------------------------------------------------------------------------------------------------------------------ (4) the text door

#[test]
fn layer2_door_the_text_form_is_read_and_written_back_and_the_dropped_fact_is_decided() {
    for fx in fixtures() {
        let g = open(&fx);
        let (mut exact, mut legal_only, mut dropped, mut relations) = (0usize, 0usize, 0usize, BTreeSet::new());
        for p in &fx.pos {
            let s = g.read_state_text(&p.text).unwrap_or_else(|e| panic!("{}: {e}", p.label));
            assert_eq!(sorted_facts(&g, &s), p.facts, "{}: the state read from the text has the reference's facts", p.label);
            assert_eq!(g.state_text(&s), p.text, "{}: the state written back is the text", p.label);
            assert_eq!(names_of(&g, &s), p.moves, "{}: the allowed moves of the state read", p.label);
            assert_eq!(g.outcome(&s), p.end.as_ref().map(|e| e.1.clone()), "{}: the end", p.label);
            exact += 1;
            // the text that keeps the passing-square only where a taking is legal (what the kernel's own writer gives): the same position by key, moves and end; at most one relation of facts is lost
            let t = g.read_state_text(&p.legal_text).unwrap_or_else(|e| panic!("{} (legal form): {e}", p.label));
            assert_eq!(g.key(&t), g.key(&s), "{}: the key does not tell the two texts apart", p.label);
            assert_eq!(names_of(&g, &t), p.moves, "{}: the same moves", p.label);
            assert_eq!(g.outcome(&t), g.outcome(&s), "{}: the same end", p.label);
            let (a, b) = (sorted_facts(&g, &s), sorted_facts(&g, &t));
            if a != b {
                let lost: Vec<&Fact> = a.symmetric_difference(&b).collect();
                assert!(lost.len() == 1 && a.contains(lost[0]), "{}: the legal form only loses one fact: {lost:?}", p.label);
                relations.insert(lost[0].relation.clone());
                dropped += 1;
            }
            legal_only += 1;
        }
        println!("OBS 79.23 B4 text door {}: {exact} positions read from their text with the reference's facts, moves and end and written back identical; the text that writes the passing-square only where a taking is legal was read for {legal_only}: same key, moves and end, and it dropped one fact (of relation {:?}) in {dropped}", fx.name, relations);
        assert!(exact > 80);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (1) the allowed moves and the counts

#[test]
fn layer2_the_counts_of_positions_and_the_allowed_moves_equal_the_reference() {
    for fx in fixtures() {
        let g = open(&fx);
        let (mut nodes, mut moves_seen) = (0u64, 0usize);
        for (label, text, want) in &fx.perft {
            let s = g.read_state_text(text).unwrap();
            let mut got = vec![0u64; want.len() + 1];
            count(&g, &s, want.len(), 1, &mut got);
            assert_eq!(&got[1..], &want[..], "{}: {label}: positions reached after each ply", fx.name);
            nodes += want.iter().sum::<u64>();
        }
        for p in &fx.pos {
            let s = g.read_state_text(&p.text).unwrap();
            let got = names_of(&g, &s);
            assert_eq!(got, p.moves, "{}: allowed moves at {}", fx.name, p.label);
            for n in &p.moves { assert!(g.find_move(&s, n).is_some(), "{}: {n} is found by its name at {}", fx.name, p.label); }
            assert_eq!(g.mover(&s).map(|m| g.roles()[m].clone()), if p.end.is_none() { Some(p.side.clone()) } else { None }, "{}: the side to move at {}", fx.name, p.label);
            moves_seen += got.len();
        }
        println!("OBS 79.23 B1 {}: counts of positions equal the reference for {} starts to depths {:?} ({} positions counted); the allowed moves equal at {} positions ({} moves named and found)", fx.name, fx.perft.len(), fx.perft.iter().map(|p| p.2.len()).collect::<Vec<_>>(), nodes, fx.pos.len(), moves_seen);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (2) every way a game ends

#[test]
fn layer2_every_way_a_game_ends_is_reached_and_its_result_equals_the_reference() {
    for fx in fixtures() {
        let g = open(&fx);
        let mut reached: BTreeMap<String, usize> = BTreeMap::new();
        for p in &fx.pos {
            let s = g.read_state_text(&p.text).unwrap();
            assert_eq!(end_text(&g, &s), p.end.as_ref().map(|e| e.1.clone()), "{}: the end at {}", fx.name, p.label);
            if let Some(e) = &p.end { assert!(g.moves(&s).is_empty()); *reached.entry(e.0.clone()).or_default() += 1; }
        }
        for st in &fx.steps {
            let parent = fx.pos.iter().find(|p| p.label == st.parent).unwrap();
            let s = g.read_state_text(&parent.text).unwrap();
            let after = g.apply(&s, &g.find_move(&s, &st.name).unwrap());
            assert_eq!(g.outcome(&after), st.end.as_ref().map(|e| e.1.clone()), "{}: the end after {} at {}", fx.name, st.name, st.parent);
            if let Some(e) = &st.end { *reached.entry(format!("{}(by a move)", e.0)).or_default() += 1; }
        }
        let labels: BTreeSet<String> = reached.keys().map(|k| k.split('(').next().unwrap().to_string()).collect();
        assert_eq!(labels, fx.ends.iter().cloned().collect::<BTreeSet<_>>(), "{}: every way this game ends is reached, and no other", fx.name);
        // the repeat: the rules' limit and layer 1's count of earlier occurrences, along the games of the file that return to a position
        let mut repeat_ends = 0;
        for (label, moves, want, limit) in &fx.records {
            let mut s = g.initial();
            let mut h = History::new(g.key(&s));
            let mut got = vec![h.repeats()];
            for n in moves {
                let m = g.find_move(&s, n).unwrap();
                let (before, after) = (sorted_facts(&g, &s), g.apply(&s, &m));
                let now = sorted_facts(&g, &after);
                h.push(Change { added: now.difference(&before).cloned().collect(), removed: before.difference(&now).cloned().collect(), keys: vec![] }, g.key(&after));
                got.push(h.repeats());
                s = after;
            }
            assert_eq!(&got, want, "{}: {label}: earlier occurrences of each position, layer 1's count", fx.name);
            assert_eq!(g.repetition_limit(), Some(*limit));
            for (i, r) in got.iter().enumerate() { assert_eq!(g.ends_by_record(*r), *r >= *limit, "{}: {label}: ply {i}", fx.name); }
            repeat_ends += got.iter().filter(|r| g.ends_by_record(**r)).count();
        }
        println!("OBS 79.23 B2 {}: ends reached (positions of the file, and by a move): {:?}; every way of the game's list reached: {:?}; positions ended by the repeat rule over {} record games: {repeat_ends}", fx.name, reached, fx.ends, fx.records.len());
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (3) the position after a move, as a change to layer 1's graph

#[test]
fn layer2_the_position_after_a_move_is_a_change_to_layer_1s_graph_with_undo_and_keys() {
    for fx in fixtures() {
        let g = open(&fx);
        let board: Static = g.board().unwrap();
        let decl = g.fact_decl();
        let is_tag = |f: &Fact| f.relation != decl.units.relation && decl.tempo.as_deref() != Some(f.relation.as_str());
        let pos: HashMap<&str, &Pos> = fx.pos.iter().map(|p| (p.label.as_str(), p)).collect();
        let (mut key_of_identity, mut identity_of_key): (HashMap<String, u64>, HashMap<u64, String>) = (HashMap::new(), HashMap::new());
        let mut note = |identity: &str, key: u64| {
            assert_eq!(*key_of_identity.entry(identity.to_string()).or_insert(key), key, "{}: positions the reference says are one have one key", fx.name);
            assert_eq!(identity_of_key.entry(key).or_insert_with(|| identity.to_string()), identity, "{}: two positions the reference tells apart have two keys", fx.name);
        };
        let (mut made, mut undone, mut graphs_equal, mut tags_equal, mut units_equal) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut by_parent: BTreeMap<&str, Vec<&Step>> = BTreeMap::new();
        for st in &fx.steps { by_parent.entry(st.parent.as_str()).or_default().push(st); }
        for (label, steps) in by_parent {
            let p = pos[label];
            let s = g.read_state_text(&p.text).unwrap();
            note(&p.identity, g.key(&s));
            let mut graph = GraphKeeper::new(&board, &g, &s);
            let before_graph = graph.part(&board).definitions(board.board());
            let mut tags = TagKeeper::new(&g, &s, &decl, GameMemory::new(g.rules_key()));
            let tags_before = tags.tags();
            let key_before = tags.history().current_key();
            assert_eq!(key_before, g.key(&s));
            for st in steps {
                let m = g.find_move(&s, &st.name).unwrap();
                let after = g.apply(&s, &m);
                // the facts after are the facts before with the reference's change applied
                let mut want = p.facts.clone();
                for f in &st.removed { assert!(want.remove(f), "{}: {} at {label}: a removed fact was there", fx.name, st.name); }
                for f in &st.added { assert!(want.insert(f.clone()), "{}: {} at {label}: an added fact was new", fx.name, st.name); }
                assert_eq!(sorted_facts(&g, &after), want, "{}: facts after {} at {label}", fx.name, st.name);
                // layer 1's graph: the change is written, the graph equals the graph made afresh and holds the reference's units and tempo
                graph.make(&board, &g, &after);
                let kept = graph.part(&board);
                assert_eq!(kept.definitions(board.board()), board.position(&g, &after).definitions(board.board()), "{}: the graph kept after {} at {label} equals the one made afresh", fx.name, st.name);
                let (mover, mut cells) = board.read_facts(&want.iter().cloned().collect::<Vec<_>>()).unwrap();
                cells.sort();
                let mut live: Vec<_> = graph.live().units().iter().map(|u| u.key()).collect();
                live.sort();
                assert_eq!(live, cells, "{}: units of the graph after {} at {label} equal the reference's", fx.name, st.name);
                assert_eq!(graph.live().tempo(), if st.end.is_none() { mover } else { None }, "{}: tempo after {} at {label}", fx.name, st.name);
                graph_equal_note(&mut graphs_equal, &mut units_equal);
                // layer 1's history: the tags, the key
                tags.make(&g, &after);
                assert_eq!(tags.tags(), want.iter().filter(|f| is_tag(f)).cloned().collect::<BTreeSet<_>>(), "{}: tags after {} at {label}", fx.name, st.name);
                assert_eq!(tags.history().current_key(), g.key(&after));
                note(&st.identity, g.key(&after));
                tags_equal += 1;
                made += 1;
                // undo: the graph, the tags, the key are what they were
                assert!(graph.undo());
                assert_eq!(graph.part(&board).definitions(board.board()), before_graph, "{}: undo of {} at {label} gives the graph before", fx.name, st.name);
                assert!(tags.undo());
                assert_eq!((tags.tags(), tags.history().current_key()), (tags_before.clone(), key_before), "{}: undo of {} at {label} gives the tags and the key before", fx.name, st.name);
                undone += 1;
            }
        }
        println!("OBS 79.23 B3 {}: {made} moves made as a change to layer 1's graph from {} positions; the graph after equals the graph made afresh and holds the reference's units and tempo ({graphs_equal}, {units_equal}); tags equal the reference's ({tags_equal}); {undone} undos give back the graph, the tags and the key; {} distinct reference positions have {} distinct keys, one for one", fx.name, fx.pos.len(), key_of_identity.len(), identity_of_key.len());
        assert_eq!(key_of_identity.len(), identity_of_key.len());
    }
}

fn graph_equal_note(a: &mut usize, b: &mut usize) { *a += 1; *b += 1; }

// ------------------------------------------------------------------------------------------------------------------------------ (5) the two readers agree

#[test]
fn layer2_the_fast_reader_and_the_readers_of_the_description_agree_on_moves_ends_and_facts() {
    for fx in fixtures() {
        let g = open(&fx);
        let oracle = GdlGame::load(&fx.name, g.description()).unwrap();
        // the games of the file replayed in both readers' notations: at every position the side to move, the allowed moves, the end and the facts are equal
        let (mut plies, mut ends) = (0usize, 0usize);
        for (label, line) in &fx.lines {
            let (mut a, mut b) = (g.initial(), oracle.initial());
            for i in 0..=line.len() {
                let mapped: BTreeSet<String> = g.moves(&a).iter().map(|m| g.description_name(&g.move_name(&a, m))).collect();
                let theirs: BTreeSet<String> = oracle.moves(&b).iter().map(|m| oracle.move_name(&b, m)).collect();
                assert_eq!(mapped, theirs, "{}: {label}: moves at ply {i}", fx.name);
                assert_eq!(g.mover(&a).map(|m| g.roles()[m].clone()), oracle.mover(&b).map(|m| oracle.roles()[m].clone()), "{}: {label}: mover at ply {i}", fx.name);
                assert_eq!(g.outcome(&a), oracle.outcome(&b), "{}: {label}: end at ply {i}", fx.name);
                let (fa, mut fb) = (g.base_facts(&a), oracle.base_facts(&b));
                fb.sort();
                assert_eq!(fa, fb, "{}: {label}: facts at ply {i}", fx.name);
                plies += 1;
                if g.outcome(&a).is_some() { ends += 1; }
                let Some((name, term)) = line.get(i) else { break };
                a = g.apply(&a, &g.find_move(&a, name).unwrap_or_else(|| panic!("{label}: {name}")));
                b = oracle.apply(&b, &oracle.find_move(&b, term).unwrap_or_else(|| panic!("{label}: {term}")));
            }
        }
        // and the certify stage's own comparison, from the start, by seeded random play
        let r = coplay(g.game(), &oracle, &|n| g.description_name(n), 3, 2026_11_20);
        assert!(r.disagreements.is_empty(), "{}: {:?}", fx.name, r.disagreements);
        println!("OBS 79.23 B5 {}: both readers equal at {plies} positions of {} named games ({ends} of them ended), and by co-play from the start over {} games, {} plies, {} states ({} with facts compared), {} disagreements", fx.name, fx.lines.len(), r.games, r.plies, r.states, r.fact_states, r.disagreements.len());
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (6) the rate

#[test]
fn layer2_rate_positions_a_second_of_the_count() {
    for fx in fixtures() {
        let g = open(&fx);
        let starts: Vec<(Vec<u64>, <R as Game>::State)> = fx.perft.iter().map(|(_, t, w)| (w.clone(), g.read_state_text(t).unwrap())).collect();
        let mut rates = vec![];
        for run in 0..3 {
            let t0 = Instant::now();
            let mut positions = 0u64;
            for (want, s) in &starts {
                let mut got = vec![0u64; want.len() + 1];
                count(&g, s, want.len(), 1, &mut got);
                positions += got.iter().sum::<u64>();
            }
            let secs = t0.elapsed().as_secs_f64();
            rates.push(positions as f64 / secs);
            println!("OBS 79.23 B6 rate {} run {run}: {positions} positions counted in {:.0} ms: {:.0} positions a second", fx.name, secs * 1000.0, positions as f64 / secs);
        }
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("OBS 79.23 B6 rate {}: median of 3 runs {:.0} positions a second of the count", fx.name, rates[1]);
    }
}
