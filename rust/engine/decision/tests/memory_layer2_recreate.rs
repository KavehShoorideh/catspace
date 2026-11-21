//! PLAN 79.20 PART C: TESTS OF LAYER 2 (the rules and what they say of a position), moved up from plan 79.19's part A (SYSTEM_SPEC 11 "a layer's test uses only that layer and what is below it"; piece: representation, read through the rules).
//! What leaned on the rules is here, named a test of layer 2: a position recreated from the record of its moves, from the game's text form and from the description's state (and the finding that the text form drops one fact), and the fixtures of layer 1 compared with what the rules say.
//! The other four tests of 79.19 part A (statements written and read back, falls, patterns against a plain reference, one game's statement not seen by the other's reader) need no rules: they are layer 1's tests in cs_rep (rust/engine/rep/tests/layer1.rs), reading fixtures.
//! (This file is in cs_decision's tests because it needs cs_gdl to compose a variant's text and the kernel; cs_memory has no such dependency, and adding one changes Cargo.lock, which the kernels' certificates are keyed by.)
use cs_game::gdl_game::GdlGame;
use cs_game::kernel::{uci_to_spec_term, KernelGame};
use cs_game::rng::Rng;
use cs_game::{Fact, Game};
use cs_memory::statement::{decode_facts, encode_facts};
use cs_memory::store::GameKey;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn rules(rel: &str) -> String { std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules").join(rel)).unwrap() }

fn text_of(variant: &str) -> String {
    match variant { "chess" => rules("chess.gdl"), v => cs_gdl::compose(&rules("chess.gdl"), &rules(&format!("variants/{v}.gdl"))).unwrap() }
}

const GAMES: [&str; 2] = ["chess", "kingofthehill"];

/// each game gets its own random games
fn seed_of(v: &str) -> u64 { v.bytes().fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64)) }

/// One generated position: the moves from the start (as the kernel names them), its text, its key and its facts.
struct Pos { moves: Vec<String>, text: String, key: u64, facts: Vec<Fact> }

/// `games` random games of at most `plies` plies each, every position kept.
fn generated(variant: &str, games: usize, plies: usize, seed: u64) -> Vec<Vec<Pos>> {
    let k = KernelGame::new(variant, 518).unwrap();
    let mut rng = Rng::new(seed);
    let mut out = vec![];
    for _ in 0..games {
        let (mut s, mut moves, mut line) = (k.initial(), vec![], vec![]);
        for _ in 0..plies {
            line.push(Pos { moves: moves.clone(), text: k.fen(&s), key: k.key(&s), facts: k.base_facts(&s) });
            let legal = k.moves(&s);
            if legal.is_empty() || k.outcome(&s).is_some() { break; }
            let m = legal[rng.below(legal.len())].clone();
            moves.push(k.move_name(&s, &m));
            s = k.apply(&s, &m);
        }
        out.push(line);
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------------------ (1) a position is encoded and recreated

#[test]
fn layer2_a_position_is_recreated_exactly_by_the_record_of_its_moves_by_its_text_and_by_the_description() {
    for v in GAMES {
        let k = KernelGame::new(v, 518).unwrap();
        let g = GdlGame::load(v, &text_of(v)).unwrap();
        let sample = generated(v, 30, 40, 7919 + seed_of(v));
        let (mut n, mut by_replay, mut by_text, mut by_description, mut by_facts, mut dropped_ep) = (0, 0, 0, 0, 0, 0);
        for pos in sample.iter().flatten() {
            n += 1;
            // the game's own state by replaying its moves from the start
            let mut s = k.initial();
            for m in &pos.moves { s = k.apply(&s, &k.parse(&s, m).expect("a move of the record is legal")); }
            assert_eq!((k.key(&s), k.fen(&s)), (pos.key, pos.text.clone()));
            by_replay += 1;
            // by its text
            let k2 = KernelGame::from_fen(v, &pos.text).unwrap();
            assert_eq!(k2.key(&k2.initial()), pos.key, "{}", pos.text);
            // the text keeps the en passant square only where a capture is legal: the key and every other fact are exact, that one fact may be dropped (counted, printed)
            let (mut a, mut b) = (k2.base_facts(&k2.initial()), pos.facts.clone());
            if a != b { dropped_ep += 1; a.retain(|f| f.relation != "ep_square"); b.retain(|f| f.relation != "ep_square"); }
            assert_eq!(a, b, "{}", pos.text);
            by_text += 1;
            // the facts through their canonical text
            let mut want = pos.facts.clone();
            want.sort();
            want.dedup();
            assert_eq!(decode_facts(&encode_facts(&pos.facts)).unwrap(), want);
            by_facts += 1;
            // the description's state, reached by the same moves, holds the same facts (a sample of the positions: the evaluator is slow)
            if n % 12 == 0 {
                let mut t = g.initial();
                for m in &pos.moves { t = g.apply(&t, &g.find_move(&t, &uci_to_spec_term(m)).expect("the description has the move")); }
                let mut gf = g.base_facts(&t);
                gf.sort();
                assert_eq!(gf, want, "the description's facts at {}", pos.text);
                by_description += 1;
            }
        }
        println!("OBS 79.20 C1 layer 2 {v}: positions {n}, recreated by replay {by_replay}, by text {by_text} (the text dropped the en passant square fact in {dropped_ep}: no capture was legal), facts through their text {by_facts}, the description's state compared on {by_description}");
        assert!(n >= 600);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ the fixtures of layer 1 are what the rules say

/// The fixtures layer 1 reads (rust/engine/rep/tests/fixtures) are data made by a script from the rules; this test, of layer 2, plays each game's record of moves again and compares every position's facts with the file's, and the file's address of the rules with the rules' text now (a changed rule makes the fixtures stale and this fails).
#[test]
fn layer2_the_fixtures_of_layer_1_are_what_the_rules_say() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rep/tests/fixtures");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "fixture")).collect();
    files.sort();
    assert!(files.len() >= 2);
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let (mut key, mut moves, mut cur, mut state, mut k, mut n, mut games): (Option<GameKey>, Vec<String>, BTreeSet<Fact>, Option<<KernelGame as Game>::State>, Option<KernelGame>, usize, usize) = (None, vec![], BTreeSet::new(), None, None, 0, 0);
        let mut ply = 0;
        for line in text.lines() {
            let (tag, rest) = line.split_once('\t').unwrap_or((line, ""));
            match tag {
                "key" => {
                    let kk = GameKey::from_json(&serde_json::from_str(rest).unwrap()).unwrap();
                    assert_eq!(kk, GameKey::of(&kk.name, &text_of(&kk.name)), "{:?}: the file's address of the rules is the rules' now", path.file_name());
                    k = Some(KernelGame::new(&kk.name, 518).unwrap());
                    key = Some(kk);
                }
                "game" => { games += 1; ply = 0; }
                "moves" => { moves = rest.split(' ').filter(|m| !m.is_empty()).map(|m| m.to_string()).collect(); state = Some(k.as_ref().unwrap().initial()); }
                "full" | "delta" => {
                    let k = k.as_ref().unwrap();
                    if tag == "full" { cur = decode_facts(rest).unwrap().into_iter().collect(); } else {
                        let v: serde_json::Value = serde_json::from_str(rest).unwrap();
                        for f in decode_facts(&v["removed"].to_string()).unwrap() { cur.remove(&f); }
                        for f in decode_facts(&v["added"].to_string()).unwrap() { cur.insert(f); }
                        let s = state.take().unwrap();
                        let m = k.parse(&s, &moves[ply - 1]).expect("a move of the record is legal");
                        state = Some(k.apply(&s, &m));
                    }
                    let mut want = k.base_facts(state.as_ref().unwrap());
                    want.sort();
                    want.dedup();
                    assert_eq!(want, cur.iter().cloned().collect::<Vec<_>>(), "{:?}: game {games} ply {ply}", path.file_name());
                    ply += 1;
                    n += 1;
                }
                _ => {}
            }
        }
        println!("OBS 79.20 C2 layer 2 {}: the file's positions {n} of {games} games equal the facts the rules give on replaying each record of moves; its address of the rules is the rules' now", key.unwrap().name);
        assert!(n >= 300);
    }
}
