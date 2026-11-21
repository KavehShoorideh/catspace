//! PLAN 79.21 STAGE C: THE TESTS OF LAYER 1'S HISTORY AND OF THE GRAPH KEPT STEP BY STEP, CALLING NOTHING OF THE RULES (SYSTEM_SPEC 11 "layer 1 holds the history", "a layer's test uses only that layer and what is below it"; piece: representation).
//! The positions of the fixtures are sets of facts; a tempo is the change between two of them; the key of a position, which the rules make in play, is here the content address of its facts (layer 1 holds whatever key it is given).
//! (1) A history written and read back is identical; the facts of each position derived from the whole history equal the file's; the record every search takes is the keys before the current position; repeats are counted.
//! (2) Append and undo return the same key and the same change, in the order they were made.
//! (3) The graph kept step by step (`LiveGraph`: units and tempo, no relation of the rules) after each tempo equals the graph built from that position's facts, and after undo equals the one before.
mod common;
use common::{fixtures, Fixture, Pos};
use cs_rep::fact::Fact;
use cs_rep::graph::Board;
use cs_rep::history::History;
use cs_rep::live::{Edit, LiveGraph, LiveUnit};
use cs_rep::sha::content_id;
use cs_rep::statement::{encode_facts, Change};

/// the key of a position in these tests: the first 64 bits of the content address of its facts
fn key_of(facts: &[Fact]) -> u64 { u64::from_str_radix(&content_id(encode_facts(facts).as_bytes())[..16], 16).unwrap() }

fn history_of(game: &[Pos]) -> History {
    let mut h = History::new(key_of(&game[0].facts));
    for p in &game[1..] {
        let (added, removed) = p.since.clone().expect("a position after the first has its change");
        h.push(Change { added, removed, keys: vec![] }, key_of(&p.facts));
    }
    h
}

#[test]
fn a_history_is_written_and_read_back_identical_and_its_facts_derive_from_the_whole_of_it() {
    for fx in fixtures() {
        let (mut plies, mut repeated_games) = (0usize, 0usize);
        for game in &fx.games {
            let h = history_of(game);
            assert_eq!(h.len(), game.len() - 1);
            assert_eq!(History::from_json(&h.to_json()).unwrap(), h, "{}: written and read back", fx.file);
            for (i, p) in game.iter().enumerate() {
                assert_eq!(h.facts_at(&game[0].facts, i).unwrap().into_iter().collect::<Vec<_>>(), p.facts, "{}: facts of position {i} from the whole history", fx.file);
                assert_eq!(h.key_at(i), Some(key_of(&p.facts)));
            }
            assert_eq!(h.record().len(), h.len());
            assert_eq!(h.current_key(), key_of(&game.last().unwrap().facts));
            // a repeat is a position met again: counted from the record
            let reps = h.record().iter().filter(|k| **k == h.current_key()).count();
            assert_eq!(h.repeats(), reps);
            repeated_games += (reps > 0) as usize;
            plies += h.len();
        }
        println!("OBS 79.21 C-1 {}: {} histories of {plies} plies in all, each written and read back identical, each position's facts derived from the whole history equal to the file's; games ending on a repeat: {repeated_games}", fx.file, fx.games.len());
    }
}

#[test]
fn append_and_undo_return_the_same_key_and_the_same_change_in_order() {
    for fx in fixtures() {
        let mut checked = 0usize;
        for game in &fx.games {
            let mut h = History::new(key_of(&game[0].facts));
            let mut made: Vec<(Change, u64)> = vec![];
            for p in &game[1..] {
                let (added, removed) = p.since.clone().unwrap();
                let (ch, key) = (Change { added, removed, keys: vec![] }, key_of(&p.facts));
                h.push(ch.clone(), key);
                made.push((ch, key));
                assert_eq!(h.current_key(), key);
            }
            for (i, (ch, key)) in made.iter().enumerate().rev() {
                let (back_c, back_k) = h.undo().expect("a tempo to take back");
                assert_eq!((&back_c, &back_k), (ch, key), "{}: undo returns what was made", fx.file);
                assert_eq!((h.len(), h.current_key()), (i, key_of(&game[i].facts)));
                checked += 1;
            }
            assert!(h.undo().is_none() && h.record().is_empty());
        }
        println!("OBS 79.21 C-2 {}: {checked} appends each taken back by an undo returning the same change and key, the history back at its first position", fx.file);
        assert!(checked > 300);
    }
}

/// the edit a tempo makes of the graph, from the change between two positions' facts alone: the units gone and come, the new tempo
fn edit_of(b: &Board, ch: &Change, after: &[Fact]) -> Edit {
    let (_, removed) = b.read_facts(&ch.removed).unwrap();
    let (_, added) = b.read_facts(&ch.added).unwrap();
    let (tempo, _) = b.read_facts(after).unwrap();
    Edit { tempo, removed, added, rewrite: vec![], clear_shuts: true, shuts: vec![] }
}

fn live_of(b: &Board, fx_pos: &Pos) -> LiveGraph {
    let (tempo, cells) = b.read_facts(&fx_pos.facts).unwrap();
    LiveGraph::new(tempo, cells.into_iter().map(|(owner, kind, at)| LiveUnit { owner, kind, at, bears: vec![], reach: vec![], shuts: vec![] }).collect())
}

#[test]
fn the_graph_kept_step_by_step_equals_the_graph_built_from_each_positions_facts_and_undo_returns_the_one_before() {
    for fx in fixtures() {
        let b = &fx.board;
        let (mut made, mut undone) = (0usize, 0usize);
        for game in &fx.games {
            let mut live = live_of(b, &game[0]);
            assert_eq!(live.part(b).definitions(b), b.position_from_facts(&game[0].facts).unwrap().definitions(b));
            for (i, p) in game.iter().enumerate().skip(1) {
                let (added, removed) = p.since.clone().unwrap();
                live.make(edit_of(b, &Change { added, removed, keys: vec![] }, &p.facts));
                assert_eq!(live.part(b).definitions(b), b.position_from_facts(&p.facts).unwrap().definitions(b), "{}: position {i}: kept step by step", fx.file);
                made += 1;
            }
            for i in (1..game.len()).rev() {
                assert!(live.undo());
                assert_eq!(live.part(b).definitions(b), b.position_from_facts(&game[i - 1].facts).unwrap().definitions(b), "{}: undo to position {}", fx.file, i - 1);
                undone += 1;
            }
            assert!(!live.undo());
        }
        println!("OBS 79.21 C-3 {}: the graph kept step by step equals the graph built from the facts after each of {made} tempos and after each of {undone} undos", fx.file);
        assert!(made > 300);
    }
}

#[allow(dead_code)]
fn _uses(_: &Fixture) {}
