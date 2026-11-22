//! PLAN 79.21 STAGE C, THE HISTORY OF THE GAME AT HAND (SYSTEM_SPEC 11 "layer 1 holds the history; the rights are layer 2's tags", "two memories"; ledger row 36; piece: representation). The history is a SEQUENCE OF CHANGES, one a tempo: for each position reached after the first, the facts
//! added and removed (a `Change`) and the KEY of the position reached. Layer 1 knows no legality: it holds what it is given. What the rules read off it (a right still held, a taking in passing, the counters, a repeat) is said by layer 2 as TAGS on nodes (statements with what they
//! read), through the memory's four changes; the history is the truth and the tags are kept step by step for speed.
//!
//! INTERFACE (written before the body; who calls it is said at each).
//!   `History::new(first_key) -> History`                    the history of a game that has reached its first position (the key of that position).
//!   `History::push(&mut self, Change, key)`                 a tempo is made: the position reached has this change from the one before and this key. Called by the wake's binding when the game moves on, by layer 2's tag keeper, by tests.
//!   `History::undo(&mut self) -> Option<(Change, u64)>`     the last tempo is taken back; what it was is returned so the caller can take back what it kept (None at the first position).
//!   `History::len / key_at(i) / current_key / keys()`       the plies made; the key of position i (0 is the first); the key of the position reached; the keys of every position from the first to the current.
//!   `History::record() -> &[u64]`                           THE RECORD OF THE GAME every search takes: the keys of the positions before the current one (a position met again is the rules' draw). ONE COPY: the search reads it from here.
//!   `History::repeats() -> usize`                           how many earlier positions have the current key.
//!   `History::seen() -> &Arc<Seen>`                         PLAN 79.22 PART B: THE KEYS MET, the record as a counted set, kept in step by `push`, `undo` and `sync`, and handed by reference count (never copied) to every reader of a repeat: the search, the measure's lines, the wake. ONE WAY to read a repeat; `Seen::contains / count`.
//!   `Path` (`push / pop / contains`)                       the keys on the line a search walks, made and taken back with its tempos: the search's own repeats, read the way `Seen` reads the record's.
//!   `History::push_unknown(key)`                            a position reached whose change was not seen (the caller skipped it): the key is kept, the change is marked not known.
//!   `History::facts_at(start, i) -> Set<Fact>`         the facts of position i, derived from the whole history: the facts of the first position and every change up to i applied in order.
//!   `History::sync(&mut self, before: &[u64], current: u64, change: Option<Change>)`   bring the history to a record a caller holds as keys only (the binding's way in): keeps what agrees, takes back what does not, pushes the new positions (the last with `change`, the others with none known).
//!   `to_json / from_json`                                   a history written and read back identical (keys as text, since a key is a 64 bit number).
use crate::fact::Fact;
use crate::statement::Change;
use serde_json::{json, Value};
use cs_base::sets::Set;   // plan 79.31 part G: the facts of a position are a set of facts, which is what layer 0's set is (the same ordered set underneath, so nothing is slower)
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

/// A position's key is already a hash: it goes through as it is.
#[derive(Default, Clone, Copy)]
pub struct KeyHasher(u64);
impl Hasher for KeyHasher {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, bytes: &[u8]) { for b in bytes { self.0 = self.0.rotate_left(8) ^ *b as u64; } }
    fn write_u64(&mut self, i: u64) { self.0 = i; }
}

/// PLAN 79.22 PART B. THE KEYS MET: the keys of the positions the game was in before the one it is at, with how often each was met. What a repeat is read from (the rules' draw: a position met again). Layer 1 holds it with the history; every reader borrows it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Seen { n: HashMap<u64, u32, BuildHasherDefault<KeyHasher>> }

impl Seen {
    pub fn new() -> Seen { Seen::default() }
    pub fn contains(&self, key: &u64) -> bool { self.n.contains_key(key) }
    pub fn count(&self, key: u64) -> usize { self.n.get(&key).copied().unwrap_or(0) as usize }
    pub fn is_empty(&self) -> bool { self.n.is_empty() }
    pub fn len(&self) -> usize { self.n.len() }
    pub fn add(&mut self, key: u64) { *self.n.entry(key).or_insert(0) += 1; }
    pub fn remove(&mut self, key: u64) { if let Some(c) = self.n.get_mut(&key) { if *c <= 1 { self.n.remove(&key); } else { *c -= 1; } } }
    pub fn keys(&self) -> impl Iterator<Item = u64> + '_ { self.n.keys().copied() }
    /// The keys met of a record held as a list (the way in of a caller that has no history of its own: a test, a measure); the wake reads `History::seen` instead, which is never made again.
    pub fn of(keys: &[u64]) -> Arc<Seen> { Arc::new(keys.iter().copied().collect()) }
}

impl FromIterator<u64> for Seen {
    fn from_iter<I: IntoIterator<Item = u64>>(it: I) -> Seen { let mut s = Seen::new(); for k in it { s.add(k); } s }
}

/// PLAN 79.22 PART B. THE LINE A SEARCH IS WALKING: the keys of the positions from the root to the one it is at, `push`ed by a tempo made and `pop`ped by a tempo taken back, so that a repeat on its own line is read the same way as a repeat in the game's record (`Seen`): by key, from layer 1. A line is short, so the keys are a list and `contains` scans it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Path { keys: Vec<u64> }

impl Path {
    pub fn new() -> Path { Path::default() }
    pub fn push(&mut self, key: u64) { self.keys.push(key); }
    pub fn pop(&mut self) -> Option<u64> { self.keys.pop() }
    pub fn contains(&self, key: &u64) -> bool { self.keys.contains(key) }
    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }
}

/// The tempos made: `changes[i]` leads from position i to position i + 1, whose key is `keys[i + 1]`; `known[i]` says whether that change was given (a record that came as keys only has no change to give).
#[derive(Debug, Clone, PartialEq)]
pub struct History { keys: Vec<u64>, changes: Vec<Change>, known: Vec<bool>, seen: Arc<Seen> }

/// Apply a change to a set of facts: what it removed goes, what it added stays.
pub fn apply_change(set: &mut Set<Fact>, ch: &Change) {
    for f in &ch.removed { set.remove(f); }
    for f in &ch.added { set.insert(f.clone()); }
}

impl History {
    pub fn new(first_key: u64) -> History { History { keys: vec![first_key], changes: vec![], known: vec![], seen: Arc::new(Seen::new()) } }
    /// The plies made (the positions after the first).
    pub fn len(&self) -> usize { self.changes.len() }
    pub fn is_empty(&self) -> bool { self.changes.is_empty() }
    pub fn key_at(&self, i: usize) -> Option<u64> { self.keys.get(i).copied() }
    pub fn current_key(&self) -> u64 { *self.keys.last().expect("a history has a first position") }
    /// The key of every position from the first to the current.
    pub fn keys(&self) -> &[u64] { &self.keys }
    /// THE RECORD every search takes: the keys of the positions before the current one.
    pub fn record(&self) -> &[u64] { &self.keys[..self.keys.len() - 1] }
    pub fn changes(&self) -> &[Change] { &self.changes }
    pub fn change_known(&self, i: usize) -> bool { self.known.get(i).copied().unwrap_or(false) }
    pub fn repeats(&self) -> usize { self.seen.count(self.current_key()) }
    /// The keys met, shared: a reader holds the reference count, nothing is copied (a later `push` while one is held copies once, then is alone again).
    pub fn seen(&self) -> &Arc<Seen> { &self.seen }

    fn step(&mut self, change: Change, known: bool, key: u64) {
        let c = self.current_key();
        Arc::make_mut(&mut self.seen).add(c);
        self.changes.push(change);
        self.known.push(known);
        self.keys.push(key);
    }

    pub fn push(&mut self, change: Change, key: u64) { self.step(change, true, key); }

    /// A position reached whose change was not seen: its key is kept and the change is marked not known (`facts_at` refuses to cross it).
    pub fn push_unknown(&mut self, key: u64) { self.step(Change::default(), false, key); }

    pub fn undo(&mut self) -> Option<(Change, u64)> {
        let ch = self.changes.pop()?;
        self.known.pop();
        let key = self.keys.pop().expect("a key for every change");
        let c = self.current_key();
        Arc::make_mut(&mut self.seen).remove(c);
        Some((ch, key))
    }

    /// The facts of position `i`, derived from the whole history: the facts of the first position and every change up to `i` applied in order. Positions whose change is not known cannot be derived: an error.
    pub fn facts_at(&self, start: &[Fact], i: usize) -> Result<Set<Fact>, String> {
        if i > self.changes.len() { return Err(format!("no position {i}: the history has {}", self.keys.len())); }
        let mut set: Set<Fact> = start.iter().cloned().collect();
        for j in 0..i {
            if !self.known[j] { return Err(format!("the change into position {} was not given", j + 1)); }
            apply_change(&mut set, &self.changes[j]);
        }
        Ok(set)
    }

    /// Bring the history to a record held as keys only: `before` are the keys of the positions before the current one, `current` the key of the current one. What agrees from the start stays; the rest is taken back and the new positions pushed (the last with `change`, the others with none known).
    pub fn sync(&mut self, before: &[u64], current: u64, change: Option<Change>) {
        let mut want: Vec<u64> = before.to_vec();
        want.push(current);
        let mut agree = 0;
        while agree < want.len() && agree < self.keys.len() && want[agree] == self.keys[agree] { agree += 1; }
        if agree == 0 { *self = History::new(want[0]); agree = 1; }
        while self.keys.len() > agree { self.undo(); }
        let last = want.len() - 1;
        for (i, k) in want.iter().enumerate().skip(agree) {
            match (i == last, &change) {
                (true, Some(c)) => self.push(c.clone(), *k),
                _ => self.push_unknown(*k),
            }
        }
    }

    pub fn to_json(&self) -> Value {
        json!({"keys": self.keys.iter().map(|k| format!("{k:016x}")).collect::<Vec<_>>(),
               "changes": self.changes.iter().zip(&self.known).map(|(c, k)| if *k { json!({"added": c.added.iter().map(fact_json).collect::<Vec<_>>(), "removed": c.removed.iter().map(fact_json).collect::<Vec<_>>(), "keys": c.keys}) } else { Value::Null }).collect::<Vec<_>>()})
    }

    pub fn from_json(v: &Value) -> Result<History, String> {
        let keys = v.get("keys").and_then(|x| x.as_array()).ok_or("history.keys is a list")?.iter().map(|k| k.as_str().ok_or("a key is text".to_string()).and_then(|s| u64::from_str_radix(s, 16).map_err(|e| e.to_string()))).collect::<Result<Vec<_>, _>>()?;
        if keys.is_empty() { return Err("a history has a first position".into()); }
        let mut h = History::new(keys[0]);
        let cs = v.get("changes").and_then(|x| x.as_array()).ok_or("history.changes is a list")?;
        if cs.len() + 1 != keys.len() { return Err("one change for every position after the first".into()); }
        for (c, k) in cs.iter().zip(&keys[1..]) {
            if c.is_null() { h.push_unknown(*k); continue; }
            let facts = |name: &str| -> Result<Vec<Fact>, String> { c.get(name).and_then(|x| x.as_array()).ok_or(format!("change.{name} is a list"))?.iter().map(fact_from).collect() };
            let ks = c.get("keys").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|s| s.as_str().map(|t| t.to_string())).collect()).unwrap_or_default();
            h.push(Change { added: facts("added")?, removed: facts("removed")?, keys: ks }, *k);
        }
        Ok(h)
    }
}

fn fact_json(f: &Fact) -> Value { json!([f.relation, f.args]) }
fn fact_from(v: &Value) -> Result<Fact, String> {
    let a = v.as_array().ok_or("a fact is a pair")?;
    Ok(Fact { relation: a.first().and_then(|x| x.as_str()).ok_or("a relation is text")?.to_string(),
              args: a.get(1).and_then(|x| x.as_array()).ok_or("arguments are a list")?.iter().map(|x| x.as_str().map(|s| s.to_string()).ok_or("an argument is text")).collect::<Result<Vec<_>, _>>()? })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ch(add: &[&str], rem: &[&str]) -> Change { Change { added: add.iter().map(|r| Fact::new(r, &[])).collect(), removed: rem.iter().map(|r| Fact::new(r, &[])).collect(), keys: vec![] } }

    #[test]
    fn push_undo_record_and_repeats() {
        let mut h = History::new(7);
        h.push(ch(&["a"], &[]), 8);
        h.push(ch(&["b"], &["a"]), 7);
        assert_eq!((h.len(), h.current_key(), h.record().to_vec(), h.repeats()), (2, 7, vec![7, 8], 1));
        let (c, k) = h.undo().unwrap();
        assert_eq!((c, k, h.len(), h.repeats()), (ch(&["b"], &["a"]), 7, 1, 0));
        assert!(History::new(1).undo().is_none());
    }

    #[test]
    fn the_keys_met_are_kept_in_step_with_push_undo_and_sync_and_shared_without_copying() {
        let mut h = History::new(7);
        let held = h.seen().clone();                                        // a reader holds the set; a later push copies once and leaves the reader's set as it was
        h.push(ch(&["a"], &[]), 8);
        h.push(ch(&["b"], &["a"]), 7);
        assert_eq!((held.len(), h.seen().len(), h.seen().count(7), h.seen().count(8), h.seen().contains(&9)), (0, 2, 1, 1, false));
        let record: Seen = h.record().iter().copied().collect();
        assert_eq!(**h.seen(), record, "the set is the record");
        h.push_unknown(9);
        h.push_unknown(7);
        assert_eq!((h.repeats(), h.seen().count(7), h.change_known(2), h.facts_at(&[], 3).is_err()), (2, 2, false, true));
        h.undo();
        h.undo();
        assert_eq!((h.repeats(), h.seen().count(7), h.seen().contains(&9)), (1, 1, false));
        h.sync(&[7, 8], 5, None);
        let record: Seen = h.record().iter().copied().collect();
        assert_eq!((**h.seen() == record, h.keys().to_vec()), (true, vec![7, 8, 5]));
        let again: Seen = History::from_json(&h.to_json()).unwrap().seen().as_ref().clone();
        assert_eq!(again, record);
        assert!(!Seen::of(&[1, 2]).contains(&3) && Seen::of(&[1, 1]).count(1) == 2);
    }

    #[test]
    fn the_path_of_a_walk_is_made_and_taken_back_with_its_tempos() {
        let mut p = Path::new();
        p.push(4);
        p.push(5);
        assert!(p.contains(&4) && p.contains(&5) && !p.contains(&6) && p.len() == 2);
        assert_eq!(p.pop(), Some(5));
        assert!(!p.contains(&5) && p.contains(&4));
    }

    #[test]
    fn facts_derive_from_the_whole_history_and_json_round_trips() {
        let mut h = History::new(1);
        h.push(ch(&["a"], &[]), 2);
        h.push(ch(&["b"], &["a"]), 3);
        let start = vec![Fact::new("z", &[])];
        let at2: Vec<String> = h.facts_at(&start, 2).unwrap().iter().map(|f| f.to_string()).collect();
        assert_eq!(at2, vec!["b".to_string(), "z".to_string()]);
        assert_eq!(History::from_json(&h.to_json()).unwrap(), h);
        assert!(h.facts_at(&start, 3).is_err());
    }

    #[test]
    fn sync_keeps_what_agrees_and_takes_back_the_rest() {
        let mut h = History::new(1);
        h.push(ch(&["a"], &[]), 2);
        h.sync(&[1, 2], 3, Some(ch(&["b"], &[])));
        assert_eq!((h.keys().to_vec(), h.change_known(0), h.change_known(1)), (vec![1, 2, 3], true, true));
        h.sync(&[1], 5, None);                                   // a different second position: taken back and pushed again, no change known
        assert_eq!((h.keys().to_vec(), h.change_known(0)), (vec![1, 5], false));
        h.sync(&[9], 10, None);                                  // nothing agrees: a new history
        assert_eq!(h.keys().to_vec(), vec![9, 10]);
    }
}
