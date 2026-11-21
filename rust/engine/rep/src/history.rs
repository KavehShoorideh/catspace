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
//!   `History::facts_at(start, i) -> BTreeSet<Fact>`         the facts of position i, derived from the whole history: the facts of the first position and every change up to i applied in order.
//!   `History::sync(&mut self, before: &[u64], current: u64, change: Option<Change>)`   bring the history to a record a caller holds as keys only (the binding's way in): keeps what agrees, takes back what does not, pushes the new positions (the last with `change`, the others with none known).
//!   `to_json / from_json`                                   a history written and read back identical (keys as text, since a key is a 64 bit number).
use crate::fact::Fact;
use crate::statement::Change;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// The tempos made: `changes[i]` leads from position i to position i + 1, whose key is `keys[i + 1]`; `known[i]` says whether that change was given (a record that came as keys only has no change to give).
#[derive(Debug, Clone, PartialEq)]
pub struct History { keys: Vec<u64>, changes: Vec<Change>, known: Vec<bool> }

/// Apply a change to a set of facts: what it removed goes, what it added stays.
pub fn apply_change(set: &mut BTreeSet<Fact>, ch: &Change) {
    for f in &ch.removed { set.remove(f); }
    for f in &ch.added { set.insert(f.clone()); }
}

impl History {
    pub fn new(first_key: u64) -> History { History { keys: vec![first_key], changes: vec![], known: vec![] } }
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
    pub fn repeats(&self) -> usize { let c = self.current_key(); self.record().iter().filter(|k| **k == c).count() }

    pub fn push(&mut self, change: Change, key: u64) { self.changes.push(change); self.known.push(true); self.keys.push(key); }

    pub fn undo(&mut self) -> Option<(Change, u64)> {
        let ch = self.changes.pop()?;
        self.known.pop();
        let key = self.keys.pop().expect("a key for every change");
        Some((ch, key))
    }

    /// The facts of position `i`, derived from the whole history: the facts of the first position and every change up to `i` applied in order. Positions whose change is not known cannot be derived: an error.
    pub fn facts_at(&self, start: &[Fact], i: usize) -> Result<BTreeSet<Fact>, String> {
        if i > self.changes.len() { return Err(format!("no position {i}: the history has {}", self.keys.len())); }
        let mut set: BTreeSet<Fact> = start.iter().cloned().collect();
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
                _ => { self.changes.push(Change::default()); self.known.push(false); self.keys.push(*k); }
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
            if c.is_null() { h.changes.push(Change::default()); h.known.push(false); h.keys.push(*k); continue; }
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
