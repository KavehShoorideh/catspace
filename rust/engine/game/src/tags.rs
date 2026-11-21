//! PLAN 79.21 STAGE C, LAYER 2 WRITES THE TAGS (SYSTEM_SPEC 11 "layer 1 holds the history; the rights are layer 2's tags"; ledger row 36; piece: representation, the rules above it). Layer 1 holds the game at hand as a SEQUENCE OF CHANGES with the key of each position (`cs_rep::history::History`) and
//! knows no legality. What the rules read off it (a right still held, a taking in passing, the counters, a repeat) is said HERE, by layer 2, as TAGS: statements about the position that are not the units or the tempo, each with what it read (its own fact), so that it falls by change. The
//! history is the truth; the tags are kept step by step, in make and undo, for speed, and are tested equal to the facts the rules give and to the tags derived from the whole history from the start.
//!
//! INTERFACE.
//!   `TagKeeper::new(g, s, &FactDecl, GameMemory) -> TagKeeper`     the tags of a first position are the facts the rules give that are neither a unit nor the tempo (`FactDecl` says which relations those are); each is written to the game's memory as a statement. A position GIVEN WITHOUT ITS HISTORY (a test position, a start in the middle) states its tags as plain facts this way.
//!   `TagKeeper::make(&mut self, g, after)`                          one tempo: the change between the facts of the position before and the one reached is pushed on the history with the key the rules give; the tags the change touched FALL (`GameMemory::fall_by_change`) and the tags it brought are added.
//!   `TagKeeper::undo(&mut self) -> bool`                            the last tempo taken back: the history, the tags and the memory return to what they were.
//!   `TagKeeper::tags() -> BTreeSet<Fact>`                           the tags as they stand; `repeat_tag() -> Fact`: `(repeats n)`, how many earlier positions have the current key (read off the history, never off the facts).
//!   `TagKeeper::history() / memory()`                                what layer 1 holds.
//!   PLAN 79.22 PART B, THE KEEPER LIVES WITH THE GAME'S MEMORY AND IS KEPT ACROSS DECISIONS:
//!   `TagKeeper::over(g, s, &FactDecl, GameMemory, keys_before) -> TagKeeper`   a keeper for a position reached with a record of keys before it and no changes seen (a game begun in the middle, or a history that did not continue): the tags are STATED AS FACTS from what the rules give, the history holds the keys as not-known changes, any old tag of the memory is retracted first.
//!   `TagKeeper::resume(&FactDecl, facts, History, GameMemory) -> TagKeeper`    the keeper taken up again from what layer 1 kept (the position's facts, the history, the memory with its tag statements).
//!   `TagKeeper::into_parts(self) -> (facts, History, GameMemory)`              what goes back into layer 1 between decisions.
//!   `TagKeeper::make_over(&mut self, g, after, skipped)`                      `make` for a position reached by several tempos of which the keys of the earlier ones (`skipped`) were seen but not the positions: they are pushed as not-known changes, the change into `after` is the facts' difference.
//! Called by: the tests, and the wake's keeper (cs_decision::keep) when a game at hand moves on. A tag's statement id is `tag:<fact>`, its pattern the fact itself (constants), what it says `Holds`, its standing `Fact` (it follows from the rules at this position), its holder ours.
use crate::{Fact, Game};
use cs_rep::graph::FactDecl;
use cs_rep::history::History;
use cs_rep::lemma::{Arg, Lit};
use cs_rep::statement::{Change, Holder, Says, Standing, Statement};
use cs_rep::store::GameMemory;
use std::collections::BTreeSet;

fn id_of(f: &Fact) -> String { format!("tag:{f}") }

fn statement_of(f: &Fact) -> Statement {
    Statement { id: id_of(f), pattern: vec![Lit { neg: false, relation: f.relation.clone(), args: f.args.iter().map(|a| Arg::Const(a.clone())).collect() }], says: Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: vec![] }
}

pub struct TagKeeper {
    decl: FactDecl,
    facts: BTreeSet<Fact>,
    history: History,
    memory: GameMemory,
}

impl TagKeeper {
    fn is_tag(decl: &FactDecl, f: &Fact) -> bool { f.relation != decl.units.relation && decl.tempo.as_deref() != Some(f.relation.as_str()) }

    pub fn new<G: Game>(g: &G, s: &G::State, decl: &FactDecl, mut memory: GameMemory) -> TagKeeper {
        let facts: BTreeSet<Fact> = g.base_facts(s).into_iter().collect();
        for f in facts.iter().filter(|f| Self::is_tag(decl, f)) { memory.add(statement_of(f), "a tag of the first position, as the rules give it").expect("a tag is written once"); }
        TagKeeper { decl: decl.clone(), facts, history: History::new(g.key(s)), memory }
    }

    /// A keeper for a position whose changes were not seen: see the interface above.
    pub fn over<G: Game>(g: &G, s: &G::State, decl: &FactDecl, mut memory: GameMemory, keys_before: &[u64]) -> TagKeeper {
        memory.retract_prefix("tag:", "the history did not continue: the tags are stated again as the rules give them");
        let mut k = TagKeeper::new(g, s, decl, memory);
        if let Some(first) = keys_before.first() {
            let mut h = History::new(*first);
            for key in &keys_before[1..] { h.push_unknown(*key); }
            h.push_unknown(g.key(s));
            k.history = h;
        }
        k
    }

    pub fn resume(decl: &FactDecl, facts: BTreeSet<Fact>, history: History, memory: GameMemory) -> TagKeeper { TagKeeper { decl: decl.clone(), facts, history, memory } }

    pub fn into_parts(self) -> (BTreeSet<Fact>, History, GameMemory) { (self.facts, self.history, self.memory) }

    pub fn make_over<G: Game>(&mut self, g: &G, after: &G::State, skipped: &[u64]) {
        for k in skipped { self.history.push_unknown(*k); }
        self.make(g, after);
    }

    pub fn make<G: Game>(&mut self, g: &G, after: &G::State) {
        let now: BTreeSet<Fact> = g.base_facts(after).into_iter().collect();
        let change = Change { added: now.difference(&self.facts).cloned().collect(), removed: self.facts.difference(&now).cloned().collect(), keys: vec![] };
        let tag_change = Change { added: change.added.iter().filter(|f| Self::is_tag(&self.decl, f)).cloned().collect(), removed: change.removed.iter().filter(|f| Self::is_tag(&self.decl, f)).cloned().collect(), keys: vec![] };
        self.memory.fall_by_change(&tag_change, "a tempo was made");
        for f in &tag_change.added { self.memory.add(statement_of(f), "a tag the tempo brought").expect("a tag brought is new"); }
        self.history.push(change, g.key(after));
        self.facts = now;
    }

    pub fn undo(&mut self) -> bool {
        let Some((change, _)) = self.history.undo() else { return false };
        for f in &change.added { self.facts.remove(f); }
        for f in &change.removed { self.facts.insert(f.clone()); }
        for f in change.added.iter().filter(|f| Self::is_tag(&self.decl, f)) { self.memory.retract(&id_of(f), "a tempo was taken back").expect("the tag brought was there"); }
        for f in change.removed.iter().filter(|f| Self::is_tag(&self.decl, f)) { self.memory.add(statement_of(f), "a tag the tempo had dropped, back").expect("a tag returned is not there"); }
        true
    }

    /// The tags as they stand: the facts that are neither a unit nor the tempo.
    pub fn tags(&self) -> BTreeSet<Fact> { self.facts.iter().filter(|f| Self::is_tag(&self.decl, f)).cloned().collect() }

    /// How many earlier positions have the current key, as a fact `(repeats n)`: said by layer 2 from the history's keys.
    pub fn repeat_tag(&self) -> Fact { Fact { relation: "repeats".to_string(), args: vec![self.history.repeats().to_string()] } }

    pub fn history(&self) -> &History { &self.history }
    /// The facts of the position the keeper is at (every fact the rules give, units and tempo included).
    pub fn facts(&self) -> &BTreeSet<Fact> { &self.facts }
    pub fn memory(&self) -> &GameMemory { &self.memory }
}
