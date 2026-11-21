//! THE TWO MEMORIES AND THE FOUR CHANGES (plan 79.19 part B; SYSTEM_SPEC 11 "two memories", "the go on layer 1"). The interface is in the crate's doc comment (lib.rs); this module is its body. Game-free.
//! THE LONG-TERM MEMORY is one store for every game the system learns, partitioned by the key the game itself reports (`GameKey`: its name and the content address of its rules). A partition is immutable and content-addressed; a change makes a new one with the old as its parent. The wake holds a `Reader`
//! opened for one game: it sees that game's partition and has no way to change it. The only door of change is `Sleep`. THE MEMORY OF THE GAME AT HAND is `GameMemory`: statements and typed slots, written by the wake, changed in place with every change logged, emptied by `end()`.
//! A typed thing a memory keeps (a trained model, a table, the open tactics) is a `Kept`: it is reached through its own interface as its own type; its JSON is how a partition is written, and `Decoders` read it back (a type of a crate above is decoded by a function that crate registers; it is never copied here).
use crate::sha::content_id;
use crate::statement::{Change, Holder, Standing, Statement};
use crate::fact::Fact;
use serde_json::{json, Value};
use std::any::Any;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub const FORMAT: u64 = 2;

/// What the game itself reports: its name and the content address of its rules. Never a word typed in a layer crate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GameKey { pub name: String, pub rules: String }

impl GameKey {
    /// The key of a game from its name and the text of its rules.
    pub fn of(name: &str, rules_text: &str) -> GameKey { GameKey { name: name.to_string(), rules: content_id(rules_text.as_bytes()) } }
    /// The folder name of the partition of this game.
    pub fn folder(&self) -> String { format!("{}__{}", self.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect::<String>(), self.rules) }
    pub fn to_json(&self) -> Value { json!({"name": self.name, "rules": self.rules}) }
    pub fn from_json(v: &Value) -> Result<GameKey, String> {
        Ok(GameKey { name: v.get("name").and_then(|x| x.as_str()).ok_or("a game key has a name")?.to_string(), rules: v.get("rules").and_then(|x| x.as_str()).ok_or("a game key has the address of its rules")?.to_string() })
    }
}

/// A typed thing a memory keeps, reached only through its own interface.
pub trait Kept: Any + Send + Sync {
    /// the name of its kind (the key `Decoders` read it back by)
    fn kind(&self) -> &'static str;
    /// how it came to be held: a table counted from results is a belief with its count
    fn standing(&self) -> Standing;
    /// (what it takes, what it returns), in words
    fn describe(&self) -> (&'static str, &'static str);
    /// its text form (the way a partition is written)
    fn to_json(&self) -> Value;
    fn as_any(&self) -> &dyn Any;
}

/// The functions that read a kept thing back from its JSON, by kind.
#[derive(Default, Clone)]
pub struct Decoders { by_kind: BTreeMap<String, fn(&Value) -> Result<Arc<dyn Kept>, String>> }

impl Decoders {
    pub fn new() -> Decoders { Decoders::default() }
    pub fn with(mut self, kind: &str, f: fn(&Value) -> Result<Arc<dyn Kept>, String>) -> Decoders { self.by_kind.insert(kind.to_string(), f); self }
    fn decode(&self, kind: &str, v: &Value) -> Result<Arc<dyn Kept>, String> { (self.by_kind.get(kind).ok_or_else(|| format!("no decoder for the kind {kind}"))?)(v) }
}

/// One partition of the long-term memory: the statements and the named entries of one game, at one moment, with the snapshot it came from.
pub struct Partition { pub key: GameKey, pub parent: Option<String>, statements: BTreeMap<String, Statement>, entries: BTreeMap<String, Arc<dyn Kept>>, id: OnceLock<String> }

impl std::fmt::Debug for Partition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Partition({} with {} statements, {} entries)", self.key.name, self.statements.len(), self.entries.len()) }
}

impl Partition {
    pub fn new(key: GameKey) -> Partition { Partition { key, parent: None, statements: BTreeMap::new(), entries: BTreeMap::new(), id: OnceLock::new() } }
    /// A partition built in place from its parts (no snapshot is made, no hash is computed until the id is asked): how a caller that is given the tables of a game makes the reader of one decision.
    pub fn with_entry(mut self, name: &str, e: Arc<dyn Kept>) -> Partition { self.entries.insert(name.to_string(), e); self }
    pub fn with_statement(mut self, s: Statement) -> Partition { self.statements.insert(s.id.clone(), s); self }
    fn child(&self) -> Partition { Partition { key: self.key.clone(), parent: Some(self.id().to_string()), statements: self.statements.clone(), entries: self.entries.clone(), id: OnceLock::new() } }
    pub fn statements(&self) -> impl Iterator<Item = &Statement> { self.statements.values() }
    pub fn statement(&self, id: &str) -> Option<&Statement> { self.statements.get(id) }
    pub fn entry(&self, name: &str) -> Option<&dyn Kept> { self.entries.get(name).map(|e| e.as_ref()) }
    pub fn entry_names(&self) -> Vec<String> { self.entries.keys().cloned().collect() }
    /// The text a partition is written as; its address is the address of these bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let entries: Vec<Value> = self.entries.iter().map(|(n, e)| json!({"name": n, "kind": e.kind(), "standing": e.standing().to_json(), "payload": e.to_json()})).collect();
        json!({"format": FORMAT, "game": self.key.to_json(), "parent": self.parent, "statements": self.statements.values().map(Statement::to_json).collect::<Vec<_>>(), "entries": entries}).to_string().into_bytes()
    }
    /// The content address (computed once, on the first ask: building a partition for a decision costs no hashing).
    pub fn id(&self) -> &str { self.id.get_or_init(|| content_id(&self.to_bytes())) }

    /// Read a partition from its bytes; the id is the bytes' address.
    pub fn parse(bytes: &[u8], decoders: &Decoders) -> Result<Partition, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("not JSON: {e}"))?;
        if v.get("format").and_then(|x| x.as_u64()) != Some(FORMAT) { return Err(format!("the partition format is {FORMAT}")); }
        let key = GameKey::from_json(v.get("game").ok_or("a partition names its game")?)?;
        let parent = match v.get("parent") { None | Some(Value::Null) => None, Some(p) => Some(p.as_str().ok_or("parent is an id")?.to_string()) };
        let mut p = Partition { key, parent, statements: BTreeMap::new(), entries: BTreeMap::new(), id: OnceLock::new() };
        for s in v.get("statements").and_then(|x| x.as_array()).ok_or("a partition has statements")? {
            let s = Statement::from_json(s)?;
            if p.statements.insert(s.id.clone(), s).is_some() { return Err("two statements have one id".into()); }
        }
        for e in v.get("entries").and_then(|x| x.as_array()).ok_or("a partition has entries")? {
            let name = e.get("name").and_then(|x| x.as_str()).ok_or("an entry has a name")?.to_string();
            let kind = e.get("kind").and_then(|x| x.as_str()).ok_or("an entry has a kind")?;
            let kept = decoders.decode(kind, e.get("payload").ok_or("an entry has a payload")?)?;
            if p.entries.insert(name, kept).is_some() { return Err("two entries have one name".into()); }
        }
        let _ = p.id.set(content_id(bytes));
        Ok(p)
    }

    /// Read `<id>.json`: the file must hash to its name.
    pub fn load(path: &Path, decoders: &Decoders) -> Result<Partition, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let p = Partition::parse(&bytes, decoders).map_err(|e| format!("{}: {e}", path.display()))?;
        let stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("");
        if stem != p.id() { return Err(format!("{}: the file hashes to {} -- a partition is named by its content", path.display(), p.id())); }
        Ok(p)
    }

    /// Write the partition as `<root>/<game folder>/<id>.json`; returns the path.
    pub fn write(&self, root: &Path) -> Result<PathBuf, String> {
        let dir = root.join(self.key.folder());
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(format!("{}.json", self.id()));
        std::fs::write(&path, self.to_bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(path)
    }
}

/// What the wake holds of the long-term memory: opened for one game, it sees that game's partition and cannot change it.
#[derive(Clone)]
pub struct Reader { part: Arc<Partition> }

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Reader({} @ {} entries, {} statements)", self.part.key.name, self.part.entries.len(), self.part.statements.len()) }
}

impl Reader {
    /// The reader of one partition (the partition is shared, never copied).
    pub fn new(p: Partition) -> Reader { Reader { part: Arc::new(p) } }
    pub fn key(&self) -> &GameKey { &self.part.key }
    pub fn id(&self) -> &str { self.part.id() }
    /// Write this partition as a file under `root` (a copy of what it holds; the memory is not changed).
    pub fn write(&self, root: &Path) -> Result<PathBuf, String> { self.part.write(root) }
    pub fn parent(&self) -> Option<&str> { self.part.parent.as_deref() }
    pub fn statements(&self) -> impl Iterator<Item = &Statement> { self.part.statements() }
    pub fn statement(&self, id: &str) -> Option<&Statement> { self.part.statement(id) }
    pub fn entry(&self, name: &str) -> Option<&dyn Kept> { self.part.entry(name) }
    /// An entry as its own type (None: no such entry, or another type).
    pub fn get<T: Kept>(&self, name: &str) -> Option<&T> { self.part.entries.get(name).and_then(|e| e.as_any().downcast_ref::<T>()) }
    /// The statements of this game's partition whose pattern binds on a position, with their bindings (a limit a statement).
    pub fn bound(&self, idx: &crate::matcher::FactIndex, limit: usize) -> Vec<(&Statement, Vec<crate::matcher::Binding>)> {
        let mut steps = 0u64;
        self.part.statements().map(|s| (s, s.bindings(idx, limit, &mut steps))).filter(|(_, b)| !b.is_empty()).collect()
    }
}

/// The long-term memory: one store for every game, partitioned by `GameKey`. Every partition ever made stays (by its id) as the parent of the next.
#[derive(Default)]
pub struct LongTerm { heads: BTreeMap<GameKey, String>, all: BTreeMap<String, Arc<Partition>> }

impl LongTerm {
    pub fn new() -> LongTerm { LongTerm::default() }
    /// Put a partition in as the head of its game (the first of a game, or one read from a file).
    pub fn insert(&mut self, p: Partition) -> String {
        let id = p.id().to_string();
        self.heads.insert(p.key.clone(), id.clone());
        self.all.insert(id.clone(), Arc::new(p));
        id
    }
    /// The reader of one game (None: this store knows nothing of it).
    pub fn reader(&self, key: &GameKey) -> Option<Reader> { self.heads.get(key).and_then(|id| self.all.get(id)).map(|p| Reader { part: p.clone() }) }
    /// A reader of a snapshot by its id (the parent chain can be read back).
    pub fn reader_at(&self, id: &str) -> Option<Reader> { self.all.get(id).map(|p| Reader { part: p.clone() }) }
    pub fn games(&self) -> Vec<GameKey> { self.heads.keys().cloned().collect() }
    /// Write the head partition of a game under `root` (see `Partition::write`).
    pub fn write_head(&self, key: &GameKey, root: &Path) -> Result<PathBuf, String> { self.heads.get(key).and_then(|id| self.all.get(id)).ok_or_else(|| format!("no partition of {}", key.name))?.write(root) }
    /// The door of change: open to sleep, never taken in the wake.
    pub fn sleep(&mut self) -> Sleep<'_> { Sleep { store: self } }
}

/// The four changes of the long-term memory. Each names the partition it changes (the game's key) and returns the id of the NEW snapshot; the old one stays as its parent.
pub struct Sleep<'a> { store: &'a mut LongTerm }

impl Sleep<'_> {
    fn base(&mut self, key: &GameKey) -> Partition { match self.store.heads.get(key).and_then(|id| self.store.all.get(id)) { Some(p) => p.child(), None => Partition::new(key.clone()) } }
    fn commit(&mut self, p: Partition) -> String { self.store.insert(p) }
    /// ADD a statement that is not there.
    pub fn add(&mut self, key: &GameKey, s: Statement) -> Result<String, String> {
        s.validate()?;
        let mut p = self.base(key);
        if p.statements.contains_key(&s.id) { return Err(format!("{}: a statement of this id is there; replace it", s.id)); }
        p.statements.insert(s.id.clone(), s);
        Ok(self.commit(p))
    }
    /// REPLACE a statement that is there.
    pub fn replace(&mut self, key: &GameKey, s: Statement) -> Result<String, String> {
        s.validate()?;
        let mut p = self.base(key);
        if p.statements.insert(s.id.clone(), s.clone()).is_none() { return Err(format!("{}: no such statement to replace; add it", s.id)); }
        Ok(self.commit(p))
    }
    /// RETRACT a statement or an entry by id (an id that is neither is an error).
    pub fn retract(&mut self, key: &GameKey, id: &str) -> Result<String, String> {
        let mut p = self.base(key);
        if p.statements.remove(id).is_none() && p.entries.remove(id).is_none() { return Err(format!("{id}: nothing of this id to retract")); }
        Ok(self.commit(p))
    }
    /// FALL BY CHANGE: given what a move changed, every statement that read it is dropped and listed (with the new snapshot's id; nothing fell: the same snapshot, no new one).
    pub fn fall_by_change(&mut self, key: &GameKey, c: &Change) -> (Vec<Statement>, Option<String>) {
        let mut p = self.base(key);
        let fallen: Vec<String> = p.statements.values().filter(|s| s.falls_by(c)).map(|s| s.id.clone()).collect();
        if fallen.is_empty() { return (vec![], None); }
        let gone: Vec<Statement> = fallen.iter().filter_map(|id| p.statements.remove(id)).collect();
        (gone, Some(self.commit(p)))
    }
    /// ADD an entry (a typed thing behind its own interface) under a name that is not taken.
    pub fn add_entry(&mut self, key: &GameKey, name: &str, e: Arc<dyn Kept>) -> Result<String, String> {
        let mut p = self.base(key);
        if p.entries.contains_key(name) { return Err(format!("{name}: an entry of this name is there; replace it")); }
        p.entries.insert(name.to_string(), e);
        Ok(self.commit(p))
    }
    /// REPLACE an entry that is there.
    pub fn replace_entry(&mut self, key: &GameKey, name: &str, e: Arc<dyn Kept>) -> Result<String, String> {
        let mut p = self.base(key);
        if p.entries.insert(name.to_string(), e).is_none() { return Err(format!("{name}: no such entry to replace; add it")); }
        Ok(self.commit(p))
    }
}

/// One line of the memory of the game at hand: what was changed and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Logged { pub op: &'static str, pub id: String, pub why: String }

/// What a game's memory hands to sleep when the game ends: the statements that fell (what surprised it) and the log.
#[derive(Debug, Clone, Default)]
pub struct Handed { pub surprised: Vec<Statement>, pub log: Vec<Logged> }

/// THE MEMORY OF THE GAME AT HAND: statements with what they read, and typed slots (the open tactics, the goal held). Written by the wake, changed in place, every change logged.
#[derive(Clone)]
pub struct GameMemory { key: Option<GameKey>, statements: BTreeMap<String, Statement>, slots: BTreeMap<String, Arc<dyn Kept>>, log: Vec<Logged>, fallen: Vec<Statement>, history: Option<crate::history::History>, at_hand: Option<AtHand> }

/// PLAN 79.22 PARTS A AND B. THE POSITION AT HAND AS LAYER 1 KEEPS IT BETWEEN DECISIONS: the key of the position the history ends at, the facts the rules gave for it (what the tags are kept from) and the graph kept step by step (`LiveGraph`, the relations layer 2 wrote). Changed by the moves played, never rebuilt unless the history does not continue.
#[derive(Debug, Clone)]
pub struct AtHand { pub key: u64, pub facts: std::collections::BTreeSet<Fact>, pub graph: crate::live::LiveGraph }

impl Default for GameMemory { fn default() -> GameMemory { GameMemory { key: None, statements: BTreeMap::new(), slots: BTreeMap::new(), log: vec![], fallen: vec![], history: None, at_hand: None } } }

impl std::fmt::Debug for GameMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "GameMemory({} statements, slots {:?}, {} logged)", self.statements.len(), self.slots.keys().collect::<Vec<_>>(), self.log.len()) }
}

impl GameMemory {
    pub fn new(key: GameKey) -> GameMemory { GameMemory { key: Some(key), ..GameMemory::default() } }
    pub fn key(&self) -> Option<&GameKey> { self.key.as_ref() }
    /// PLAN 79.21 STAGE C: the history of the game at hand (a sequence of changes with each position's key), kept with the rest of the game's memory; None until the first position is known. THE RECORD every search takes is `history().record()`: one copy.
    pub fn history(&self) -> Option<&crate::history::History> { self.history.as_ref() }
    /// Bring the history to the record a caller holds as keys (the keys of the positions before the current one, and the current one's): what agrees is kept, the rest is taken back and pushed again (see `History::sync`). Creates the history at the first call.
    pub fn sync_history(&mut self, before: &[u64], current: u64, change: Option<Change>) {
        match &mut self.history { Some(h) => h.sync(before, current, change), None => { let mut h = crate::history::History::new(before.first().copied().unwrap_or(current)); h.sync(before, current, change); self.history = Some(h); } }
    }
    /// PLAN 79.22: the position at hand kept between decisions (None: nothing kept, or the history did not continue). `take_at_hand` hands it to layer 2's keeper, which gives it back with `put_at_hand` once it has brought it to the new position.
    pub fn at_hand(&self) -> Option<&AtHand> { self.at_hand.as_ref() }
    pub fn take_at_hand(&mut self) -> Option<AtHand> { self.at_hand.take() }
    pub fn put_at_hand(&mut self, mut a: AtHand) { a.graph.settle(); self.at_hand = Some(a); }
    /// The history taken out to be kept by layer 2's keeper (None if there is none), and put back.
    pub fn take_history(&mut self) -> Option<crate::history::History> { self.history.take() }
    pub fn put_history(&mut self, h: crate::history::History) { self.history = Some(h); }
    /// RETRACT every statement whose id begins with `prefix` (a layer's own kind of statement, all at once: the history did not continue); returns how many.
    pub fn retract_prefix(&mut self, prefix: &str, why: &str) -> usize {
        let ids: Vec<String> = self.statements.keys().filter(|k| k.starts_with(prefix)).cloned().collect();
        for id in &ids { self.statements.remove(id); self.log.push(Logged { op: "retract", id: id.clone(), why: why.to_string() }); }
        ids.len()
    }
    /// The history, to push a tempo on or take one back (layer 2's tag keeper holds the history itself; this is for a caller that keeps the game's memory only).
    pub fn history_mut(&mut self) -> &mut crate::history::History { self.history.get_or_insert_with(|| crate::history::History::new(0)) }
    pub fn statements(&self) -> impl Iterator<Item = &Statement> { self.statements.values() }
    pub fn statement(&self, id: &str) -> Option<&Statement> { self.statements.get(id) }
    pub fn log(&self) -> &[Logged] { &self.log }
    /// A slot as its own type (None: empty, or another type).
    pub fn slot<T: Kept>(&self, name: &str) -> Option<&T> { self.slots.get(name).and_then(|e| e.as_any().downcast_ref::<T>()) }
    pub fn slot_names(&self) -> Vec<String> { self.slots.keys().cloned().collect() }
    /// Put a slot in place of what was there (logged). `None` empties it.
    pub fn put_slot(&mut self, name: &str, e: Option<Arc<dyn Kept>>, why: &str) {
        match e { Some(e) => { let op = if self.slots.insert(name.to_string(), e).is_some() { "replace" } else { "add" }; self.log.push(Logged { op, id: name.to_string(), why: why.to_string() }); }
                  None => if self.slots.remove(name).is_some() { self.log.push(Logged { op: "retract", id: name.to_string(), why: why.to_string() }); } }
    }
    /// ADD a statement that is not there.
    pub fn add(&mut self, s: Statement, why: &str) -> Result<(), String> {
        s.validate()?;
        if self.statements.contains_key(&s.id) { return Err(format!("{}: a statement of this id is there; replace it", s.id)); }
        self.log.push(Logged { op: "add", id: s.id.clone(), why: why.to_string() });
        self.statements.insert(s.id.clone(), s);
        Ok(())
    }
    /// REPLACE a statement that is there.
    pub fn replace(&mut self, s: Statement, why: &str) -> Result<(), String> {
        s.validate()?;
        if !self.statements.contains_key(&s.id) { return Err(format!("{}: no such statement to replace; add it", s.id)); }
        self.log.push(Logged { op: "replace", id: s.id.clone(), why: why.to_string() });
        self.statements.insert(s.id.clone(), s);
        Ok(())
    }
    /// RETRACT a statement by id.
    pub fn retract(&mut self, id: &str, why: &str) -> Result<Statement, String> {
        let s = self.statements.remove(id).ok_or_else(|| format!("{id}: nothing of this id to retract"))?;
        self.log.push(Logged { op: "retract", id: id.to_string(), why: why.to_string() });
        Ok(s)
    }
    /// FALL BY CHANGE: every statement that read what the change touched is dropped, listed and logged; the list is what surprised this memory.
    pub fn fall_by_change(&mut self, c: &Change, why: &str) -> Vec<String> {
        let ids: Vec<String> = self.statements.values().filter(|s| s.falls_by(c)).map(|s| s.id.clone()).collect();
        for id in &ids {
            if let Some(s) = self.statements.remove(id) { self.fallen.push(s); }
            self.log.push(Logged { op: "fall", id: id.clone(), why: why.to_string() });
        }
        ids
    }
    /// The game ends: the memory is emptied and what surprised it is handed to sleep.
    pub fn end(&mut self) -> Handed {
        let h = Handed { surprised: std::mem::take(&mut self.fallen), log: std::mem::take(&mut self.log) };
        self.statements.clear();
        self.slots.clear();
        self.history = None;
        self.at_hand = None;
        h
    }
}

/// The facts of a position are not kept by a memory; a layer that wants one kept writes it as a statement (the position's key as a read, `Standing::Fact`).
pub fn position_statement(id: &str, facts: &[Fact], key: &str) -> Statement {
    Statement { id: id.to_string(), pattern: facts.iter().map(|f| crate::lemma::Lit { neg: false, relation: f.relation.clone(), args: f.args.iter().map(|a| crate::lemma::Arg::Const(a.clone())).collect() }).collect(),
                says: crate::statement::Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: vec![key.to_string()] }
}
