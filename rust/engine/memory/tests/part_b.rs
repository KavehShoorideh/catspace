//! PLAN 79.19 PART B: THE TWO MEMORIES AND THE FOUR CHANGES (SYSTEM_SPEC 11 "two memories", "the go on layer 1"). The reader, the four changes on the long-term memory (a new content-addressed snapshot with its parent, named by the partition of the game), the memory of the game at hand with its log, a typed thing behind its own interface.
use cs_game::Fact;
use cs_memory::lemma::Lit;
use cs_memory::statement::{Change, Holder, Says, Standing, Statement};
use cs_memory::store::{Decoders, GameKey, GameMemory, Kept, LongTerm, Partition};
use serde_json::{json, Value};
use std::any::Any;
use std::sync::Arc;

/// A counted table behind its own interface: it takes a key, it returns a probability, its standing is a belief.
#[derive(Debug, Clone, PartialEq)]
struct Table { cells: Vec<(String, u32, u32)> }
impl Table { fn chance(&self, key: &str) -> Option<f64> { self.cells.iter().find(|c| c.0 == key).map(|c| (c.2 as f64 + 1.0) / (c.1 as f64 + 2.0)) } }
impl Kept for Table {
    fn kind(&self) -> &'static str { "test_table" }
    fn standing(&self) -> Standing { Standing::Belief { probability: 0.25, count: self.cells.iter().map(|c| c.1 as u64).sum() } }
    fn describe(&self) -> (&'static str, &'static str) { ("a key", "a probability") }
    fn to_json(&self) -> Value { json!(self.cells.iter().map(|c| json!([c.0, c.1, c.2])).collect::<Vec<_>>()) }
    fn as_any(&self) -> &dyn Any { self }
}
fn decode_table(v: &Value) -> Result<Arc<dyn Kept>, String> {
    let cells = v.as_array().ok_or("cells")?.iter().map(|c| Ok((c[0].as_str().ok_or("key")?.to_string(), c[1].as_u64().ok_or("n")? as u32, c[2].as_u64().ok_or("hits")? as u32))).collect::<Result<Vec<_>, String>>()?;
    Ok(Arc::new(Table { cells }))
}
/// another type of entry, to show a reader asking for the wrong type gets nothing
struct Other;
impl Kept for Other {
    fn kind(&self) -> &'static str { "other" }
    fn standing(&self) -> Standing { Standing::Fact }
    fn describe(&self) -> (&'static str, &'static str) { ("nothing", "nothing") }
    fn to_json(&self) -> Value { Value::Null }
    fn as_any(&self) -> &dyn Any { self }
}

fn key(name: &str) -> GameKey { GameKey::of(name, &format!("rules of {name}")) }
fn st(id: &str, relation: &str, standing: Standing) -> Statement { Statement { id: id.into(), pattern: vec![Lit::pos(relation, &["?a", "?b"])], says: Says::Holds, standing, holder: Holder::Ours, also: vec![] } }
fn gone(relation: &str) -> Change { Change { removed: vec![Fact::new(relation, &["x", "y"])], ..Default::default() } }

#[test]
fn the_four_changes_each_make_a_new_snapshot_with_the_old_as_its_parent() {
    let (a, b) = (key("game-a"), key("game-b"));
    let mut store = LongTerm::new();
    let id1 = store.sleep().add(&a, st("s1", "r", Standing::Fact)).unwrap();
    let id2 = store.sleep().add(&a, st("s2", "q", Standing::Conjecture("a sample".into()))).unwrap();
    let id3 = store.sleep().replace(&a, st("s1", "r", Standing::Proved("a_theorem".into()))).unwrap();
    let id4 = store.sleep().retract(&a, "s2").unwrap();
    let ids = [id1.clone(), id2.clone(), id3.clone(), id4.clone()];
    assert_eq!(ids.iter().collect::<std::collections::BTreeSet<_>>().len(), 4, "four changes, four snapshots");
    let head = store.reader(&a).unwrap();
    assert_eq!((head.id(), head.parent()), (id4.as_str(), Some(id3.as_str())));
    assert_eq!(store.reader_at(&id3).unwrap().parent(), Some(id2.as_str()));
    assert_eq!(store.reader_at(&id2).unwrap().parent(), Some(id1.as_str()));
    assert_eq!(store.reader_at(&id1).unwrap().parent(), None);
    // the older readers still see what they saw
    assert!(store.reader_at(&id2).unwrap().statement("s2").is_some() && head.statement("s2").is_none());
    assert_eq!(head.statement("s1").unwrap().standing, Standing::Proved("a_theorem".into()));
    // the wrong calls are refused: adding what is there, replacing what is not, retracting what is not
    assert!(store.sleep().add(&a, st("s1", "r", Standing::Fact)).is_err());
    assert!(store.sleep().replace(&a, st("nope", "r", Standing::Fact)).is_err());
    assert!(store.sleep().retract(&a, "nope").is_err());
    // another game's partition is not touched by any of it
    assert!(store.reader(&b).is_none());
    store.sleep().add(&b, st("s1", "r", Standing::Fact)).unwrap();
    assert_eq!(store.reader(&a).unwrap().id(), id4);
    assert_eq!(store.games().len(), 2);
}

#[test]
fn fall_by_change_drops_and_lists_what_read_it_and_makes_a_snapshot_only_when_something_fell() {
    let a = key("game-a");
    let mut store = LongTerm::new();
    store.sleep().add(&a, st("reads-r", "r", Standing::Fact)).unwrap();
    let before = store.sleep().add(&a, st("reads-q", "q", Standing::Fact)).unwrap();
    let (none, id) = store.sleep().fall_by_change(&a, &gone("elsewhere"));
    assert!(none.is_empty() && id.is_none());
    assert_eq!(store.reader(&a).unwrap().id(), before, "nothing fell: the same snapshot");
    let (fell, id) = store.sleep().fall_by_change(&a, &gone("r"));
    assert_eq!(fell.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["reads-r"]);
    let r = store.reader(&a).unwrap();
    assert_eq!((Some(r.id()), r.parent()), (id.as_deref(), Some(before.as_str())));
    assert!(r.statement("reads-r").is_none() && r.statement("reads-q").is_some());
}

#[test]
fn a_trained_thing_is_an_entry_reached_as_its_own_type_through_its_own_interface() {
    let a = key("game-a");
    let mut store = LongTerm::new();
    store.sleep().add_entry(&a, "chance", Arc::new(Table { cells: vec![("k".into(), 8, 3)] })).unwrap();
    store.sleep().add_entry(&a, "other", Arc::new(Other)).unwrap();
    let r = store.reader(&a).unwrap();
    assert_eq!(r.get::<Table>("chance").unwrap().chance("k"), Some(0.4));
    assert!(r.get::<Table>("other").is_none() && r.get::<Table>("missing").is_none(), "another type or no such entry: nothing");
    let e = r.entry("chance").unwrap();
    assert_eq!((e.kind(), e.describe(), e.standing()), ("test_table", ("a key", "a probability"), Standing::Belief { probability: 0.25, count: 8 }));
    assert!(store.sleep().add_entry(&a, "chance", Arc::new(Other)).is_err());
    assert!(store.sleep().replace_entry(&a, "chance", Arc::new(Table { cells: vec![("k".into(), 9, 9)] })).is_ok());
    assert!(store.sleep().replace_entry(&a, "nope", Arc::new(Other)).is_err());
    assert!(store.sleep().retract(&a, "other").is_ok());
    assert_eq!(store.reader(&a).unwrap().get::<Table>("chance").unwrap().chance("k"), Some(10.0 / 11.0));
}

#[test]
fn a_partition_with_statements_and_entries_round_trips_through_its_file() {
    let a = key("game-a");
    let decoders = Decoders::new().with("test_table", decode_table);
    let mut store = LongTerm::new();
    store.sleep().add(&a, st("s1", "r", Standing::Belief { probability: 0.1 + 0.2, count: 3 })).unwrap();
    store.sleep().add(&a, st("s2", "q", Standing::Proved("t".into()))).unwrap();
    store.sleep().add_entry(&a, "chance", Arc::new(Table { cells: vec![("k".into(), 8, 3)] })).unwrap();
    let root = std::env::temp_dir().join(format!("cs_memory_part_b2_{}", std::process::id()));
    let path = store.write_head(&a, &root).unwrap();
    assert!(path.to_string_lossy().contains(&a.folder()), "a partition lives in its game's folder");
    let back = Partition::load(&path, &decoders).unwrap();
    assert_eq!(back.id(), store.reader(&a).unwrap().id());
    assert_eq!(back.statements().count(), 2);
    assert_eq!(back.entry("chance").unwrap().to_json(), json!([["k", 8, 3]]));
    // a file that does not hash to its name is refused
    let moved = path.with_file_name("0123456789abcdef.json");
    std::fs::copy(&path, &moved).unwrap();
    assert!(Partition::load(&moved, &decoders).unwrap_err().contains("named by its content"));
    // a kind nobody registered a decoder for is refused, not skipped
    assert!(Partition::parse(&std::fs::read(&path).unwrap(), &Decoders::new()).is_err());
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn the_memory_of_the_game_at_hand_changes_in_place_logs_every_change_and_hands_what_surprised_it_to_sleep() {
    let mut m = GameMemory::new(key("game-a"));
    m.add(st("believes-r", "r", Standing::Belief { probability: 0.6, count: 5 }), "opened").unwrap();
    m.add(st("reads-q", "q", Standing::Fact), "opened").unwrap();
    assert!(m.add(st("reads-q", "q", Standing::Fact), "again").is_err());
    m.replace(st("reads-q", "q", Standing::Conjecture("this game".into())), "learned more").unwrap();
    assert!(m.replace(st("nope", "q", Standing::Fact), "x").is_err());
    m.put_slot("chance", Some(Arc::new(Table { cells: vec![("k".into(), 1, 1)] })), "a table for this opponent");
    assert_eq!(m.slot::<Table>("chance").unwrap().chance("k"), Some(2.0 / 3.0));
    let fell = m.fall_by_change(&gone("r"), "a move removed an r");
    assert_eq!(fell, vec!["believes-r".to_string()]);
    assert!(m.statement("believes-r").is_none() && m.statement("reads-q").is_some());
    assert!(m.fall_by_change(&gone("elsewhere"), "x").is_empty());
    let r = m.retract("reads-q", "no longer wanted").unwrap();
    assert_eq!(r.id, "reads-q");
    assert!(m.retract("reads-q", "again").is_err());
    let ops: Vec<(&str, &str)> = m.log().iter().map(|l| (l.op, l.id.as_str())).collect();
    assert_eq!(ops, vec![("add", "believes-r"), ("add", "reads-q"), ("replace", "reads-q"), ("add", "chance"), ("fall", "believes-r"), ("retract", "reads-q")]);
    assert!(m.log().iter().all(|l| !l.why.is_empty()), "every change says why");
    let handed = m.end();
    assert_eq!(handed.surprised.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["believes-r"], "what fell is what surprised it");
    assert_eq!(handed.log.len(), 6);
    assert!(m.statements().next().is_none() && m.slot_names().is_empty() && m.log().is_empty(), "emptied at the game's end");
}
