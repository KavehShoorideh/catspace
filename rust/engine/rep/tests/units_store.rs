//! PLAN 79.36 PART C, LAYER 1, THE TWO MEMORIES: the game key, the partition, the reader, the long-term store and its four changes, the memory of the game at hand (SYSTEM_SPEC 11, last block; piece: representation; ledger row 43).
//! Expected values are written here by hand. No object of a game is named (tests/test_plan78.py scans for it).
use cs_rep::fact::Fact;
use cs_rep::history::History;
use cs_rep::lemma::Lit;
use cs_rep::matcher::FactIndex;
use cs_rep::serde_json::{json, Value};
use cs_rep::statement::{Change, Holder, Says, Standing, Statement};
use cs_rep::store::{self, AtHand, Decoders, GameKey, GameMemory, Kept, LongTerm, Partition, Reader};
use cs_rep::live::{LiveGraph, LiveUnit};
use std::any::Any;
use std::sync::Arc;

#[derive(Debug, PartialEq)]
struct Table { rows: u64 }

impl Kept for Table {
    fn kind(&self) -> &'static str { "table" }
    fn standing(&self) -> Standing { Standing::Belief { probability: 0.5, count: self.rows } }
    fn describe(&self) -> (&'static str, &'static str) { ("a key", "a number") }
    fn to_json(&self) -> Value { json!({"rows": self.rows}) }
    fn as_any(&self) -> &dyn Any { self }
}

fn decode_table(v: &Value) -> Result<Arc<dyn Kept>, String> { Ok(Arc::new(Table { rows: v.get("rows").and_then(|x| x.as_u64()).ok_or("a table has rows")? })) }
fn decoders() -> Decoders { Decoders::new().with("table", decode_table) }
fn key() -> GameKey { GameKey::of("toy game", "the rules text") }
fn st(id: &str, rel: &str) -> Statement { Statement { id: id.into(), pattern: vec![Lit::pos(rel, &["?x"])], says: Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: vec![] } }
fn dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("cs_rep_units_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ------------------------------------------------------------------------------------------------------------------------------ GameKey

#[test]
fn the_key_of_a_game_is_its_name_and_the_content_address_of_its_rules_and_other_rules_give_another_key() {
    let k = GameKey::of("g", "rules");
    assert_eq!(k.name, "g");
    assert_eq!(k.rules.len(), 16);
    assert_eq!(k, GameKey::of("g", "rules"));
    assert_ne!(k, GameKey::of("g", "rulez"));
    assert_ne!(k, GameKey::of("h", "rules"));
}

#[test]
fn the_folder_of_a_game_replaces_what_is_not_a_letter_a_digit_or_a_dash_and_ends_with_the_address() {
    let k = GameKey { name: "a b/c-d.e".into(), rules: "0123456789abcdef".into() };
    assert_eq!(GameKey::folder(&k), "a_b_c-d_e__0123456789abcdef");
    assert_eq!(GameKey::folder(&GameKey { name: String::new(), rules: "ff".into() }), "__ff");
}

#[test]
fn a_game_key_is_written_and_read_back_and_one_without_its_parts_is_refused() {
    let k = key();
    let j = GameKey::to_json(&k);
    assert_eq!(j["name"], "toy game");
    assert_eq!(GameKey::from_json(&j), Ok(k));
    assert_eq!(GameKey::from_json(&json!({"rules": "x"})), Err("a game key has a name".to_string()));
    assert_eq!(GameKey::from_json(&json!({"name": "x"})), Err("a game key has the address of its rules".to_string()));
}

// ------------------------------------------------------------------------------------------------------------------------------ Partition

#[test]
fn an_empty_partition_has_no_parent_no_statements_and_no_entries() {
    let p = Partition::new(key());
    assert!(p.parent.is_none());
    assert_eq!(p.statements().count(), 0);
    assert!(Partition::entry_names(&p).is_empty());
    assert!(Partition::statement(&p, "s").is_none());
    assert!(Partition::entry(&p, "t").is_none());
}

#[test]
fn a_partition_built_with_a_statement_and_an_entry_gives_them_back_by_id_and_by_name() {
    let p = Partition::new(key()).with_statement(st("s1", "a")).with_statement(st("s2", "b")).with_entry("t", Arc::new(Table { rows: 7 }));
    assert_eq!(Partition::statement(&p, "s2").map(|s| s.pattern[0].relation.as_str()), Some("b"));
    assert!(Partition::statement(&p, "s3").is_none());
    assert_eq!(Partition::entry_names(&p), vec!["t".to_string()]);
    assert_eq!(Partition::entry(&p, "t").map(|e| e.kind()), Some("table"));
    assert_eq!(p.statements().map(|s| s.id.clone()).collect::<Vec<_>>(), vec!["s1".to_string(), "s2".to_string()], "in the order of the ids");
}

#[test]
fn a_statement_with_the_same_id_replaces_the_one_built_before() {
    let p = Partition::new(key()).with_statement(st("s", "a")).with_statement(st("s", "b"));
    assert_eq!(p.statements().count(), 1);
    assert_eq!(Partition::statement(&p, "s").unwrap().pattern[0].relation, "b");
}

#[test]
fn the_bytes_of_a_partition_name_its_format_its_game_and_its_parts_and_its_address_is_the_address_of_the_bytes() {
    let p = Partition::new(key()).with_statement(st("s", "a")).with_entry("t", Arc::new(Table { rows: 3 }));
    let v: Value = cs_rep::serde_json::from_slice(&Partition::to_bytes(&p)).unwrap();
    assert_eq!(v["format"], 2);
    assert_eq!(v["game"]["name"], "toy game");
    assert_eq!(v["statements"].as_array().unwrap().len(), 1);
    assert_eq!(v["entries"][0]["kind"], "table");
    assert_eq!(v["entries"][0]["payload"]["rows"], 3);
    assert_eq!(Partition::id(&p), cs_rep::sha::content_id(&Partition::to_bytes(&p)));
    assert_eq!(store::FORMAT, 2);
}

#[test]
fn a_partition_read_from_its_bytes_is_the_one_written_and_keeps_its_address() {
    let p = Partition::new(key()).with_statement(st("s", "a")).with_entry("t", Arc::new(Table { rows: 3 }));
    let back = Partition::parse(&Partition::to_bytes(&p), &decoders()).unwrap();
    assert_eq!(Partition::id(&back), Partition::id(&p));
    assert_eq!(Partition::statement(&back, "s"), Partition::statement(&p, "s"));
    assert_eq!(Partition::entry(&back, "t").unwrap().to_json(), json!({"rows": 3}));
}

#[test]
fn a_partition_that_is_not_json_of_another_format_with_an_unknown_kind_or_a_twin_id_is_refused() {
    assert!(Partition::parse(b"not json", &decoders()).unwrap_err().starts_with("not JSON"));
    assert!(Partition::parse(br#"{"format":1}"#, &decoders()).unwrap_err().contains("partition format is 2"));
    let with_table = Partition::new(key()).with_entry("t", Arc::new(Table { rows: 1 }));
    assert_eq!(Partition::parse(&Partition::to_bytes(&with_table), &Decoders::new()).unwrap_err(), "no decoder for the kind table");
    let twin = Partition::new(key()).with_statement(st("s", "a"));
    let mut v: Value = cs_rep::serde_json::from_slice(&Partition::to_bytes(&twin)).unwrap();
    let again = v["statements"][0].clone();
    v["statements"].as_array_mut().unwrap().push(again);
    assert_eq!(Partition::parse(v.to_string().as_bytes(), &decoders()).unwrap_err(), "two statements have one id");
}

#[test]
fn a_partition_is_written_under_its_game_folder_named_by_its_content_and_read_back_by_that_file_and_not_under_another_name() {
    let root = dir("write");
    let p = Partition::new(key()).with_statement(st("s", "a"));
    let path = Partition::write(&p, &root).unwrap();
    assert_eq!(path, root.join(GameKey::folder(&p.key)).join(format!("{}.json", Partition::id(&p))));
    let back = Partition::load(&path, &decoders()).unwrap();
    assert_eq!(Partition::id(&back), Partition::id(&p));
    let renamed = path.with_file_name("0000000000000000.json");
    std::fs::copy(&path, &renamed).unwrap();
    assert!(Partition::load(&renamed, &decoders()).unwrap_err().contains("named by its content"));
    assert!(Partition::load(&root.join("none.json"), &decoders()).is_err());
    std::fs::remove_dir_all(&root).ok();
}

// ------------------------------------------------------------------------------------------------------------------------------ Reader

fn reader() -> Reader { Reader::new(Partition::new(key()).with_statement(st("s1", "a")).with_statement(st("s2", "b")).with_entry("t", Arc::new(Table { rows: 9 }))) }

#[test]
fn a_reader_shows_the_key_the_address_and_the_statements_of_its_partition() {
    let r = reader();
    assert_eq!(Reader::key(&r), &key());
    assert_eq!(Reader::id(&r).len(), 16);
    assert_eq!(Reader::statements(&r).count(), 2);
    assert_eq!(Reader::statement(&r, "s1").map(|s| s.id.as_str()), Some("s1"));
    assert!(Reader::statement(&r, "none").is_none());
    assert!(Reader::parent(&r).is_none());
}

#[test]
fn a_reader_gives_an_entry_by_name_and_as_its_own_type_and_none_for_another_type_or_a_missing_name() {
    #[derive(Debug)]
    struct Other;
    impl Kept for Other {
        fn kind(&self) -> &'static str { "other" }
        fn standing(&self) -> Standing { Standing::Fact }
        fn describe(&self) -> (&'static str, &'static str) { ("", "") }
        fn to_json(&self) -> Value { Value::Null }
        fn as_any(&self) -> &dyn Any { self }
    }
    let r = reader();
    assert_eq!(Reader::entry(&r, "t").map(|e| e.kind()), Some("table"));
    assert_eq!(Reader::get::<Table>(&r, "t"), Some(&Table { rows: 9 }));
    assert!(Reader::get::<Other>(&r, "t").is_none(), "an entry of another type is none");
    assert!(Reader::get::<Table>(&r, "missing").is_none());
    assert!(Reader::entry(&r, "missing").is_none());
}

#[test]
fn a_kept_thing_names_its_kind_its_text_form_and_gives_itself_back_as_its_own_type() {
    let t = Table { rows: 4 };
    assert_eq!(Kept::kind(&t), "table");
    assert_eq!(Kept::to_json(&t), json!({"rows": 4}));
    assert_eq!(Kept::standing(&t), Standing::Belief { probability: 0.5, count: 4 });
    assert_eq!(Kept::describe(&t), ("a key", "a number"));
    assert_eq!(Kept::as_any(&t).downcast_ref::<Table>(), Some(&Table { rows: 4 }));
    assert!(Kept::as_any(&t).downcast_ref::<String>().is_none());
}

#[test]
fn a_reader_binds_the_statements_whose_pattern_matches_a_position_and_drops_the_rest() {
    let r = reader();
    let idx = FactIndex::new(&[Fact::new("a", &["u"]), Fact::new("a", &["v"])]);
    let got = Reader::bound(&r, &idx, 10);
    assert_eq!(got.len(), 1, "only the statement over a binds");
    assert_eq!(got[0].0.id, "s1");
    assert_eq!(got[0].1.len(), 2);
    let none = Reader::bound(&r, &FactIndex::new(&[]), 10);
    assert!(none.is_empty());
}

#[test]
fn a_reader_writes_its_partition_as_a_file_under_a_root() {
    let root = dir("reader_write");
    let r = reader();
    let path = Reader::write(&r, &root).unwrap();
    assert!(path.exists());
    assert!(path.to_string_lossy().ends_with(&format!("{}.json", Reader::id(&r))));
    std::fs::remove_dir_all(&root).ok();
}

// ------------------------------------------------------------------------------------------------------------------------------ the four changes

#[test]
fn a_statement_added_makes_a_new_snapshot_with_the_old_as_parent_and_the_same_id_twice_is_refused() {
    let mut m = LongTerm::new();
    let first = m.sleep().add(&key(), st("s", "a")).unwrap();
    let second = m.sleep().add(&key(), st("t", "b")).unwrap();
    assert_ne!(first, second);
    let head = m.reader(&key()).unwrap();
    assert_eq!(Reader::id(&head), second);
    assert_eq!(Reader::parent(&head), Some(first.as_str()));
    assert_eq!(Reader::statements(&head).count(), 2);
    assert_eq!(Reader::statements(&m.reader_at(&first).unwrap()).count(), 1, "the old snapshot stays");
    assert!(m.sleep().add(&key(), st("s", "z")).unwrap_err().contains("replace it"));
}

#[test]
fn a_statement_that_is_not_well_formed_is_not_added() {
    let mut m = LongTerm::new();
    let mut bad = st("s", "a");
    bad.id.clear();
    assert_eq!(m.sleep().add(&key(), bad), Err("a statement needs an id".to_string()));
    assert!(m.reader(&key()).is_none(), "nothing was made");
}

#[test]
fn a_statement_replaced_is_the_new_one_in_a_new_snapshot_and_one_that_is_not_there_is_refused() {
    let mut m = LongTerm::new();
    m.sleep().add(&key(), st("s", "a")).unwrap();
    let id = m.sleep().replace(&key(), st("s", "b")).unwrap();
    assert_eq!(Reader::statement(&m.reader(&key()).unwrap(), "s").unwrap().pattern[0].relation, "b");
    assert_eq!(Reader::id(&m.reader(&key()).unwrap()), id);
    assert!(m.sleep().replace(&key(), st("none", "b")).unwrap_err().contains("add it"));
}

#[test]
fn a_statement_or_an_entry_retracted_is_gone_from_the_new_snapshot_and_a_name_that_is_neither_is_refused() {
    let mut m = LongTerm::new();
    m.sleep().add(&key(), st("s", "a")).unwrap();
    m.sleep().add_entry(&key(), "t", Arc::new(Table { rows: 1 })).unwrap();
    m.sleep().retract(&key(), "s").unwrap();
    assert!(Reader::statement(&m.reader(&key()).unwrap(), "s").is_none());
    assert!(Reader::entry(&m.reader(&key()).unwrap(), "t").is_some());
    m.sleep().retract(&key(), "t").unwrap();
    assert!(Reader::entry(&m.reader(&key()).unwrap(), "t").is_none());
    assert_eq!(m.sleep().retract(&key(), "t"), Err("t: nothing of this id to retract".to_string()));
}

#[test]
fn what_a_change_touched_falls_and_is_listed_and_a_change_that_touches_nothing_makes_no_snapshot() {
    let mut m = LongTerm::new();
    m.sleep().add(&key(), st("s", "a")).unwrap();
    let kept = m.sleep().add(&key(), st("t", "b")).unwrap();
    let touch_nothing = Change { added: vec![Fact::new("z", &["1"])], ..Default::default() };
    assert_eq!(m.sleep().fall_by_change(&key(), &touch_nothing), (vec![], None));
    assert_eq!(Reader::id(&m.reader(&key()).unwrap()), kept, "the head did not move");
    let touch_a = Change { removed: vec![Fact::new("a", &["1"])], ..Default::default() };
    let (gone, id) = m.sleep().fall_by_change(&key(), &touch_a);
    assert_eq!(gone.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["s"]);
    assert_eq!(Reader::id(&m.reader(&key()).unwrap()), id.unwrap());
    assert_eq!(Reader::statements(&m.reader(&key()).unwrap()).count(), 1);
}

#[test]
fn an_entry_added_under_a_free_name_and_replaced_under_a_taken_one_and_the_other_two_cases_are_refused() {
    let mut m = LongTerm::new();
    m.sleep().add_entry(&key(), "t", Arc::new(Table { rows: 1 })).unwrap();
    assert!(m.sleep().add_entry(&key(), "t", Arc::new(Table { rows: 2 })).unwrap_err().contains("replace it"));
    m.sleep().replace_entry(&key(), "t", Arc::new(Table { rows: 5 })).unwrap();
    assert_eq!(Reader::get::<Table>(&m.reader(&key()).unwrap(), "t"), Some(&Table { rows: 5 }));
    assert!(m.sleep().replace_entry(&key(), "free", Arc::new(Table { rows: 5 })).unwrap_err().contains("add it"));
}

#[test]
fn the_store_lists_its_games_reads_a_head_by_key_and_none_for_a_game_it_does_not_know_and_writes_a_head() {
    let mut m = LongTerm::new();
    let other = GameKey::of("other", "x");
    assert!(m.reader(&other).is_none() && m.games().is_empty());
    m.insert(Partition::new(key()));
    m.insert(Partition::new(other.clone()));
    let mut games = m.games();
    games.sort();
    assert_eq!(games.len(), 2);
    assert!(games.contains(&other));
    assert!(m.reader_at("0000").is_none());
    let root = dir("head");
    let path = m.write_head(&key(), &root).unwrap();
    assert!(path.exists());
    assert_eq!(m.write_head(&GameKey::of("none", "x"), &root), Err("no partition of none".to_string()));
    std::fs::remove_dir_all(&root).ok();
}

// ------------------------------------------------------------------------------------------------------------------------------ GameMemory

#[test]
fn a_memory_made_for_a_game_holds_its_key_and_a_default_one_holds_none() {
    assert_eq!(GameMemory::key(&GameMemory::new(key())), Some(&key()));
    assert!(GameMemory::key(&GameMemory::default()).is_none());
}

#[test]
fn a_statement_added_replaced_and_retracted_in_the_game_at_hand_is_logged_each_time_with_its_reason() {
    let mut g = GameMemory::new(key());
    g.add(st("s", "a"), "first").unwrap();
    assert!(g.add(st("s", "a"), "again").is_err());
    g.replace(st("s", "b"), "second").unwrap();
    assert_eq!(GameMemory::statement(&g, "s").unwrap().pattern[0].relation, "b");
    assert!(g.replace(st("none", "b"), "x").is_err());
    let gone = g.retract("s", "third").unwrap();
    assert_eq!(gone.id, "s");
    assert!(g.retract("s", "x").is_err());
    let log: Vec<(&str, &str)> = GameMemory::log(&g).iter().map(|l| (l.op, l.why.as_str())).collect();
    assert_eq!(log, vec![("add", "first"), ("replace", "second"), ("retract", "third")]);
    assert_eq!(g.statements().count(), 0);
}

#[test]
fn a_prefix_retracts_every_statement_whose_id_begins_with_it_and_returns_how_many() {
    let mut g = GameMemory::new(key());
    for id in ["tag:1", "tag:2", "other"] { g.add(st(id, "a"), "w").unwrap(); }
    assert_eq!(g.retract_prefix("tag:", "history broke"), 2);
    assert_eq!(g.statements().map(|s| s.id.clone()).collect::<Vec<_>>(), vec!["other".to_string()]);
    assert_eq!(g.retract_prefix("zzz", "none"), 0);
    assert_eq!(GameMemory::log(&g).iter().filter(|l| l.op == "retract" && l.why == "history broke").count(), 2);
}

#[test]
fn a_slot_is_put_replaced_and_emptied_with_a_log_line_each_and_read_as_its_own_type() {
    let mut g = GameMemory::new(key());
    g.put_slot("open", Some(Arc::new(Table { rows: 1 })), "made");
    g.put_slot("open", Some(Arc::new(Table { rows: 2 })), "again");
    assert_eq!(g.slot::<Table>("open"), Some(&Table { rows: 2 }));
    assert_eq!(g.slot_names(), vec!["open".to_string()]);
    g.put_slot("open", None, "emptied");
    g.put_slot("open", None, "nothing to empty");
    assert!(g.slot::<Table>("open").is_none() && g.slot_names().is_empty());
    let ops: Vec<&str> = GameMemory::log(&g).iter().map(|l| l.op).collect();
    assert_eq!(ops, vec!["add", "replace", "retract"], "emptying an empty slot logs nothing");
}

#[test]
fn what_a_change_touched_falls_from_the_game_at_hand_and_is_handed_to_sleep_when_the_game_ends() {
    let mut g = GameMemory::new(key());
    g.add(st("s", "a"), "w").unwrap();
    g.add(st("t", "b"), "w").unwrap();
    let fell = g.fall_by_change(&Change { added: vec![Fact::new("a", &["1"])], ..Default::default() }, "a move");
    assert_eq!(fell, vec!["s".to_string()]);
    assert_eq!(g.statements().count(), 1);
    let handed = g.end();
    assert_eq!(handed.surprised.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["s"]);
    assert!(handed.log.iter().any(|l| l.op == "fall" && l.why == "a move"));
    assert_eq!(g.statements().count(), 0, "the memory is emptied");
    assert!(GameMemory::log(&g).is_empty());
    assert!(g.end().surprised.is_empty(), "nothing is handed twice");
}

#[test]
fn the_history_of_the_game_at_hand_is_made_by_the_first_sync_taken_out_and_put_back() {
    let mut g = GameMemory::new(key());
    assert!(g.history().is_none());
    g.sync_history(&[1, 2], 3, None);
    assert_eq!(g.history().unwrap().keys().to_vec(), vec![1, 2, 3]);
    let taken: History = g.take_history().unwrap();
    assert!(g.history().is_none() && g.take_history().is_none());
    g.put_history(taken);
    assert_eq!(g.history().unwrap().current_key(), 3);
    g.end();
    assert!(g.history().is_none(), "ending drops it");
    assert_eq!(g.history_mut().keys().to_vec(), vec![0], "a history made when asked for starts at the key nought");
    g.history_mut().push(Change::default(), 4);
    assert_eq!(g.history().unwrap().len(), 1);
}

#[test]
fn the_position_at_hand_is_put_settled_taken_and_gone_after_the_game_ends() {
    let mut g = GameMemory::new(key());
    assert!(g.at_hand().is_none() && g.take_at_hand().is_none());
    let mut live = LiveGraph::new(Some(0), vec![LiveUnit::new(0, 0, 1)]);
    live.make(Default::default());
    assert_eq!(live.depth(), 1);
    g.put_at_hand(AtHand { key: 11, facts: Default::default(), graph: live });
    assert_eq!(g.at_hand().unwrap().key, 11);
    assert_eq!(g.at_hand().unwrap().graph.depth(), 0, "put settles the graph: nothing to take back");
    let back = g.take_at_hand().unwrap();
    assert_eq!(back.graph.units().len(), 1);
    assert!(g.at_hand().is_none());
    g.put_at_hand(back);
    g.end();
    assert!(g.at_hand().is_none());
}

#[test]
fn a_position_kept_as_a_statement_holds_its_facts_as_constants_and_its_key_as_what_it_read() {
    let facts = [Fact::new("p", &["1", "2"]), Fact::new("q", &[])];
    let s = store::position_statement("pos", &facts, "key:5");
    assert_eq!(s.id, "pos");
    assert_eq!(s.pattern.len(), 2);
    assert_eq!(s.pattern[0], Lit::pos("p", &["1", "2"]));
    assert_eq!((s.says, s.standing, s.also), (Says::Holds, Standing::Fact, vec!["key:5".to_string()]));
    assert!(Statement::falls_by(&store::position_statement("pos", &facts, "key:5"), &Change { keys: vec!["key:5".into()], ..Default::default() }));
}
