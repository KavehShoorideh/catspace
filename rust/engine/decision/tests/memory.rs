//! PLAN 79.19 PART C, UNIT TESTS (SYSTEM_SPEC 11 "two memories"; ledger rows 13, 34; piece: the memory, read by the wake). The types of this crate are things the two memories keep: each is written to a partition file and read back identical, a wake that reads its tables from a file decides as one given them in place,
//! a reader of another game gives the wake nothing, the open tactics and the goal held are a slot of the game's memory (the decision hands the memory on), and a position recreated from its text has the same part of the base graph (the second half of part A's first test).
use cs_decision::estimate::ChanceTable;
use cs_decision::goal_table::{Field, GoalTable, Level};
use cs_decision::memory::{decoders, Parts, WakeMemory, CARRIED, CHANCES, GOAL_TABLE, REASONS, RECORD};
use cs_decision::pick::{Dir, Reason, Reasons, Side};
use cs_decision::threat_verdict::Record;
use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::rng::Rng;
use cs_game::Game;
use cs_memory::store::{GameKey, Kept, LongTerm, Partition, Reader};
use cs_memory::statement::Says;
use cs_tactics::ledger::Worth;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;
use std::sync::Arc;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn parts() -> Parts {
    let by_key: HashMap<String, (u32, u32)> = [("a|b".to_string(), (30, 9)), ("c".to_string(), (22, 3))].into_iter().collect();
    let goal_table = GoalTable { k: 16, floor: 20, total: (300, 61), levels: vec![Level { fields: vec![Field::Source, Field::Mine], cells: [("x|1".to_string(), (40, 12)), ("y|0".to_string(), (25, 2))].into_iter().collect() }, Level { fields: vec![Field::Mine], cells: [("1".to_string(), (150, 40))].into_iter().collect() }] };
    Parts { record: Some(Record { worth: table(), half: [("n".to_string(), 12.5), ("p".to_string(), 0.1 + 0.2)].into_iter().collect() }), chances: Some(ChanceTable { by_key, by_facts: [("bears".to_string(), (50, 10))].into_iter().collect(), total: (100, 25), floor: 20 }),
            reasons: Some(Reasons { keys: vec![3, 9, 77], reasons: vec![Reason { column: 1, side: Side::Mover, dir: Dir::Up, weight: 0.037 }, Reason { column: 2, side: Side::Other, dir: Dir::Down, weight: -1.5e-3 }], max_size: 2, levels: vec![0, 1, 2, 3], tie_digits: 9 }),
            goal_table: Some(goal_table), measure: Some(("[tuple region(1,1)]".to_string(), 3, "a sample of games".to_string())), more_measures: vec![("measure_two".to_string(), "[tuple region(1,1) corner(1,1)]".to_string(), 3, "another sample".to_string())] }
}
fn key() -> GameKey { GameKey::of("a game", "its rules") }

#[test]
fn every_table_the_wake_reads_is_written_to_a_partition_file_and_read_back_identical() {
    let m = WakeMemory::of(key(), parts(), None);
    let r = m.long.as_ref().unwrap();
    let root = std::env::temp_dir().join(format!("cs_decision_memory_{}", std::process::id()));
    let path = r.write(&root).unwrap();
    let back = Partition::load(&path, &decoders()).unwrap();
    assert_eq!(back.id(), r.id(), "the file is named by its content and reads back to the same address");
    let back = Reader::new(back);
    for name in [RECORD, CHANCES, REASONS, GOAL_TABLE] {
        let (a, b) = (r.entry(name).unwrap(), back.entry(name).unwrap());
        assert_eq!((a.kind(), a.standing(), a.describe(), a.to_json()), (b.kind(), b.standing(), b.describe(), b.to_json()), "{name}");
    }
    assert_eq!(back.statement("measure"), r.statement("measure"));
    assert!(matches!(back.statement("measure_two").map(|s| &s.says), Some(Says::Lower { within: 3, .. })), "a measure kept beside the held one");
    // the typed readings: the same chance from the table read back
    let (t0, t1) = (r.get::<GoalTable>(GOAL_TABLE).unwrap(), back.get::<GoalTable>(GOAL_TABLE).unwrap());
    assert_eq!((t0.k, t0.floor, t0.total, t0.levels.len()), (t1.k, t1.floor, t1.total, t1.levels.len()));
    assert_eq!(r.get::<Record>(RECORD).unwrap().half, back.get::<Record>(RECORD).unwrap().half);
    println!("OBS 79.19 C tables: record, chance table, reasons, goal table and the mined measure written to {} and read back with the same address {}", path.file_name().unwrap().to_string_lossy(), back.id());
    std::fs::remove_dir_all(&root).ok();
}

fn params(memory: WakeMemory, g: &KernelGame) -> WakeParams {
    let goal = g.kind_names().iter().position(|k| k == "king").map(|i| i as u8);
    WakeParams { seconds: 30.0, depth: 4, pieces: Pieces { goals: true, threats: true, direct_names: true, ..Pieces::old_player() }, places: vec![], inputs: WakeInputs { memory, goal, budget: 4, ..WakeInputs::default() } }
}

fn said(d: &cs_decision::wake::WakeDecision) -> String {
    format!("{:?} {} {} {} {:?} {:?} {:?} {:?}", d.mv, d.value, d.depth, d.nodes, d.equal, d.named_mine, d.log.goals.iter().map(|x| &x.chance).collect::<Vec<_>>(), d.log.notes)
}

const FENS: [&str; 3] = ["r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "r3k2r/pp3ppp/2n1b3/q2pP3/3P4/2P2N2/P1Q2PPP/R3KB1R w KQkq - 0 12"];

#[test]
fn a_wake_that_reads_its_tables_from_a_file_decides_as_one_given_them_in_place_and_a_reader_of_another_game_gives_it_nothing() {
    let (g, st) = world();
    let judge = Worth { name: "r".into(), table: table(), ours_first: false };
    let root = std::env::temp_dir().join(format!("cs_decision_memory_wake_{}", std::process::id()));
    let path = WakeMemory::of(key(), parts(), None).long.as_ref().unwrap().write(&root).unwrap();
    let from_file = || WakeMemory { long: Some(Reader::new(Partition::load(&path, &decoders()).unwrap())), game: Default::default(), hold: None };
    let other_game = || { let mut store = LongTerm::new(); store.insert(Partition::new(GameKey::of("another game", "other rules")).with_entry(CHANCES, Arc::new(parts().chances.unwrap())).with_entry(GOAL_TABLE, Arc::new(parts().goal_table.unwrap()))); WakeMemory { long: store.reader(&GameKey::of("another game", "other rules")), game: cs_memory::store::GameMemory::new(key()), hold: None } };
    let (mut with_tables, mut equal) = (0, 0);
    for fen in FENS {
        let s = KernelGame::from_fen("chess", fen).unwrap().initial();
        let mut given = parts();
        given.measure = None;
        let a = decide(&st, &g, &s, &[], &judge, &params(WakeMemory::of(key(), given, None), &g));
        let b = decide(&st, &g, &s, &[], &judge, &params(from_file(), &g));
        let c = decide(&st, &g, &s, &[], &judge, &params(WakeMemory::default(), &g));
        let d = decide(&st, &g, &s, &[], &judge, &params(other_game(), &g));
        with_tables += a.log.goals.iter().filter(|x| x.chance.is_some()).count();
        // the file's partition also holds the mined measure, but the wake holds it only when the caller names it (`hold`), so nothing changes
        assert_eq!(said(&a), said(&b), "{fen}: the file's tables read as the tables given");
        assert_eq!(said(&c), said(&d), "{fen}: a reader of another game's partition is read as no memory of this one: no table speaks");
        equal += 1;
    }
    assert!(with_tables > 0, "the tables did speak somewhere (a goal logged a chance)");
    println!("OBS 79.19 C wake: {equal} positions: the tables from a file decide as the tables in place; {with_tables} goals logged a chance from them");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn the_open_tactics_and_the_goal_held_are_a_slot_of_the_games_memory_and_the_decision_hands_the_memory_on() {
    let (g, st) = world();
    let judge = Worth { name: "r".into(), table: table(), ours_first: false };
    let s = KernelGame::from_fen("chess", FENS[0]).unwrap().initial();
    let mut p = params(WakeMemory::of(key(), Parts { record: Some(Record { worth: table(), half: HashMap::new() }), measure: Some(("region(1,1)".to_string(), 2, "x".to_string())), ..Default::default() }, None), &g);
    p.inputs.memory.hold = Some("measure".to_string());
    assert!(p.inputs.memory.previous().is_none() && p.inputs.memory.game.slot_names().is_empty());
    let d = decide(&st, &g, &s, &[], &judge, &p);
    let held = d.carried().and_then(|c| c.held.as_ref()).expect("the goal held is in the slot after the first decision");
    assert_eq!(d.game.slot_names(), vec![CARRIED.to_string()]);
    assert_eq!(held.since, 0);
    assert!(d.game.log().iter().any(|l| l.op == "add" && l.id == CARRIED), "the slot was written and the change logged");
    // handed on, the next decision reads it and replaces it
    p.inputs.memory.game = d.game.clone();
    assert!(p.inputs.memory.previous().is_some());
    let d2 = decide(&st, &g, &s, &[], &judge, &p);
    assert!(d2.game.log().iter().any(|l| l.op == "replace" && l.id == CARRIED));
    assert!(d.log.measure.is_some(), "a held goal is logged");
    // the game ends: the memory is emptied
    let mut game = d2.game.clone();
    let handed = game.end();
    assert!(game.slot_names().is_empty() && handed.log.len() >= 2);
    // with no goal named, the statement is not held and no goal is logged
    let mut q = params(WakeMemory::of(key(), Parts { measure: Some(("region(1,1)".to_string(), 2, "x".to_string())), ..Default::default() }, None), &g);
    q.inputs.memory.hold = None;
    assert!(decide(&st, &g, &s, &[], &judge, &q).log.measure.is_none());
}

#[test]
fn a_position_recreated_from_its_text_has_the_same_part_of_the_base_graph() {
    let (g, st) = world();
    let mut rng = Rng::new(7919);
    let (mut n, mut same) = (0, 0);
    for _ in 0..12 {
        let mut s = g.initial();
        for _ in 0..40 {
            let (text, key) = (g.fen(&s), g.key(&s));
            let g2 = KernelGame::from_fen("chess", &text).unwrap();
            let s2 = g2.initial();
            assert_eq!(g2.key(&s2), key);
            n += 1;
            if format!("{:?}", st.position(&g, &s)) == format!("{:?}", st.position(&g2, &s2)) { same += 1; }
            let legal = g.moves(&s);
            if legal.is_empty() { break; }
            s = g.apply(&s, &legal[rng.below(legal.len())]);
        }
    }
    println!("OBS 79.19 A1 base graph part: {n} positions recreated from their text, the part of the base graph equal in {same}");
    assert_eq!(n, same);
}
