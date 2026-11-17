//! Plan 78.77: a unit changing kind is an event of the ledger (`Game::changed`, scored by `Judge::change`); `NoChange` takes it out and nothing else; the line read lists it.
use cs_game::kernel::KernelGame;
use cs_game::{Game, Role};
use cs_tactics::ledger::{Judge, NoChange, Search, Worth};
use cs_tactics::lines::{read_all, LineParams};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("q", 4343), ("r", 1605), ("b", 1878), ("n", 1661), ("p", 1045)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "t".into(), table, ours_first: false }
}

fn value_of<J: Judge>(fen: &str, judge: &J, depth: i32) -> i64 {
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    let me: Role = g.mover(&s).unwrap();
    Search::new(&g, judge, me, 5.0).run(&s, depth).value
}

#[test]
fn the_judge_scores_the_new_kind_less_the_old_for_the_side_that_moved() {
    let w = record();
    assert_eq!(w.change(0, 0, "p", "q"), 4343 - 1045);
    assert_eq!(w.change(0, 1, "p", "q"), -(4343 - 1045));       // the other side's change is a loss for `me`
    assert_eq!(w.change(0, 0, "", "n"), 1661);                  // a unit that arrives from nothing
    assert_eq!(NoChange(record()).change(0, 0, "p", "q"), 0);
    assert_eq!(NoChange(record()).delta(0, 1, "q"), w.delta(0, 1, "q"));
    assert_eq!(NoChange(record()).worth("r"), 1605);
}

#[test]
fn a_move_that_can_change_a_units_kind_scores_above_the_same_position_where_it_cannot() {
    let (can, cannot) = ("4k3/8/P7/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/P7/4K3 w - - 0 1");
    assert_eq!(value_of(can, &record(), 3), 4343 - 1045);
    assert_eq!(value_of(cannot, &record(), 3), 0);
    assert_eq!(value_of(can, &NoChange(record()), 3), 0);
}

#[test]
fn the_line_read_lists_the_change_whatever_the_judge() {
    let g = KernelGame::from_fen("chess", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let s = g.initial();
    for on in [true, false] {
        let p = LineParams { depth: 2, fuse: 100_000 };
        let reads = if on { read_all(&g, &record(), &s, 0, p) } else { read_all(&g, &NoChange(record()), &s, 0, p) };
        let q = reads.iter().find(|r| r.names[0] == "a7a8q").unwrap();
        assert_eq!(q.changes, vec![(true, "p".to_string(), "q".to_string())]);
        assert_eq!(q.value, if on { 4343 - 1045 } else { 0 });
    }
}
