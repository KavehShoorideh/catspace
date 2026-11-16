//! Plan 78.61: the judge that counts nothing prefers nothing but a forced end of the game.
use cs_game::kernel::KernelGame;
use cs_game::{Game, Role};
use cs_tactics::ledger::{Count, Judge, Nothing, Search, WIN};

fn value_of<J: Judge>(fen: &str, judge: &J, depth: i32) -> i64 {
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    let me: Role = g.mover(&s).unwrap();
    Search::new(&g, judge, me, 5.0).run(&s, depth).value
}

#[test]
fn a_free_capture_is_worth_nothing_to_the_judge_that_counts_nothing_and_one_to_the_count() {
    let free = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1";
    assert_eq!(value_of(free, &Nothing, 2), 0);
    assert!(value_of(free, &Count, 2) > 0);
}

#[test]
fn a_forced_end_is_valued_above_every_ledger_and_a_forced_loss_below() {
    let ends = "7k/8/6K1/8/8/8/8/R7 w - - 0 1";
    assert!(value_of(ends, &Nothing, 2) >= WIN - 1000);
    let lost = "7k/8/6K1/8/8/8/8/R7 b - - 0 1";
    let v = value_of(lost, &Nothing, 3);
    assert!(v.abs() < WIN - 1000 || v < 0, "the side to move is not on the winning side: {v}");
}

#[test]
fn the_judge_has_no_worth_and_no_change() {
    assert_eq!(Nothing.worth("anything"), 0);
    assert_eq!(Nothing.delta(0, 0, "anything"), 0);
    assert_eq!(Nothing.delta(0, 1, "anything"), 0);
    assert_eq!(Nothing.change(0, 0, "a", "b"), 0);
}
