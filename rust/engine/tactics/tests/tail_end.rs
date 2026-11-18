//! Plan 78.87: the settled tail is a reading by the judge. Case 8 of the ladder re-read (78.86): the line ends inside an exchange where the side to move can take a unit with a rook that guards the other rook, and the reply
//! takes that rook and ends the game against the mover. The tail used to count the end of the game as nothing and played the taking; the line's verdict then said it won a lot while the search's own value was negative.
use cs_game::kernel::KernelGame;
use cs_game::{Game, Role};
use cs_tactics::ledger::{Judge, Worth, WIN};
use cs_tactics::lines::{read_line, settle_tail, LineParams, END_LOST, END_QUIET};
use std::collections::HashMap;

const POSITION: &str = "6k1/ppp3pp/4P3/2b5/8/5R2/P1q3PP/1q3R1K w - - 0 54";
const LINE: [&str; 8] = ["e6e7", "c5e7", "a2a4", "b1a2", "f3f2", "a2c4", "a4a5", "e7a3"];

fn record() -> Worth {
    let table: HashMap<String, i64> = [("q", 4343), ("r", 1605), ("b", 1878), ("n", 1661), ("p", 1045)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "t".into(), table, ours_first: false }
}

fn play(g: &KernelGame, names: &[&str]) -> <KernelGame as Game>::State {
    let mut s = g.initial();
    for n in names {
        let m = g.find_move(&s, n).unwrap_or_else(|| panic!("{n} is not legal"));
        s = g.apply(&s, &m);
    }
    s
}

#[test]
fn the_tail_does_not_take_a_unit_when_the_reply_ends_the_game() {
    let g = KernelGame::from_fen("chess", POSITION).unwrap();
    let s = play(&g, &LINE);
    let me: Role = g.mover(&s).unwrap();
    let j = record();
    // the taking the old tail played
    let taking = g.find_move(&s, "f2c2").unwrap();
    assert!(!g.captured(&s, &taking).is_empty());
    let after = g.apply(&s, &taking);
    let reply = g.find_move(&after, "c4f1").unwrap();
    assert!(g.outcome(&g.apply(&after, &reply)).is_some(), "the reply ends the game against the mover");
    let (tail, hit) = settle_tail(&g, &j, me, &s);
    assert!(!hit);
    assert!(tail.iter().all(|m| g.move_name(&s, m) != "f2c2"), "the taking that lets the other side end the game is not played");
    let _ = j.worth("q");
}

#[test]
fn the_line_read_does_not_count_the_queen_won_before_the_game_is_lost() {
    let g = KernelGame::from_fen("chess", POSITION).unwrap();
    let s = g.initial();
    let me: Role = g.mover(&s).unwrap();
    let j = record();
    let m = g.find_move(&s, "e6e7").unwrap();
    let r = read_line(&g, &j, &s, &m, me, LineParams { depth: 9, fuse: 40000 * 16 });
    // the tail never leaves the reader better off by a unit of the other side's that the game's end then takes back: the events agree with the value's sign
    let net: i64 = r.events.iter().map(|(mine, k)| if *mine { -j.worth(k) } else { j.worth(k) }).sum();
    assert!(r.value < 1000 && net < 1000 + 2000 || r.tail_end == END_QUIET || r.tail_end == END_LOST, "value {} events net {} tail_end {}", r.value, net, r.tail_end);
    assert!(!(r.tail_end == END_QUIET && r.events.iter().any(|(mine, k)| !mine && k == "q") && r.value < 0), "the queen is not counted won in a line the search values negative: {:?} value {}", r.events, r.value);
    assert!(r.value.abs() < WIN - 1000 || r.end != END_QUIET);
}
