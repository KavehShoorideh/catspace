//! Plan 78.85 part one: the equal reading. A move the search holds equal to its best at its depth can be one that falls under the best at that depth, because the late-move reduction reads a late quiet reply
//! of the other side two plies shallower and keeps the shallow answer. The traced position (plan 78.85): white king g4, knight d3, pawns h5 and f3 against black king d6 and a rook on b1, white to move; the
//! push h5h6 is held at 0 at depth 12 by the reduced reading of the late quiet reply d6e6, and is under it (the reply wins material) when that reply is read in full.
use cs_game::kernel::KernelGame;
use cs_game::{Game, Role};
use cs_tactics::ledger::{Judge, Search, Worth};
use std::collections::HashMap;

const POSITION: &str = "8/8/3k4/7P/6K1/3N1P2/8/1r6 w - - 1 54";
const DEPTH: i32 = 12;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("q", 4343), ("r", 1605), ("b", 1878), ("n", 1661), ("p", 1045)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "t".into(), table, ours_first: false }
}

fn index_of(g: &KernelGame, s: &<KernelGame as Game>::State, name: &str) -> usize {
    g.moves(s).iter().position(|m| g.move_name(s, m) == name).unwrap()
}

/// The probe "at least 0 at depth 12?" of the push, on a fresh table, with the late-move reduction on and off.
fn probe(late: bool) -> i64 {
    let g = KernelGame::from_fen("chess", POSITION).unwrap();
    let s = g.initial();
    let me: Role = g.mover(&s).unwrap();
    let j = record();
    let mut x = Search::with_fuse(&g, &j, me, 50_000_000);
    x.late_reduce = late;
    let i = index_of(&g, &s, "h5h6");
    let r = x.probes(&s, DEPTH, 0, Some(&[i]));
    assert_eq!(r.len(), 1);
    r[0].1
}

#[test]
fn the_reduced_reading_holds_the_push_equal_and_the_full_reading_does_not() {
    assert!(probe(true) >= 0, "the search's own reading holds the push at the best");
    assert!(probe(false) < 0, "read without the reduction the push is under the best");
}

#[test]
fn the_second_reading_of_ties_drops_the_push() {
    let g = KernelGame::from_fen("chess", POSITION).unwrap();
    let s = g.initial();
    let me: Role = g.mover(&s).unwrap();
    let j = record();
    let i = index_of(&g, &s, "h5h6");
    let mut x = Search::with_fuse(&g, &j, me, 50_000_000);
    assert_eq!(x.verify(&s, DEPTH, 0, &[i]), Vec::<usize>::new());
}

/// Part two: a line read to one reply is read on through the captures its horizon resolves, so what the line shows is what the search counted: black pawns on c6 and d5, a white pawn on e4 and a white knight on c3;
/// the taking e4d5, the retaking c6d5, and the knight takes on d5 in the tail (the search's value counts it; the printed line stopped before it).
#[test]
fn a_line_read_to_one_reply_is_read_on_through_its_captures() {
    use cs_tactics::lines::{read_all, LineParams};
    let g = KernelGame::from_fen("chess", "4k3/8/2p5/3p4/4P3/2N5/8/4K3 w - - 0 1").unwrap();
    let s = g.initial();
    let j = record();
    let reads = read_all(&g, &j, &s, 0, LineParams { depth: 1, fuse: 100_000 });
    let take = reads.iter().find(|r| r.names[0] == "e4d5").unwrap();
    let shown: i64 = take.events.iter().map(|(mine, k)| if *mine { -j.worth(k) } else { j.worth(k) }).sum();
    assert_eq!(shown, take.value, "the events of the line add up to the line's value: {:?} {:?}", take.names, take.events);
    assert!(!take.unsettled);
    assert!(take.names.len() >= 3, "the line is read on past its first reply: {:?}", take.names);
}

/// Plan 78.85 follow-up: with the second reading off `ties` is the equal reading as it was (one null-window reading per root move on the table of the run), on several positions; with it on the set is within that.
#[test]
fn with_the_second_reading_off_ties_is_the_old_equal_reading() {
    let fens = [POSITION, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1", "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", "4k3/8/2p5/3p4/4P3/2N5/8/4K3 w - - 0 1"];
    for fen in fens {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let me: Role = g.mover(&s).unwrap();
        let j = record();
        let make = |verify: bool| {
            let mut x = Search::with_fuse(&g, &j, me, 100_000);
            x.verify_ties = verify;
            let ln = x.run(&s, 64);
            x.rebudget(50_000_000);
            (x, ln)
        };
        let (mut off, ln) = make(false);
        let (mut reference, _) = make(true);
        let old: Vec<usize> = reference.probes(&s, ln.depth, ln.value, None).into_iter().filter(|(_, p)| *p >= ln.value).map(|(i, _)| i).collect();
        let got = off.ties(&s, ln.depth, ln.value);
        assert_eq!(got, old, "{fen}");
        let (mut on, _) = make(true);
        let new = on.ties(&s, ln.depth, ln.value);
        assert!(new.iter().all(|i| old.contains(i)), "{fen}: {new:?} not within {old:?}");
    }
}
