//! Plan 79.09 B (the clock is held, SYSTEM_SPEC 7 the budget; piece: the wake): UNIT TESTS of the clock of a reading that is a step before the search. A reading past its clock is cut and says so (a gap, never a refutation); with a distant clock it reads exactly as with none.
//! Constructed positions only.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use cs_tactics::lines::{end_answers, end_answers_by, forced_end, forced_end_by, read_best, read_best_by, read_line_in, Forced, LineParams};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn record() -> Worth {
    let table: HashMap<String, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "human".into(), table, ours_first: false }
}

const POS: &str = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7";
/// a forced end of the game within a few plies (the back rank)
const FORCED: &str = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1";

fn past() -> Instant { Instant::now() - Duration::from_millis(1) }
fn far() -> Instant { Instant::now() + Duration::from_secs(3600) }

#[test]
fn a_line_read_past_its_clock_is_cut_and_a_distant_clock_changes_nothing() {
    let g = KernelGame::from_fen("chess", POS).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let j = record();
    let p = LineParams { depth: 5, fuse: 40_000 };
    let m = g.moves(&s).into_iter().find(|m| g.move_name(&s, m) == "c4f7").unwrap();
    let mut none = Search::with_fuse(&g, &j, me, 1);
    let mut cut = Search::with_fuse(&g, &j, me, 1);
    cut.stop_at(past());
    let mut distant = Search::with_fuse(&g, &j, me, 1);
    distant.stop_at(far());
    let (a, b, c) = (read_line_in(&mut none, &s, &m, me, p), read_line_in(&mut cut, &s, &m, me, p), read_line_in(&mut distant, &s, &m, me, p));
    assert!(b.cut, "past its clock the line is cut: a gap");
    assert!(b.nodes < a.nodes, "and it stopped early");
    assert!(!a.cut && !c.cut);
    assert_eq!((a.value, a.names.clone(), a.nodes), (c.value, c.names.clone(), c.nodes), "a distant clock reads as none");
}

#[test]
fn the_baseline_read_stops_at_its_clock_and_reads_as_before_without_one() {
    let g = KernelGame::from_fen("chess", POS).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let j = record();
    let p = LineParams { depth: 4, fuse: 12_000 };
    let a = read_best(&g, &j, &s, me, p);
    let none = read_best_by(&g, &j, &s, me, p, None);
    let distant = read_best_by(&g, &j, &s, me, p, Some(far()));
    let cut = read_best_by(&g, &j, &s, me, p, Some(past()));
    assert_eq!((a.value, a.names.clone(), a.nodes), (none.value, none.names.clone(), none.nodes));
    assert_eq!((a.value, a.names.clone()), (distant.value, distant.names.clone()));
    assert!(cut.cut && cut.nodes < a.nodes);
}

#[test]
fn the_forced_end_is_a_gap_past_its_clock_and_found_as_before_with_a_distant_one() {
    let g = KernelGame::from_fen("chess", FORCED).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let a = forced_end(&g, &s, me, 5, 1_500_000);
    assert!(matches!(a, Forced::Proved { .. } | Forced::Suspected { .. }), "the end is forced here: {a:?}");
    let distant = forced_end_by(&g, &s, me, 5, 1_500_000, Some(far()));
    assert_eq!(format!("{a:?}"), format!("{distant:?}"));
    // a position where no end is forced and the search is long: past its clock the answer is a GAP (the search was cut before the plies asked), never "none"; with a distant clock it is "none"
    let g2 = KernelGame::from_fen("chess", POS).unwrap();
    let s2 = g2.initial();
    let me2 = g2.mover(&s2).unwrap();
    let cut = forced_end_by(&g2, &s2, me2, 7, 1_500_000, Some(past()));
    assert!(matches!(cut, Forced::Gap { .. }), "the clock cuts the search: a gap, not a proof that there is none: {cut:?}");
    let full = forced_end_by(&g2, &s2, me2, 7, 1_500_000, Some(far()));
    assert!(matches!(full, Forced::None { .. }), "{full:?}");
}

#[test]
fn the_answers_to_a_forced_end_are_unread_past_the_clock() {
    let g = KernelGame::from_fen("chess", FORCED).unwrap();
    let s = g.pass(&g.initial()).unwrap();
    let a = end_answers(&g, &s, 0, 5, 200_000);
    let distant = end_answers_by(&g, &s, 0, 5, 200_000, Some(far()));
    assert_eq!(a, distant);
    let cut = end_answers_by(&g, &s, 0, 5, 200_000, Some(past()));
    assert_eq!(cut.len(), a.len(), "every move is listed");
    assert!(cut.iter().all(|x| x.2 == 2 || x.2 == 3 || x.2 == 0 && a.iter().any(|y| y.1 == x.1 && y.2 == 0)), "a move the clock did not reach is unread (2), never an answer");
    assert!(cut.iter().filter(|x| x.2 == 2).count() > 0);
}
