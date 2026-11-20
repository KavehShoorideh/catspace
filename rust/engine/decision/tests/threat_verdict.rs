//! Plan 79.05 (SYSTEM_SPEC 8 step 3, 4): UNIT TESTS of the port of the concrete threat list's judgment (cs_decision::threat_verdict): small constructed positions read by the engine's own line reader, the verdict asserted against its Python definition
//! (threat_concrete.verdict_of: like for like cancels, the net of the kinds left in the record's worths, the resolution, the words).
use cs_decision::threat_verdict::{best_taking, cancel_like, verdict_of, Record, VerdictKind};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::ledger::Worth;
use cs_tactics::lines::{read_takings, LineParams};
use std::collections::HashMap;

fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }

fn record() -> Record { Record { worth: table(), half: HashMap::new() } }

fn reads(fen: &str, on: &str) -> Vec<cs_tactics::lines::LineRead> {
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    let me = g.mover(&s).unwrap();
    let place = g.place_names().iter().position(|p| p == on).unwrap() as u16;
    read_takings(&g, &Worth { name: "t".into(), table: table(), ours_first: false }, &s, place, me, LineParams { depth: 6, fuse: 40_000 })
}

#[test]
fn like_for_like_cancels_exactly_and_what_is_left_is_counted() {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let (ll, lw, pairs) = cancel_like(&s(&["p", "n"]), &s(&["p", "q"]));
    assert_eq!((ll, lw, pairs), (s(&["n"]), s(&["q"]), 1));
}

#[test]
fn an_undefended_unit_taken_is_confirmed_and_a_defended_one_taken_by_a_larger_is_refuted() {
    let rec = record();
    let free = reads("k7/8/8/3p4/8/8/8/K2R4 w - - 0 1", "d5");
    let v = best_taking(&free, &rec).unwrap();
    assert_eq!(v.kind, VerdictKind::Confirmed);
    assert_eq!((v.taker_won.clone(), v.taker_lost.clone()), (vec!["p".to_string()], vec![]));
    assert_eq!(v.net, 475);
    let guarded = reads("k7/8/2p5/3p4/8/8/8/K2R4 w - - 0 1", "d5");
    let v = best_taking(&guarded, &rec).unwrap();
    assert_eq!(v.kind, VerdictKind::Refuted, "the rook is lost for a pawn: net {}", v.net);
    assert!(v.net < 0);
}

#[test]
fn a_gain_inside_the_resolution_is_not_told_from_nothing() {
    let mut rec = record();
    rec.half.insert("p".to_string(), 600.0);                         // a record that cannot tell a pawn from nothing
    let v = verdict_of(&reads("k7/8/8/3p4/8/8/8/K2R4 w - - 0 1", "d5")[0], &rec);
    assert!(v.net == 475 && v.resolution == 600.0);
    // as in the Python definition: inside the resolution the words EXCHANGE (units traded) or, for a plain taking, CONFIRMED by the last branch; only a clear gain passes the first test
    assert!((v.net as f64) <= v.resolution && v.kind == VerdictKind::Confirmed && v.taker_lost.is_empty());
}

#[test]
fn a_cut_line_is_unread_and_a_confirmed_line_beats_it() {
    let rec = record();
    let mut a = reads("k7/8/8/3p4/8/8/8/K2R4 w - - 0 1", "d5");
    let mut cut = a[0].clone();
    cut.cut = true;
    assert_eq!(verdict_of(&cut, &rec).kind, VerdictKind::Unread);
    a.push(cut.clone());
    assert_eq!(best_taking(&a, &rec).unwrap().kind, VerdictKind::Confirmed);
    assert_eq!(best_taking(&[cut], &rec).unwrap().kind, VerdictKind::Unread);
    assert!(best_taking(&[], &rec).is_none());
}
