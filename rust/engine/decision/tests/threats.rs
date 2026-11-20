//! Plan 79.06 (SYSTEM_SPEC 8 step 3 "the threat list", 4 how a line ends): UNIT TESTS of the Rust concrete threat list (cs_decision::threats): small positions built through the engine's own game interface, each piece's own contract asserted.
//! No puzzle, no stored game, no match, no integration of the wake.
use cs_decision::threat_verdict::{Record, VerdictKind};
use cs_decision::threats::{concrete_threats, every_unit, kinds_ours_less_theirs, stake_closure, stake_of_kinds, ConcreteList, FrameRow, RespState, Source, ThreatParams};
use cs_decision::wake::{Pieces, UnitRef};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}

fn record() -> Record { Record { worth: [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect(), half: HashMap::new() } }
fn at(g: &KernelGame, name: &str) -> u16 { g.place_names().iter().position(|p| p == name).unwrap() as u16 }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn row_at<'a>(rows: &'a [cs_decision::threats::ThreatRow], g: &KernelGame, place: &str) -> Option<&'a cs_decision::threats::ThreatRow> { let p = at(g, place); rows.iter().find(|r| r.unit.is_some_and(|u| u.at == p)) }
fn list_of(fen: &str, params: &ThreatParams) -> (KernelGame, ConcreteList) {
    let (g, st) = world();
    let s = state(fen);
    let l = concrete_threats(&st, &g, &s, &record(), params);
    (g, l)
}
fn no_time(l: &ConcreteList) -> String { format!("{:?} {:?} {:?} {:?} {:?}", l.mover, l.other, l.exchange, l.dropped, l.goal) }

#[test]
fn an_undefended_unit_of_the_other_side_is_a_confirmed_row_whose_line_starts_with_the_taking() {
    let (g, st) = world();
    let params = ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() };
    let l = concrete_threats(&st, &g, &state("k7/8/8/3p4/8/8/8/K2R4 w - - 0 1"), &record(), &params);
    let r = row_at(&l.other, &g, "d5").expect("the pawn the rook takes for nothing is a row against the other side");
    assert_eq!(r.verdict, VerdictKind::Confirmed);
    assert_eq!(r.line[0], "d1d5");
    assert_eq!(r.net, Some(475));
    assert!(!r.two_moves && r.lead.is_none());
    assert!(r.responses.is_empty() || r.responses.iter().all(|x| x.state == RespState::Kept), "only kept responses stay on a row");
}

#[test]
fn a_taking_that_loses_the_larger_unit_for_the_smaller_is_dropped_as_refuted_with_its_code() {
    let (g, st) = world();
    let params = ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() };
    let l = concrete_threats(&st, &g, &state("k7/8/2p5/3p4/8/8/8/K2R4 w - - 0 1"), &record(), &params);
    assert!(row_at(&l.other, &g, "d5").is_none(), "a refuted claim is not in the list");
    let d = row_at(&l.dropped, &g, "d5").expect("kept for the proposer");
    assert_eq!(d.why, Some("refuted"));
    assert!(d.net.unwrap() < 0);
    assert!(l.dropped_counts.keys().any(|k| k.contains("refuted")), "the count says why");
}

#[test]
fn like_for_like_is_an_exchange_apart_from_the_threats() {
    let (g, st) = world();
    let params = ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() };
    // the knight takes the knight, which is taken back: units of the same kind cancel exactly
    let l = concrete_threats(&st, &g, &state("k7/8/2p5/3n4/8/4N3/8/K7 w - - 0 1"), &record(), &params);
    let r = row_at(&l.exchange, &g, "d5").expect("an exchange is listed apart");
    assert_eq!(r.verdict, VerdictKind::Exchange);
    assert!(row_at(&l.other, &g, "d5").is_none());
    assert_eq!(r.outcome.as_ref().unwrap().cancelled, 1);
}

#[test]
fn a_threat_against_the_mover_is_read_after_the_pass_and_its_kept_responses_save_the_unit() {
    let (g, st) = world();
    let params = ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() };
    // white to move; the bishop on b2 takes the undefended rook on a1 if white passes
    let l = concrete_threats(&st, &g, &state("4k3/8/8/8/8/8/1b6/R3K3 w - - 0 1"), &record(), &params);
    let r = row_at(&l.mover, &g, "a1").expect("the rook is threatened: a row against the mover");
    assert_eq!(r.verdict, VerdictKind::Confirmed);
    assert_eq!(r.line[0], "b2a1");
    assert!(r.resp_read >= 1 && !r.responses.is_empty(), "the sweep reads every legal move of the owner and keeps the answers");
    let names: Vec<&str> = r.responses.iter().map(|x| x.names[0].as_str()).collect();
    assert!(names.contains(&"a1a2") || names.contains(&"a1b1"), "the rook leaving with a threat is kept: {names:?}");
    assert!(!names.contains(&"e1d2"), "a move that leaves the rook where it is is not kept: {names:?}");
    assert!(r.responses.iter().all(|x| x.state == RespState::Kept && x.source == Source::Sweep), "no frame read was given: every response is the sweep's");
    assert!(r.resp_states.contains_key("STILL WON"), "the moves that leave the unit won are counted");
}

#[test]
fn a_unit_nobody_bears_on_gets_a_row_from_every_unit_with_the_quiet_move_that_makes_a_taking_stand() {
    let (g, st) = world();
    let params = ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() };
    // the black bishop stands where no white unit bears on it; the rook can go to the file of the bishop: a threat in two moves
    let l = concrete_threats(&st, &g, &state("7k/8/8/8/8/5b2/8/R3K3 w - - 0 1"), &record(), &params);
    let r = row_at(&l.other, &g, "f3").or_else(|| row_at(&l.dropped, &g, "f3")).expect("every unit of the other side has a row somewhere");
    assert!(r.every, "the row came from every_unit (the list had none)");
    assert!(r.two_moves, "no taking is legal, so a quiet move that makes one stand is the line");
    assert!(r.lead.is_some() && !r.deferred, "the quiet move that makes it is named");
    let in_list = row_at(&l.other, &g, "f3").is_some();
    assert_eq!(in_list, matches!(r.verdict, VerdictKind::Confirmed | VerdictKind::Unread), "a quiet move that does not end ahead leaves the list and is kept as a dropped row for the proposer");
    if !in_list { assert_eq!(r.why, Some("two_move_not_ahead")); }
    assert!(row_at(&l.other, &g, "h8").is_none() && row_at(&l.dropped, &g, "h8").is_none(), "the goal unit gets no row (it cannot be taken)");
}

#[test]
fn every_unit_adds_only_the_units_the_list_has_no_row_for() {
    let (g, st) = world();
    let s = state("k7/8/7n/3p4/8/8/8/K2R4 w - - 0 1");
    let params = ThreatParams { goal: Some(kind(&g, "king")), every_sides: Some(vec![]), ..Default::default() };
    let mut l = concrete_threats(&st, &g, &s, &record(), &params);
    assert!(row_at(&l.other, &g, "d5").is_some() && row_at(&l.other, &g, "h6").is_none() && row_at(&l.dropped, &g, "h6").is_none(), "no side asked: the unit nothing bears on has no row");
    let before = l.other.len() + l.exchange.len() + l.dropped.len();
    let n = every_unit(&st, &g, &s, &mut l, &[0], &record(), &params);
    let white = g.roles().iter().position(|r| r == "white").unwrap();
    assert_eq!(n, 1, "only the knight was without a row (the pawn has one, the goal unit is not read)");
    assert_eq!(l.other.len() + l.exchange.len() + l.dropped.len(), before + 1);
    assert!(l.other.iter().chain(&l.dropped).chain(&l.exchange).filter(|r| r.every).all(|r| r.owner != white));
    // asked again: nothing is added twice
    assert_eq!(every_unit(&st, &g, &s, &mut l, &[0], &record(), &params), 0);
}

#[test]
fn the_end_of_the_game_is_a_row_of_the_goal_with_the_highest_stake_and_comes_first() {
    let (g, st) = world();
    let params = ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() };
    let l = concrete_threats(&st, &g, &state("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"), &record(), &params);
    assert!(!l.goal.is_empty(), "the end of the game is forced by the side to move");
    let gr = &l.goal[0];
    assert!(gr.goal && gr.unit.is_none() && gr.stake == Some(f64::INFINITY));
    assert_eq!(gr.verdict, VerdictKind::Confirmed);
    assert_eq!(gr.line[0], "a1a8");
    let black = g.roles().iter().position(|r| r == "black").unwrap();
    assert_eq!(gr.owner, black);
    assert!(l.other[0].goal, "the goal row is first in its side's list");
}

#[test]
fn a_side_forced_against_has_its_answers_listed_on_the_goal_row() {
    let (g, st) = world();
    let params = ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() };
    // black to move; white would end the game on the back rank if black passed: black's answers make room or move the goal unit
    let l = concrete_threats(&st, &g, &state("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1"), &record(), &params);
    let black = g.roles().iter().position(|x| x == "black").unwrap();
    let gr = l.goal.iter().find(|r| r.owner == black).expect("the end of the game is forced against the side to move after its pass");
    assert_eq!(gr.line[0], "a1a8");
    let names: Vec<&str> = gr.responses.iter().map(|x| x.names[0].as_str()).collect();
    assert!(names.contains(&"h7h6"), "making room answers it: {names:?}");
    assert!(gr.responses.iter().all(|x| x.source == Source::ForcingRead && x.state == RespState::Kept));
    assert!(gr.resp_read > gr.responses.len(), "the moves after which it is still forced are read and not listed");
    assert!(l.mover.iter().any(|r| r.goal), "the goal row is also in the list of the side it is against");
}

#[test]
fn the_rows_come_in_the_order_of_stake_and_a_given_frame_row_is_read_as_the_frame_read_named_it() {
    let (g, st) = world();
    let rook = UnitRef { owner: g.roles().iter().position(|r| r == "white").unwrap(), kind: kind(&g, "rook"), at: at(&g, "a1") };
    let bish = UnitRef { owner: g.roles().iter().position(|r| r == "black").unwrap(), kind: kind(&g, "bishop"), at: at(&g, "b2") };
    let row = FrameRow { unit: rook, class: "threat".into(), by: vec![bish], response_ix: vec![(at(&g, "a1"), at(&g, "b1"), 1, true)], stake: Some(2.0) };
    let params = ThreatParams { goal: Some(kind(&g, "king")), rows: Some(vec![row]), every_sides: Some(vec![]), ..Default::default() };
    let l = concrete_threats(&st, &g, &state("4k3/8/8/8/8/8/1b6/R3K3 w - - 0 1"), &record(), &params);
    assert_eq!(l.mover.len(), 1, "the frame read's rows replace the graph's candidates");
    let r = &l.mover[0];
    assert_eq!((r.class.as_str(), r.stake), ("threat", Some(2.0)));
    assert!(r.responses.iter().any(|x| x.source == Source::FrameRead && x.names[0] == "a1b1" && x.how == "the unit moves"), "the named answer is read and kept, with how it was found");
    assert!(r.responses.iter().any(|x| x.source == Source::Sweep), "the sweep reads the other moves");
    let mut firsts: Vec<_> = r.responses.iter().map(|x| x.first).collect();
    let n = firsts.len(); firsts.sort(); firsts.dedup();
    assert_eq!(firsts.len(), n, "a move is listed once: the frame read's reading comes before the sweep's of the same move");
    // the stake orders two rows: the greater first
    let params = ThreatParams { goal: Some(kind(&g, "king")), every_sides: Some(vec![]), stake: Some(&|k: &str, own: bool| Some(if k == "pawn" { 1.0 } else { 3.0 } + if own { 0.0 } else { 0.5 })), ..Default::default() };
    let l = concrete_threats(&st, &g, &state("7k/8/8/3p4/8/8/1b6/R2RK3 w - - 0 1"), &record(), &params);
    let st_of: Vec<f64> = l.other.iter().filter_map(|r| r.stake).collect();
    assert!(st_of.windows(2).all(|w| w[0] >= w[1]), "descending by stake: {st_of:?}");
}

#[test]
fn the_reads_are_batched_by_round_and_the_list_is_deterministic() {
    let (g, st) = world();
    let params = ThreatParams { goal: Some(kind(&g, "king")), every_sides: Some(vec![]), ..Default::default() };
    let s = state("7k/8/8/3p4/8/8/1b6/R2RK3 w - - 0 1");
    let a = concrete_threats(&st, &g, &s, &record(), &params);
    let b = concrete_threats(&st, &g, &s, &record(), &params);
    assert_eq!(no_time(&a), no_time(&b), "the same position gives the same list");
    let names: Vec<&str> = a.cost.iter().map(|c| c.name).collect();
    assert_eq!(names.len(), 5, "four rounds, then every_unit: {names:?}");
    let rows = (a.mover.len() + a.other.len() + a.dropped.len() + a.exchange.len()) as u64;
    assert!(a.cost[0].queries <= rows + 2 && a.cost[0].queries >= 2, "round 1 is one query a row and one per side for the end of the game, not one per move");
    assert!(a.cost[2].queries <= rows, "the sweep is asked once for every place, however many rows share it");
}

#[test]
fn the_stake_of_a_kind_is_the_change_in_the_owners_chance_if_one_is_lost() {
    // a head that is linear in the kinds (ours less theirs): the chance of the side to move less the other's
    let w = [0.25, 0.5, 1.0];
    let head = move |k: &[f64]| k.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f64>();
    let t = stake_of_kinds(&head, &[1.0, 0.0, -2.0]);
    for (i, wi) in w.iter().enumerate() {
        assert!((t[i][1] - wi).abs() < 1e-12, "losing one of ours lowers our chance by its weight");
        assert!((t[i][0] - wi).abs() < 1e-12, "losing one of theirs raises it by the same");
    }
    let f = stake_closure(t, vec!["pawn".into(), "knight".into(), "bishop".into()]);
    assert_eq!(f("knight", true), Some(0.5));
    assert_eq!(f("queen", true), None, "a kind the table does not have has no stake");
    let (g, st) = world();
    let s = state("k7/8/8/3p4/8/8/8/K2R4 w - - 0 1");
    let k = kinds_ours_less_theirs(&st, &g, &s, &["pawn".to_string(), "rook".to_string(), "queen".to_string()]);
    assert_eq!(k, vec![-1.0, 1.0, 0.0]);
}

#[test]
fn the_wake_with_the_threat_list_on_still_lists_the_other_pieces_not_ported_and_the_old_player_is_unchanged() {
    assert!(Pieces::old_player().unported().is_empty());
    let only_threats = Pieces { threats: true, ..Pieces::old_player() };
    assert!(only_threats.unported().is_empty(), "the threat list is ported: it is no longer listed as missing");
    assert!(Pieces::all().unported().iter().all(|n| !n.contains("concrete")));
    assert!(!Pieces::all().unported().is_empty(), "the proposer, the pick, the heads are still to port and are listed");
}

#[test]
fn bench_the_time_of_the_whole_list_on_a_fixed_position_one_thread() {
    let (g, st) = world();
    // a constructed middle position, both sides developed
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let params = ThreatParams { goal: Some(kind(&g, "king")), ..Default::default() };
    let t0 = std::time::Instant::now();
    let l = concrete_threats(&st, &g, &s, &record(), &params);
    let total = t0.elapsed().as_secs_f64();
    let rows = l.mover.len() + l.other.len() + l.exchange.len() + l.dropped.len();
    let parts: Vec<String> = l.cost.iter().map(|c| format!("{} {:.3} s ({} queries, {} nodes)", c.name, c.seconds, c.queries, c.nodes)).collect();
    println!("OBS threat list 79.06: fixed constructed position, one thread, depth {} fuse {}: whole list {:.3} s; rows {} (mover {}, other {}, exchange {}, dropped {}); responses read {} dropped {}; by round: {}", cs_decision::threats::DEPTH, cs_decision::threats::FUSE, total, rows, l.mover.len(), l.other.len(), l.exchange.len(), l.dropped.len(), l.resp_read, l.resp_dropped, parts.join("; "));
    assert!(total > 0.0 && l.cost.len() == 5);
}
