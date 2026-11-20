//! Plan 79.08 PART B (the remaining pieces of the wake, SYSTEM_SPEC 8 steps 1, 2 and 6, 7 "Calculation and estimate"): UNIT TESTS of each piece on small constructed positions or on hand-built inputs, its own contract. No puzzle, no stored game, no match.
//! B1 the frame read as the source of the rows (frame_rows), B2 the carry (carry), B3 the chance of a goal (estimate) and the reasons' pick (pick). The heads over whole-position counts are not ported and not called.
use cs_decision::carry::{carry, Bucket, Carried, CarriedRow};
use cs_decision::estimate::ChanceTable;
use cs_decision::frame_rows::{frame_read, rows_of, FrameRead, FrameRows, ReadSet};
use cs_decision::goals::{goals, GoalParams};
use cs_decision::pick::{pick, scores, Dir, MoveReasons, Reason, Reasons, Side};
use cs_decision::threat_verdict::{Record, VerdictKind};
use cs_decision::threats::{ConcreteList, FrameRow, ThreatRow};
use cs_decision::wake::UnitRef;
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::patterns::Board;
use cs_tactics::sheet::questions::Rules;
use cs_tactics::sheet::threat_frames::{FrameReadParams, ThreatClaim, ThreatFrame, ThreatFrames};
use cs_tactics::sheet::Names;
use std::collections::{BTreeMap, HashMap};

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let mut st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    // a profile of the kinds is learned from a few positions (the frame read needs one; the binding learns it from a sample of games)
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 2 3", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "6k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1"] {
        let s = KernelGame::from_fen("chess", fen).unwrap().initial();
        let p = st.position(&g, &s);
        st.learn(&p);
    }
    (g, st)
}
fn record() -> Record { Record { worth: [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect(), half: HashMap::new() } }
fn at(g: &KernelGame, name: &str) -> u16 { g.place_names().iter().position(|p| p == name).unwrap() as u16 }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn role(g: &KernelGame, name: &str) -> usize { g.roles().iter().position(|r| r == name).unwrap() }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn unit(g: &KernelGame, r: &str, k: &str, p: &str) -> UnitRef { UnitRef { owner: role(g, r), kind: kind(g, k), at: at(g, p) } }

/// every pattern key (a seed of one or two units) found on the board of `s`: the stand-in for the learned seeds
fn keys_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Vec<u64> {
    let part = st.position(g, s);
    let b = Board::new(st, &part, Some(kind(g, "king")));
    let mut keys: Vec<u64> = vec![];
    for k in 1..=2 { b.connected_sets(k, &mut |its| { for key in b.keys(its, &[0, 1, 2, 3]) { if !keys.contains(&key.hash) { keys.push(key.hash); } } }); }
    keys
}

fn frame_params(g: &KernelGame) -> FrameReadParams {
    let worth: HashMap<&str, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].into_iter().collect();
    let kv: Vec<i64> = g.kind_names().iter().map(|k| worth.get(k.as_str()).copied().unwrap_or(10_000)).collect();
    FrameReadParams { max_size: 3, fuse: 1_500, row_fuse: 30_000, answer_fuse: 1_500, conditional: true, kv: Some(kv), bands: None, rules: Rules { settle: true, progress: true, cause: true, band: true, root_cap: 0 } }
}

// ---------------------------------------------------------------- B1 the frame read

#[test]
fn b1_the_rows_of_a_hand_built_frame_read_follow_the_threat_list_rule() {
    let (g, _st) = world();
    let (me, other) = (role(&g, "white"), role(&g, "black"));
    let (n, b, p) = (kind(&g, "knight"), kind(&g, "bishop"), kind(&g, "pawn"));
    let (c4, e2, a7) = (at(&g, "c4"), at(&g, "e2"), at(&g, "a7"));
    // two frames about the knight on c4 (the other side's unit): a small one that reads it as a threat (class 4), a wider one that also has an answer; and a frame about the pawn on a7 that reads it safe (class 1): no row
    let small = ThreatFrame { id: "f1".into(), members: vec![(false, n, c4), (true, b, e2)], places: vec![c4, e2], unread: 0, claims: vec![ThreatClaim { member: 0, class: 4, basis: 255, take_basis: 255, bearers: vec![(b, e2)], answers: vec![], tried: 0, unread: 0 }] };
    let wide = ThreatFrame { id: "f2".into(), members: vec![(false, n, c4), (true, b, e2), (false, p, a7)], places: vec![c4, e2, a7], unread: 0,
        claims: vec![ThreatClaim { member: 0, class: 4, basis: 255, take_basis: 255, bearers: vec![(b, e2)], answers: vec![(c4, a7, 1, true)], tried: 3, unread: 0 }, ThreatClaim { member: 2, class: 1, basis: 0, take_basis: 255, bearers: vec![], answers: vec![], tried: 0, unread: 0 }] };
    let raw = ThreatFrames { frames: vec![small, wide], questions: 7 };
    let out = rows_of(&raw, me, other, g.kind_names().len(), None, &g.kind_names());
    assert_eq!(out.rows.len(), 1, "a unit read safe in every frame makes no row");
    let r = &out.rows[0];
    assert_eq!(r.unit, unit(&g, "black", "knight", "c4"));
    assert_eq!(r.class, "threat");
    assert_eq!(r.by, vec![unit(&g, "white", "bishop", "e2")], "the bearers are of the other side");
    assert_eq!(r.response_ix, vec![(c4, a7, 1, true)], "the answer comes from the wider frame: a unit lost in the smallest frame and answered in a wider one is one row with a response");
    let rs = out.read_sets.get(&r.unit).unwrap();
    assert_eq!((rs.mover, rs.places.clone()), (me, vec![c4.min(e2), c4.max(e2)]), "the read set is that of the smallest frame that explains the row");
    assert_eq!(out.questions, 7);
}

#[test]
fn b1_the_frame_read_in_process_makes_rows_from_the_learned_seeds_and_gives_the_read_set() {
    let (g, st) = world();
    // the white bishop on c3 bears on the black queen on d5? no: it bears along the long diagonal; the rook on b2 backs it. The black queen bears on the bishop and on the king's neighbourhood.
    let s = state("6k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1");
    let fr = FrameRead { want: keys_of(&st, &g, &s), params: frame_params(&g) };
    assert!(!fr.want.is_empty());
    let out = frame_read(&st, &g, &s, Some(kind(&g, "king")), &fr, None);
    assert!(out.frames > 0 && out.questions > 0, "the learned seeds name units on this board and the read asks questions");
    assert!(!out.rows.is_empty(), "at least one unit is read as lost, threatened, conditional or contested");
    for r in &out.rows { assert!(out.read_sets.contains_key(&r.unit), "every row has the read set of the frame that explains it"); }
    // without seeds nothing is read
    let none = frame_read(&st, &g, &s, Some(kind(&g, "king")), &FrameRead { want: vec![], params: frame_params(&g) }, None);
    assert!(none.rows.is_empty() && none.frames == 0);
}

// ---------------------------------------------------------------- B2 the carry

fn a_row(g: &KernelGame, u: UnitRef) -> ThreatRow { let mut r = ThreatRow::new(u.owner, Some(u), "threat", vec![], None); r.verdict = VerdictKind::Confirmed; r.net = Some(100); let _ = g; r }
fn read_set(me: usize, members: Vec<(usize, Kind, u16)>, places: Vec<u16>) -> ReadSet { ReadSet { mover: me, members, places } }

#[test]
fn b2_a_claim_whose_read_set_did_not_change_is_held_and_the_others_are_read_again() {
    let (g, _st) = world();
    let me = role(&g, "white");
    let (u1, u2) = (unit(&g, "black", "knight", "c4"), unit(&g, "black", "pawn", "a7"));
    let (rs1, rs2, rs2_changed) = (read_set(me, vec![(0, 1, 10), (1, 2, 20)], vec![10, 20]), read_set(me, vec![(1, 0, 30)], vec![30]), read_set(me, vec![(1, 0, 30), (0, 3, 31)], vec![30, 31]));
    let mut list = ConcreteList::default();
    list.other.push(a_row(&g, u1));
    list.dropped.push(a_row(&g, u2));
    let mut sets = BTreeMap::new();
    sets.insert(u1, rs1.clone());
    sets.insert(u2, rs2.clone());
    let prev = Carried::of(&list, &sets);
    assert_eq!(prev.rows.len(), 2);
    assert!(prev.rows.iter().any(|r| r.bucket == Bucket::Dropped), "a dropped row is carried in its own list");
    // now: the unit u1 has the same read set, u2 has a changed one, a third unit is new
    let u3 = unit(&g, "white", "bishop", "e2");
    let mut now = FrameRows::default();
    for (u, rs) in [(u1, rs1), (u2, rs2_changed), (u3, read_set(me, vec![(0, 2, 5)], vec![5]))] {
        now.rows.push(FrameRow { unit: u, class: "threat".into(), by: vec![], response_ix: vec![], stake: None });
        now.read_sets.insert(u, rs);
    }
    let c = carry(&prev, &now);
    assert_eq!(c.held.iter().map(|h| h.row.unit.unwrap()).collect::<Vec<_>>(), vec![u1], "only the claim whose read set is the same is held");
    assert_eq!(c.read.iter().map(|r| r.unit).collect::<Vec<_>>(), vec![u2, u3], "the changed one and the new one are read again");
    // the held row goes back into the list it was in
    let mut fresh = ConcreteList::default();
    c.merge_into(&mut fresh);
    assert_eq!((fresh.other.len(), fresh.mover.len(), fresh.dropped.len()), (1, 0, 0));
    // nothing carried when there is no last decision
    let c0 = carry(&Carried::default(), &now);
    assert!(c0.held.is_empty() && c0.read.len() == 3);
    let _: Option<CarriedRow> = c.held.first().cloned();
}

// ---------------------------------------------------------------- B3 the estimates

#[test]
fn b3_the_chance_of_a_goal_is_read_from_the_table_by_its_key_then_its_facts_then_the_whole_and_is_smoothed() {
    let (g, st) = world();
    let s = state("k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1");
    let mut l = ConcreteList::default();
    let mut r = ThreatRow::new(role(&g, "black"), Some(unit(&g, "black", "knight", "e7")), "contested", vec![], None);
    r.verdict = VerdictKind::Exchange;
    l.other.push(r);
    let gs = goals(&st, &g, &s, &l, role(&g, "white"), Some(kind(&g, "king")), 1, 8);
    let gl = gs.iter().find(|x| x.target == unit(&g, "black", "knight", "e7")).unwrap();
    let mut t = ChanceTable { floor: 20, total: (100, 30), ..Default::default() };
    let e = t.chance(gl);
    assert_eq!((e.source, e.n), ("total", 100));
    assert!((e.chance - 31.0 / 102.0).abs() < 1e-12, "(hits + 1) / (n + 2) of the whole table");
    t.by_key.insert(gl.key.clone(), (19, 19));
    assert_eq!(t.chance(gl).source, "total", "a cell with fewer than the floor does not speak");
    t.by_key.insert(gl.key.clone(), (40, 10));
    let e = t.chance(gl);
    assert_eq!((e.source, e.n), ("key", 40));
    assert!((e.chance - 11.0 / 42.0).abs() < 1e-12);
    let _ = GoalParams::default();
}

#[test]
fn b3_the_reasons_score_is_the_sum_of_the_weights_of_the_reasons_a_move_carries_and_the_pick_is_the_top_with_the_search_move_first_on_a_tie() {
    let (g, st) = world();
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let keys = keys_of(&st, &g, &s);
    // every column, every event, a weight that depends on the column: the score of a move is then a function of its events only
    let mut reasons = vec![];
    for c in 0..keys.len() { for side in [Side::Mover, Side::Other] { for dir in [Dir::Up, Dir::Down] { reasons.push(Reason { column: c, side, dir, weight: 0.001 * (1 + c % 7) as f64 * if dir == Dir::Up { 1.0 } else { -1.0 } }); } } }
    let r = Reasons { keys, reasons, max_size: 2, levels: vec![0, 1, 2, 3], tie_digits: 9 };
    let moves: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    let sc = scores(&st, &g, &s, Some(kind(&g, "king")), &r, &moves);
    assert_eq!(sc.len(), moves.len());
    for m in &sc { assert!((m.score - m.carried.iter().map(|i| r.reasons[*i].weight).sum::<f64>()).abs() < 1e-12, "the score is the sum of the weights of the reasons carried"); }
    assert!(sc.iter().any(|m| !m.carried.is_empty()), "some move changes some pattern count");
    // the pick: the top of the tied moves; on equal scores the first of the tied
    let hand = vec![MoveReasons { mv: "x".into(), carried: vec![], score: 0.5 }, MoveReasons { mv: "y".into(), carried: vec![], score: 0.7 }, MoveReasons { mv: "z".into(), carried: vec![], score: 0.7 }];
    let tied = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    assert_eq!(pick(&hand, &tied, 9), Some("y".to_string()));
    let level = vec![MoveReasons { mv: "x".into(), carried: vec![], score: 0.5 }, MoveReasons { mv: "y".into(), carried: vec![], score: 0.5 }];
    assert_eq!(pick(&level, &["x".to_string(), "y".to_string()], 9), Some("x".to_string()), "the search's own first move when the scores agree");
    assert_eq!(pick(&level, &["x".to_string()], 9), None, "fewer than two tied: nothing to pick");
}
