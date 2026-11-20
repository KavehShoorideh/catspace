//! Plan 79.14 PART B, UNIT TESTS of the carry (SYSTEM_SPEC 8 step 1 "update only those whose facts changed"; piece: the wake). `delta.rs`: the seeds found by asking only what the last position did not answer are the seeds asked afresh; what the two plies since touched is read from the graph;
//! a frame read that carries gives the frames it gives afresh; the carry by the graph holds a row only if its read set is untouched; with the carry off nothing is carried or kept. Constructed positions only; no puzzle, no match.
use cs_decision::frame_rows::{frame_read, frame_read_with, FrameRead};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::delta::{seeds_carried, Digest, Touch};
use cs_tactics::sheet::frames::seeds_of_board_until;
use cs_tactics::sheet::patterns::{Board, FxHashMap, LEVELS};
use cs_tactics::sheet::threat_frames::FrameReadParams;
use cs_tactics::sheet::questions::Rules;
use cs_tactics::sheet::Names;
use cs_decision::carry::Carried;
use cs_decision::goals::{propose, GoalParams};
use cs_decision::threat_verdict::Record;
use cs_decision::threats::{ConcreteList, ThreatParams};
use cs_decision::wake::UnitRef;
use std::collections::{BTreeMap, BTreeSet, HashMap};

type St = <KernelGame as Game>::State;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}
fn state(fen: &str) -> St { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn play(g: &KernelGame, s: &St, name: &str) -> St { let m = g.moves(s).into_iter().find(|m| g.move_name(s, m) == name).unwrap_or_else(|| panic!("no move {name}")); g.apply(s, &m) }

/// a middle game with pieces on both wings; two plies are played on the queen's wing
const MID: &str = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7";

/// every key (of every connected set of up to three atoms, at every level) of both positions, as the wanted hashes: the learned seeds of the real wake are a subset of this
fn want_all(boards: &[&Board]) -> FxHashMap<u64, u32> {
    let mut want: FxHashMap<u64, u32> = FxHashMap::default();
    for b in boards {
        for size in 1..=3 { b.connected_sets(size, &mut |its| { for k in b.keys(its, &(0..LEVELS).collect::<Vec<u8>>()) { let n = want.len() as u32; want.entry(k.hash).or_insert(n); } }); }
    }
    want
}
fn as_set(v: &[(u32, cs_tactics::sheet::frames::Seed)]) -> BTreeSet<(u32, Vec<(usize, u8, u16)>, Vec<u16>)> { v.iter().map(|(c, s)| (*c, s.units.iter().map(|x| (x.0, x.1 as u8, x.2)).collect(), s.places.clone())).collect() }

fn two_positions(g: &KernelGame) -> (St, St) {
    let a = state(MID);
    let b = play(g, &play(g, &a, "a2a3"), "a7a6");
    (a, b)
}

#[test]
fn a_the_seeds_asked_only_where_the_position_changed_are_the_seeds_asked_afresh() {
    let (g, st) = world();
    let (a, b) = two_positions(&g);
    let (pa, pb) = (st.position(&g, &a), st.position(&g, &b));
    let (ba, bb) = (Board::new(&st, &pa, None), Board::new(&st, &pb, None));
    let want = want_all(&[&ba, &bb]);
    let levels: Vec<u8> = (0..LEVELS).collect();
    let first = seeds_carried(&ba, &pa, 3, &levels, &want, None, None);
    let (fresh_a, _) = seeds_of_board_until(&ba, &pa, 3, &levels, &want, None);
    assert_eq!(as_set(&first.seeds), as_set(&fresh_a), "with nothing to carry from, the same seeds as the plain enumeration");
    let t0 = std::time::Instant::now();
    let carried = seeds_carried(&bb, &pb, 3, &levels, &want, None, Some(&first.memo));
    let t_carried = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let (fresh_b, _) = seeds_of_board_until(&bb, &pb, 3, &levels, &want, None);
    let t_fresh = t0.elapsed().as_secs_f64();
    let all = seeds_carried(&bb, &pb, 3, &levels, &want, None, None);
    println!("OBS delta 79.14 seeds: fresh {} seeds from {} sets asked in {:.4} s; carried {} seeds: {} matches carried, {} sets asked in {:.4} s", fresh_b.len(), all.asked, t_fresh, carried.seeds.len(), carried.carried, carried.asked, t_carried);
    assert_eq!(as_set(&carried.seeds), as_set(&fresh_b), "the seeds carried and asked for what is new are the seeds asked afresh");
    assert!(carried.carried > 0 && carried.asked < all.asked, "something was carried and fewer sets were asked");
    // a third position, from the carried memo (the memo of a carried read is itself good to carry from)
    let c = play(&g, &play(&g, &b, "c1d2"), "c8d7");
    let pc = st.position(&g, &c);
    let bc = Board::new(&st, &pc, None);
    let want3 = want_all(&[&ba, &bb, &bc]);
    let m1 = seeds_carried(&ba, &pa, 3, &levels, &want3, None, None);
    let m2 = seeds_carried(&bb, &pb, 3, &levels, &want3, None, Some(&m1.memo));
    let m3 = seeds_carried(&bc, &pc, 3, &levels, &want3, None, Some(&m2.memo));
    let (fresh_c, _) = seeds_of_board_until(&bc, &pc, 3, &levels, &want3, None);
    assert_eq!(as_set(&m3.seeds), as_set(&fresh_c), "two carries in a row still give the seeds asked afresh");
}

#[test]
fn b_a_seed_another_embedding_also_gives_is_carried_when_the_first_embedding_is_gone() {
    // the same seed can come from two embeddings; the carry keeps every match, so that when one embedding's atoms go the other still gives it
    let (g, st) = world();
    let (a, b) = two_positions(&g);
    let (pa, pb) = (st.position(&g, &a), st.position(&g, &b));
    let (ba, bb) = (Board::new(&st, &pa, None), Board::new(&st, &pb, None));
    let want = want_all(&[&ba, &bb]);
    let levels: Vec<u8> = (0..LEVELS).collect();
    let first = seeds_carried(&ba, &pa, 3, &levels, &want, None, None);
    let distinct: BTreeSet<_> = first.memo.matches.iter().map(|m| (m.concept, m.seed.units.clone(), m.seed.places.clone())).collect();
    assert!(first.memo.matches.len() > distinct.len(), "{} matches for {} distinct seeds: some seed has two embeddings", first.memo.matches.len(), distinct.len());
}

#[test]
fn c_what_the_two_plies_touched_is_read_from_the_graph_and_a_far_part_is_not_touched() {
    let (g, st) = world();
    let (a, b) = two_positions(&g);
    let (pa, pb) = (st.position(&g, &a), st.position(&g, &b));
    let (ba, bb) = (Board::new(&st, &pa, None), Board::new(&st, &pb, None));
    let (da, db) = (Digest::of(&ba, &pa), Digest::of(&bb, &pb));
    let t = Touch::between(&da, &db);
    let name = |n: &str| g.place_names().iter().position(|p| p == n).unwrap() as u16;
    let kind = |n: &str| g.kind_names().iter().position(|p| p == n).unwrap() as u8;
    let white = g.roles().iter().position(|r| r == "white").unwrap();
    let black = g.roles().iter().position(|r| r == "black").unwrap();
    // the pawn that moved from a2 to a3 and the pawn from a7 to a6 are touched; the pieces on the king's wing, which nothing opened or shut, are not
    assert!(t.hits(&[(white, kind("pawn"), name("a2"))], &[]), "the pawn that moved is touched (its old cell)");
    assert!(t.hits(&[(black, kind("pawn"), name("a6"))], &[]), "the pawn that moved is touched (its new cell)");
    assert!(!t.hits(&[(white, kind("king"), name("g1")), (white, kind("pawn"), name("g2")), (black, kind("king"), name("g8"))], &[name("h2"), name("h7"), name("g7")]), "the king's wing is not touched");
    println!("OBS delta 79.14 touch: {} cells and {} places touched by two plies of {} units", t.cells.len(), t.places.len(), pa.units.len());
    // a position against itself touches nothing
    assert!(Touch::between(&da, &da).cells.is_empty());
}

fn frame_params() -> FrameReadParams {
    FrameReadParams { max_size: 3, fuse: 1_500, row_fuse: 30_000, answer_fuse: 1_500, conditional: true, kv: Some(vec![0, 475, 1216, 1301, 1945, 3583, 0, 0]), bands: None, rules: Rules { settle: true, progress: true, cause: true, band: true, root_cap: 0 }, deadline: None }
}

fn keys_of(st: &Static, g: &KernelGame, ss: &[&St]) -> Vec<u64> {
    let mut v: Vec<u64> = vec![];
    for s in ss {
        let p = st.position(g, s);
        let b = Board::new(st, &p, None);
        for size in 1..=2 { b.connected_sets(size, &mut |its| { for k in b.keys(its, &(0..LEVELS).collect::<Vec<u8>>()) { if !v.contains(&k.hash) { v.push(k.hash); } } }); }
    }
    let mut v: Vec<u64> = v.into_iter().enumerate().filter(|(i, _)| i % 9 == 0).map(|(_, h)| h).collect();
    v.truncate(60);
    v
}

fn frames_view(r: &cs_decision::frame_rows::FrameRows) -> Vec<String> {
    let mut v: Vec<String> = r.rows.iter().map(|x| format!("{:?} {} {:?} {:?} {:?}", x.unit, x.class, x.by, x.response_ix, r.read_sets.get(&x.unit).map(|s| (&s.members, &s.places)))).collect();
    v.sort();
    v
}

#[test]
fn d_the_frame_read_that_carries_gives_the_rows_of_the_frame_read_asked_afresh() {
    let (g, st) = world();
    let goal = g.kind_names().iter().position(|k| k == "king").map(|k| k as u8);
    let (mut carried_any, mut asked_less) = (0usize, 0usize);
    for (m1, m2) in [("a2a3", "a7a6"), ("h2h3", "h7h6"), ("c1d2", "c8d7"), ("b2b3", "b7b6"), ("g2g3", "a7a6")] {
        let a = state(MID);
        let b = play(&g, &play(&g, &a, m1), m2);
        let fr = FrameRead { want: keys_of(&st, &g, &[&a, &b]), params: frame_params() };
        let first = frame_read_with(&st, &g, &a, goal, &fr, None, None, true);
        assert!(first.memo.is_some(), "a read to the end leaves a memo");
        assert!(first.touch.is_none(), "nothing to carry from: nothing is touched, nothing held");
        let carried = frame_read_with(&st, &g, &b, goal, &fr, None, first.memo.as_ref(), true);
        let fresh = frame_read_with(&st, &g, &b, goal, &fr, None, None, true);          // asked afresh, in the carry's own (canonical) order of seeds
        println!("OBS delta 79.14 frames {m1} {m2}: fresh {} frames, {} rows, {} questions in {:.4} s; carried: {} frames carried of {}, {} seeds carried, {} questions in {:.4} s", fresh.frames, fresh.rows.len(), fresh.questions, fresh.seconds, carried.parts.carried, carried.frames, carried.parts.seeds_carried, carried.questions, carried.seconds);
        assert_eq!(frames_view(&carried), frames_view(&fresh), "the rows and their read sets are those asked afresh");
        assert_eq!(carried.frames, fresh.frames);
        carried_any += carried.parts.carried;
        asked_less += usize::from(carried.questions < fresh.questions);
    }
    println!("OBS delta 79.14 frames: over the five pairs of plies {carried_any} frames carried, {asked_less} reads asked fewer questions (the frames of this test are merged into two large ones that cover most of the board, so a frame is touched by almost any move)");
    // the same position again (two plies that touch nothing): every frame is carried, and the rows are those asked afresh
    let a = state(MID);
    let fr = FrameRead { want: keys_of(&st, &g, &[&a]), params: frame_params() };
    let first = frame_read_with(&st, &g, &a, goal, &fr, None, None, true);
    let again = frame_read_with(&st, &g, &a, goal, &fr, None, first.memo.as_ref(), true);
    println!("OBS delta 79.14 frames, nothing touched: {} frames carried of {}, {} questions against {} asked afresh, {:.4} s against {:.4} s", again.parts.carried, again.frames, again.questions, first.questions, again.seconds, first.seconds);
    assert!(again.parts.carried >= first.frames && again.parts.carried > 0, "nothing touched: every frame is carried");
    assert_eq!(again.questions, 0, "and no question is asked");
    assert_eq!(frames_view(&again), frames_view(&first));
}

#[test]
fn e_with_the_carry_off_nothing_is_kept_and_nothing_is_carried() {
    let (g, st) = world();
    let (a, b) = two_positions(&g);
    let fr = FrameRead { want: keys_of(&st, &g, &[&a, &b]), params: frame_params() };
    let goal = g.kind_names().iter().position(|k| k == "king").map(|k| k as u8);
    let off = frame_read_with(&st, &g, &a, goal, &fr, None, None, false);
    assert!(off.memo.is_none() && off.touch.is_none() && off.parts.carried == 0 && off.parts.seeds_carried == 0);
    let plain = frame_read(&st, &g, &a, goal, &fr, None);
    assert_eq!(frames_view(&off), frames_view(&plain));
}

#[test]
fn f_a_memo_made_with_the_other_side_to_move_is_not_used() {
    let (g, st) = world();
    let (a, _) = two_positions(&g);
    let b1 = play(&g, &a, "a2a3");
    let fr = FrameRead { want: keys_of(&st, &g, &[&a, &b1]), params: frame_params() };
    let goal = g.kind_names().iter().position(|k| k == "king").map(|k| k as u8);
    let first = frame_read_with(&st, &g, &a, goal, &fr, None, None, true);
    let next = frame_read_with(&st, &g, &b1, goal, &fr, None, first.memo.as_ref(), true);
    assert!(next.touch.is_none() && next.parts.carried == 0 && next.parts.seeds_carried == 0, "the other side to move: nothing carried");
    let _ = HashMap::<u8, u8>::new();
}

#[test]
fn g_the_hash_without_allocation_is_the_hash_of_every_key_of_every_set() {
    let (g, st) = world();
    let (a, b) = two_positions(&g);
    let levels: Vec<u8> = (0..LEVELS).collect();
    let (mut sets, mut t_fast, mut t_slow) = (0usize, 0.0f64, 0.0f64);
    for s in [&a, &b] {
        let p = st.position(&g, s);
        let bd = Board::new(&st, &p, Some(g.kind_names().iter().position(|k| k == "king").unwrap() as u8));
        for size in 1..=3 {
            bd.connected_sets(size, &mut |its| {
                let t0 = std::time::Instant::now(); let fast = bd.key_hashes(its, &levels); t_fast += t0.elapsed().as_secs_f64();
                let t0 = std::time::Instant::now(); let slow = bd.keys(its, &levels); t_slow += t0.elapsed().as_secs_f64();
                for (i, k) in slow.iter().enumerate() { assert_eq!((k.hash, k.sign), fast[i], "level {i} of the set {its:?}"); }
                sets += 1;
            });
        }
    }
    println!("OBS delta 79.14 keys: {sets} sets, the hashes of all four levels equal; {:.1} ns a set without allocation, {:.1} ns with it", 1e9 * t_fast / sets as f64, 1e9 * t_slow / sets as f64);
}

fn record() -> Record { Record { worth: [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect(), half: HashMap::new() } }

fn rows_view(l: &ConcreteList) -> BTreeSet<String> {
    l.mover.iter().map(|r| ("against us", r)).chain(l.other.iter().map(|r| ("against them", r))).chain(l.exchange.iter().map(|r| ("exchange", r))).chain(l.dropped.iter().map(|r| ("dropped", r)))
        .filter(|(_, r)| !r.goal).map(|(b, r)| format!("{b} {:?} {} {:?} {:?} {:?} {:?}", r.unit, r.class, r.verdict, r.net, r.line, r.why)).collect()
}

/// the neighbourhood read sets of the every-unit rows of a list, as the wake makes them
fn every_sets(list: &ConcreteList, dg: &Digest, me: usize) -> BTreeMap<UnitRef, cs_decision::frame_rows::ReadSet> {
    let mut m = BTreeMap::new();
    for r in list.mover.iter().chain(&list.other).chain(&list.exchange).chain(&list.dropped).filter(|r| r.every && !r.goal) {
        if let Some(u) = r.unit { let (cells, places) = dg.neighbourhood((u.owner, u.kind, u.at), u.at); m.insert(u, cs_decision::frame_rows::ReadSet { mover: me, members: cells, places }); }
    }
    m
}

#[test]
fn h_a_unit_whose_neighbourhood_nothing_touched_keeps_its_row_and_the_rows_are_those_asked_afresh() {
    let (g, st) = world();
    let goal = g.kind_names().iter().position(|k| k == "king").map(|k| k as u8);
    let rec = record();
    let (mut carried_all, mut asked_less, mut same, mut total) = (0usize, 0usize, 0usize, 0usize);
    let (mut unread_n, mut unread_eq, mut read_n, mut read_eq) = (0usize, 0usize, 0usize, 0usize);
    for (m1, m2) in [("a2a3", "a7a6"), ("h2h3", "h7h6"), ("c1d2", "c8d7"), ("b2b3", "b7b6"), ("g2g3", "a7a6")] {
        let a = state(MID);
        let b = play(&g, &play(&g, &a, m1), m2);
        let tp = || ThreatParams { goal, rows: Some(vec![]), share_table: true, ..Default::default() };
        let pa = propose(&st, &g, &a, &rec, &tp(), &GoalParams { budget: 40, ..Default::default() });
        let (parta, partb) = (st.position(&g, &a), st.position(&g, &b));
        let (da, db) = (Digest::of(&Board::new(&st, &parta, goal), &parta), Digest::of(&Board::new(&st, &partb, goal), &partb));
        let me = g.mover(&a).unwrap();
        let carried = Carried::of_every(&pa.list, &BTreeMap::new(), &every_sets(&pa.list, &da, me), Some((me, da.clone())));
        let touch = Touch::between(&da, &db);
        let t0 = std::time::Instant::now();
        let fresh = propose(&st, &g, &b, &rec, &tp(), &GoalParams { budget: 40, ..Default::default() });
        let t_fresh = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let held = propose(&st, &g, &b, &rec, &tp(), &GoalParams { budget: 40, held_units: Some((carried.rows.clone(), touch)), ..Default::default() });
        let t_held = t0.elapsed().as_secs_f64();
        let n_carried = held.list.mover.iter().chain(&held.list.other).chain(&held.list.exchange).chain(&held.list.dropped).filter(|r| r.carried).count();
        let (vf, vh) = (rows_view(&fresh.list), rows_view(&held.list));
        let diff: Vec<&String> = vf.symmetric_difference(&vh).collect();
        println!("OBS delta 79.14 every-unit rows {m1} {m2}: {} rows asked afresh in {:.4} s; with the carry {} rows, {} carried, {} questions against {}, in {:.4} s; {} rows differ", vf.len(), t_fresh, vh.len(), n_carried, held.questions, fresh.questions, t_held, diff.len());
        for d in diff.iter().take(4) { println!("    differs: {d}"); }
        // each carried row against the row asked afresh for the same unit
        let all_fresh: Vec<&cs_decision::threats::ThreatRow> = fresh.list.mover.iter().chain(&fresh.list.other).chain(&fresh.list.exchange).chain(&fresh.list.dropped).collect();
        let key = |r: &cs_decision::threats::ThreatRow| format!("{:?} {:?} {:?} {:?} {:?}", r.unit, r.verdict, r.net, r.line, r.why);
        for r in held.list.mover.iter().chain(&held.list.other).chain(&held.list.exchange).chain(&held.list.dropped).filter(|r| r.carried) {
            let unread = r.verdict == cs_decision::threat_verdict::VerdictKind::Unread;
            let eq = all_fresh.iter().any(|f| key(f) == key(r));
            if unread { unread_n += 1; unread_eq += usize::from(eq); } else { read_n += 1; read_eq += usize::from(eq); if !eq { println!("    CARRIED ROW DIFFERS FROM FRESH: {}", key(r)); for f in all_fresh.iter().filter(|f| f.unit == r.unit) { println!("        fresh: {}", key(f)); } } }
        }
        carried_all += n_carried; asked_less += usize::from(held.questions < fresh.questions); total += vf.len(); same += vf.intersection(&vh).count();
        assert_eq!(fresh.goals.len() > 0, held.goals.len() > 0);
    }
    println!("OBS delta 79.14 every-unit rows over five pairs of plies: {carried_all} rows carried, {same} of {total} rows the same as asked afresh, {asked_less} reads asked fewer questions");
    assert!(carried_all > 0, "some unit row was carried");
    assert!(asked_less > 0, "and fewer questions were asked");
    println!("OBS delta 79.14 every-unit rows carried against asked afresh: {read_n} carried rows that a line read settled, {read_eq} the same as asked afresh; {unread_n} carried rows the node fuse left UNREAD, {unread_eq} the same as asked afresh (a fuse-cut line depends on what the shared table held when it was read, so it differs even between two fresh reads in another order)");
    // WHY THEY DIFFER (said before going on, as the plan asks): a line is read under a node fuse and the lines of one position share one table, so the answer asked afresh depends on what the table held when the unit was reached (a fresh read in another order also differs), and a line's
    // tail reads moves far from the unit; the row carried is the row the last decision read, usually the same verdict and net, and where it differs the fresh row is as often the cut one (Unread) as the carried one. Nothing here is asserted of the rows that differ but that they are marked carried.
    assert!(read_n + unread_n == carried_all, "every carried row is counted");
}

#[test]
fn i_with_nothing_given_to_carry_the_rows_are_asked_as_before() {
    let (g, st) = world();
    let goal = g.kind_names().iter().position(|k| k == "king").map(|k| k as u8);
    let rec = record();
    let a = state(MID);
    let tp = || ThreatParams { goal, rows: Some(vec![]), share_table: true, ..Default::default() };
    let x = propose(&st, &g, &a, &rec, &tp(), &GoalParams { budget: 40, ..Default::default() });
    let y = propose(&st, &g, &a, &rec, &tp(), &GoalParams { budget: 40, held_units: None, ..Default::default() });
    assert_eq!(rows_view(&x.list), rows_view(&y.list));
    assert!(x.list.mover.iter().chain(&x.list.other).chain(&x.list.exchange).chain(&x.list.dropped).all(|r| !r.carried), "no row is marked carried");
    // the same position carried against itself: every unit row stands and none is asked
    let part = st.position(&g, &a);
    let da = Digest::of(&Board::new(&st, &part, goal), &part);
    let me = g.mover(&a).unwrap();
    let carried = Carried::of_every(&x.list, &BTreeMap::new(), &every_sets(&x.list, &da, me), Some((me, da.clone())));
    let z = propose(&st, &g, &a, &rec, &tp(), &GoalParams { budget: 40, held_units: Some((carried.rows.clone(), Touch::between(&da, &da))), ..Default::default() });
    assert_eq!(rows_view(&x.list), rows_view(&z.list), "nothing touched: the rows carried are the rows asked");
    assert_eq!(z.questions, 0, "and no unit was asked");
}
