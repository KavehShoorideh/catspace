//! PLAN 79.17 PART B, UNIT TESTS (SYSTEM_SPEC 7, the measure of progress; piece: the measure): the atoms read from the base graph on constructed positions, the term's text, and October's question (`lowering`) answered by the search of toward.rs.
//! The readings are also compared with October's Python in tests/test_plan79_17.py (through the binding). Constructed positions only.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_operator::Budget;
use cs_tactics::measure::{atom_values, groups_of, library, lower, lowering, question, value, Atom, Found, Geometry, Group, Reader, Term};
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashSet;

type St = <KernelGame as Game>::State;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}
fn state(fen: &str) -> St { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn kind(g: &KernelGame, name: &str) -> u8 { g.kind_names().iter().position(|k| k == name).unwrap() as u8 }
fn grp(theirs: bool, k: u8) -> Group { Group { theirs, kind: k } }
fn play(g: &KernelGame, s: &St, name: &str) -> St { let m = g.moves(s).into_iter().find(|m| g.move_name(s, m) == name).unwrap_or_else(|| panic!("no move {name}")); g.apply(s, &m) }

/// their leader in the corner, ours and the long-mover far: the distances are the board's, the region is what the long-mover leaves it
#[test]
fn the_atoms_on_a_constructed_position() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let kk = kind(&g, "king");
    let s = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let r = Reader::new(&g, &geo, &s, 0);
    let theirs_k = grp(true, kk);
    let ours_k = grp(false, kk);
    assert_eq!(r.atom(&Atom::Corner(theirs_k)), 0, "their leader stands on a place of fewest steps");
    assert_eq!(r.atom(&Atom::Edge(theirs_k)), 0);
    assert_eq!(r.atom(&Atom::Edge(ours_k)), 0);
    assert_eq!(r.atom(&Atom::Dist(ours_k, theirs_k)), 7, "seven steps along the long diagonal");
    // the long-mover bears on the second line and the first column: their leader walks over the six lines above it and the seven columns beside it
    assert_eq!(r.atom(&Atom::Region(theirs_k)), 42);
    assert_eq!(r.atom(&Atom::Turn), 0);
    let s2 = play(&g, &s, "a2a7");
    let r2 = Reader::new(&g, &geo, &s2, 0);
    assert_eq!(r2.atom(&Atom::Turn), 1);
    assert_eq!(r2.atom(&Atom::Region(theirs_k)), 48, "the long-mover stands loose beside it: their leader may take it, so the region leaks through its place");
    // the same with the long-mover held by our leader: the seventh line is shut to it, the last line is what is left, less the column the long-mover bears on
    let held = play(&g, &state("7k/8/1K6/8/8/8/8/R7 w - - 0 1"), "a1a7");
    assert_eq!(Reader::new(&g, &geo, &held, 0).atom(&Atom::Region(theirs_k)), 7);
}

#[test]
fn the_span_is_the_extent_of_the_region_along_each_line_family() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let kk = kind(&g, "king");
    let s = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let r = Reader::new(&g, &geo, &s, 0);
    let spans: Vec<i64> = (0..geo.family_count()).map(|f| r.atom(&Atom::Span(grp(true, kk), f))).collect();
    // the region is six lines by seven columns: the two families of straight lines read 6 and 7
    assert!(spans.contains(&6) && spans.contains(&7), "{spans:?}");
}

#[test]
fn a_term_has_a_text_that_parses_back_and_a_tuple_is_compared_in_order() {
    let a = Atom::parse("dist(0,5,1,5)").unwrap();
    assert_eq!(a.name(), "dist(0,5,1,5)");
    for text in ["turn", "[sum region(1,5) edge(1,5)]", "[product span(1,5,0) span(1,5,1)]", "[tuple region(1,5) corner(1,5) dist(0,5,1,5)]"] {
        let t = Term::parse(text).unwrap();
        assert_eq!(t.text(), text);
    }
    assert!(Term::parse("[sum [tuple turn turn] turn]").is_err(), "a tuple is not a scalar");
    assert!(Term::parse("[tuple]").is_err());
    assert!(Term::parse("region(1)").is_err());
    assert!(lower(&[3, 9], &[4, 0]) && lower(&[3, 1], &[3, 2]) && !lower(&[3, 2], &[3, 2]) && !lower(&[5, 0], &[4, 9]));
}

#[test]
fn the_library_is_the_same_code_for_every_class_and_names_no_unit() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let (kk, rk) = (kind(&g, "king"), kind(&g, "rook"));
    let one = groups_of(&g, &state("7k/8/8/8/8/8/R7/K7 w - - 0 1"), 0);
    let two = groups_of(&g, &state("7k/8/8/8/8/R7/R7/K7 w - - 0 1"), 0);
    assert_eq!(one, two, "two long-movers are the same group");
    assert_eq!(one.len(), 3);
    let lib = library(&one, &geo);
    let names: HashSet<String> = lib.iter().map(|a| a.name()).collect();
    assert_eq!(names.len(), lib.len(), "every atom has its own name");
    for a in &lib { assert_eq!(Atom::parse(&a.name()).unwrap(), *a); }
    let s = state("7k/8/8/8/8/R7/R7/K7 w - - 0 1");
    let r = Reader::new(&g, &geo, &s, 0);
    let v = atom_values(&lib, &r);
    assert_eq!(v.len(), lib.len());
    assert!(lib.contains(&Atom::Dist(grp(false, kk), grp(true, kk))) && lib.contains(&Atom::Region(grp(true, kk))) && lib.contains(&Atom::Region(grp(false, rk))));
}

/// their leader in the corner with our long-mover on the second line: its region can be made smaller within one of our moves; against a term no move of ours can lower the question says no
#[test]
fn the_question_finds_a_move_that_lowers_the_term_against_every_reply_and_says_no_where_there_is_none() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let kk = kind(&g, "king");
    let s = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let t = Term::parse(&format!("region(1,{kk})")).unwrap();
    let held = value(&t, &Reader::new(&g, &geo, &s, 0));
    let mut b = Budget::new(20.0, 50_000_000);
    let (found, cut) = lowering(&g, &geo, &s, 0, &t, &held, 2, &cs_rep::history::Seen::new(), &mut b);
    assert!(!cut);
    let f = found.expect("a line lowers the region");
    assert!(f.moves <= 2);
    assert!(f.first < g.moves(&s).len());
    println!("OBS measure question: the region of their leader ({} places) is lowered within {} of our moves by the first move {}", held[0], f.moves, g.move_name(&s, &g.moves(&s)[f.first]));
    // a term nothing can lower: the turn is 0 at every position where it is our turn, and nothing is below 0
    let turn = Term::parse("turn").unwrap();
    let mut b2 = Budget::new(20.0, 50_000_000);
    let (none, cut2) = lowering(&g, &geo, &s, 0, &turn, &[0], 2, &cs_rep::history::Seen::new(), &mut b2);
    assert!(none.is_none() && !cut2);
    // not our move: no question
    let theirs = play(&g, &s, "a2a3");
    let mut b3 = Budget::new(1.0, 1_000_000);
    assert!(lowering(&g, &geo, &theirs, 0, &t, &held, 2, &cs_rep::history::Seen::new(), &mut b3).0.is_none());
}

/// the game's own end counts: with the end one move off, the question finds it even with a term nothing can lower
#[test]
fn the_game_end_found_counts() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let s = state("7k/R7/6K1/8/8/8/8/8 w - - 0 1");
    let turn = Term::parse("turn").unwrap();
    let mut b = Budget::new(20.0, 50_000_000);
    let (found, _) = lowering(&g, &geo, &s, 0, &turn, &[0], 1, &cs_rep::history::Seen::new(), &mut b);
    let f = found.expect("a move ends the game");
    assert!(f.game_end && f.moves == 1);
    assert_eq!(g.move_name(&s, &g.moves(&s)[f.first]), "a7a8");
}

/// NEGATIVE CONTROL: a held value below every reading (nothing can be lower) finds no lowering
#[test]
fn control_a_held_value_below_every_reading_finds_no_lowering() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let kk = kind(&g, "king");
    let s = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let t = Term::parse(&format!("region(1,{kk})")).unwrap();
    let mut b = Budget::new(20.0, 50_000_000);
    let (found, _) = lowering(&g, &geo, &s, 0, &t, &[0], 2, &cs_rep::history::Seen::new(), &mut b);
    assert!(found.is_none());
    println!("OBS measure control: a held value of 0 is below every reading of the region: no lowering found");
}

// ---------------------------------------------------------------------------------------------------------------------------- PLAN 79.18 PART B: the fall counted against the lowest held; a recovery; a position met before

/// THE FALL IS COUNTED AGAINST THE LOWEST HELD. With the lowest held equal to the value now the question is October's (a fall). With the lowest held below what any line reaches there is no fall, and a line that brings the term below the value now is a recovery, which is not a fall.
#[test]
fn a_fall_is_counted_against_the_lowest_held_and_a_line_below_the_value_now_only_is_a_recovery() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let kk = kind(&g, "king");
    let s = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let t = Term::parse(&format!("region(1,{kk})")).unwrap();
    let now = value(&t, &Reader::new(&g, &geo, &s, 0));
    let mut b = Budget::new(20.0, 50_000_000);
    let (f, cut) = question(&g, &geo, &s, 0, &t, &now, &now, 2, &cs_rep::history::Seen::new(), &mut b);
    assert!(!cut && matches!(f, Some(Found::Fall(_))), "the lowest held is the value now: a fall is found: {f:?}");
    let mut b2 = Budget::new(20.0, 50_000_000);
    let (r, cut2) = question(&g, &geo, &s, 0, &t, &[0], &now, 2, &cs_rep::history::Seen::new(), &mut b2);
    assert!(!cut2);
    let r = r.expect("a line below the value now exists");
    assert!(!r.is_fall(), "nothing is below the lowest held of 0: it is a recovery, not a fall: {r:?}");
    println!("OBS measure question 79.18: lowest held {now:?}: a fall; lowest held [0], value now {now:?}: a recovery of {} of our moves, first move {}", r.lowering().moves, g.move_name(&s, &g.moves(&s)[r.lowering().first]));
}

/// A REPLY HAS RAISED THE TERM: the long-mover steps up and their leader answers; the value now is above the lowest held, and the question is asked against the lowest, not against the value now.
#[test]
fn a_reply_that_raised_the_term_is_asked_against_the_lowest_held() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let kk = kind(&g, "king");
    let t = Term::parse(&format!("region(1,{kk})")).unwrap();
    let s0 = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let lowest = value(&t, &Reader::new(&g, &geo, &s0, 0));
    let s = play(&g, &play(&g, &s0, "a2a7"), "h8g8");
    let now = value(&t, &Reader::new(&g, &geo, &s, 0));
    assert!(lower(&lowest, &now), "the reply raised the term: {lowest:?} then {now:?}");
    let mut b = Budget::new(20.0, 50_000_000);
    let (f, cut) = question(&g, &geo, &s, 0, &t, &lowest, &now, 2, &cs_rep::history::Seen::new(), &mut b);
    assert!(!cut);
    let f = f.expect("some line lowers it");
    // against the value now the old question finds a line; against the lowest the same line is a fall only if it goes below the lowest
    let mut b2 = Budget::new(20.0, 50_000_000);
    let old = lowering(&g, &geo, &s, 0, &t, &now, 2, &cs_rep::history::Seen::new(), &mut b2).0.expect("the old question finds a line below the value now");
    println!("OBS measure question 79.18: lowest {lowest:?}, value now {now:?}: found {} (old question's first move {}, new {})", if f.is_fall() { "a fall" } else { "a recovery" }, g.move_name(&s, &g.moves(&s)[old.first]), g.move_name(&s, &g.moves(&s)[f.lowering().first]));
    if f.is_fall() {
        // a fall below the lowest is below the value now as well: the old question finds a line too
        assert!(lower(&lowest, &now));
    }
}

/// A POSITION MET BEFORE IS NEVER THE END OF A LINE THAT COUNTS: with every position two plies on already in the game's record the question finds no line within one of our moves; without the record it does.
#[test]
fn a_position_met_before_is_never_the_end_of_a_line_that_counts() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let kk = kind(&g, "king");
    let s = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let t = Term::parse(&format!("region(1,{kk})")).unwrap();
    let now = value(&t, &Reader::new(&g, &geo, &s, 0));
    let mut b = Budget::new(20.0, 50_000_000);
    let (free, _) = question(&g, &geo, &s, 0, &t, &now, &now, 1, &cs_rep::history::Seen::new(), &mut b);
    assert!(free.is_some(), "without a record a line of one move lowers the region");
    let mut seen = cs_rep::history::Seen::new();
    for m in g.moves(&s) { let q = g.apply(&s, &m); for r in g.moves(&q) { seen.add(g.key(&g.apply(&q, &r))); } }
    let mut b2 = Budget::new(20.0, 50_000_000);
    let (met, cut) = question(&g, &geo, &s, 0, &t, &now, &now, 1, &seen, &mut b2);
    assert!(!cut && met.is_none(), "every end of a line was met before: none counts: {met:?}");
    println!("OBS measure question 79.18: a line of one move is found without the record and none with every position two plies on in the record");
}

/// NEGATIVE CONTROL (a deliberately wrong build would pass this): a lowest held below every reading and a value now with nothing below it finds nothing; a cut budget is a gap and never a recovery.
#[test]
fn control_nothing_below_the_lowest_or_the_value_now_finds_nothing_and_a_cut_is_a_gap() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let s = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let turn = Term::parse("turn").unwrap();
    let mut b = Budget::new(20.0, 50_000_000);
    let (f, cut) = question(&g, &geo, &s, 0, &turn, &[0], &[0], 2, &cs_rep::history::Seen::new(), &mut b);
    assert!(f.is_none() && !cut);
    let kk = kind(&g, "king");
    let t = Term::parse(&format!("region(1,{kk})")).unwrap();
    let now = value(&t, &Reader::new(&g, &geo, &s, 0));
    let mut tiny = Budget::new(20.0, 3);
    let (f2, cut2) = question(&g, &geo, &s, 0, &t, &[0], &now, 2, &cs_rep::history::Seen::new(), &mut tiny);
    assert!(f2.is_none() && cut2, "a cut question names nothing: {f2:?}");
    println!("OBS measure control 79.18: nothing below the lowest or the value now: no line; a cut budget: a gap");
}

// ---------------------------------------------------------------------------------------------------------------------------- PLAN 79.18: the cost of the question (a memo of the answers read in an iteration)

/// THE MEMO CHANGES THE COST AND NOT THE ANSWER: the search of toward.rs with and without the memo gives the same class and value for every root move, on constructed positions, at several depths, with and without a record of the game; the memo reads fewer nodes.
#[test]
fn the_memo_of_the_search_gives_the_same_answers_in_fewer_nodes() {
    use cs_tactics::toward::{toward_memo, toward_with, Verdict};
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let kk = kind(&g, "king");
    let t = Term::parse(&format!("[tuple region(1,{kk}) dist(0,{kk},1,{kk})]")).unwrap();
    let mut plain_nodes = 0u64;
    let mut memo_nodes = 0u64;
    for fen in ["7k/8/8/8/8/8/R7/K7 w - - 0 1", "8/8/R4K2/8/8/3k4/8/8 w - - 0 1", "8/8/8/3R4/2k1K3/8/8/8 w - - 0 1", "8/3k4/5R2/5K2/8/8/8/8 w - - 0 1"] {
        let s = state(fen);
        let held = value(&t, &Reader::new(&g, &geo, &s, 0));
        let oracle = |gm: &KernelGame, st: &St| -> Option<Verdict> { if gm.mover(st) != Some(0) { return None; } if lower(&value(&t, &Reader::new(gm, &geo, st, 0)), &held) { Some(Verdict::Win) } else { None } };
        // a record: the position two plies on of the first move and the first reply
        let mut hist = cs_rep::history::Seen::new();
        let m0 = g.moves(&s)[0].clone();
        let q = g.apply(&s, &m0);
        if let Some(r0) = g.moves(&q).first() { hist.add(g.key(&g.apply(&q, r0))); }
        for rec in [&cs_rep::history::Seen::new(), &hist] {
            for depth in [2u32, 4, 5] {
                let mut b1 = Budget::new(60.0, 400_000_000);
                let a = toward_with(&g, &s, 0, &oracle, rec, depth, None, &mut b1);
                let mut b2 = Budget::new(60.0, 400_000_000);
                let m = toward_memo(&g, &s, 0, &oracle, rec, depth, None, &mut b2);
                assert!(!a.cut && !m.cut);
                assert_eq!(a.depth, m.depth, "{fen} depth {depth}");
                assert_eq!(a.root, m.root, "{fen} depth {depth}: the same class and value for every root move");
                plain_nodes += a.nodes;
                memo_nodes += m.nodes;
            }
        }
    }
    assert!(memo_nodes < plain_nodes, "the memo reads fewer nodes: {memo_nodes} against {plain_nodes}");
    println!("OBS measure memo 79.18: the same roots on 4 positions, 3 depths, with and without a record; nodes read {plain_nodes} without the memo and {memo_nodes} with it");
}
