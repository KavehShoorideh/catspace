//! PLAN 79.17 PART B, UNIT TESTS (SYSTEM_SPEC 7, the measure of progress; piece: the measure): the atoms read from the base graph on constructed positions, the term's text, and October's question (`lowering`) answered by the search of toward.rs.
//! The readings are also compared with October's Python in tests/test_plan79_17.py (through the binding). Constructed positions only.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_operator::Budget;
use cs_tactics::measure::{atom_values, groups_of, library, lower, lowering, value, Atom, Geometry, Group, Reader, Term};
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
    let (found, cut) = lowering(&g, &geo, &s, 0, &t, &held, 2, &HashSet::new(), &mut b);
    assert!(!cut);
    let f = found.expect("a line lowers the region");
    assert!(f.moves <= 2);
    assert!(f.first < g.moves(&s).len());
    println!("OBS measure question: the region of their leader ({} places) is lowered within {} of our moves by the first move {}", held[0], f.moves, g.move_name(&s, &g.moves(&s)[f.first]));
    // a term nothing can lower: the turn is 0 at every position where it is our turn, and nothing is below 0
    let turn = Term::parse("turn").unwrap();
    let mut b2 = Budget::new(20.0, 50_000_000);
    let (none, cut2) = lowering(&g, &geo, &s, 0, &turn, &[0], 2, &HashSet::new(), &mut b2);
    assert!(none.is_none() && !cut2);
    // not our move: no question
    let theirs = play(&g, &s, "a2a3");
    let mut b3 = Budget::new(1.0, 1_000_000);
    assert!(lowering(&g, &geo, &theirs, 0, &t, &held, 2, &HashSet::new(), &mut b3).0.is_none());
}

/// the game's own end counts: with the end one move off, the question finds it even with a term nothing can lower
#[test]
fn the_game_end_found_counts() {
    let (g, st) = world();
    let geo = Geometry::new(&st);
    let s = state("7k/R7/6K1/8/8/8/8/8 w - - 0 1");
    let turn = Term::parse("turn").unwrap();
    let mut b = Budget::new(20.0, 50_000_000);
    let (found, _) = lowering(&g, &geo, &s, 0, &turn, &[0], 1, &HashSet::new(), &mut b);
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
    let (found, _) = lowering(&g, &geo, &s, 0, &t, &[0], 2, &HashSet::new(), &mut b);
    assert!(found.is_none());
    println!("OBS measure control: a held value of 0 is below every reading of the region: no lowering found");
}
