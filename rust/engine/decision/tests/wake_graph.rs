//! Plan 79.05 (SYSTEM_SPEC 8 step 3, the proposer's graph reads), UNIT TESTS of the ported pieces `lines_onto`, `goal_unit_lines` and `new_bearer_moves` (cs_decision::wake): small constructed positions built through the engine's own game interface,
//! asserting each piece's own contract. No puzzle, no stored game, no match.
use cs_decision::wake::{goal_unit_lines, lines_onto, new_bearer_moves};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}

fn at(g: &KernelGame, name: &str) -> u16 { g.place_names().iter().position(|p| p == name).unwrap() as u16 }
fn kind(g: &KernelGame, name: &str) -> u8 { g.kind_names().iter().position(|k| k == name).unwrap() as u8 }

#[test]
fn a_line_onto_the_goal_unit_shut_by_exactly_one_unit_is_found_with_who_bears_and_who_holds() {
    let (g, st) = world();
    // the bishop's diagonal runs f4 e5 d6 c7 b8: the pawn on c7 is the only unit between it and the goal unit on b8
    let s = KernelGame::from_fen("chess", "1k6/2p5/8/8/5B2/8/8/K7 w - - 0 1").unwrap().initial();
    let part = st.position(&g, &s);
    let white = g.roles().iter().position(|r| r == "white").unwrap();
    let ls = lines_onto(&part, at(&g, "b8"), white);
    assert_eq!(ls.len(), 1, "exactly one line onto the goal unit's place is shut by one unit");
    assert_eq!((ls[0].unit.at, ls[0].unit.kind), (at(&g, "f4"), kind(&g, "bishop")));
    assert_eq!((ls[0].blocker.at, ls[0].blocker.kind), (at(&g, "c7"), kind(&g, "pawn")));
    assert_eq!(ls[0].bearers.len(), 1, "the bishop bears on the blocker's place");
    assert_eq!(ls[0].holders.len(), 1, "the goal unit holds the blocker's place");
    assert_eq!(ls[0].holders[0].kind, kind(&g, "king"));
}

#[test]
fn a_second_bearer_outnumbers_the_one_holder() {
    let (g, st) = world();
    let s = KernelGame::from_fen("chess", "1k6/2p5/8/8/5B2/8/8/K1R5 w - - 0 1").unwrap().initial();
    let white = g.roles().iter().position(|r| r == "white").unwrap();
    let ls = lines_onto(&st.position(&g, &s), at(&g, "b8"), white);
    assert_eq!(ls.len(), 1);
    assert_eq!((ls[0].bearers.len(), ls[0].holders.len()), (2, 1));
}

#[test]
fn a_line_shut_by_two_units_or_by_none_is_not_listed() {
    let (g, st) = world();
    let white = g.roles().iter().position(|r| r == "white").unwrap();
    // two units between (c7 and e5): not exactly one
    let s = KernelGame::from_fen("chess", "1k6/2p5/8/4p3/5B2/8/8/K7 w - - 0 1").unwrap().initial();
    assert!(lines_onto(&st.position(&g, &s), at(&g, "b8"), white).is_empty());
    // a place no line of the side reaches: nothing listed
    let s = KernelGame::from_fen("chess", "1k6/2p5/8/8/5B2/8/8/K7 w - - 0 1").unwrap().initial();
    assert!(lines_onto(&st.position(&g, &s), at(&g, "h1"), white).is_empty());
}

#[test]
fn the_side_not_to_move_is_read_after_the_games_pass() {
    let (g, st) = world();
    let s = KernelGame::from_fen("chess", "1k6/2p5/8/8/5B2/8/8/K1R5 b - - 0 1").unwrap().initial();
    let rows = goal_unit_lines(&st, &g, &s, kind(&g, "king"));
    let black = g.roles().iter().position(|r| r == "black").unwrap();
    let mine = rows.iter().find(|r| r.0 == black).unwrap();
    assert_eq!(mine.1, at(&g, "b8"));
    let ls = mine.2.as_ref().expect("the game gives a pass, so the lines of the other side are read");
    assert_eq!(ls.len(), 1);
    assert_eq!((ls[0].bearers.len(), ls[0].holders.len()), (2, 1));
}

#[test]
fn the_moves_that_bring_a_bearer_onto_a_place_are_the_moves_after_which_more_units_bear_on_it() {
    let (g, st) = world();
    let s = KernelGame::from_fen("chess", "1k6/2p5/8/8/8/8/8/R6K w - - 0 1").unwrap().initial();
    let names: Vec<String> = new_bearer_moves(&st, &g, &s, &[at(&g, "c7")]).into_iter().map(|(m, _)| m).collect();
    assert!(names.contains(&"a1a7".to_string()), "the rook on the seventh rank bears on c7 along the rank: {names:?}");
    assert!(names.contains(&"a1c1".to_string()), "the rook on the c file bears on c7 along the file: {names:?}");
    assert!(!names.contains(&"a1a2".to_string()), "a move that adds no bearer is not named");
    assert!(new_bearer_moves(&st, &g, &s, &[]).is_empty(), "no place asked: no move named");
}

#[test]
fn a_bearer_that_a_discovered_line_brings_counts_too() {
    let (g, st) = world();
    // the white rook on c1 is shut from c7 by its own bishop on c4; the bishop's move off the file opens the line
    let s = KernelGame::from_fen("chess", "1k6/2p5/8/8/2B5/8/8/K1R5 w - - 0 1").unwrap().initial();
    let before = st.position(&g, &s);
    let white = g.roles().iter().position(|r| r == "white").unwrap();
    let on_c7 = |p: &cs_tactics::sheet::base_graph::PositionPart| p.bears.iter().filter(|r| r.place == at(&g, "c7") && p.units[r.unit].owner == white).count();
    assert_eq!(on_c7(&before), 0);
    let names: Vec<String> = new_bearer_moves(&st, &g, &s, &[at(&g, "c7")]).into_iter().map(|(m, _)| m).collect();
    assert!(names.iter().any(|m| m.starts_with("c4") && m != "c4c5" && m != "c4c3" && m != "c4c2"), "a bishop move off the file opens the rook's line onto c7: {names:?}");
}
