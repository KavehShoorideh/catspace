//! PLAN 79.36 PART B (LAYER 2), CONTINUED: the hypothetical boards, the counter at a hundred, the checks counted a state late, the repeat that ends a game (SYSTEM_SPEC 11, last block; piece: rules; ledger rows 43, 41). This file names the games on purpose (a test file may).
mod variants_common;
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::rules::{Declaration, Ruled, Rules};
use cs_game::{Fact, Game};
use variants_common::rules_root;

type R = Ruled<KernelGame>;

const DECL: &str = "learned/chess/graph_vocabulary.json";

fn open_game(game: &str, fen: &str) -> (R, <KernelGame as Game>::State) {
    let kernel = KernelGame::from_fen(game, fen).unwrap();
    let delta = if game == "chess" { None } else { Some(format!("variants/{game}.gdl")) };
    let r = Ruled::open_variant(kernel, game, &rules_root(), DECL, delta.as_deref(), None).unwrap();
    let s = Game::initial(&r);
    (r, s)
}

fn place(g: &R, text: &str) -> u16 { Placed::place_names(g).iter().position(|n| n == text).unwrap() as u16 }

// ------------------------------------------------------------------------------------------------------------------------------ the hypothetical boards answer nothing where the library refuses the board (the counts of the 79.35 smoke test)

#[test]
fn a_board_with_a_unit_taken_off_is_a_position_unless_the_library_refuses_it_and_the_side_to_move_is_kept() {
    let (g, s) = open_game("chess", "r3k3/8/8/8/8/8/3B4/R3K3 w - - 0 1");
    let off = Placed::without(&g, &s, place(&g, "d2")).expect("a bishop off the board leaves a position");
    assert!(Placed::kind_at(&g, &off, place(&g, "d2")).is_none());
    assert_eq!(Game::mover(&g, &off), Some(0), "the side to move is kept");
    assert!(Placed::without(&g, &s, place(&g, "c5")).is_some(), "a place holding nothing is left as it is");
    assert!(Placed::without(&g, &s, place(&g, "e1")).is_none(), "the white leader off: the library refuses a board with no leader");
    assert!(Placed::without(&g, &s, place(&g, "e8")).is_none());
}

#[test]
fn a_unit_put_elsewhere_stands_there_and_a_unit_that_was_there_is_gone_and_nothing_to_move_is_none() {
    let (g, s) = open_game("chess", "4k3/8/8/8/3n4/8/3R4/4K3 w - - 0 1");
    let moved = Placed::relocate(&g, &s, place(&g, "d2"), place(&g, "d4")).unwrap();
    let rook = Placed::kind_names(&g).iter().position(|k| k == "rook").unwrap() as u8;
    assert_eq!(Placed::kind_at(&g, &moved, place(&g, "d4")), Some(rook), "the rook stands on d4 and the knight that was there is gone");
    assert!(Placed::kind_at(&g, &moved, place(&g, "d2")).is_none());
    assert_eq!(Placed::cells(&g, &moved).len(), 3, "two leaders and the rook: the knight is gone");
    assert!(Placed::relocate(&g, &s, place(&g, "a3"), place(&g, "a4")).is_none(), "nothing stands on a3");
}

#[test]
fn a_board_the_library_refuses_is_none_the_side_not_to_move_left_struck() {
    // the black leader on e8 and a white rook put on e2 with the BLACK side to move is fine; with the WHITE side to move the black side would stand struck, which no position allows
    let (g, s) = open_game("chess", "4k3/8/8/8/8/8/3R4/4K3 w - - 0 1");
    assert!(Placed::relocate(&g, &s, place(&g, "d2"), place(&g, "e2")).is_none(), "white to move and black struck: refused");
    let (b, sb) = open_game("chess", "4k3/8/8/8/8/8/3R4/4K3 b - - 0 1");
    assert!(Placed::relocate(&b, &sb, place(&b, "d2"), place(&b, "e2")).is_some(), "black to move and struck is a position");
}

#[test]
fn a_hypothetical_board_keeps_the_right_to_the_two_unit_move_only_where_the_leader_and_the_corner_unit_still_stand() {
    let (g, s) = open_game("chess", "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let both = g.game().state_facts(&s);
    assert_eq!(both.iter().filter(|f| f.relation == "castling_right").count(), 2);
    let moved = Placed::relocate(&g, &s, place(&g, "a1"), place(&g, "a2")).unwrap();
    let facts = g.game().state_facts(&moved);
    let rights: Vec<&Fact> = facts.iter().filter(|f| f.relation == "castling_right").collect();
    assert_eq!(rights.len(), 1, "the corner unit left its corner: that right is gone");
    assert_eq!(rights[0].args, vec!["white".to_string(), "kingside".to_string()]);
}

// ------------------------------------------------------------------------------------------------------------------------------ the counter at a hundred, the checks counted a state late, the repeat that ends the game

#[test]
fn the_game_is_drawn_when_the_clock_reaches_a_hundred_and_not_before() {
    let (g99, s99) = open_game("chess", "4k3/8/8/8/8/8/8/4K2R w - - 99 80");
    assert_eq!(Game::outcome(&g99, &s99), None);
    let (g100, s100) = open_game("chess", "4k3/8/8/8/8/8/8/4K2R w - - 100 80");
    assert_eq!(Game::outcome(&g100, &s100), Some(vec![50, 50]));
    assert!(Game::moves(&g100, &s100).is_empty() && Game::mover(&g100, &s100).is_none(), "a game that is over has no moves and nobody to move");
}

#[test]
fn in_three_check_a_check_just_given_is_counted_into_the_next_state_and_not_into_the_one_it_was_given_in() {
    let (r, s) = open_game("3check", "4k3/8/8/8/8/8/8/R3K3 w - - 3+3 0 1");
    let count = |s: &<R as Game>::State| -> Option<String> { Game::base_facts(&r, s).iter().find(|f| f.relation == "checks" && f.args[0] == "white").map(|f| f.args[1].clone()) };
    assert_eq!(count(&s), Some("0".to_string()));
    let m = Game::find_move(&r, &s, "a1a8").unwrap();
    let checked = Game::apply(&r, &s, &m);
    assert_eq!(count(&checked), Some("0".to_string()), "black stands checked and to move: the check just given is not counted yet");
    let escape = Game::find_move(&r, &checked, "e8d7").unwrap();
    let after = Game::apply(&r, &checked, &escape);
    assert_eq!(count(&after), Some("1".to_string()), "the next state counts it");
}

#[test]
fn the_repeat_rule_ends_a_game_at_the_limit_the_description_gives_and_a_description_with_no_limit_never_ends_one() {
    let (g, _s) = open_game("chess", "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert_eq!(Rules::repetition_limit(&g), Some(2));
    assert!(!Rules::ends_by_record(&g, 0) && !Rules::ends_by_record(&g, 1));
    assert!(Rules::ends_by_record(&g, 2) && Rules::ends_by_record(&g, 3));
    let decl = Declaration::from_json(&std::fs::read_to_string(rules_root().join(DECL)).unwrap()).unwrap();
    let none = Ruled::new(KernelGame::new("chess", 0).unwrap(), "plain", "(role white) (role black)", decl).unwrap();
    assert_eq!(Rules::repetition_limit(&none), None);
    assert!(!Rules::ends_by_record(&none, 100));
}
