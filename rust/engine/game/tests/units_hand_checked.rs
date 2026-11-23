//! PLAN 79.36 PART B (LAYER 2): WHAT WAS CHECKED BY HAND IN PLANS 79.23 TO 79.35 AND HAD NO TEST (SYSTEM_SPEC 11, last block; piece: rules; ledger rows 43, 41). Each test says in its name the thing that was read by hand. This file names the games on purpose (a test file may).
//! A test marked to be skipped holds a fault that is NOT mended here, with its cause in the mark.
mod variants_common;
use cs_game::gdl_game::GdlGame;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::placed::{NoStep, Placed};
use cs_game::rules::{restart_delta, Ruled, StateText};
use cs_game::{Fact, Game};
use std::collections::BTreeSet;
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
fn kernel_moves(g: &R, s: &<R as Game>::State) -> BTreeSet<String> { Game::moves(g, s).iter().map(|m| spec_term(&Game::move_name(g, s, m))).collect() }

/// the moves the description allows at the position the kernel is at, begun from the position's facts
fn described_moves(g: &R, s: &<R as Game>::State) -> BTreeSet<String> {
    let facts: Vec<Fact> = g.game().state_facts(s);
    let delta = restart_delta(g.description(), &facts).unwrap();
    let d = GdlGame::load("shown", &cs_gdl::compose(g.description(), &delta).unwrap()).unwrap();
    let st = d.initial();
    d.moves(&st).iter().map(|m| d.move_name(&st, m)).collect()
}

// ------------------------------------------------------------------------------------------------------------------------------ in atomic, whether a side strikes a place (the 79.35 READ, item ii)

#[test]
fn in_chess_a_unit_strikes_the_places_it_bears_on_and_no_others() {
    let (g, s) = open_game("chess", "7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    assert!(Placed::strikes_place(&g, &s, 0, place(&g, "a8")), "the rook on a1 bears on the whole a-file");
    assert!(!Placed::strikes_place(&g, &s, 0, place(&g, "g7")));
    assert!(Placed::strikes_place(&g, &s, 1, place(&g, "g7")), "the black leader on h8 bears on g7");
}

#[test]
#[ignore = "awaiting the decision on rules from the description only: strikes_place (outside.rs) reads the chess attack tables for every game; the atomic rule would be a branch on the game written by hand"]
fn in_atomic_a_leader_beside_the_other_leader_strikes_nothing_because_taking_it_would_remove_both() {
    let (g, s) = open_game("atomic", "8/8/8/8/8/8/8/6Kk w - - 0 1");
    assert!(!Placed::strikes_place(&g, &s, 0, place(&g, "h1")), "white's leader beside black's strikes nothing in atomic: it may not take");
    assert!(!Placed::strikes_place(&g, &s, 1, place(&g, "g1")), "and black's strikes nothing either");
}

#[test]
fn in_atomic_a_unit_that_is_not_the_leader_still_strikes_the_other_leader_beside_it() {
    let (g, s) = open_game("atomic", "8/8/8/8/8/8/Rk6/K7 w - - 0 1");
    assert!(Placed::strikes_place(&g, &s, 0, place(&g, "b2")), "a rook beside the leader strikes the place it stands on");
}

// ------------------------------------------------------------------------------------------------------------------------------ the refusal reasons that fall to one coarse word

#[test]
fn a_step_refused_has_the_reason_the_rules_give_unit_gone_place_taken_line_shut_or_none_of_these() {
    let (g, s) = open_game("chess", "4k3/8/8/3p4/8/3R4/3P4/4K3 w - - 0 1");
    let why = |from: &str, to: &str| Placed::why_not(&g, &s, place(&g, from), place(&g, to));
    assert_eq!(why("a4", "a5").0, NoStep::UnitGone, "nothing of the side to move stands on a4");
    assert_eq!(why("d4", "d5").0, NoStep::UnitGone, "d4 is empty here");
    assert_eq!(why("d3", "d2"), (NoStep::PlaceTaken, vec![place(&g, "d2")]), "its own unit stands there");
    assert_eq!(why("d3", "d6").0, NoStep::LineShut, "the black unit on d5 stands on the line");
    assert_eq!(why("d3", "e5").0, NoStep::MayNotMove, "a rook does not move like that");
    assert_eq!(why("d2", "d3").0, NoStep::PlaceTaken, "a smallest unit may not step onto a unit straight ahead");
}

#[test]
fn a_pinned_units_step_and_a_leaders_step_onto_a_struck_place_are_refused_with_the_one_coarse_word() {
    // the true reasons (the unit is pinned to its leader; the place is struck) have no word of their own: both fall to `MayNotMove`, naming the place the unit stands on
    let (g, s) = open_game("chess", "3rk3/8/8/8/8/8/4B3/4RK2 w - - 0 1");
    assert_eq!(Placed::why_not(&g, &s, place(&g, "e2"), place(&g, "d3")), (NoStep::MayNotMove, vec![place(&g, "e2")]), "a bishop's step, but the leader would be left in the black rook's line");
    let (g2, s2) = open_game("chess", "3rk3/8/8/8/8/8/8/5K2 w - - 0 1");
    assert_eq!(Placed::why_not(&g2, &s2, place(&g2, "f1"), place(&g2, "e1")), (NoStep::MayNotMove, vec![place(&g2, "f1")]), "e1 is on the line of the black rook on e8");
}

// ------------------------------------------------------------------------------------------------------------------------------ Chess960: the right of a corner unit that is not the outermost (the 79.35 READ, item iii)

#[test]
fn in_chess960_the_description_allows_the_two_unit_move_of_the_leader_with_the_outermost_corner_unit() {
    let (g, s) = open_game("chess960", "4k3/8/8/8/8/8/8/1R2K2R w KQ - 0 1");
    assert_eq!(described_moves(&g, &s), kernel_moves(&g, &s));
}

#[test]
#[ignore = "awaiting the decision on rules from the description only: the description reads the corner unit of a wing as the outermost one of the start and a right names no file, so a state shown with another holder of the right is not said; mending needs the right's file as a fluent (state_facts in outside.rs, the kernel's facts) "]
fn in_chess960_the_description_and_the_kernel_agree_where_the_right_belongs_to_a_corner_unit_that_is_not_the_outermost() {
    // the leader on e1, units on b1 and c1, the right held by the one on c1 (the inner one): the kernel allows the two-unit move with c1, the description reads the outermost only
    let (g, s) = open_game("chess960", "4k3/8/8/8/8/8/8/1RR1K3 w C - 0 1");
    assert_eq!(described_moves(&g, &s), kernel_moves(&g, &s));
}
