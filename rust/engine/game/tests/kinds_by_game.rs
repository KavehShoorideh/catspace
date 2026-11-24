//! PLAN 79.36 PART D: THE TESTS OF EVERY KIND IN EVERY GAME (SYSTEM_SPEC 11, last block, second sense of "unit": every kind of unit of every game the rules say; piece: rules; ledger rows 43, 41, 40).
//! The seven games layer 2 says: chess, king of the hill, three-check (the kinds move as in chess), Chess960, horde, atomic, antichess. For each game and each of the six kinds: ALONE on the board, with a unit of ITS OWN SIDE in the way, with a unit of the OTHER SIDE in the way:
//! the places it BEARS ON (the kind's geometry on the board as it stands: its own units are guarded, what lies behind a unit is hidden) and the places it MAY MOVE TO (where its step reaches under the game's rule of what may be taken; compulsion, what else a taking removes and the leader left
//! struck are not read), against sets WRITTEN OUT BY HAND below. Then the games' own differences: the leader that takes nothing in atomic, the leader that may be taken in atomic and antichess, the leader that is an ordinary unit in antichess (and may be the kind a smallest unit becomes), the smallest kind on the first
//! row in horde, the compulsory taking of antichess (not read by `may move to`, read by the moves). This file names the games on purpose (a test file may); it prints one OBS line a game and kind.
//! THE TABLE OF KINDS BY GAME is `SCENARIOS`: every scenario names its kind and its family; `the_table_has_each_kind_in_each_family_for_each_game` counts them (tests/test_plan79_36.py reads it).
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Place, Placed};
use cs_game::Game;
use std::collections::BTreeSet;

pub const GAMES: [&str; 7] = ["chess", "kingofthehill", "3check", "chess960", "horde", "atomic", "antichess"];
pub const KINDS: [&str; 6] = ["bishop", "king", "knight", "pawn", "queen", "rook"];
pub const FAMILIES: [&str; 3] = ["alone", "own in the way", "other in the way"];

const WHITE: usize = 0;
const BLACK: usize = 1;

/// One scenario of the table: the units (place, letter: upper case white's, lower case black's), the side to move, whose unit is asked, its kind and place, and the places it bears on and may move to as written by hand.
struct Scenario { name: &'static str, kind: &'static str, family: &'static str, units: Vec<(&'static str, char)>, turn: char, owner: usize, at: &'static str, bears: Vec<&'static str>, reach: Vec<&'static str> }

fn diagonals_of_d4() -> Vec<&'static str> { vec!["a1", "b2", "c3", "e5", "f6", "g7", "h8", "a7", "b6", "c5", "e3", "f2", "g1"] }
fn lines_of_d4() -> Vec<&'static str> { vec!["d1", "d2", "d3", "d5", "d6", "d7", "d8", "a4", "b4", "c4", "e4", "f4", "g4", "h4"] }
fn knight_of_d4() -> Vec<&'static str> { vec!["b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"] }
fn around_d4() -> Vec<&'static str> { vec!["c3", "c4", "c5", "d3", "d5", "e3", "e4", "e5"] }

/// the backdrop of the scenarios: the two leaders, far from the unit asked (the board of the unit alone)
fn leaders() -> Vec<(&'static str, char)> { vec![("h1", 'K'), ("a8", 'k')] }
fn with(extra: &[(&'static str, char)]) -> Vec<(&'static str, char)> { let mut v = leaders(); v.extend_from_slice(extra); v }

/// THE TABLE (chess: the standard game; the other games differ where `differences` says so)
fn scenarios() -> Vec<Scenario> {
    let queen: Vec<&str> = diagonals_of_d4().into_iter().chain(lines_of_d4()).collect();
    let s = |name, kind, family, units, turn, owner, at, bears: Vec<&'static str>, reach: Vec<&'static str>| Scenario { name, kind, family, units, turn, owner, at, bears, reach };
    vec![
        // ------------------------------------------------------------ alone
        s("bishop alone", "bishop", "alone", with(&[("d4", 'B')]), 'w', WHITE, "d4", diagonals_of_d4(), diagonals_of_d4()),
        s("rook alone", "rook", "alone", with(&[("d4", 'R')]), 'w', WHITE, "d4", lines_of_d4(), lines_of_d4()),
        s("queen alone", "queen", "alone", with(&[("d4", 'Q')]), 'w', WHITE, "d4", queen.clone(), queen),
        s("knight alone", "knight", "alone", with(&[("d4", 'N')]), 'w', WHITE, "d4", knight_of_d4(), knight_of_d4()),
        s("leader alone", "king", "alone", vec![("d4", 'K'), ("a8", 'k')], 'w', WHITE, "d4", around_d4(), around_d4()),
        s("smallest kind alone, white", "pawn", "alone", with(&[("d4", 'P')]), 'w', WHITE, "d4", vec!["c5", "e5"], vec!["d5"]),
        s("smallest kind alone, black", "pawn", "alone", with(&[("d4", 'p')]), 'b', BLACK, "d4", vec!["c3", "e3"], vec!["d3"]),
        // ------------------------------------------------------------ a unit of its own side in the way
        s("rook with its own in the way", "rook", "own in the way", with(&[("d4", 'R'), ("d6", 'P'), ("f4", 'P')]), 'w', WHITE, "d4",
          vec!["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4"], vec!["d1", "d2", "d3", "d5", "a4", "b4", "c4", "e4"]),
        s("bishop with its own in the way", "bishop", "own in the way", with(&[("d4", 'B'), ("f6", 'P'), ("c3", 'P')]), 'w', WHITE, "d4",
          vec!["c3", "e5", "f6", "a7", "b6", "c5", "e3", "f2", "g1"], vec!["e5", "a7", "b6", "c5", "e3", "f2", "g1"]),
        s("queen hemmed in by its own", "queen", "own in the way", with(&[("d4", 'Q'), ("d5", 'N'), ("e4", 'B'), ("c3", 'R'), ("e5", 'P'), ("c4", 'P'), ("d3", 'P'), ("c5", 'P'), ("e3", 'P')]), 'w', WHITE, "d4",
          vec!["d5", "e4", "c3", "e5", "c4", "d3", "c5", "e3"], vec![]),
        s("knight with its own beside and on its places", "knight", "own in the way", with(&[("d4", 'N'), ("d5", 'P'), ("e4", 'P'), ("e6", 'P'), ("b3", 'P')]), 'w', WHITE, "d4",
          knight_of_d4(), vec!["b5", "c2", "c6", "e2", "f3", "f5"]),
        s("leader beside its own", "king", "own in the way", vec![("d4", 'K'), ("d5", 'P'), ("c3", 'N'), ("a8", 'k')], 'w', WHITE, "d4", around_d4(), vec!["c4", "c5", "d3", "e3", "e4", "e5"]),
        s("smallest kind shut by its own ahead", "pawn", "own in the way", with(&[("d4", 'P'), ("d5", 'P'), ("e5", 'N')]), 'w', WHITE, "d4", vec!["c5", "e5"], vec![]),
        // ------------------------------------------------------------ a unit of the other side in the way
        s("rook with the other side's in the way", "rook", "other in the way", with(&[("d4", 'R'), ("d6", 'n'), ("f4", 'p')]), 'w', WHITE, "d4",
          vec!["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4"], vec!["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4"]),
        s("bishop with the other side's in the way", "bishop", "other in the way", with(&[("d4", 'B'), ("f6", 'r'), ("c3", 'q')]), 'w', WHITE, "d4",
          vec!["c3", "e5", "f6", "a7", "b6", "c5", "e3", "f2", "g1"], vec!["c3", "e5", "f6", "a7", "b6", "c5", "e3", "f2", "g1"]),
        s("queen with the other side's in the way", "queen", "other in the way", with(&[("d4", 'Q'), ("d6", 'n'), ("f6", 'p'), ("b4", 'r')]), 'w', WHITE, "d4",
          vec!["d1", "d2", "d3", "d5", "d6", "b4", "c4", "e4", "f4", "g4", "h4", "a1", "b2", "c3", "e5", "f6", "a7", "b6", "c5", "e3", "f2", "g1"],
          vec!["d1", "d2", "d3", "d5", "d6", "b4", "c4", "e4", "f4", "g4", "h4", "a1", "b2", "c3", "e5", "f6", "a7", "b6", "c5", "e3", "f2", "g1"]),
        s("knight with the other side's on its places", "knight", "other in the way", with(&[("d4", 'N'), ("b3", 'b'), ("f5", 'q')]), 'w', WHITE, "d4", knight_of_d4(), knight_of_d4()),
        s("leader beside the other side's units", "king", "other in the way", vec![("d4", 'K'), ("d5", 'n'), ("c3", 'b'), ("a8", 'k')], 'w', WHITE, "d4", around_d4(), around_d4()),
        s("smallest kind with the other side's ahead and aside", "pawn", "other in the way", with(&[("d4", 'P'), ("d5", 'r'), ("c5", 'n'), ("e5", 'b')]), 'w', WHITE, "d4", vec!["c5", "e5"], vec!["c5", "e5"]),
        s("smallest kind taking aside with the way ahead open", "pawn", "other in the way", with(&[("d4", 'P'), ("c5", 'n')]), 'w', WHITE, "d4", vec!["c5", "e5"], vec!["c5", "d5"]),
    ]
}

/// The games' own differences from the table above: (game, scenario name) -> the places it may move to there, written by hand. Where a game is not listed for a scenario it is as in chess.
fn differences(game: &str, name: &str) -> Option<Vec<&'static str>> {
    match (game, name) {
        // in atomic the leader takes nothing: beside the other side's units it moves only to empty places (it still bears on them)
        ("atomic", "leader beside the other side's units") => Some(vec!["c4", "c5", "e3", "e4", "e5", "d3"]),
        _ => None,
    }
}

/// the position of a scenario as the game's text form, for the game's kernel (the backdrop leaders are left out of the games that have none of one side: horde's white side has no leader)
fn fen_of(units: &[(&str, char)], turn: char, game: &str) -> String {
    let mut rows = vec![];
    for rank in (1..=8).rev() {
        let (mut row, mut empty) = (String::new(), 0);
        for file in 'a'..='h' {
            let sq = format!("{file}{rank}");
            match units.iter().find(|(s, c)| *s == sq && !(game == "horde" && *c == 'K')) {
                Some((_, c)) => { if empty > 0 { row.push_str(&empty.to_string()); empty = 0; } row.push(*c); }
                None => empty += 1,
            }
        }
        if empty > 0 { row.push_str(&empty.to_string()); }
        rows.push(row);
    }
    format!("{} {turn} - - 0 1", rows.join("/"))
}

fn open(game: &str, units: &[(&str, char)], turn: char) -> (KernelGame, <KernelGame as Game>::State) {
    let name = if game == "chess960" { "chess960" } else { game };
    let g = KernelGame::from_fen(name, &fen_of(units, turn, game)).unwrap_or_else(|e| panic!("{game}: {e}"));
    let s = g.initial();
    (g, s)
}

fn place_of(g: &KernelGame, text: &str) -> Place { Placed::place_names(g).iter().position(|n| n == text).unwrap_or_else(|| panic!("no place {text}")) as Place }
fn kind_of(g: &KernelGame, text: &str) -> Kind { Placed::kind_names(g).iter().position(|n| n == text).unwrap_or_else(|| panic!("no kind {text}")) as Kind }
fn names(g: &KernelGame, v: Vec<Place>) -> BTreeSet<String> { v.into_iter().map(|p| Placed::place_names(g)[p as usize].clone()).collect() }
fn set(v: &[&str]) -> BTreeSet<String> { v.iter().map(|s| s.to_string()).collect() }

fn asked(game: &str, sc: &Scenario) -> (BTreeSet<String>, BTreeSet<String>) {
    let (g, s) = open(game, &sc.units, sc.turn);
    let (k, at) = (kind_of(&g, sc.kind), place_of(&g, sc.at));
    (names(&g, Placed::bearing_from(&g, &s, sc.owner, k, at)), names(&g, Placed::reach_from(&g, &s, sc.owner, k, at)))
}

/// the games whose scenario involves a leader of the white side that the game does not have
fn applies(game: &str, sc: &Scenario) -> bool { !(game == "horde" && sc.units.iter().any(|(_, c)| *c == 'K') && sc.kind == "king" && sc.owner == WHITE) }

// ------------------------------------------------------------------------------------------------------------------------------ the table is complete

#[test]
fn the_table_has_each_kind_in_each_family_for_each_game() {
    let table = scenarios();
    for kind in KINDS {
        for family in FAMILIES {
            let n = table.iter().filter(|s| s.kind == kind && s.family == family).count();
            assert!(n >= 1, "no scenario of the kind {kind} in the family {family}");
        }
    }
    assert_eq!(GAMES.len(), 7);
    let mut per_game = 0;
    for game in GAMES { per_game += table.iter().filter(|s| applies(game, s)).count(); }
    // seven games, each asked every scenario that applies to it (horde's white side has no leader of its own)
    assert_eq!(per_game, 7 * table.len() - 3, "horde skips the three scenarios of a white leader, the other six games none");
}

// ------------------------------------------------------------------------------------------------------------------------------ the kinds move as in chess: the standard game, king of the hill, three-check

#[test]
fn every_kind_alone_with_its_own_side_in_the_way_and_with_the_other_side_in_the_way_bears_on_and_may_move_to_the_places_written_here_in_chess() {
    for sc in scenarios() {
        let (b, r) = asked("chess", &sc);
        assert_eq!(b, set(&sc.bears), "chess: {} bears on", sc.name);
        assert_eq!(r, set(&sc.reach), "chess: {} may move to", sc.name);
    }
    println!("OBS 79.36 D chess: {} scenarios, every kind in three families, bears on and may move to as written by hand", scenarios().len());
}

#[test]
fn king_of_the_hill_and_three_check_move_every_kind_as_chess_does_the_same_table() {
    for game in ["kingofthehill", "3check"] {
        for sc in scenarios() {
            let (b, r) = asked(game, &sc);
            assert_eq!((b, r), (set(&sc.bears), set(&sc.reach)), "{game}: {}", sc.name);
        }
        println!("OBS 79.36 D {game}: the table of chess, every scenario equal");
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ Chess960: the same kinds on the same board

#[test]
fn chess960_moves_every_kind_as_chess_does_the_same_table() {
    for sc in scenarios() {
        let (b, r) = asked("chess960", &sc);
        assert_eq!((b, r), (set(&sc.bears), set(&sc.reach)), "chess960: {}", sc.name);
    }
    println!("OBS 79.36 D chess960: the table of chess, every scenario equal");
}

// ------------------------------------------------------------------------------------------------------------------------------ horde: the side of many units, with no leader

#[test]
fn horde_moves_every_kind_as_chess_does_where_both_sides_have_what_the_scenario_needs() {
    for sc in scenarios().iter().filter(|s| applies("horde", s)) {
        let (b, r) = asked("horde", sc);
        assert_eq!((b, r), (set(&sc.bears), set(&sc.reach)), "horde: {}", sc.name);
    }
    println!("OBS 79.36 D horde: the table of chess for the scenarios its sides can have; white has no leader");
}

#[test]
fn in_horde_the_black_leader_bears_on_and_may_move_to_what_the_black_leader_of_chess_does_and_a_white_unit_beside_it_may_be_taken() {
    let (g, s) = open("horde", &[("d4", 'k'), ("d5", 'N'), ("c3", 'B')], 'b');
    let k = kind_of(&g, "king");
    assert_eq!(names(&g, Placed::bearing_from(&g, &s, BLACK, k, place_of(&g, "d4"))), set(&around_d4()));
    assert_eq!(names(&g, Placed::reach_from(&g, &s, BLACK, k, place_of(&g, "d4"))), set(&around_d4()), "units of the white side may be taken by the black leader");
}

#[test]
fn in_horde_the_smallest_kind_on_the_first_row_may_advance_one_or_two_places_and_the_one_on_the_second_row_the_same_and_one_on_the_third_only_one() {
    let (g, s) = open("horde", &[("a1", 'P'), ("h8", 'k')], 'w');
    let moves: BTreeSet<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    assert_eq!(moves, set(&["a1a2", "a1a3"]), "from the first row two places are allowed");
    let (g2, s2) = open("horde", &[("a2", 'P'), ("h8", 'k')], 'w');
    let moves2: BTreeSet<String> = g2.moves(&s2).iter().map(|m| g2.move_name(&s2, m)).collect();
    assert_eq!(moves2, set(&["a2a3", "a2a4"]));
    let (g3, s3) = open("horde", &[("a3", 'P'), ("h8", 'k')], 'w');
    let moves3: BTreeSet<String> = g3.moves(&s3).iter().map(|m| g3.move_name(&s3, m)).collect();
    assert_eq!(moves3, set(&["a3a4"]), "from the third row one place");
    // 2026-11-24 (plan 79.50): what the kind may move to is the description's `move_by`, the first move of two places included (it was the one place ahead only, and the double step was said to be the move's)
    let k = kind_of(&g, "pawn");
    assert_eq!(names(&g, Placed::reach_from(&g, &s, WHITE, k, place_of(&g, "a1"))), set(&["a2", "a3"]));
    assert_eq!(names(&g, Placed::bearing_from(&g, &s, WHITE, k, place_of(&g, "a1"))), set(&["b2"]));
}

// ------------------------------------------------------------------------------------------------------------------------------ atomic: the leader takes nothing and may be taken

#[test]
fn atomic_moves_every_kind_as_chess_does_where_a_unit_of_the_other_side_is_not_the_leader_and_the_leader_takes_nothing() {
    for sc in scenarios() {
        let (b, r) = asked("atomic", &sc);
        assert_eq!(b, set(&sc.bears), "atomic: {} bears on", sc.name);
        let want = differences("atomic", sc.name).map(|v| set(&v)).unwrap_or_else(|| set(&sc.reach));
        assert_eq!(r, want, "atomic: {} may move to", sc.name);
    }
    println!("OBS 79.36 D atomic: the table of chess, with the leader that takes nothing");
}

#[test]
fn in_atomic_the_leader_beside_a_unit_of_the_other_side_bears_on_it_and_does_not_take_it_and_in_chess_it_takes_it() {
    let units = [("d4", 'K'), ("d5", 'n'), ("a8", 'k')];
    let (g, s) = open("atomic", &units, 'w');
    let (k, at) = (kind_of(&g, "king"), place_of(&g, "d4"));
    assert!(names(&g, Placed::bearing_from(&g, &s, WHITE, k, at)).contains("d5"));
    assert!(!names(&g, Placed::reach_from(&g, &s, WHITE, k, at)).contains("d5"), "the leader takes nothing in atomic");
    let (c, cs) = open("chess", &units, 'w');
    assert!(names(&c, Placed::reach_from(&c, &cs, WHITE, kind_of(&c, "king"), place_of(&c, "d4"))).contains("d5"));
}

#[test]
fn in_atomic_the_leader_of_the_other_side_is_a_place_to_move_to_for_every_other_kind() {
    let units = [("h1", 'K'), ("d4", 'R'), ("d6", 'k')];
    let (g, s) = open("atomic", &units, 'b');
    let reach = names(&g, Placed::reach_from(&g, &s, WHITE, kind_of(&g, "rook"), place_of(&g, "d4")));
    assert!(reach.contains("d6"), "a taking of the leader is a move in atomic (it removes it)");
    assert_eq!(reach, set(&["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4", "g4", "h4"]));
    let (c, cs) = open("chess", &units, 'b');
    assert!(!names(&c, Placed::reach_from(&c, &cs, WHITE, kind_of(&c, "rook"), place_of(&c, "d4"))).contains("d6"), "in chess it is not");
}

// ------------------------------------------------------------------------------------------------------------------------------ antichess: the leader is an ordinary unit; taking is compulsory

#[test]
fn antichess_moves_every_kind_as_chess_does_the_leader_of_the_other_side_being_a_place_to_move_to() {
    for sc in scenarios() {
        let (b, r) = asked("antichess", &sc);
        assert_eq!(b, set(&sc.bears), "antichess: {} bears on", sc.name);
        assert_eq!(r, set(&sc.reach), "antichess: {} may move to", sc.name);
    }
    // the one place the leader of the other side makes a difference: it is taken like any unit
    let (g, s) = open("antichess", &[("h1", 'K'), ("d4", 'R'), ("d6", 'k')], 'w');
    let reach = names(&g, Placed::reach_from(&g, &s, WHITE, kind_of(&g, "rook"), place_of(&g, "d4")));
    assert_eq!(reach, set(&["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4", "g4", "h4"]));
    println!("OBS 79.36 D antichess: the table of chess, with the leader taken like any unit");
}

#[test]
fn in_antichess_the_leader_takes_a_unit_of_the_other_side_like_any_other_unit() {
    let (g, s) = open("antichess", &[("d4", 'K'), ("d5", 'n'), ("a8", 'k')], 'w');
    assert!(names(&g, Placed::reach_from(&g, &s, WHITE, kind_of(&g, "king"), place_of(&g, "d4"))).contains("d5"));
}

#[test]
fn in_antichess_a_smallest_unit_reaching_the_last_row_may_become_the_leader_too_so_five_kinds_and_in_chess_four() {
    let (g, s) = open("antichess", &[("a7", 'P'), ("h7", 'p')], 'w');
    let moves: BTreeSet<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    assert_eq!(moves, set(&["a7a8q", "a7a8r", "a7a8b", "a7a8n", "a7a8k"]));
    let (c, cs) = open("chess", &[("a7", 'P'), ("h1", 'K'), ("h8", 'k')], 'w');
    let cm: BTreeSet<String> = c.moves(&cs).iter().map(|m| c.move_name(&cs, m)).collect();
    assert!(cm.contains("a7a8q") && !cm.contains("a7a8k"), "chess has four kinds to become");
}

#[test]
fn in_antichess_a_taking_that_is_possible_is_the_only_kind_of_move_though_the_unit_may_move_to_more_places_than_that() {
    let (g, s) = open("antichess", &[("d4", 'R'), ("d6", 'p'), ("h8", 'n')], 'w');
    let rook = kind_of(&g, "rook");
    let reach = names(&g, Placed::reach_from(&g, &s, WHITE, rook, place_of(&g, "d4")));
    assert_eq!(reach, set(&["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4", "g4", "h4"]), "the places it may move to do not read the compulsion");
    let moves: BTreeSet<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    assert_eq!(moves, set(&["d4d6"]), "the moves are the taking alone: no other move is allowed while one is possible");
}

// ------------------------------------------------------------------------------------------------------------------------------ the kinds by game, as a count

#[test]
fn every_game_names_the_same_six_kinds_and_the_same_sixty_four_places() {
    for game in GAMES {
        let g = KernelGame::new(game, 518).unwrap();
        assert_eq!(Placed::kind_names(&g), KINDS.iter().map(|s| s.to_string()).collect::<Vec<_>>(), "{game}");
        assert_eq!(Placed::place_names(&g).len(), 64, "{game}");
        assert_eq!(Placed::place_names(&g)[0], "a1");
        assert_eq!(Placed::place_names(&g)[63], "h8");
    }
    println!("OBS 79.36 D kinds: {} games name {} kinds and 64 places alike", GAMES.len(), KINDS.len());
}
