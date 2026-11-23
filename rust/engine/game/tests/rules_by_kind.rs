//! PLAN 79.34 PART E: THE UNIT TESTS OF LAYER 2, THE RULES (SYSTEM_SPEC 11 "The cut of layers 1 and 2 is carried into the code"; ledger rows 37, 36; piece: rules).
//! Layer 2 says how each kind moves, what a move changes, whose turn it is, what ends the game, and where the game starts, and it refers to the kinds and the places by layer 1's names. This file names the game of chess on purpose (a test file may): it
//! makes positions from the game's text form and compares what layer 2 says with sets of places WRITTEN OUT BY HAND here. It uses layer 1 (cs_rep) and layer 2 (cs_game) and no crate above.
//! (1) the start of the game: where each unit stands, how many of each kind; (2) each kind alone on the board, with a unit of its own side in the way and with a unit of the other side in the way: the places it bears on and may move to;
//! (3) the special rules: the double first step, taking in passing, the two-unit move of the leader with a corner unit, reaching the last row; (4) the turn passes; (5) a move changes the state as said; (6) each end of the game;
//! (7) held against the kernel's own moves on every position of the fixtures. Each prints OBS lines.
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Place, Placed};
use cs_game::rules::{Ruled, StateText};
use cs_game::Game;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

type R = Ruled<KernelGame>;
const WHITE: usize = 0;
const BLACK: usize = 1;

fn rules_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules") }

fn open(variant: &str) -> R {
    Ruled::open(KernelGame::new(variant, 518).unwrap(), variant, &rules_root(), "learned/chess/graph_vocabulary.json").unwrap()
}

fn chess() -> R { open("chess") }

fn place_of(g: &R, text: &str) -> Place { g.place_names().iter().position(|n| n == text).unwrap_or_else(|| panic!("no place {text}")) as Place }
fn kind_of(g: &R, text: &str) -> Kind { g.kind_names().iter().position(|n| n == text).unwrap_or_else(|| panic!("no kind {text}")) as Kind }
fn text_of(g: &R, p: Place) -> String { g.place_names()[p as usize].clone() }
fn set(v: &[&str]) -> BTreeSet<String> { v.iter().map(|s| s.to_string()).collect() }

/// The game's text form of a position made from (place, letter) pairs (an upper-case letter is white's unit, a lower-case one black's), the side to move, the rights, the passed-over place and the clock.
fn text(pieces: &[(&str, char)], turn: char, rights: &str, passed: &str, clock: u32) -> String {
    let mut rows = vec![];
    for rank in (1..=8).rev() {
        let mut row = String::new();
        let mut empty = 0;
        for file in 'a'..='h' {
            let sq = format!("{file}{rank}");
            match pieces.iter().find(|(s, _)| *s == sq) {
                Some((_, c)) => { if empty > 0 { row.push_str(&empty.to_string()); empty = 0; } row.push(*c); }
                None => empty += 1,
            }
        }
        if empty > 0 { row.push_str(&empty.to_string()); }
        rows.push(row);
    }
    format!("{} {turn} {rights} {passed} {clock} 1", rows.join("/"))
}

fn state(g: &R, pieces: &[(&str, char)], turn: char) -> <R as Game>::State { g.read_state_text(&text(pieces, turn, "-", "-", 0)).unwrap() }

fn names(g: &R, v: Vec<Place>) -> BTreeSet<String> { v.into_iter().map(|p| text_of(g, p)).collect() }
fn bears(g: &R, s: &<R as Game>::State, owner: usize, kind: &str, at: &str) -> BTreeSet<String> { names(g, g.bearing_from(s, owner, kind_of(g, kind), place_of(g, at))) }
fn reach(g: &R, s: &<R as Game>::State, owner: usize, kind: &str, at: &str) -> BTreeSet<String> { names(g, g.reach_from(s, owner, kind_of(g, kind), place_of(g, at))) }
fn move_names(g: &R, s: &<R as Game>::State) -> BTreeSet<String> { g.moves(s).iter().map(|m| g.move_name(s, m)).collect() }
fn cells(g: &R, s: &<R as Game>::State) -> BTreeSet<(usize, String, String)> { g.cells(s).into_iter().map(|(o, k, p)| (o, g.kind_names()[k as usize].clone(), text_of(g, p))).collect() }
fn play(g: &R, s: &<R as Game>::State, uci: &str) -> <R as Game>::State {
    let m = g.moves(s).into_iter().find(|m| g.move_name(s, m) == uci).unwrap_or_else(|| panic!("{uci} is not a move"));
    g.apply(s, &m)
}

// ------------------------------------------------------------------------------------------------------------------------------ (1) the start

#[test]
fn the_start_puts_each_unit_where_the_rules_say_and_the_counts_are_eight_two_two_two_one_one_a_side() {
    let back = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'];
    let mut want: BTreeSet<(usize, String, String)> = BTreeSet::new();
    let name = |c: char| match c { 'r' => "rook", 'n' => "knight", 'b' => "bishop", 'q' => "queen", 'k' => "king", _ => "pawn" }.to_string();
    for (i, file) in ('a'..='h').enumerate() {
        want.insert((WHITE, name(back[i]), format!("{file}1")));
        want.insert((WHITE, "pawn".into(), format!("{file}2")));
        want.insert((BLACK, "pawn".into(), format!("{file}7")));
        want.insert((BLACK, name(back[i]), format!("{file}8")));
    }
    for variant in ["chess", "kingofthehill", "3check"] {
        let g = open(variant);
        let s = g.initial();
        assert_eq!(cells(&g, &s), want, "{variant}: where each unit stands at the start");
        let mut count: BTreeMap<(usize, String), usize> = BTreeMap::new();
        for (o, k, _) in cells(&g, &s) { *count.entry((o, k)).or_default() += 1; }
        for owner in [WHITE, BLACK] {
            let of = |k: &str| count[&(owner, k.to_string())];
            assert_eq!((of("pawn"), of("rook"), of("knight"), of("bishop"), of("queen"), of("king")), (8, 2, 2, 2, 1, 1), "{variant}: side {owner}");
        }
        assert_eq!(g.mover(&s), Some(WHITE), "{variant}: white is to move at the start");
        assert_eq!(g.moves(&s).len(), 20, "{variant}: the twenty first moves");
        assert!(g.outcome(&s).is_none());
    }
    println!("OBS 79.34 E start: chess, king of the hill and three-check start from one arrangement (32 units, 8 2 2 2 1 1 a side, white to move, 20 moves); no fixture of a variant with another start exists in the repo, so none is added");
}

// ------------------------------------------------------------------------------------------------------------------------------ (2) each kind, alone and with units in the way
// The white leader stands on h1 and the black leader on a8 in every position below, far from the unit under test (the units must exist for a position to be one); a unit of either side stands in the way where the test says so.

fn kings() -> Vec<(&'static str, char)> { vec![("h1", 'K'), ("a8", 'k')] }
fn with(extra: &[(&'static str, char)]) -> Vec<(&'static str, char)> { let mut v = kings(); v.extend_from_slice(extra); v }

/// (kind, letter, the places it bears on alone on d4, and may move to alone)
fn diagonals_of_d4() -> Vec<&'static str> { vec!["a1", "b2", "c3", "e5", "f6", "g7", "h8", "a7", "b6", "c5", "e3", "f2", "g1"] }
fn lines_of_d4() -> Vec<&'static str> { vec!["d1", "d2", "d3", "d5", "d6", "d7", "d8", "a4", "b4", "c4", "e4", "f4", "g4", "h4"] }

#[test]
fn each_kind_alone_on_the_board_bears_on_and_may_move_to_exactly_the_places_written_here() {
    let g = chess();
    let queen: Vec<&str> = diagonals_of_d4().into_iter().chain(lines_of_d4()).collect();
    let table: Vec<(&str, char, Vec<&str>)> = vec![
        ("bishop", 'B', diagonals_of_d4()),
        ("rook", 'R', lines_of_d4()),
        ("queen", 'Q', queen),
        ("knight", 'N', vec!["b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"]),
    ];
    for (kind, letter, places) in table {
        let s = state(&g, &with(&[("d4", letter)]), 'w');
        assert_eq!(bears(&g, &s, WHITE, kind, "d4"), set(&places), "{kind} bears on");
        assert_eq!(reach(&g, &s, WHITE, kind, "d4"), set(&places), "{kind} may move to");
    }
    // the leader: the eight places around it (the black leader stands far off)
    let s = state(&g, &[("d4", 'K'), ("a8", 'k')], 'w');
    let around = set(&["c3", "c4", "c5", "d3", "d5", "e3", "e4", "e5"]);
    assert_eq!((bears(&g, &s, WHITE, "king", "d4"), reach(&g, &s, WHITE, "king", "d4")), (around.clone(), around));
    // the smallest kind: white's bears on the two places diagonally ahead and may move to the one straight ahead; black's the mirror
    let s = state(&g, &with(&[("d4", 'P')]), 'w');
    assert_eq!(bears(&g, &s, WHITE, "pawn", "d4"), set(&["c5", "e5"]));
    assert_eq!(reach(&g, &s, WHITE, "pawn", "d4"), set(&["d5"]));
    let s = state(&g, &with(&[("d4", 'p')]), 'b');
    assert_eq!(bears(&g, &s, BLACK, "pawn", "d4"), set(&["c3", "e3"]));
    assert_eq!(reach(&g, &s, BLACK, "pawn", "d4"), set(&["d3"]));
    println!("OBS 79.34 E alone: bishop, rook, queen, knight, leader and smallest kind (both sides) bear on and may move to the places written in the test");
}

#[test]
fn a_unit_of_its_own_side_in_the_way_is_borne_on_but_never_moved_to_and_hides_what_is_behind_it() {
    let g = chess();
    // a rook on d4 with its own smallest unit on d6 and another on f4; a bishop on d4 with its own on f6 and c3 (white to move; the position is white's to be played)
    let s = state(&g, &with(&[("d4", 'R'), ("d6", 'P'), ("f4", 'P')]), 'w');
    assert_eq!(bears(&g, &s, WHITE, "rook", "d4"), set(&["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4"]), "a rook bears on what its own side holds: it guards it");
    assert_eq!(reach(&g, &s, WHITE, "rook", "d4"), set(&["d1", "d2", "d3", "d5", "a4", "b4", "c4", "e4"]), "it does not move onto it, nor past it");
    let s = state(&g, &with(&[("d4", 'B'), ("f6", 'P'), ("c3", 'P')]), 'w');
    assert_eq!(bears(&g, &s, WHITE, "bishop", "d4"), set(&["c3", "e5", "f6", "a7", "b6", "c5", "e3", "f2", "g1"]));
    assert_eq!(reach(&g, &s, WHITE, "bishop", "d4"), set(&["e5", "a7", "b6", "c5", "e3", "f2", "g1"]));
    let s = state(&g, &with(&[("d4", 'Q'), ("d5", 'N'), ("e4", 'B'), ("c3", 'R'), ("e5", 'P'), ("c4", 'P'), ("d3", 'P'), ("c5", 'P'), ("e3", 'P')]), 'w');
    assert_eq!(bears(&g, &s, WHITE, "queen", "d4"), set(&["d5", "e4", "c3", "e5", "c4", "d3", "c5", "e3"]), "a queen hemmed in by its own units bears on each and on nothing more");
    assert!(reach(&g, &s, WHITE, "queen", "d4").is_empty(), "and may move nowhere");
    // a knight leaps: units next to it are no way
    let s = state(&g, &with(&[("d4", 'N'), ("d5", 'P'), ("e4", 'P'), ("e6", 'P'), ("b3", 'P')]), 'w');
    assert_eq!(reach(&g, &s, WHITE, "knight", "d4"), set(&["b5", "c2", "c6", "e2", "f3", "f5"]), "it leaps over d5 and e4, and does not land on e6 or b3, its own units");
    assert_eq!(bears(&g, &s, WHITE, "knight", "d4"), set(&["b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"]));
    // the leader beside its own units
    let s = state(&g, &[("d4", 'K'), ("d5", 'P'), ("c3", 'N'), ("a8", 'k')], 'w');
    assert_eq!(reach(&g, &s, WHITE, "king", "d4"), set(&["c4", "c5", "d3", "e3", "e4", "e5"]));
    assert_eq!(bears(&g, &s, WHITE, "king", "d4"), set(&["c3", "c4", "c5", "d3", "d5", "e3", "e4", "e5"]));
    // the smallest kind: a unit of its own side ahead shuts the step, and one diagonally ahead is guarded and not taken
    let s = state(&g, &with(&[("d4", 'P'), ("d5", 'P'), ("e5", 'N')]), 'w');
    assert_eq!(reach(&g, &s, WHITE, "pawn", "d4"), set(&[]));
    assert_eq!(bears(&g, &s, WHITE, "pawn", "d4"), set(&["c5", "e5"]));
    println!("OBS 79.34 E own unit in the way: borne on (guarded), never moved to, what lies behind it is hidden; a knight leaps");
}

#[test]
fn a_unit_of_the_other_side_in_the_way_may_be_taken_and_hides_what_is_behind_it() {
    let g = chess();
    let s = state(&g, &with(&[("d4", 'R'), ("d6", 'n'), ("f4", 'p')]), 'w');
    assert_eq!(bears(&g, &s, WHITE, "rook", "d4"), set(&["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4"]));
    assert_eq!(reach(&g, &s, WHITE, "rook", "d4"), set(&["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4"]), "it may move onto each and stops there");
    let s = state(&g, &with(&[("d4", 'B'), ("f6", 'r'), ("c3", 'q')]), 'w');
    assert_eq!(reach(&g, &s, WHITE, "bishop", "d4"), set(&["c3", "e5", "f6", "a7", "b6", "c5", "e3", "f2", "g1"]));
    assert!(!bears(&g, &s, WHITE, "bishop", "d4").contains("g7") && !bears(&g, &s, WHITE, "bishop", "d4").contains("b2"), "what is behind is hidden");
    let s = state(&g, &with(&[("d4", 'N'), ("b3", 'b'), ("f5", 'q')]), 'w');
    assert_eq!(reach(&g, &s, WHITE, "knight", "d4"), set(&["b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"]));
    // the smallest kind takes only diagonally ahead, and a unit straight ahead shuts it
    let s = state(&g, &with(&[("d4", 'P'), ("d5", 'r'), ("c5", 'n'), ("e5", 'b')]), 'w');
    assert_eq!(reach(&g, &s, WHITE, "pawn", "d4"), set(&["c5", "e5"]));
    assert_eq!(bears(&g, &s, WHITE, "pawn", "d4"), set(&["c5", "e5"]));
    let s = state(&g, &with(&[("d4", 'P'), ("c5", 'n')]), 'w');
    assert_eq!(reach(&g, &s, WHITE, "pawn", "d4"), set(&["c5", "d5"]));
    // the leader of the other side is borne on and is not a place to move to
    let s = state(&g, &[("h1", 'K'), ("d4", 'R'), ("d6", 'k')], 'b');
    assert_eq!(bears(&g, &s, WHITE, "rook", "d4"), set(&["d1", "d2", "d3", "d5", "d6", "a4", "b4", "c4", "e4", "f4", "g4", "h4"]));
    assert_eq!(reach(&g, &s, WHITE, "rook", "d4"), set(&["d1", "d2", "d3", "d5", "a4", "b4", "c4", "e4", "f4", "g4", "h4"]), "the other side's leader is never a place to move to");
    println!("OBS 79.34 E other side's unit in the way: may be taken and stops the line; its leader is borne on, never a place to move to");
}

// ------------------------------------------------------------------------------------------------------------------------------ (3) the special rules

#[test]
fn the_smallest_kind_may_advance_two_places_on_its_first_step_only_and_only_over_empty_places() {
    let g = chess();
    let s = state(&g, &with(&[("e2", 'P')]), 'w');
    assert!(move_names(&g, &s).is_superset(&set(&["e2e3", "e2e4"])));
    let s = state(&g, &with(&[("e3", 'P')]), 'w');
    let m = move_names(&g, &s);
    assert!(m.contains("e3e4") && !m.contains("e3e5"), "off its first row it advances one place");
    let s = state(&g, &with(&[("e2", 'P'), ("e3", 'n')]), 'w');
    let m = move_names(&g, &s);
    assert!(!m.contains("e2e3") && !m.contains("e2e4"), "a unit on the first place shuts both");
    let s = state(&g, &with(&[("e2", 'P'), ("e4", 'n')]), 'w');
    let m = move_names(&g, &s);
    assert!(m.contains("e2e3") && !m.contains("e2e4"), "a unit on the second place shuts the long step only");
    let s = state(&g, &with(&[("e7", 'p')]), 'b');
    assert!(move_names(&g, &s).is_superset(&set(&["e7e6", "e7e5"])), "black's first step is the mirror");
    println!("OBS 79.34 E double first step: two places from the first row over empty places only, one place elsewhere, mirrored for the other side");
}

#[test]
fn taking_in_passing_is_offered_for_one_move_only_and_removes_the_passed_unit() {
    let g = chess();
    // black has just played d7d5 beside white's e5: the passed-over place is d6
    let s = g.read_state_text(&text(&with(&[("e5", 'P'), ("d5", 'p')]), 'w', "-", "d6", 0)).unwrap();
    assert!(move_names(&g, &s).contains("e5d6"), "the unit beside may take as if the other had advanced one place");
    let after = play(&g, &s, "e5d6");
    let c = cells(&g, &after);
    assert!(c.contains(&(WHITE, "pawn".into(), "d6".into())) && !c.contains(&(BLACK, "pawn".into(), "d5".into())) && !c.contains(&(WHITE, "pawn".into(), "e5".into())), "the taken unit left d5, not the place the taker landed on");
    assert_eq!(c.len(), cells(&g, &s).len() - 1);
    // without the passed-over place it is not a move; and it is not on offer if the right was not made at once (no place named)
    let s = g.read_state_text(&text(&with(&[("e5", 'P'), ("d5", 'p')]), 'w', "-", "-", 0)).unwrap();
    assert!(!move_names(&g, &s).contains("e5d6"));
    // it is the kernel's own two-step that makes the place: after d7d5 from the start of a column the right is there
    let s = g.read_state_text(&text(&with(&[("e5", 'P'), ("d7", 'p')]), 'b', "-", "-", 0)).unwrap();
    let s = play(&g, &s, "d7d5");
    assert!(move_names(&g, &s).contains("e5d6"));
    // and it is gone after any other move
    let s = play(&g, &s, "h1h2");
    let s = play(&g, &s, "a8a7");
    assert!(!move_names(&g, &s).contains("e5d6"));
    println!("OBS 79.34 E taking in passing: offered right after a two-place step beside the taker, removes the passed unit from its own place, gone one move later");
}

#[test]
fn the_leader_moves_two_places_with_the_corner_unit_when_the_rights_hold_and_the_way_is_clear_and_safe() {
    let g = chess();
    let ps: Vec<(&str, char)> = vec![("e1", 'K'), ("a1", 'R'), ("h1", 'R'), ("d8", 'k')];
    let s = g.read_state_text(&text(&ps, 'w', "KQ", "-", 0)).unwrap();
    let m = move_names(&g, &s);
    assert!(m.contains("e1g1") && m.contains("e1c1"));
    let after = play(&g, &s, "e1g1");
    let c = cells(&g, &after);
    assert!(c.contains(&(WHITE, "king".into(), "g1".into())) && c.contains(&(WHITE, "rook".into(), "f1".into())) && !c.contains(&(WHITE, "rook".into(), "h1".into())) && !c.contains(&(WHITE, "king".into(), "e1".into())), "two units change: the leader to g1 and the corner unit to f1");
    let after = play(&g, &s, "e1c1");
    let c = cells(&g, &after);
    assert!(c.contains(&(WHITE, "king".into(), "c1".into())) && c.contains(&(WHITE, "rook".into(), "d1".into())) && !c.contains(&(WHITE, "rook".into(), "a1".into())));
    // no right, no move
    let s2 = g.read_state_text(&text(&ps, 'w', "-", "-", 0)).unwrap();
    assert!(!move_names(&g, &s2).contains("e1g1"));
    // a unit between
    let mut ps2 = ps.clone(); ps2.push(("f1", 'B'));
    let s3 = g.read_state_text(&text(&ps2, 'w', "KQ", "-", 0)).unwrap();
    assert!(!move_names(&g, &s3).contains("e1g1") && move_names(&g, &s3).contains("e1c1"));
    // a place it passes over is borne on by the other side
    let mut ps3 = ps.clone(); ps3.push(("f8", 'r'));
    let s4 = g.read_state_text(&text(&ps3, 'w', "KQ", "-", 0)).unwrap();
    assert!(!move_names(&g, &s4).contains("e1g1") && move_names(&g, &s4).contains("e1c1"));
    // the other side's leader is not on the first row: the mirrored move for black
    let ps4 = vec![("e8", 'k'), ("h8", 'r'), ("a2", 'K')];
    let s5 = g.read_state_text(&text(&ps4, 'b', "k", "-", 0)).unwrap();
    assert!(move_names(&g, &s5).contains("e8g8"));
    println!("OBS 79.34 E leader with corner unit: both wings, two units change; refused without the right, with a unit between, over a place borne on");
}

#[test]
fn reaching_the_last_row_turns_the_smallest_kind_into_one_of_four_kinds() {
    let g = chess();
    let s = state(&g, &[("h1", 'K'), ("g6", 'k'), ("b7", 'P'), ("c8", 'r')], 'w');
    let m = move_names(&g, &s);
    assert!(set(&["b7b8q", "b7b8r", "b7b8b", "b7b8n", "b7c8q", "b7c8r", "b7c8b", "b7c8n"]).is_subset(&m), "the step and the taking each give four kinds");
    assert!(!m.contains("b7b8") && !m.contains("b7b8k") && !m.contains("b7b8p"), "never left as it is, never the leader");
    let after = play(&g, &s, "b7b8n");
    let c = cells(&g, &after);
    assert!(c.contains(&(WHITE, "knight".into(), "b8".into())) && !c.contains(&(WHITE, "pawn".into(), "b7".into())), "the unit now stands as the chosen kind");
    assert_eq!(c.len(), cells(&g, &s).len());
    let after = play(&g, &s, "b7c8q");
    let c = cells(&g, &after);
    assert!(c.contains(&(WHITE, "queen".into(), "c8".into())) && !c.contains(&(BLACK, "rook".into(), "c8".into())), "taking and turning in one move");
    assert_eq!(c.len(), cells(&g, &s).len() - 1);
    let s = state(&g, &with(&[("b2", 'p')]), 'b');
    assert!(move_names(&g, &s).is_superset(&set(&["b2b1q", "b2b1r", "b2b1b", "b2b1n"])), "the mirror for the other side");
    println!("OBS 79.34 E last row: four kinds, by step or by taking, the unit stands as the chosen kind");
}

// ------------------------------------------------------------------------------------------------------------------------------ (4)(5) the turn and what a move changes

#[test]
fn the_turn_passes_with_every_move_and_a_move_changes_the_state_as_said() {
    let g = chess();
    let mut s = g.initial();
    let mut who = g.mover(&s);
    assert_eq!(who, Some(WHITE));
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"] {
        let before = cells(&g, &s);
        let after = play(&g, &s, uci);
        let now = cells(&g, &after);
        let (from, to) = (&uci[..2], &uci[2..4]);
        let mover = who.unwrap();
        let moved: Vec<&(usize, String, String)> = before.iter().filter(|c| c.2 == from).collect();
        assert_eq!(moved.len(), 1);
        assert_eq!(before.difference(&now).cloned().collect::<Vec<_>>(), vec![(mover, moved[0].1.clone(), from.to_string())], "{uci}: exactly the unit that moved leaves its place");
        assert_eq!(now.difference(&before).cloned().collect::<Vec<_>>(), vec![(mover, moved[0].1.clone(), to.to_string())], "{uci}: and stands on the other, as the same kind");
        assert_eq!(g.mover(&after), Some(1 - mover), "{uci}: the turn passes");
        who = g.mover(&after);
        s = after;
    }
    // a taking removes the taken unit and no other
    let s = state(&g, &with(&[("d4", 'R'), ("d6", 'n')]), 'w');
    let before = cells(&g, &s);
    let now = cells(&g, &play(&g, &s, "d4d6"));
    assert_eq!(before.difference(&now).cloned().collect::<BTreeSet<_>>(), [(WHITE, "rook".to_string(), "d4".to_string()), (BLACK, "knight".to_string(), "d6".to_string())].into_iter().collect());
    assert_eq!(now.difference(&before).cloned().collect::<Vec<_>>(), vec![(WHITE, "rook".to_string(), "d6".to_string())]);
    // what the move says of itself
    let m = g.moves(&s).into_iter().find(|m| g.move_name(&s, m) == "d4d6").unwrap();
    assert_eq!(g.move_info(&s, &m), (kind_of(&g, "rook"), Some(kind_of(&g, "knight"))));
    assert_eq!(g.move_ends(&s, &m), (place_of(&g, "d4"), place_of(&g, "d6")));
    println!("OBS 79.34 E turn and change: five moves each take one unit off its place and put it on another as the same kind, the turn passes each time; a taking removes the taken unit");
}

// ------------------------------------------------------------------------------------------------------------------------------ (6) each end of the game

#[test]
fn each_end_of_the_game_is_said_with_its_result_and_nobody_is_to_move() {
    let g = chess();
    let ended = |s: &<R as Game>::State, want: Vec<i64>, why: &str| {
        assert_eq!(g.outcome(s), Some(want), "{why}");
        assert_eq!(g.mover(s), None, "{why}: nobody is to move");
        assert!(g.moves(s).is_empty(), "{why}: no move");
    };
    // the leader of the side to move is attacked and nothing helps (beaten)
    let s = g.read_state_text("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3").unwrap();
    ended(&s, vec![0, 100], "beaten");
    // no move and not attacked (no move left): a draw
    let s = g.read_state_text("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    ended(&s, vec![50, 50], "no move left");
    // no material that could give an end: a draw
    let s = g.read_state_text("k7/8/8/8/8/8/8/7K w - - 0 1").unwrap();
    ended(&s, vec![50, 50], "dead material");
    let s = g.read_state_text("k7/8/8/8/8/8/8/6NK w - - 0 1").unwrap();
    ended(&s, vec![50, 50], "dead material with one small unit");
    // the counter of moves with no taking and no advance of the smallest kind
    let s = g.read_state_text("1k6/8/8/8/8/8/8/R6K w - - 100 80").unwrap();
    ended(&s, vec![50, 50], "the counter");
    let s = g.read_state_text("1k6/8/8/8/8/8/8/R6K w - - 99 80").unwrap();
    assert!(g.outcome(&s).is_none() && g.mover(&s) == Some(WHITE), "one move short of the counter the game goes on");
    // a variant's own end: the leader reaches the middle
    let h = open("kingofthehill");
    let s = h.read_state_text("k7/8/8/8/4K3/8/8/8 b - - 0 1").unwrap();
    assert_eq!(h.outcome(&s), Some(vec![100, 0]), "king of the hill: the leader on the middle");
    assert_eq!(h.mover(&s), None);
    // the same position in the standard game is only two leaders alone: dead material, a draw
    let s = g.read_state_text("k7/8/8/8/4K3/8/8/8 b - - 0 1").unwrap();
    assert_eq!(g.outcome(&s), Some(vec![50, 50]));
    println!("OBS 79.34 E ends: beaten, no move left, dead material, the counter, and the variant's own end each give their result with nobody to move");
}

// ------------------------------------------------------------------------------------------------------------------------------ (7) held against the kernel's own moves, on every position of the fixtures

fn fixture_texts(name: &str) -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(format!("{name}.fixture"));
    std::fs::read_to_string(path).unwrap().lines().filter(|l| l.starts_with("pos\t")).map(|l| l.split('\t').nth(2).unwrap().to_string()).collect()
}

#[test]
fn layer_2s_places_to_move_to_agree_with_the_kernels_own_moves_on_every_position_of_the_fixtures() {
    let (mut positions, mut moves, mut special, mut explained, mut unexplained_skipped) = (0, 0, 0, 0, 0);
    for variant in ["chess", "kingofthehill", "3check"] {
        let g = open(variant);
        for t in fixture_texts(variant) {
            let s = g.read_state_text(&t).unwrap();
            positions += 1;
            let mover = match g.mover(&s) { Some(m) => m, None => continue };
            let mut legal: BTreeSet<(Place, Place)> = BTreeSet::new();
            let units: Vec<(usize, Kind, Place)> = g.cells(&s);
            for m in g.moves(&s) {
                moves += 1;
                let (from, to) = g.move_ends(&s, &m);
                legal.insert((from, to));
                let (kind, _) = g.move_info(&s, &m);
                assert!(units.iter().any(|u| *u == (mover, kind, from)), "{variant} {t}: a move starts on a unit of the mover of the kind it says");
                let (fx, fy, tx, ty) = (from / 8, from % 8, to / 8, to % 8);
                let two_ahead = kind == kind_of(&g, "pawn") && fx == tx && (fy as i32 - ty as i32).abs() == 2;
                let diagonal_to_empty = kind == kind_of(&g, "pawn") && fx != tx && !units.iter().any(|u| u.2 == to);
                let two_aside = kind == kind_of(&g, "king") && (fx as i32 - tx as i32).abs() == 2;
                if two_ahead || diagonal_to_empty || two_aside { special += 1; continue; }
                assert!(g.reach_from(&s, mover, kind, from).contains(&to), "{variant} {t}: a move of the kernel is a place layer 2 says it may move to ({} to {})", text_of(&g, from), text_of(&g, to));
            }
            // the converse: a place layer 2 says a unit may move to is either a move of the kernel or is refused because the mover's leader would then be attacked
            let leader = kind_of(&g, "king");
            let own_leader = units.iter().find(|u| u.0 == mover && u.1 == leader).map(|u| u.2);
            for &(o, k, at) in units.iter().filter(|u| u.0 == mover) {
                for to in g.reach_from(&s, o, k, at) {
                    if legal.contains(&(at, to)) { continue; }
                    match g.relocate(&s, at, to) {
                        Some(s2) => {
                            let lp = if k == leader { to } else { own_leader.expect("a leader") };
                            // the other side to move would bear on the leader: layer 2's own `strikes_place` says it
                            assert!(g.strikes_place(&s2, 1 - mover, lp), "{variant} {t}: {} to {} is in layer 2's reach and not a move, yet the leader is not attacked", text_of(&g, at), text_of(&g, to));
                            explained += 1;
                        }
                        None => unexplained_skipped += 1,
                    }
                }
            }
        }
    }
    println!("OBS 79.34 E kernel: {positions} fixture positions, {moves} kernel moves each a place layer 2 may move to (or one of the {special} of the three special rules); {explained} reach places the kernel refuses are each a move that leaves the leader attacked, {unexplained_skipped} not testable");
}

#[test]
fn bearing_on_a_place_is_the_same_as_striking_it_on_every_position_of_the_fixtures() {
    let (mut positions, mut cases) = (0, 0);
    for variant in ["chess", "kingofthehill", "3check"] {
        let g = open(variant);
        for t in fixture_texts(variant) {
            let s = g.read_state_text(&t).unwrap();
            positions += 1;
            let units = g.cells(&s);
            for owner in [WHITE, BLACK] {
                for p in 0..64u16 {
                    let by_units = units.iter().filter(|u| u.0 == owner).any(|&(o, k, at)| g.bearing_from(&s, o, k, at).contains(&p));
                    assert_eq!(g.strikes_place(&s, owner, p), by_units, "{variant} {t}: side {owner} on {}", text_of(&g, p));
                    cases += 1;
                }
            }
        }
    }
    println!("OBS 79.34 E bearing: {cases} (side, place) cases over {positions} positions: bearing on a place by some unit equals the game's own striking of it");
}
