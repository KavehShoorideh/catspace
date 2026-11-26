//! PLAN 79.36 PART C, LAYER 1, THE BOARD, THE GRAPH OF A STATE, THE KEPT GRAPH AND THE HISTORY: every public item asked on a small board made by hand, and at its edges (SYSTEM_SPEC 11, last block; piece: representation; ledger row 43).
//! The board is a grid of three by three places `n0` to `n8` (place = 3 * row + column) with the directions east, north, south, southeast, west; the expected numbers are worked out by hand in the comments.
//! No object of a game is named (tests/test_plan78.py scans for it).
use cs_rep::fact::{self, Fact, UnitDecl};
use cs_rep::graph::{Board, Classes, FactDecl, LinkType, NodeType, PositionPart, Rel, LinkRec, REL_SLOTS};
use cs_rep::history::{apply_change, History, KeyHasher, Path, Seen};
use cs_rep::live::{Edit, LiveGraph, LiveUnit, LIVE_SLOTS};
use cs_rep::names::Names;
use cs_rep::statement::Change;
use cs_rep::steps::Steps;
use cs_base::sets::Set;
use std::hash::Hasher;

const W: Rel = Rel::new(0, "w");

fn names() -> Names {
    Names::new((0..9).map(|i| format!("n{i}")).collect(), vec!["k0".into(), "k1".into()], vec!["r0".into(), "r1".into()])
}

/// rows of one direction: every (row, column) -> (row + dr, column + dc) that stays on the grid
fn rows(d: u16, dr: i32, dc: i32) -> Vec<(u16, u16, u16)> {
    let mut out = vec![];
    for r in 0..3i32 { for c in 0..3i32 {
        let (nr, nc) = (r + dr, c + dc);
        if (0..3).contains(&nr) && (0..3).contains(&nc) { out.push((d, (3 * r + c) as u16, (3 * nr + nc) as u16)); }
    } }
    out
}

fn steps() -> Steps {
    let mut r = vec![];
    r.extend(rows(0, 0, 1));    // east
    r.extend(rows(1, -1, 0));   // north
    r.extend(rows(2, 1, 0));    // south
    r.extend(rows(3, 1, 1));    // southeast
    r.extend(rows(4, 0, -1));   // west
    Steps { directions: ["east", "north", "south", "southeast", "west"].iter().map(|s| s.to_string()).collect(), rows: r, relation: String::new(), object: String::new() }
}

fn board() -> Board { Board::new(&steps(), &names()).unwrap() }
fn decl() -> FactDecl { FactDecl { units: UnitDecl { relation: "u".into(), owner: 0, kind: Some(1) }, tempo: Some("t".into()) } }

// ------------------------------------------------------------------------------------------------------------------------------ the words of what is seen

#[test]
fn the_words_of_the_facts_of_what_is_seen_are_unit_on_line_and_on_edge() {
    assert_eq!((fact::UNIT, fact::ON_LINE, fact::ON_EDGE), ("unit", "on_line", "on_edge"));
}

#[test]
fn a_node_type_has_a_word_and_a_relation_node_says_the_word_of_its_relation_and_only_that_one_is_a_relation() {
    let words: Vec<&str> = [NodeType::Place, NodeType::Direction, NodeType::Line, NodeType::Edge, NodeType::Kind, NodeType::Side, NodeType::Unit].iter().map(|t| NodeType::word(*t)).collect();
    assert_eq!(words, vec!["place", "direction", "line", "edge", "kind", "side", "unit"]);
    assert_eq!(NodeType::word(NodeType::Relation(W)), "w");
    assert!(NodeType::is_relation(NodeType::Relation(W)));
    assert!(![NodeType::Place, NodeType::Direction, NodeType::Line, NodeType::Edge, NodeType::Kind, NodeType::Side, NodeType::Unit].iter().any(|t| NodeType::is_relation(*t)));
}

#[test]
fn the_fifteen_kinds_of_link_have_fifteen_different_words_in_the_order_of_the_list() {
    assert_eq!(LinkType::ALL.len(), 15);
    let words: Vec<&str> = LinkType::ALL.iter().map(|l| LinkType::word(*l)).collect();
    assert_eq!(words, vec!["on_line", "next", "on_edge", "edge_distance", "line_along", "reverse", "edge_of", "unit_on", "unit_kind", "unit_side", "by", "at_place", "at_unit", "along", "between"]);
    let mut sorted = words.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 15);
}

#[test]
fn a_relation_as_a_value_keeps_its_slot_and_its_word_and_a_layer_may_write_four_kinds() {
    let r = Rel::new(3, "far");
    assert_eq!((r.slot, r.word), (3, "far"));
    assert_eq!(REL_SLOTS, 5);      // 2026-11-25 (plan 79.58 B3): the count of slots rose from four to five (a number only); the test's name still says four
    assert_ne!(Rel::new(0, "a"), Rel::new(1, "a"));
}

// ------------------------------------------------------------------------------------------------------------------------------ the board

#[test]
fn a_board_of_nine_places_has_nine_lines_the_reverse_of_each_direction_and_the_lines_in_order() {
    // east and west share three lines (one a row), north and south three (one a column), southeast three diagonals of two or more places: nine
    let b = board();
    assert_eq!(b.lines.len(), 9);
    assert_eq!(b.reverse, vec![Some(4), Some(2), Some(1), None, Some(0)]);
    assert!(b.lines.iter().any(|l| l.places == vec![0, 1, 2] && b.directions[l.direction] == "east"));
    assert!(b.lines.iter().any(|l| l.places == vec![0, 4, 8] && b.directions[l.direction] == "southeast"));
    assert!(b.lines.iter().all(|l| l.places.len() >= 2));
}

#[test]
fn the_steps_to_the_end_of_each_direction_are_what_is_left_of_the_chain() {
    let b = board();
    assert_eq!(b.edge_distance[0], vec![2, 1, 0, 2, 1, 0, 2, 1, 0], "east: two steps from the first column, none from the last");
    assert_eq!(b.edge_distance[3][4], 1, "southeast from the centre: one step");
    assert_eq!(b.edge_distance[3][8], 0);
    assert_eq!(b.edge_distance[3][2], 0, "the corner at the end of a short diagonal");
}

#[test]
fn the_nearest_edge_is_nought_on_the_rim_and_one_at_the_centre() {
    let b = board();
    assert_eq!(Board::nearest_edge(&b, 4), 1);
    for p in [0u16, 1, 2, 3, 5, 6, 7, 8] { assert_eq!(Board::nearest_edge(&b, p), 0, "place {p}"); }
}

#[test]
fn two_places_share_the_lines_they_are_both_on_and_none_when_no_line_joins_them() {
    let b = board();
    assert_eq!(Board::common_lines(&b, 0, 8).len(), 1, "the long diagonal");
    assert_eq!(Board::common_lines(&b, 0, 2).len(), 1, "the top row");
    assert_eq!(Board::common_lines(&b, 0, 5).len(), 0, "two steps and one aside: no line");
    assert_eq!(Board::common_lines(&b, 4, 4).len(), 3, "a place shares with itself every line it is on: the row, the column and the long diagonal (there is no direction for the other diagonal)");
}

#[test]
fn the_places_strictly_between_two_places_of_a_line_are_in_the_order_of_the_line_and_empty_for_neighbours_and_for_places_not_on_it() {
    let b = board();
    let diag = Board::common_lines(&b, 0, 8)[0];
    assert_eq!(Board::strictly_between(&b, diag, 0, 8), vec![4]);
    assert_eq!(Board::strictly_between(&b, diag, 8, 0), vec![4], "from either end");
    assert_eq!(Board::strictly_between(&b, diag, 0, 4), Vec::<u16>::new());
    assert_eq!(Board::strictly_between(&b, diag, 4, 4), Vec::<u16>::new());
    assert_eq!(Board::strictly_between(&b, diag, 0, 1), Vec::<u16>::new(), "n1 is not on the diagonal");
    assert_eq!(Board::index_on(&b, diag, 4), 1);
    assert_eq!(Board::index_on(&b, diag, 1), -1);
}

#[test]
fn the_two_classes_of_places_alternate_and_a_step_made_of_two_others_does_not_change_the_class() {
    let c: Classes = Board::classes(&board()).unwrap();
    assert_eq!(c.class, (0..9).map(|p| ((p / 3 + p % 3) % 2) as u8).collect::<Vec<_>>());
    assert_eq!(c.flips, vec![true, true, true, false, true], "southeast is an east step and a south step: two changes are none");
    assert_eq!(c.class.iter().filter(|x| **x == 0).count(), 5);
}

#[test]
fn a_board_whose_steps_make_an_odd_circle_says_no_classes_and_a_board_with_no_steps_says_none() {
    // a: n0 -> n1, b: n1 -> n2, c: n2 -> n0, each a change of class: three changes come back to the start
    let odd = Steps { directions: vec!["a".into(), "b".into(), "c".into()], rows: vec![(0, 0, 1), (1, 1, 2), (2, 2, 0)], relation: String::new(), object: String::new() };
    let b = Board::new(&odd, &Names::new(vec!["n0".into(), "n1".into(), "n2".into()], vec![], vec![])).unwrap();
    assert_eq!(Board::classes(&b), None);
    let bare = Board::new(&Steps::default(), &Names::new(vec!["n0".into()], vec![], vec![])).unwrap();
    assert_eq!(Board::classes(&bare), None);
}

#[test]
fn a_direction_that_is_not_a_chain_is_refused_with_the_reason() {
    let names = Names::new(vec!["n0".into(), "n1".into(), "n2".into()], vec![], vec![]);
    let st = |rows: Vec<(u16, u16, u16)>| Steps { directions: vec!["d".into()], rows, relation: String::new(), object: String::new() };
    assert!(Board::new(&st(vec![(0, 0, 1), (0, 0, 2)]), &names).unwrap_err().contains("two places after one"));
    assert!(Board::new(&st(vec![(0, 0, 2), (0, 1, 2)]), &names).unwrap_err().contains("two places before one"));
    assert!(Board::new(&st(vec![(0, 0, 1), (0, 1, 0)]), &names).unwrap_err().contains("closed chain"));
}

#[test]
fn a_board_is_written_to_json_and_read_back_with_its_declaration_and_a_board_without_one_stays_without() {
    let b = board().with_decl(decl());
    let j = Board::to_json(&b);
    assert_eq!(j["places"].as_array().unwrap().len(), 9);
    assert_eq!(j["decl"]["relation"], "u");
    let back = Board::from_json(&j).unwrap();
    assert_eq!(back.decl, Some(decl()));
    assert_eq!(back.steps, b.steps);
    assert_eq!(back.lines.len(), 9);
    assert!(Board::from_json(&Board::to_json(&board())).unwrap().decl.is_none());
}

#[test]
fn a_board_with_a_part_missing_in_its_json_is_refused_with_the_name_of_the_part() {
    let mut j = Board::to_json(&board());
    j.as_object_mut().unwrap().remove("rows");
    assert_eq!(Board::from_json(&j).unwrap_err(), "board.rows is a list");
    let mut k = Board::to_json(&board());
    k.as_object_mut().unwrap().remove("places");
    assert_eq!(Board::from_json(&k).unwrap_err(), "board.places is a list");
    let mut m = Board::to_json(&board());
    m["kinds"] = cs_rep::serde_json::json!([1]);
    assert_eq!(Board::from_json(&m).unwrap_err(), "board.kinds holds text");
}

// ------------------------------------------------------------------------------------------------------------------------------ the graph of a state

fn state() -> PositionPart {
    // owner 0 is marked to act: its units are ours; the units come in any order and are held in the order (owner, kind, place)
    Board::position_from_units(&board(), Some(0), vec![(1, 0, 8), (0, 1, 4), (0, 0, 0)])
}

#[test]
fn a_state_from_numbers_holds_its_units_in_the_order_of_owner_kind_and_place_with_their_ids_and_sides() {
    let p = state();
    let order: Vec<(usize, u8, u16)> = p.units.iter().map(|u| (u.owner, u.kind, u.at)).collect();
    assert_eq!(order, vec![(0, 0, 0), (0, 1, 4), (1, 0, 8)]);
    let ids: Vec<&str> = p.nodes.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids, vec!["unit:r0:k0:n0", "unit:r0:k1:n4", "unit:r1:k0:n8"]);
    assert_eq!(p.units[0].side, cs_rep::names::Side::Us);
    assert_eq!(p.units[2].side, cs_rep::names::Side::Them);
    assert_eq!(p.occupant, vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
    assert_eq!(PositionPart::tempo(&p), Some(0));
}

#[test]
fn a_state_with_no_tempo_marks_nobody_and_a_state_with_no_units_is_empty() {
    let b = board();
    let p = Board::position_from_units(&b, None, vec![(1, 0, 2)]);
    assert_eq!(PositionPart::tempo(&p), None);
    assert_eq!(p.units[0].side, cs_rep::names::Side::Us, "with nobody marked every unit is read as ours");
    let empty = Board::position_from_units(&b, Some(1), vec![]);
    assert!(empty.units.is_empty() && empty.occupant.iter().all(|o| *o == 0) && empty.nodes.is_empty());
}

#[test]
fn how_many_units_of_a_kind_a_side_has_is_counted_and_zero_for_a_kind_it_has_none_of() {
    let p = Board::position_from_units(&board(), Some(0), vec![(0, 0, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]);
    assert_eq!(PositionPart::count(&p, 0, 0), 2);
    assert_eq!(PositionPart::count(&p, 0, 1), 1);
    assert_eq!(PositionPart::count(&p, 1, 0), 1);
    assert_eq!(PositionPart::count(&p, 1, 1), 0);
}

#[test]
fn the_longest_open_run_of_a_line_is_the_longest_stretch_with_no_unit_on_it() {
    let b = board();
    let top = b.lines.iter().position(|l| l.places == vec![0, 1, 2] && b.directions[l.direction] == "east").unwrap();
    let none = Board::position_from_units(&b, Some(0), vec![]);
    assert_eq!(PositionPart::open_length(&none, &b, top), 3);
    let middle = Board::position_from_units(&b, Some(0), vec![(0, 0, 1)]);
    assert_eq!(PositionPart::open_length(&middle, &b, top), 1, "n0 and n2 are open but apart");
    let end = Board::position_from_units(&b, Some(0), vec![(0, 0, 0)]);
    assert_eq!(PositionPart::open_length(&end, &b, top), 2);
    let full = Board::position_from_units(&b, Some(0), vec![(0, 0, 0), (0, 0, 1), (0, 0, 2)]);
    assert_eq!(PositionPart::open_length(&full, &b, top), 0);
}

#[test]
fn the_dump_of_a_state_is_one_line_a_node_with_its_links_by_name() {
    let b = board();
    let p = Board::position_from_units(&b, Some(0), vec![(0, 1, 4)]);
    assert_eq!(PositionPart::dump(&p, &b), "unit:r0:k1:n4 [unit]  unit_on->place:n4 unit_kind->kind:k1 unit_side->side:r0\n");
    assert_eq!(PositionPart::dump(&Board::position_from_units(&b, None, vec![]), &b), "");
}

#[test]
fn the_definitions_of_a_graph_are_its_node_ids_and_its_links_sorted_and_do_not_depend_on_the_order_the_units_came_in() {
    let b = board();
    let a = Board::position_from_units(&b, Some(0), vec![(0, 0, 0), (1, 0, 8)]);
    let c = Board::position_from_units(&b, Some(0), vec![(1, 0, 8), (0, 0, 0)]);
    assert_eq!(PositionPart::definitions(&a, &b), PositionPart::definitions(&c, &b));
    let (nodes, links) = PositionPart::definitions(&a, &b);
    assert_eq!(nodes, vec!["unit:r0:k0:n0".to_string(), "unit:r1:k0:n8".to_string()]);
    assert_eq!(links.len(), 6);
    assert_eq!(links[0], "unit_kind|unit:r0:k0:n0|kind:k0||-1", "sorted: the first by text; an absent `via` is empty and an absent value is -1");
}

#[test]
fn a_state_is_written_as_names_and_read_back_and_a_name_the_game_does_not_give_is_an_error() {
    let b = board();
    let p = state();
    let (marked, units) = Board::state_text(&b, &p);
    assert_eq!(marked, Some("r0".to_string()));
    assert_eq!(units[1], ("r0".to_string(), "k1".to_string(), "n4".to_string()));
    let back = Board::part_from_text(&b, marked.as_deref(), &units).unwrap();
    assert_eq!(PositionPart::definitions(&back, &b), PositionPart::definitions(&p, &b));
    assert_eq!(back.occupant, p.occupant);
    assert_eq!(Board::state_text(&b, &Board::position_from_units(&b, None, vec![])).0, None);
    let bad = |m: Option<&str>, u: (&str, &str, &str)| Board::part_from_text(&b, m, &[(u.0.into(), u.1.into(), u.2.into())]).unwrap_err();
    assert_eq!(bad(None, ("r0", "k0", "n9")), "no such place: n9");
    assert_eq!(bad(None, ("r0", "kx", "n0")), "no such kind: kx");
    assert_eq!(bad(None, ("rx", "k0", "n0")), "no such role: rx");
    assert_eq!(bad(Some("rx"), ("r0", "k0", "n0")), "no such role: rx");
}

// ------------------------------------------------------------------------------------------------------------------------------ the facts of a state

#[test]
fn the_units_and_the_tempo_are_read_off_facts_by_the_declaration_and_a_fact_said_twice_is_one_unit() {
    let b = board().with_decl(decl());
    let facts = [Fact::new("u", &["r1", "k0", "n8"]), Fact::new("u", &["r0", "k1", "n4"]), Fact::new("u", &["r0", "k1", "n4"]), Fact::new("t", &["r1"]), Fact::new("other", &["x"])];
    let (marked, cells) = Board::read_facts(&b, &facts).unwrap();
    assert_eq!(marked, Some(1));
    assert_eq!(cells, vec![(0, 1, 4), (1, 0, 8)]);
    let p = Board::position_from_facts(&b, &facts).unwrap();
    assert_eq!((PositionPart::tempo(&p), p.units.len()), (Some(1), 2));
    assert_eq!(p.units[0].side, cs_rep::names::Side::Them, "owner r0 is not the marked owner r1");
}

#[test]
fn a_unit_with_no_kind_in_its_declaration_is_of_kind_nought_and_the_other_arguments_make_its_place() {
    let b = board().with_decl(FactDecl { units: UnitDecl { relation: "u".into(), owner: 1, kind: None }, tempo: None });
    let (marked, cells) = Board::read_facts(&b, &[Fact::new("u", &["n", "r1", "3"])]).unwrap();
    assert_eq!((marked, cells), (None, vec![(1, 0, 3)]));
}

#[test]
fn facts_that_name_nothing_the_game_gives_or_are_too_short_or_have_no_declaration_to_read_by_are_refused() {
    let b = board().with_decl(decl());
    assert_eq!(Board::read_facts(&b, &[Fact::new("u", &["rx", "k0", "n0"])]).unwrap_err(), "no such role: rx");
    assert_eq!(Board::read_facts(&b, &[Fact::new("u", &["r0", "kx", "n0"])]).unwrap_err(), "no such kind: kx");
    assert_eq!(Board::read_facts(&b, &[Fact::new("u", &["r0", "k0", "nope"])]).unwrap_err(), "no such place: nope");
    assert_eq!(Board::read_facts(&b, &[Fact::new("u", &["r0"])]).unwrap_err(), "a unit fact is too short");
    assert_eq!(Board::read_facts(&b, &[Fact::new("t", &[])]).unwrap_err(), "a tempo fact is empty");
    assert_eq!(Board::read_facts(&board(), &[]).unwrap_err(), "the board has no declaration of where facts hold units");
    assert_eq!(Board::position_from_facts(&board(), &[]).unwrap_err(), "the board has no declaration of where facts hold units");
}

// ------------------------------------------------------------------------------------------------------------------------------ the door of change

#[test]
fn a_relation_written_makes_a_node_named_by_its_word_the_unit_and_the_place_and_links_by_the_units_between() {
    let b = board();
    let mut p = Board::position_from_units(&b, Some(0), vec![(0, 0, 0), (0, 0, 1), (0, 0, 2)]);
    Board::write(&b, &mut p, W, 0, 2);
    let rec = &PositionPart::written(&p, W)[0];
    assert_eq!((rec.unit, rec.place, rec.between.clone()), (0, 2, vec![1]));
    assert_eq!(p.nodes.last().unwrap().id, "w(unit:r0:k0:n0,place:n2)");
    assert_eq!(p.nodes.last().unwrap().kind, NodeType::Relation(W));
    let kinds: Vec<LinkType> = p.links.iter().filter(|l: &&LinkRec| l.a == rec.node).map(|l| l.kind).collect();
    assert_eq!(kinds, vec![LinkType::By, LinkType::AtPlace, LinkType::AtUnit, LinkType::Along, LinkType::Between]);
    assert_eq!(PositionPart::relation_count(&p), 1);
}

#[test]
fn a_relation_written_to_an_empty_place_has_no_unit_there_and_none_between_and_the_slots_are_kept_apart() {
    let b = board();
    let mut p = Board::position_from_units(&b, Some(0), vec![(0, 0, 0)]);
    Board::write(&b, &mut p, W, 0, 8);
    Board::write(&b, &mut p, Rel::new(3, "z"), 0, 1);
    assert_eq!(PositionPart::written(&p, W).len(), 1);
    assert_eq!(PositionPart::written(&p, Rel::new(3, "z")).len(), 1);
    assert_eq!(PositionPart::written(&p, Rel::new(1, "never")).len(), 0);
    assert_eq!(PositionPart::relation_count(&p), 2);
    let first = &PositionPart::written(&p, W)[0];
    assert!(first.between.is_empty());
    assert!(!p.links.iter().any(|l| l.a == first.node && l.kind == LinkType::AtUnit), "nothing stands on n8");
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn a_fifth_kind_of_relation_has_no_slot_and_the_door_refuses_it_by_a_panic_and_does_not_drop_it() {
    // the door has a fixed number of slots (four on a graph): a layer that needs more must raise it; a fifth is never written quietly
    let b = board();
    let mut p = Board::position_from_units(&b, Some(0), vec![(0, 0, 0)]);
    Board::write(&b, &mut p, Rel::new(REL_SLOTS as u8, "fifth"), 0, 1);
}

#[test]
fn the_four_slots_of_the_door_each_hold_their_own_relations_and_a_unit_kept_has_three_slots_of_places() {
    let b = board();
    let mut p = Board::position_from_units(&b, Some(0), vec![(0, 0, 0)]);
    for slot in 0..REL_SLOTS as u8 { Board::write(&b, &mut p, Rel::new(slot, "s"), 0, 1); }
    assert_eq!(PositionPart::relation_count(&p), REL_SLOTS);      // 2026-11-25 (plan 79.58 B3): one row in each of the slots, five since the count rose from four
    for slot in 0..REL_SLOTS as u8 { assert_eq!(PositionPart::written(&p, Rel::new(slot, "s")).len(), 1, "slot {slot}"); }
    assert_eq!(LIVE_SLOTS, 3, "a unit as the graph keeps it has three slots, one fewer than the graph has");
}

// ------------------------------------------------------------------------------------------------------------------------------ the kept graph

fn lu(o: usize, k: u8, at: u16) -> LiveUnit { LiveUnit::new(o, k, at) }

#[test]
fn a_unit_as_the_graph_keeps_it_has_empty_slots_and_its_key_is_owner_kind_place() {
    let u = LiveUnit::new(1, 2, 3);
    assert_eq!(LiveUnit::key(&u), (1, 2, 3));
    assert_eq!(LIVE_SLOTS, 3);
    assert!(u.slots.iter().all(|s| s.is_empty()) && u.slots.len() == LIVE_SLOTS);
}

#[test]
fn a_kept_graph_holds_its_units_in_canonical_order_its_tempo_and_a_depth_that_counts_the_tempos_made() {
    let mut g = LiveGraph::new(Some(1), vec![lu(1, 0, 3), lu(0, 1, 2), lu(0, 0, 5)]);
    assert_eq!(g.units().iter().map(|u| LiveUnit::key(u)).collect::<Vec<_>>(), vec![(0, 0, 5), (0, 1, 2), (1, 0, 3)]);
    assert_eq!((g.tempo(), g.depth()), (Some(1), 0));
    g.make(Edit { tempo: Some(0), ..Default::default() });
    g.make(Edit { tempo: None, ..Default::default() });
    assert_eq!((g.tempo(), g.depth()), (None, 2));
}

#[test]
fn settling_a_kept_graph_lets_go_of_what_it_could_be_taken_back_to_and_keeps_what_stands() {
    let mut g = LiveGraph::new(Some(0), vec![lu(0, 0, 1)]);
    g.make(Edit { tempo: Some(1), added: vec![(1, 0, 2)], ..Default::default() });
    assert_eq!((g.depth(), g.units().len()), (1, 2));
    g.settle();
    assert_eq!((g.depth(), g.units().len(), g.tempo()), (0, 2, Some(1)));
    assert!(!g.undo(), "nothing to take back after settling");
}

#[test]
fn a_tempo_removes_adds_rewrites_and_sets_the_last_slot_and_taking_it_back_restores_every_unit_and_slot() {
    let mut g = LiveGraph::new(Some(0), vec![lu(0, 0, 1), lu(1, 0, 4)]);
    let before = g.units().to_vec();
    g.make(Edit { tempo: Some(1), removed: vec![(1, 0, 4), (9, 9, 9)], added: vec![(1, 0, 6), (0, 0, 1)], rewrite: vec![((0, 0, 1), [vec![2, 3], vec![5]]), ((7, 7, 7), [vec![], vec![]])],
                  clear_last: true, last: vec![((1, 0, 6), vec![8])] });
    let keys: Vec<_> = g.units().iter().map(|u| LiveUnit::key(u)).collect();
    assert_eq!(keys, vec![(0, 0, 1), (1, 0, 6)], "the unit at n4 is gone, n6 stands, a unit added that was there is one unit, a removal and a rewrite of nothing do nothing");
    assert_eq!((g.units()[0].slots[0].clone(), g.units()[0].slots[1].clone()), (vec![2, 3], vec![5]));
    assert_eq!(g.units()[1].slots[LIVE_SLOTS - 1], vec![8]);
    assert!(g.units()[0].slots[LIVE_SLOTS - 1].is_empty());
    assert!(g.undo());
    assert_eq!(g.units().to_vec(), before);
    assert_eq!(g.tempo(), Some(0));
}

#[test]
fn clearing_the_last_slot_empties_every_units_last_slot_before_the_new_ones_are_set() {
    let mut g = LiveGraph::new(Some(0), vec![lu(0, 0, 1), lu(1, 0, 2)]);
    g.make(Edit { last: vec![((0, 0, 1), vec![4]), ((1, 0, 2), vec![5])], ..Default::default() });
    g.make(Edit { clear_last: true, last: vec![((0, 0, 1), vec![6])], ..Default::default() });
    assert_eq!((g.units()[0].slots[2].clone(), g.units()[1].slots[2].clone()), (vec![6], vec![]));
}

#[test]
fn who_stands_where_in_the_kept_graph_is_the_index_in_the_unit_list_plus_one_and_the_units_between_two_places_are_listed_by_index() {
    let b = board();
    let g = LiveGraph::new(Some(0), vec![lu(0, 0, 0), lu(0, 0, 1), lu(1, 0, 2), lu(1, 0, 4)]);
    let occ = LiveGraph::occupant(&g, &b);
    assert_eq!(occ, vec![1, 2, 3, 0, 4, 0, 0, 0, 0]);
    assert_eq!(LiveGraph::between(&g, &b, &occ, 0, 2), vec![1], "the unit on n1 lies between n0 and n2");
    assert_eq!(LiveGraph::between(&g, &b, &occ, 0, 8), vec![3], "the unit on n4 lies between n0 and n8");
    assert_eq!(LiveGraph::between(&g, &b, &occ, 0, 1), Vec::<usize>::new());
    assert_eq!(LiveGraph::between(&g, &b, &occ, 0, 5), Vec::<usize>::new(), "no line joins n0 and n5");
}

#[test]
fn the_bare_graph_of_a_kept_graph_is_the_graph_made_from_the_same_units_and_tempo() {
    let b = board();
    let g = LiveGraph::new(Some(1), vec![lu(1, 0, 8), lu(0, 1, 4)]);
    let bare = LiveGraph::bare_part(&g, &b);
    let made = Board::position_from_units(&b, Some(1), vec![(0, 1, 4), (1, 0, 8)]);
    assert_eq!(PositionPart::definitions(&bare, &b), PositionPart::definitions(&made, &b));
    assert_eq!((PositionPart::tempo(&bare), bare.occupant), (Some(1), made.occupant));
}

// ------------------------------------------------------------------------------------------------------------------------------ the history, the keys met, the line walked

fn ch(add: &[&str], rem: &[&str]) -> Change { Change { added: add.iter().map(|r| Fact::new(r, &[])).collect(), removed: rem.iter().map(|r| Fact::new(r, &[])).collect(), keys: vec![] } }

#[test]
fn the_keys_met_are_counted_added_and_removed_one_at_a_time_and_a_key_never_met_is_not_there() {
    let mut s = Seen::new();
    assert!(Seen::is_empty(&s));
    Seen::add(&mut s, 5);
    Seen::add(&mut s, 5);
    Seen::add(&mut s, 6);
    assert_eq!((Seen::len(&s), s.count(5), s.count(6), s.count(7)), (2, 2, 1, 0));
    Seen::remove(&mut s, 5);
    assert_eq!(s.count(5), 1);
    Seen::remove(&mut s, 5);
    assert!(!s.contains(&5) && s.contains(&6));
    Seen::remove(&mut s, 5);
    assert_eq!(Seen::len(&s), 1, "removing a key not met changes nothing");
    let mut keys: Vec<u64> = Seen::keys(&s).collect();
    keys.sort();
    assert_eq!(keys, vec![6]);
    assert!(!Seen::is_empty(&s));
}

#[test]
fn the_line_a_search_walks_is_empty_at_first_and_is_not_after_a_push() {
    let mut p = Path::new();
    assert!(Path::is_empty(&p));
    assert_eq!(Path::pop(&mut p), None);
    Path::push(&mut p, 1);
    assert!(!Path::is_empty(&p) && Path::len(&p) == 1);
}

#[test]
fn a_change_applied_removes_what_it_removed_and_keeps_what_it_added_and_removing_what_is_not_there_does_nothing() {
    let mut set: Set<Fact> = [Fact::new("a", &[]), Fact::new("b", &[])].into_iter().collect();
    apply_change(&mut set, &ch(&["c"], &["a", "zzz"]));
    let left: Vec<String> = set.iter().map(|f| f.to_string()).collect();
    assert_eq!(left, vec!["b".to_string(), "c".to_string()]);
}

#[test]
fn the_changes_of_a_history_are_those_pushed_in_order_and_an_unknown_one_is_marked_not_known() {
    let mut h = History::new(1);
    assert!(History::changes(&h).is_empty() && History::is_empty(&h));
    h.push(ch(&["a"], &[]), 2);
    h.push_unknown(3);
    assert_eq!(History::changes(&h).len(), 2);
    assert_eq!(History::changes(&h)[0], ch(&["a"], &[]));
    assert_eq!((h.change_known(0), h.change_known(1), h.change_known(2)), (true, false, false));
    assert_eq!((History::len(&h), History::is_empty(&h), h.key_at(2), h.key_at(3)), (2, false, Some(3), None));
}

#[test]
fn the_hasher_of_keys_passes_a_whole_key_through_and_folds_bytes_by_rotating_eight_places() {
    let mut h = KeyHasher::default();
    Hasher::write_u64(&mut h, 0xDEAD_BEEF);
    assert_eq!(Hasher::finish(&h), 0xDEAD_BEEF);
    let mut g = KeyHasher::default();
    Hasher::write(&mut g, &[1, 2]);
    assert_eq!(Hasher::finish(&g), 0x0102, "rotate the state left by eight, then fold the byte in");
    assert_eq!(Hasher::finish(&KeyHasher::default()), 0);
}
