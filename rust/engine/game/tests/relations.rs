//! PLAN 79.34 PART C/E: THE RELATIONS THAT FOLLOW FROM HOW THINGS MOVE ARE LAYER 2'S, WRITTEN ONTO LAYER 1'S GRAPH THROUGH ITS DOOR (SYSTEM_SPEC 11 "The cut of layers 1 and 2 is carried into the code"; ledger rows 37, 36; piece: rules).
//! This file names the game of chess on purpose (a test file may). It checks that layer 2 names the relations and makes them (bear, may_move, shut, through, the turn) from what its rules say, that layer 1's graph holds them in slots it does not name, and that their fact form keeps its
//! words and argument order.
use cs_game::graph::Static;
use cs_game::graph_facts::FactReader;
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::relations::{node_words, PartRelations, UnitRelations, BEAR, MAY_MOVE, SHUT, THROUGH};
use cs_game::rules::{Ruled, StateText};
use cs_game::keeper::GraphKeeper;
use cs_game::Game;
use cs_rep::live::LiveGraph;
use std::collections::BTreeSet;
use std::path::PathBuf;

type R = Ruled<KernelGame>;

fn open() -> R {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules");
    Ruled::open(KernelGame::new("chess", 518).unwrap(), "chess", &root, "learned/chess/graph_vocabulary.json").unwrap()
}

#[test]
fn layer_2_names_the_relations_and_layer_1_holds_them_in_slots() {
    let words = node_words();
    assert_eq!(words, vec!["place", "direction", "line", "edge", "kind", "side", "unit", "bear", "may_move", "shut", "through"], "the words of the node types are the ones the layers above read");
    let slots: BTreeSet<u8> = [BEAR, MAY_MOVE, SHUT, THROUGH].iter().map(|r| r.slot).collect();
    assert_eq!(slots.len(), 4, "four relations, four slots");
    assert!(slots.iter().all(|s| (*s as usize) < cs_rep::graph::REL_SLOTS));
    println!("OBS 79.34 E relations: bear, may_move, shut, through are layer 2's values with slots 0 to 3 of layer 1's graph and the node words the layers above read");
}

#[test]
fn a_position_written_by_the_rules_holds_what_the_rules_say_of_each_unit() {
    let g = open();
    let st = g.board().unwrap();
    // a position with a unit of each kind in play: the rules' bearing and moving of each unit equal the relations on the graph, in the order written
    let s = g.read_state_text("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let part = st.position(&g, &s);
    let (mut bears, mut moves) = (0, 0);
    for (ui, u) in part.units.iter().enumerate() {
        let want_bears = g.bearing_from(&s, u.owner, u.kind, u.at);
        let want_moves = g.reach_from(&s, u.owner, u.kind, u.at);
        let got_bears: Vec<u16> = part.bears().iter().filter(|r| r.unit == ui).map(|r| r.place).collect();
        let got_moves: Vec<u16> = part.moves().iter().filter(|r| r.unit == ui).map(|r| r.place).collect();
        assert_eq!(got_bears, want_bears);
        assert_eq!(got_moves, want_moves);
        bears += want_bears.len();
        moves += want_moves.len();
    }
    assert_eq!((part.bears().len(), part.moves().len()), (bears, moves));
    assert!(part.nodes.iter().filter(|n| n.kind.is_relation()).all(|n| ["bear(", "may_move(", "shut(", "through("].iter().any(|w| n.id.starts_with(w))), "each relation node's id begins with the word layer 2 gave it");
    assert_eq!(part.relation_count(), part.bears().len() + part.moves().len() + part.shuts().len() + part.throughs().len());
    // a shut line belongs to the side to move: every shut relation is of a unit of the side marked
    assert!(part.shuts().iter().all(|r| Some(part.units[r.unit].owner) == part.mover));
    // places struck by a side: the places some unit of the side bears on
    let np = st.names.places.len();
    let struck = part.struck_by(np, cs_rep::names::Side::Us);
    let by_hand: BTreeSet<u16> = part.bears().iter().filter(|r| part.units[r.unit].side == cs_rep::names::Side::Us).map(|r| r.place).collect();
    assert_eq!(struck.iter().collect::<BTreeSet<u16>>(), by_hand);
    println!("OBS 79.34 E relations: on a position of the rules, the {bears} bear and {moves} may_move relations on the graph are the rules' own bearing and moving of each unit; each node id begins with layer 2's word; {} shut relations all of the side to move", part.shuts().len());
}

#[test]
fn the_line_through_a_unit_and_the_profile_of_the_kinds_are_layer_2s() {
    let g = open();
    let mut st = g.board().unwrap();
    let s = g.read_state_text("4k3/8/8/8/8/2n5/8/R3K2R w KQ - 0 1").unwrap();
    let part = st.position(&g, &s);
    assert!(st.cont.is_empty() && part.throughs().is_empty(), "before a profile is learned no line through is made");
    st.learn(&part);
    assert!(!st.cont.is_empty());
    // a unit that bears two places away along a line continues along it: the corner unit on a1 along the first row, the one on h1 the same way back
    assert!(st.cont.iter().any(|c| *c));
    // a rook on a1 with a black unit on a4 and the other side's unit on a7 behind it: the line goes through the first to the second
    let s2 = g.read_state_text("4k3/n7/8/8/p7/8/8/R3K3 w - - 0 1").unwrap();
    let part2 = st.position(&g, &s2);
    let rook = part2.units.iter().position(|u| u.at == g.place_names().iter().position(|n| n == "a1").unwrap() as u16).unwrap();
    let a7 = g.place_names().iter().position(|n| n == "a7").unwrap() as u16;
    assert!(part2.throughs().iter().any(|r| r.unit == rook && r.place == a7 && r.between.len() == 1), "the line from the unit on a1 runs through the unit on a4 to the one on a7");
    println!("OBS 79.34 E relations: the profile of the kinds is learned in layer 2 (`Static::learn`) and the line through a unit is written by `Static::derive_throughs` through layer 1's door");
}

#[test]
fn the_graph_kept_in_layer_1_and_written_by_layer_2_is_the_graph_made_afresh_and_the_slots_hold_what_was_written() {
    let g = open();
    let st = g.board().unwrap();
    let mut s = g.initial();
    let mut keeper = GraphKeeper::new(&st, &g, &s);
    for uci in ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"] {
        let m = g.moves(&s).into_iter().find(|m| g.move_name(&s, m) == uci).unwrap();
        s = g.apply(&s, &m);
        keeper.make(&st, &g, &s);
        assert_eq!(keeper.part(&st).definitions(st.board()), st.position(&g, &s).definitions(st.board()), "after {uci}");
    }
    // layer 1 keeps the places layer 2 wrote for each unit in slots; layer 2 reads them by name
    let live: &LiveGraph = keeper.live();
    for u in live.units() {
        assert_eq!(u.bears(), &g.bearing_from(&s, u.owner, u.kind, u.at));
        assert_eq!(u.reach(), &g.reach_from(&s, u.owner, u.kind, u.at));
        assert_eq!(u.slots[BEAR.slot as usize].len(), u.bears().len());
    }
    // and the bare graph layer 1 gives holds no relation at all
    assert_eq!(live.bare_part(st.board()).relation_count(), 0);
    while keeper.undo() {}
    assert_eq!(keeper.part(&st).definitions(st.board()), st.position(&g, &g.initial()).definitions(st.board()), "taken back to the start");
    println!("OBS 79.34 E relations: six moves kept and taken back; the graph kept in layer 1's slots and written by layer 2 equals the graph made afresh at every step; layer 1's bare graph holds no relation");
}

#[test]
fn the_facts_of_the_graph_keep_their_words_and_their_argument_order() {
    let g = open();
    let st = g.board().unwrap();
    let s = g.read_state_text("4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1").unwrap();
    let keeper = GraphKeeper::new(&st, &g, &s);
    let reader = FactReader::new(st.board(), None);
    let rows = reader.facts(keeper.live());
    let has = |rel: &str, args: &[&str]| rows.iter().any(|f| f.relation == rel && f.args.iter().map(|a| a.as_str()).eq(args.iter().copied()));
    assert!(has("tempo", &["black"]), "whose turn it is: one argument, the side");
    assert!(has("unit", &["white", "pawn", "e4"]) && has("unit", &["black", "pawn", "d5"]), "a unit: side, kind, place");
    assert!(has("bear", &["white", "pawn", "e4", "d5"]) && has("bear", &["black", "pawn", "d5", "e4"]), "bears: side, kind, place, target");
    assert!(has("may_move", &["white", "pawn", "e4", "e5"]) && has("may_move", &["black", "pawn", "d5", "d4"]), "may move: side, kind, place, to");
    assert!(rows.iter().any(|f| f.relation == "on_line" && f.args[0] == "a1" && f.args[1].starts_with("line:")), "a place lies on a line (layer 1's word)");
    assert!(rows.iter().any(|f| f.relation == "on_edge"));
    // a reader made for a pattern makes only the rows some literal of it unifies with
    let some = cs_rep::lemma::Lit::pos("bear", &["?o", "?k", "?p", "?t"]);
    let narrow = FactReader::new(st.board(), Some(&[some])).facts(keeper.live());
    assert!(narrow.iter().all(|f| f.relation == "bear") && !narrow.is_empty());
    println!("OBS 79.34 E relations: the facts unit, bear, may_move and tempo keep their words and argument order ({} rows); a reader narrowed to a pattern makes {} rows, all bear", rows.len(), narrow.len());
}
