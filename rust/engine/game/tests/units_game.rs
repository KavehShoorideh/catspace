//! PLAN 79.36 PART C, LAYER 2, THE UNITS OF CODE: the game trait and its defaults, the kernel, the description as a game, the writing of the rules onto layer 1's graph, the keepers, the tags, the declaration, the generator (SYSTEM_SPEC 11, last block; piece: rules; ledger row 43).
//! Every test states what one item answers on a small case written by hand, and at its edges. A test file may name the games.
use cs_game::gdl_game::{parse_term, render_term, row_args, FactView, GdlGame};
use cs_game::graph::{Root, Static};
use cs_game::keeper::GraphKeeper;
use cs_game::kernel::{KernelGame, CLOCK_LIMIT, VARIANTS};
use cs_game::placed::{NoStep, Placed};
use cs_game::relations::{self, PartRelations, UnitRelations};
use cs_game::rng::Rng;
use cs_game::rules::{Declaration, Ruled, Rules, StateText};
use cs_game::tags::TagKeeper;
use cs_game::{units_changed, Fact, Game, UnitDecl, Vocabulary};
use cs_gdl::{Program, ARITY_MAX, FREE};
use cs_rep::live::LiveUnit;
use cs_rep::statement::Change;
use cs_rep::store::{GameKey, GameMemory};
use std::path::PathBuf;

type R = Ruled<KernelGame>;

fn rules_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules") }
fn open(variant: &str) -> R { Ruled::open(KernelGame::new(variant, 518).unwrap(), variant, &rules_root(), "learned/chess/graph_vocabulary.json").unwrap() }
fn chess() -> R { open("chess") }
fn declaration_text() -> String { std::fs::read_to_string(rules_root().join("learned/chess/graph_vocabulary.json")).unwrap() }
fn description() -> String { std::fs::read_to_string(rules_root().join("chess.gdl")).unwrap() }
fn from_fen(fen: &str) -> (KernelGame, <KernelGame as Game>::State) { let k = KernelGame::from_fen("chess", fen).unwrap(); let s = k.initial(); (k, s) }
fn play(g: &R, s: &<R as Game>::State, uci: &str) -> <R as Game>::State { let m = Game::find_move(g, s, uci).unwrap_or_else(|| panic!("{uci} is not a move")); Game::apply(g, s, &m) }
fn place(g: &R, text: &str) -> u16 { Placed::place_names(g).iter().position(|n| n == text).unwrap() as u16 }
fn kind(g: &R, text: &str) -> u8 { Placed::kind_names(g).iter().position(|n| n == text).unwrap() as u8 }
fn state(g: &R, fen: &str) -> <R as Game>::State { StateText::read_state_text(g, fen).unwrap() }

// ------------------------------------------------------------------------------------------------------------------------------ the generator

#[test]
fn the_generator_gives_the_stream_of_splitmix64_from_seed_nought() {
    let mut r = Rng::new(0);
    assert_eq!(Rng::next_u64(&mut r), 0xe220a8397b1dcdaf);
    assert_eq!(Rng::next_u64(&mut r), 0x6e789e6aa1b965f4);
    assert_eq!(Rng::next_u64(&mut r), 0x06c45d188009454f);
}

#[test]
fn two_generators_of_one_seed_give_one_stream_and_two_seeds_give_two() {
    let (mut a, mut b, mut c) = (Rng::new(7), Rng::new(7), Rng::new(8));
    let sa: Vec<u64> = (0..5).map(|_| Rng::next_u64(&mut a)).collect();
    assert_eq!(sa, (0..5).map(|_| Rng::next_u64(&mut b)).collect::<Vec<_>>());
    assert_ne!(sa, (0..5).map(|_| Rng::next_u64(&mut c)).collect::<Vec<_>>());
}

#[test]
fn an_index_below_a_bound_is_the_next_number_modulo_the_bound_and_one_below_one_is_nought() {
    let mut r = Rng::new(0);
    assert_eq!(Rng::below(&mut r, 10), (0xe220a8397b1dcdafu64 % 10) as usize);
    assert_eq!(Rng::below(&mut r, 1), 0);
    let mut q = Rng::new(3);
    assert!((0..200).all(|_| Rng::below(&mut q, 7) < 7));
}

// ------------------------------------------------------------------------------------------------------------------------------ the kernel

#[test]
fn the_kernel_plays_nine_variants_by_their_names_and_a_clock_of_a_hundred() {
    assert_eq!(VARIANTS, ["chess", "chess960", "crazyhouse", "atomic", "antichess", "kingofthehill", "3check", "racingkings", "horde"]);
    assert_eq!(CLOCK_LIMIT, 100);
    for v in VARIANTS { assert!(KernelGame::new(v, 518).is_ok(), "{v}"); }
    assert_eq!(KernelGame::new("nonsense", 0).err(), Some("no variant nonsense".to_string()));
}

#[test]
fn the_start_of_the_standard_game_is_written_as_its_text_form_and_the_game_has_its_name() {
    let k = KernelGame::new("chess", 0).unwrap();
    assert_eq!(KernelGame::fen(&k, &k.initial()), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    assert_eq!(Game::name(&k), "chess");
    assert_eq!(Game::name(&KernelGame::new("atomic", 0).unwrap()), "atomic");
    assert_eq!(Game::roles(&k), vec!["white".to_string(), "black".to_string()]);
}

#[test]
fn the_pass_is_the_same_board_with_the_other_side_to_act_and_none_when_the_side_is_struck_or_the_game_is_over() {
    let (k, s) = from_fen("4k3/8/8/8/8/8/4K3/R7 w - - 0 1");
    let passed = Game::pass(&k, &s).unwrap();
    assert_eq!(KernelGame::fen(&k, &passed), "4k3/8/8/8/8/8/4K3/R7 b - - 0 1");
    let (k2, struck) = from_fen("4k3/8/8/8/8/8/8/4RK2 b - - 0 1");
    assert!(Game::pass(&k2, &struck).is_none(), "the side to move is struck: it cannot pass");
    let (k3, over) = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assert!(Game::outcome(&k3, &over).is_some());
    assert!(Game::pass(&k3, &over).is_none(), "a game that is over has no pass");
}

#[test]
fn a_move_that_turns_a_unit_into_another_kind_is_listed_as_the_change_and_a_plain_move_changes_nothing() {
    let (k, s) = from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    let promote = Game::find_move(&k, &s, "a7a8q").unwrap();
    assert_eq!(Game::changed(&k, &s, &promote), vec![("p".to_string(), "q".to_string())]);
    let plain = Game::find_move(&k, &s, "e1e2").unwrap();
    assert!(Game::changed(&k, &s, &plain).is_empty());
}

#[test]
fn the_state_as_the_descriptions_fluents_is_the_units_the_turn_the_rights_the_passing_place_and_the_clock_sorted() {
    let k = KernelGame::new("chess", 0).unwrap();
    let facts = KernelGame::state_facts(&k, &k.initial());
    assert_eq!(facts.len(), 32 + 1 + 4 + 1, "thirty-two units, the turn, four rights, the clock");
    assert!(facts.windows(2).all(|w| w[0] < w[1]), "sorted and distinct");
    assert!(facts.contains(&Fact::new("cell", &["e", "1", "white", "king"])));
    assert!(facts.contains(&Fact::new("control", &["white"])));
    assert!(facts.contains(&Fact::new("castling_right", &["black", "queenside"])));
    assert!(facts.contains(&Fact::new("halfmove", &["0"])));
}

#[test]
fn the_state_facts_say_the_passing_place_after_a_two_place_advance_and_the_clock_stops_at_the_limit() {
    let (k, s) = from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    let after = k.apply(&s, &Game::find_move(&k, &s, "e2e4").unwrap());
    let facts = KernelGame::state_facts(&k, &after);
    assert!(facts.contains(&Fact::new("ep_square", &["e", "3"])));
    assert!(facts.contains(&Fact::new("control", &["black"])));
    let (k2, old) = from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 150 80");
    assert!(KernelGame::state_facts(&k2, &old).contains(&Fact::new("halfmove", &["100"])));
}

// ------------------------------------------------------------------------------------------------------------------------------ the game trait, its defaults, held on a toy game of units

/// A toy game of units only: the state is the list of (owner, kind); the mover is side 0 for ever; move 0 takes the first unit of the other side, move 1 turns the first own unit "a" into "b", move 2 places a new own unit "c".
struct Toy;
impl Game for Toy {
    type State = Vec<(usize, String)>;
    type Move = u8;
    fn name(&self) -> String { "toy".into() }
    fn roles(&self) -> Vec<String> { vec!["us".into(), "them".into()] }
    fn initial(&self) -> Self::State { vec![(0, "a".into()), (1, "x".into()), (1, "y".into())] }
    fn mover(&self, _s: &Self::State) -> Option<usize> { Some(0) }
    fn moves(&self, _s: &Self::State) -> Vec<u8> { vec![0, 1, 2] }
    fn apply(&self, s: &Self::State, m: &u8) -> Self::State {
        let mut out = s.clone();
        match m {
            0 => { if let Some(i) = out.iter().position(|(o, _)| *o == 1) { out.remove(i); } }
            1 => { if let Some(u) = out.iter_mut().find(|(o, k)| *o == 0 && k == "a") { u.1 = "b".into(); } }
            _ => out.push((0, "c".into())),
        }
        out
    }
    fn outcome(&self, _s: &Self::State) -> Option<Vec<i64>> { None }
    fn move_name(&self, _s: &Self::State, m: &u8) -> String { format!("m{m}") }
    fn key(&self, s: &Self::State) -> u64 { s.len() as u64 }
    fn units(&self, s: &Self::State) -> Vec<(usize, String)> { s.clone() }
}

#[test]
fn a_game_that_says_nothing_of_observation_pass_or_the_other_side_gets_the_defaults() {
    let g = Toy;
    let s = g.initial();
    assert_eq!(Game::observe(&g, &s, 1), s, "perfect information: a side sees the state itself");
    assert_eq!(Game::pass(&g, &s), None);
    assert_eq!((Game::other(&g, 0), Game::other(&g, 1)), (1, 0));
    assert_eq!(Game::name(&g), "toy");
}

#[test]
fn the_default_of_what_a_move_takes_is_the_units_of_the_other_sides_that_disappear() {
    let g = Toy;
    let s = g.initial();
    assert_eq!(Game::captured(&g, &s, &0), vec![(1, "x".to_string())]);
    assert!(Game::captured(&g, &s, &1).is_empty(), "the mover's own change is no loss");
    assert!(Game::captured(&g, &s, &2).is_empty(), "a placement takes nothing");
    assert_eq!(Game::taken_at(&g, &s, &0), vec![(1, "x".to_string(), String::new())], "no place: the empty text");
}

#[test]
fn the_default_of_a_change_of_kind_pairs_the_unit_that_left_with_the_one_that_came_and_a_placement_comes_from_nothing() {
    let g = Toy;
    let s = g.initial();
    assert_eq!(Game::changed(&g, &s, &1), vec![("a".to_string(), "b".to_string())]);
    assert_eq!(Game::changed(&g, &s, &2), vec![(String::new(), "c".to_string())]);
    assert!(Game::changed(&g, &s, &0).is_empty());
    assert_eq!(units_changed(&g, &s, &1), vec![("a".to_string(), "b".to_string())]);
    assert_eq!(units_changed(&g, &s, &2), vec![(String::new(), "c".to_string())]);
    assert!(units_changed(&g, &s, &0).is_empty());
}

#[test]
fn a_move_is_found_by_its_name_and_a_name_no_move_has_finds_none() {
    let g = Toy;
    let s = g.initial();
    assert_eq!(Game::find_move(&g, &s, "m1"), Some(1));
    assert_eq!(Game::find_move(&g, &s, "m9"), None);
    assert!(Game::base_facts(&g, &s).is_empty(), "a game that gives no facts gives none");
}

// ------------------------------------------------------------------------------------------------------------------------------ the description as a game

fn gdl() -> GdlGame {
    let vocab = Vocabulary { derived: vec![], delta: vec!["cell".into()], units: Some(UnitDecl { relation: "cell".into(), owner: 2, kind: Some(3) }) };
    GdlGame::load("described", &description()).unwrap().with_vocabulary(vocab).unwrap()
}

#[test]
fn a_description_is_a_game_with_its_name_its_roles_and_its_start_and_a_text_without_roles_is_refused() {
    let g = gdl();
    assert_eq!(Game::name(&g), "described");
    assert_eq!(Game::roles(&g), vec!["white".to_string(), "black".to_string()]);
    assert_eq!(Game::initial(&g).len() >= 32, true);
    assert!(GdlGame::load("x", "(sort s a b)").err().unwrap().contains("role"));
}

#[test]
fn the_units_of_a_described_state_are_listed_with_their_owner_kind_and_place_joined_by_a_colon() {
    let g = gdl();
    let units = GdlGame::units_placed(&g, &g.initial());
    assert_eq!(units.len(), 32);
    assert_eq!(units.iter().filter(|u| u.0 == 0).count(), 16);
    assert!(units.contains(&(0, "king".to_string(), "e:1".to_string())));
    assert!(units.contains(&(1, "pawn".to_string(), "h:7".to_string())));
    let bare = GdlGame::load("bare", &description()).unwrap();
    assert!(GdlGame::units_placed(&bare, &bare.initial()).is_empty(), "no vocabulary: no units are known");
}

#[test]
fn a_vocabulary_naming_a_fluent_the_description_does_not_have_is_refused() {
    let v = Vocabulary { derived: vec![], delta: vec![], units: Some(UnitDecl { relation: "nothing_of_the_kind".into(), owner: 0, kind: None }) };
    assert!(GdlGame::load("x", &description()).unwrap().with_vocabulary(v).err().unwrap().contains("no fluent"));
}

#[test]
fn the_key_of_a_described_state_is_equal_for_equal_states_and_differs_after_a_move() {
    let g = gdl();
    let s = g.initial();
    let m = Game::moves(&g, &s).into_iter().next().unwrap();
    let after = Game::apply(&g, &s, &m);
    assert_eq!(Game::key(&g, &s), Game::key(&g, &g.initial()));
    assert_ne!(Game::key(&g, &s), Game::key(&g, &after));
}

#[test]
fn a_ground_term_is_written_as_its_constant_or_as_a_functor_with_its_arguments_and_read_back() {
    let mut p = Program::load(&description()).unwrap();
    let t = parse_term(&mut p, "(move e 2 e 4)").unwrap();
    assert_eq!(render_term(&p, t), "(move e 2 e 4)");
    let nested = parse_term(&mut p, "(f a (g b))").unwrap();
    assert_eq!(render_term(&p, nested), "(f a (g b))");
    let atom = parse_term(&mut p, "white").unwrap();
    assert_eq!(render_term(&p, atom), "white");
    let g = gdl();
    let first = Game::moves(&g, &g.initial())[0];
    assert!(GdlGame::render(&g, first).starts_with("(move "), "a move of the described game, written as its term");
    assert_eq!(GdlGame::render(&g, first), Game::move_name(&g, &g.initial(), &first));
    assert!(parse_term(&mut p, "(f a").is_err());
    assert!(parse_term(&mut p, "a b").unwrap_err().contains("trailing"));
}

#[test]
fn the_arguments_of_a_row_run_up_to_its_first_free_place() {
    let mut p = Program::load(&description()).unwrap();
    let (a, b) = (parse_term(&mut p, "a").unwrap(), parse_term(&mut p, "b").unwrap());
    let mut row = [FREE; ARITY_MAX];
    assert!(row_args(&row).is_empty());
    row[0] = a;
    row[1] = b;
    assert_eq!(row_args(&row), vec![a, b]);
    row[3] = a;
    assert_eq!(row_args(&row), vec![a, b], "what follows a free place is not an argument");
}

#[test]
fn the_fact_view_keeps_the_vocabulary_it_was_made_with() {
    let vocab = Vocabulary { derived: vec!["attack".into()], delta: vec!["cell".into()], units: None };
    let view = FactView::new(&description(), vocab.clone()).unwrap();
    assert_eq!(FactView::vocabulary(&view), &vocab);
    assert!(FactView::new(&description(), Vocabulary { derived: vec!["no_such_relation".into()], delta: vec![], units: None }).err().unwrap().contains("no relation"));
}

// ------------------------------------------------------------------------------------------------------------------------------ the declaration and the game with its rules

#[test]
fn a_declaration_is_read_from_its_text_and_each_missing_part_is_refused_with_its_name() {
    let d = Declaration::from_json(&declaration_text()).unwrap();
    assert_eq!((d.fact_decl.units.relation.as_str(), d.fact_decl.units.owner, d.fact_decl.units.kind), ("cell", 2, Some(3)));
    assert_eq!(d.fact_decl.tempo.as_deref(), Some("control"));
    assert_eq!(d.static_relations, vec!["diagonal_of".to_string(), "step".to_string()]);
    assert_eq!(d.objects, vec![("square".to_string(), vec!["file".to_string(), "rank".to_string()])]);
    assert_eq!(Declaration::from_json("{}").unwrap_err(), "the declaration has no fact_decl");
    assert_eq!(Declaration::from_json(r#"{"fact_decl": {}}"#).unwrap_err(), "fact_decl has no units");
    assert_eq!(Declaration::from_json(r#"{"fact_decl": {"units": {"owner": 0}}}"#).unwrap_err(), "units has no relation");
    assert_eq!(Declaration::from_json(r#"{"fact_decl": {"units": {"relation": "u"}}}"#).unwrap_err(), "units has no owner");
    assert!(Declaration::from_json("not json").is_err());
    let bare = Declaration::from_json(r#"{"fact_decl": {"units": {"relation": "u", "owner": 1}}}"#).unwrap();
    assert_eq!((bare.fact_decl.units.kind, bare.fact_decl.tempo, bare.static_relations.len(), bare.objects.len()), (None, None, 0, 0));
}

#[test]
fn a_game_made_from_its_text_and_declaration_holds_the_text_and_the_address_of_it_and_a_text_that_is_no_description_is_refused() {
    let text = description();
    let g = Ruled::new(KernelGame::new("chess", 0).unwrap(), "chess", &text, Declaration::from_json(&declaration_text()).unwrap()).unwrap();
    assert_eq!(g.description(), text);
    assert_eq!(Rules::rules_key(&g), GameKey::of("chess", &text));
    assert_eq!(Rules::repetition_limit(&g), Some(2));
    assert!(Ruled::new(KernelGame::new("chess", 0).unwrap(), "x", "(((", Declaration::from_json(&declaration_text()).unwrap()).is_err());
}

#[test]
fn a_game_with_its_rules_delegates_the_name_the_view_of_a_side_the_pass_the_other_side_and_the_change_of_kind_to_the_game() {
    let g = chess();
    let s = Game::initial(&g);
    assert_eq!(Game::name(&g), "chess");
    assert_eq!(Game::key(&g, &Game::observe(&g, &s, 1)), Game::key(&g, &s));
    assert_eq!(Game::key(&g, &Game::pass(&g, &s).unwrap()), Game::key(g.game(), &g.game().pass(&s).unwrap()));
    assert_eq!((Game::other(&g, 0), Game::other(&g, 1)), (1, 0));
    let s2 = state(&g, "4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    let m = Game::find_move(&g, &s2, "a7a8n").unwrap();
    assert_eq!(Game::changed(&g, &s2, &m), vec![("p".to_string(), "n".to_string())]);
}

// ------------------------------------------------------------------------------------------------------------------------------ the words of the relations and what is said of one unit as kept

#[test]
fn the_relations_have_their_slots_and_words_and_the_facts_of_the_graph_say_the_same_words() {
    assert_eq!((relations::BEAR.slot, relations::BEAR.word), (0, "bear"));
    assert_eq!((relations::MAY_MOVE.slot, relations::MAY_MOVE.word), (1, "may_move"));
    assert_eq!((cs_game::graph_facts::BEAR, cs_game::graph_facts::MAY_MOVE, cs_game::graph_facts::TEMPO), ("bear", "may_move", "tempo"));
    assert_eq!((relations::SHUT.slot, relations::THROUGH.slot), (2, 3));
}

#[test]
fn what_is_said_of_a_unit_as_kept_is_what_stands_in_its_first_three_slots() {
    let mut u = LiveUnit::new(0, 1, 2);
    u.slots[0] = vec![5, 6];
    u.slots[1] = vec![7];
    u.slots[2] = vec![8, 9, 10];
    assert_eq!(UnitRelations::bears(&u), &vec![5, 6]);
    assert_eq!(UnitRelations::reach(&u), &vec![7]);
    assert_eq!(UnitRelations::shuts(&u), &vec![8, 9, 10]);
    let empty = LiveUnit::new(0, 0, 0);
    assert!(UnitRelations::bears(&empty).is_empty() && UnitRelations::reach(&empty).is_empty() && UnitRelations::shuts(&empty).is_empty());
}

#[test]
fn a_reason_a_step_is_refused_has_a_word() {
    let words: Vec<&str> = [NoStep::UnitGone, NoStep::PlaceTaken, NoStep::LineShut, NoStep::MayNotMove].iter().map(|n| NoStep::word(*n)).collect();
    assert_eq!(words, vec!["unit_gone", "place_taken", "line_shut", "may_not_move"]);
}

// ------------------------------------------------------------------------------------------------------------------------------ the writing of the rules onto layer 1's graph

/// white: the leader on h1, a rook on d4; black: the leader on h8, a smallest unit on d6, a knight on d8 (white to act)
const SHUT_FEN: &str = "3n3k/8/3p4/8/3R4/8/8/7K w - - 0 1";

#[test]
fn the_steps_of_a_game_are_read_from_its_description_by_the_names_its_declaration_gives_and_a_wrong_name_is_refused() {
    let g = chess();
    let d = Rules::declaration(&g);
    let steps = Placed::steps(g.game(), g.description(), &d.static_relations, &d.objects).unwrap();
    assert_eq!(steps.directions, g.board().unwrap().directions);
    assert!(!steps.rows.is_empty());
    let none = Placed::steps(g.game(), g.description(), &["nothing".to_string()], &d.objects);
    assert!(none.unwrap_err().contains("no static relation is a step relation"));
}

#[test]
fn a_static_part_made_from_a_board_layer_1_holds_is_the_same_board_and_has_no_profile_of_the_kinds() {
    let st = chess().board().unwrap();
    let again = Static::from_board(st.board().clone());
    assert_eq!(again.directions, st.directions);
    assert_eq!(again.lines.len(), st.lines.len());
    assert!(again.cont.is_empty());
    assert!(!Static::continues(&again, 0, 0, true), "until a profile is learned no kind continues");
}

#[test]
fn the_rules_written_onto_a_graph_give_each_unit_what_it_bears_on_and_where_it_may_move() {
    let g = chess();
    let st = g.board().unwrap();
    let s = state(&g, "7k/8/8/8/3R4/8/8/7K w - - 0 1");
    let mut part = st.board().position_from_units(Game::mover(&g, &s), Placed::cells(&g, &s));
    assert_eq!(part.relation_count(), 0, "layer 1 made the units and no relation");
    Static::write_rules(&st, &g, &s, &mut part);
    let rook = part.units.iter().position(|u| u.kind == kind(&g, "rook")).unwrap();
    assert_eq!(part.bears().iter().filter(|r| r.unit == rook).count(), 14, "a rook alone bears on fourteen places");
    assert_eq!(part.moves().iter().filter(|r| r.unit == rook).count(), 14);
    assert!(part.throughs().is_empty(), "no profile, no lines through");
    assert_eq!(part.definitions(st.board()), st.position(&g, &s).definitions(st.board()), "the same as the graph made in one go");
}

#[test]
fn the_graph_of_a_state_for_a_decision_is_the_roots_own_when_the_state_is_the_roots_and_another_one_made_otherwise() {
    let g = chess();
    let st = g.board().unwrap();
    let s = Game::initial(&g);
    let root = Root::of(&st, &g, &s);
    assert_eq!(Root::key(&root), Game::key(&g, &s));
    let same = Static::position_in(&st, &root, &g, &s);
    assert!(std::rc::Rc::ptr_eq(&same, Root::part(&root)), "the root's graph is lent, not made again");
    let after = play(&g, &s, "e2e4");
    let other = Static::position_in(&st, &root, &g, &after);
    assert!(!std::rc::Rc::ptr_eq(&other, Root::part(&root)));
    assert_eq!(other.definitions(st.board()), st.position(&g, &after).definitions(st.board()));
    assert_ne!(other.definitions(st.board()), Root::part(&root).definitions(st.board()));
}

#[test]
fn a_root_made_from_a_graph_the_memory_kept_holds_that_graph_and_its_key() {
    let g = chess();
    let st = g.board().unwrap();
    let s = Game::initial(&g);
    let part = st.position(&g, &s);
    let nodes = part.nodes.len();
    let root = Root::from_part(99, part);
    assert_eq!(Root::key(&root), 99);
    assert_eq!(Root::part(&root).nodes.len(), nodes);
}

#[test]
fn the_profile_of_the_kinds_is_learned_from_what_bears_two_or_more_places_away_and_only_for_those_kinds_and_lines() {
    let g = chess();
    let mut st = g.board().unwrap();
    let s = state(&g, "7k/8/8/8/3R4/8/8/7K w - - 0 1");
    let part = st.position(&g, &s);
    Static::learn(&mut st, &part);
    let file = st.lines.iter().position(|l| l.places.len() == 8 && l.places.contains(&place(&g, "d4")) && l.places.contains(&place(&g, "d1"))).unwrap() as u32;
    let rook = kind(&g, "rook");
    assert!(Static::continues(&st, rook, file, true) && Static::continues(&st, rook, file, false), "a rook bears far along a file in both headings");
    assert!(!Static::continues(&st, kind(&g, "bishop"), file, true), "a bishop was never seen to bear along a file");
    assert!(!Static::continues(&st, kind(&g, "knight"), file, false));
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn asking_whether_a_kind_continues_along_a_line_that_is_not_on_the_board_is_a_caller_error_and_panics() {
    let st = chess().board().unwrap();
    let _ = Static::continues(&st, 0, 10_000, true);
}

#[test]
fn the_line_through_a_unit_is_the_next_unit_beyond_the_one_a_unit_bears_on_and_only_once_a_profile_is_learned() {
    let g = chess();
    let mut st = g.board().unwrap();
    let s = state(&g, SHUT_FEN);
    let mut part = st.position(&g, &s);
    Static::derive_throughs(&st, &mut part);
    assert!(part.throughs().is_empty(), "no profile, nothing is made");
    Static::learn(&mut st, &part);
    Static::derive_throughs(&st, &mut part);
    let rook = part.units.iter().position(|u| u.kind == kind(&g, "rook")).unwrap();
    let got: Vec<(usize, u16)> = part.throughs().iter().map(|r| (r.unit, r.place)).collect();
    assert_eq!(got, vec![(rook, place(&g, "d8"))], "the rook bears on d6 and the next unit beyond it, on d8, is the place of the relation");
    assert_eq!(part.throughs()[0].between, vec![part.occupant[place(&g, "d6") as usize] as usize - 1]);
}

#[test]
fn a_graph_kept_by_layer_1_is_written_as_the_graph_made_in_one_go_by_the_part_of() {
    let g = chess();
    let st = g.board().unwrap();
    let s = play(&g, &Game::initial(&g), "g1f3");
    let keeper = GraphKeeper::new(&st, &g, &s);
    let from_live = Static::part_of(&st, GraphKeeper::live(&keeper));
    assert_eq!(from_live.definitions(st.board()), st.position(&g, &s).definitions(st.board()));
}

#[test]
fn the_places_beyond_the_far_end_of_what_a_unit_bears_on_are_shut_where_the_game_says_the_line_is_shut() {
    let g = chess();
    let st = g.board().unwrap();
    let s = state(&g, SHUT_FEN);
    let d4 = place(&g, "d4");
    let bears = Placed::bearing_from(&g, &s, 0, kind(&g, "rook"), d4);
    let mut shut = Static::shut_places(&st, &g, &s, d4, &bears);
    shut.sort();
    let mut want = vec![place(&g, "d7"), place(&g, "d8")];
    want.sort();
    assert_eq!(shut, want, "behind the unit on d6 the rook's file is shut at d7 and d8; its other lines end on the board");
    assert!(Static::shut_places(&st, &g, &s, d4, &[]).is_empty(), "a unit that bears on nothing shuts nothing");
}

#[test]
fn the_lines_shut_by_a_unit_that_does_not_move_are_read_with_the_blocker_taken_off_and_the_places_beyond_it_are_shut() {
    // black to be struck: the black rook on d8, white's unit on d4 stops it; with that unit off the rook would bear on d3, d2, d1
    let g = chess();
    let st = g.board().unwrap();
    let s = state(&g, "3r3k/8/8/8/3P4/8/8/7K w - - 0 1");
    let mut part = st.position(&g, &s);
    let rook = part.units.iter().position(|u| u.kind == kind(&g, "rook")).unwrap();
    assert_eq!(part.shuts().iter().filter(|r| r.unit == rook).count(), 0, "the side to move's lines only are written by the graph");
    Static::shut_lines_other(&st, &g, &s, 0, &mut part, 0);
    let mut shut: Vec<u16> = part.shuts().iter().filter(|r| r.unit == rook).map(|r| r.place).collect();
    shut.sort();
    let mut want = vec![place(&g, "d1"), place(&g, "d2"), place(&g, "d3")];
    want.sort();
    assert_eq!(shut, want);
    let mut one = st.position(&g, &s);
    Static::first_blocker_shuts(&st, &g, &s, rook, &mut one, 0);
    assert_eq!(one.shuts().iter().filter(|r| r.unit == rook).count(), 3, "the same for one unit asked alone");
}

// ------------------------------------------------------------------------------------------------------------------------------ the keepers

#[test]
fn a_kept_graph_taken_out_and_put_back_is_the_same_graph_and_counts_what_it_asked_of_the_rules() {
    let g = chess();
    let st = g.board().unwrap();
    let s = Game::initial(&g);
    let keeper = GraphKeeper::new(&st, &g, &s);
    assert_eq!((GraphKeeper::asked(&keeper), GraphKeeper::depth(&keeper)), ((32, 32), 0), "a first position asks the rules about every unit once");
    let live = GraphKeeper::into_live(keeper);
    assert_eq!(live.units().len(), 32);
    let mut again = GraphKeeper::from_live(live);
    assert_eq!(GraphKeeper::asked(&again), (32, 32));
    let after = play(&g, &s, "e2e4");
    again.make(&st, &g, &after);
    assert_eq!(GraphKeeper::depth(&again), 1);
    let (asked, all) = GraphKeeper::asked(&again);
    assert_eq!(all, 32);
    assert!(asked >= 1 && asked <= 32, "at least the moved unit is asked again");
    assert!(again.undo());
    assert_eq!(GraphKeeper::depth(&again), 0);
}

fn start_keeper(g: &R) -> (TagKeeper, <R as Game>::State) {
    let s = Game::initial(g);
    (TagKeeper::new(g, &s, &Rules::fact_decl(g), GameMemory::new(Rules::rules_key(g))), s)
}

#[test]
fn the_tags_of_the_first_position_are_the_rights_and_the_clock_each_a_statement_of_the_memory_and_the_repeat_tag_counts_earlier_occurrences() {
    let g = chess();
    let (k, _s) = start_keeper(&g);
    assert_eq!(k.tags().len(), 4 + 1, "four rights and the clock");
    assert_eq!(TagKeeper::memory(&k).statements().count(), 5, "each tag is a statement");
    assert_eq!(TagKeeper::repeat_tag(&k), Fact::new("repeats", &["0"]));
}

#[test]
fn a_position_met_again_is_a_repeat_the_tag_says_how_many_times_and_taking_a_tempo_back_takes_the_count_back() {
    let g = chess();
    let (mut k, mut s) = start_keeper(&g);
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        s = play(&g, &s, uci);
        k.make(&g, &s);
    }
    assert_eq!(TagKeeper::repeat_tag(&k), Fact::new("repeats", &["1"]), "the start, again");
    assert!(k.undo());
    assert_eq!(TagKeeper::repeat_tag(&k), Fact::new("repeats", &["0"]));
}

#[test]
fn a_keeper_taken_apart_and_resumed_holds_the_same_facts_tags_and_history() {
    let g = chess();
    let (mut k, s) = start_keeper(&g);
    let after = play(&g, &s, "e2e4");
    k.make(&g, &after);
    let tags = k.tags();
    let (facts, history, memory) = TagKeeper::into_parts(k);
    assert_eq!(history.len(), 1);
    let back = TagKeeper::resume(&Rules::fact_decl(&g), facts.clone(), history, memory);
    assert_eq!(back.tags(), tags);
    assert_eq!(back.facts(), &facts);
    assert_eq!(back.history().len(), 1);
    assert_eq!(back.history().current_key(), Game::key(&g, &after));
}

#[test]
fn a_keeper_for_a_position_reached_with_no_changes_seen_holds_the_keys_as_not_known_and_states_the_tags_again() {
    let g = chess();
    let s = state(&g, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    let mut memory = GameMemory::new(Rules::rules_key(&g));
    let stale = cs_rep::store::position_statement("tag:old", &[Fact::new("old", &[])], "none");
    memory.add(stale, "from an earlier history").unwrap();
    let k = TagKeeper::over(&g, &s, &Rules::fact_decl(&g), memory, &[11, 12]);
    assert_eq!(k.history().keys().to_vec(), vec![11, 12, Game::key(&g, &s)]);
    assert!(!k.history().change_known(0) && !k.history().change_known(1));
    assert!(TagKeeper::memory(&k).statement("tag:old").is_none(), "an old tag of the memory is retracted first");
    assert_eq!(TagKeeper::memory(&k).statements().count(), 1, "only the clock is a tag here: no rights are held");
    let none = TagKeeper::over(&g, &s, &Rules::fact_decl(&g), GameMemory::new(Rules::rules_key(&g)), &[]);
    assert_eq!(none.history().len(), 0, "no earlier keys: a history of the one position");
}

#[test]
fn a_tempo_made_after_skipped_ones_pushes_their_keys_as_not_known_and_then_the_change_into_the_position_reached() {
    let g = chess();
    let (mut k, s) = start_keeper(&g);
    let after = play(&g, &s, "e2e4");
    TagKeeper::make_over(&mut k, &g, &after, &[777]);
    assert_eq!(k.history().len(), 2);
    assert_eq!(k.history().keys().to_vec(), vec![Game::key(&g, &s), 777, Game::key(&g, &after)]);
    assert!(!k.history().change_known(0) && k.history().change_known(1));
    let change: &Change = &k.history().changes()[1];
    assert!(change.added.contains(&Fact::new("cell", &["e", "4", "white", "pawn"])));
}

#[test]
fn the_key_of_a_game_with_a_start_is_the_game_not_the_start_and_a_variant_open_gives_the_rules_of_the_variant() {
    // (Ruled::open_variant and restart_delta are held by layer2_variants.rs; here: one game, two starts, one address)
    let a = Ruled::open_variant(KernelGame::new("chess960", 518).unwrap(), "chess960", &rules_root(), "learned/chess/graph_vocabulary.json", Some("variants/chess960.gdl"), None).unwrap();
    assert_eq!(Rules::rules_key(&a).name, "chess960");
    assert_eq!(Rules::rules_key(&a).rules.len(), 16);
}
