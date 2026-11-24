//! PLAN 79.38 PART A (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... THE UNIT OF TEST IS A PIECE OF THE GAME'S LOGIC"; ledger row 45; piece: rules): THE PLAIN TRANSLATION IS THE REFERENCE OF EVERY PIECE.
//! For each game (chess, atomic) and each module with passes (dense, demand, copy, delta, and all four together as `fast`; tests/generated/<game>_<variant>.rs, written by discovery/gdl_to_rust.py with the passes named in its head), the module is held to
//! the plain module (tests/generated/<game>.rs) piece by piece, by the harness of translated_common/pieces.rs: the relations that do not read the state, whole; at every position of the game's fixture (and each move the fixture records) and at
//! every state of the certificate's seeded games, the five questions, every relation the module makes in full, and every relation it answers on demand in every way it is asked (keys made from the rows of the plain relation and from rows with
//! one place changed). The numbers are printed as OBS lines; any disagreement fails the test with the relation and the smallest case. The tests of 79.37 against the evaluator are translated_agree.rs (run on the plain modules and on `fast`).
//! This file names the games on purpose (a test file may).
mod variants_common;
#[macro_use]
mod translated_common;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::rng::Rng;
use translated_common::pieces::*;
use translated_common::*;
use variants_common::fixture;

include!("generated/chess.rs");
include!("generated/chess_dense.rs");
include!("generated/chess_demand.rs");
include!("generated/chess_copy.rs");
include!("generated/chess_delta.rs");
include!("generated/chess_fast.rs");
include!("generated/atomic.rs");
include!("generated/atomic_dense.rs");
include!("generated/atomic_demand.rs");
include!("generated/atomic_copy.rs");
include!("generated/atomic_delta.rs");
include!("generated/atomic_fast.rs");
impl_tr!(chess);
impl_tr!(chess_dense, passes);
impl_tr!(chess_demand, passes);
impl_tr!(chess_copy, passes);
impl_tr!(chess_delta, passes);
impl_tr!(chess_fast, passes);
impl_tr!(atomic);
impl_tr!(atomic_dense, passes);
impl_tr!(atomic_demand, passes);
impl_tr!(atomic_copy, passes);
impl_tr!(atomic_delta, passes);
impl_tr!(atomic_fast, passes);

const SEED: u64 = 77; // the certificate's (discovery/certify.py)

fn report(game: &str, variant: &str, what: &str, c: &PieceCounts) {
    println!("OBS 79.38 A pieces {game} {variant} {what}: {} piece-by-piece comparisons ({} states, {} question answers, {} next states, {} relations in full, {} key probes of demanded relations, {} static relations whole), 0 disagreements; {} relations answered on demand",
             c.total(), c.states, c.questions, c.nexts, c.relations, c.keys, c.statics, c.on_demand.len());
}

fn statics<A: Tr, B: Tr>(game: &str, variant: &str) {
    let (a, b) = (A::make(), B::make());
    let mut c = PieceCounts::default();
    statics_equal(&a, &b, &mut c).unwrap_or_else(|e| panic!("{game} {variant}: {e}"));
    report(game, variant, "static", &c);
    assert!(c.statics > 20);
    println!("OBS 79.38 A passes {game} {variant}: {:?}, answered on demand {:?}, copy kept for {:?}", b.passes_t(), b.demanded_t().iter().map(|d| (d.0, d.1.len())).collect::<Vec<_>>().len(), b.copied_t());
}

fn fixtures<A: Tr, B: Tr>(game: &str, variant: &str) {
    let fx = fixture(game);
    let (mut a, mut b) = (A::make(), B::make());
    let mut c = PieceCounts::default();
    for p in &fx.pos {
        let facts = facts_pairs(&p.facts);
        let label = format!("{game} {variant} fixture {} ({})", p.label, p.text);
        pieces_at(&mut a, &mut b, &facts, &None, &label, true, &mut c).unwrap_or_else(|e| panic!("{e}"));
        for st in fx.steps.iter().filter(|st| st.parent == p.label) {
            let mv = Some(spec_term(&st.name));
            pieces_at(&mut a, &mut b, &facts, &mv, &format!("{label} {}", st.name), true, &mut c).unwrap_or_else(|e| panic!("{e}"));
        }
    }
    report(game, variant, "fixture", &c);
    assert!(c.states > 100);
}

fn random_games<A: Tr, B: Tr>(game: &str, variant: &str, games: usize, rel_games: usize) {
    let kernel = KernelGame::new(game, 518).unwrap();
    let (mut a, mut b) = (A::make(), B::make());
    let walk = corpus(&kernel, &mut a, games, SEED);
    let mut c = PieceCounts::default();
    let t0 = std::time::Instant::now();
    for (g, line) in walk.iter().enumerate() {
        for (i, (facts, mv)) in line.iter().enumerate() {
            let label = format!("{game} {variant} game {g} ply {i}");
            pieces_at(&mut a, &mut b, facts, mv, &label, g < rel_games, &mut c).unwrap_or_else(|e| panic!("{e}"));
        }
    }
    report(game, variant, &format!("seeded games (seed {SEED}, {games} games, relations in the first {rel_games}; {:.1} s)", t0.elapsed().as_secs_f64()), &c);
    for (r, (full, keys)) in &c.per_relation { println!("OBS 79.38 A relation {game} {variant} {r}: {full} comparisons in full, {keys} key probes"); }
}

macro_rules! pieces_tests {
    ($game:literal, $a:ident, $variant:literal, $b:ident, $s:ident, $f:ident, $r:ident) => {
        #[test]
        fn $s() { statics::<$a::Game, $b::Game>($game, $variant); }
        #[test]
        fn $f() { fixtures::<$a::Game, $b::Game>($game, $variant); }
        #[test]
        fn $r() { random_games::<$a::Game, $b::Game>($game, $variant, 20, 6); }
    };
}

pieces_tests!("chess", chess, "dense", chess_dense, chess_dense_static_relations_are_the_plain_ones, chess_dense_every_piece_at_every_fixture_position_is_the_plain_one, chess_dense_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("chess", chess, "demand", chess_demand, chess_demand_static_relations_are_the_plain_ones, chess_demand_every_piece_at_every_fixture_position_is_the_plain_one, chess_demand_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("chess", chess, "copy", chess_copy, chess_copy_static_relations_are_the_plain_ones, chess_copy_every_piece_at_every_fixture_position_is_the_plain_one, chess_copy_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("chess", chess, "delta", chess_delta, chess_delta_static_relations_are_the_plain_ones, chess_delta_every_piece_at_every_fixture_position_is_the_plain_one, chess_delta_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("chess", chess, "fast", chess_fast, chess_fast_static_relations_are_the_plain_ones, chess_fast_every_piece_at_every_fixture_position_is_the_plain_one, chess_fast_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("atomic", atomic, "dense", atomic_dense, atomic_dense_static_relations_are_the_plain_ones, atomic_dense_every_piece_at_every_fixture_position_is_the_plain_one, atomic_dense_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("atomic", atomic, "demand", atomic_demand, atomic_demand_static_relations_are_the_plain_ones, atomic_demand_every_piece_at_every_fixture_position_is_the_plain_one, atomic_demand_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("atomic", atomic, "copy", atomic_copy, atomic_copy_static_relations_are_the_plain_ones, atomic_copy_every_piece_at_every_fixture_position_is_the_plain_one, atomic_copy_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("atomic", atomic, "delta", atomic_delta, atomic_delta_static_relations_are_the_plain_ones, atomic_delta_every_piece_at_every_fixture_position_is_the_plain_one, atomic_delta_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("atomic", atomic, "fast", atomic_fast, atomic_fast_static_relations_are_the_plain_ones, atomic_fast_every_piece_at_every_fixture_position_is_the_plain_one, atomic_fast_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
