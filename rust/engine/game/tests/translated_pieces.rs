//! PLAN 79.38 PART A (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... THE UNIT OF TEST IS A PIECE OF THE GAME'S LOGIC"; ledger row 45; piece: rules): THE PLAIN TRANSLATION IS THE REFERENCE OF EVERY PIECE.
//! For each game (chess, atomic) and each module with passes (`fast`, every pass: tests/generated/<game>_fast.rs, kept in the repository; and, when this file is compiled with `--cfg singles` (tests/test_plan79_40.py does it), the single passes dense, demand, copy, delta, bits, order, keep and the four of 79.38 together, `prev`, made under tests/generated/scratch by scripts/gdl_to_rust_make.py --scratch), the module is held to
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
include!("generated/chess_fast.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev.rs");
#[cfg(singles)]
include!("generated/scratch/chess_dense.rs");
#[cfg(singles)]
include!("generated/scratch/chess_demand.rs");
#[cfg(singles)]
include!("generated/scratch/chess_copy.rs");
#[cfg(singles)]
include!("generated/scratch/chess_delta.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits.rs");
#[cfg(singles)]
include!("generated/scratch/chess_order.rs");
#[cfg(singles)]
include!("generated/scratch/chess_keep.rs");
include!("generated/atomic.rs");
include!("generated/atomic_fast.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_dense.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_demand.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_copy.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_delta.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_order.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_keep.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits_scan.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits_cut.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits_shift.rs");
#[cfg(singles)]
include!("generated/scratch/chess_bits_base.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev41_orient.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev41_ends.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev41_start.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev_42.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev_project.rs");
#[cfg(singles)]
include!("generated/scratch/chess_prev_asked.rs");
impl_tr!(chess);
impl_tr!(chess_fast, passes);
#[cfg(singles)]
impl_tr!(chess_prev, passes);
#[cfg(singles)]
impl_tr!(chess_dense, passes);
#[cfg(singles)]
impl_tr!(chess_demand, passes);
#[cfg(singles)]
impl_tr!(chess_copy, passes);
#[cfg(singles)]
impl_tr!(chess_delta, passes);
#[cfg(singles)]
impl_tr!(chess_bits, passes);
#[cfg(singles)]
impl_tr!(chess_order, passes);
#[cfg(singles)]
impl_tr!(chess_keep, passes);
#[cfg(singles)]
include!("generated/scratch/atomic_bits_scan.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits_cut.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits_shift.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_bits_base.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev41_orient.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev41_ends.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev41_start.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev_42.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev_project.rs");
#[cfg(singles)]
include!("generated/scratch/atomic_prev_asked.rs");
impl_tr!(atomic);
impl_tr!(atomic_fast, passes);
#[cfg(singles)]
impl_tr!(atomic_prev, passes);
#[cfg(singles)]
impl_tr!(atomic_dense, passes);
#[cfg(singles)]
impl_tr!(atomic_demand, passes);
#[cfg(singles)]
impl_tr!(atomic_copy, passes);
#[cfg(singles)]
impl_tr!(atomic_delta, passes);
#[cfg(singles)]
impl_tr!(atomic_bits, passes);
#[cfg(singles)]
impl_tr!(atomic_order, passes);
#[cfg(singles)]
impl_tr!(atomic_keep, passes);
#[cfg(singles)]
impl_tr!(chess_bits_scan, passes);
#[cfg(singles)]
impl_tr!(atomic_bits_scan, passes);
#[cfg(singles)]
impl_tr!(chess_bits_cut, passes);
#[cfg(singles)]
impl_tr!(atomic_bits_cut, passes);
#[cfg(singles)]
impl_tr!(chess_bits_shift, passes);
#[cfg(singles)]
impl_tr!(atomic_bits_shift, passes);
#[cfg(singles)]
impl_tr!(chess_bits_base, passes);
#[cfg(singles)]
impl_tr!(chess_prev41_orient, passes);
#[cfg(singles)]
impl_tr!(chess_prev41_ends, passes);
#[cfg(singles)]
impl_tr!(chess_prev41_start, passes);
#[cfg(singles)]
impl_tr!(chess_prev_42, passes);
#[cfg(singles)]
impl_tr!(chess_prev_project, passes);
#[cfg(singles)]
impl_tr!(chess_prev_asked, passes);
#[cfg(singles)]
impl_tr!(atomic_bits_base, passes);
#[cfg(singles)]
impl_tr!(atomic_prev41_orient, passes);
#[cfg(singles)]
impl_tr!(atomic_prev41_ends, passes);
#[cfg(singles)]
impl_tr!(atomic_prev41_start, passes);
#[cfg(singles)]
impl_tr!(atomic_prev_42, passes);
#[cfg(singles)]
impl_tr!(atomic_prev_project, passes);
#[cfg(singles)]
impl_tr!(atomic_prev_asked, passes);

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

pieces_tests!("chess", chess, "fast", chess_fast, chess_fast_static_relations_are_the_plain_ones, chess_fast_every_piece_at_every_fixture_position_is_the_plain_one, chess_fast_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "prev", chess_prev, chess_prev_static_relations_are_the_plain_ones, chess_prev_every_piece_at_every_fixture_position_is_the_plain_one, chess_prev_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "dense", chess_dense, chess_dense_static_relations_are_the_plain_ones, chess_dense_every_piece_at_every_fixture_position_is_the_plain_one, chess_dense_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "demand", chess_demand, chess_demand_static_relations_are_the_plain_ones, chess_demand_every_piece_at_every_fixture_position_is_the_plain_one, chess_demand_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "copy", chess_copy, chess_copy_static_relations_are_the_plain_ones, chess_copy_every_piece_at_every_fixture_position_is_the_plain_one, chess_copy_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "delta", chess_delta, chess_delta_static_relations_are_the_plain_ones, chess_delta_every_piece_at_every_fixture_position_is_the_plain_one, chess_delta_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "bits", chess_bits, chess_bits_static_relations_are_the_plain_ones, chess_bits_every_piece_at_every_fixture_position_is_the_plain_one, chess_bits_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "order", chess_order, chess_order_static_relations_are_the_plain_ones, chess_order_every_piece_at_every_fixture_position_is_the_plain_one, chess_order_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "keep", chess_keep, chess_keep_static_relations_are_the_plain_ones, chess_keep_every_piece_at_every_fixture_position_is_the_plain_one, chess_keep_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
pieces_tests!("atomic", atomic, "fast", atomic_fast, atomic_fast_static_relations_are_the_plain_ones, atomic_fast_every_piece_at_every_fixture_position_is_the_plain_one, atomic_fast_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "prev", atomic_prev, atomic_prev_static_relations_are_the_plain_ones, atomic_prev_every_piece_at_every_fixture_position_is_the_plain_one, atomic_prev_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "dense", atomic_dense, atomic_dense_static_relations_are_the_plain_ones, atomic_dense_every_piece_at_every_fixture_position_is_the_plain_one, atomic_dense_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "demand", atomic_demand, atomic_demand_static_relations_are_the_plain_ones, atomic_demand_every_piece_at_every_fixture_position_is_the_plain_one, atomic_demand_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "copy", atomic_copy, atomic_copy_static_relations_are_the_plain_ones, atomic_copy_every_piece_at_every_fixture_position_is_the_plain_one, atomic_copy_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "delta", atomic_delta, atomic_delta_static_relations_are_the_plain_ones, atomic_delta_every_piece_at_every_fixture_position_is_the_plain_one, atomic_delta_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "bits", atomic_bits, atomic_bits_static_relations_are_the_plain_ones, atomic_bits_every_piece_at_every_fixture_position_is_the_plain_one, atomic_bits_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "order", atomic_order, atomic_order_static_relations_are_the_plain_ones, atomic_order_every_piece_at_every_fixture_position_is_the_plain_one, atomic_order_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "keep", atomic_keep, atomic_keep_static_relations_are_the_plain_ones, atomic_keep_every_piece_at_every_fixture_position_is_the_plain_one, atomic_keep_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "bits_scan", chess_bits_scan, chess_bits_scan_static_relations_are_the_plain_ones, chess_bits_scan_every_piece_at_every_fixture_position_is_the_plain_one, chess_bits_scan_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "bits_scan", atomic_bits_scan, atomic_bits_scan_static_relations_are_the_plain_ones, atomic_bits_scan_every_piece_at_every_fixture_position_is_the_plain_one, atomic_bits_scan_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "bits_cut", chess_bits_cut, chess_bits_cut_static_relations_are_the_plain_ones, chess_bits_cut_every_piece_at_every_fixture_position_is_the_plain_one, chess_bits_cut_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "bits_cut", atomic_bits_cut, atomic_bits_cut_static_relations_are_the_plain_ones, atomic_bits_cut_every_piece_at_every_fixture_position_is_the_plain_one, atomic_bits_cut_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "bits_shift", chess_bits_shift, chess_bits_shift_static_relations_are_the_plain_ones, chess_bits_shift_every_piece_at_every_fixture_position_is_the_plain_one, chess_bits_shift_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "bits_shift", atomic_bits_shift, atomic_bits_shift_static_relations_are_the_plain_ones, atomic_bits_shift_every_piece_at_every_fixture_position_is_the_plain_one, atomic_bits_shift_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "bits_base", chess_bits_base, chess_bits_base_static_relations_are_the_plain_ones, chess_bits_base_every_piece_at_every_fixture_position_is_the_plain_one, chess_bits_base_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "bits_base", atomic_bits_base, atomic_bits_base_static_relations_are_the_plain_ones, atomic_bits_base_every_piece_at_every_fixture_position_is_the_plain_one, atomic_bits_base_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "prev41_start", chess_prev41_start, chess_prev41_start_static_relations_are_the_plain_ones, chess_prev41_start_every_piece_at_every_fixture_position_is_the_plain_one, chess_prev41_start_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "prev_42", chess_prev_42, chess_prev_42_static_relations_are_the_plain_ones, chess_prev_42_every_piece_at_every_fixture_position_is_the_plain_one, chess_prev_42_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "prev_project", chess_prev_project, chess_prev_project_static_relations_are_the_plain_ones, chess_prev_project_every_piece_at_every_fixture_position_is_the_plain_one, chess_prev_project_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "prev_asked", chess_prev_asked, chess_prev_asked_static_relations_are_the_plain_ones, chess_prev_asked_every_piece_at_every_fixture_position_is_the_plain_one, chess_prev_asked_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "prev41_start", atomic_prev41_start, atomic_prev41_start_static_relations_are_the_plain_ones, atomic_prev41_start_every_piece_at_every_fixture_position_is_the_plain_one, atomic_prev41_start_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "prev_42", atomic_prev_42, atomic_prev_42_static_relations_are_the_plain_ones, atomic_prev_42_every_piece_at_every_fixture_position_is_the_plain_one, atomic_prev_42_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "prev_project", atomic_prev_project, atomic_prev_project_static_relations_are_the_plain_ones, atomic_prev_project_every_piece_at_every_fixture_position_is_the_plain_one, atomic_prev_project_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "prev_asked", atomic_prev_asked, atomic_prev_asked_static_relations_are_the_plain_ones, atomic_prev_asked_every_piece_at_every_fixture_position_is_the_plain_one, atomic_prev_asked_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "prev41_ends", chess_prev41_ends, chess_prev41_ends_static_relations_are_the_plain_ones, chess_prev41_ends_every_piece_at_every_fixture_position_is_the_plain_one, chess_prev41_ends_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "prev41_ends", atomic_prev41_ends, atomic_prev41_ends_static_relations_are_the_plain_ones, atomic_prev41_ends_every_piece_at_every_fixture_position_is_the_plain_one, atomic_prev41_ends_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("chess", chess, "prev41_orient", chess_prev41_orient, chess_prev41_orient_static_relations_are_the_plain_ones, chess_prev41_orient_every_piece_at_every_fixture_position_is_the_plain_one, chess_prev41_orient_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
#[cfg(singles)]
pieces_tests!("atomic", atomic, "prev41_orient", atomic_prev41_orient, atomic_prev41_orient_static_relations_are_the_plain_ones, atomic_prev41_orient_every_piece_at_every_fixture_position_is_the_plain_one, atomic_prev41_orient_every_piece_at_every_state_of_the_seeded_games_is_the_plain_one);
