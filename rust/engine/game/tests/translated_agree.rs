//! PLAN 79.37 PART D AND PART F (SYSTEM_SPEC 11, "The crux of layer 2 is a translator from the description to Rust ... THE UNIT OF TEST IS A PIECE OF THE GAME'S LOGIC"; ledger rows 45, 44; piece: rules): THE TRANSLATED CODE GIVES THE SAME
//! ANSWERS AS THE EVALUATOR. The files tests/generated/chess.rs and atomic.rs are written by discovery/gdl_to_rust.py from rules/chess.gdl and from it composed with rules/variants/atomic.gdl; the translator is the same for both and knows
//! neither. For each: (1) every relation that does not read the state equals the evaluator's on its WHOLE domain (the rows, then every tuple of the sorts of its signature, counted); (2) at every position of the game's fixture
//! (the outside library's file) the translated code, the evaluator and the file give the same allowed moves, end and facts, and the same next facts after each move the file records, and every relation that reads the state equals the evaluator's;
//! (3) on seeded random games (the certificate's walk, seed and kernel) the five questions agree at every state and every relation agrees at the first games; (4) the counts of positions after a few plies equal the fixture's.
//! The numbers are printed as OBS lines. A disagreement fails the test with the relation and the smallest case. This file names the games on purpose (a test file may).
mod variants_common;
#[macro_use]
mod translated_common;
use cs_game::kernel::KernelGame;
use cs_game::Game as _;
use cs_game::outside::spec_term;
use cs_gdl::Program;
use std::collections::{BTreeMap, BTreeSet};
use translated_common::*;
use variants_common::fixture;

include!("generated/chess.rs");
include!("generated/atomic.rs");
impl_tr!(chess);
impl_tr!(atomic);

const SEED: u64 = 77; // the certificate's (discovery/certify.py)

fn statics<T: Tr>(game: &str, text: &str) {
    let t = T::make();
    let mut prog = Program::load(text).unwrap();
    let rep = statics_whole_domain(&t, &mut prog).unwrap_or_else(|e| panic!("{game}: {e}"));
    let (mut rows, mut dom, mut typed) = (0usize, 0u64, 0usize);
    for r in &rep {
        println!("OBS 79.37 D static {game} {}: rows {} whole domain {}", r.name, r.rows, r.domain.map(|d| d.to_string()).unwrap_or("-".into()));
        rows += r.rows;
        dom += r.domain.unwrap_or(0);
        typed += r.domain.is_some() as usize;
    }
    println!("OBS 79.37 D static {game}: {} relations that do not read the state, {rows} rows, {typed} with a whole domain of {dom} tuples in all; every one equal to the evaluator's", rep.len());
    assert!(rep.len() > 20);
}

fn fixture_positions<T: Tr>(game: &str, text: &str) {
    let fx = fixture(game);
    let mut t = T::make();
    let mut tally = Tally::default();
    let mut rels = BTreeMap::new();
    let (mut steps, mut ended) = (0usize, 0usize);
    for p in &fx.pos {
        let mut prog = program_at(text, &p.facts);                          // 2026-11-24 (plan 79.47): the evaluator's program begun at the position, as the module is
        t.restart_t(&translated_common::facts_pairs(&p.facts)).unwrap();      // 2026-11-24 (plan 79.47): begun at the position's facts as `evaluator_at` begins the evaluator (a game whose rules read its start)
        let ts = t.state_of_facts_t(&translated_common::facts_pairs(&p.facts)).unwrap();
        let ev = evaluator_at(text, &p.facts);
        let es = ev.initial();
        let label = format!("{game} fixture {} ({})", p.label, p.text);
        compare_at(&mut t, &ev, &ts, &es, &label, false, &mut tally).unwrap_or_else(|e| panic!("{e}"));
        // against the outside library's file
        let theirs: BTreeSet<String> = t.moves_t(&ts).iter().map(|m| t.render_t(*m)).collect();
        let file: BTreeSet<String> = p.moves.iter().map(|m| spec_term(m)).collect();
        assert_eq!(theirs, file, "{label}: the moves of the file");
        assert_eq!(t.outcome_t(&ts), p.end.as_ref().map(|e| e.1.clone()), "{label}: the end of the file");
        ended += p.end.is_some() as usize;
        relations_at(&mut t, &mut prog, &ts, None, &label, &mut rels).unwrap_or_else(|e| panic!("{e}"));
        for st in fx.steps.iter().filter(|st| st.parent == p.label) {
            let name = spec_term(&st.name);
            let m = t.term_t(&name).unwrap();
            assert!(t.moves_t(&ts).contains(&m), "{label}: {name} of the file is not allowed by the translated code");
            let child = t.next_t(&ts, m);
            let mut want = p.facts.clone();
            for f in &st.removed { assert!(want.remove(f)); }
            for f in &st.added { want.insert(f.clone()); }
            assert_eq!(fact_set(t.facts_of_t(&child)), want, "{label}: the facts after {name} (the file's)");
            assert_eq!(t.outcome_t(&child), st.end.as_ref().map(|e| e.1.clone()), "{label}: the end after {name}");
            // and the evaluator's, and every relation of the move
            let em = ev.find_move(&es, &name).unwrap();
            let echild = cs_game::Game::apply(&ev, &es, &em);
            assert_eq!(fact_set(t.facts_of_t(&child)), cs_game::Game::base_facts(&ev, &echild).into_iter().collect::<BTreeSet<_>>(), "{label}: the facts after {name} (the evaluator's)");
            relations_at(&mut t, &mut prog, &ts, Some(m), &format!("{label} {name}"), &mut rels).unwrap_or_else(|e| panic!("{e}"));
            steps += 1;
        }
    }
    println!("OBS 79.37 D fixture {game}: {} positions ({ended} ended), {} allowed moves, {steps} moves made, {} nexts, every one equal to the evaluator and to the file", tally.positions, tally.moves, steps);
    for (r, (s, n)) in &rels { println!("OBS 79.37 D relation {game} fixture {r}: states {s}, rows compared {n}, differing 0"); }
    assert!(tally.positions >= 100);                                 // 2026-11-24 (plan 79.47): the file of the game with captures that are forced has exactly 100 positions
}

fn random_play<T: Tr>(game: &str, text: &str, games: usize, rel_games: usize, all_every: usize) {
    let kernel = KernelGame::new(game, 518).unwrap();
    let mut t = T::make();
    let ev = evaluator(text);
    let mut prog = Program::load(text).unwrap();
    let (mut tally, mut rels) = (Tally::default(), BTreeMap::new());
    let t0 = std::time::Instant::now();
    let corpus = coplay(&kernel, &mut t, &ev, Some(&mut prog), games, SEED, all_every, rel_games, &mut tally, &mut rels).unwrap_or_else(|e| panic!("{e}"));
    let plies: usize = corpus.iter().map(|l| l.len()).sum();
    println!("OBS 79.37 D coplay {game}: {games} seeded games (seed {SEED}), {plies} plies, {} states, {} allowed moves, {} next states compared, ends at {} states, 0 disagreements ({:.1} s)",
             tally.positions, tally.moves, tally.nexts, tally.terminal, t0.elapsed().as_secs_f64());
    let mut dyn_total = 0usize;
    for (r, (s, n)) in &rels { println!("OBS 79.37 D relation {game} coplay {r}: states {s}, rows compared {n}, differing 0"); dyn_total += *s; }
    println!("OBS 79.37 D coplay {game}: {} relations that read the state, {dyn_total} relation-by-state comparisons in the first {rel_games} games", rels.len());
    assert_eq!(tally.positions, plies + games);
}

fn counts<T: Tr>(game: &str, depth_start: usize, depth_other: usize) {
    let fx = fixture(game);
    let mut t = T::make();
    let mut n = 0u64;
    for (label, fen, counts) in &fx.perft {
        let depth = if label == "start" { depth_start } else { depth_other }.min(counts.len());
        let k = KernelGame::from_fen(game, fen).unwrap();
        let facts: Vec<(String, Vec<String>)> = k.state_facts(&cs_game::Game::initial(&k)).into_iter().map(|f| (f.relation, f.args)).collect();
        t.restart_t(&facts).unwrap();                                     // 2026-11-24 (plan 79.47): the module begun at the position's facts, as the kernel's `from_fen` begins the kernel
        let s = t.state_of_facts_t(&facts).unwrap();
        for d in 1..=depth {
            let c = perft(&mut t, &s, d);
            assert_eq!(c, counts[d - 1], "{game}: positions after {d} plies from {label} ({fen})");
            n += c;
        }
        println!("OBS 79.37 D counts {game} {label}: after 1..{depth} plies {:?} equal to the fixture's", &counts[..depth]);
    }
    println!("OBS 79.37 D counts {game}: {n} positions counted, all equal to the fixture's");
}

#[test]
fn chess_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { statics::<chess::Game>("chess", &description(None)); }
#[test]
fn atomic_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { statics::<atomic::Game>("atomic", &description(Some("variants/atomic.gdl"))); }

#[test]
fn chess_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { fixture_positions::<chess::Game>("chess", &description(None)); }
#[test]
fn atomic_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { fixture_positions::<atomic::Game>("atomic", &description(Some("variants/atomic.gdl"))); }

#[test]
fn chess_the_certificates_random_games_agree_at_every_state() { random_play::<chess::Game>("chess", &description(None), 20, 4, 10); }
#[test]
fn atomic_the_certificates_random_games_agree_at_every_state() { random_play::<atomic::Game>("atomic", &description(Some("variants/atomic.gdl")), 20, 4, 10); }

#[test]
fn chess_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { counts::<chess::Game>("chess", 4, 3); }
#[test]
fn atomic_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { counts::<atomic::Game>("atomic", 4, 3); }

// ------------------------------------------------------------------------------------------------------------------------------------------ PLAN 79.38: the same tests on the modules with every pass (the game's name is the fixture's and the kernel's; the OBS line names the module)

include!("generated/chess_fast.rs");
include!("generated/atomic_fast.rs");
impl_tr!(chess_fast, passes);
impl_tr!(atomic_fast, passes);

fn module_line<T: Tr>(module: &str) { println!("OBS 79.38 D module {module}: passes {:?}, {} relations answered on demand, copy kept for {:?}", T::make().passes_t(), T::make().demanded_t().len(), T::make().copied_t()); }

#[test]
fn chess_with_the_passes_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { module_line::<chess_fast::Game>("chess_fast"); statics::<chess_fast::Game>("chess", &description(None)); }
#[test]
fn atomic_with_the_passes_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { module_line::<atomic_fast::Game>("atomic_fast"); statics::<atomic_fast::Game>("atomic", &description(Some("variants/atomic.gdl"))); }

#[test]
fn chess_with_the_passes_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { module_line::<chess_fast::Game>("chess_fast"); fixture_positions::<chess_fast::Game>("chess", &description(None)); }
#[test]
fn atomic_with_the_passes_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { module_line::<atomic_fast::Game>("atomic_fast"); fixture_positions::<atomic_fast::Game>("atomic", &description(Some("variants/atomic.gdl"))); }

#[test]
fn chess_with_the_passes_the_certificates_random_games_agree_at_every_state() { module_line::<chess_fast::Game>("chess_fast"); random_play::<chess_fast::Game>("chess", &description(None), 20, 4, 10); }
#[test]
fn atomic_with_the_passes_the_certificates_random_games_agree_at_every_state() { module_line::<atomic_fast::Game>("atomic_fast"); random_play::<atomic_fast::Game>("atomic", &description(Some("variants/atomic.gdl")), 20, 4, 10); }

#[test]
fn chess_with_the_passes_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { module_line::<chess_fast::Game>("chess_fast"); counts::<chess_fast::Game>("chess", 4, 3); }
#[test]
fn atomic_with_the_passes_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { module_line::<atomic_fast::Game>("atomic_fast"); counts::<atomic_fast::Game>("atomic", 4, 3); }

// ------------------------------------------------------------------------------------------------------------------------------------------ PLAN 79.47 PART B: the five more games, the plain module and the module with every pass, held to the evaluator as the first two are
// 2026-11-24 (plan 79.47): the five more games of the seven certified kernels (3check, kingofthehill, antichess, chess960, horde), each a delta on the first game's description, the modules made by the same translator.
include!("generated/3check.rs");
include!("generated/3check_fast.rs");
impl_tr!(threecheck);
impl_tr!(threecheck_fast, passes);
include!("generated/kingofthehill.rs");
include!("generated/kingofthehill_fast.rs");
impl_tr!(kingofthehill);
impl_tr!(kingofthehill_fast, passes);
include!("generated/antichess.rs");
include!("generated/antichess_fast.rs");
impl_tr!(antichess);
impl_tr!(antichess_fast, passes);
include!("generated/chess960.rs");
include!("generated/chess960_fast.rs");
impl_tr!(chess960);
impl_tr!(chess960_fast, passes);
include!("generated/horde.rs");
include!("generated/horde_fast.rs");
impl_tr!(horde);
impl_tr!(horde_fast, passes);
#[test]
fn threecheck_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { statics::<threecheck::Game>("3check", &description(Some("variants/3check.gdl"))); }
#[test]
fn threecheck_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { fixture_positions::<threecheck::Game>("3check", &description(Some("variants/3check.gdl"))); }
#[test]
fn threecheck_the_certificates_random_games_agree_at_every_state() { random_play::<threecheck::Game>("3check", &description(Some("variants/3check.gdl")), 20, 4, 10); }
#[test]
fn threecheck_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { counts::<threecheck::Game>("3check", 4, 3); }
#[test]
fn threecheck_with_the_passes_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { module_line::<threecheck_fast::Game>("threecheck_fast"); statics::<threecheck_fast::Game>("3check", &description(Some("variants/3check.gdl"))); }
#[test]
fn threecheck_with_the_passes_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { module_line::<threecheck_fast::Game>("threecheck_fast"); fixture_positions::<threecheck_fast::Game>("3check", &description(Some("variants/3check.gdl"))); }
#[test]
fn threecheck_with_the_passes_the_certificates_random_games_agree_at_every_state() { module_line::<threecheck_fast::Game>("threecheck_fast"); random_play::<threecheck_fast::Game>("3check", &description(Some("variants/3check.gdl")), 20, 4, 10); }
#[test]
fn threecheck_with_the_passes_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { module_line::<threecheck_fast::Game>("threecheck_fast"); counts::<threecheck_fast::Game>("3check", 4, 3); }
#[test]
fn kingofthehill_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { statics::<kingofthehill::Game>("kingofthehill", &description(Some("variants/kingofthehill.gdl"))); }
#[test]
fn kingofthehill_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { fixture_positions::<kingofthehill::Game>("kingofthehill", &description(Some("variants/kingofthehill.gdl"))); }
#[test]
fn kingofthehill_the_certificates_random_games_agree_at_every_state() { random_play::<kingofthehill::Game>("kingofthehill", &description(Some("variants/kingofthehill.gdl")), 20, 4, 10); }
#[test]
fn kingofthehill_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { counts::<kingofthehill::Game>("kingofthehill", 4, 3); }
#[test]
fn kingofthehill_with_the_passes_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { module_line::<kingofthehill_fast::Game>("kingofthehill_fast"); statics::<kingofthehill_fast::Game>("kingofthehill", &description(Some("variants/kingofthehill.gdl"))); }
#[test]
fn kingofthehill_with_the_passes_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { module_line::<kingofthehill_fast::Game>("kingofthehill_fast"); fixture_positions::<kingofthehill_fast::Game>("kingofthehill", &description(Some("variants/kingofthehill.gdl"))); }
#[test]
fn kingofthehill_with_the_passes_the_certificates_random_games_agree_at_every_state() { module_line::<kingofthehill_fast::Game>("kingofthehill_fast"); random_play::<kingofthehill_fast::Game>("kingofthehill", &description(Some("variants/kingofthehill.gdl")), 20, 4, 10); }
#[test]
fn kingofthehill_with_the_passes_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { module_line::<kingofthehill_fast::Game>("kingofthehill_fast"); counts::<kingofthehill_fast::Game>("kingofthehill", 4, 3); }
#[test]
fn antichess_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { statics::<antichess::Game>("antichess", &description(Some("variants/antichess.gdl"))); }
#[test]
fn antichess_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { fixture_positions::<antichess::Game>("antichess", &description(Some("variants/antichess.gdl"))); }
#[test]
fn antichess_the_certificates_random_games_agree_at_every_state() { random_play::<antichess::Game>("antichess", &description(Some("variants/antichess.gdl")), 20, 4, 10); }
#[test]
fn antichess_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { counts::<antichess::Game>("antichess", 4, 3); }
#[test]
fn antichess_with_the_passes_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { module_line::<antichess_fast::Game>("antichess_fast"); statics::<antichess_fast::Game>("antichess", &description(Some("variants/antichess.gdl"))); }
#[test]
fn antichess_with_the_passes_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { module_line::<antichess_fast::Game>("antichess_fast"); fixture_positions::<antichess_fast::Game>("antichess", &description(Some("variants/antichess.gdl"))); }
#[test]
fn antichess_with_the_passes_the_certificates_random_games_agree_at_every_state() { module_line::<antichess_fast::Game>("antichess_fast"); random_play::<antichess_fast::Game>("antichess", &description(Some("variants/antichess.gdl")), 20, 4, 10); }
#[test]
fn antichess_with_the_passes_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { module_line::<antichess_fast::Game>("antichess_fast"); counts::<antichess_fast::Game>("antichess", 4, 3); }
#[test]
fn chess960_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { statics::<chess960::Game>("chess960", &description(Some("variants/chess960.gdl"))); }
#[test]
fn chess960_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { fixture_positions::<chess960::Game>("chess960", &description(Some("variants/chess960.gdl"))); }
#[test]
fn chess960_the_certificates_random_games_agree_at_every_state() { random_play::<chess960::Game>("chess960", &description(Some("variants/chess960.gdl")), 20, 4, 10); }
#[test]
fn chess960_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { counts::<chess960::Game>("chess960", 4, 3); }
#[test]
fn chess960_with_the_passes_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { module_line::<chess960_fast::Game>("chess960_fast"); statics::<chess960_fast::Game>("chess960", &description(Some("variants/chess960.gdl"))); }
#[test]
fn chess960_with_the_passes_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { module_line::<chess960_fast::Game>("chess960_fast"); fixture_positions::<chess960_fast::Game>("chess960", &description(Some("variants/chess960.gdl"))); }
#[test]
fn chess960_with_the_passes_the_certificates_random_games_agree_at_every_state() { module_line::<chess960_fast::Game>("chess960_fast"); random_play::<chess960_fast::Game>("chess960", &description(Some("variants/chess960.gdl")), 20, 4, 10); }
#[test]
fn chess960_with_the_passes_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { module_line::<chess960_fast::Game>("chess960_fast"); counts::<chess960_fast::Game>("chess960", 4, 3); }
#[test]
fn horde_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { statics::<horde::Game>("horde", &description(Some("variants/horde.gdl"))); }
#[test]
fn horde_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { fixture_positions::<horde::Game>("horde", &description(Some("variants/horde.gdl"))); }
#[test]
fn horde_the_certificates_random_games_agree_at_every_state() { random_play::<horde::Game>("horde", &description(Some("variants/horde.gdl")), 20, 4, 10); }
#[test]
fn horde_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { counts::<horde::Game>("horde", 4, 3); }
#[test]
fn horde_with_the_passes_every_relation_that_does_not_read_the_state_equals_the_evaluators_on_its_whole_domain() { module_line::<horde_fast::Game>("horde_fast"); statics::<horde_fast::Game>("horde", &description(Some("variants/horde.gdl"))); }
#[test]
fn horde_with_the_passes_every_fixture_position_and_every_move_of_the_file_and_every_relation_agree() { module_line::<horde_fast::Game>("horde_fast"); fixture_positions::<horde_fast::Game>("horde", &description(Some("variants/horde.gdl"))); }
#[test]
fn horde_with_the_passes_the_certificates_random_games_agree_at_every_state() { module_line::<horde_fast::Game>("horde_fast"); random_play::<horde_fast::Game>("horde", &description(Some("variants/horde.gdl")), 20, 4, 10); }
#[test]
fn horde_with_the_passes_the_counts_of_positions_after_a_few_plies_equal_the_fixtures() { module_line::<horde_fast::Game>("horde_fast"); counts::<horde_fast::Game>("horde", 4, 3); }
