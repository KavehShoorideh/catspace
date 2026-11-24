//! PLAN 79.48 PART E (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 42): THE KNOWN CAUSES OF THE DISAGREEMENTS OF THE PLACED RECORDS, HELD AS STATES.
//! The records (translated_placed.rs, discovery/placed_translated.py) name the causes they met; each is held here as one state in an ignored test: run by hand it asks the kernel and the module of the definitions at that one state, prints both answers
//! and the evaluator's, and asserts that the two differ as the cause says, so the comparison is shown to SEE the cause. A mending is not this file's: the first cause is a choice of what the question `reach_from` means (one step, as the kernel and the
//! hand tests of plan 79.36 read it, or one move, as the description's `move_by` says it), the second (atomic) is NOT a cause of the record any more: the kernel answers `bears on`, equal to the description's `attacked`; the fault of 79.36 is a difference between two meanings, held here as a state.
//! This file names the games on purpose (a test file may). It is not part of the hashes of a record (it makes no count).
mod translated_common;
use cs_game::gdl_game::{parse_term, render_term, row_args};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game as _;
use cs_gdl::{Program, Rel, Row, ARITY_MAX, FREE};
use std::collections::BTreeSet;
use translated_common::*;

include!("generated_placed/chess.rs");
include!("generated_placed/atomic.rs");
impl_tr!(placed_chess, passes);
impl_tr!(placed_atomic, passes);

fn delta_of(name: &str) -> Option<String> {
    let text = std::fs::read_to_string(rules_path(&format!("placed/{name}.json"))).ok()?;
    let (mut out, mut inside) = (vec![], false);
    for ln in text.lines() {
        let t = ln.trim();
        if t.starts_with("\"delta\": [") { inside = true; continue; }
        if inside {
            if t == "]" || t == "]," { break; }
            out.push(t.trim_end_matches(',').trim_matches('"').replace("\\\"", "\"").replace("\\\\", "\\"));
        }
    }
    Some(out.join("\n"))
}

fn text_of(delta: Option<&str>, game: &str) -> String {
    let mut text = cs_gdl::compose(&description(delta), &delta_of("base").unwrap()).unwrap();
    if let Some(d) = delta_of(game) { text = cs_gdl::compose(&text, &d).unwrap(); }
    text
}

type Pairs = Vec<(String, Vec<String>)>;

/// The rows of a relation of the definitions, from the module and from the evaluator, at the state `pairs`.
fn rows_both<T: Tr>(t: &mut T, text: &str, pairs: &Pairs, rel: &str, arity: usize) -> (BTreeSet<Vec<String>>, BTreeSet<Vec<String>>) {
    let s = t.state_of_facts_t(pairs).unwrap();
    let got = t.ask_t(&s, &[], &[(rel.to_string(), vec![u32::MAX; arity])]);
    let module: BTreeSet<Vec<String>> = got[0].0.iter().map(|r| r.iter().map(|x| t.render_t(*x)).collect()).collect();
    let mut prog = Program::load(text).unwrap();
    let mut facts: Vec<(Rel, Row)> = vec![];
    for (r, args) in pairs {
        let id = prog.rel_id(&format!("true.{r}")).unwrap();
        let mut row = [FREE; ARITY_MAX];
        for (i, a) in args.iter().enumerate() { row[i] = parse_term(&mut prog, a).unwrap(); }
        facts.push((id, row));
    }
    let db = prog.run(&facts);
    let ev: BTreeSet<Vec<String>> = db.facts(prog.rel_id(rel).unwrap()).iter().map(|row| row_args(row).iter().map(|x| render_term(&prog, *x)).collect()).collect();
    (module, ev)
}

fn pairs_of(k: &KernelGame, fen: &str, control_only: Option<&str>) -> (KernelGame, Pairs) {
    let kk = KernelGame::from_fen(&k.label, fen).unwrap();
    let facts = kk.state_facts(&kk.initial());
    let mut v: Pairs = facts.iter().filter(|f| !(control_only.is_some() && (f.relation == "control" || f.relation == "ep_square"))).map(|f| (f.relation.clone(), f.args.clone())).collect();
    if let Some(side) = control_only { v.push(("control".into(), vec![side.to_string()])); }
    (kk, v)
}

#[test]
#[ignore = "known cause (plan 79.48, reach_from): the kernel's reach_from reads a unit's STEP (a smallest unit one place ahead) while the description's `move_by` gives its first move of two places too; run by hand to see the comparison find it; the choice is Kaveh's (it changes the graph's may-move nodes in every game)"]
fn chess_known_cause_the_smallest_kind_on_its_start_row_reaches_two_places_by_the_description_and_one_by_the_kernel() {
    let text = text_of(None, "chess");
    let k = KernelGame::new("chess", 518).unwrap();
    let (kk, pairs) = pairs_of(&k, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", Some("white"));
    let mut t = placed_chess::Game::make();
    let (module, ev) = rows_both(&mut t, &text, &pairs, "placed_reaches", 6);
    let places = kk.place_names();
    let kernel: BTreeSet<String> = kk.reach_from(&kk.initial(), 0, 3, 1).into_iter().map(|p| places[p as usize].clone()).collect();      // the pawn on the second row of the file a (place 1), kind 3 = the smallest
    let from_a2 = |s: &BTreeSet<Vec<String>>| -> BTreeSet<String> { s.iter().filter(|r| r[0] == "pawn" && r[1] == "white" && r[2] == "a" && r[3] == "2").map(|r| format!("{}{}", r[4], r[5])).collect() };
    println!("OBS 79.48 E known cause chess: the kernel's reach of the pawn on a2 {kernel:?}, the module's {:?}, the evaluator's {:?}", from_a2(&module), from_a2(&ev));
    assert_eq!(from_a2(&module), from_a2(&ev), "the module is the description's");
    assert!(from_a2(&module).contains("a4") && !kernel.contains("a4"), "the kernel and the module agree here: the cause is gone");
}

#[test]
#[ignore = "known cause (plan 79.36 fault 2, plan 79.48 addendum, two meanings of a strike in atomic): the kernel's strikes_place BEARS ON (the description's `attacked`, which the record holds it to and finds equal); the description's `struck` (the check notion) leaves out a place beside the owner's own leader, which is what the hand test of 79.36 asked for; the callers above decide which they need (ledger row 42); run by hand to see the two differ"]
fn atomic_known_difference_a_leader_beside_the_other_leader_bears_on_its_place_but_does_not_strike_it_in_the_check_sense() {
    let text = text_of(Some("variants/atomic.gdl"), "atomic");
    let k = KernelGame::new("atomic", 518).unwrap();
    let (kk, pairs) = pairs_of(&k, "8/8/8/3kK3/8/8/8/8 w - - 0 1", None);
    let mut t = placed_atomic::Game::make();
    let (m_bears, e_bears) = rows_both(&mut t, &text, &pairs, "placed_strikes", 3);
    let (m_end, e_end) = rows_both(&mut t, &text, &pairs, "placed_strikes_end", 3);
    let d5 = 3 * 8 + 4;                                                                      // file d, rank 5: the black leader's place
    let kernel = kk.strikes_place(&kk.initial(), 0, d5);
    let has = |s: &BTreeSet<Vec<String>>| s.contains(&vec!["white".to_string(), "d".to_string(), "5".to_string()]);
    println!("OBS 79.48 E known difference atomic (8/8/8/3kK3/8/8/8/8 w): the kernel says white strikes d5: {kernel}; placed_strikes (bears on) module {} evaluator {}; placed_strikes_end (check notion) module {} evaluator {}", has(&m_bears), has(&e_bears), has(&m_end), has(&e_end));
    assert!(has(&m_bears) == has(&e_bears) && has(&m_end) == has(&e_end), "the module is the description's");
    assert!(kernel && has(&m_bears) && !has(&m_end), "the two meanings agree here: the difference is gone");
}
