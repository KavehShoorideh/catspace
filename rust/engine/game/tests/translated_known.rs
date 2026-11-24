//! PLAN 79.47 PART C (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 45): THE KNOWN CAUSE OF A DISAGREEMENT BETWEEN A KERNEL AND THE DESCRIPTION'S MODULE, HELD AS A STATE.
//! The fault of plan 79.36 in the game with a start by number (fault 3 of the 79.36 READ): the description reads the corner unit of a wing as the OUTERMOST one of the start, and a right names no file, so a state shown with another holder of the right is not said; the kernel
//! allows the two-unit move with that unit. The kernel's record (translated_kernel.rs, kernel_translated.py) did NOT meet this cause: the seeded games from the start numbered 518, from the 31 listed starts and its large run never reach a state where a right is held by a corner unit that is
//! not the outermost (the large run's walk plays from the kernel's start and the corner starts of the list give the right to the outermost unit). This file holds the state where it is, as an ignored test: run by hand it compares the kernel and the module at that one state and prints
//! both answers, so that the comparison is shown to SEE the cause when a state reaches it. A mending is the description's (it needs the right's file as a fluent, `state_facts` of outside.rs, and a clause), not this file's.
//! This file names the game on purpose (a test file may). It is not part of the hashes of a record (it makes no count).
mod translated_common;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::Game as _;
use std::collections::BTreeSet;
use translated_common::*;

include!("generated/chess960_fast.rs");
impl_tr!(chess960_fast, passes);

#[test]
#[ignore = "known cause (plan 79.36, fault 3): the description of the game with a start by number reads the corner unit of a wing as the outermost one, so a right held by another corner unit is not said while the kernel allows the two-unit move with it; run by hand to see the comparison find it"]
fn chess960_known_cause_a_right_held_by_a_corner_unit_that_is_not_the_outermost() {
    let text = description(Some("variants/chess960.gdl"));
    let fen = "4k3/8/8/8/8/8/8/1RR1K3 w C - 0 1";
    let k = KernelGame::from_fen("chess960", fen).unwrap();
    let ks = k.initial();
    let facts = k.state_facts(&ks);
    let mut t = chess960_fast::Game::make();
    t.restart_t(&facts.iter().map(|f| (f.relation.clone(), f.args.clone())).collect::<Vec<_>>()).unwrap();
    let ts = t.state_of_facts_t(&facts.iter().map(|f| (f.relation.clone(), f.args.clone())).collect::<Vec<_>>()).unwrap();
    let km: BTreeSet<String> = k.moves(&ks).iter().map(|m| spec_term(&k.move_name(&ks, m))).collect();
    let tm: BTreeSet<String> = t.moves_t(&ts).iter().map(|m| t.render_t(*m)).collect();
    let ev = evaluator_at(&text, &facts.iter().cloned().collect());
    let em: BTreeSet<String> = ev.moves(&ev.initial()).iter().map(|m| ev.move_name(&ev.initial(), m)).collect();
    println!("OBS 79.47 C known cause chess960 ({fen}): only the kernel allows {:?}, only the module allows {:?}; the evaluator begun there allows the module's moves: {}", km.difference(&tm).collect::<Vec<_>>(), tm.difference(&km).collect::<Vec<_>>(), em == tm);
    assert!(km != tm, "the kernel and the module agree here: the cause is gone");
    assert_eq!(em, tm, "the module is the description's");
}
