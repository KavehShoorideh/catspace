//! Plan 78.93: the mode of the verification of the choice (`ChoiceMode`: Slot or Main). Slot is 78.92's form and the default everywhere; Main takes the verification's time out of the main search's clock.
use cs_game::kernel::KernelGame;
use cs_tactics::ledger::{ChoiceMode, Search, Worth, CHOICE_MAIN_SHARE};
use std::collections::HashMap;

fn record() -> Worth {
    let table: HashMap<String, i64> = [("q", 4343), ("r", 1605), ("b", 1878), ("n", 1661), ("p", 1045)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
    Worth { name: "t".into(), table, ours_first: false }
}

#[test]
fn the_default_mode_is_slot_and_the_switch_is_off() {
    let g = KernelGame::from_fen("chess", "4k3/8/8/8/8/8/8/4K2R w - - 0 1").unwrap();
    let j = record();
    for s in [Search::new(&g, &j, 0, 1.0), Search::with_fuse(&g, &j, 0, 10)] {
        assert_eq!(s.choice_mode, ChoiceMode::Slot);
        assert!(!s.verify_choice);
    }
    assert_eq!(cs_tactics::operators::LedgerOp::new(&j).choice_mode, ChoiceMode::Slot);
}

#[test]
fn the_names_parse() {
    assert_eq!(ChoiceMode::parse("slot"), Some(ChoiceMode::Slot));
    assert_eq!(ChoiceMode::parse("main"), Some(ChoiceMode::Main));
    assert_eq!(ChoiceMode::parse("other"), None);
}

#[test]
fn slot_changes_no_clock_and_main_keeps_part_of_the_main_clock_for_the_verification() {
    // switch off: no mode changes the deepening's clock
    for m in [ChoiceMode::Slot, ChoiceMode::Main] { assert_eq!(m.deepening_seconds(false, 0.08), 0.08); }
    // slot with the switch on: the deepening has all of it, the verification the equal reading's slot
    assert_eq!(ChoiceMode::Slot.deepening_seconds(true, 0.08), 0.08);
    assert_eq!(ChoiceMode::Slot.verification_seconds(0.08, 0.08, 0.12, 0.5), 0.06);
    // main: the deepening stops with the share left; the verification gets what is left of the whole clock (never more than the rest of the decision)
    let d = ChoiceMode::Main.deepening_seconds(true, 0.08);
    assert!((d - 0.08 * (1.0 - CHOICE_MAIN_SHARE)).abs() < 1e-12);
    assert!((ChoiceMode::Main.verification_seconds(0.08, 0.04, 0.12, 0.5) - 0.04).abs() < 1e-12);
    assert_eq!(ChoiceMode::Main.verification_seconds(0.08, 0.04, 0.02, 0.5), 0.02);
    assert_eq!(ChoiceMode::Main.verification_seconds(0.08, 0.09, 0.12, 0.5), 0.0);
}
