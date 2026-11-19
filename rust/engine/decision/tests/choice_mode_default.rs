//! Plan 78.93: the decision's defaults keep the verification of the choice off and its mode at Slot (78.92's form).
use cs_tactics::ledger::ChoiceMode;

#[test]
fn the_defaults_keep_the_search_as_it_is() {
    let p = cs_decision::Params::default();
    assert!(!p.verify_choice && !p.verify_ties);
    assert!(!p.reask, "plan 79.02: the second asking of the equal reading is off by default");
    assert_eq!(p.choice_mode, ChoiceMode::Slot);
}
