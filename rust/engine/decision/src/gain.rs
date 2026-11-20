//! PLAN 79.13 C, THE VALUE OF A SIDE'S CHAINS: STAKE TIMES CHANCE (SYSTEM_SPEC 7 "among equal moves we optimize for the prediction ... whichever gets us closer to our goals", point 2 the estimate per goal; 8 the rule for choosing a move; piece: the estimate per goal). It is a NUMBER ABOUT A POSITION that the last layer reads as an input; it is never added to the judge's value (rule 7).
//! A CHAIN is a goal with the goals of its stoppers under it (the recursion of goals.rs); the chain's stake is the stake of the goal at its top (rule.rs `chain_top`), and its chance is the chance of that top goal (the estimate per goal). The VALUE of a side's chains is the sum over its chains of stake times chance. The stake of the goal kind is above every worth and is never a number here
//! (no worth is assumed, so no worth is invented for it): the chains whose top is the goal kind are summed apart, by their chance alone.
//!
//! INTERFACE (written before the bodies; who calls: `after.rs` for each position it reads, `wake::decide`, the unit tests).
//!   `ChainValue { finite, goal }`                      finite: the sum of stake times chance over the chains whose top stake is a worth; goal: the sum of the chances of the chains whose top is the goal kind.
//!   `chain_value(goals, side, stakes, chance)`         the value of the chains of `side` among `goals` (a goal counts when it has an unmet fact, as in `rule::named_fixes`); `chance(goal, stakes)` is the chance of a goal.
//!   `ChainValue::less(&self, other)`                   self minus other, field by field: the gain from a position to another.
use crate::goals::Goal;
use crate::rule::{chain_top, Stakes};
use cs_game::Role;
use std::collections::BTreeSet;

/// The value of a side's chains: the finite part and the part of the goal kind.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChainValue { pub finite: f64, pub goal: f64 }

impl ChainValue {
    /// This value less another: the gain from the other to this.
    pub fn less(&self, other: &ChainValue) -> ChainValue { ChainValue { finite: self.finite - other.finite, goal: self.goal - other.goal } }
}

/// The value of the chains of `side` among `goals`: one term a chain, at its top goal.
pub fn chain_value(goals: &[Goal], side: Role, stakes: &Stakes, chance: &dyn Fn(&Goal, &Stakes) -> f64) -> ChainValue {
    let tops: BTreeSet<usize> = goals.iter().enumerate().filter(|(_, x)| x.side == side && !x.unmet.is_empty()).map(|(i, _)| chain_top(goals, i)).collect();
    let mut v = ChainValue::default();
    for t in tops {
        let g = &goals[t];
        let c = chance(g, stakes);
        let stake = stakes.of(g.target);
        if stake.is_finite() { v.finite += stake * c; } else { v.goal += c; }
    }
    v
}
