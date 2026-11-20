//! PLAN 79.15 B, THE WALL KEPT (SYSTEM_SPEC 8 "the wall" and "the rule for choosing a move"; piece: the wake). The wake never plays a move its own calculation has put a unit lower. The search's first move was put at the head of the moves held equal whether the equal reading kept it or not
//! (wake.rs, 79.12 A); when the clock cut the verification the set was empty and the first move played alone, even where its own probe was far below the search's value and other moves' probes held (game bc_v_old_A:A9100:1, ply 21 of 79.14: the first move -1301, nine others 0, the value 0).
//!
//! INTERFACE (written before the body). `candidates(own, value, probes, held, verified)` takes the search's first root move (an index into the root's legal moves; None: the search read no depth), the search's value, the null-window answers of the equal reading
//! (`probes`: root move, answer, in the order they were asked; a move not asked has none), the moves the verification held (`held`; empty where it did not run or was cut) and whether the verification ran to its end. It returns `Candidates`: the moves among which the rule, the chooser and the
//! reasons work (the old `equal` set), how the first move stood (`Lowered`), whether the candidates are held by their first reading only (`unverified`: the clock cut the verification) and by what the set was made (`Held`).
//!   (1) THE FIRST MOVE IS NOT LOWERED (its probe holds the value and the verification, if it ran, held it; or it was not asked): the set is main's: the moves held, the first at their head, whether the equal reading kept it or not is not asked of a move that was not read.
//!   (2) THE FIRST MOVE IS LOWERED (its probe is below the value: `ByProbe`; or its probe held and the verification did not: `ByVerification`): it is not a candidate while any other move holds. The candidates are the moves the verification held; where the verification was cut, the moves whose probe held,
//!       marked unverified; where no other move holds, the one move with the highest probe (the first in the order they were asked on a tie).
//! Called by: `wake::decide` (step 5). Game-free; reads nothing but the numbers it is given; no clock, no search.

/// How the search's first move stood at the equal reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lowered { No, ByProbe, ByVerification }

/// What the candidate set was made from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Held { /// as before: the moves held (the search's first move at their head)
    AsBefore,
    /// the moves the verification held, the first move not among them
    Verified,
    /// the verification was cut: the moves whose probe held
    ProbeOnly,
    /// no other move holds: the one move with the highest probe
    Highest }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidates { pub moves: Vec<usize>, pub lowered: Lowered, pub unverified: bool, pub held: Held }

/// The candidates of the choice (see the module doc).
pub fn candidates(own: Option<usize>, value: i64, probes: &[(usize, i64)], held: &[usize], verified: bool) -> Candidates {
    let main = |held: &[usize]| { let mut v = held.to_vec(); if let Some(o) = own { v.retain(|i| *i != o); v.insert(0, o); } Candidates { moves: v, lowered: Lowered::No, unverified: false, held: Held::AsBefore } };
    let Some(o) = own else { return main(held) };
    let Some(own_probe) = probes.iter().find(|(i, _)| *i == o).map(|(_, v)| *v) else { return main(held) };
    let lowered = if own_probe < value { Lowered::ByProbe } else if verified && !held.contains(&o) { Lowered::ByVerification } else { Lowered::No };
    if lowered == Lowered::No { return main(held); }
    if verified {
        let c: Vec<usize> = held.iter().copied().filter(|i| *i != o).collect();
        if !c.is_empty() { return Candidates { moves: c, lowered, unverified: false, held: Held::Verified }; }
    } else {
        let c: Vec<usize> = probes.iter().filter(|(i, v)| *v >= value && *i != o).map(|(i, _)| *i).collect();
        if !c.is_empty() { return Candidates { moves: c, lowered, unverified: true, held: Held::ProbeOnly }; }
    }
    // no other move holds: the highest probe (the first asked on a tie)
    let best = probes.iter().fold(None::<(usize, i64)>, |b, (i, v)| match b { Some((_, bv)) if bv >= *v => b, _ => Some((*i, *v)) });
    Candidates { moves: best.into_iter().map(|(i, _)| i).collect(), lowered, unverified: !verified, held: Held::Highest }
}
