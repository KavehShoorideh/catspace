//! CERTIFY BY CO-PLAY (plan 78.1): a fast binding against its oracle, both behind the `Game` trait, played side by side from their initial states -- at every
//! state the side to move, the legal move names (the fast one's names mapped by the binding's naming), the terminal verdict and the outcome compared; a
//! seeded random legal move then played in both. No state of one is translated into the other: the two games agree when every trajectory reads the same. A
//! game's first disagreement ends that game (the states have parted). Generic: nothing here knows a game.
use crate::rng::Rng;
use crate::Game;
use std::collections::BTreeSet;

#[derive(Debug, Default, Clone)]
pub struct Report {
    pub games: usize,
    pub plies: usize,
    pub states: usize,
    /// The states whose facts both sides gave and agreed on (the fact layer's certificate).
    pub fact_states: usize,
    pub terminals: Vec<String>,
    pub disagreements: Vec<String>,
}

/// Co-play `games` seeded random games of `fast` against `oracle`; `name` maps a move name of `fast` to the oracle's name of the same move.
pub fn coplay<A: Game, B: Game>(fast: &A, oracle: &B, name: &dyn Fn(&str) -> String, games: usize, seed: u64) -> Report {
    let mut rep = Report::default();
    let mut rng = Rng::new(seed);
    let roles = fast.roles();
    for g in 0..games {
        rep.games += 1;
        let (mut a, mut b) = (fast.initial(), oracle.initial());
        let mut line: Vec<String> = vec![];
        loop {
            rep.states += 1;
            let (oa, ob) = (fast.outcome(&a), oracle.outcome(&b));
            if oa != ob {
                rep.disagreements.push(format!("game {g} after {:?}: outcome {oa:?} against the oracle's {ob:?}", line));
                break;
            }
            if let Some(o) = oa {
                rep.terminals.push(format!("{o:?}"));
                break;
            }
            let (fa, fb) = (fast.base_facts(&a), oracle.base_facts(&b));
            if !fa.is_empty() && !fb.is_empty() && fa != fb {
                let only_a: Vec<String> = fa.iter().filter(|f| !fb.contains(f)).map(|f| f.to_string()).collect();
                let only_b: Vec<String> = fb.iter().filter(|f| !fa.contains(f)).map(|f| f.to_string()).collect();
                rep.disagreements.push(format!("game {g} after {:?}: facts -- the binding only {only_a:?}, the oracle only {only_b:?}", line));
                break;
            }
            rep.fact_states += (!fa.is_empty() && !fb.is_empty()) as usize;
            let (ma, mb) = (fast.mover(&a), oracle.mover(&b));
            if ma.map(|r| roles[r].clone()) != mb.map(|r| oracle.roles()[r].clone()) {
                rep.disagreements.push(format!("game {g} after {:?}: the mover {ma:?} against the oracle's {mb:?}", line));
                break;
            }
            let la = fast.moves(&a);
            let lb = oracle.moves(&b);
            let na: BTreeSet<String> = la.iter().map(|m| name(&fast.move_name(&a, m))).collect();
            let nb: BTreeSet<String> = lb.iter().map(|m| oracle.move_name(&b, m)).collect();
            if na != nb {
                let only_a: Vec<_> = na.difference(&nb).cloned().collect();
                let only_b: Vec<_> = nb.difference(&na).cloned().collect();
                rep.disagreements.push(format!("game {g} after {:?}: legal moves -- the binding only {only_a:?}, the oracle only {only_b:?}", line));
                break;
            }
            let mut sorted: Vec<(String, usize)> = la.iter().enumerate().map(|(i, m)| (fast.move_name(&a, m), i)).collect();
            sorted.sort();
            let (nm, i) = sorted[rng.below(sorted.len())].clone();
            let mapped = name(&nm);
            let j = lb.iter().position(|m| oracle.move_name(&b, m) == mapped).expect("the move sets agree");
            a = fast.apply(&a, &la[i]);
            b = oracle.apply(&b, &lb[j]);
            line.push(nm);
            rep.plies += 1;
        }
    }
    rep
}
