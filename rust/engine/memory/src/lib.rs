//! PLAN 78.14 THE MEMORY LAYER: what the engine keeps of what it learned, in a form a person can read and check. Game-free: it reads only the `Game` trait.
//!   `sha`       the content address of a snapshot.
//!   `lemma`     the lemma language (restricted Datalog over the game's fact vocabulary) and its JSON.
//!   `matcher`   the one join that binds a lemma's body to facts.
//!   `snapshot`  an immutable, content-addressed store state: the game's vocabulary, its judge, its lemmas, its parent.
//!   `cards`     the corpus a lemma is verified on: positions with their facts and the exact value of every legal move; the evaluation of a lemma over it.
//!   `op`        the lemma operator: binds the snapshot's lemmas to the position and proposes moves (a prior, never a verdict).
pub mod cards;
pub mod lemma;
pub mod matcher;
pub mod op;
pub mod sha;
pub mod snapshot;

use cs_game::Fact;

/// THE PERSPECTIVE of the side to move (plan 78.14): the roles' names in the facts replaced by `us` (the mover) and `them` (the other side; with more than two
/// roles `r1`, `r2` ... in seat order after the mover). A lemma is written in this perspective, so one lemma serves both colours and the same fact reads the same
/// from either seat. A game whose roles are not alike loses nothing but reach: verification judges every lemma in this encoding on the cards made the same way.
pub fn perspective(facts: &[Fact], roles: &[String], mover: usize) -> Vec<Fact> {
    let name_of = |c: &str| -> Option<String> {
        let i = roles.iter().position(|r| r == c)?;
        let seat = (i + roles.len() - mover) % roles.len();
        Some(if seat == 0 { "us".to_string() } else if roles.len() == 2 { "them".to_string() } else { format!("r{seat}") })
    };
    let mut out: Vec<Fact> = facts.iter().map(|f| Fact { relation: f.relation.clone(), args: f.args.iter().map(|a| name_of(a).unwrap_or_else(|| a.clone())).collect() }).collect();
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_mover_is_us_whichever_seat() {
        let roles = vec!["white".to_string(), "black".to_string()];
        let f = vec![Fact::new("cell", &["a", "1", "white", "rook"]), Fact::new("cell", &["a", "8", "black", "rook"])];
        let w = perspective(&f, &roles, 0);
        let b = perspective(&f, &roles, 1);
        assert!(w.contains(&Fact::new("cell", &["a", "1", "us", "rook"])) && w.contains(&Fact::new("cell", &["a", "8", "them", "rook"])));
        assert!(b.contains(&Fact::new("cell", &["a", "1", "them", "rook"])) && b.contains(&Fact::new("cell", &["a", "8", "us", "rook"])));
    }
}
