//! PLAN 78.14 THE MEMORY LAYER: what the engine keeps of what it learned, in a form a person can read and check. Game-free: it reads only the `Game` trait.
//!   `sha`       the content address of a snapshot.
//!   `lemma`     the lemma language (restricted Datalog over the game's fact vocabulary) and its JSON.
//!   `matcher`   the one join that binds a lemma's body to facts.
//!   `snapshot`  an immutable, content-addressed store state: the game's vocabulary, its judge, its lemmas, its parent.
//!   `cards`     the corpus a lemma is verified on: positions with their facts and the exact value of every legal move; the evaluation of a lemma over it.
//!   `claim`     (plan 78.36) the claim rule -- a position's class for the side to move -- and the reader the decision's search uses at its leaves.
//!   `op`        the lemma operator: binds the snapshot's lemmas to the position and proposes moves (a prior, never a verdict).
//!
//! PLAN 79.19 LAYER 1: ONE FORM OF STATEMENT, TWO MEMORIES, AND A WAY TO CHANGE THEM (SYSTEM_SPEC 11 "the representation says all of this", "two memories", "the go on layer 1"; ledger rows 36, 34, 14, 13). THE INTERFACE, written before any body; the bodies are in `statement` and `store`.
//!   `statement`  THE STATEMENT, the one form in which every layer writes what it knows.
//!       `Statement { id, pattern, says, standing, holder, also }`
//!         pattern   literals of the lemma language over the graph's facts (variables `?x`, constants, negation as failure): the placeholders. Bound by `matcher`, the one join; no second matcher.
//!         says      `Says::Holds` (the pattern holds), `Proposes(Response)` (a move of this effect is proposed: a lemma's head), `Number(f64)` (a number about the pattern), `Lower { term, within }` (a term, in the text of its own layer, to be lowered within `within` of our moves).
//!         standing  `Standing::Fact` (follows from the rules at the position) | `Proved(theorem)` (the name of the Lean theorem) | `Conjecture(sample)` (held on the named sample, not proved) | `Belief { probability, count }` (counted from results).
//!         holder    `Holder::Ours` | `Holder::OursAboutOther` (our belief about what the other side holds: never their state itself; one level, no nesting). Sayable and tested; nothing is put in it by this plan.
//!         also      what else it READ beside the facts its pattern names (`Read::Key(text)`: a key of the position, a count, a file's address).
//!       `Statement::reads() -> Vec<Read>`           every literal of its pattern as `Read::Facts(lit)` (the facts that unify with it, positive or negated) and `also`.
//!       `Statement::bindings(&FactIndex, limit, &mut steps) -> Vec<Binding>`   where the pattern binds on a position.
//!       `Statement::falls_by(&Change) -> bool`      true exactly when a fact the change added or removed unifies with a literal of its pattern, or a key the change names is in its `also`.
//!       `Change { added, removed, keys }`; `Change::between(before, after)` from two lists of facts. `Statement::to_json / from_json`: a statement is written and read back identical.
//!   `store`  THE TWO MEMORIES and the four changes.
//!       `GameKey { name, rules }`   what the game itself reports: its name and the content address of its rules (`GameKey::of(name, rules_text)`). The key is never a word typed in a layer crate.
//!       `Kept` (trait)              a typed thing a memory keeps (a trained model, a table, a held goal): `kind()`, `standing()`, `describe() -> (takes, returns)`, `to_json()`, `as_any()`. It is reached only through its own interface; a layer reads it as its own type (`Reader::get::<T>(name)`).
//!       `LongTerm`                  ONE store for every game the system learns, partitioned by `GameKey`. A partition is an immutable, content-addressed `Partition` (statements and entries) with its parent. `LongTerm::reader(&GameKey) -> Option<Reader>`.
//!       `Reader`                    what the wake holds: opened for ONE game, it sees that game's partition only and has no way to change it. `statement(id)`, `statements()`, `entry(name)`, `get::<T>(name)`, `id()`, `parent()`, `bound(&FactIndex)`.
//!       `LongTerm::sleep() -> Sleep` the only door of change, open to sleep and never taken in the wake: `Sleep::add / replace / retract / fall_by_change`, each naming its partition and returning the id of the NEW snapshot (the old one stays, as the new one's parent).
//!       `GameMemory`                the memory of the game at hand: statements and typed slots; written by the wake, emptied by `end()`, which hands back what surprised it (the statements that fell). Its four changes are in place and each is logged: `add / replace / retract / fall_by_change`.
//!   THE FOUR CHANGES (the mutation interface of the representation layer): ADD a statement or entry that is not there; REPLACE one that is; RETRACT one by id; FALL BY CHANGE: given what a move changed (`Change`), every statement that read it is dropped and listed. A layer's scratch that it does not want kept (the search's table for one decision) stays inside that layer.
//!   Called by: cs_decision (the wake reads one `Reader` and writes one `GameMemory`), the sleep side (the only caller of `Sleep`), the unit tests. Dependency direction: cs_memory depends on cs_game and cs_operator only; a type of a crate above (a table, the open tactics) is kept behind `Kept` and decoded by a registered function (`Decoders`), never copied here.
pub mod cards;
pub mod claim;
pub mod op;
// plan 79.21: the lemma language, the matcher, the statement, the store with both memories and the snapshot moved down to the representation crate (cs_rep); re-exported here so every path `cs_memory::lemma::..` keeps resolving
pub use cs_rep::{history, lemma, matcher, sha, snapshot, statement, store};

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
