//! LAYER 0, MODULE 1: LOGIC (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 1). The statements layers 1 and up write and read: the fact (a relation and its arguments, as text), the argument (a placeholder or a constant), the literal, the one literal that is not
//! a row (`differ`), the well-formedness of a body, the index of facts, the matcher, and a FORMULA over literals. Moved here from cs_rep (fact.rs, lemma.rs, matcher.rs), which hands them on under their old paths; nothing above changes.
//!
//! INTERFACE (written before the bodies; callers: cs_rep re-exports the first block; cs_memory, cs_game, cs_tactics use them through cs_rep; the second block is new and has no caller yet outside the tests).
//!   MOVED, THE SAME TEXT AS BEFORE (mod.rs, matcher.rs, json.rs):
//!   `Fact { relation: String, args: Vec<String> }`, `Fact::new(&str, &[&str]) -> Fact`, `Display`      a ground fact as text.
//!   `Arg::{Var(String), Const(String)}`, `Arg::parse(&str) -> Arg`, `Arg::text(&self) -> String`          a placeholder (a text that starts with `?`) or a constant.
//!   `Lit { neg: bool, relation: String, args: Vec<Arg> }`, `Lit::pos`, `Lit::differ`, `Lit::is_differ`, `Lit::vars(&self) -> BTreeSet<String>`   a literal.
//!   `Lit::to_json(&self) -> serde_json::Value`, `Lit::from_json(&Value) -> Result<Lit, String>`            its JSON form (json.rs; the one outside crate, serde_json).
//!   `DIFFER: &str`                                                                                        the one literal that is not a row: "these two stand for different things".
//!   `validate_body(at: &str, body: &[Lit]) -> Result<(), String>`                                         a name, at least one positive row literal, every placeholder of a negated or differ literal bound by one.
//!   `Binding = BTreeMap<String, String>`, `FactIndex::{new(&[Fact]), rows(&str) -> &[Vec<String>], contains(&str, &[String]) -> bool}`.
//!   `bindings(body, &init, &idx, limit, &mut steps) -> (Vec<Binding>, bool)`, `bindings_fused(.., fuse) -> (Vec<Binding>, bool)`   all bindings of a body of literals; the bool says the limit or the fuse cut the search.
//!   NEW, THE FORMULA (formula.rs):
//!   `Domain::{Set(Set<String>), Relation(String)}`             the finite set a quantifier ranges over: a given set of constants, or the first arguments of the rows of a relation of one argument in the index.
//!   `Formula::{Lit, And, Or, Not, Some, Every, AtLeast}`        and, or, not, some (a placeholder over a finite set), every (likewise), at least so many (`AtLeast { count, var, over, body }`).
//!   `Formula::free_vars(&self) -> BTreeSet<String>`            the placeholders the formula leaves free.
//!   `Formula::needs(&self) -> BTreeSet<String>`                the placeholders the caller's initial binding must hold: what the formula's filters (`Not`, `Every`, `AtLeast`, a negated literal, a `differ`) read that nothing in the formula binds.
//!   `Formula::validate(&self)` / `Formula::check(&self, bound)` -> `Result<(), FormulaError>`   the well-formedness: every branch of an `Or` binds the same placeholders besides those already bound outside it. A filter that reads a placeholder nothing binds is
//!                                                              `FormulaError::Unbound` when the formula is read.
//!   `answers(&Formula, &Binding, &FactIndex, limit, &mut steps) -> Result<(Vec<Binding>, bool), FormulaError>`   every binding of the free placeholders that makes the formula true over the index, extending the initial binding, as a SET (a binding once; the matcher gives
//!                                                              one for each row that matches). On a body of literals alone it gives the bindings the matcher gives, as a set (the literals of an `And` are handed to the matcher, the one part that binds a body to facts).
//!   `holds(&Formula, &Binding, &FactIndex) -> Result<bool, FormulaError>`   whether at least one answer exists.
//!   LAWS: lean/ideas/Ideas/Layer0/Logic.lean (the quantifier laws used, De Morgan over a finite set, "at least so many" is monotone in the count).
use std::collections::BTreeSet;

pub mod formula;
pub mod json;
pub mod matcher;

pub use formula::{answers, holds, Domain, Formula, FormulaError};
pub use matcher::{bindings, bindings_fused, Binding, FactIndex};

/// A ground fact as text: a relation name and its arguments (the store's language; the layers know no game).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fact { pub relation: String, pub args: Vec<String> }

impl Fact {
    pub fn new(relation: &str, args: &[&str]) -> Fact { Fact { relation: relation.to_string(), args: args.iter().map(|a| a.to_string()).collect() } }
}

impl std::fmt::Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.args.is_empty() { write!(f, "{}", self.relation) } else { write!(f, "({} {})", self.relation, self.args.join(" ")) }
    }
}

/// An argument of a literal: a placeholder or a constant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arg { Var(String), Const(String) }

impl Arg {
    /// An argument starting with `?` is a placeholder, any other text a constant.
    pub fn parse(s: &str) -> Arg { match s.strip_prefix('?') { Some(v) => Arg::Var(v.to_string()), None => Arg::Const(s.to_string()) } }
    pub fn text(&self) -> String { match self { Arg::Var(v) => format!("?{v}"), Arg::Const(c) => c.clone() } }
}

/// PLAN 79.28 PART C (SYSTEM_SPEC 11 "Reached means the goal outlasts the reply", 7 goals; ledger row 14), moved to layer 0 by plan 79.30. THE ONE LITERAL OF THE LANGUAGE THAT IS NOT A ROW: `differ(a, b)`, two arguments (placeholders or constants), says that the two stand for
/// DIFFERENT things. It is written and read back as every literal is ({"relation": "differ", "args": ["?x", "?y"], "neg": false}), so a statement stored before it existed reads unchanged (no stored statement used the word as a relation). It is not matched over rows: the matcher
/// tests it once both arguments are bound (a placeholder it names must be bound by a positive literal that is a row, as for a negated literal). Negated, `differ(a, b)` says they are the SAME. No fact of any game has this relation.
pub const DIFFER: &str = "differ";

/// A literal: a relation over arguments, positive or negated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lit { pub neg: bool, pub relation: String, pub args: Vec<Arg> }

impl Lit {
    /// The literal that says two placeholders (or a placeholder and a constant) stand for different things.
    pub fn differ(a: &str, b: &str) -> Lit { Lit::pos(DIFFER, &[a, b]) }
    /// Whether this is the language's one literal that is not a row.
    pub fn is_differ(&self) -> bool { self.relation == DIFFER }
    pub fn pos(relation: &str, args: &[&str]) -> Lit { Lit { neg: false, relation: relation.to_string(), args: args.iter().map(|a| Arg::parse(a)).collect() } }
    /// The negated literal over the same relation and arguments (a literal negated twice is the literal).
    pub fn negated(&self) -> Lit { Lit { neg: !self.neg, relation: self.relation.clone(), args: self.args.clone() } }
    pub fn vars(&self) -> BTreeSet<String> { self.args.iter().filter_map(|a| if let Arg::Var(v) = a { Some(v.clone()) } else { None }).collect() }
}

/// The well-formedness of a body (a lemma's or a claim's): a name, at least one positive literal, negated literals safe (every variable bound by a positive one).
pub fn validate_body(at: &str, body: &[Lit]) -> Result<(), String> {
    if at.is_empty() { return Err("a lemma needs a name".into()); }
    if !body.iter().any(|l| !l.neg && !l.is_differ()) { return Err(format!("{at}: the body needs a positive literal")); }
    let bound: BTreeSet<String> = body.iter().filter(|l| !l.neg && !l.is_differ()).flat_map(|l| l.vars()).collect();
    for l in body.iter().filter(|l| l.neg || l.is_differ()) {
        if l.is_differ() && l.args.len() != 2 { return Err(format!("{at}: differ takes two arguments")); }
        if let Some(v) = l.vars().into_iter().find(|v| !bound.contains(v)) { return Err(format!("{at}: the {} literal {} uses ?{v}, which no positive literal binds", if l.neg { "negated" } else { "differ" }, l.relation)); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_argument_is_a_placeholder_when_it_starts_with_a_question_mark() {
        assert_eq!(Arg::parse("?x"), Arg::Var("x".into()));
        assert_eq!(Arg::parse("x"), Arg::Const("x".into()));
        assert_eq!(Arg::parse("?x").text(), "?x");
        assert_eq!(Fact::new("r", &["a", "b"]).to_string(), "(r a b)");
        assert_eq!(Fact::new("p", &[]).to_string(), "p");
    }

    #[test]
    fn a_body_needs_a_positive_row_and_safe_negations() {
        let pos = Lit::pos("r", &["?x", "?y"]);
        assert!(validate_body("b", &[pos.clone()]).is_ok());
        assert!(validate_body("", &[pos.clone()]).is_err());
        assert!(validate_body("b", &[]).is_err());
        assert!(validate_body("b", &[Lit::differ("?x", "?y")]).unwrap_err().contains("positive"));
        assert!(validate_body("b", &[pos.clone(), Lit::differ("?x", "?y")]).is_ok());
        assert!(validate_body("b", &[pos.clone(), Lit::differ("?x", "?z")]).unwrap_err().contains("binds"));
        assert!(validate_body("b", &[pos.clone(), pos.negated().negated()]).is_ok());
        let mut loose = Lit::pos("s", &["?z"]);
        loose.neg = true;
        assert!(validate_body("b", &[pos.clone(), loose]).unwrap_err().contains("negated"));
        let mut bad = Lit::differ("?x", "?y");
        bad.args.push(Arg::Const("w".into()));
        assert!(validate_body("b", &[pos, bad]).unwrap_err().contains("two arguments"));
    }

    #[test]
    fn the_differ_literal_is_the_one_that_is_not_a_row() {
        assert!(Lit::differ("?x", "y").is_differ());
        assert!(!Lit::pos("r", &["?x"]).is_differ());
        assert_eq!(Lit::differ("?x", "?y").vars().len(), 2);
        assert_eq!(DIFFER, "differ");
    }
}
