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
//!   `Formula::validate(&self) -> Result<(), FormulaError>`        the well-formedness: every branch of an `Or` binds the same placeholders; a `Not`, an `Every`, an `AtLeast`, a negated literal and a `differ` read only placeholders that a positive part of the same
//!                                                              `And` binds (or that the caller's initial binding holds).
//!   `answers(&Formula, &Binding, &FactIndex, limit, &mut steps) -> Result<(Vec<Binding>, bool), FormulaError>`   every binding of the free placeholders that makes the formula true over the index, extending the initial binding; on a body of literals alone
//!                                                              it gives the bindings the matcher gives (the literals of an `And` are handed to the matcher, the one part that binds a body to facts).
//!   `holds(&Formula, &Binding, &FactIndex) -> Result<bool, FormulaError>`   whether at least one answer exists.
//!   LAWS: lean/ideas/Ideas/Layer0/Logic.lean (the quantifier laws used, De Morgan over a finite set, "at least so many" is monotone in the count).
