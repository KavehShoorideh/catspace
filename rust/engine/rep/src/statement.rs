//! THE STATEMENT (plan 79.19 part A; SYSTEM_SPEC 11 "what the representation layer must be able to say", "the representation says all of this for the opponent too"). The interface is in the crate's doc comment (lib.rs); this module is its body. Game-free: it reads facts only through `Fact`.
//! A statement is a pattern over the graph's facts with placeholders (the lemma language as it is: `lemma::Lit`, bound by `matcher`, the one join), what it says of the pattern, its standing, its holder, and what it read. It is written to JSON and read back identical.
//! A statement FALLS when something it read changes (`falls_by`): a fact a change added or removed that unifies with a literal of its pattern (positive or negated), or a key the change names that is in its `also`. The reading is conservative on purpose: a statement that did not fall has the
//! same bindings on the position after as before (tested over generated games of two games), a statement that fell may still bind the same (the over-count is printed by the test).
use crate::lemma::{validate_body, Arg, Lit, LitJson, Response};
use crate::matcher::{bindings_fused, Binding, FactIndex};
use crate::fact::Fact;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// What a statement is worth: how it came to be held.
#[derive(Debug, Clone, PartialEq)]
pub enum Standing {
    /// follows from the rules at the position
    Fact,
    /// proved: the name of the Lean theorem
    Proved(String),
    /// held on a named sample and not proved
    Conjecture(String),
    /// counted from results: a probability and the count behind it
    Belief { probability: f64, count: u64 },
}

/// Whose it is: ours, or ours about the other side (our belief about what they hold; never their state itself; one level).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Holder { Ours, OursAboutOther }

/// What the statement says of its pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum Says {
    /// the pattern holds
    Holds,
    /// a move with this effect on the delta relations is proposed (a lemma's head)
    Proposes(Response),
    /// a number about the pattern
    Number(f64),
    /// a term (in the text of its own layer) to be lowered within `within` of our moves
    Lower { term: String, within: u32 },
}

/// What a statement read: the facts that unify with a literal, or a key (of a position, a count, a file's address).
#[derive(Debug, Clone, PartialEq)]
pub enum Read { Facts(Lit), Key(String) }

#[derive(Debug, Clone, PartialEq)]
pub struct Statement { pub id: String, pub pattern: Vec<Lit>, pub says: Says, pub standing: Standing, pub holder: Holder,
    /// what it read beside the facts its pattern names
    pub also: Vec<String> }

/// What a move changed: the facts added and removed, and keys named (a key a statement read that is not a fact).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Change { pub added: Vec<Fact>, pub removed: Vec<Fact>, pub keys: Vec<String> }

impl Change {
    /// The change from one list of facts to the next.
    pub fn between(before: &[Fact], after: &[Fact]) -> Change {
        let (b, a): (BTreeSet<&Fact>, BTreeSet<&Fact>) = (before.iter().collect(), after.iter().collect());
        Change { added: a.difference(&b).map(|f| (*f).clone()).collect(), removed: b.difference(&a).map(|f| (*f).clone()).collect(), keys: vec![] }
    }
    pub fn is_empty(&self) -> bool { self.added.is_empty() && self.removed.is_empty() && self.keys.is_empty() }
}

/// Whether a fact unifies with a literal read as a pattern: the same relation and arity, equal constants, one value for a variable used twice.
pub fn unifies(lit: &Lit, f: &Fact) -> bool {
    if lit.relation != f.relation || lit.args.len() != f.args.len() { return false; }
    let mut seen: Binding = Binding::new();
    for (a, v) in lit.args.iter().zip(&f.args) {
        match a {
            Arg::Const(c) => if c != v { return false; },
            Arg::Var(x) => match seen.get(x) { Some(b) => if b != v { return false; }, None => { seen.insert(x.clone(), v.clone()); } },
        }
    }
    true
}

fn standing_json(s: &Standing) -> Value {
    match s {
        Standing::Fact => json!({"kind": "fact"}),
        Standing::Proved(t) => json!({"kind": "proved", "theorem": t}),
        Standing::Conjecture(x) => json!({"kind": "conjecture", "sample": x}),
        Standing::Belief { probability, count } => json!({"kind": "belief", "probability": probability, "count": count}),
    }
}

fn standing_from(v: &Value) -> Result<Standing, String> {
    let text = |k: &str| v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string()).ok_or_else(|| format!("a standing has {k}"));
    match v.get("kind").and_then(|x| x.as_str()).ok_or("a standing has a kind")? {
        "fact" => Ok(Standing::Fact),
        "proved" => Ok(Standing::Proved(text("theorem")?)),
        "conjecture" => Ok(Standing::Conjecture(text("sample")?)),
        "belief" => Ok(Standing::Belief { probability: v.get("probability").and_then(|x| x.as_f64()).ok_or("a belief has a probability")?, count: v.get("count").and_then(|x| x.as_u64()).ok_or("a belief has a count")? }),
        k => Err(format!("no standing {k}")),
    }
}

impl Standing {
    pub fn to_json(&self) -> Value { standing_json(self) }
    pub fn from_json(v: &Value) -> Result<Standing, String> { standing_from(v) }
    pub fn word(&self) -> &'static str { match self { Standing::Fact => "fact", Standing::Proved(_) => "proved", Standing::Conjecture(_) => "conjecture", Standing::Belief { .. } => "belief" } }
}

impl Holder {
    pub fn word(&self) -> &'static str { match self { Holder::Ours => "ours", Holder::OursAboutOther => "ours_about_other" } }
    pub fn parse(s: &str) -> Option<Holder> { match s { "ours" => Some(Holder::Ours), "ours_about_other" => Some(Holder::OursAboutOther), _ => None } }
}

impl Says {
    pub fn to_json(&self) -> Value {
        match self {
            Says::Holds => json!({"kind": "holds"}),
            Says::Proposes(r) => json!({"kind": "proposes", "removed": r.removed.iter().map(Lit::to_json).collect::<Vec<_>>(), "added": r.added.iter().map(Lit::to_json).collect::<Vec<_>>()}),
            Says::Number(x) => json!({"kind": "number", "value": x}),
            Says::Lower { term, within } => json!({"kind": "lower", "term": term, "within": within}),
        }
    }
    pub fn from_json(v: &Value) -> Result<Says, String> {
        let lits = |k: &str| -> Result<Vec<Lit>, String> { v.get(k).and_then(|x| x.as_array()).ok_or_else(|| format!("proposes has {k}"))?.iter().map(Lit::from_json).collect() };
        match v.get("kind").and_then(|x| x.as_str()).ok_or("a saying has a kind")? {
            "holds" => Ok(Says::Holds),
            "proposes" => Ok(Says::Proposes(Response { removed: lits("removed")?, added: lits("added")? })),
            "number" => Ok(Says::Number(v.get("value").and_then(|x| x.as_f64()).ok_or("a number is a number")?)),
            "lower" => Ok(Says::Lower { term: v.get("term").and_then(|x| x.as_str()).ok_or("lower has a term")?.to_string(), within: v.get("within").and_then(|x| x.as_u64()).ok_or("lower has within")? as u32 }),
            k => Err(format!("no saying {k}")),
        }
    }
}

impl Statement {
    /// What the statement read: each literal of its pattern, and `also`.
    pub fn reads(&self) -> Vec<Read> { self.pattern.iter().cloned().map(Read::Facts).chain(self.also.iter().cloned().map(Read::Key)).collect() }

    /// The well-formedness: an id; a non-empty pattern is a well-formed body (a positive literal, negations safe); a proposal's literals are positive.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() { return Err("a statement needs an id".into()); }
        if !self.pattern.is_empty() { validate_body(&self.id, &self.pattern)?; }
        if let Says::Proposes(r) = &self.says {
            if r.removed.is_empty() && r.added.is_empty() { return Err(format!("{}: a proposal says nothing", self.id)); }
            if r.removed.iter().chain(r.added.iter()).any(|l| l.neg) { return Err(format!("{}: a proposal is positive literals", self.id)); }
        }
        if let Standing::Belief { probability, .. } = &self.standing { if !(0.0..=1.0).contains(probability) { return Err(format!("{}: a probability is in 0..1", self.id)); } }
        Ok(())
    }

    /// Where the pattern binds on a position (an empty pattern holds with the empty binding).
    pub fn bindings(&self, idx: &FactIndex, limit: usize, steps: &mut u64) -> Vec<Binding> {
        if self.pattern.is_empty() { return vec![Binding::new()]; }
        bindings_fused(&self.pattern, &Binding::new(), idx, limit, steps, u64::MAX).0
    }

    /// Whether something it read changed.
    pub fn falls_by(&self, c: &Change) -> bool {
        c.added.iter().chain(c.removed.iter()).any(|f| self.pattern.iter().any(|l| unifies(l, f))) || c.keys.iter().any(|k| self.also.contains(k))
    }

    pub fn to_json(&self) -> Value {
        json!({"id": self.id, "pattern": self.pattern.iter().map(Lit::to_json).collect::<Vec<_>>(), "says": self.says.to_json(), "standing": self.standing.to_json(), "holder": self.holder.word(), "also": self.also})
    }

    pub fn from_json(v: &Value) -> Result<Statement, String> {
        let id = v.get("id").and_then(|x| x.as_str()).ok_or("a statement has an id")?.to_string();
        let pattern = v.get("pattern").and_then(|x| x.as_array()).ok_or_else(|| format!("{id}: no pattern"))?.iter().map(Lit::from_json).collect::<Result<Vec<_>, _>>()?;
        let holder = v.get("holder").and_then(|x| x.as_str()).and_then(Holder::parse).ok_or_else(|| format!("{id}: a holder is ours or ours_about_other"))?;
        let also = v.get("also").and_then(|x| x.as_array()).ok_or_else(|| format!("{id}: no also"))?.iter().map(|k| k.as_str().map(|s| s.to_string()).ok_or("a key is text")).collect::<Result<Vec<_>, _>>()?;
        let s = Statement { says: Says::from_json(v.get("says").ok_or_else(|| format!("{id}: no says"))?)?, standing: standing_from(v.get("standing").ok_or_else(|| format!("{id}: no standing"))?)?, id, pattern, holder, also };
        s.validate()?;
        Ok(s)
    }

    /// A lemma as a statement: its body the pattern, its response what it says; the standing is the caller's (a lemma's own stats and meta are kept in the lemma).
    pub fn from_lemma(l: &crate::lemma::Lemma, standing: Standing) -> Statement {
        Statement { id: l.name.clone(), pattern: l.body.clone(), says: Says::Proposes(l.response.clone()), standing, holder: Holder::Ours, also: vec![] }
    }
}

/// A position's facts as the canonical text the memories compare by: a JSON array of [relation, args], sorted and distinct.
pub fn encode_facts(facts: &[Fact]) -> String {
    let set: BTreeSet<&Fact> = facts.iter().collect();
    Value::Array(set.iter().map(|f| json!([f.relation, f.args])).collect()).to_string()
}

/// The facts of a canonical text (the inverse of `encode_facts`).
pub fn decode_facts(text: &str) -> Result<Vec<Fact>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
    v.as_array().ok_or("facts are a list")?.iter().map(|row| {
        let r = row.as_array().filter(|r| r.len() == 2).ok_or("a fact is [relation, args]")?;
        let relation = r[0].as_str().ok_or("a relation is text")?.to_string();
        let args = r[1].as_array().ok_or("args are a list")?.iter().map(|a| a.as_str().map(|s| s.to_string()).ok_or("an argument is text")).collect::<Result<Vec<_>, _>>()?;
        Ok(Fact { relation, args })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(standing: Standing, holder: Holder, says: Says) -> Statement {
        Statement { id: "s".into(), pattern: vec![Lit::pos("cell", &["?f", "?r", "us", "?k"])], says, standing, holder, also: vec!["key:7".into()] }
    }
    #[test]
    fn every_standing_holder_and_saying_is_written_and_read_back_identical() {
        let standings = [Standing::Fact, Standing::Proved("exchange_law".into()), Standing::Conjecture("games 1..40".into()), Standing::Belief { probability: 0.123456789012345, count: 17 }];
        let sayings = [Says::Holds, Says::Proposes(Response { removed: vec![Lit::pos("cell", &["?f", "?r", "us", "?k"])], added: vec![Lit::pos("cell", &["?x", "?y", "us", "?k"])] }), Says::Number(-2.5e-3), Says::Lower { term: "[tuple region(1,1)]".into(), within: 3 }];
        let mut n = 0;
        for st in &standings { for h in [Holder::Ours, Holder::OursAboutOther] { for sa in &sayings {
            let s = sample(st.clone(), h, sa.clone());
            assert_eq!(Statement::from_json(&s.to_json()).unwrap(), s);
            assert_eq!(Statement::from_json(&serde_json::from_str::<Value>(&s.to_json().to_string()).unwrap()).unwrap(), s, "by way of the text too");
            n += 1;
        } } }
        assert_eq!(n, 32);
    }
    #[test]
    fn a_belief_outside_zero_to_one_and_an_empty_proposal_are_refused() {
        assert!(sample(Standing::Belief { probability: 1.5, count: 1 }, Holder::Ours, Says::Holds).validate().is_err());
        assert!(sample(Standing::Fact, Holder::Ours, Says::Proposes(Response::default())).validate().is_err());
    }
    #[test]
    fn a_statement_falls_when_a_literal_unifies_with_a_changed_fact_and_not_otherwise() {
        let s = sample(Standing::Fact, Holder::Ours, Says::Holds);
        let hit = Change { added: vec![Fact::new("cell", &["a", "1", "us", "kind_x"])], ..Default::default() };
        let other_relation = Change { added: vec![Fact::new("attack", &["a", "1", "us", "kind_x"])], ..Default::default() };
        let other_constant = Change { removed: vec![Fact::new("cell", &["a", "1", "them", "kind_x"])], ..Default::default() };
        assert!(s.falls_by(&hit) && !s.falls_by(&other_relation) && !s.falls_by(&other_constant));
        assert!(s.falls_by(&Change { keys: vec!["key:7".into()], ..Default::default() }) && !s.falls_by(&Change { keys: vec!["key:8".into()], ..Default::default() }));
    }
    #[test]
    fn a_lemma_is_a_statement_whose_body_is_its_pattern_and_whose_head_is_what_it_says() {
        let l = crate::lemma::Lemma { name: "l".into(), body: vec![Lit::pos("cell", &["?f", "?r", "us", "?k"])], response: Response { removed: vec![Lit::pos("cell", &["?f", "?r", "us", "?k"])], added: vec![Lit::pos("cell", &["?x", "?y", "us", "?k"])] }, stats: json!({}), meta: json!({}), suggest: false };
        let s = Statement::from_lemma(&l, Standing::Proved("a_theorem".into()));
        assert_eq!((s.id.as_str(), &s.pattern, &s.says, s.holder), ("l", &l.body, &Says::Proposes(l.response.clone()), Holder::Ours));
        assert!(s.validate().is_ok());
        assert_eq!(Statement::from_json(&s.to_json()).unwrap(), s);
    }
    #[test]
    fn the_position_text_round_trips() {
        let f = vec![Fact::new("cell", &["b", "(sq 1 2)", "us", "x y"]), Fact::new("cell", &["a", "1", "them", "k"])];
        let back = decode_facts(&encode_facts(&f)).unwrap();
        let mut want = f.clone();
        want.sort();
        assert_eq!(back, want);
    }
}
