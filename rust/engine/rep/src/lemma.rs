//! THE LEMMA LANGUAGE (plan 78.14): a lemma is "if these facts hold, a move of this kind is among the best" -- a body of literals over the game's fact vocabulary
//! (restricted Datalog: positive literals, and negated ones whose variables the positive ones bind), and a response, the effect a move has on the state's facts
//! (the facts it removes and the facts it adds, over the delta relations), written in the same variables. A variable of the response the body does not bind is free:
//! any value will do. A lemma is READ (matched against a position) by `matcher`, never run as code; it proposes moves, it is not a verdict.
//!
//! JSON: {"name", "body": [lit...], "response": {"removed": [lit...], "added": [lit...]}, "stats": {...}, "meta": {...}}; a literal is
//! {"relation": "attack", "args": ["?s", "e", "?r"], "neg": false}; an argument starting with `?` is a variable, any other text a constant.
use serde_json::{json, Value};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arg { Var(String), Const(String) }

impl Arg {
    pub fn parse(s: &str) -> Arg { match s.strip_prefix('?') { Some(v) => Arg::Var(v.to_string()), None => Arg::Const(s.to_string()) } }
    pub fn text(&self) -> String { match self { Arg::Var(v) => format!("?{v}"), Arg::Const(c) => c.clone() } }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lit { pub neg: bool, pub relation: String, pub args: Vec<Arg> }

impl Lit {
    pub fn pos(relation: &str, args: &[&str]) -> Lit { Lit { neg: false, relation: relation.to_string(), args: args.iter().map(|a| Arg::parse(a)).collect() } }
    pub fn vars(&self) -> BTreeSet<String> { self.args.iter().filter_map(|a| if let Arg::Var(v) = a { Some(v.clone()) } else { None }).collect() }
    pub fn to_json(&self) -> Value { json!({"relation": self.relation, "args": self.args.iter().map(|a| a.text()).collect::<Vec<_>>(), "neg": self.neg}) }
    pub fn from_json(v: &Value) -> Result<Lit, String> {
        let relation = v.get("relation").and_then(|x| x.as_str()).ok_or("a literal has a relation")?.to_string();
        let args = v.get("args").and_then(|x| x.as_array()).ok_or("a literal has args")?
            .iter().map(|a| a.as_str().map(Arg::parse).ok_or_else(|| "an argument is text".to_string())).collect::<Result<Vec<_>, _>>()?;
        Ok(Lit { neg: v.get("neg").and_then(|x| x.as_bool()).unwrap_or(false), relation, args })
    }
}

/// The effect of a move on the state's facts, over the delta relations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Response { pub removed: Vec<Lit>, pub added: Vec<Lit> }

#[derive(Debug, Clone, PartialEq)]
pub struct Lemma {
    pub name: String,
    pub body: Vec<Lit>,
    pub response: Response,
    /// What the verification measured (free-form JSON, kept as written).
    pub stats: Value,
    /// Where it came from (free-form JSON, kept as written).
    pub meta: Value,
    /// Plan 78.36: a SUGGESTION -- a rule that was not admitted with zero wrong. It never acts unverified: the lemma operator reports its moves apart from the candidates and the
    /// decision uses them only where a claim search classed them a win (or, with every move drawn or refuted, a draw). JSON `"suggest": true` at the rule's top level; absent = false.
    pub suggest: bool,
}

impl Lemma {
    pub fn to_json(&self) -> Value {
        let mut j = json!({"name": self.name, "body": self.body.iter().map(Lit::to_json).collect::<Vec<_>>(),
               "response": {"removed": self.response.removed.iter().map(Lit::to_json).collect::<Vec<_>>(), "added": self.response.added.iter().map(Lit::to_json).collect::<Vec<_>>()},
               "stats": self.stats, "meta": self.meta});
        if self.suggest { if let Some(o) = j.as_object_mut() { o.insert("suggest".into(), Value::Bool(true)); } }
        j
    }

    pub fn from_json(v: &Value) -> Result<Lemma, String> {
        let name = v.get("name").and_then(|x| x.as_str()).ok_or("a lemma has a name")?.to_string();
        let lits = |key: &str, from: &Value| -> Result<Vec<Lit>, String> {
            from.get(key).and_then(|x| x.as_array()).ok_or_else(|| format!("{name}: no {key}"))?.iter().map(Lit::from_json).collect()
        };
        let suggest = match v.get("suggest") { None | Some(Value::Null) => false, Some(b) => b.as_bool().ok_or_else(|| format!("{name}: suggest is true or false"))? };
        let resp = v.get("response").ok_or_else(|| format!("{name}: no response"))?;
        let l = Lemma { body: lits("body", v)?, response: Response { removed: lits("removed", resp)?, added: lits("added", resp)? },
                        stats: v.get("stats").cloned().unwrap_or(Value::Null), meta: v.get("meta").cloned().unwrap_or(Value::Null), suggest, name };
        l.validate()?;
        Ok(l)
    }

    /// The well-formedness of a lemma: a body with at least one positive literal, negated literals safe (every variable bound by a positive one), a response of
    /// positive literals with something in it.
    pub fn validate(&self) -> Result<(), String> {
        let at = &self.name;
        validate_body(at, &self.body)?;
        if self.response.removed.is_empty() && self.response.added.is_empty() { return Err(format!("{at}: the response says nothing")); }
        if self.response.removed.iter().chain(self.response.added.iter()).any(|l| l.neg) { return Err(format!("{at}: the response is positive literals")); }
        Ok(())
    }

    /// The relations the body reads.
    pub fn body_relations(&self) -> BTreeSet<String> { self.body.iter().map(|l| l.relation.clone()).collect() }
}

/// The well-formedness of a body (a lemma's or a claim's): a name, at least one positive literal, negated literals safe (every variable bound by a positive one).
pub fn validate_body(at: &str, body: &[Lit]) -> Result<(), String> {
    if at.is_empty() { return Err("a lemma needs a name".into()); }
    if !body.iter().any(|l| !l.neg) { return Err(format!("{at}: the body needs a positive literal")); }
    let bound: BTreeSet<String> = body.iter().filter(|l| !l.neg).flat_map(|l| l.vars()).collect();
    for l in body.iter().filter(|l| l.neg) {
        if let Some(v) = l.vars().into_iter().find(|v| !bound.contains(v)) { return Err(format!("{at}: the negated literal {} uses ?{v}, which no positive literal binds", l.relation)); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Lemma {
        Lemma { name: "hang-x".into(), body: vec![Lit::pos("cell", &["?f", "?r", "?s", "kind_y"]), Lit::pos("attack", &["?o", "?g", "?q", "?f", "?r"])],
                response: Response { removed: vec![Lit::pos("cell", &["?f", "?r", "?s", "kind_y"])], added: vec![Lit::pos("cell", &["?x", "?y", "?s", "kind_y"])] },
                stats: json!({}), meta: json!({}), suggest: false }
    }
    #[test]
    fn json_round_trip_and_checks() {
        let l = sample();
        assert!(l.validate().is_ok());
        assert_eq!(Lemma::from_json(&l.to_json()).unwrap(), l);
        let mut bad = sample();
        bad.body.push(Lit { neg: true, relation: "cell".into(), args: vec![Arg::Var("zzz".into())] });
        assert!(bad.validate().unwrap_err().contains("binds"));
        let mut empty = sample();
        empty.response = Response::default();
        assert!(empty.validate().is_err());
        let mut nobody = sample();
        nobody.body.clear();
        assert!(nobody.validate().is_err());
        // a suggestion carries `suggest` and is written only when true: a rule without it serialises as it always did
        assert!(l.to_json().get("suggest").is_none());
        let mut sg = sample();
        sg.suggest = true;
        assert_eq!(sg.to_json()["suggest"], true);
        assert_eq!(Lemma::from_json(&sg.to_json()).unwrap(), sg);
    }
}
