//! THE LEMMA LANGUAGE (plan 78.14): a lemma is "if these facts hold, a move of this kind is among the best" -- a body of literals over the game's fact vocabulary
//! (restricted Datalog: positive literals, and negated ones whose variables the positive ones bind), and a response, the effect a move has on the state's facts
//! (the facts it removes and the facts it adds, over the delta relations), written in the same variables. A variable of the response the body does not bind is free:
//! any value will do. A lemma is READ (matched against a position) by `matcher`, never run as code; it proposes moves, it is not a verdict.
//!
//! JSON: {"name", "body": [lit...], "response": {"removed": [lit...], "added": [lit...]}, "stats": {...}, "meta": {...}}; a literal is
//! {"relation": "attack", "args": ["?s", "e", "?r"], "neg": false}; an argument starting with `?` is a variable, any other text a constant.
use serde_json::{json, Value};
use std::collections::BTreeSet;

// plan 79.30: the argument, the literal, the one literal that is not a row, and the check of a body moved down to layer 0 (cs_base::logic); every old path resolves here.
pub use cs_base::logic::{validate_body, Arg, Lit, DIFFER};

/// THE JSON FORM OF A LITERAL (plan 79.31 part E; it was `Lit::to_json` and `Lit::from_json` inside cs_base, which now has no outside crate): {"relation": "attack", "args": ["?s", "e", "?r"], "neg": false}; an argument starting with `?` is a placeholder,
/// any other text a constant. A trait here, because a literal is cs_base's: a caller brings it into scope (`use cs_rep::lemma::LitJson;`) and writes `Lit::to_json` and `Lit::from_json` as before.
pub trait LitJson: Sized {
    fn to_json(&self) -> Value;
    fn from_json(v: &Value) -> Result<Self, String>;
}

impl LitJson for Lit {
    fn to_json(&self) -> Value { json!({"relation": self.relation, "args": self.args.iter().map(|a| a.text()).collect::<Vec<_>>(), "neg": self.neg}) }

    fn from_json(v: &Value) -> Result<Lit, String> {
        let relation = v.get("relation").and_then(|x| x.as_str()).ok_or("a literal has a relation")?.to_string();
        let args = v.get("args").and_then(|x| x.as_array()).ok_or("a literal has args")?
            .iter().map(|a| a.as_str().map(Arg::parse).ok_or_else(|| "an argument is text".to_string())).collect::<Result<Vec<_>, _>>()?;
        Ok(Lit { neg: v.get("neg").and_then(|x| x.as_bool()).unwrap_or(false), relation, args })
    }
}

#[cfg(test)]
mod lit_json_tests {
    use super::*;

    #[test]
    fn a_literal_is_written_and_read_back_identical() {
        for l in [Lit::pos("r", &["?x", "c"]), Lit::differ("?x", "?y"), Lit::pos("r", &["?x"]).negated(), Lit::pos("p", &[])] {
            assert_eq!(Lit::from_json(&l.to_json()).unwrap(), l);
        }
        assert_eq!(Lit::pos("r", &["?x", "c"]).to_json(), json!({"relation": "r", "args": ["?x", "c"], "neg": false}));
        assert!(!Lit::from_json(&json!({"relation": "r", "args": []})).unwrap().neg, "neg may be left out");
        assert!(Lit::from_json(&json!({"args": []})).is_err());
        assert!(Lit::from_json(&json!({"relation": "r", "args": [1]})).is_err());
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
