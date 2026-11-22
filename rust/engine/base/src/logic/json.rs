//! THE JSON FORM OF A LITERAL (moved from cs_rep's lemma.rs, plan 79.30): {"relation": "attack", "args": ["?s", "e", "?r"], "neg": false}; an argument starting with `?` is a placeholder, any other text a constant.
//! This is the one place cs_base uses an outside crate (serde_json 1, `float_roundtrip`, already in the lock): `Lit::to_json` and `Lit::from_json` are called by name from the layers above and must keep resolving on `Lit`.
use super::{Arg, Lit};
use serde_json::{json, Value};

impl Lit {
    pub fn to_json(&self) -> Value { json!({"relation": self.relation, "args": self.args.iter().map(|a| a.text()).collect::<Vec<_>>(), "neg": self.neg}) }

    pub fn from_json(v: &Value) -> Result<Lit, String> {
        let relation = v.get("relation").and_then(|x| x.as_str()).ok_or("a literal has a relation")?.to_string();
        let args = v.get("args").and_then(|x| x.as_array()).ok_or("a literal has args")?
            .iter().map(|a| a.as_str().map(Arg::parse).ok_or_else(|| "an argument is text".to_string())).collect::<Result<Vec<_>, _>>()?;
        Ok(Lit { neg: v.get("neg").and_then(|x| x.as_bool()).unwrap_or(false), relation, args })
    }
}

#[cfg(test)]
mod tests {
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
