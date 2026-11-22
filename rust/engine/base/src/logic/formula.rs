//! THE FORMULA (plan 79.30, new; SYSTEM_SPEC 11 "Layer 0 holds six things" item 1): and, or, not, some, every (over a finite set), at least so many, over the literals of the language, read against the same index of facts the matcher reads.
//! Placeholders are written without the `?` here, as the keys of a `Binding` are. A formula is a CLOSED reading when its filters (`Not`, `Every`, `AtLeast`, a negated literal, a `differ`) read only placeholders that the rest of the formula, or the caller's initial binding, binds.
//!
//! Reading, in the order `answers` takes the parts of an `And`: first the literals as one body handed to the matcher (the positive rows joined; the negated and `differ` literals whose placeholders that body or the initial binding binds), then the parts that
//! make bindings (`Or`, `Some`) in the order written, then the filters. On a body of literals alone it therefore gives the bindings the matcher gives.
use super::matcher::{bindings, Binding, FactIndex};
use super::Lit;
use crate::sets::Set;
use std::collections::BTreeSet;

/// The finite set a quantifier ranges over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Domain {
    /// A given set of constants.
    Set(Set<String>),
    /// The first arguments of the rows of a relation of one argument, in the index.
    Relation(String),
}

impl Domain {
    fn members(&self, idx: &FactIndex) -> Vec<String> {
        match self {
            Domain::Set(s) => s.iter().cloned().collect(),
            Domain::Relation(r) => { let s: Set<String> = idx.rows(r).iter().filter(|row| row.len() == 1).map(|row| row[0].clone()).collect(); s.iter().cloned().collect() }
        }
    }
}

/// A formula over literals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Formula {
    Lit(Lit),
    And(Vec<Formula>),
    Or(Vec<Formula>),
    Not(Box<Formula>),
    /// Some value of `var` in `over` makes `body` true.
    Some { var: String, over: Domain, body: Box<Formula> },
    /// Every value of `var` in `over` makes `body` true.
    Every { var: String, over: Domain, body: Box<Formula> },
    /// At least `count` values of `var` in `over` make `body` true.
    AtLeast { count: usize, var: String, over: Domain, body: Box<Formula> },
}

/// Why a formula could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormulaError {
    /// A filter reads a placeholder that nothing binds.
    Unbound(String),
    /// The branches of an `Or` do not bind the same placeholders.
    BranchesDiffer,
    /// A quantifier's placeholder is already bound outside it.
    Shadowed(String),
}

impl std::fmt::Display for FormulaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormulaError::Unbound(v) => write!(f, "the placeholder ?{v} is read before anything binds it"),
            FormulaError::BranchesDiffer => f.write_str("the branches of an or bind different placeholders"),
            FormulaError::Shadowed(v) => write!(f, "the placeholder ?{v} of a quantifier is already bound"),
        }
    }
}

fn is_filter(l: &Lit) -> bool { l.neg || l.is_differ() }

impl Formula {
    /// The placeholders the formula leaves free (a quantifier's own placeholder is not free in it).
    pub fn free_vars(&self) -> BTreeSet<String> {
        match self {
            Formula::Lit(l) => l.vars(),
            Formula::And(ps) | Formula::Or(ps) => ps.iter().flat_map(|p| p.free_vars()).collect(),
            Formula::Not(f) => f.free_vars(),
            Formula::Some { var, body, .. } | Formula::Every { var, body, .. } | Formula::AtLeast { var, body, .. } => { let mut s = body.free_vars(); s.remove(var); s }
        }
    }

    /// The placeholders a reading of the formula binds (the answers hold a value for each).
    fn binds(&self) -> BTreeSet<String> {
        match self {
            Formula::Lit(l) => if is_filter(l) { BTreeSet::new() } else { l.vars() },
            Formula::And(ps) => ps.iter().flat_map(|p| p.binds()).collect(),
            Formula::Or(ps) => ps.first().map(|p| p.binds()).unwrap_or_default(),
            Formula::Not(_) | Formula::Every { .. } | Formula::AtLeast { .. } => BTreeSet::new(),
            Formula::Some { var, body, .. } => { let mut s = body.binds(); s.remove(var); s }
        }
    }

    /// The placeholders the caller's initial binding must hold for the formula to be read: what its filters read that nothing in it binds.
    pub fn needs(&self) -> BTreeSet<String> {
        match self {
            Formula::Lit(l) => if is_filter(l) { l.vars() } else { BTreeSet::new() },
            Formula::And(ps) => { let bound: BTreeSet<String> = ps.iter().flat_map(|p| p.binds()).collect(); ps.iter().flat_map(|p| p.needs()).filter(|v| !bound.contains(v)).collect() }
            Formula::Or(ps) => ps.iter().flat_map(|p| p.needs()).collect(),
            Formula::Not(f) => f.free_vars(),
            Formula::Some { var, body, .. } => { let mut s = body.needs(); s.remove(var); s }
            Formula::Every { var, body, .. } | Formula::AtLeast { var, body, .. } => { let mut s = body.free_vars(); s.remove(var); s }
        }
    }

    /// The well-formedness that needs no initial binding: see `check`.
    pub fn validate(&self) -> Result<(), FormulaError> { self.check(&BTreeSet::new()) }

    /// The well-formedness given the placeholders already bound outside (the caller's initial binding, a quantifier's own placeholder, what the rest of an `And` binds): every branch of an `Or` binds the same placeholders besides those
    /// (a placeholder bound in one branch only would leave the answers of the other without it).
    pub fn check(&self, bound: &BTreeSet<String>) -> Result<(), FormulaError> {
        match self {
            Formula::Lit(_) => Ok(()),
            Formula::And(ps) => {
                // a part is read beside the others: what they bind is bound for it (never what it binds itself)
                for (i, p) in ps.iter().enumerate() {
                    let mut inner = bound.clone();
                    inner.extend(ps.iter().enumerate().filter(|(j, _)| *j != i).flat_map(|(_, q)| q.binds()));
                    p.check(&inner)?;
                }
                Ok(())
            }
            Formula::Or(ps) => {
                ps.iter().try_for_each(|p| p.check(bound))?;
                let new = |p: &Formula| -> BTreeSet<String> { p.binds().into_iter().filter(|v| !bound.contains(v)).collect() };
                if ps.windows(2).any(|w| new(&w[0]) != new(&w[1])) { Err(FormulaError::BranchesDiffer) } else { Ok(()) }
            }
            Formula::Not(f) => f.check(bound),
            Formula::Some { var, body, .. } | Formula::Every { var, body, .. } | Formula::AtLeast { var, body, .. } => {
                let mut inner = bound.clone();
                inner.insert(var.clone());
                body.check(&inner)
            }
        }
    }
}

/// Every binding of the free placeholders that makes the formula true over the index, extending `init`. At most `limit` (the second result says the limit cut the answers). `steps` counts the rows examined (the matcher's) and the values of a domain gone through.
pub fn answers(f: &Formula, init: &Binding, idx: &FactIndex, limit: usize, steps: &mut u64) -> Result<(Vec<Binding>, bool), FormulaError> {
    f.check(&init.keys().cloned().collect())?;
    let mut cut = false;
    let mut all = dedup(eval(f, init, idx, limit, steps, &mut cut)?);                  // the answers are a set: a binding once, where the matcher gives one for each row that matches
    if all.len() > limit { all.truncate(limit); cut = true; }
    Ok((all, cut))
}

/// Whether the formula is true under the binding: at least one answer exists.
pub fn holds(f: &Formula, b: &Binding, idx: &FactIndex) -> Result<bool, FormulaError> {
    let mut steps = 0;
    Ok(!answers(f, b, idx, 1, &mut steps)?.0.is_empty())
}

fn require(vars: &BTreeSet<String>, b: &Binding) -> Result<(), FormulaError> {
    match vars.iter().find(|v| !b.contains_key(*v)) { Some(v) => Err(FormulaError::Unbound(v.clone())), None => Ok(()) }
}

fn dedup(v: Vec<Binding>) -> Vec<Binding> { let mut seen = BTreeSet::new(); v.into_iter().filter(|b| seen.insert(b.clone())).collect() }

/// The values of a quantifier's placeholder for which the body is true under the binding.
fn witnesses(var: &str, over: &Domain, body: &Formula, b: &Binding, idx: &FactIndex, steps: &mut u64, enough: usize) -> Result<usize, FormulaError> {
    if b.contains_key(var) { return Err(FormulaError::Shadowed(var.to_string())); }
    let mut found = 0;
    let mut cut = false;
    for v in over.members(idx) {
        *steps += 1;
        let mut b2 = b.clone();
        b2.insert(var.to_string(), v);
        if !eval(body, &b2, idx, 1, steps, &mut cut)?.is_empty() { found += 1; if found >= enough { break; } }
    }
    Ok(found)
}

fn eval(f: &Formula, b: &Binding, idx: &FactIndex, limit: usize, steps: &mut u64, cut: &mut bool) -> Result<Vec<Binding>, FormulaError> {
    match f {
        Formula::Lit(l) => {
            if is_filter(l) { require(&l.vars(), b)?; }
            let (out, c) = bindings(std::slice::from_ref(l), b, idx, limit, steps);
            *cut |= c;
            Ok(out)
        }
        Formula::And(parts) => eval_and(parts, b, idx, limit, steps, cut),
        Formula::Or(parts) => {
            let mut out = vec![];
            for p in parts { out.extend(eval(p, b, idx, limit, steps, cut)?); }
            Ok(dedup(out))
        }
        Formula::Not(inner) => { require(&inner.free_vars(), b)?; Ok(if eval(inner, b, idx, 1, steps, &mut false)?.is_empty() { vec![b.clone()] } else { vec![] }) }
        Formula::Some { var, over, body } => {
            if b.contains_key(var) { return Err(FormulaError::Shadowed(var.clone())); }
            let mut out = vec![];
            for v in over.members(idx) {
                *steps += 1;
                let mut b2 = b.clone();
                b2.insert(var.clone(), v);
                for mut r in eval(body, &b2, idx, limit, steps, cut)? { r.remove(var); out.push(r); }
            }
            Ok(dedup(out))
        }
        Formula::Every { var, over, body } => {
            let mut need = body.free_vars();
            need.remove(var);
            require(&need, b)?;
            let n = over.members(idx).len();
            Ok(if witnesses(var, over, body, b, idx, steps, n)? == n { vec![b.clone()] } else { vec![] })
        }
        Formula::AtLeast { count, var, over, body } => {
            let mut need = body.free_vars();
            need.remove(var);
            require(&need, b)?;
            Ok(if witnesses(var, over, body, b, idx, steps, *count)? >= *count { vec![b.clone()] } else { vec![] })
        }
    }
}

fn flatten<'a>(parts: &'a [Formula], out: &mut Vec<&'a Formula>) {
    for p in parts { if let Formula::And(inner) = p { flatten(inner, out); } else { out.push(p); } }
}

fn eval_and(parts: &[Formula], b: &Binding, idx: &FactIndex, limit: usize, steps: &mut u64, cut: &mut bool) -> Result<Vec<Binding>, FormulaError> {
    let mut flat = vec![];
    flatten(parts, &mut flat);
    // 1. the literals: positive rows, and the negated and `differ` literals the rows and the initial binding bind
    let pos: Vec<&Lit> = flat.iter().filter_map(|p| if let Formula::Lit(l) = p { Some(l) } else { None }).filter(|l| !is_filter(l)).collect();
    let mut known: BTreeSet<String> = b.keys().cloned().collect();
    known.extend(pos.iter().flat_map(|l| l.vars()));
    let (inline, deferred): (Vec<&Formula>, Vec<&Formula>) = flat.iter().partition(|p| matches!(p, Formula::Lit(l) if !is_filter(l) || l.vars().is_subset(&known)));
    let body: Vec<Lit> = inline.iter().filter_map(|p| if let Formula::Lit(l) = p { Some((*l).clone()) } else { None }).collect();
    let mut results = if body.is_empty() { vec![b.clone()] } else { let (out, c) = bindings(&body, b, idx, limit, steps); *cut |= c; out };
    // 2. the parts that make bindings, in the order written; 3. the filters
    let (makers, filters): (Vec<&Formula>, Vec<&Formula>) = deferred.into_iter().partition(|p| matches!(p, Formula::Or(_) | Formula::Some { .. }));
    for m in makers {
        let mut next = vec![];
        for r in &results { next.extend(eval(m, r, idx, limit, steps, cut)?); }
        results = dedup(next);
    }
    for fl in filters {
        let mut kept = vec![];
        for r in results { if !eval(fl, &r, idx, 1, steps, &mut false)?.is_empty() { kept.push(r); } }
        results = kept;
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::Fact;
    use crate::maker::Maker;

    fn l(r: &str, a: &[&str]) -> Formula { Formula::Lit(Lit::pos(r, a)) }
    fn dom(items: &[&str]) -> Domain { Domain::Set(items.iter().map(|s| s.to_string()).collect()) }
    fn idx() -> FactIndex {
        FactIndex::new(&[Fact::new("thing", &["a"]), Fact::new("thing", &["b"]), Fact::new("thing", &["c"]),
                         Fact::new("red", &["a"]), Fact::new("red", &["b"]), Fact::new("near", &["a", "b"]), Fact::new("near", &["b", "c"])])
    }
    fn run(f: &Formula) -> Vec<Binding> { let mut s = 0; let mut v = answers(f, &Binding::new(), &idx(), 100, &mut s).unwrap().0; v.sort(); v }
    fn bind(pairs: &[(&str, &str)]) -> Binding { pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() }

    #[test]
    fn and_or_not_over_rows() {
        assert_eq!(run(&Formula::And(vec![l("thing", &["?x"]), l("red", &["?x"])])), vec![bind(&[("x", "a")]), bind(&[("x", "b")])]);
        assert_eq!(run(&Formula::And(vec![l("thing", &["?x"]), Formula::Not(Box::new(l("red", &["?x"])))])), vec![bind(&[("x", "c")])]);
        assert_eq!(run(&Formula::Or(vec![l("red", &["?x"]), l("near", &["c", "?x"])])), vec![bind(&[("x", "a")]), bind(&[("x", "b")])]);
        assert_eq!(run(&Formula::Or(vec![l("red", &["?x"]), l("near", &["?x", "c"])])), vec![bind(&[("x", "a")]), bind(&[("x", "b")])], "b is red and b is near c: one answer for b");
    }

    #[test]
    fn some_every_and_at_least_over_a_finite_set() {
        // some y near x, over the things
        let some = Formula::Some { var: "y".into(), over: Domain::Relation("thing".into()), body: Box::new(l("near", &["?x", "?y"])) };
        assert_eq!(run(&Formula::And(vec![l("thing", &["?x"]), some])), vec![bind(&[("x", "a")]), bind(&[("x", "b")])]);
        let every = Formula::Every { var: "y".into(), over: dom(&["a", "b"]), body: Box::new(l("red", &["?y"])) };
        assert_eq!(run(&every), vec![Binding::new()], "true: no free placeholder, one empty answer");
        let not_every = Formula::Every { var: "y".into(), over: Domain::Relation("thing".into()), body: Box::new(l("red", &["?y"])) };
        assert!(run(&not_every).is_empty());
        let at_least = |n: usize| Formula::AtLeast { count: n, var: "y".into(), over: Domain::Relation("thing".into()), body: Box::new(l("red", &["?y"])) };
        assert_eq!(run(&at_least(2)).len(), 1);
        assert!(run(&at_least(3)).is_empty());
        assert_eq!(run(&at_least(0)).len(), 1);
        // an empty set: every is true, some is false
        let empty = Formula::Every { var: "y".into(), over: dom(&[]), body: Box::new(l("red", &["?y"])) };
        assert_eq!(run(&empty).len(), 1);
        assert!(run(&Formula::Some { var: "y".into(), over: dom(&[]), body: Box::new(l("red", &["?y"])) }).is_empty());
        // an every that reads a placeholder the rest binds: the things all near-reached from x are red? (x = a: near b, which is red)
        let all_near_red = Formula::Every { var: "y".into(), over: Domain::Relation("thing".into()), body: Box::new(Formula::Or(vec![Formula::Not(Box::new(l("near", &["?x", "?y"]))), l("red", &["?y"])])) };
        assert_eq!(run(&Formula::And(vec![l("thing", &["?x"]), all_near_red])), vec![bind(&[("x", "a")]), bind(&[("x", "c")])], "b is near c, c is not red");
    }

    #[test]
    fn a_filter_that_reads_an_unbound_placeholder_is_an_error_and_so_are_unequal_branches() {
        let mut s = 0;
        let e = answers(&Formula::Not(Box::new(l("red", &["?x"]))), &Binding::new(), &idx(), 10, &mut s).unwrap_err();
        assert_eq!(e, FormulaError::Unbound("x".into()));
        let e = answers(&Formula::Or(vec![l("red", &["?x"]), l("near", &["?x", "?y"])]), &Binding::new(), &idx(), 10, &mut s).unwrap_err();
        assert_eq!(e, FormulaError::BranchesDiffer);
        let e = answers(&Formula::Some { var: "x".into(), over: dom(&["a"]), body: Box::new(l("red", &["?x"])) }, &bind(&[("x", "a")]), &idx(), 10, &mut s).unwrap_err();
        assert_eq!(e, FormulaError::Shadowed("x".into()));
        assert_eq!(Formula::Not(Box::new(l("red", &["?x"]))).needs(), ["x".to_string()].into_iter().collect());
        assert!(holds(&Formula::Not(Box::new(l("red", &["?x"]))), &bind(&[("x", "c")]), &idx()).unwrap(), "the initial binding supplies it");
    }

    #[test]
    fn the_limit_cuts_and_says_so() {
        let mut s = 0;
        let (r, cut) = answers(&l("thing", &["?x"]), &Binding::new(), &idx(), 2, &mut s).unwrap();
        assert_eq!((r.len(), cut), (2, true));
        let (r, cut) = answers(&Formula::Or(vec![l("thing", &["?x"])]), &Binding::new(), &idx(), 2, &mut s).unwrap();
        assert_eq!((r.len(), cut), (2, true));
    }

    /// A brute-force reading, written apart from the code above: every assignment of the free placeholders to the constants of the index, each tested by the plain meaning of the connectives.
    fn naive(f: &Formula, b: &Binding, idx: &FactIndex, consts: &[String]) -> bool {
        match f {
            Formula::Lit(l) => {
                let vals: Vec<String> = l.args.iter().map(|a| match a { crate::logic::Arg::Const(c) => c.clone(), crate::logic::Arg::Var(v) => b[v].clone() }).collect();
                let hit = if l.is_differ() { vals[0] != vals[1] } else { idx.contains(&l.relation, &vals) };
                hit != (l.neg && !l.is_differ())
            }
            Formula::And(ps) => ps.iter().all(|p| naive(p, b, idx, consts)),
            Formula::Or(ps) => ps.iter().any(|p| naive(p, b, idx, consts)),
            Formula::Not(p) => !naive(p, b, idx, consts),
            Formula::Some { var, over, body } => over.members(idx).iter().any(|v| { let mut c = b.clone(); c.insert(var.clone(), v.clone()); naive(body, &c, idx, consts) }),
            Formula::Every { var, over, body } => over.members(idx).iter().all(|v| { let mut c = b.clone(); c.insert(var.clone(), v.clone()); naive(body, &c, idx, consts) }),
            Formula::AtLeast { count, var, over, body } => over.members(idx).iter().filter(|v| { let mut c = b.clone(); c.insert(var.clone(), (*v).clone()); naive(body, &c, idx, consts) }).count() >= *count,
        }
    }

    fn random_formula(m: &mut Maker, depth: u32) -> Formula {
        let rel = |m: &mut Maker| -> (String, usize) { m.pick(&[("p".to_string(), 1usize), ("q".to_string(), 1), ("r".to_string(), 2)]) };
        let lit = |m: &mut Maker| -> Formula {
            let (name, arity) = rel(m);
            let args: Vec<String> = (0..arity).map(|_| m.pick(&["?x".to_string(), "?y".to_string(), "c0".to_string(), "c1".to_string()])).collect();
            Formula::Lit(Lit { neg: false, relation: name, args: args.iter().map(|a| crate::logic::Arg::parse(a)).collect() })
        };
        if depth == 0 { return lit(m); }
        match m.below(6) {
            0 => lit(m),
            1 => Formula::And((0..2).map(|_| random_formula(m, depth - 1)).collect()),
            2 => Formula::Or((0..2).map(|_| random_formula(m, depth - 1)).collect()),
            3 => Formula::Not(Box::new(random_formula(m, depth - 1))),
            4 => Formula::Some { var: "z".into(), over: Domain::Relation("p".into()), body: Box::new(random_formula(m, depth - 1)) },
            _ => Formula::Every { var: "z".into(), over: dom(&["c0", "c1"]), body: Box::new(random_formula(m, depth - 1)) },
        }
    }

    /// Every answer `answers` gives satisfies the plain meaning, and (for a formula with nothing left free to the filters) none is missed among the assignments of the constants.
    #[test]
    fn the_answers_are_exactly_the_assignments_the_plain_meaning_accepts() {
        let mut m = Maker::new(61);
        let consts: Vec<String> = ["c0", "c1", "c2"].iter().map(|s| s.to_string()).collect();
        let mut tried = 0;
        for _ in 0..1500 {
            let facts: Vec<Fact> = (0..m.below(8)).map(|_| { match m.below(3) { 0 => Fact::new("p", &[&m.pick(&consts)]), 1 => Fact::new("q", &[&m.pick(&consts)]), _ => Fact::new("r", &[&m.pick(&consts), &m.pick(&consts)]) } }).collect();
            let idx = FactIndex::new(&facts);
            let f = random_formula(&mut m, 3);
            let free: Vec<String> = f.free_vars().into_iter().collect();
            if f.needs().len() > 0 || f.validate().is_err() { continue; }
            let mut s = 0;
            let Ok((got, cut)) = answers(&f, &Binding::new(), &idx, 10_000, &mut s) else { continue };
            assert!(!cut);
            tried += 1;
            // every answer binds exactly the free placeholders and is accepted by the plain meaning
            let mut expected: Vec<Binding> = vec![];
            let n = consts.len();
            for code in 0..n.pow(free.len() as u32) {
                let mut b = Binding::new();
                let mut c = code;
                for v in &free { b.insert(v.clone(), consts[c % n].clone()); c /= n; }
                if naive(&f, &b, &idx, &consts) { expected.push(b); }
            }
            let mut got_set: Vec<Binding> = got.into_iter().collect();
            got_set.sort();
            expected.sort();
            // a placeholder bound by a row can take a constant outside the three only if a row holds it; the facts here use the three, so the sets agree
            assert_eq!(got_set, expected, "{f:?} over {facts:?}");
        }
        assert!(tried > 300, "the maker makes enough readable formulas ({tried})");
    }

    /// On a body of literals alone the formula gives the bindings the matcher gives (the same set, over many bodies and fact sets).
    #[test]
    fn on_a_body_of_literals_alone_the_formula_gives_the_matcher_bindings() {
        let mut m = Maker::new(62);
        let consts = ["c0", "c1", "c2", "c3"];
        for _ in 0..1500 {
            let facts: Vec<Fact> = (0..m.below(14)).map(|_| if m.chance() { Fact::new("p", &[m.pick(&consts)]) } else { Fact::new("r", &[m.pick(&consts), m.pick(&consts)]) }).collect();
            let idx = FactIndex::new(&facts);
            let vars = ["?x", "?y", "?z"];
            let mut body: Vec<Lit> = (0..1 + m.below(3)).map(|_| if m.chance() { Lit::pos("p", &[m.pick(&vars)]) } else { Lit::pos("r", &[m.pick(&vars), m.pick(&vars)]) }).collect();
            for _ in 0..m.below(3) {
                let bound: Vec<String> = body.iter().flat_map(|l| l.vars()).map(|v| format!("?{v}")).collect();
                let v1 = m.pick(&bound);
                let extra = if m.chance() { Lit::differ(&v1, &m.pick(&bound)) } else { Lit { neg: true, relation: "p".into(), args: vec![crate::logic::Arg::parse(&v1)] } };
                body.push(extra);
            }
            if crate::logic::validate_body("t", &body).is_err() { continue; }
            let (mut s1, mut s2) = (0, 0);
            let (mut want, _) = bindings(&body, &Binding::new(), &idx, 10_000, &mut s1);
            let (mut got, _) = answers(&Formula::And(body.iter().cloned().map(Formula::Lit).collect()), &Binding::new(), &idx, 10_000, &mut s2).unwrap();
            want.sort();
            want.dedup();
            got.sort();
            assert_eq!(got, want, "{body:?}");
            // the same body reached through nested connectives that change nothing: a conjunction of conjunctions, and an or of one branch
            let nested = Formula::And(vec![Formula::And(body.iter().cloned().map(Formula::Lit).collect()), Formula::Or(vec![Formula::And(vec![])])]);
            let (mut got2, _) = answers(&nested, &Binding::new(), &idx, 10_000, &mut s2).unwrap();
            got2.sort();
            assert_eq!(got2, want, "nested {body:?}");
        }
    }

    /// Mirrors Ideas.Layer0.Logic.not_every_iff_some_not, not_some_iff_every_not (De Morgan over a finite set) and atLeast_mono.
    #[test]
    fn de_morgan_over_a_finite_set_and_at_least_is_monotone_in_the_count() {
        let mut m = Maker::new(63);
        let consts = ["c0", "c1", "c2"];
        for _ in 0..800 {
            let facts: Vec<Fact> = (0..m.below(6)).map(|_| Fact::new("p", &[m.pick(&consts)])).collect();
            let idx = FactIndex::new(&facts);
            let over = dom(&consts);
            let body = || Box::new(l("p", &["?y"]));
            let nbody = || Box::new(Formula::Not(Box::new(l("p", &["?y"]))));
            let every = Formula::Every { var: "y".into(), over: over.clone(), body: body() };
            let some_not = Formula::Some { var: "y".into(), over: over.clone(), body: nbody() };
            assert_eq!(holds(&every, &Binding::new(), &idx).unwrap(), !holds(&some_not, &Binding::new(), &idx).unwrap());
            let some = Formula::Some { var: "y".into(), over: over.clone(), body: body() };
            let every_not = Formula::Every { var: "y".into(), over: over.clone(), body: nbody() };
            assert_eq!(holds(&some, &Binding::new(), &idx).unwrap(), !holds(&every_not, &Binding::new(), &idx).unwrap());
            let at = |n: usize| holds(&Formula::AtLeast { count: n, var: "y".into(), over: over.clone(), body: body() }, &Binding::new(), &idx).unwrap();
            for n in 0..4 { if at(n + 1) { assert!(at(n)); } }
            assert_eq!(at(1), holds(&some, &Binding::new(), &idx).unwrap());
            assert_eq!(at(3), holds(&every, &Binding::new(), &idx).unwrap());
        }
    }
}
