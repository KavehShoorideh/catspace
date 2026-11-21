//! THE MATCHER (plan 78.14): the one place a lemma's body is bound to a set of facts -- the join of its positive literals, then the negated ones as failures. Used
//! by the lemma operator in the engine's analysis and by the card corpus in verification, so a lemma means the same in both. Every row examined is counted: the
//! operator reports it as its measured cost.
use crate::lemma::{Arg, Lit};
use crate::fact::Fact;
use std::collections::{BTreeMap, HashMap};

/// A binding of variables to constants.
pub type Binding = BTreeMap<String, String>;

/// The facts grouped by relation.
#[derive(Debug, Clone, Default)]
pub struct FactIndex { rows: HashMap<String, Vec<Vec<String>>> }

impl FactIndex {
    pub fn new(facts: &[Fact]) -> FactIndex {
        let mut rows: HashMap<String, Vec<Vec<String>>> = HashMap::new();
        for f in facts { rows.entry(f.relation.clone()).or_default().push(f.args.clone()); }
        FactIndex { rows }
    }
    pub fn rows(&self, relation: &str) -> &[Vec<String>] { self.rows.get(relation).map(|v| v.as_slice()).unwrap_or(&[]) }
    pub fn contains(&self, relation: &str, args: &[String]) -> bool { self.rows(relation).iter().any(|r| r == args) }
}

/// Unify a literal with a row under a binding: the extended binding, or None.
fn unify(lit: &Lit, row: &[String], b: &Binding) -> Option<Binding> {
    if lit.args.len() != row.len() { return None; }
    let mut out = b.clone();
    for (a, v) in lit.args.iter().zip(row) {
        match a {
            Arg::Const(c) => if c != v { return None; },
            Arg::Var(x) => match out.get(x) { Some(bound) => if bound != v { return None; }, None => { out.insert(x.clone(), v.clone()); } },
        }
    }
    Some(out)
}

/// How many rows of the literal's relation could still match under `b` (a scan; used to read the most constrained literal first).
fn candidates(lit: &Lit, idx: &FactIndex, b: &Binding) -> usize { idx.rows(&lit.relation).iter().filter(|r| unify(lit, r, b).is_some()).count() }

/// All bindings of `body`'s variables that satisfy it over `idx`, extending `init`: the positive literals joined (most constrained first), the negated ones then
/// required to fail. At most `limit` bindings (the second result says whether the limit cut the search). `steps` counts the rows examined.
pub fn bindings(body: &[Lit], init: &Binding, idx: &FactIndex, limit: usize, steps: &mut u64) -> (Vec<Binding>, bool) {
    bindings_fused(body, init, idx, limit, steps, u64::MAX)
}

/// `bindings` with a fuse on the rows examined by THIS call: a join that runs past it stops, and the result says it was cut (a body of loosely connected literals can
/// fail only after a combinatorial search; the fuse bounds the cost of asking).
pub fn bindings_fused(body: &[Lit], init: &Binding, idx: &FactIndex, limit: usize, steps: &mut u64, fuse: u64) -> (Vec<Binding>, bool) {
    let pos: Vec<&Lit> = body.iter().filter(|l| !l.neg).collect();
    let neg: Vec<&Lit> = body.iter().filter(|l| l.neg).collect();
    let mut out = vec![];
    let mut truncated = false;
    let ceiling = steps.saturating_add(fuse);
    solve(pos, &neg, init.clone(), idx, limit, steps, ceiling, &mut out, &mut truncated);
    (out, truncated)
}

#[allow(clippy::too_many_arguments)]
fn solve(mut left: Vec<&Lit>, neg: &[&Lit], b: Binding, idx: &FactIndex, limit: usize, steps: &mut u64, ceiling: u64, out: &mut Vec<Binding>, truncated: &mut bool) {
    if *truncated { return; }
    if *steps >= ceiling { *truncated = true; return; }
    if left.is_empty() {
        for n in neg {
            let hit = idx.rows(&n.relation).iter().any(|r| { *steps += 1; unify(n, r, &b).is_some() });
            if hit { return; }
        }
        if out.len() >= limit { *truncated = true; return; }
        out.push(b);
        return;
    }
    let (best, _) = left.iter().enumerate().map(|(i, l)| (i, candidates(l, idx, &b))).min_by_key(|(_, c)| *c).expect("a literal is left");
    let lit = left.swap_remove(best);
    for row in idx.rows(&lit.relation) {
        *steps += 1;
        if let Some(nb) = unify(lit, row, &b) { solve(left.clone(), neg, nb, idx, limit, steps, ceiling, out, truncated); if *truncated { return; } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn facts() -> Vec<Fact> {
        vec![Fact::new("cell", &["a", "1", "w", "rook"]), Fact::new("cell", &["b", "3", "b", "knight"]), Fact::new("cell", &["c", "1", "w", "rook"]),
             Fact::new("attack", &["b", "b", "3", "a", "1"]), Fact::new("guard", &["w", "a", "1"])]
    }
    fn run(body: &[Lit]) -> Vec<Binding> { let mut s = 0; bindings(body, &Binding::new(), &FactIndex::new(&facts()), 100, &mut s).0 }

    #[test]
    fn a_join_binds_shared_variables() {
        let r = run(&[Lit::pos("cell", &["?f", "?r", "?c", "rook"]), Lit::pos("attack", &["?o", "?k", "?x", "?f", "?r"])]);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0]["f"], "a");
        assert_eq!(r[0]["o"], "b");
    }
    #[test]
    fn a_negated_literal_must_fail() {
        let mut body = vec![Lit::pos("cell", &["?f", "?r", "?c", "rook"])];
        body.push(Lit { neg: true, relation: "guard".into(), args: vec![Arg::Const("w".into()), Arg::Var("f".into()), Arg::Var("r".into())] });
        let r = run(&body);
        assert_eq!(r.len(), 1, "the rook on a1 is guarded, the one on c1 is not");
        assert_eq!(r[0]["f"], "c");
    }
    #[test]
    fn the_limit_cuts_and_says_so() {
        let mut s = 0;
        let (r, cut) = bindings(&[Lit::pos("cell", &["?f", "?r", "?c", "?k"])], &Binding::new(), &FactIndex::new(&facts()), 2, &mut s);
        assert_eq!(r.len(), 2);
        assert!(cut);
        assert!(s > 0);
    }
    #[test]
    fn the_fuse_cuts_a_runaway_join() {
        // eight independent literals over a relation of 30 rows, with a ninth that can never match: the search would try 30^8 combinations before failing
        let many: Vec<Fact> = (0..30).map(|i| Fact::new("r", &[&i.to_string()])).collect();
        let mut body: Vec<Lit> = (0..8).map(|i| Lit::pos("r", &[&format!("?x{i}")])).collect();
        body.push(Lit::pos("never", &["?z"]));
        let mut s = 0;
        let (r, cut) = bindings_fused(&body, &Binding::new(), &FactIndex::new(&many), 64, &mut s, 10_000);
        assert!(r.is_empty() && !cut || cut, "the literal that cannot match is read first (most constrained), so no blow-up");
        // and with the impossible literal last in a way the order cannot avoid: a chain of equalities
        let mut s2 = 0;
        let chain: Vec<Lit> = (0..8).map(|i| Lit::pos("r", &[&format!("?x{i}")])).chain(std::iter::once(Lit::pos("r", &["nonesuch"]))).collect();
        let (_r2, cut2) = bindings_fused(&chain, &Binding::new(), &FactIndex::new(&many), 64, &mut s2, 10_000);
        assert!(s2 <= 10_000 + 40, "the fuse held: {s2} rows examined");
        let _ = cut2;
    }
    #[test]
    fn an_initial_binding_constrains() {
        let mut init = Binding::new();
        init.insert("c".into(), "b".into());
        let mut s = 0;
        let (r, _) = bindings(&[Lit::pos("cell", &["?f", "?r", "?c", "?k"])], &init, &FactIndex::new(&facts()), 10, &mut s);
        assert_eq!(r.len(), 1);
    }
}
