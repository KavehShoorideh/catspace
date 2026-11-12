//! PLAN 78.39 THE GOAL'S TARGET, READ FROM THE DESCRIPTION. Walk the program from the goal clauses (those of the highest goal value) through the positive conditions of
//! their bodies, level by level, to the first clause that names a UNIT UNDER ATTACK: a state fluent with exactly one constant argument (the unit's kind, the other arguments
//! variables) joined, in the same body, with a derived relation that shares at least two of its variables (the unit's place) -- and a third, the unit's owner, that is not
//! shared. The kind, the fluent and the argument positions are READ; nothing here names a game's unit. If the walk does not end at one clear target the answer is Err
//! with the reason: unreadable, never a guess.
use cs_gdl::{Lit, Program, Rel, Term};
use std::collections::{BTreeSet, HashMap};

/// What the goal attacks: where the description keeps the unit and which argument is what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalTarget {
    /// The state fluent that holds units (its name without `true.`), and its arity.
    pub fluent: String,
    pub arity: usize,
    /// The argument that is the unit's kind, and the kind (the constant the clause names).
    pub kind_arg: usize,
    pub kind: String,
    /// The argument that is the unit's owner (a role).
    pub owner_arg: usize,
    /// The arguments that are the unit's place, in order.
    pub place_args: Vec<usize>,
    /// The derived relation the clause joins the unit with (the one that says "under attack").
    pub under: String,
    /// The relations walked from `goal` to the clause that names it (goal first).
    pub path: Vec<String>,
}

fn goal_value(p: &Program, c: usize) -> Option<i64> {
    match p.clauses[c].head.1.last() { Some(Term::Const(t)) => p.terms.name(*t).parse().ok(), _ => None }
}

/// The matches of one clause (usually none or one): (GoalTarget without its path).
fn matches(p: &Program, c: usize) -> Vec<GoalTarget> {
    let cl = &p.clauses[c];
    let mut out = vec![];
    for lit in &cl.body {
        let Lit::Pos(r, args) = lit else { continue };
        let Some(fluent) = p.rel_name(*r).strip_prefix("true.") else { continue };
        let consts: Vec<usize> = args.iter().enumerate().filter(|(_, a)| matches!(a, Term::Const(_))).map(|(i, _)| i).collect();
        if consts.len() != 1 || args.iter().any(|a| matches!(a, Term::Fun(..))) { continue; }
        let kind_arg = consts[0];
        let Term::Const(kt) = &args[kind_arg] else { continue };
        let vars: BTreeSet<String> = args.iter().filter_map(|a| if let Term::Var(v) = a { Some(v.clone()) } else { None }).collect();
        if vars.len() != args.len() - 1 { continue; }                            // (a variable repeated: not a plain unit)
        // the derived literals of the same body that share variables with the unit
        let mut best: Vec<(usize, Rel, BTreeSet<String>)> = vec![];
        for (j, other) in cl.body.iter().enumerate() {
            let Lit::Pos(r2, args2) = other else { continue };
            let n2 = p.rel_name(*r2);
            if n2.starts_with("true.") || n2 == "does" || n2 == "record" { continue; }
            let mut vs = BTreeSet::new();
            for a in args2 { a.vars(&mut vs); }
            let shared: BTreeSet<String> = vs.intersection(&vars).cloned().collect();
            if !shared.is_empty() { best.push((j, *r2, shared)); }
        }
        let Some(top) = best.iter().map(|b| b.2.len()).max() else { continue };
        let tops: Vec<&(usize, Rel, BTreeSet<String>)> = best.iter().filter(|b| b.2.len() == top).collect();
        if top < 2 || tops.len() != 1 || vars.len() - top != 1 { continue; }
        let (_, under, shared) = tops[0];
        let place_args: Vec<usize> = args.iter().enumerate().filter(|(_, a)| matches!(a, Term::Var(v) if shared.contains(v))).map(|(i, _)| i).collect();
        let owner_arg = (0..args.len()).find(|i| *i != kind_arg && !place_args.contains(i)).unwrap();
        out.push(GoalTarget { fluent: fluent.to_string(), arity: args.len(), kind_arg, kind: p.terms.name(*kt).to_string(), owner_arg, place_args, under: p.rel_name(*under).to_string(), path: vec![] });
    }
    out
}

/// The goal's target of a description text (see the module): Err names why it is unreadable.
pub fn goal_target(text: &str) -> Result<GoalTarget, String> {
    let p = Program::load(text).map_err(|e| e.0)?;
    let goal = p.rel_id("goal").ok_or("unreadable: the description has no goal")?;
    let mut level: Vec<usize> = p.rules_of[goal as usize].clone();
    if level.is_empty() { return Err("unreadable: the description has no goal clause".into()); }
    if let Some(top) = level.iter().filter_map(|c| goal_value(&p, *c)).max() { level.retain(|c| goal_value(&p, *c) == Some(top)); }
    let mut parent: HashMap<Rel, Rel> = HashMap::new();
    let mut seen: BTreeSet<Rel> = BTreeSet::new();
    seen.insert(goal);
    while !level.is_empty() {
        let mut found: Vec<(GoalTarget, Rel)> = vec![];
        for &c in &level { for m in matches(&p, c) { found.push((m, p.clauses[c].head.0)); } }
        found.dedup_by(|a, b| a.0 == b.0);
        let mut distinct: Vec<(GoalTarget, Rel)> = vec![];
        for f in found { if !distinct.iter().any(|d| d.0 == f.0) { distinct.push(f); } }
        match distinct.len() {
            0 => {}
            1 => {
                let (mut t, head) = distinct.pop().unwrap();
                let mut path = vec![p.rel_name(head).to_string()];
                let mut cur = head;
                while let Some(up) = parent.get(&cur) { path.push(p.rel_name(*up).to_string()); cur = *up; }
                path.reverse();
                t.path = path;
                return Ok(t);
            }
            n => return Err(format!("unreadable: {n} different targets at the same depth of the goal")),
        }
        let mut next: Vec<usize> = vec![];
        for &c in &level {
            let head = p.clauses[c].head.0;
            for lit in &p.clauses[c].body {
                if let Lit::Pos(r, _) = lit {
                    if p.rel_name(*r).starts_with("true.") || p.rel_name(*r) == "does" || p.rel_name(*r) == "record" { continue; }
                    if seen.insert(*r) { parent.insert(*r, head); next.extend(p.rules_of[*r as usize].iter().copied()); }
                }
            }
        }
        level = next;
    }
    Err("unreadable: no condition of the goal names a unit under attack".into())
}
