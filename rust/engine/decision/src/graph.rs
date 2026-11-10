//! PLAN 78.21 THE TYPED FACT GRAPH, BUILT IN RUST (the one graph builder; the torch side trains on what this builds). GENERIC: everything is read from the game
//! description's sorts, signatures and static relations and from a graph vocabulary (which relations to show); nothing here knows a game.
//!
//!   nodes   term   TYPED BY SORT: token "<sort>:<constant>". Product sorts (`objects`: name -> component sorts) FUSE consecutive slots of that sort sequence into one
//!                  node, token "<name>:<c1><c2>..", linked to its component nodes by `part<i>` edges.
//!           fact   one per row of the state relations (the base facts), the derived relations (the fact layer), the static relations (identical in every graph) and
//!                  the rows a move adds that the position lacks. Feature [relation id, kind (0 state, 1 derived, 2 static), present].
//!           move   one per legal move in the game's move order; edges `removes` / `adds` over the vocabulary's delta relations (base facts only, as cards.rs).
//!           claim  the type exists and is empty (a later learner binds lemmas to facts: `binds` edges).
//!   edges   (fact, arg<i>, term) i < MAX_ARITY by position after fusion; (term, part<i>, term); (move, removes|adds, fact); (claim, binds, fact); each with its reverse.
//!   roles   facts are in the mover's perspective (`us` / `them`, `r<k>` for more roles); the role sort's nodes are us / them and the absolute roles (static rows keep them).
//!   vocab   CONSTANT, read from the description: `term_vocab` (every "<sort>:<member>" of the sorts in use, every object token) and `rel_vocab` ("move", "claim", then
//!           the graph relations sorted). Ids are index + 1 (0 = UNK, a token the vocabularies lack).
//!   always-present terms: every member of every sort in use (and every object whose components exist: the rows of a static relation named like the object with exactly
//!           its signature, else the cross product), except the `scales` sorts and sorts with pattern members (move terms), which become nodes only where they occur
//!           (a static row occurring counts). So the node set of a graph is identical in every position apart from those, and an unmentioned constant is an edgeless node.
use cs_game::gdl_game::{render_term, row_args, FactView};
use cs_game::{Game, Vocabulary};
use cs_gdl::{Program, SortMember};
use cs_memory::op::Effects;
use cs_memory::perspective;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

pub const MAX_ARITY: usize = 6;
pub const UNK: &str = "<unk>";
pub const MOVE: &str = "move";
pub const CLAIM: &str = "claim";
pub const KIND_STATE: u32 = 0;
pub const KIND_DERIVED: u32 = 1;
pub const KIND_STATIC: u32 = 2;

/// One argument group of a relation after fusion: a plain sort, or an object over component sorts.
#[derive(Clone, Debug)]
struct Slot { name: String, parts: Vec<String> }

#[derive(Clone, Debug)]
struct RelPlan { id: u32, kind: u32, slots: Vec<Slot>, width: usize }

/// The edge types of the graph (forward names; the reverses are `rev_<name>`), in a fixed order.
pub fn edge_names(max_parts: usize) -> Vec<String> {
    let mut v: Vec<String> = (0..MAX_ARITY).map(|i| format!("arg{i}")).collect();
    v.extend((0..max_parts).map(|i| format!("part{i}")));
    v.extend(["removes", "adds", "binds"].map(String::from));
    v
}

/// The graph of one position: integer features and edge lists (forward types; the net reads reverses by swapping).
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub terms: Vec<String>,
    pub term_ids: Vec<u32>,
    /// per fact node: [relation vocab id, kind, present]
    pub facts: Vec<[u32; 3]>,
    pub moves: Vec<String>,
    /// (forward edge name, sources, destinations), in `edge_names` order
    pub edges: Vec<(String, Vec<u32>, Vec<u32>)>,
    pub n_static_terms: usize,
    pub n_static_facts: usize,
    /// seconds spent in the fact layer while building
    pub secs_facts: f64,
}

impl Graph {
    pub fn edge(&self, name: &str) -> Option<(&[u32], &[u32])> { self.edges.iter().find(|e| e.0 == name).map(|e| (&e.1[..], &e.2[..])) }
}

#[derive(Clone)]
struct Frozen { terms: Vec<String>, term_ids: Vec<u32>, facts: Vec<[u32; 3]>, edges: Vec<(String, Vec<u32>, Vec<u32>)>, ix: HashMap<String, u32> }

pub struct GraphSpec {
    view: FactView,
    vocab: Vocabulary,
    plans: HashMap<String, RelPlan>,
    /// token -> vocab id (1-based)
    term_vocab: Vec<String>,
    term_id: HashMap<String, u32>,
    rel_vocab: Vec<String>,
    /// sorts (and objects) whose members become nodes only where they occur
    open: HashSet<String>,
    objects: BTreeMap<String, Vec<String>>,
    members: HashMap<String, Vec<String>>,
    max_parts: usize,
    base: Frozen,
}

fn strs(v: &Value, key: &str) -> Result<Vec<String>, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(vec![]),
        Some(a) => a.as_array().ok_or_else(|| format!("graph vocabulary.{key} is a list"))?.iter()
            .map(|x| x.as_str().map(String::from).ok_or_else(|| format!("graph vocabulary.{key} holds text"))).collect(),
    }
}

impl GraphSpec {
    /// The spec of a description (its text) under a graph vocabulary (JSON: state, derived, static, delta, objects, scales, units).
    pub fn new(description_text: &str, graph_vocabulary_json: &str) -> Result<GraphSpec, String> {
        let cfg: Value = serde_json::from_str(graph_vocabulary_json).map_err(|e| format!("graph vocabulary: {e}"))?;
        let (state, derived, statik, dl, scales) = (strs(&cfg, "state")?, strs(&cfg, "derived")?, strs(&cfg, "static")?, strs(&cfg, "delta")?, strs(&cfg, "scales")?);
        let mut objects: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if let Some(o) = cfg.get("objects").filter(|o| !o.is_null()) {
            for (name, comps) in o.as_object().ok_or("graph vocabulary.objects is a map")? {
                let comps: Vec<String> = comps.as_array().ok_or("an object lists its component sorts")?.iter().filter_map(|c| c.as_str().map(String::from)).collect();
                if comps.len() < 2 { return Err(format!("object {name} needs at least two component sorts")); }
                objects.insert(name.clone(), comps);
            }
        }
        let mut vjson = cfg.clone();
        if let Some(m) = vjson.as_object_mut() { m.remove("objects"); }
        let vocab = cs_memory::snapshot::vocabulary_from_json(&vjson)?;
        let prog = Program::load(description_text).map_err(|e| e.0)?;
        let view = FactView::new(description_text, vocab.clone())?;

        // the relations shown, their kinds and signatures
        let mut kinds: BTreeMap<String, u32> = BTreeMap::new();
        for (list, kind) in [(&state, KIND_STATE), (&derived, KIND_DERIVED), (&statik, KIND_STATIC)] {
            for r in list {
                if kinds.insert(r.clone(), kind).is_some_and(|k| k != kind) { return Err(format!("relation {r} is listed under two kinds")); }
            }
        }
        for r in &dl { if kinds.get(r) != Some(&KIND_STATE) { return Err(format!("delta relation {r} must be a state relation (a move's effect is read from base facts)")); } }
        let rel_vocab: Vec<String> = [MOVE.to_string(), CLAIM.to_string()].into_iter().chain(kinds.keys().cloned()).collect();
        // objects by fusion priority: longest first, then name
        let mut order: Vec<(&String, &Vec<String>)> = objects.iter().collect();
        order.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
        let mut plans = HashMap::new();
        let mut used_objects: BTreeSet<String> = BTreeSet::new();
        let mut in_use: BTreeSet<String> = BTreeSet::new();
        for (rel, &kind) in &kinds {
            let sig = prog.signature(rel).ok_or_else(|| format!("relation {rel} has no signature in the description (every relation of the graph needs one)"))?;
            let mut slots = vec![];
            let mut i = 0;
            while i < sig.len() {
                if let Some((name, comps)) = order.iter().find(|(_, c)| sig.len() >= i + c.len() && sig[i..i + c.len()] == c[..]) {
                    for c in comps.iter() { in_use.insert(c.clone()); }
                    used_objects.insert((*name).clone());
                    slots.push(Slot { name: (*name).clone(), parts: (*comps).clone() });
                    i += comps.len();
                } else {
                    in_use.insert(sig[i].clone());
                    slots.push(Slot { name: sig[i].clone(), parts: vec![] });
                    i += 1;
                }
            }
            let id = 1 + rel_vocab.iter().position(|r| r == rel).unwrap() as u32;
            plans.insert(rel.clone(), RelPlan { id, kind, slots, width: sig.len() });
        }
        // sorts: members, openness
        let mut members: HashMap<String, Vec<String>> = HashMap::new();
        let mut open: HashSet<String> = HashSet::new();
        for s in &in_use {
            let ms = prog.sorts.get(s).ok_or_else(|| format!("the description has no sort {s}"))?;
            members.insert(s.clone(), ms.iter().filter_map(|m| if let SortMember::Const(c) = m { Some(c.clone()) } else { None }).collect());
            if scales.contains(s) || ms.iter().any(|m| matches!(m, SortMember::Pattern(_))) { open.insert(s.clone()); }
        }
        // the role sort and its perspective names
        let (mut role_sort, mut nroles) = (None, 0);
        if let (Some(sig), Some(r)) = (prog.signature("role"), prog.rel_id("role")) {
            if let (Some(s), Some(rel)) = (sig.first(), prog.statics.get(&r)) { nroles = rel.order.len(); role_sort = Some(s.clone()); }
        }
        if let Some(rs) = role_sort.as_ref().filter(|s| in_use.contains(*s)) {
            let names: Vec<String> = (0..nroles).map(|i| if i == 0 { "us".into() } else if nroles == 2 { "them".into() } else { format!("r{i}") }).collect();
            let ms = members.get_mut(rs).unwrap();
            for n in names { if !ms.contains(&n) { ms.push(n); } }
        }
        // object members: the rows of a static relation named like the object (exact signature) or the cross product
        let static_rows = |name: &str, comps: &[String]| -> Option<Vec<Vec<String>>> {
            let r = prog.rel_id(name)?;
            if prog.signature(name)?.as_slice() != comps { return None; }
            Some(prog.statics.get(&r)?.order.iter().map(|row| row_args(row).iter().map(|t| render_term(&prog, *t)).collect()).collect())
        };
        let mut object_rows: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
        for name in &used_objects {
            let comps = &objects[name];
            if comps.iter().any(|c| open.contains(c)) { open.insert(name.clone()); continue; }
            let rows = static_rows(name, comps).unwrap_or_else(|| {
                let mut acc: Vec<Vec<String>> = vec![vec![]];
                for c in comps { acc = acc.iter().flat_map(|p| members[c].iter().map(move |m| { let mut q = p.clone(); q.push(m.clone()); q })).collect(); }
                acc
            });
            object_rows.insert(name.clone(), rows);
        }
        // vocabularies
        let mut tv: BTreeSet<String> = BTreeSet::new();
        for (s, ms) in &members { for m in ms { tv.insert(format!("{s}:{m}")); } }
        for (name, rows) in &object_rows { for r in rows { tv.insert(format!("{name}:{}", r.concat())); } }
        let term_vocab: Vec<String> = tv.into_iter().collect();
        let term_id: HashMap<String, u32> = term_vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32 + 1)).collect();
        let max_parts = used_objects.iter().map(|o| objects[o].len()).max().unwrap_or(0);

        let mut spec = GraphSpec { view, vocab, plans, term_vocab, term_id, rel_vocab, open, objects, members,
                                   max_parts, base: Frozen { terms: vec![], term_ids: vec![], facts: vec![], edges: vec![], ix: HashMap::new() } };
        // the static base: constants (sorted by token), part edges of the closed objects, the static facts
        let empty = Frozen { terms: vec![], term_ids: vec![], facts: vec![], edges: vec![], ix: HashMap::new() };
        let mut b = Build::new(&spec, &empty);
        let mut consts: BTreeSet<String> = BTreeSet::new();
        for (s, ms) in &spec.members { if !spec.open.contains(s) { for m in ms { consts.insert(format!("{s}:{m}")); } } }
        for (name, rows) in &object_rows { for r in rows { consts.insert(format!("{name}:{}", r.concat())); } }
        for t in &consts { b.push_term(t.clone()); }
        for (name, rows) in &object_rows {
            for r in rows {
                let obj = b.local[&format!("{name}:{}", r.concat())];
                for (i, (c, sort)) in r.iter().zip(&spec.objects[name]).enumerate() {
                    let comp = b.local[&format!("{sort}:{c}")];
                    b.edge(&format!("part{i}"), obj, comp);
                }
            }
        }
        let mut stat: Vec<(String, Vec<Vec<String>>)> = vec![];
        for r in &statik {
            let rid = prog.rel_id(r).ok_or_else(|| format!("static relation {r} is not in the description"))?;
            let rel = prog.statics.get(&rid).ok_or_else(|| format!("{r} is not a static relation (it reads the state or the moves)"))?;
            let mut rows: Vec<Vec<String>> = rel.order.iter().map(|row| row_args(row).iter().map(|t| render_term(&prog, *t)).collect()).collect();
            rows.sort();
            stat.push((r.clone(), rows));
        }
        for (rel, rows) in &stat { for row in rows { b.fact(rel, row, 1)?; } }
        let frozen = b.freeze();
        spec.base = frozen;
        Ok(spec)
    }

    pub fn term_vocab(&self) -> &[String] { &self.term_vocab }
    pub fn rel_vocab(&self) -> &[String] { &self.rel_vocab }
    pub fn max_parts(&self) -> usize { self.max_parts }
    pub fn vocabulary(&self) -> &Vocabulary { &self.vocab }
    /// The relation vocab id of "move" and "claim" (1-based; the graph relations follow).
    pub fn move_rel_id(&self) -> u32 { 1 }
    pub fn claim_rel_id(&self) -> u32 { 2 }
    /// How many term nodes and fact nodes every graph of the game starts with (the constants and the static facts).
    pub fn n_static(&self) -> (usize, usize) { (self.base.terms.len(), self.base.facts.len()) }

    /// The graph of the position `s`: the fact layer in the mover's perspective, the legal moves' effects on the delta relations.
    pub fn graph<G: Game>(&self, g: &G, s: &G::State) -> Result<Graph, String> {
        let me = g.mover(s).ok_or("the position is terminal")?;
        let base = g.base_facts(s);
        if base.is_empty() { return Err("the game gives no facts for this position".into()); }
        let roles = g.roles();
        let t0 = Instant::now();
        let facts = perspective(&self.view.facts(&base)?, &roles, me);
        let secs_facts = t0.elapsed().as_secs_f64();
        let mut b = Build::new(self, &self.base);
        for f in &facts {
            if let Some(p) = self.plans.get(&f.relation) { if p.kind != KIND_STATIC { b.fact(&f.relation, &f.args, 1)?; } }
        }
        let moves = g.moves(s);
        let mut names = Vec::with_capacity(moves.len());
        let base_after = |m: &G::Move| g.base_facts(&g.apply(s, m));
        let effects = Effects::at(&self.view, &self.vocab.delta, &base)?;
        for (mi, m) in moves.iter().enumerate() {
            names.push(g.move_name(s, m));
            let (r, a) = effects.of(&base_after(m))?;
            for (kind, rows) in [("removes", perspective(&r, &roles, me)), ("adds", perspective(&a, &roles, me))] {
                for f in rows {
                    let fi = b.fact(&f.relation, &f.args, 0)?;
                    b.edge(kind, mi as u32, fi);
                }
            }
        }
        let mut out = b.finish(names);
        out.secs_facts = secs_facts;
        Ok(out)
    }
}

struct Build<'a> {
    spec: &'a GraphSpec,
    fixed: &'a Frozen,
    terms: Vec<String>,
    term_ids: Vec<u32>,
    local: HashMap<String, u32>,
    facts: Vec<[u32; 3]>,
    fact_ix: HashMap<(u32, Vec<u32>), u32>,
    edges: Vec<(String, Vec<u32>, Vec<u32>)>,
}

impl<'a> Build<'a> {
    fn new(spec: &'a GraphSpec, fixed: &'a Frozen) -> Build<'a> {
        let edges = edge_names(spec.max_parts).into_iter().map(|n| (n, vec![], vec![])).collect();
        Build { spec, fixed, terms: vec![], term_ids: vec![], local: HashMap::new(), facts: vec![], fact_ix: HashMap::new(), edges }
    }
    fn n_fixed_terms(&self) -> u32 { self.fixed.terms.len() as u32 }
    fn push_term(&mut self, token: String) -> u32 {
        let ix = self.n_fixed_terms() + self.terms.len() as u32;
        self.term_ids.push(self.spec.term_id.get(&token).copied().unwrap_or(0));
        self.local.insert(token.clone(), ix);
        self.terms.push(token);
        ix
    }
    fn edge(&mut self, name: &str, src: u32, dst: u32) {
        let e = self.edges.iter_mut().find(|e| e.0 == name).expect("an edge type of the graph");
        e.1.push(src);
        e.2.push(dst);
    }
    /// The node of the term `token` of sort/object `sort`: a constant of the base, one made earlier, or (an open sort only) a new node.
    fn term(&mut self, sort: &str, token: String, parts: &[String], args: &[String]) -> Result<u32, String> {
        if let Some(&i) = self.fixed.ix.get(&token) { return Ok(i); }
        if let Some(&i) = self.local.get(&token) { return Ok(i); }
        if !self.spec.open.contains(sort) { return Err(format!("{token}: not a member of {sort}")); }
        let ix = self.push_term(token);
        for (i, (c, cs)) in args.iter().zip(parts).enumerate() {
            let comp = self.term(cs, format!("{cs}:{c}"), &[], &[])?;
            self.edge(&format!("part{i}"), ix, comp);
        }
        Ok(ix)
    }
    /// The fact node of a row (added as new with `present`, or the existing one).
    fn fact(&mut self, rel: &str, args: &[String], present: u32) -> Result<u32, String> {
        let spec = self.spec;
        let plan = spec.plans.get(rel).ok_or_else(|| format!("relation {rel} is not in the graph vocabulary"))?;
        if args.len() != plan.width { return Err(format!("{rel} has {} arguments, its signature {}", args.len(), plan.width)); }
        let mut nodes = Vec::with_capacity(plan.slots.len());
        let mut at = 0;
        for sl in &plan.slots {
            if sl.parts.is_empty() {
                nodes.push(self.term(&sl.name, format!("{}:{}", sl.name, args[at]), &[], &[])?);
                at += 1;
            } else {
                let k = sl.parts.len();
                nodes.push(self.term(&sl.name, format!("{}:{}", sl.name, args[at..at + k].concat()), &sl.parts, &args[at..at + k])?);
                at += k;
            }
        }
        let key = (plan.id, nodes);
        if plan.kind != KIND_STATIC { if let Some(&i) = self.fact_ix.get(&key) { return Ok(i); } }
        let fi = self.fixed.facts.len() as u32 + self.facts.len() as u32;
        for (p, &n) in key.1.iter().enumerate() { self.edge(&format!("arg{}", p.min(MAX_ARITY - 1)), fi, n); }
        self.facts.push([plan.id, plan.kind, present]);
        if plan.kind != KIND_STATIC { self.fact_ix.insert(key, fi); }
        Ok(fi)
    }
    fn freeze(self) -> Frozen {
        let mut ix = self.fixed.ix.clone();
        ix.extend(self.local);
        Frozen { terms: self.terms, term_ids: self.term_ids, facts: self.facts, edges: self.edges, ix }
    }
    fn finish(self, moves: Vec<String>) -> Graph {
        let f = self.fixed;
        let mut edges = self.edges;
        for (e, base) in edges.iter_mut().zip(&f.edges) {
            let mut s = base.1.clone();
            s.extend_from_slice(&e.1);
            let mut d = base.2.clone();
            d.extend_from_slice(&e.2);
            e.1 = s;
            e.2 = d;
        }
        let mut terms = f.terms.clone();
        terms.extend(self.terms);
        let mut term_ids = f.term_ids.clone();
        term_ids.extend(self.term_ids);
        let mut facts = f.facts.clone();
        facts.extend(self.facts);
        Graph { terms, term_ids, facts, moves, edges, n_static_terms: f.terms.len(), n_static_facts: f.facts.len(), secs_facts: 0.0 }
    }
}
