//! PLAN 79.24 PART B: THE MEASURE OF PROGRESS IN LAYER 3 (moved here from cs_tactics::measure, plan 79.17 part B; SYSTEM_SPEC 7 "the small endings as the test of means and ends, and a measure of progress", 11 "layer 3 holds two implementations"). Game-free: it
//! reads a game only through `Game` and `Placed` and the board only through layer 2's graph maker's `Static` (places, steps by direction, lines, the edge distance); no unit, no place, no kind is named. A unit is only a GROUP: the units of one kind on one side,
//! the side being "the side the measure is for" or the other. The code of the atoms, the terms and their reading is the code that was in cs_tactics::measure, moved as it was (the means-ends crate's tests still call it through the old path, which re-exports
//! this module).
//!
//! WHAT A MEASURE IS (October's lesson, plans 73.5 and 73.5b, Lean `forced_by_measure`): a number or a tuple of numbers read from a position, such that from every position of a class some play of ours of at most d moves LOWERS it against every reply, until the
//! game's end is found. It is not a value: it never enters a line's value (SYSTEM_SPEC 7 and AGENTS rule 7); it orders and it chooses among moves the calculation holds equal.
//!
//! INTERFACE (what this module gives; the question that needs a search, `lowering` and `question`, stays in cs_tactics::measure).
//!   `Geometry::new(&Static)`                     what the board gives once: the line families, a place's coordinate along each, its distance to the end of each family, the distance in steps between any two places, the distance to the places of fewest steps,
//!                                                the nearest edge.
//!   `Group { theirs, kind }`                     the units of one kind of the measure's side (`theirs` false) or of the other side (true).
//!   `Atom`                                       one reading of a position; `Atom::name`/`Atom::parse` (a name is its definition: `region(0,3)`, `dist(0,3,1,3)`, `span(1,3,0)`, `turn`).
//!   `Term`                                       an atom; the sum or the product of two scalar terms; a tuple of scalar terms compared in order. `Term::parse`/`Term::text` (`[sum A B]`, `[product A B]`, `[tuple A B C]`).
//!   `library(groups, &Geometry)`                 every atom the groups and the families give: the same code for every class.
//!   `value(&Term, &Reader) -> Vec<i64>`          the tuple of numbers of a position (a pure function of it); `Reader::new(g, geo, s, side)`.
//!   `lower(a, b)`                                a is strictly below b in the order of the tuple.
//!   `Progress { term, within, geo, standing }`   the measure as an implementation of layer 3's one question (`Assess`): `Progress::from_memory(reader, id, geo)` reads the statement `id` of the long-term memory of one game (`Says::Lower { term, within }`, its
//!                                                standing the sample it was mined on); `assess` returns the tuple negated as the rank, the tuple as `raw`, the atoms it read, and the standing as `sure`.
use crate::{Answer, Assess, Question, Unanswered};
use cs_game::graph::Static;
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use cs_rep::names::{Kind, Place, PlaceSet};
use cs_rep::statement::Says;
use cs_rep::store::Reader as Memory;
use std::cell::RefCell;
use std::collections::HashMap;

const FAR: i32 = 99;

/// What the board gives once (see the module doc).
#[derive(Debug, Clone)]
pub struct Geometry {
    /// the direction (an index into the static part's directions) each line family is named in, in order
    pub families: Vec<usize>,
    /// [family][place]: the place's index along its line of that family (0 where it is on none)
    coord: Vec<Vec<i32>>,
    /// [family][place]: the steps to the end of the family's lines, the nearer way
    edge_f: Vec<Vec<i32>>,
    /// [place][place]: the steps between two places along any direction
    dist: Vec<Vec<i32>>,
    corner: Vec<i32>,
    edge: Vec<i32>,
    np: usize,
}

impl Geometry {
    pub fn new(st: &Static) -> Geometry {
        let np = st.names.places.len();
        let mut families: Vec<usize> = st.lines.iter().map(|l| l.direction).collect();
        families.sort();
        families.dedup();
        let coord: Vec<Vec<i32>> = families.iter().map(|&d| {
            let mut c = vec![0i32; np];
            for p in 0..np { for &(li, ix) in &st.place_lines[p] { if st.lines[li as usize].direction == d { c[p] = ix as i32; } } }
            c
        }).collect();
        let edge_f: Vec<Vec<i32>> = families.iter().map(|&d| (0..np).map(|p| { let a = st.edge_distance[d][p] as i32; match st.reverse[d] { Some(e) => a.min(st.edge_distance[e][p] as i32), None => a } }).collect()).collect();
        let nd = st.directions.len();
        let mut dist = vec![vec![FAR; np]; np];
        for s in 0..np {
            dist[s][s] = 0;
            let mut queue = std::collections::VecDeque::from([s]);
            while let Some(p) = queue.pop_front() {
                for d in 0..nd { if let Some(q) = st.next[d][p] { let q = q as usize; if dist[s][q] == FAR { dist[s][q] = dist[s][p] + 1; queue.push_back(q); } } }
            }
        }
        let degree: Vec<usize> = (0..np).map(|p| (0..nd).filter(|&d| st.next[d][p].is_some()).count()).collect();
        let least = degree.iter().copied().min().unwrap_or(0);
        let corners: Vec<usize> = (0..np).filter(|&p| degree[p] == least).collect();
        let corner = (0..np).map(|p| corners.iter().map(|&c| dist[p][c]).min().unwrap_or(FAR)).collect();
        let edge = (0..np).map(|p| st.nearest_edge(p as Place) as i32).collect();
        Geometry { families, coord, edge_f, dist, corner, edge, np }
    }
    pub fn family_count(&self) -> usize { self.families.len() }
}

/// The units of one kind on one side: `theirs` false the measure's side, true the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Group { pub theirs: bool, pub kind: Kind }

impl Group {
    fn text(&self) -> String { format!("{},{}", self.theirs as u8, self.kind) }
    fn parse(side: &str, kind: &str) -> Result<Group, String> {
        Ok(Group { theirs: match side { "0" => false, "1" => true, o => return Err(format!("no side {o}")) }, kind: kind.parse::<Kind>().map_err(|e| e.to_string())? })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Atom { Region(Group), Span(Group, usize), Edge(Group), Corner(Group), Dist(Group, Group), Axis(Group, Group, usize), AxisEdge(Group, usize), Turn }

impl Atom {
    pub fn name(&self) -> String {
        match self {
            Atom::Region(g) => format!("region({})", g.text()), Atom::Span(g, f) => format!("span({},{f})", g.text()), Atom::Edge(g) => format!("edge({})", g.text()), Atom::Corner(g) => format!("corner({})", g.text()),
            Atom::Dist(a, b) => format!("dist({},{})", a.text(), b.text()), Atom::Axis(a, b, f) => format!("axis({},{},{f})", a.text(), b.text()), Atom::AxisEdge(g, f) => format!("axisedge({},{f})", g.text()), Atom::Turn => "turn".to_string(),
        }
    }
    pub fn parse(s: &str) -> Result<Atom, String> {
        if s == "turn" { return Ok(Atom::Turn); }
        let (head, rest) = s.split_once('(').ok_or_else(|| format!("no atom {s}"))?;
        let args: Vec<&str> = rest.strip_suffix(')').ok_or_else(|| format!("no atom {s}"))?.split(',').collect();
        let fam = |x: &str| x.parse::<usize>().map_err(|e| e.to_string());
        match (head, args.len()) {
            ("region", 2) => Ok(Atom::Region(Group::parse(args[0], args[1])?)), ("edge", 2) => Ok(Atom::Edge(Group::parse(args[0], args[1])?)), ("corner", 2) => Ok(Atom::Corner(Group::parse(args[0], args[1])?)),
            ("span", 3) => Ok(Atom::Span(Group::parse(args[0], args[1])?, fam(args[2])?)), ("axisedge", 3) => Ok(Atom::AxisEdge(Group::parse(args[0], args[1])?, fam(args[2])?)),
            ("dist", 4) => Ok(Atom::Dist(Group::parse(args[0], args[1])?, Group::parse(args[2], args[3])?)),
            ("axis", 5) => Ok(Atom::Axis(Group::parse(args[0], args[1])?, Group::parse(args[2], args[3])?, fam(args[4])?)),
            _ => Err(format!("no atom {s}")),
        }
    }
}

/// Every atom the groups and the families give (the same for every class): per group the region, its span along each family, the edge and corner distances, the distance to the end of each family; per pair of different groups the distance and the distance along each family; the turn.
pub fn library(groups: &[Group], geo: &Geometry) -> Vec<Atom> {
    let nf = geo.family_count();
    let mut v = vec![];
    for &g in groups {
        v.push(Atom::Region(g));
        for f in 0..nf { v.push(Atom::Span(g, f)); }
        v.push(Atom::Edge(g));
        v.push(Atom::Corner(g));
        for f in 0..nf { v.push(Atom::AxisEdge(g, f)); }
    }
    for (i, &a) in groups.iter().enumerate() {
        for &b in &groups[i + 1..] {
            v.push(Atom::Dist(a, b));
            for f in 0..nf { v.push(Atom::Axis(a, b, f)); }
        }
    }
    v.push(Atom::Turn);
    v
}

/// The groups present in a position, the measure's side first: (theirs, kind) in the order of the numbers.
pub fn groups_of<G: Game + Placed>(g: &G, s: &G::State, side: Role) -> Vec<Group> {
    let mut v: Vec<Group> = g.cells(s).into_iter().map(|(o, k, _)| Group { theirs: o != side, kind: k }).collect();
    v.sort();
    v.dedup();
    v
}

/// A position read for the atoms (the regions are made once and kept).
pub struct Reader<'a, G: Game + Placed> {
    g: &'a G,
    geo: &'a Geometry,
    s: &'a G::State,
    side: Role,
    cells: Vec<(Role, Kind, Place)>,
    regions: RefCell<HashMap<Group, PlaceSet>>,
}

impl<'a, G: Game + Placed> Reader<'a, G> {
    pub fn new(g: &'a G, geo: &'a Geometry, s: &'a G::State, side: Role) -> Reader<'a, G> { Reader { g, geo, s, side, cells: g.cells(s), regions: RefCell::new(HashMap::new()) } }
    fn owner(&self, grp: Group) -> Role { if grp.theirs { self.g.other(self.side) } else { self.side } }
    fn places(&self, grp: Group) -> Vec<Place> { let o = self.owner(grp); self.cells.iter().filter(|(ow, k, _)| *ow == o && *k == grp.kind).map(|c| c.2).collect() }
    /// Whether the group has a unit in the position.
    pub fn present(&self, grp: Group) -> bool { !self.places(grp).is_empty() }
    fn region(&self, grp: Group) -> PlaceSet {
        if let Some(r) = self.regions.borrow().get(&grp) { return r.clone(); }
        let owner = self.owner(grp);
        let foe = self.g.other(owner);
        let mut all = PlaceSet::new(self.geo.np);
        for at in self.places(grp) {
            let lifted = self.g.without(self.s, at).unwrap_or_else(|| self.s.clone());
            let mut seen = PlaceSet::new(self.geo.np);
            seen.insert(at);
            let mut stack = vec![at];
            while let Some(p) = stack.pop() {
                for q in self.g.reach_from(&lifted, owner, grp.kind, p) {
                    if seen.contains(q) || self.g.strikes_place(&lifted, foe, q) { continue; }
                    seen.insert(q);
                    stack.push(q);
                }
            }
            all.union_with(&seen);
        }
        self.regions.borrow_mut().insert(grp, all.clone());
        all
    }
    pub fn atom(&self, a: &Atom) -> i64 {
        let least = |it: &mut dyn Iterator<Item = i32>| it.min().unwrap_or(FAR) as i64;
        match *a {
            Atom::Region(grp) => self.region(grp).len() as i64,
            Atom::Span(grp, f) => { let r = self.region(grp); let (lo, hi) = r.iter().fold((i32::MAX, i32::MIN), |(lo, hi), p| (lo.min(self.geo.coord[f][p as usize]), hi.max(self.geo.coord[f][p as usize]))); if lo > hi { 0 } else { (hi - lo + 1) as i64 } }
            Atom::Edge(grp) => least(&mut self.places(grp).into_iter().map(|p| self.geo.edge[p as usize])),
            Atom::Corner(grp) => least(&mut self.places(grp).into_iter().map(|p| self.geo.corner[p as usize])),
            Atom::AxisEdge(grp, f) => least(&mut self.places(grp).into_iter().map(|p| self.geo.edge_f[f][p as usize])),
            Atom::Dist(a, b) => { let (pa, pb) = (self.places(a), self.places(b)); least(&mut pa.iter().flat_map(|&x| pb.iter().map(move |&y| (x, y))).map(|(x, y)| self.geo.dist[x as usize][y as usize])) }
            Atom::Axis(a, b, f) => { let (pa, pb) = (self.places(a), self.places(b)); least(&mut pa.iter().flat_map(|&x| pb.iter().map(move |&y| (x, y))).map(|(x, y)| (self.geo.coord[f][x as usize] - self.geo.coord[f][y as usize]).abs())) }
            Atom::Turn => if self.g.mover(self.s) == Some(self.side) { 0 } else { 1 },
        }
    }
}

/// A term: an atom, the sum or the product of two scalar terms, or a tuple of scalar terms compared in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term { Atom(Atom), Sum(Box<Term>, Box<Term>), Product(Box<Term>, Box<Term>), Tuple(Vec<Term>) }

impl Term {
    fn scalar(&self) -> bool { !matches!(self, Term::Tuple(_)) }
    pub fn text(&self) -> String {
        match self {
            Term::Atom(a) => a.name(), Term::Sum(a, b) => format!("[sum {} {}]", a.text(), b.text()), Term::Product(a, b) => format!("[product {} {}]", a.text(), b.text()),
            Term::Tuple(v) => format!("[tuple {}]", v.iter().map(|t| t.text()).collect::<Vec<_>>().join(" ")),
        }
    }
    /// The atoms the term reads.
    pub fn atoms(&self) -> Vec<Atom> {
        match self { Term::Atom(a) => vec![*a], Term::Sum(a, b) | Term::Product(a, b) => { let mut v = a.atoms(); v.extend(b.atoms()); v } Term::Tuple(t) => t.iter().flat_map(|x| x.atoms()).collect() }
    }
    pub fn parse(s: &str) -> Result<Term, String> {
        let toks = tokens(s);
        let (t, used) = parse_at(&toks, 0)?;
        if used != toks.len() { return Err(format!("text left over after the term: {s}")); }
        Ok(t)
    }
    fn eval(&self, r: &dyn Fn(&Atom) -> i64) -> Vec<i64> {
        match self {
            Term::Atom(a) => vec![r(a)],
            Term::Sum(a, b) => vec![a.eval(r)[0] + b.eval(r)[0]],
            Term::Product(a, b) => vec![a.eval(r)[0] * b.eval(r)[0]],
            Term::Tuple(v) => v.iter().map(|t| t.eval(r)[0]).collect(),
        }
    }
}

fn tokens(s: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    for c in s.chars() {
        match c { '[' | ']' => { if !cur.is_empty() { out.push(std::mem::take(&mut cur)); } out.push(c.to_string()); } c if c.is_whitespace() => { if !cur.is_empty() { out.push(std::mem::take(&mut cur)); } } c => cur.push(c) }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

fn parse_at(t: &[String], at: usize) -> Result<(Term, usize), String> {
    let first = t.get(at).ok_or("a term is missing")?;
    if first != "[" { return Ok((Term::Atom(Atom::parse(first)?), at + 1)); }
    let head = t.get(at + 1).ok_or("a term has no head")?.clone();
    let mut args = vec![];
    let mut i = at + 2;
    while t.get(i).ok_or("a bracket is not closed")? != "]" { let (a, n) = parse_at(t, i)?; args.push(a); i = n; }
    let term = match head.as_str() {
        "sum" | "product" if args.len() == 2 && args.iter().all(Term::scalar) => { let b = args.pop().unwrap(); let a = args.pop().unwrap(); if head == "sum" { Term::Sum(Box::new(a), Box::new(b)) } else { Term::Product(Box::new(a), Box::new(b)) } }
        "tuple" if !args.is_empty() && args.iter().all(Term::scalar) => Term::Tuple(args),
        h => return Err(format!("no term form {h} with {} scalar arguments", args.len())),
    };
    Ok((term, i + 1))
}

/// The tuple of numbers of a position: a pure function of it.
pub fn value<G: Game + Placed>(t: &Term, r: &Reader<G>) -> Vec<i64> { t.eval(&|a| r.atom(a)) }

/// The value of every atom of a list in one position (the miner's table row).
pub fn atom_values<G: Game + Placed>(atoms: &[Atom], r: &Reader<G>) -> Vec<i64> { atoms.iter().map(|a| r.atom(a)).collect() }

/// a strictly below b, in the order of the tuple (the first number that differs decides).
pub fn lower(a: &[i64], b: &[i64]) -> bool { a < b }

/// THE MEASURE AS AN IMPLEMENTATION OF LAYER 3'S ONE QUESTION. The term and the depth it is to fall within come from a statement of the long-term memory of one game; the geometry of the board is given (made once from layer 2's graph).
#[derive(Debug, Clone)]
pub struct Progress<'a> {
    pub term: Term,
    /// the term is to be lowered within this many of our moves (what the statement says; this crate does not search)
    pub within: u32,
    pub geo: &'a Geometry,
    /// the sample the statement was mined on, as its standing says
    pub standing: cs_rep::statement::Standing,
    pub id: String,
}

impl<'a> Progress<'a> {
    /// Read the statement `id` of the long-term memory of one game: it must say a term to be lowered (`Says::Lower`) and the term must parse.
    pub fn from_memory(memory: &Memory, id: &str, geo: &'a Geometry) -> Result<Progress<'a>, Unanswered> {
        let st = memory.statement(id).ok_or_else(|| Unanswered(format!("the long-term memory of {} holds no statement {id}", memory.key().name)))?;
        let Says::Lower { term, within } = &st.says else { return Err(Unanswered(format!("the statement {id} does not say a term to lower"))) };
        let term = Term::parse(term).map_err(|e| Unanswered(format!("the term of {id} does not parse: {e}")))?;
        Ok(Progress { term, within: *within, geo, standing: st.standing.clone(), id: id.to_string() })
    }
    /// A measure given directly (a term and the depth it falls within), for a caller that holds no memory.
    pub fn given(term: Term, within: u32, geo: &'a Geometry, standing: cs_rep::statement::Standing) -> Progress<'a> { Progress { term, within, geo, standing, id: String::new() } }
}

impl<'a> Progress<'a> {
    /// Whether every group the term reads is present in the position: the position is of the class the measure was mined for (the question of the means-ends crate asks it before it reads a leaf).
    pub fn in_class<G: Game + Placed>(&self, g: &G, s: &G::State, side: Role) -> bool {
        let r = Reader::new(g, self.geo, s, side);
        !self.term.atoms().iter().any(|a| match a { Atom::Region(x) | Atom::Span(x, _) | Atom::Edge(x) | Atom::Corner(x) | Atom::AxisEdge(x, _) => !r.present(*x), Atom::Dist(x, y) | Atom::Axis(x, y, _) => !r.present(*x) || !r.present(*y), Atom::Turn => false })
    }
}

impl<'a, G: Game + Placed> Assess<G> for Progress<'a> {
    fn name(&self) -> String { format!("progress {}", self.term.text()) }
    /// The tuple of the term at the position, for the side asked: always answered, as `value` always was (an atom of a group that is absent reads as the far value; `in_class` says whether that is the measure's own class).
    fn assess(&self, q: &Question<G>) -> Result<Answer, Unanswered> {
        let raw = value(&self.term, &Reader::new(q.game, self.geo, q.state, q.side));
        Ok(Answer { rank: raw.iter().map(|x| -(*x as f64)).collect(), raw: Some(raw), chance: None, sure: Some(self.standing.clone()), read: self.term.atoms().iter().map(|a| a.name()).collect() })
    }
}
