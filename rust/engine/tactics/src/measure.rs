//! PLAN 79.17 PART B, THE MEASURE OF PROGRESS AS A PIECE (SYSTEM_SPEC 7, "the small endings as the test of means and ends, and a measure of progress"; 8 the rule for choosing a move; piece: the opportunity policy, read by the wake). Game-free: it reads a game only through `Game` and `Placed`
//! and the board only through the base graph's `Static` (places, steps by direction, lines, the edge distance); no unit, no place, no kind is named. A unit is only a GROUP: the units of one kind on one side, the side being "the side the measure is for" or the other.
//!
//! WHAT A MEASURE IS (October's lesson, plans 73.5 and 73.5b, Lean `forced_by_measure`): a number or a tuple of numbers read from a position, such that from every position of a class some play of ours of at most d moves LOWERS it against every reply, until the game's end is found.
//! It is not a value: it never enters a line's value (SYSTEM_SPEC 7 and rule 7); it orders and it chooses among moves the calculation holds equal.
//!
//! INTERFACE (written before the bodies).
//!   `Geometry::new(&Static)`                     what the board gives once: the line families, a place's coordinate along each, its distance to the end of each family, the distance in steps between any two places, the distance to the places of fewest steps, the nearest edge.
//!   `Group { theirs, kind }`                     the units of one kind of the measure's side (`theirs` false) or of the other side (true).
//!   `Atom`                                       one reading of a position; `Atom::name`/`Atom::parse` (a name is its definition: `region(0,3)`, `dist(0,3,1,3)`, `span(1,3,0)`, `turn`).
//!                                                  region(g)      the count of places a unit of the group can walk to by its own steps over places the other side does not bear on (the unit lifted from the board, the others held still)
//!                                                  span(g,f)      the extent of that set along line family f (the board's lines)
//!                                                  edge(g)        a unit's distance to the nearest edge, in steps
//!                                                  corner(g)      a unit's distance in steps to the nearest place of fewest steps
//!                                                  dist(a,b)      the distance in steps between a unit of a and a unit of b (least over the pairs)
//!                                                  axis(a,b,f)    the distance along line family f between a unit of a and a unit of b (least over the pairs)
//!                                                  axisedge(g,f)  a unit's distance to the end of line family f (either way)
//!                                                  turn           0 where the measure's side is to move, 1 where the other is
//!   `Term`                                       an atom; the sum or the product of two scalar terms; a tuple of scalar terms compared in order. `Term::parse`/`Term::text` (`[sum A B]`, `[product A B]`, `[tuple A B C]`).
//!   `library(groups, &Geometry)`                 every atom the groups and the families give: the same code for every class.
//!   `value(&Term, &Reader) -> Vec<i64>`          the tuple of numbers of a position (a pure function of it); `Reader::new(g, geo, s, side)`.
//!   `lower(a, b)`                                a is strictly below b in the order of the tuple.
//!   `lowering(...) -> Option<Lowering>`          the question of October's criterion: is there a move of ours after which, within d of our moves and against every reply, the term is lower than `held` or the game's end is found? It is the search of toward.rs (`toward_with`) with the leaf
//!                                                  "the term is below the held value" read where it is our turn. The answer names the first move.
//!   `question(..) -> (Option<Found>, cut)`      PLAN 79.18 B: the question of the held goal. The fall is counted against the LOWEST value held: is there a line of at most d of our moves after which, against every reply, the term is below the lowest held (`Found::Fall`)? Where there is none, read to its end, and the value now is
//!                                                  above the lowest, is there one that brings it below the value now (`Found::Recovery`, which is not a fall)? A position met before in the game is a draw for the search and never the end of a line that counts.
//! Called by: the binding (`measure_atoms`, `measure_lowers`), the wake (`cs_decision::wake`, the held goal), the miner's driver in Python.
use crate::sheet::base_graph::Static;
use crate::sheet::{Kind, Place, PlaceSet};
use crate::toward::{toward_memo, Class, Verdict};
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use cs_operator::Budget;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

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

/// What October's question found: the first move (an index into the root's legal moves), how many of our moves the line takes, and whether the game's own end is what was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lowering { pub first: usize, pub moves: u32, pub game_end: bool, pub nodes: u64 }

/// Is there a move of ours at `s` (we are to move; `side` is the measure's side and the side to move) after which, within `d` of our moves and against every reply, the term is below `held` where it is our turn, or the game's end is ours? The search of toward.rs, deepened ply by ply to 2d - 1 plies and
/// one more for the reply; a leaf where it is our turn answers "win" when the term there is below `held`. None: no such line in the budget (a cut is a gap, never a result: `Err` of the budget is None as well, `cut` tells which).
pub fn lowering<G: Game + Placed>(g: &G, geo: &Geometry, s: &G::State, side: Role, term: &Term, held: &[i64], d: u32, history: &HashSet<u64>, b: &mut Budget) -> (Option<Lowering>, bool) {
    if g.mover(s) != Some(side) || d == 0 { return (None, false); }
    let oracle = |gm: &G, st: &G::State| -> Option<Verdict> {
        if gm.mover(st) != Some(side) { return None; }
        let r = Reader::new(gm, geo, st, side);
        if term.atoms().iter().any(|a| match a { Atom::Region(x) | Atom::Span(x, _) | Atom::Edge(x) | Atom::Corner(x) | Atom::AxisEdge(x, _) => !r.present(*x), Atom::Dist(x, y) | Atom::Axis(x, y, _) => !r.present(*x) || !r.present(*y), Atom::Turn => false }) { return None; }
        if lower(&value(term, &r), held) { Some(Verdict::Win) } else { None }
    };
    let t = toward_memo(g, s, side, &oracle, history, 2 * d, None, b);          // PLAN 79.18: the same answers as `toward_with`, a position reached by another order of moves read once
    let best = t.of_class(Class::Win).into_iter().next();
    let found = best.map(|i| Lowering { first: i, moves: (t.depth + 1) / 2, game_end: t.root.iter().find(|r| r.0 == i).is_some_and(|r| crate::toward::is_game_win(r.2)), nodes: t.nodes });
    (found, t.cut)
}

/// PLAN 79.18 B (SYSTEM_SPEC 7, the measure of progress; piece: the measure): what the question of the held goal found. A FALL goes below the lowest value held and is the subgoal reached; a RECOVERY goes only below the value now (the term had risen above the lowest) and is not a subgoal reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Found { Fall(Lowering), Recovery(Lowering) }

impl Found {
    pub fn lowering(&self) -> &Lowering { match self { Found::Fall(l) | Found::Recovery(l) => l }
    }
    pub fn is_fall(&self) -> bool { matches!(self, Found::Fall(_)) }
}

/// PLAN 79.18 B: THE QUESTION OF THE HELD GOAL, counted against what was held. `lowest` is the lowest value held so far and `now` the value in the position (never below `lowest`: a value below it is registered as an event before the question is asked). The first reading asks for a line of at most `d` of our moves after which
/// the term is below `lowest` against every reply; only where that was read to its end and found nothing, and `now` is above `lowest`, a second asks for a line below `now`. A cut (the budget) is a gap: it is returned as `cut` and nothing is said of what was not read. The history keys are the game's record: a position met before is a draw for the search, so it is never the end of a line that counts.
#[allow(clippy::too_many_arguments)]
pub fn question<G: Game + Placed>(g: &G, geo: &Geometry, s: &G::State, side: Role, term: &Term, lowest: &[i64], now: &[i64], d: u32, history: &HashSet<u64>, b: &mut Budget) -> (Option<Found>, bool) {
    let (fall, cut) = lowering(g, geo, s, side, term, lowest, d, history, b);
    if let Some(l) = fall { return (Some(Found::Fall(l)), cut); }
    if cut || !lower(lowest, now) { return (None, cut); }
    let (rec, cut) = lowering(g, geo, s, side, term, now, d, history, b);
    (rec.map(Found::Recovery), cut)
}
