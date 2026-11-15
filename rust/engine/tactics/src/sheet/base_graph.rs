//! PLAN 78.52 THE GRAPH, LAYER 1 (SYSTEM_SPEC section 11): the concepts and basic facts that the RULES give, as a graph, game-free. Every node is derived; none is supplied. The
//! module names no object of any game: not a line family, not a unit, not a place. Every word it holds of a game (a place's text, a direction, a kind, a side) comes from the rules
//! or from the game's own names, at run time.
//!
//! WHERE THE STATIC PART COMES FROM. The places and the steps by direction are the rules' own (`Placed::steps`, read from the description's static relations through the bound names
//! of its graph vocabulary: no state is read). `Static::new(&Steps, &Names)` is built ONCE per game; `Static::position` builds the per-position part from a state through `Placed`.
//!
//! NODES, with their ids. An id is its DEFINITION: the kind of node and the ids it is defined from, never a counter, so it is the same in every position and every run.
//!   place        `place:<place text>`                                  one per place of the rules
//!   direction    `direction:<word>`                                    one per step relation (the rules' own word)
//!   line         `line:<direction word>@<id of its first place>`       a maximal chain of at least two places under one direction and its reverse, named in the direction of
//!                                                                      the pair that sorts first and begun at the place that has no place before it
//!   edge         `edge:<direction word>`                               the set of places with no next place under that direction
//!   kind, side   `kind:<word>`, `side:<role word>`                     one per kind of unit and one per side of the rules
//!   unit         `unit:<side>:<kind>:<place text>`                     one per unit standing, per position
//!   bear         `bear(<unit id>,<place id>)`                          a RELATION node: the unit bears on the place (every bearing `Placed::bearing_from` reports, once)
//!   may_move     `may_move(<unit id>,<place id>)`                      a RELATION node: the unit may move to the place (`Placed::reach_from`)
//!   shut         `shut(<unit id>,<place id>)`                          a RELATION node, for the side to move: the unit runs along a line and the line is shut before the place, by units
//!                                                                      standing between (`Placed::why_not` says LineShut)
//! PLAIN LINKS (a kind, two nodes, optionally a node it goes by and a number):
//!   on_line (place -> line, the index along the line), next (place -> place, by a direction), on_edge (place -> edge), edge_distance (place -> edge, the steps until the chain
//!   ends under that direction), line_along (line -> direction), reverse (direction -> direction), edge_of (edge -> direction), unit_on (unit -> place), unit_kind, unit_side.
//! RELATION LINKS (the same for bear, may_move and shut): by (relation -> its unit), at_place (-> the place), at_unit (-> the unit standing there), along (-> the line it runs along,
//!   where from and to lie on one line), between (-> each unit standing strictly between on that line). A bearing that follows no line (a leap) has no `along` and no `between`.
use super::{Kind, Names, Place, PlaceSet, Side};
use cs_game::placed::{NoStep, Placed, Steps};
use cs_game::{Game, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeType { Place, Direction, Line, Edge, Kind, Side, Unit, Bear, MayMove, Shut }

impl NodeType {
    pub const ALL: [NodeType; 10] = [NodeType::Place, NodeType::Direction, NodeType::Line, NodeType::Edge, NodeType::Kind, NodeType::Side, NodeType::Unit, NodeType::Bear, NodeType::MayMove, NodeType::Shut];
    pub fn word(self) -> &'static str {
        match self { NodeType::Place => "place", NodeType::Direction => "direction", NodeType::Line => "line", NodeType::Edge => "edge", NodeType::Kind => "kind", NodeType::Side => "side",
                     NodeType::Unit => "unit", NodeType::Bear => "bear", NodeType::MayMove => "may_move", NodeType::Shut => "shut" }
    }
    pub fn is_relation(self) -> bool { matches!(self, NodeType::Bear | NodeType::MayMove | NodeType::Shut) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LinkType { OnLine, Next, OnEdge, EdgeDistance, LineAlong, Reverse, EdgeOf, UnitOn, UnitKind, UnitSide, By, AtPlace, AtUnit, Along, Between }

impl LinkType {
    pub const ALL: [LinkType; 15] = [LinkType::OnLine, LinkType::Next, LinkType::OnEdge, LinkType::EdgeDistance, LinkType::LineAlong, LinkType::Reverse, LinkType::EdgeOf, LinkType::UnitOn,
                                     LinkType::UnitKind, LinkType::UnitSide, LinkType::By, LinkType::AtPlace, LinkType::AtUnit, LinkType::Along, LinkType::Between];
    pub fn word(self) -> &'static str {
        match self { LinkType::OnLine => "on_line", LinkType::Next => "next", LinkType::OnEdge => "on_edge", LinkType::EdgeDistance => "edge_distance", LinkType::LineAlong => "line_along",
                     LinkType::Reverse => "reverse", LinkType::EdgeOf => "edge_of", LinkType::UnitOn => "unit_on", LinkType::UnitKind => "unit_kind", LinkType::UnitSide => "unit_side",
                     LinkType::By => "by", LinkType::AtPlace => "at_place", LinkType::AtUnit => "at_unit", LinkType::Along => "along", LinkType::Between => "between" }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRec { pub kind: NodeType, pub id: String }

/// A link between two nodes by index into the graph's node list (the static nodes first, then the position's); `via` is the node it goes by (a direction), `value` a number; -1 when none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkRec { pub kind: LinkType, pub a: u32, pub b: u32, pub via: i32, pub value: i32 }

/// A line: the direction it is named in and its places in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line { pub direction: usize, pub places: Vec<Place> }

/// THE STATIC PART: what the rules say of the board and the game says of its kinds and sides; built once per game.
#[derive(Debug, Clone)]
pub struct Static {
    pub nodes: Vec<NodeRec>,
    pub links: Vec<LinkRec>,
    pub names: Names,
    pub directions: Vec<String>,
    pub lines: Vec<Line>,
    /// per place: (line, index along it)
    pub place_lines: Vec<Vec<(u32, u16)>>,
    /// [direction][place]: the steps until the chain ends under that direction (0: the place is on the edge)
    pub edge_distance: Vec<Vec<u16>>,
    pub next: Vec<Vec<Option<Place>>>,
    pub reverse: Vec<Option<usize>>,
    pub place_node: Vec<u32>,
    pub direction_node: Vec<u32>,
    pub line_node: Vec<u32>,
    pub edge_node: Vec<u32>,
    pub kind_node: Vec<u32>,
    pub side_node: Vec<u32>,
}

/// One unit of a position: the owner as the game names it, the side as the reading names it (the side to move is `Us`), the kind, the place and its node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitRec { pub owner: Role, pub side: Side, pub kind: Kind, pub at: Place, pub node: u32 }

/// A relation node of a position: the unit (an index into `units`), the place, the lines it runs along (indices into the static lines), the units standing between (indices into `units`),
/// and its node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelRec { pub unit: usize, pub place: Place, pub along: Vec<u32>, pub between: Vec<usize>, pub node: u32 }

/// THE PER-POSITION PART.
#[derive(Debug, Clone)]
pub struct PositionPart {
    pub mover: Option<Role>,
    pub nodes: Vec<NodeRec>,
    pub links: Vec<LinkRec>,
    pub units: Vec<UnitRec>,
    pub bears: Vec<RelRec>,
    pub moves: Vec<RelRec>,
    pub shuts: Vec<RelRec>,
    /// place -> index in `units` + 1 (0: nothing stands there)
    pub occupant: Vec<u16>,
}

impl Static {
    /// Build the static part from the rules' steps and the game's names. Errors: a direction that is not a chain (two places after one), a closed chain (not supported in layer 1).
    pub fn new(steps: &Steps, names: &Names) -> Result<Static, String> {
        let (np, nd) = (names.places.len(), steps.directions.len());
        let mut next: Vec<Vec<Option<Place>>> = vec![vec![None; np]; nd];
        let mut prev: Vec<Vec<Option<Place>>> = vec![vec![None; np]; nd];
        for &(d, f, t) in &steps.rows {
            let d = d as usize;
            if next[d][f as usize].replace(t).is_some_and(|x| x != t) { return Err(format!("direction {} gives two places after one: it is not a chain", steps.directions[d])); }
            if prev[d][t as usize].replace(f).is_some_and(|x| x != f) { return Err(format!("direction {} gives two places before one: it is not a chain", steps.directions[d])); }
        }
        let reverse: Vec<Option<usize>> = (0..nd).map(|d| (0..nd).find(|&e| next[e] == prev[d] && prev[d].iter().any(|x| x.is_some()))).collect();
        // lines: for the first direction of each pair (or a direction with no reverse), the maximal chains of at least two places
        let mut lines: Vec<Line> = vec![];
        for d in 0..nd {
            if reverse[d].is_some_and(|e| e < d) { continue; }
            let mut seen = vec![false; np];
            for p in 0..np {
                if next[d][p].is_none() || prev[d][p].is_some() { continue; }
                let mut places = vec![p as Place];
                seen[p] = true;
                let mut at = p;
                while let Some(q) = next[d][at] { places.push(q); seen[q as usize] = true; at = q as usize; }
                lines.push(Line { direction: d, places });
            }
            if (0..np).any(|p| next[d][p].is_some() && !seen[p]) { return Err(format!("direction {} has a closed chain: layer 1 does not read it", steps.directions[d])); }
        }
        let mut place_lines: Vec<Vec<(u32, u16)>> = vec![vec![]; np];
        for (li, l) in lines.iter().enumerate() { for (i, p) in l.places.iter().enumerate() { place_lines[*p as usize].push((li as u32, i as u16)); } }
        let edge_distance: Vec<Vec<u16>> = (0..nd).map(|d| (0..np).map(|p| { let (mut n, mut at) = (0u16, p); while let Some(q) = next[d][at] { n += 1; at = q as usize; if n as usize > np { break; } } n }).collect()).collect();

        let mut nodes: Vec<NodeRec> = vec![];
        let mut push = |kind: NodeType, id: String| -> u32 { nodes.push(NodeRec { kind, id }); nodes.len() as u32 - 1 };
        let place_node: Vec<u32> = names.places.iter().map(|p| push(NodeType::Place, format!("place:{p}"))).collect();
        let direction_node: Vec<u32> = steps.directions.iter().map(|d| push(NodeType::Direction, format!("direction:{d}"))).collect();
        let line_node: Vec<u32> = lines.iter().map(|l| push(NodeType::Line, format!("line:{}@place:{}", steps.directions[l.direction], names.places[l.places[0] as usize]))).collect();
        let edge_node: Vec<u32> = steps.directions.iter().map(|d| push(NodeType::Edge, format!("edge:{d}"))).collect();
        let kind_node: Vec<u32> = names.kinds.iter().map(|k| push(NodeType::Kind, format!("kind:{k}"))).collect();
        let side_node: Vec<u32> = names.roles.iter().map(|r| push(NodeType::Side, format!("side:{r}"))).collect();

        let mut links: Vec<LinkRec> = vec![];
        let mut link = |kind: LinkType, a: u32, b: u32, via: i32, value: i32| links.push(LinkRec { kind, a, b, via, value });
        for (li, l) in lines.iter().enumerate() {
            link(LinkType::LineAlong, line_node[li], direction_node[l.direction], -1, -1);
            for (i, p) in l.places.iter().enumerate() { link(LinkType::OnLine, place_node[*p as usize], line_node[li], -1, i as i32); }
        }
        for d in 0..nd {
            if let Some(e) = reverse[d] { link(LinkType::Reverse, direction_node[d], direction_node[e], -1, -1); }
            link(LinkType::EdgeOf, edge_node[d], direction_node[d], -1, -1);
            for p in 0..np {
                if let Some(q) = next[d][p] { link(LinkType::Next, place_node[p], place_node[q as usize], direction_node[d] as i32, -1); }
                if edge_distance[d][p] == 0 { link(LinkType::OnEdge, place_node[p], edge_node[d], -1, -1); }
                link(LinkType::EdgeDistance, place_node[p], edge_node[d], -1, edge_distance[d][p] as i32);
            }
        }
        Ok(Static { nodes, links, names: names.clone(), directions: steps.directions.clone(), lines, place_lines, edge_distance, next, reverse, place_node, direction_node, line_node, edge_node, kind_node, side_node })
    }

    /// The lines on which both places lie.
    pub fn common_lines(&self, a: Place, b: Place) -> Vec<u32> {
        self.place_lines[a as usize].iter().filter(|(l, _)| self.place_lines[b as usize].iter().any(|(m, _)| m == l)).map(|(l, _)| *l).collect()
    }

    /// The least of a place's edge distances over every direction.
    pub fn nearest_edge(&self, p: Place) -> u16 { self.edge_distance.iter().map(|d| d[p as usize]).min().unwrap_or(0) }

    fn index_on(&self, line: u32, p: Place) -> i32 { self.place_lines[p as usize].iter().find(|(l, _)| *l == line).map_or(-1, |x| x.1 as i32) }

    /// The places strictly between two places on a line, in order of the line.
    pub fn strictly_between(&self, line: u32, a: Place, b: Place) -> Vec<Place> {
        let (i, j) = (self.index_on(line, a), self.index_on(line, b));
        if i < 0 || j < 0 { return vec![]; }
        let (lo, hi) = (i.min(j) as usize, i.max(j) as usize);
        self.lines[line as usize].places[lo + 1..hi].to_vec()
    }

    /// Build the part of one position through `Placed`.
    pub fn position<G: Game + Placed>(&self, g: &G, s: &G::State) -> PositionPart {
        let np = self.names.places.len();
        let ns = self.nodes.len() as u32;
        let mover = g.mover(s);
        let mut cells = g.cells(s);
        cells.sort_by_key(|(o, k, p)| (*o, *k, *p));
        let mut nodes: Vec<NodeRec> = vec![];
        let mut links: Vec<LinkRec> = vec![];
        let mut units: Vec<UnitRec> = Vec::with_capacity(cells.len());
        let mut occupant = vec![0u16; np];
        for (owner, kind, at) in cells {
            let side = match mover { Some(m) if m != owner => Side::Them, _ => Side::Us };
            let id = format!("unit:{}:{}:{}", self.names.roles[owner], self.names.kinds[kind as usize], self.names.places[at as usize]);
            let node = ns + nodes.len() as u32;
            nodes.push(NodeRec { kind: NodeType::Unit, id });
            links.push(LinkRec { kind: LinkType::UnitOn, a: node, b: self.place_node[at as usize], via: -1, value: -1 });
            links.push(LinkRec { kind: LinkType::UnitKind, a: node, b: self.kind_node[kind as usize], via: -1, value: -1 });
            links.push(LinkRec { kind: LinkType::UnitSide, a: node, b: self.side_node[owner], via: -1, value: -1 });
            occupant[at as usize] = units.len() as u16 + 1;
            units.push(UnitRec { owner, side, kind, at, node });
        }
        let mut part = PositionPart { mover, nodes, links, units, bears: vec![], moves: vec![], shuts: vec![], occupant };
        let nu = part.units.len();
        for ui in 0..nu {
            let u = part.units[ui];
            for p in g.bearing_from(s, u.owner, u.kind, u.at) { let r = self.relation(&mut part, NodeType::Bear, ui, p, ns); part.bears.push(r); }
            for p in g.reach_from(s, u.owner, u.kind, u.at) { let r = self.relation(&mut part, NodeType::MayMove, ui, p, ns); part.moves.push(r); }
        }
        if let Some(m) = mover { self.shut_lines(g, s, m, &mut part, ns); }
        part
    }

    /// One relation node with its links.
    fn relation(&self, part: &mut PositionPart, kind: NodeType, ui: usize, to: Place, ns: u32) -> RelRec {
        let u = part.units[ui];
        let id = format!("{}({},place:{})", kind.word(), part.nodes[(u.node - ns) as usize].id, self.names.places[to as usize]);
        let node = ns + part.nodes.len() as u32;
        part.nodes.push(NodeRec { kind, id });
        let mut push = |kind: LinkType, b: u32| part.links.push(LinkRec { kind, a: node, b, via: -1, value: -1 });
        push(LinkType::By, u.node);
        push(LinkType::AtPlace, self.place_node[to as usize]);
        let target = part.occupant[to as usize];
        if target != 0 { push(LinkType::AtUnit, part.units[target as usize - 1].node); }
        let along = self.common_lines(u.at, to);
        let mut between = vec![];
        for &l in &along {
            push(LinkType::Along, self.line_node[l as usize]);
            for q in self.strictly_between(l, u.at, to) {
                let o = part.occupant[q as usize];
                if o != 0 { between.push(o as usize - 1); push(LinkType::Between, part.units[o as usize - 1].node); }
            }
        }
        RelRec { unit: ui, place: to, along, between, node }
    }

    /// The shut lines of the side to move: for each of its units and each line it bears along, the places beyond the farthest bearing on either side of the unit, where the game says the
    /// line is shut (`why_not` = LineShut), with the units standing between.
    fn shut_lines<G: Game + Placed>(&self, g: &G, s: &G::State, mover: Role, part: &mut PositionPart, ns: u32) {
        for ui in 0..part.units.len() {
            let u = part.units[ui];
            if u.owner != mover { continue; }
            for &(l, i) in &self.place_lines[u.at as usize] {
                let ends: Vec<i32> = part.bears.iter().filter(|r| r.unit == ui && r.along.contains(&l)).map(|r| self.index_on(l, r.place) - i as i32).collect();
                for sign in [1i32, -1] {
                    let Some(far) = ends.iter().filter(|d| d.signum() == sign).map(|d| d.abs()).max() else { continue };
                    let line = &self.lines[l as usize];
                    let mut k = i as i32 + sign * (far + 1);
                    while k >= 0 && (k as usize) < line.places.len() {
                        let q = line.places[k as usize];
                        if let (NoStep::LineShut, _) = g.why_not(s, u.at, q) {
                            let r = self.relation(part, NodeType::Shut, ui, q, ns);
                            part.shuts.push(r);
                        }
                        k += sign;
                    }
                }
            }
        }
    }
}

impl Static {
    /// PLAN 78.53, A DIAGNOSTIC AND NOT BUILT INTO THE GRAPH: the shut lines of the units that do NOT move, which `why_not` cannot say (it answers for the side to move only), tried through
    /// `Placed::without`. It fails where the game refuses the board with the blocker off (a side that is not to move left exposed to a line), which is why the position part keeps the
    /// shut lines of the side to move only (`first_blocker_check` measures the gap against `why_not`). Game-free, through `Placed::without`: for a unit and a line it
    /// bears along, the unit standing on its farthest bearing place on one side is the blocker; the places the unit would bear on beyond the blocker with the blocker taken off the board (and
    /// not held by the unit's own side) are shut by it, the blocker the one unit between. So these relation nodes are the first-blocker form: the nodes of the mover with exactly one unit between
    /// are the same set (tested on the mover against `why_not`).
    pub fn shut_lines_other<G: Game + Placed>(&self, g: &G, s: &G::State, mover: Role, part: &mut PositionPart, ns: u32) {
        for ui in 0..part.units.len() {
            let u = part.units[ui];
            if u.owner == mover { continue; }
            self.first_blocker_shuts(g, s, ui, part, ns);
        }
    }

    /// The first-blocker shut relations of one unit, added to the part.
    pub fn first_blocker_shuts<G: Game + Placed>(&self, g: &G, s: &G::State, ui: usize, part: &mut PositionPart, ns: u32) {
        let u = part.units[ui];
        for &(l, i) in &self.place_lines[u.at as usize] {
            let ends: Vec<i32> = part.bears.iter().filter(|r| r.unit == ui && r.along.contains(&l)).map(|r| self.index_on(l, r.place) - i as i32).collect();
            for sign in [1i32, -1] {
                let Some(far) = ends.iter().filter(|d| d.signum() == sign).map(|d| d.abs()).max() else { continue };
                let line = &self.lines[l as usize];
                let k = i as i32 + sign * far;
                if k < 0 || k as usize >= line.places.len() { continue; }
                let blocker = line.places[k as usize];
                if part.occupant[blocker as usize] == 0 { continue; }
                let Some(s2) = g.without(s, blocker) else { continue };
                for q in g.bearing_from(&s2, u.owner, u.kind, u.at) {
                    let j = self.index_on(l, q);
                    if j < 0 || (j - k) * sign <= 0 { continue; }
                    let held = part.occupant[q as usize];
                    if held != 0 && part.units[held as usize - 1].owner == u.owner { continue; }
                    let r = self.relation(part, NodeType::Shut, ui, q, ns);
                    part.shuts.push(r);
                }
            }
        }
    }
}

impl PositionPart {
    /// The longest run of places on a line with no unit standing on them.
    pub fn open_length(&self, st: &Static, line: usize) -> usize {
        let (mut best, mut run) = (0, 0);
        for p in &st.lines[line].places { if self.occupant[*p as usize] == 0 { run += 1; best = best.max(run); } else { run = 0; } }
        best
    }

    /// Places borne on by some unit of `side`.
    pub fn struck_by(&self, np: usize, side: Side) -> PlaceSet {
        let mut out = PlaceSet::new(np);
        for r in &self.bears { if self.units[r.unit].side == side { out.insert(r.place); } }
        out
    }

    /// A text dump of the position part, one node per line with its links (the ids are the definitions).
    pub fn dump(&self, st: &Static) -> String {
        let ns = st.nodes.len() as u32;
        let id = |n: u32| -> &str { if n < ns { &st.nodes[n as usize].id } else { &self.nodes[(n - ns) as usize].id } };
        let mut out = String::new();
        for (i, n) in self.nodes.iter().enumerate() {
            let me = ns + i as u32;
            let outs: Vec<String> = self.links.iter().filter(|l| l.a == me).map(|l| format!("{}->{}", l.kind.word(), id(l.b))).collect();
            out.push_str(&format!("{} [{}]  {}\n", n.id, n.kind.word(), outs.join(" ")));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cs_game::placed::read_steps;

    /// A board of four columns and three rows with four headings (no diagonal), written as a description: nothing like the board of the game the engine plays.
    const SMALL: &str = "
(sort col x y z w) (sort row 1 2 3) (sort heading up down left right)
(signature col col) (signature row row) (signature col_succ col col) (signature row_succ row row) (signature step heading col row col row)
(col x) (col y) (col z) (col w) (row 1) (row 2) (row 3)
(col_succ x y) (col_succ y z) (col_succ z w) (row_succ 1 2) (row_succ 2 3)
(<= (step up ?c ?r1 ?c ?r2) (col ?c) (row_succ ?r1 ?r2))
(<= (step down ?c ?r1 ?c ?r2) (col ?c) (row_succ ?r2 ?r1))
(<= (step right ?c1 ?r ?c2 ?r) (row ?r) (col_succ ?c1 ?c2))
(<= (step left ?c1 ?r ?c2 ?r) (row ?r) (col_succ ?c2 ?c1))
";

    fn small() -> Static {
        let names: Vec<String> = ["x", "y", "z", "w"].iter().flat_map(|c| ["1", "2", "3"].iter().map(move |r| format!("{c}{r}"))).collect();
        let objects = vec![("cell".to_string(), vec!["col".to_string(), "row".to_string()])];
        let steps = read_steps(SMALL, &["step".to_string()], &objects, &names).unwrap();
        Static::new(&steps, &Names::new(names, vec!["piece".into()], vec!["a".into(), "b".into()])).unwrap()
    }

    #[test]
    fn a_four_by_three_board_has_its_lines_and_edges_from_the_steps_alone() {
        let st = small();
        assert_eq!(st.directions, vec!["down", "left", "right", "up"]);
        assert_eq!(st.lines.len(), 3 + 4);                                   // three rows of four, four columns of three
        let mut lens: Vec<usize> = st.lines.iter().map(|l| l.places.len()).collect();
        lens.sort();
        assert_eq!(lens, vec![3, 3, 3, 3, 4, 4, 4]);
        let on_edge = |d: &str| { let di = st.directions.iter().position(|x| x == d).unwrap(); (0..12).filter(|&p| st.edge_distance[di][p] == 0).count() };
        assert_eq!((on_edge("up"), on_edge("down"), on_edge("left"), on_edge("right")), (4, 4, 3, 3));
        let (right, left) = (st.directions.iter().position(|x| x == "right").unwrap(), st.directions.iter().position(|x| x == "left").unwrap());
        let y2 = st.names.place("y2").unwrap() as usize;
        assert_eq!((st.edge_distance[right][y2], st.edge_distance[left][y2]), (2, 1));
        assert_eq!(st.reverse.iter().filter(|r| r.is_some()).count(), 4);
    }

    #[test]
    fn the_static_part_is_the_same_on_two_builds_and_ids_are_their_definitions() {
        let (a, b) = (small(), small());
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.links, b.links);
        assert!(a.nodes.iter().any(|n| n.id == "place:y2") && a.nodes.iter().any(|n| n.id == "edge:up") && a.nodes.iter().any(|n| n.id == "line:left@place:w1"));
        let mut ids: Vec<&String> = a.nodes.iter().map(|n| &n.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), a.nodes.len());
    }

    #[test]
    fn a_step_relation_that_is_not_a_chain_is_refused() {
        let names = vec!["x1".to_string(), "y1".to_string(), "z1".to_string()];
        let steps = cs_game::placed::Steps { directions: vec!["d".into()], rows: vec![(0, 0, 1), (0, 0, 2)], ..Default::default() };
        assert!(Static::new(&steps, &Names::new(names, vec![], vec![])).is_err());
    }
}
