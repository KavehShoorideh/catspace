//! PLAN 79.21 STAGE B, THE GRAPH IN LAYER 1 (SYSTEM_SPEC 11 "one graph is the representation; the rules are the next layer", "the layers", 13 item 41; ledger row 36; piece: representation). One graph holds the BOARD (its places, its lines, which place is next to which,
//! how far each place is from an edge, the two classes of places), the UNITS standing on it and the side the state marks as to act. It is built from a board read once and a set of facts (or the same units as numbers), and from nothing else: no rule of play is called. HOW A UNIT STEPS is a rule, so the
//! relations that follow from it are made, named and owned by layer 2 (cs_game::relations) and WRITTEN ONTO THE GRAPH through `Board::write` (the door of change), which takes the relation as a value (`Rel`); layer 1 knows neither their names nor their meaning (plan 79.34).
//!
//! INTERFACES (written before the bodies; who calls them is said at each).
//!   `Board::new(&Steps, &Names) -> Result<Board, String>`                     the static part, once per game: the places, the directions, the lines (a maximal chain of two places or more under a direction and its reverse), the edges. Called by the layer that reads the description (layer 2 hands the steps over) and by `Board::from_json`.
//!   `Board::with_decl(self, FactDecl) -> Board`                                says where a game's facts hold its units and its tempo (`FactDecl { units: UnitDecl, tempo: Option<String> }`); data supplied by the rules' binding, never a word written in a layer.
//!   `Board::to_json / from_json`                                               the board written and read back identical (the fixtures hold boards in this form).
//!   `Board::position_from_units(&self, mover, cells) -> PositionPart`          the graph of a position from the units as numbers (owner, kind, place) and the tempo: the entry the engine uses.
//!   `Board::position_from_facts(&self, &[Fact]) -> Result<PositionPart, String>`  the same from a set of facts (needs the `FactDecl`): units and tempo are read off the facts by the declaration; every other fact is not the graph's (layer 2 tags it).
//!   `Board::write(&self, &mut PositionPart, Rel, unit, place)`                 the door of change: layer 2 writes one relation of a unit onto the graph; the node, its id and its links are made here.
//!   `Board::classes(&self) -> Option<Classes>`                                 the two alternating classes of places, from the links alone (plan 79.34).
//!   `PositionPart`                                                             the per-position part: units, relation nodes, links, who stands where.
//! Node ids are DEFINITIONS (the kind of node and the ids it is defined from), never a counter, so they are the same in every position and every run.
//!   place `place:<text>`; direction `direction:<word>`; line `line:<direction word>@<id of its first place>`; edge `edge:<direction word>`; kind `kind:<word>`, side `side:<role word>`;
//!   unit `unit:<side>:<kind>:<place text>`; relation nodes `<word of the relation>(<unit id>,<place id>)`, the word being the layer above's.
//! PLAIN LINKS: on_line, next, on_edge, edge_distance, line_along, reverse, edge_of, unit_on, unit_kind, unit_side. RELATION LINKS (the same for every relation): by, at_place, at_unit, along, between.
use crate::fact::{Fact, UnitDecl};
use cs_base::structure::ChainError;
use crate::names::{Kind, Names, Place, Role, Side};
use crate::steps::Steps;
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeType { Place, Direction, Line, Edge, Kind, Side, Unit, Relation(Rel) }

/// How many kinds of relation a layer above may write onto one position's graph.
pub const REL_SLOTS: usize = 5;

/// A RELATION AS THE LAYER ABOVE NAMES IT (plan 79.34): the slot it is kept in and the word the ids of its nodes begin with. A value, made as a constant by the layer above: layer 1 never compares its text and never looks it up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rel { pub slot: u8, pub word: &'static str }

impl Rel {
    pub const fn new(slot: u8, word: &'static str) -> Rel { Rel { slot, word } }
}

impl NodeType {
    pub fn word(self) -> &'static str {
        match self { NodeType::Place => "place", NodeType::Direction => "direction", NodeType::Line => "line", NodeType::Edge => "edge", NodeType::Kind => "kind", NodeType::Side => "side",
                     NodeType::Unit => "unit", NodeType::Relation(r) => r.word }
    }
    pub fn is_relation(self) -> bool { matches!(self, NodeType::Relation(_)) }
}

/// THE TWO CLASSES OF PLACES (plan 79.34), from the board's links alone: the class (0 or 1) of each place and, for each direction, whether a step along it changes the class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classes { pub class: Vec<u8>, pub flips: Vec<bool> }

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

/// Where a game's facts hold its units and its tempo: the unit's declaration (the relation, the argument of the owner, the argument of the kind; the other arguments are its place) and the relation (of one argument, the owner) that says whose turn it is. Supplied by the binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactDecl { pub units: UnitDecl, pub tempo: Option<String> }

/// THE BOARD: what the description says of the places and the steps, and the game says of its kinds and sides; built once per game.
#[derive(Debug, Clone)]
pub struct Board {
    pub nodes: Vec<NodeRec>,
    pub links: Vec<LinkRec>,
    pub names: Names,
    pub steps: Steps,
    pub decl: Option<FactDecl>,
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

/// A relation node of a position: the unit (an index into `units`), the place, the lines it runs along (indices into the static lines), the units standing between (indices into `units`), and its node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelRec { pub unit: usize, pub place: Place, pub along: Vec<u32>, pub between: Vec<usize>, pub node: u32 }

/// THE PER-POSITION PART.
#[derive(Debug, Clone)]
pub struct PositionPart {
    pub mover: Option<Role>,
    pub nodes: Vec<NodeRec>,
    pub links: Vec<LinkRec>,
    pub units: Vec<UnitRec>,
    /// the relations a layer above wrote, one list a slot (`Rel::slot`), each in the order they were written
    pub rels: [Vec<RelRec>; REL_SLOTS],
    /// place -> index in `units` + 1 (0: nothing stands there)
    pub occupant: Vec<u16>,
}

impl Board {
    /// Build the static part from the steps and the game's names. Errors: a direction that is not a chain (two places after one), a closed chain (not supported in layer 1).
    pub fn new(steps: &Steps, names: &Names) -> Result<Board, String> {
        let (np, nd) = (names.places.len(), steps.directions.len());
        let mut next: Vec<Vec<Option<Place>>> = vec![vec![None; np]; nd];
        let mut prev: Vec<Vec<Option<Place>>> = vec![vec![None; np]; nd];
        for &(d, f, t) in &steps.rows {
            let d = d as usize;
            if next[d][f as usize].replace(t).is_some_and(|x| x != t) { return Err(format!("direction {} gives two places after one: it is not a chain", steps.directions[d])); }
            if prev[d][t as usize].replace(f).is_some_and(|x| x != f) { return Err(format!("direction {} gives two places before one: it is not a chain", steps.directions[d])); }
        }
        // plan 79.30: the chains of the successor maps (the reverse of each direction, the maximal lines, the steps to the end of each) are cs_base's, the pure core of this graph; what stays here is the reading of the rows, the names and the node ids.
        let next_by_number: Vec<Vec<Option<u32>>> = next.iter().map(|v| v.iter().map(|o| o.map(|p| p as u32)).collect()).collect();
        let found = cs_base::structure::chains(&next_by_number).map_err(|e| match e {
            ChainError::Closed(d) => format!("direction {} has a closed chain: layer 1 does not read it", steps.directions[d]),
            ChainError::TwoBefore(d) => format!("direction {} gives two places before one: it is not a chain", steps.directions[d]),
            ChainError::TwoAfter(d) => format!("direction {} gives two places after one: it is not a chain", steps.directions[d]),
        })?;
        let reverse: Vec<Option<usize>> = found.reverse;
        let lines: Vec<Line> = found.lines.into_iter().map(|c| Line { direction: c.label, places: c.nodes.into_iter().map(|p| p as Place).collect() }).collect();
        let mut place_lines: Vec<Vec<(u32, u16)>> = vec![vec![]; np];
        for (li, l) in lines.iter().enumerate() { for (i, p) in l.places.iter().enumerate() { place_lines[*p as usize].push((li as u32, i as u16)); } }
        let edge_distance: Vec<Vec<u16>> = found.distance_to_end;

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
        Ok(Board { nodes, links, names: names.clone(), steps: steps.clone(), decl: None, directions: steps.directions.clone(), lines, place_lines, edge_distance, next, reverse, place_node, direction_node, line_node, edge_node, kind_node, side_node })
    }

    /// Say where the game's facts hold its units and its tempo.
    pub fn with_decl(mut self, decl: FactDecl) -> Board { self.decl = Some(decl); self }

    /// The board as JSON (the places, kinds and roles in their order, the directions, every step row, the declaration of facts): written and read back identical.
    pub fn to_json(&self) -> Value {
        json!({"places": self.names.places, "kinds": self.names.kinds, "roles": self.names.roles, "directions": self.steps.directions,
               "rows": self.steps.rows.iter().map(|r| json!([r.0, r.1, r.2])).collect::<Vec<_>>(), "relation": self.steps.relation, "object": self.steps.object,
               "decl": self.decl.as_ref().map(|d| json!({"relation": d.units.relation, "owner": d.units.owner, "kind": d.units.kind, "tempo": d.tempo}))})
    }

    pub fn from_json(v: &Value) -> Result<Board, String> {
        let strs = |k: &str| -> Result<Vec<String>, String> { v.get(k).and_then(|x| x.as_array()).ok_or_else(|| format!("board.{k} is a list"))?.iter().map(|x| x.as_str().map(|s| s.to_string()).ok_or_else(|| format!("board.{k} holds text"))).collect() };
        let rows = v.get("rows").and_then(|x| x.as_array()).ok_or("board.rows is a list")?.iter().map(|r| {
            let a = r.as_array().ok_or("a row is a list")?;
            Ok::<_, String>((a.first().and_then(|x| x.as_u64()).ok_or("row.direction")? as u16, a.get(1).and_then(|x| x.as_u64()).ok_or("row.from")? as Place, a.get(2).and_then(|x| x.as_u64()).ok_or("row.to")? as Place))
        }).collect::<Result<Vec<_>, String>>()?;
        let steps = Steps { directions: strs("directions")?, rows, relation: v.get("relation").and_then(|x| x.as_str()).unwrap_or("").to_string(), object: v.get("object").and_then(|x| x.as_str()).unwrap_or("").to_string() };
        let mut b = Board::new(&steps, &Names::new(strs("places")?, strs("kinds")?, strs("roles")?))?;
        if let Some(d) = v.get("decl").filter(|d| !d.is_null()) {
            b.decl = Some(FactDecl {
                units: UnitDecl { relation: d.get("relation").and_then(|x| x.as_str()).ok_or("decl.relation")?.to_string(), owner: d.get("owner").and_then(|x| x.as_u64()).ok_or("decl.owner")? as usize,
                                  kind: match d.get("kind") { None | Some(Value::Null) => None, Some(k) => Some(k.as_u64().ok_or("decl.kind")? as usize) } },
                tempo: d.get("tempo").and_then(|x| x.as_str()).map(|s| s.to_string()) });
        }
        Ok(b)
    }

    /// THE TWO ALTERNATING CLASSES OF PLACES (plan 79.34), derived from the board's own links and no game: a step along a direction that is not made of two other steps (a direction whose every step is exactly the one step of another direction followed by one of a third) CHANGES the class
    /// (the classes alternate along every line of such steps); a direction that is made of two others changes the class as the two together do (two changes are none: along it the class is kept). Where the places can then be coloured so that every step does what its direction says, the classes
    /// are those colours (a place's class is 0 where the colouring of its connected part starts); where they cannot, or a direction is made in a circle, the board does not say and the answer is None.
    pub fn classes(&self) -> Option<Classes> {
        let (nd, np) = (self.directions.len(), self.names.places.len());
        let linked: Vec<bool> = (0..nd).map(|d| self.next[d].iter().any(|x| x.is_some())).collect();
        if np == 0 || !linked.iter().any(|l| *l) { return None; }
        // d is made of a then b when, at every place, the step of d is the step of a followed by the step of b, and exists exactly where that does
        let made_of = |d: usize| -> Option<(usize, usize)> {
            for a in (0..nd).filter(|a| *a != d && linked[*a]) {
                for b in (0..nd).filter(|b| *b != d && linked[*b]) {
                    if (0..np).all(|p| self.next[d][p] == self.next[a][p].and_then(|q| self.next[b][q as usize])) { return Some((a, b)); }
                }
            }
            None
        };
        let parts: Vec<Option<(usize, usize)>> = (0..nd).map(|d| if linked[d] { made_of(d) } else { None }).collect();
        let mut flips: Vec<Option<bool>> = (0..nd).map(|d| if !linked[d] { Some(false) } else if parts[d].is_none() { Some(true) } else { None }).collect();
        for _ in 0..nd {
            for d in 0..nd {
                if let (None, Some((a, b))) = (flips[d], parts[d]) { if let (Some(x), Some(y)) = (flips[a], flips[b]) { flips[d] = Some(x ^ y); } }
            }
        }
        let flips: Vec<bool> = flips.into_iter().collect::<Option<Vec<bool>>>()?;
        let mut class = vec![2u8; np];
        for start in 0..np {
            if class[start] != 2 { continue; }
            class[start] = 0;
            let mut stack = vec![start];
            while let Some(p) = stack.pop() {
                for d in 0..nd {
                    let ends = self.next[d][p].into_iter().map(|q| q as usize).chain((0..np).filter(|q| self.next[d][*q] == Some(p as Place)));
                    for q in ends {
                        let want = class[p] ^ flips[d] as u8;
                        if class[q] == 2 { class[q] = want; stack.push(q); } else if class[q] != want { return None; }
                    }
                }
            }
        }
        Some(Classes { class, flips })
    }

    /// A STATE AS NAMES (plan 79.34): the side marked to act, if any, and each unit as (side, kind, place) names, in the order the graph holds them.
    pub fn state_text(&self, part: &PositionPart) -> (Option<String>, Vec<(String, String, String)>) {
        (part.mover.map(|m| self.names.roles[m].clone()), part.units.iter().map(|u| (self.names.roles[u.owner].clone(), self.names.kinds[u.kind as usize].clone(), self.names.places[u.at as usize].clone())).collect())
    }

    /// A STATE READ BACK FROM ITS NAMES: the graph of the units, with no relation written. A name the game does not give is an error.
    pub fn part_from_text(&self, marked: Option<&str>, units: &[(String, String, String)]) -> Result<PositionPart, String> {
        let mover = match marked { Some(m) => Some(self.names.role(m)?), None => None };
        let mut cells = vec![];
        for (o, k, p) in units { cells.push((self.names.role(o)?, self.names.kind(k)?, self.names.place(p)?)); }
        Ok(self.position_from_units(mover, cells))
    }

    /// The lines on which both places lie.
    pub fn common_lines(&self, a: Place, b: Place) -> Vec<u32> {
        self.place_lines[a as usize].iter().filter(|(l, _)| self.place_lines[b as usize].iter().any(|(m, _)| m == l)).map(|(l, _)| *l).collect()
    }

    /// The least of a place's edge distances over every direction.
    pub fn nearest_edge(&self, p: Place) -> u16 { self.edge_distance.iter().map(|d| d[p as usize]).min().unwrap_or(0) }

    /// The index of a place along a line (-1: it is not on it).
    pub fn index_on(&self, line: u32, p: Place) -> i32 { self.place_lines[p as usize].iter().find(|(l, _)| *l == line).map_or(-1, |x| x.1 as i32) }

    /// The places strictly between two places on a line, in order of the line.
    pub fn strictly_between(&self, line: u32, a: Place, b: Place) -> Vec<Place> {
        let (i, j) = (self.index_on(line, a), self.index_on(line, b));
        if i < 0 || j < 0 { return vec![]; }
        let (lo, hi) = (i.min(j) as usize, i.max(j) as usize);
        if hi <= lo + 1 { return vec![]; }                                     // the same place, or two neighbours: nothing lies between
        self.lines[line as usize].places[lo + 1..hi].to_vec()
    }

    /// THE GRAPH OF A POSITION FROM ITS UNITS AS NUMBERS (owner, kind, place) AND THE TEMPO: the unit nodes with their place, kind and side links, and who stands where. No relation is written (layer 2 does that through `write`).
    pub fn position_from_units(&self, mover: Option<Role>, mut cells: Vec<(Role, Kind, Place)>) -> PositionPart {
        let np = self.names.places.len();
        let ns = self.nodes.len() as u32;
        cells.sort_by_key(|(o, k, p)| (*o, *k, *p));
        let mut nodes: Vec<NodeRec> = Vec::with_capacity(cells.len() * 8);
        let mut links: Vec<LinkRec> = Vec::with_capacity(cells.len() * 3);
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
        PositionPart { mover, nodes, links, units, rels: Default::default(), occupant }
    }

    /// THE UNITS AND THE TEMPO OF A SET OF FACTS, by the board's declaration: (the tempo, the units as numbers). A fact of the unit relation must name a known owner, kind and place, else an error; facts of other relations are not the graph's.
    pub fn read_facts(&self, facts: &[Fact]) -> Result<(Option<Role>, Vec<(Role, Kind, Place)>), String> {
        let d = self.decl.as_ref().ok_or("the board has no declaration of where facts hold units")?;
        let (mut mover, mut cells) = (None, vec![]);
        for f in facts {
            if f.relation == d.units.relation {
                let owner = self.names.role(f.args.get(d.units.owner).ok_or("a unit fact is too short")?)?;
                let kind = match d.units.kind { Some(i) => self.names.kind(f.args.get(i).ok_or("a unit fact is too short")?)?, None => 0 };
                let mut text = String::new();
                for (i, a) in f.args.iter().enumerate() { if i != d.units.owner && Some(i) != d.units.kind { text.push_str(a); } }
                cells.push((owner, kind, self.names.place(&text)?));
            } else if d.tempo.as_deref() == Some(f.relation.as_str()) {
                mover = Some(self.names.role(f.args.first().ok_or("a tempo fact is empty")?)?);
            }
        }
        cells.sort_unstable();
        cells.dedup();                                                           // a fact said twice is one unit
        Ok((mover, cells))
    }

    /// THE GRAPH OF A POSITION FROM A SET OF FACTS ALONE (and the board): units and tempo read by the declaration, then `position_from_units`.
    pub fn position_from_facts(&self, facts: &[Fact]) -> Result<PositionPart, String> {
        let (mover, cells) = self.read_facts(facts)?;
        Ok(self.position_from_units(mover, cells))
    }

    /// THE DOOR OF CHANGE: write one relation of a unit onto the position's graph: the node, its id and its links (by, at_place, at_unit, along, between). The layer above calls it with what its rules say and names the relation by a value (`Rel`).
    pub fn write(&self, part: &mut PositionPart, rel: Rel, ui: usize, to: Place) {
        let r = self.relation(part, rel, ui, to);
        part.rels[rel.slot as usize].push(r);
    }

    /// One relation node with its links.
    fn relation(&self, part: &mut PositionPart, rel: Rel, ui: usize, to: Place) -> RelRec {
        let ns = self.nodes.len() as u32;
        let u = part.units[ui];
        let id = format!("{}({},place:{})", rel.word, part.nodes[(u.node - ns) as usize].id, self.names.places[to as usize]);
        let node = ns + part.nodes.len() as u32;
        part.nodes.push(NodeRec { kind: NodeType::Relation(rel), id });
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
}

impl PositionPart {
    /// The relations written in a slot, in the order they were written.
    #[inline(always)]
    pub fn written(&self, rel: Rel) -> &Vec<RelRec> { &self.rels[rel.slot as usize] }

    /// How many relations a layer above has written onto the graph.
    pub fn relation_count(&self) -> usize { self.rels.iter().map(|v| v.len()).sum() }

    /// How many units of a kind a side has in the state.
    pub fn count(&self, owner: Role, kind: Kind) -> usize { self.units.iter().filter(|u| u.owner == owner && u.kind == kind).count() }

    /// The tempo: whose turn it is (None: nobody's, or not said).
    pub fn tempo(&self) -> Option<Role> { self.mover }

    /// The longest run of places on a line with no unit standing on them.
    pub fn open_length(&self, st: &Board, line: usize) -> usize {
        let (mut best, mut run) = (0, 0);
        for p in &st.lines[line].places { if self.occupant[*p as usize] == 0 { run += 1; best = best.max(run); } else { run = 0; } }
        best
    }

    /// A text dump of the position part, one node per line with its links (the ids are the definitions).
    pub fn dump(&self, st: &Board) -> String {
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

    /// THE GRAPH AS A SET OF DEFINITIONS, order-free: every node id and every link as (kind, from id, to id, via id, value), sorted. Two graphs of one board are the same graph exactly when these are equal.
    pub fn definitions(&self, st: &Board) -> (Vec<String>, Vec<String>) {
        let ns = st.nodes.len() as u32;
        let id = |n: i64| -> String { if n < 0 { String::new() } else if (n as u32) < ns { st.nodes[n as usize].id.clone() } else { self.nodes[(n as u32 - ns) as usize].id.clone() } };
        let mut nodes: Vec<String> = self.nodes.iter().map(|n| n.id.clone()).collect();
        let mut links: Vec<String> = self.links.iter().map(|l| format!("{}|{}|{}|{}|{}", l.kind.word(), id(l.a as i64), id(l.b as i64), id(l.via as i64), l.value)).collect();
        nodes.sort();
        links.sort();
        (nodes, links)
    }
}
