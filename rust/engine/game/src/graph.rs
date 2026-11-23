//! PLAN 79.21 STAGE B: THE GRAPH IS LAYER 1'S (cs_rep::graph: the board, the units, the tempo, the nodes and their ids, the door `Board::write`); WHAT STAYS HERE IS LAYER 2: how a unit steps. `Static` is the board plus the reading of a game's rules onto it: `Static::position` takes a game
//! and a state, hands the units and the tempo to layer 1 (`Board::position_from_units`) and writes the relations the rules give (bear, may_move, shut) through the door. The types below are re-exported from cs_rep; their documentation is there.
//!
//! (the text that follows is the description of plan 78.52, the graph as it was first made; the node ids and links are unchanged.)
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
//!   through      `through(<unit id>,<place id>)`                      PLAN 78.54, a RELATION node, for BOTH sides, no hypothetical board: the unit bears on a unit b along a line (a bear node), the unit is of a kind that
//!                                                                      CONTINUES along that line in that heading, and the next place beyond b that holds a unit is the place of the node. Its links: by, at_place, at_unit (the unit
//!                                                                      there), along (the line), between (b). It exists only once a profile of the kinds has been learned (`Static::learn`): see below.
//! HOW A UNIT IS KNOWN TO CONTINUE ALONG A LINE (78.54 A). From the bear nodes alone: a kind continues along a line family in a heading (the line's direction and the sign of the way along it) when some
//! unit of that kind, in some position of the sample the profile is learned from, bears on a place two or more steps from it along a line of that family in that heading (the places between are then passed
//! over). No list of units or movements is written; a kind the sample never shows bearing at two steps in a heading is read as not continuing there (a limit: it is a sample).
//! PLAIN LINKS (a kind, two nodes, optionally a node it goes by and a number):
//!   on_line (place -> line, the index along the line), next (place -> place, by a direction), on_edge (place -> edge), edge_distance (place -> edge, the steps until the chain
//!   ends under that direction), line_along (line -> direction), reverse (direction -> direction), edge_of (edge -> direction), unit_on (unit -> place), unit_kind, unit_side.
//! RELATION LINKS (the same for bear, may_move and shut): by (relation -> its unit), at_place (-> the place), at_unit (-> the unit standing there), along (-> the line it runs along,
//!   where from and to lie on one line), between (-> each unit standing strictly between on that line). A bearing that follows no line (a leap) has no `along` and no `between`.
use crate::placed::{NoStep, Placed, Steps};
use crate::{Game, Role};
use cs_rep::graph::Board;
use cs_rep::live::LiveGraph;
use cs_rep::names::{Kind, Names, Place};
pub use cs_rep::graph::{Classes, FactDecl, Line, LinkRec, LinkType, NodeRec, NodeType, PositionPart, Rel, RelRec, UnitRec};
pub use crate::relations::{node_words, PartRelations, UnitRelations, BEAR, MAY_MOVE, SHUT, THROUGH};

/// PLAN 79.22 PART A, THE COUNT OF GRAPH BUILDS (a measure, never read by a decision): while a counter is on (`builds::start(root key)`) every graph the rules' reading builds is noted with the place that asked and whether it is the graph of the root position (the key given) or another's (a position after a move read), so that a test or a measure can say how many times a decision built the root's graph and where. Off, a call costs one relaxed load.
pub mod builds {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    static ON: AtomicBool = AtomicBool::new(false);
    static SEEN: Mutex<(u64, BTreeMap<(bool, String), usize>)> = Mutex::new((0, BTreeMap::new()));
    /// Begin counting; `root` is the key of the position the decision is made at (the counts of an earlier run are dropped).
    pub fn start(root: u64) { *SEEN.lock().unwrap() = (root, BTreeMap::new()); ON.store(true, Ordering::Relaxed); }
    /// Stop counting and return the builds as (is the root's graph, the place that asked as file:line, how many).
    pub fn take() -> Vec<(bool, String, usize)> { ON.store(false, Ordering::Relaxed); std::mem::take(&mut SEEN.lock().unwrap().1).into_iter().map(|((r, p), n)| (r, p, n)).collect() }
    pub fn on() -> bool { ON.load(Ordering::Relaxed) }
    pub fn note(at: &'static std::panic::Location<'static>, key: u64) { let mut g = SEEN.lock().unwrap(); let root = g.0 == key; *g.1.entry((root, format!("{}:{}", at.file().rsplit('/').next().unwrap_or(""), at.line()))).or_insert(0) += 1; }
}

/// PLAN 79.22 PART A. THE ROOT'S GRAPH, MADE ONCE A DECISION AND LENT TO EVERY PART THAT READS IT. INTERFACE.
///   `Root::of(&Static, g, s) -> Root`            the graph of the position the decision is made at, built from the rules (once).
///   `Root::from_part(key, PositionPart) -> Root` the same from a graph the game's memory kept (see cs_decision::keep): no build.
///   `Root::key() / part()`                       the key of the position; the graph, shared by reference count.
///   `Static::position_in(&self, &Root, g, s) -> Rc<PositionPart>`   the graph of `s`: the root's own (no build) when `s` has the root's key, otherwise built.
/// WHO OWNS IT: the wake's decision (`cs_decision::wake::decide`) makes it and keeps it for the decision; every part (the threat list, the goals, the reasons' pick, the frame read) BORROWS it as `&Root`. A part that reads other positions (the position after a move) calls `position_in` with the same root and gets a build for them only.
#[derive(Debug, Clone)]
pub struct Root { key: u64, part: std::rc::Rc<PositionPart> }

impl Root {
    pub fn of<G: Game + Placed>(st: &Static, g: &G, s: &G::State) -> Root { Root { key: g.key(s), part: std::rc::Rc::new(st.position(g, s)) } }
    pub fn from_part(key: u64, part: PositionPart) -> Root { Root { key, part: std::rc::Rc::new(part) } }
    pub fn key(&self) -> u64 { self.key }
    pub fn part(&self) -> &std::rc::Rc<PositionPart> { &self.part }
}

/// THE STATIC PART plus the rules' reading of it: the board of layer 1 (every field of it is reached through `Deref`), the functions that ask a game how its units step, and (plan 79.34, moved up from layer 1's board) the PROFILE of the kinds learned from the bearing relations.
/// Built once per game.
#[derive(Debug, Clone)]
pub struct Static {
    board: Board,
    /// [kind][direction][heading]: the kind has been seen bearing two or more steps along a line named in that direction, heading 1 forward and 0 backward; empty until `learn`
    pub cont: Vec<bool>,
}

impl std::ops::Deref for Static { type Target = Board; fn deref(&self) -> &Board { &self.board } }
impl std::ops::DerefMut for Static { fn deref_mut(&mut self) -> &mut Board { &mut self.board } }

impl Static {
    /// Build the static part from the rules' steps and the game's names (layer 1's `Board::new`).
    pub fn new(steps: &Steps, names: &Names) -> Result<Static, String> { Ok(Static { board: Board::new(steps, names)?, cont: vec![] }) }
    /// The same from a board that layer 1 holds already (read from a file or made elsewhere).
    pub fn from_board(board: Board) -> Static { Static { board, cont: vec![] } }
    pub fn board(&self) -> &Board { &self.board }

    /// Build the part of one position: the units and the tempo from the game's state as numbers (layer 1 makes the unit nodes), then the relations the rules give, written onto the graph through the door of change.
    #[track_caller]
    pub fn position<G: Game + Placed>(&self, g: &G, s: &G::State) -> PositionPart {
        if builds::on() { builds::note(std::panic::Location::caller(), g.key(s)); }
        let mut part = self.board.position_from_units(g.mover(s), g.cells(s));
        self.write_rules(g, s, &mut part);
        part
    }

    /// The graph of `s` for a decision that holds `root`: the root's own when `s` is its position, else built (see `Root`).
    #[track_caller]
    pub fn position_in<G: Game + Placed>(&self, root: &Root, g: &G, s: &G::State) -> std::rc::Rc<PositionPart> {
        if g.key(s) == root.key { root.part.clone() } else { std::rc::Rc::new(self.position(g, s)) }
    }

    /// What the rules say of how each unit steps, written onto a graph that holds the units: bear and may_move for every unit, the shut lines of the side to move, then (once a profile of the kinds is learned) the lines through, by geometry.
    pub fn write_rules<G: Game + Placed>(&self, g: &G, s: &G::State, part: &mut PositionPart) {
        for ui in 0..part.units.len() {
            let u = part.units[ui];
            for p in g.bearing_from(s, u.owner, u.kind, u.at) { self.board.write(part, BEAR, ui, p); }
            for p in g.reach_from(s, u.owner, u.kind, u.at) { self.board.write(part, MAY_MOVE, ui, p); }
        }
        if let Some(m) = part.mover { self.shut_lines(g, s, m, part); }
        self.derive_throughs(part);
    }

    /// THE GRAPH OF THE POSITION A `LiveGraph` KEEPS (plan 79.34; this replaces layer 1's `LiveGraph::part`): layer 1's bare graph, then every relation written through the door in the order a fresh build writes them (per unit its bears then its moves; then the shut lists; then the lines through).
    pub fn part_of(&self, live: &LiveGraph) -> PositionPart {
        let mut p = live.bare_part(&self.board);
        for (ui, u) in live.units().iter().enumerate() {
            for &q in u.bears() { self.board.write(&mut p, BEAR, ui, q); }
            for &q in u.reach() { self.board.write(&mut p, MAY_MOVE, ui, q); }
        }
        for (ui, u) in live.units().iter().enumerate() { for &q in u.shuts() { self.board.write(&mut p, SHUT, ui, q); } }
        self.derive_throughs(&mut p);
        p
    }

    fn cont_ix(&self, kind: Kind, line: u32, forward: bool) -> usize { (kind as usize * self.board.directions.len() + self.board.lines[line as usize].direction) * 2 + forward as usize }

    /// Whether the kind continues along the line in the heading (false until a profile is learned).
    pub fn continues(&self, kind: Kind, line: u32, forward: bool) -> bool { self.cont.get(self.cont_ix(kind, line, forward)).copied().unwrap_or(false) }

    /// LEARN THE PROFILE of the kinds from the bear nodes of a position (78.54 A): a unit bearing on a place two or more steps away along a line marks its kind as continuing along that line family
    /// in that heading. Call it on the parts of a sample of positions.
    pub fn learn(&mut self, p: &PositionPart) {
        if self.cont.is_empty() { self.cont = vec![false; self.board.names.kinds.len() * self.board.directions.len() * 2]; }
        for r in p.bears() {
            let u = p.units[r.unit];
            for &l in &r.along {
                let (i, j) = (self.board.index_on(l, u.at), self.board.index_on(l, r.place));
                if i >= 0 && j >= 0 && (j - i).abs() >= 2 { let ix = self.cont_ix(u.kind, l, j > i); self.cont[ix] = true; }
            }
        }
    }

    /// THE LINE THROUGH ONE UNIT, FOR BOTH SIDES (78.54 A), written onto the graph from the bear relations already there: for each bearing of a unit on a unit b along a line, where the unit's kind continues along that line in that heading, the next
    /// place beyond b that holds a unit; one relation node per (unit, that place). Only where b is the one unit between (a unit that jumps others is left out: a limit). Geometry only: nothing of how a unit steps is asked.
    pub fn derive_throughs(&self, part: &mut PositionPart) {
        if self.cont.is_empty() { return; }
        let mut found: Vec<(usize, Place)> = vec![];
        for r in part.bears() {
            let u = part.units[r.unit];
            let b = part.occupant[r.place as usize];
            if b == 0 { continue; }
            for &l in &r.along {
                let (i, j) = (self.board.index_on(l, u.at), self.board.index_on(l, r.place));
                if i < 0 || j < 0 || j == i || !self.continues(u.kind, l, j > i) { continue; }
                let sign = if j > i { 1 } else { -1 };
                let line = &self.board.lines[l as usize];
                let mut k = j + sign;
                while k >= 0 && (k as usize) < line.places.len() {
                    let q = line.places[k as usize];
                    if part.occupant[q as usize] != 0 {
                        if self.board.strictly_between(l, u.at, q).iter().filter(|x| part.occupant[**x as usize] != 0).count() == 1 { found.push((r.unit, q)); }
                        break;
                    }
                    k += sign;
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        for (ui, q) in found { self.board.write(part, THROUGH, ui, q); }
    }

    /// The places of a unit's shut lines, as the rules say (the places beyond the farthest bearing on either side, where the game says the line is shut): in the order `write_rules` writes them. `bears` are the places the unit bears on.
    pub fn shut_places<G: Game + Placed>(&self, g: &G, s: &G::State, at: Place, bears: &[Place]) -> Vec<Place> {
        let mut out = vec![];
        for &(l, i) in &self.board.place_lines[at as usize] {
            let ends: Vec<i32> = bears.iter().filter(|p| self.board.index_on(l, **p) >= 0).map(|p| self.board.index_on(l, *p) - i as i32).collect();
            for sign in [1i32, -1] {
                let Some(far) = ends.iter().filter(|d| d.signum() == sign).map(|d| d.abs()).max() else { continue };
                let line = &self.board.lines[l as usize];
                let mut k = i as i32 + sign * (far + 1);
                while k >= 0 && (k as usize) < line.places.len() {
                    let q = line.places[k as usize];
                    if let (NoStep::LineShut, _) = g.why_not(s, at, q) { out.push(q); }
                    k += sign;
                }
            }
        }
        out
    }

    /// The shut lines of the side to move: for each of its units and each line it bears along, the places beyond the farthest bearing on either side of the unit, where the game says the
    /// line is shut (`why_not` = LineShut), with the units standing between.
    fn shut_lines<G: Game + Placed>(&self, g: &G, s: &G::State, mover: Role, part: &mut PositionPart) {
        for ui in 0..part.units.len() {
            let u = part.units[ui];
            if u.owner != mover { continue; }
            for &(l, i) in &self.board.place_lines[u.at as usize] {
                let ends: Vec<i32> = part.bears().iter().filter(|r| r.unit == ui && r.along.contains(&l)).map(|r| self.board.index_on(l, r.place) - i as i32).collect();
                for sign in [1i32, -1] {
                    let Some(far) = ends.iter().filter(|d| d.signum() == sign).map(|d| d.abs()).max() else { continue };
                    let line = &self.board.lines[l as usize];
                    let mut k = i as i32 + sign * (far + 1);
                    while k >= 0 && (k as usize) < line.places.len() {
                        let q = line.places[k as usize];
                        if let (NoStep::LineShut, _) = g.why_not(s, u.at, q) { self.board.write(part, SHUT, ui, q); }
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
    pub fn shut_lines_other<G: Game + Placed>(&self, g: &G, s: &G::State, mover: Role, part: &mut PositionPart, _ns: u32) {
        for ui in 0..part.units.len() {
            let u = part.units[ui];
            if u.owner == mover { continue; }
            self.first_blocker_shuts(g, s, ui, part, _ns);
        }
    }

    /// The first-blocker shut relations of one unit, added to the part.
    pub fn first_blocker_shuts<G: Game + Placed>(&self, g: &G, s: &G::State, ui: usize, part: &mut PositionPart, _ns: u32) {
        let u = part.units[ui];
        for &(l, i) in &self.place_lines[u.at as usize] {
            let ends: Vec<i32> = part.bears().iter().filter(|r| r.unit == ui && r.along.contains(&l)).map(|r| self.index_on(l, r.place) - i as i32).collect();
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
                    self.board.write(part, SHUT, ui, q);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placed::read_steps;

    /// A board of four columns and three rows with four headings (no diagonal), written as a description: nothing like the board of the game the engine plays.
    const SMALL: &str = "
(sort col x y z w) (sort row 1 2 3) (sort heading up down left right)
(signature col col) (signature row row) (signature col_succ col col) (signature row_succ row row) (signature step heading col row col row)
(col x) (col y) (col z) (col w) (row 1) (row 2) (row 3)
(col_succ x y) (col_succ y z) (col_succ z w) (row_succ 1 2) (row_succ 2 3)
(<= (step up ?c ?ra ?c ?rb) (col ?c) (row_succ ?ra ?rb))
(<= (step down ?c ?ra ?c ?rb) (col ?c) (row_succ ?rb ?ra))
(<= (step right ?ca ?r ?cb ?r) (row ?r) (col_succ ?ca ?cb))
(<= (step left ?ca ?r ?cb ?r) (row ?r) (col_succ ?cb ?ca))
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
        let steps = crate::placed::Steps { directions: vec!["d".into()], rows: vec![(0, 0, 1), (0, 0, 2)], ..Default::default() };
        assert!(Static::new(&steps, &Names::new(names, vec![], vec![])).is_err());
    }
}
