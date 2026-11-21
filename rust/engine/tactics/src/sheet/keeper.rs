//! PLAN 79.21 STAGE B, LAYER 2'S KEEPER OF THE GRAPH (SYSTEM_SPEC 11 "one graph is the representation", "layer 1 holds the history"; Kaveh 2026-11-20: "the graph of a position must be kept step by step in make and undo, never rebuilt from facts at each node"; piece: representation, the rules above it).
//! Layer 1 keeps the graph (`cs_rep::live::LiveGraph`: units, tempo, the relations written through the door, make and undo). WHAT A TEMPO CHANGES IN THE RELATIONS is a question of the rules, so it is asked here: after a tempo the places whose holder changed are `C`; a unit's relations (what
//! it bears on, where it may move) can have changed only if the unit is new, or a place of `C` lies on a line it stands on, or a place of `C` is one of its relations' places; those units are asked again of the rules (`bearing_from`, `reach_from`) and every other unit keeps what it had. The shut
//! lines belong to the side to move, which changes every tempo, so they are asked again for that side's units (`Static::shut_places`). The rule is a SUPERSET of what can change, tested equal to a fresh build at every position of the fixture games and on undo.
//!
//! INTERFACE.
//!   `GraphKeeper::new(&Static, g, s) -> GraphKeeper`        the graph of a first position, every relation asked of the rules once.
//!   `GraphKeeper::make(&mut self, &Static, g, after)`        one tempo: `after` is the state reached; only the units the tempo can have changed are asked again.
//!   `GraphKeeper::undo(&mut self) -> bool`                   the last tempo taken back (no rule is asked).
//!   `GraphKeeper::part(&self, &Static) -> PositionPart`      the graph as it now is, equal to `Static::position` on the state (the lines through included once a profile is learned).
//!   `GraphKeeper::asked() -> (units asked again, units in all)`   what the last make cost in questions to the rules.
//! Called by: the tests, and the layers that walk a sequence of positions (a game being played, a line being read) instead of building the graph of each from nothing.
use super::base_graph::Static;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_rep::live::{Edit, LiveGraph, LiveUnit};
use cs_rep::graph::PositionPart;
use cs_rep::names::Place;

pub struct GraphKeeper { live: LiveGraph, asked: (usize, usize) }

impl GraphKeeper {
    pub fn new<G: Game + Placed>(st: &Static, g: &G, s: &G::State) -> GraphKeeper {
        let mover = g.mover(s);
        let mut units: Vec<LiveUnit> = g.cells(s).into_iter().map(|(owner, kind, at)| LiveUnit { owner, kind, at, bears: g.bearing_from(s, owner, kind, at), reach: g.reach_from(s, owner, kind, at), shuts: vec![] }).collect();
        units.sort_by_key(|u| u.key());
        if let Some(m) = mover { for u in units.iter_mut().filter(|u| u.owner == m) { u.shuts = st.shut_places(g, s, u.at, &u.bears); } }
        let n = units.len();
        GraphKeeper { live: LiveGraph::new(mover, units), asked: (n, n) }
    }

    pub fn make<G: Game + Placed>(&mut self, st: &Static, g: &G, after: &G::State) {
        let mover = g.mover(after);
        let mut cells = g.cells(after);
        cells.sort_by_key(|(o, k, p)| (*o, *k, *p));
        let before: Vec<(usize, u8, Place)> = self.live.units().iter().map(|u| u.key()).collect();
        let removed: Vec<_> = before.iter().filter(|k| cells.binary_search(k).is_err()).copied().collect();
        let added: Vec<_> = cells.iter().filter(|k| before.binary_search(k).is_err()).copied().collect();
        let changed: Vec<Place> = removed.iter().chain(added.iter()).map(|k| k.2).collect();
        let mut rewrite = vec![];
        let mut asked = 0;
        // the relations of every unit of the state reached, as they will stand: the old ones kept, the ones the tempo can have changed asked again
        let mut bears_of: Vec<((usize, u8, Place), Vec<Place>)> = Vec::with_capacity(cells.len());
        for &(owner, kind, at) in &cells {
            let key = (owner, kind, at);
            let old = self.live.units().binary_search_by_key(&key, |u| u.key()).ok().map(|i| &self.live.units()[i]);
            let again = match old {
                None => true,
                Some(u) => changed.iter().any(|&c| !st.common_lines(at, c).is_empty() || u.bears.contains(&c) || u.reach.contains(&c)),
            };
            if again {
                asked += 1;
                let (bears, moves) = (g.bearing_from(after, owner, kind, at), g.reach_from(after, owner, kind, at));
                bears_of.push((key, bears.clone()));
                rewrite.push((key, bears, moves));
            } else { bears_of.push((key, old.unwrap().bears.clone())); }
        }
        let mut shuts = vec![];
        if let Some(m) = mover {
            for (key, bears) in &bears_of { if key.0 == m { shuts.push((*key, st.shut_places(g, after, key.2, bears))); } }
        }
        self.asked = (asked, cells.len());
        self.live.make(Edit { tempo: mover, removed, added, rewrite, clear_shuts: true, shuts });
    }

    pub fn undo(&mut self) -> bool { self.live.undo() }
    pub fn asked(&self) -> (usize, usize) { self.asked }
    pub fn depth(&self) -> usize { self.live.depth() }
    /// The graph kept, for a question asked of it as it is (no node made).
    pub fn live(&self) -> &LiveGraph { &self.live }

    /// The graph as it now is. The lines through are derived from the bears written and the profile of the kinds (geometry, no rule).
    pub fn part(&self, st: &Static) -> PositionPart { self.live.part(st.board()) }
}
