//! PLAN 79.22 PARTS A AND B: THE GAME'S MEMORY KEEPS THE POSITION AT HAND BETWEEN DECISIONS (SYSTEM_SPEC 11 "layer 1 holds the history", "one graph is the representation", the two memories; ledger row 36; piece: representation and its use by the wake).
//! What layer 1 keeps of the game at hand is the history (a change and a key for each position), the facts of the position the history ends at and the graph of that position kept step by step (`cs_rep::store::AtHand`). Layer 2's keepers (`cs_game::tags::TagKeeper`, which writes the tags, and
//! `cs_tactics::sheet::keeper::GraphKeeper`, which asks the rules how the units step) change them by the moves played; nothing is built from nothing unless the history does not continue. The wake reads the root graph from here once a decision and the record of the game as `History::seen`.
//!
//! INTERFACE (written before the body).
//!   `bring(st, g, decl, memory, keys_before, s) -> Brought`   takes the game's memory, the keys of the positions before `s` as the caller gives them and the position `s`; brings the memory's history, tags and graph to `s`:
//!        the history continues (the kept position is one of the keys before, and every key up to it is the history's own): the tempos since are made, ONE CHANGE for the tempo into `s` (the facts' difference) and the keys of the tempos skipped before it as not-known changes; the tags fall and are brought by that change; the graph asks the rules again only about the units the change can have touched;
//!        it does not continue (a game begun in the middle, another game, a record taken back): the history is the keys given with no change known, the tags are STATED AS FACTS from what the rules give, the graph is built from the rules.
//!        Returns how it was brought (`Brought`). Called by the wake's binding before `decide`.
//!   `root_of(st, g, s, memory) -> Root`                       the root graph of the decision: the kept graph's own when the memory is at `s`, otherwise built. Called by `wake::decide_seen` once a decision.
//!   `advance(st, g, decl, memory, after)`                      the move the decision chose is played in the memory: the history gets the tempo (its change and key), the tags and the graph follow. The next decision then finds the memory one tempo before the position it is asked about, and brings it there by the other side's move alone. Called by `wake::decide_seen` at its end.
//! A `FactDecl` says which relation of the game's facts holds its units and which its tempo (data the game gives: the binding is handed it); every other fact is a tag.
use cs_game::placed::Placed;
use cs_game::tags::TagKeeper;
use cs_game::Game;
use cs_rep::graph::FactDecl;
use cs_rep::store::{AtHand, GameMemory};
use cs_tactics::sheet::base_graph::{Root, Static};
use cs_tactics::sheet::keeper::GraphKeeper;

/// How the memory was brought to the position asked about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Brought {
    /// the memory was already at this position (it was asked again)
    Same,
    /// the history continued: `tempos` positions were reached since the kept one; the last by its change, the `tempos - 1` before it with only their keys
    Continued { tempos: usize },
    /// the history did not continue: everything stated afresh from the rules
    Fresh,
}

/// The keys the memory's history holds, if the kept position is the one it ends at.
fn kept_keys(memory: &GameMemory) -> Option<Vec<u64>> {
    let (at, h) = (memory.at_hand()?, memory.history()?);
    if h.current_key() == at.key { Some(h.keys().to_vec()) } else { None }
}

pub fn bring<G: Game + Placed>(st: &Static, g: &G, decl: &FactDecl, memory: &mut GameMemory, keys_before: &[u64], s: &G::State) -> Brought {
    let now = g.key(s);
    let mut want = keys_before.to_vec();
    want.push(now);
    if let Some(have) = kept_keys(memory) {
        if want.len() >= have.len() && want[..have.len()] == have[..] {
            let new = &want[have.len()..];
            if new.is_empty() { return Brought::Same; }
            let at = memory.take_at_hand().expect("kept");
            let h = memory.take_history().expect("kept");
            let mut tk = TagKeeper::resume(decl, at.facts, h, std::mem::take(memory));
            let mut gk = GraphKeeper::from_live(at.graph);
            tk.make_over(g, s, &new[..new.len() - 1]);
            gk.make(st, g, s);
            store(memory, tk, gk, now);
            return Brought::Continued { tempos: new.len() };
        }
    }
    // the history does not continue: state everything afresh (a kept graph or history of another game is let go)
    memory.take_at_hand();
    memory.take_history();
    let tk = TagKeeper::over(g, s, decl, std::mem::take(memory), keys_before);
    let gk = GraphKeeper::new(st, g, s);
    store(memory, tk, gk, now);
    Brought::Fresh
}

fn store(memory: &mut GameMemory, tk: TagKeeper, gk: GraphKeeper, key: u64) {
    let (facts, history, mem) = tk.into_parts();
    *memory = mem;
    memory.put_history(history);
    memory.put_at_hand(AtHand { key, facts, graph: gk.into_live() });
}

pub fn root_of<G: Game + Placed>(st: &Static, g: &G, s: &G::State, memory: &GameMemory) -> Root {
    let key = g.key(s);
    match memory.at_hand() {
        Some(at) if at.key == key => Root::from_part(key, at.graph.part(st.board())),
        _ => Root::of(st, g, s),
    }
}

pub fn advance<G: Game + Placed>(st: &Static, g: &G, decl: &FactDecl, memory: &mut GameMemory, after: &G::State) {
    let (Some(at), Some(_)) = (memory.at_hand(), memory.history()) else { return };
    if kept_keys(memory).is_none() || at.key == g.key(after) { return; }
    let at = memory.take_at_hand().expect("kept");
    let h = memory.take_history().expect("kept");
    let mut tk = TagKeeper::resume(decl, at.facts, h, std::mem::take(memory));
    let mut gk = GraphKeeper::from_live(at.graph);
    tk.make(g, after);
    gk.make(st, g, after);
    store(memory, tk, gk, g.key(after));
}
