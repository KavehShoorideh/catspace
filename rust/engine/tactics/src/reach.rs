//! PLAN 79.27 PART B AND PLAN 79.28 PARTS A AND B, THE BODY OF LAYER 4'S SEARCH TO STATEMENTS (SYSTEM_SPEC 11 "Layer 4 takes a goal as an input", "Reached means the goal outlasts the reply", "the layers"; ledger rows 36, 14; piece: tactical reasoner). The entry is `meansends::ask`; this file is what it calls for the goals that are statements.
//!
//! INTERFACE (written before the body). It takes the game through layer 2 (`Game`, `Placed`, layer 2's `Static` for the board), a state, the side that wants the goals, the goals as layer 1's statements (a pattern over the facts of the graph, `cs_rep::statement::Statement`),
//! a budget in nodes (and optionally a clock that may only stop it earlier), the deepest depth in plies, the history (layer 1's `Seen`) and the narrowing (`Narrow`). It returns, for each goal, one of four answers: REACHED (a line of the two sides' moves to a position where the pattern is HELD, the bindings there, what the judge says the
//! line took and cost, and the goals that are also held somewhere along the line), TOUCHED (the pattern binds after a move of the side and a reply unbinds it, and the goal is not reached within the depth: the line to the position where it binds, that reply, the bindings there), OUT OF REACH within the depth read in full (every line of at most that many
//! plies read, the goal neither reached nor touched: a statement of that depth only), or NOT SETTLED (the budget cut the reading; a cut is a gap, never a verdict).
//! Called by: `meansends::ask` (the one entry), and the layer's test, which reads it with the narrowing off to measure the narrowing against it.
//!
//! WHAT A GOAL IS HERE (plan 79.28: it must outlast the reply). A pattern is HELD at a position when it binds there AND the side asking is to move there, or the game is over there. Side S REACHES goal G from position P in n plies when S has a way of moving such that, whatever the other side replies, a position where G is held is met within n
//! plies (P itself counts, at 0: a pattern that binds at the root with S to move is reached in nought plies). A pattern that binds right after a move of S, with the other side to move, is not held there: it is held only if it still binds after every reply, at the position where S is to move again. The line stops where the pattern is held.
//! A position where the game is over ends a line (the pattern is read there first); a position met again (in layer 1's record, or earlier on the line) ends it too, as the rules' draw; a position at the depth limit ends it unread further.
//! A goal TOUCHED: S has a way of moving such that, whatever the other side replies, a position is met where G is held, or where G binds after a move of S (the other side to move, the game not over, ply of at least one) and some reply of the other side leaves it unbound (the reply is read, so it needs one ply more than the line);
//! and the goal is not reached. So a touched goal is one that S can force to bind, and that the other side can always undo. The line stops at the position where it binds; the reply that unbinds it is said apart; they are given only for a goal not reached.
//! THE BODY. (1) THE SOUND READING: an and-or search, deepened a ply at a time inside the budget, every move read, the patterns matched on the LIVE graph (`cs_game::keeper::GraphKeeper`: made and undone one tempo at a time, never built anew) through layer 1's `FactReader` and the matcher's one join. ONE TREE SERVES THE LIST: at a
//! position the patterns of all goals still open are matched together (one set of rows, one join each), and the search returns for every open goal its plies; a child is read for the goals still open only (an or-node stops asking for a goal once a child reaches it; an and-node stops asking for a goal once a reply leaves it out of reach).
//! TWO READINGS IN THE ONE TREE: the first asks for the goals held (reached); the second, run only for the goals the first settled as not reached, asks for "held, or touched" in the same tree, the leaf being the one more kind. The second reading is a superset of the first, so a goal it does not find is neither reached nor touched.
//! (2) THE NARROWING, measured against (1) on the same questions: (a) a table of answers by position, goal and reading, kept across the deepening, of two kinds (reached within n, out of reach within n), never stored for an answer that rests on the line (a repeat on the line); and the rows' matching kept by position; (b) an order of moves: those that arrive on a place the first open
//! goal's pattern names, that leave one, that take a unit, that are of a kind it names, first. An order is never a verdict. NO CUT OF MOVES: a cut that is sound for the last ply (a move that changes no row of the pattern cannot make it bind) can be known only from the position the move makes, so it saves matching and not nodes; it is the same test the table's rows make (the rows
//! equal to the parent's are not matched again). A cut at an earlier ply is unsound (a move that changes no row may still open a later move), and none is kept.
use crate::ledger::Judge;
use cs_game::graph::Static;
use cs_game::keeper::GraphKeeper;
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use cs_rep::graph_facts::FactReader;
use cs_rep::history::{Path, Seen};
use cs_rep::lemma::{Arg, Lit};
use cs_rep::matcher::{Binding, FactIndex};
use cs_rep::statement::Statement;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// No answer for a goal (not reached within the depth asked).
const NONE: u8 = u8::MAX;
/// The most goals one tree serves (a mask of positions).
pub const MOST_GOALS: usize = 64;

/// The narrowing: the table of answers and the order of moves. `PLAIN` is the sound reading with neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Narrow { pub table: bool, pub order: bool }
pub const NARROWED: Narrow = Narrow { table: true, order: true };
pub const PLAIN: Narrow = Narrow { table: false, order: false };

/// What the layer asks of the body.
pub struct Ask<'a, G: Game + Placed, J: Judge> {
    pub game: &'a G,
    pub board: &'a Static,
    pub state: &'a G::State,
    pub judge: &'a J,
    pub side: Role,
    pub goals: &'a [Statement],
    pub nodes: u64,
    pub seconds: Option<f64>,
    pub max_depth: i32,
    pub history: Arc<Seen>,
    pub narrow: Narrow,
}

/// A line to a position where a pattern is held (reached) or binds (touched).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    pub first: Option<usize>,
    pub names: Vec<String>,
    /// plies of the line (the number of moves it holds)
    pub plies: i32,
    /// where the pattern binds at the line's end (at most `BINDINGS` of them)
    pub bindings: Vec<Binding>,
    /// the units the line takes, (owner, kind)
    pub taken: Vec<(Role, String)>,
    /// what the judge says the line cost or won for the side, read after the goal holds (units taken, kinds changed)
    pub value: i64,
    /// the goals (by place in the list) held at some position of the line, this goal's own included (for a reached goal)
    pub meets: u64,
}

/// A goal touched: the line to the position where the pattern binds after a move of the side, and the reply that unbinds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Touch {
    pub line: Line,
    pub reply: String,
}

/// The most bindings kept of a reached goal.
pub const BINDINGS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Reached(Line),
    Touched(Touch),
    /// every line of at most `depth` plies read, the goal neither reached nor touched
    Out { depth: i32 },
    /// the budget cut the reading; `depth` is the deepest read in full (0: none)
    Unsettled { depth: i32 },
}

/// What the body returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Read {
    pub outcomes: Vec<Outcome>,
    pub nodes: u64,
    /// positions made onto the live graph / rows matched (where the cost goes)
    pub makes: u64,
    pub matches: u64,
    /// the deepest iteration read in full
    pub depth: i32,
    pub cut: bool,
    /// no line met the depth limit in the last iteration: deeper reading changes nothing
    pub exhausted: bool,
    /// nodes spent by the first reading (held) and by the second (touched)
    pub reads: Vec<(i32, u64)>,
}

#[derive(Clone, Copy)]
struct Entry { reach: u8, fail: i16 }

struct Cut;

struct Tree<'a, G: Game + Placed, J: Judge> {
    g: &'a G,
    st: &'a Static,
    me: Role,
    judge: &'a J,
    goals: Vec<Statement>,
    reader: FactReader,
    keeper: GraphKeeper,
    history: Arc<Seen>,
    path: Path,
    nodes: u64,
    fuse: u64,
    deadline: Option<Instant>,
    narrow: Narrow,
    table: HashMap<(u64, u16), Entry>,
    holds: HashMap<u64, u64>,
    makes: u64,
    matches: u64,
    horizon: bool,
    taint: bool,
    /// the places and kinds the goals' literals name, by goal (for the order)
    names: Vec<Vec<String>>,
    place_names: Vec<String>,
    kind_names: Vec<String>,
}

fn consts(pattern: &[Lit]) -> Vec<String> { pattern.iter().flat_map(|l| l.args.iter().filter_map(|a| if let Arg::Const(c) = a { Some(c.clone()) } else { None })).collect() }

/// The table's index of a goal under a reading: the first reading by goal; the second reading by goal and by whether the position was made by a move of the side (the leaf of the second reading depends on it).
fn tix(goal: usize, touch: bool, after_mine: bool) -> u16 { goal as u16 + if !touch { 0 } else if after_mine { 128 } else { 64 } }

impl<'a, G: Game + Placed, J: Judge> Tree<'a, G, J> {
    fn new(a: &Ask<'a, G, J>) -> Tree<'a, G, J> {
        assert!(a.goals.len() <= MOST_GOALS, "one tree serves at most {MOST_GOALS} goals");
        let all: Vec<Lit> = a.goals.iter().flat_map(|s| s.pattern.iter().cloned()).collect();
        Tree {
            g: a.game, st: a.board, me: a.side, judge: a.judge, goals: a.goals.to_vec(), reader: FactReader::new(a.board, Some(&all)), keeper: GraphKeeper::new(a.board, a.game, a.state), history: a.history.clone(), path: Path::new(),
            nodes: 0, fuse: a.nodes, deadline: a.seconds.map(|s| Instant::now() + std::time::Duration::from_secs_f64(s.max(0.0))), narrow: a.narrow, table: HashMap::new(), holds: HashMap::new(), makes: 0, matches: 0, horizon: false, taint: false,
            names: a.goals.iter().map(|s| consts(&s.pattern)).collect(), place_names: a.game.place_names(), kind_names: a.game.kind_names(),
        }
    }

    fn spend(&mut self) -> Result<(), Cut> {
        self.nodes += 1;
        if self.nodes > self.fuse { return Err(Cut); }
        if self.nodes & 63 == 0 { if let Some(d) = self.deadline { if Instant::now() >= d { return Err(Cut); } } }
        Ok(())
    }

    /// The goals whose pattern binds on the graph as it now stands.
    fn mask_now(&mut self) -> u64 {
        self.matches += 1;
        let rows = self.reader.rows(self.keeper.live());
        let idx = FactIndex::new(&rows);
        let mut m = 0u64;
        for (i, s) in self.goals.iter().enumerate() {
            let mut steps = 0u64;
            if !s.bindings(&idx, 1, &mut steps).is_empty() { m |= 1 << i; }
        }
        m
    }

    /// The goals whose pattern binds at a child of the position the keeper holds (made and taken back; kept by position when the table is on).
    fn bind_mask_of(&mut self, q: &G::State) -> u64 {
        let key = self.g.key(q);
        if self.narrow.table { if let Some(m) = self.holds.get(&key) { return *m; } }
        self.keeper.make(self.st, self.g, q);
        self.makes += 1;
        let m = self.mask_now();
        self.keeper.undo();
        if self.narrow.table { self.holds.insert(key, m); }
        m
    }

    /// The goals held at `s` given the goals that bind there: all that bind if the side is to move or the game is over, else none.
    fn held(&self, s: &G::State, bound: u64) -> u64 { if self.g.outcome(s).is_some() || self.g.mover(s) == Some(self.me) { bound } else { 0 } }

    /// The moves of `s` with the states they make, in the order read: the game's own, or (narrowed) those the first open goal names first.
    fn children(&self, s: &G::State, steer: Option<usize>) -> Vec<(usize, G::Move, G::State)> {
        let mut v: Vec<(usize, i32, G::Move)> = self.g.moves(s).into_iter().enumerate().map(|(i, m)| (i, 0, m)).collect();
        if self.narrow.order {
            if let Some(gi) = steer {
                let names = &self.names[gi];
                for e in v.iter_mut() {
                    let (from, to) = self.g.move_ends(s, &e.2);
                    let (fp, tp) = (&self.place_names[from as usize], &self.place_names[to as usize]);
                    let (mover_kind, took) = self.g.move_info(s, &e.2);
                    let mut sc = 0;
                    if names.iter().any(|n| n == tp) { sc += 4; }
                    if names.iter().any(|n| n == fp) { sc += 2; }
                    if took.is_some() { sc += 2; }
                    if names.iter().any(|n| *n == self.kind_names[mover_kind as usize]) { sc += 1; }
                    e.1 = -sc;
                }
                v.sort_by_key(|e| (e.1, e.0));
            }
        }
        v.into_iter().map(|(i, _, m)| { let q = self.g.apply(s, &m); (i, m, q) }).collect()
    }

    /// The answers for the goals of `open` at `s` (the keeper holds the parent of `s`, or `s` itself at the root: `at_root`), `depth` plies left. `res[g]` is the plies to reach g, or NONE; goals not asked are left NONE. `after_mine`: `s` was made by a move of the side asking.
    /// `touch` selects the reading: false, a goal is reached where it is held; true, it is also met where it binds after a move of the side and a reply unbinds it.
    #[allow(clippy::too_many_arguments)]
    fn go(&mut self, s: &G::State, depth: u8, open: u64, ply: i32, at_root: bool, after_mine: bool, touch: bool, res: &mut [u8]) -> Result<(), Cut> {
        self.spend()?;
        let key = self.g.key(s);
        let mut made = at_root;
        let cached = if self.narrow.table { self.holds.get(&key).copied() } else { None };
        let mask = match cached { Some(m) => m, None => {
            if !made { self.keeper.make(self.st, self.g, s); self.makes += 1; made = true; }
            let m = self.mask_now();
            if self.narrow.table { self.holds.insert(key, m); }
            m
        } };
        let undo = |t: &mut Self, made_here: bool| { if made_here && !at_root { t.keeper.undo(); } };
        let made_here = made && !at_root;
        let hit = self.held(s, mask);
        for i in 0..self.goals.len() { if open & (1 << i) != 0 && hit & (1 << i) != 0 { res[i] = 0; } }
        let mut left = open & !hit;
        if left == 0 { undo(self, made_here); return Ok(()); }
        if self.g.outcome(s).is_some() { undo(self, made_here); return Ok(()); }
        if ply > 0 {
            if self.history.contains(&key) { undo(self, made_here); return Ok(()); }
            if self.path.contains(&key) { self.taint = true; undo(self, made_here); return Ok(()); }
        }
        if depth == 0 { self.horizon = true; undo(self, made_here); return Ok(()); }
        if self.narrow.table {
            for i in 0..self.goals.len() {
                if left & (1 << i) == 0 { continue; }
                if let Some(e) = self.table.get(&(key, tix(i, touch, after_mine))) {
                    if e.reach != NONE && e.reach <= depth { res[i] = e.reach; left &= !(1 << i); }
                    else if e.fail >= depth as i16 { left &= !(1 << i); }
                }
            }
            if left == 0 { undo(self, made_here); return Ok(()); }
        }
        if !made { self.keeper.make(self.st, self.g, s); self.makes += 1; made = true; }
        let made_here = made && !at_root;
        let taint_before = self.taint;
        self.taint = false;
        let steer = (0..self.goals.len()).find(|i| left & (1 << i) != 0);
        let kids = self.children(s, steer);
        let k = self.goals.len();
        // the second reading's own leaf: the pattern binds here after a move of the side (the other side to move, the game not over) and some reply leaves it unbound
        if touch && after_mine && ply >= 1 {
            let cand = left & mask;
            if cand != 0 {
                let mut unbound = 0u64;
                for (_, _, q) in &kids { unbound |= cand & !self.bind_mask_of(q); if unbound == cand { break; } }
                for i in 0..k { if unbound & (1 << i) != 0 { res[i] = 0; } }
                left &= !unbound;
                if left == 0 { self.taint |= taint_before; undo(self, made_here); return Ok(()); }
            }
        }
        let asked = left;
        let maxing = self.g.mover(s) == Some(self.me);
        self.path.push(key);
        let mut result: Result<(), Cut> = Ok(());
        if maxing {
            let mut todo = left;
            for (_, _, q) in &kids {
                let mut sub = vec![NONE; k];
                match self.go(q, depth - 1, todo, ply + 1, false, true, touch, &mut sub) { Ok(()) => {}, Err(c) => { result = Err(c); break; } }
                for i in 0..k { if todo & (1 << i) != 0 && sub[i] != NONE { res[i] = sub[i] + 1; todo &= !(1 << i); } }
                if todo == 0 { break; }
            }
            left = todo;
        } else {
            let mut alive = left;
            let mut worst = vec![0u8; k];
            if kids.is_empty() { alive = 0; }
            for (_, _, q) in &kids {
                let mut sub = vec![NONE; k];
                match self.go(q, depth - 1, alive, ply + 1, false, false, touch, &mut sub) { Ok(()) => {}, Err(c) => { result = Err(c); break; } }
                for i in 0..k { if alive & (1 << i) != 0 { if sub[i] == NONE { alive &= !(1 << i); } else { worst[i] = worst[i].max(sub[i] + 1); } } }
                if alive == 0 { break; }
            }
            if result.is_ok() { for i in 0..k { if alive & (1 << i) != 0 { res[i] = worst[i]; } } }
            left = if result.is_ok() { left & !alive } else { left };
        }
        self.path.pop();
        if made_here { self.keeper.undo(); }
        if result.is_ok() && self.narrow.table {
            for i in 0..k {
                if asked & (1 << i) == 0 { continue; }
                let ix = tix(i, touch, after_mine);
                if res[i] != NONE { self.table.entry((key, ix)).and_modify(|e| e.reach = e.reach.min(res[i])).or_insert(Entry { reach: res[i], fail: -1 }); }
                else if left & (1 << i) != 0 && !self.taint { self.table.entry((key, ix)).and_modify(|e| e.fail = e.fail.max(depth as i16)).or_insert(Entry { reach: NONE, fail: depth as i16 }); }
            }
        }
        self.taint |= taint_before;
        result
    }

    /// The extractions are not cut by the budget: the fuse and the clock are set aside and given back at their end.
    fn begin(&mut self) -> (u64, Option<Instant>) {
        let saved = (self.fuse, self.deadline);
        self.fuse = u64::MAX; self.deadline = None;
        saved
    }

    /// The line to goal `gi` from the root, walked through the answers: the side takes a move after which the goal is reached within the plies left; the other side's reply is the one that keeps it longest. Extraction is not cut by the budget.
    fn extract(&mut self, root: &G::State, gi: usize, depth: u8) -> Line {
        let saved = self.begin();
        let one = 1u64 << gi;
        let mut s = root.clone();
        let (mut names, mut taken, mut value, mut meets) = (vec![], vec![], 0i64, 0u64);
        let mut first = None;
        let mut left = depth;
        let (mut pushed, mut made, mut ply) = (0, 0, 0);
        loop {
            let mask = self.mask_now();
            let hit = self.held(&s, mask);
            meets |= hit;
            if left == 0 || hit & one != 0 { break; }
            let mover = self.g.mover(&s);
            let kids = { let ms: Vec<(usize, G::Move)> = self.g.moves(&s).into_iter().enumerate().collect(); ms };
            let mut pick: Option<(usize, G::Move, G::State, u8)> = None;
            let key = self.g.key(&s);
            self.path.push(key); pushed += 1;
            for (i, m) in kids {
                let q = self.g.apply(&s, &m);
                let mut sub = vec![NONE; self.goals.len()];
                let before = self.taint;
                self.go(&q, left - 1, one, ply + 1, false, mover == Some(self.me), false, &mut sub).ok();
                self.taint = before;
                if mover == Some(self.me) {
                    if sub[gi] != NONE { pick = Some((i, m, q, sub[gi])); break; }
                } else if sub[gi] != NONE && pick.as_ref().is_none_or(|p| sub[gi] > p.3) { pick = Some((i, m, q, sub[gi])); }
            }
            let Some((i, m, q, _)) = pick else { break };
            if first.is_none() { first = Some(i); }
            names.push(self.g.move_name(&s, &m));
            taken.extend(self.g.captured(&s, &m));
            let moved = mover.unwrap_or(self.me);
            value += self.g.captured(&s, &m).iter().map(|(o, k)| self.judge.delta(self.me, *o, k)).sum::<i64>() + self.g.changed(&s, &m).iter().map(|(a, b)| self.judge.change(self.me, moved, a, b)).sum::<i64>();
            self.keeper.make(self.st, self.g, &q); made += 1;
            s = q; left -= 1; ply += 1;
        }
        let bindings = self.bindings_here(gi);
        for _ in 0..made { self.keeper.undo(); }
        for _ in 0..pushed { self.path.pop(); }
        self.fuse = saved.0; self.deadline = saved.1;
        Line { first, plies: names.len() as i32, names, bindings, taken, value, meets }
    }

    /// The bindings of goal `gi` at the position the keeper now holds.
    fn bindings_here(&mut self, gi: usize) -> Vec<Binding> {
        let rows = self.reader.rows(self.keeper.live());
        let idx = FactIndex::new(&rows);
        let mut steps = 0u64;
        if self.goals[gi].pattern.is_empty() { vec![Binding::new()] } else { cs_rep::matcher::bindings_fused(&self.goals[gi].pattern, &Binding::new(), &idx, BINDINGS, &mut steps, u64::MAX).0 }
    }

    /// The line to goal `gi` as a goal TOUCHED, from the root: the walk keeps to positions where the goal is not reached within the plies left yet is met by the second reading (so the goal is not reached and is touched), the side taking a move after which the second reading
    /// meets it, the other side's reply one after which the goal is not reached (one exists wherever the goal is not reached, and the second reading met it after every reply). It ends at the position where the pattern binds after a move of the side, and the reply that unbinds it is named. Extraction is not cut by the budget.
    fn extract_touch(&mut self, root: &G::State, gi: usize, depth: u8) -> Touch {
        let saved = self.begin();
        let one = 1u64 << gi;
        let mut s = root.clone();
        let (mut names, mut taken, mut value) = (vec![], vec![], 0i64);
        let mut first = None;
        let mut left = depth;
        let (mut pushed, mut made, mut ply) = (0, 0, 0);
        let mut after_mine = false;
        let mut reply = String::new();
        loop {
            let mask = self.mask_now();
            let mover = self.g.mover(&s);
            if ply >= 1 && after_mine && left >= 1 && mover != Some(self.me) && self.g.outcome(&s).is_none() && mask & one != 0 {
                let found = self.g.moves(&s).into_iter().find(|m| { let q = self.g.apply(&s, m); self.bind_mask_of(&q) & one == 0 }).map(|m| self.g.move_name(&s, &m));
                if let Some(n) = found { reply = n; break; }
            }
            if left == 0 || self.g.outcome(&s).is_some() { break; }
            let kids = { let ms: Vec<(usize, G::Move)> = self.g.moves(&s).into_iter().enumerate().collect(); ms };
            let mut pick: Option<(usize, G::Move, G::State)> = None;
            let key = self.g.key(&s);
            self.path.push(key); pushed += 1;
            for (i, m) in kids {
                let q = self.g.apply(&s, &m);
                let (mut sub_t, mut sub_r) = (vec![NONE; self.goals.len()], vec![NONE; self.goals.len()]);
                let before = self.taint;
                self.go(&q, left - 1, one, ply + 1, false, mover == Some(self.me), true, &mut sub_t).ok();
                if sub_t[gi] != NONE && mover != Some(self.me) { self.go(&q, left - 1, one, ply + 1, false, false, false, &mut sub_r).ok(); }
                self.taint = before;
                if sub_t[gi] != NONE && (mover == Some(self.me) || sub_r[gi] == NONE) { pick = Some((i, m, q)); break; }
            }
            let Some((i, m, q)) = pick else { break };
            if first.is_none() { first = Some(i); }
            names.push(self.g.move_name(&s, &m));
            taken.extend(self.g.captured(&s, &m));
            let moved = mover.unwrap_or(self.me);
            value += self.g.captured(&s, &m).iter().map(|(o, k)| self.judge.delta(self.me, *o, k)).sum::<i64>() + self.g.changed(&s, &m).iter().map(|(a, b)| self.judge.change(self.me, moved, a, b)).sum::<i64>();
            self.keeper.make(self.st, self.g, &q); made += 1;
            after_mine = mover == Some(self.me);
            s = q; left -= 1; ply += 1;
        }
        let bindings = self.bindings_here(gi);
        for _ in 0..made { self.keeper.undo(); }
        for _ in 0..pushed { self.path.pop(); }
        self.fuse = saved.0; self.deadline = saved.1;
        Touch { line: Line { first, plies: names.len() as i32, names, bindings, taken, value, meets: 0 }, reply }
    }
}

/// THE BODY (see the file's doc).
pub fn read<G: Game + Placed, J: Judge>(a: &Ask<G, J>) -> Read {
    let k = a.goals.len();
    let mut t = Tree::new(a);
    let mut outcomes: Vec<Option<Outcome>> = vec![None; k];
    let mut found_at = vec![NONE; k];
    let mut touch_at = vec![NONE; k];
    let mut depth_done = 0;
    let mut cut = false;
    let mut exhausted = false;
    let all: u64 = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let mut open = all;
    // THE FIRST READING: the goals held. A goal held at the root is reached in nought plies, so the loop starts at depth 0 and reads it the same way
    for d in 0..=(a.max_depth.clamp(0, 254)) {
        if open == 0 { break; }
        t.horizon = false;
        t.taint = false;
        let mut res = vec![NONE; k];
        let r = t.go(a.state, d as u8, open, 0, true, false, false, &mut res);
        match r {
            Ok(()) => {
                depth_done = d;
                for i in 0..k { if open & (1 << i) != 0 && res[i] != NONE { found_at[i] = d as u8; outcomes[i] = Some(Outcome::Unsettled { depth: d }); open &= !(1 << i); } }
                if !t.horizon { exhausted = true; break; }
            }
            Err(Cut) => {
                cut = true;
                break;
            }
        }
    }
    let first_nodes = t.nodes;
    // THE SECOND READING, for the goals the first settled as not reached: held or touched, to the depth the first read in full
    let mut cut2 = false;
    if !cut && open != 0 {
        let mut open2 = open;
        for d in 0..=depth_done {
            if open2 == 0 { break; }
            t.horizon = false;
            t.taint = false;
            let mut res = vec![NONE; k];
            match t.go(a.state, d as u8, open2, 0, true, false, true, &mut res) {
                Ok(()) => { for i in 0..k { if open2 & (1 << i) != 0 && res[i] != NONE { touch_at[i] = d as u8; open2 &= !(1 << i); } } }
                Err(Cut) => { cut2 = true; break; }
            }
        }
    }
    // the line of every reached goal, and of every goal touched
    let mut done: Vec<Option<Outcome>> = vec![None; k];
    for i in 0..k {
        if found_at[i] != NONE { done[i] = Some(Outcome::Reached(t.extract(a.state, i, found_at[i]))); }
        else if touch_at[i] != NONE { done[i] = Some(Outcome::Touched(t.extract_touch(a.state, i, touch_at[i]))); }
    }
    let outcomes = (0..k).map(|i| match done[i].take() {
        Some(o) => o,
        None => if cut || cut2 { Outcome::Unsettled { depth: depth_done } } else { Outcome::Out { depth: if exhausted { a.max_depth } else { depth_done } } },
    }).collect();
    Read { outcomes, nodes: t.nodes, makes: t.makes, matches: t.matches, depth: depth_done, cut: cut || cut2, exhausted, reads: vec![(0, first_nodes), (1, t.nodes - first_nodes)] }
}

/// Whether one reply of the other side leaves both goals out of reach: read in full from the position after `first`, a state of the line (the other side to move), `plies_left` plies to read. None where the budget cut the reading.
pub fn one_reply_meets<G: Game + Placed, J: Judge>(a: &Ask<G, J>, after_first: &G::State, goals: (usize, usize), plies_left: i32) -> (Option<bool>, u64) {
    let mut t = Tree::new(&Ask { game: a.game, board: a.board, state: after_first, judge: a.judge, side: a.side, goals: a.goals, nodes: a.nodes, seconds: a.seconds, max_depth: a.max_depth, history: a.history.clone(), narrow: a.narrow });
    let both = (1u64 << goals.0) | (1u64 << goals.1);
    let k = a.goals.len();
    let mut meets = false;
    for m in a.game.moves(after_first) {
        let q = a.game.apply(after_first, &m);
        let mut sub = vec![NONE; k];
        t.keeper.make(t.st, t.g, &q);
        let r = t.go(&q, (plies_left - 1).max(0) as u8, both, 1, true, false, false, &mut sub);
        if r.is_err() { return (None, t.nodes); }
        if sub[goals.0] == NONE && sub[goals.1] == NONE { meets = true; }
        t.keeper.undo();
        // `at_root` made the keeper hold the child itself: take it back by hand (the keeper was made above)
    }
    (Some(meets), t.nodes)
}
