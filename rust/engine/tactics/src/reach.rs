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
//!
//! PLAN 79.32 (SYSTEM_SPEC 11 "The pure search is mathematics and lives in layer 0; what steers it does not", 13 item 50; piece: the tactical reasoner as a case of the base). THE WALK IS NOT HERE: the and-or recursion, the answers kept by position and the marks of a deepening are layer 0's (`cs_base::choice::walk::Walk`), one copy for every game.
//! What this file hands it: `Position`, the node (layer 2's position walked in place on the live graph, made on first need and taken back by `leave`; the bound patterns kept by position), and `Policy`, what steers (the budget in nodes and the clock, the count of calls, the order of moves, whether answers are kept).
//! What stays here besides: the goals, the deepening over depths, the lines of the answers (`Tree::extract`, `Tree::extract_touch`: a line walked by hand with the walk asked at each candidate move) and the pair reading (`one_reply_meets`).
use crate::ledger::Judge;
use cs_game::graph::Static;
use cs_game::keeper::GraphKeeper;
use cs_game::placed::Placed;
use cs_base::choice::walk::{Ask as Open, Cut, Look, Node, Out, Space, Steer, Walk};
use cs_game::{Game, Role};
use cs_game::graph_facts::FactReader;
use cs_rep::history::Seen;
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
    /// where the calls went (plan 79.29 part A)
    pub diag: Diag,
}

/// PLAN 79.29 PART A: where the calls of the one tree go, by what they are asked: the goals held only, the goals touched only (already known to bind after a move; the other answer is still asked), or both.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Diag { pub calls_held: u64, pub calls_touched: u64, pub calls_both: u64 }

/// THE NODE (layer 2's position, walked in place on the live graph). It is layer 0's `Space`: the stack holds the positions of the line being walked, each with its key and whether it was made on the live graph (`made`: made on first need, taken back by `leave`, so a position
/// whose answer is kept is never built). The bound patterns of a position are kept by its key when the table is on.
struct Position<'a, G: Game + Placed> {
    g: &'a G,
    st: &'a Static,
    me: Role,
    goals: Vec<Statement>,
    reader: FactReader,
    keeper: GraphKeeper,
    history: Arc<Seen>,
    stack: Vec<Frame<G::State>>,
    /// the goals whose pattern binds, by position (kept when the table is on)
    holds: HashMap<u64, u64>,
    table: bool,
    makes: u64,
    matches: u64,
}

struct Frame<S> { state: S, key: u64, made: bool }

impl<'a, G: Game + Placed> Position<'a, G> {
    fn top(&self) -> &Frame<G::State> { self.stack.last().expect("a position is current") }

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

    /// Make the current position live (counted) if it is not.
    fn make_top(&mut self) {
        let f = self.stack.last_mut().expect("a position is current");
        if !f.made { self.keeper.make(self.st, self.g, &f.state); self.makes += 1; f.made = true; }
    }

    /// The goals whose pattern binds at the current position (made on first need, kept by position when the table is on).
    fn bound(&mut self) -> u64 {
        let key = self.top().key;
        if self.table { if let Some(m) = self.holds.get(&key) { return *m; } }
        self.make_top();
        let m = self.mask_now();
        if self.table { self.holds.insert(key, m); }
        m
    }

    /// The goals held at `s` given the goals that bind there: all that bind if the side is to move or the game is over, else none.
    fn held_of(&self, s: &G::State, bound: u64) -> u64 { if self.g.outcome(s).is_some() || self.g.turn(s) == self.me { bound } else { 0 } }

    /// The goals that bind at the position after `m` from the current one (made and taken back).
    fn bound_after(&mut self, m: &G::Move) -> u64 { self.enter(m); let b = self.bound(); self.leave(); b }

    /// Step into `m` and make that position live at once, not counted (the extractions walk a line by hand).
    fn step_live(&mut self, m: &G::Move) {
        self.enter(m);
        let f = self.stack.last_mut().expect("a position is current");
        self.keeper.make(self.st, self.g, &f.state);
        f.made = true;
    }

    /// The bindings of goal `gi` at the position the keeper now holds.
    fn bindings_here(&mut self, gi: usize) -> Vec<Binding> {
        let rows = self.reader.rows(self.keeper.live());
        let idx = FactIndex::new(&rows);
        let mut steps = 0u64;
        if self.goals[gi].pattern.is_empty() { vec![Binding::new()] } else { cs_rep::matcher::bindings_fused(&self.goals[gi].pattern, &Binding::new(), &idx, BINDINGS, &mut steps, u64::MAX).0 }
    }
}

impl<'a, G: Game + Placed> Node for Position<'a, G> {
    type Opt = G::Move;

    fn own_turn(&self) -> bool { self.g.mover(&self.top().state) == Some(self.me) }

    fn options(&mut self, out: &mut Vec<G::Move>) { self.make_top(); *out = self.g.moves(&self.top().state); }

    fn enter(&mut self, m: &G::Move) {
        let q = self.g.apply(&self.top().state, m);
        let key = self.g.key(&q);
        self.stack.push(Frame { state: q, key, made: false });
    }

    fn leave(&mut self) {
        let f = self.stack.pop().expect("a position is current");
        if f.made { self.keeper.undo(); }
    }
}

impl<'a, G: Game + Placed> Space for Position<'a, G> {
    const CYCLES: bool = true;

    fn key(&self) -> u64 { self.top().key }

    fn look(&mut self) -> Look {
        let bound = self.bound();
        let s = &self.top().state;
        let ended = self.g.outcome(s).is_some();
        let held = if ended || self.g.turn(s) == self.me { bound } else { 0 };
        Look { bound, held, ended }
    }

    fn met_before(&self, key: u64) -> bool { self.history.contains(&key) }

    fn breaks(&mut self, cand: u64, opts: &[G::Move]) -> u64 {
        let mut unbound = 0u64;
        for m in opts { unbound |= cand & !self.bound_after(m); if unbound == cand { break; } }
        unbound
    }
}

/// WHAT STEERS THE WALK, layer 4's: the budget in nodes and the clock (a cut is a gap), the count of calls by what they asked, and the order of moves (the narrowing: the moves that arrive on a place the first open goal's pattern names, that leave one, that take a unit, that are of a kind it
/// names, first; the game's own order breaks ties). An order is never a verdict.
struct Policy<'a, G: Game + Placed> {
    g: &'a G,
    narrow: Narrow,
    nodes: u64,
    fuse: u64,
    deadline: Option<Instant>,
    diag: Diag,
    /// the places and kinds the goals' literals name, by goal (for the order)
    names: Vec<Vec<String>>,
    place_names: Vec<String>,
    kind_names: Vec<String>,
    /// the scores of the moves being ordered (kept for its room)
    scored: Vec<(i32, G::Move)>,
}

impl<'a, G: Game + Placed> Steer<Position<'a, G>> for Policy<'a, G> {
    fn spend(&mut self, asked: Open) -> Result<(), Cut> {
        self.nodes += 1;
        if self.nodes > self.fuse { return Err(Cut); }
        if self.nodes & 63 == 0 { if let Some(d) = self.deadline { if Instant::now() >= d { return Err(Cut); } } }
        match (asked.strict != 0, asked.loose != 0) { (true, true) => self.diag.calls_both += 1, (true, false) => self.diag.calls_held += 1, _ => self.diag.calls_touched += 1 }
        Ok(())
    }

    fn order(&mut self, space: &Position<'a, G>, opts: &mut Vec<G::Move>, first: usize) {
        if !self.narrow.order { return; }
        let s = &space.top().state;
        let names = &self.names[first];
        self.scored.clear();
        for m in opts.drain(..) {
            let (from, to) = self.g.move_ends(s, &m);
            let (fp, tp) = (&self.place_names[from as usize], &self.place_names[to as usize]);
            let (mover_kind, took) = self.g.move_info(s, &m);
            let mut sc = 0;
            if names.iter().any(|n| n == tp) { sc += 4; }
            if names.iter().any(|n| n == fp) { sc += 2; }
            if took.is_some() { sc += 2; }
            if names.iter().any(|n| *n == self.kind_names[mover_kind as usize]) { sc += 1; }
            self.scored.push((-sc, m));
        }
        // a stable sort: moves of one score keep the game's order
        self.scored.sort_by_key(|e| e.0);
        opts.extend(self.scored.drain(..).map(|e| e.1));
    }

    fn keeps(&self) -> bool { self.narrow.table }
}

impl<'a, G: Game + Placed> Policy<'a, G> {
    /// The extractions are not cut by the budget: the fuse and the clock are set aside and given back at their end.
    fn begin(&mut self) -> (u64, Option<Instant>) {
        let saved = (self.fuse, self.deadline);
        self.fuse = u64::MAX; self.deadline = None;
        saved
    }
}

/// The walk of layer 0 given layer 4's node and steering, and the judge for what a line costs.
struct Tree<'a, G: Game + Placed, J: Judge> {
    w: Walk<Position<'a, G>, Policy<'a, G>, u8>,
    judge: &'a J,
}

fn consts(pattern: &[Lit]) -> Vec<String> { pattern.iter().flat_map(|l| l.args.iter().filter_map(|a| if let Arg::Const(c) = a { Some(c.clone()) } else { None })).collect() }

impl<'a, G: Game + Placed, J: Judge> Tree<'a, G, J> {
    fn new(a: &Ask<'a, G, J>) -> Tree<'a, G, J> {
        assert!(a.goals.len() <= MOST_GOALS, "one tree serves at most {MOST_GOALS} goals");
        let all: Vec<Lit> = a.goals.iter().flat_map(|s| s.pattern.iter().cloned()).collect();
        let root = Frame { state: a.state.clone(), key: a.game.key(a.state), made: true };
        let space = Position { g: a.game, st: a.board, me: a.side, goals: a.goals.to_vec(), reader: FactReader::new(a.board, Some(&all)), keeper: GraphKeeper::new(a.board, a.game, a.state), history: a.history.clone(), stack: vec![root], holds: HashMap::new(), table: a.narrow.table, makes: 0, matches: 0 };
        let steer = Policy {
            g: a.game, narrow: a.narrow, nodes: 0, fuse: a.nodes, deadline: a.seconds.map(|s| Instant::now() + std::time::Duration::from_secs_f64(s.max(0.0))), diag: Diag::default(),
            names: a.goals.iter().map(|s| consts(&s.pattern)).collect(), place_names: a.game.place_names(), kind_names: a.game.kind_names(), scored: Vec::new(),
        };
        Tree { w: Walk::new(space, steer), judge: a.judge }
    }

    /// The line to goal `gi` from the root, walked through the answers: the side takes a move after which the goal is reached within the plies left; the other side's reply is the one that keeps it longest. Extraction is not cut by the budget.
    fn extract(&mut self, gi: usize, depth: u8) -> Line {
        let saved = self.w.steer.begin();
        let (g, me) = (self.w.space.g, self.w.space.me);
        let one = 1u64 << gi;
        let (mut names, mut taken, mut value, mut meets) = (vec![], vec![], 0i64, 0u64);
        let mut first = None;
        let mut left = depth;
        let (mut pushed, mut made) = (0, 0);
        loop {
            let s = self.w.space.top().state.clone();
            let mask = self.w.space.mask_now();
            let hit = self.w.space.held_of(&s, mask);
            meets |= hit;
            if left == 0 || hit & one != 0 { break; }
            let mover = Some(g.turn(&s));
            let kids: Vec<(usize, G::Move)> = g.moves(&s).into_iter().enumerate().collect();
            let mut pick: Option<(usize, G::Move, u8)> = None;
            let key = g.key(&s);
            self.w.line_push(key); pushed += 1;
            for (i, m) in kids {
                let mut sub = Out::<u8>::none();
                let before = self.w.taint;
                self.w.space.enter(&m);
                self.w.go(left - 1, Open { strict: one, loose: 0 }, true, mover == Some(me), &mut sub).ok();
                self.w.space.leave();
                self.w.taint = before;
                if mover == Some(me) {
                    if sub.strict[gi] != NONE { pick = Some((i, m, sub.strict[gi])); break; }
                } else if sub.strict[gi] != NONE && pick.as_ref().is_none_or(|p| sub.strict[gi] > p.2) { pick = Some((i, m, sub.strict[gi])); }
            }
            let Some((i, m, _)) = pick else { break };
            if first.is_none() { first = Some(i); }
            names.push(g.move_name(&s, &m));
            taken.extend(g.captured(&s, &m));
            let moved = mover.unwrap_or(me);
            value += g.captured(&s, &m).iter().map(|(o, k)| self.judge.delta(me, *o, k)).sum::<i64>() + g.changed(&s, &m).iter().map(|(a, b)| self.judge.change(me, moved, a, b)).sum::<i64>();
            self.w.space.step_live(&m); made += 1;
            left -= 1;
        }
        let bindings = self.w.space.bindings_here(gi);
        for _ in 0..made { self.w.space.leave(); }
        for _ in 0..pushed { self.w.line_pop(); }
        self.w.steer.fuse = saved.0; self.w.steer.deadline = saved.1;
        Line { first, plies: names.len() as i32, names, bindings, taken, value, meets }
    }

    /// The line to goal `gi` as a goal TOUCHED, from the root: the walk keeps to positions where the goal is not reached within the plies left yet is met by the second reading (so the goal is not reached and is touched), the side taking a move after which the second reading
    /// meets it, the other side's reply one after which the goal is not reached (one exists wherever the goal is not reached, and the second reading met it after every reply). It ends at the position where the pattern binds after a move of the side, and the reply that unbinds it is named. Extraction is not cut by the budget.
    fn extract_touch(&mut self, gi: usize, depth: u8) -> Touch {
        let saved = self.w.steer.begin();
        let (g, me) = (self.w.space.g, self.w.space.me);
        let one = 1u64 << gi;
        let (mut names, mut taken, mut value) = (vec![], vec![], 0i64);
        let mut first = None;
        let mut left = depth;
        let (mut pushed, mut made, mut ply) = (0, 0, 0);
        let mut after_mine = false;
        let mut reply = String::new();
        loop {
            let s = self.w.space.top().state.clone();
            let mask = self.w.space.mask_now();
            let mover = Some(g.turn(&s));
            if ply >= 1 && after_mine && left >= 1 && mover != Some(me) && g.outcome(&s).is_none() && mask & one != 0 {
                let found = g.moves(&s).into_iter().find(|m| self.w.space.bound_after(m) & one == 0).map(|m| g.move_name(&s, &m));
                if let Some(n) = found { reply = n; break; }
            }
            if left == 0 || g.outcome(&s).is_some() { break; }
            let kids: Vec<(usize, G::Move)> = g.moves(&s).into_iter().enumerate().collect();
            let mut pick: Option<(usize, G::Move)> = None;
            let key = g.key(&s);
            self.w.line_push(key); pushed += 1;
            for (i, m) in kids {
                let mut sub_r = Out::<u8>::none();
                let before = self.w.taint;
                // the one reading asks both answers where the reply is read (the goal must be met and not reached), the second alone where the side moves
                let ask_held = if mover != Some(me) { one } else { 0 };
                self.w.space.enter(&m);
                self.w.go(left - 1, Open { strict: ask_held, loose: one }, true, mover == Some(me), &mut sub_r).ok();
                self.w.space.leave();
                self.w.taint = before;
                if sub_r.loose[gi] != NONE && (mover == Some(me) || sub_r.strict[gi] == NONE) { pick = Some((i, m)); break; }
            }
            let Some((i, m)) = pick else { break };
            if first.is_none() { first = Some(i); }
            names.push(g.move_name(&s, &m));
            taken.extend(g.captured(&s, &m));
            let moved = mover.unwrap_or(me);
            value += g.captured(&s, &m).iter().map(|(o, k)| self.judge.delta(me, *o, k)).sum::<i64>() + g.changed(&s, &m).iter().map(|(a, b)| self.judge.change(me, moved, a, b)).sum::<i64>();
            self.w.space.step_live(&m); made += 1;
            after_mine = mover == Some(me);
            left -= 1; ply += 1;
        }
        let bindings = self.w.space.bindings_here(gi);
        for _ in 0..made { self.w.space.leave(); }
        for _ in 0..pushed { self.w.line_pop(); }
        self.w.steer.fuse = saved.0; self.w.steer.deadline = saved.1;
        Touch { line: Line { first, plies: names.len() as i32, names, bindings, taken, value, meets: 0 }, reply }
    }
}

/// THE BODY (see the file's doc).
pub fn read<G: Game + Placed, J: Judge>(a: &Ask<G, J>) -> Read {
    let k = a.goals.len();
    let mut t = Tree::new(a);
    let mut found_at = vec![NONE; k];
    let mut touch_at = vec![NONE; k];
    let mut depth_done = 0;
    let mut cut = false;
    let mut exhausted = false;
    let all: u64 = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    // the goals still open for reached (held), and for touched (held or touched: until it is met once)
    let (mut open_h, mut open_t) = (all, all);
    // ONE READING, deepened a ply at a time: it asks both answers of the goals open. A goal held at the root is reached in nought plies, so the loop starts at depth 0. A goal is touched only if it is not reached within the depth the reading finishes: a goal met by the second answer at some depth
    // is touched when the loop ends having read in full and not reached it.
    for d in 0..=(a.max_depth.clamp(0, 254)) {
        if open_h == 0 { break; }
        t.w.horizon = false;
        t.w.taint = 0;
        let mut out = Out::<u8>::none();
        match t.w.go(d as u8, Open { strict: open_h, loose: open_t }, false, false, &mut out) {
            Ok(()) => {
                depth_done = d;
                for i in 0..k {
                    let b = 1u64 << i;
                    if open_h & b != 0 && out.strict[i] != NONE { found_at[i] = d as u8; open_h &= !b; open_t &= !b; }
                    if open_t & b != 0 && out.loose[i] != NONE { touch_at[i] = d as u8; open_t &= !b; }
                }
                if !t.w.horizon { exhausted = true; break; }
            }
            Err(Cut) => { cut = true; break; }
        }
    }
    // the line of every reached goal, and of every goal touched
    let mut done: Vec<Option<Outcome>> = vec![None; k];
    for i in 0..k {
        if found_at[i] != NONE { done[i] = Some(Outcome::Reached(t.extract(i, found_at[i]))); }
        else if !cut && touch_at[i] != NONE { done[i] = Some(Outcome::Touched(t.extract_touch(i, touch_at[i]))); }
    }
    let outcomes = (0..k).map(|i| match done[i].take() {
        Some(o) => o,
        None => if cut { Outcome::Unsettled { depth: depth_done } } else { Outcome::Out { depth: if exhausted { a.max_depth } else { depth_done } } },
    }).collect();
    Read { outcomes, nodes: t.w.steer.nodes, makes: t.w.space.makes, matches: t.w.space.matches, depth: depth_done, cut, exhausted, diag: t.w.steer.diag }
}

/// Whether one reply of the other side leaves both goals out of reach: read in full from the position after `first`, a state of the line (the other side to move), `plies_left` plies to read. None where the budget cut the reading.
pub fn one_reply_meets<G: Game + Placed, J: Judge>(a: &Ask<G, J>, after_first: &G::State, goals: (usize, usize), plies_left: i32) -> (Option<bool>, u64) {
    let mut t = Tree::new(&Ask { game: a.game, board: a.board, state: after_first, judge: a.judge, side: a.side, goals: a.goals, nodes: a.nodes, seconds: a.seconds, max_depth: a.max_depth, history: a.history.clone(), narrow: a.narrow });
    let both = (1u64 << goals.0) | (1u64 << goals.1);
    let mut meets = false;
    for m in a.game.moves(after_first) {
        let mut sub = Out::<u8>::none();
        // the reply is made on the live graph by hand and read as the start of its own walk (not counted)
        t.w.space.step_live(&m);
        let r = t.w.go((plies_left - 1).max(0) as u8, Open { strict: both, loose: 0 }, true, false, &mut sub);
        if r.is_err() { return (None, t.w.steer.nodes); }
        if sub.strict[goals.0] == NONE && sub.strict[goals.1] == NONE { meets = true; }
        t.w.space.leave();
    }
    (Some(meets), t.w.steer.nodes)
}
