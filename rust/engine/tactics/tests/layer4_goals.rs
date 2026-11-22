//! PLAN 79.27 PART C AND PLAN 79.28 PART D: THE TEST OF LAYER 4 WITH GOALS (SYSTEM_SPEC 11 "Layer 4 takes a goal as an input", its amendment, "Reached means the goal outlasts the reply", "a layer's test uses only that layer and what is below it"; ledger rows 36, 14; piece: tactical reasoner).
//! It uses layer 1 (cs_rep: statements, the board), layer 2 (cs_game: the game through the text door, `Placed`, the board) and this layer's one entry (`meansends::ask`). Every run is bounded by NODES, never by a clock. The goals are DATA (tests/fixtures/goals.fixture, made by
//! discovery/layer4_goals_print.py: positions of two games with lists of patterns that are neither a gain nor the end) and tests/fixtures/pairs.fixture (positions where one line reaches two goals, MADE BY THE REFERENCE below: run this file with LAYER4_MAKE_PAIRS=1 to write it).
//! THE MEANING (plan 79.28). A pattern is HELD at a position where it binds and the side asking is to move, or the game is over. REACHED: on every reply a position where it is held is met within the depth (the position asked counts). TOUCHED: not reached, but the side can force the pattern to bind right after a move of its own and
//! some reply then unbinds it. OUT OF REACH: neither. NOT SETTLED: the budget cut the reading.
//! THE REFERENCE is written apart, here: a slow plain search of every move (no memo, no order, no live graph: the facts of each position are made from nothing out of the game's own questions) with a naive matcher of its own, to the same meaning. It answers each goal ALONE: reached in n plies, touched, or neither within the depth.
//! THE TESTS: the layer never says reached where the reference does not, with a line the rules replay to a position where the pattern is held; never touched where the reference does not say touched, with a line and a reply the rules replay; never out of reach where the reference reaches or touches; at the top budget it says exactly what the reference does;
//! the same answer asked twice; the list never contradicts a goal asked alone; the narrowing answers as the plain tree where both settle; the two old goals answer as the ledger search did (list of one); counts at three budgets are kept as floors. This file names no object of any game.
use cs_game::graph::Static;
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::rules::{Ruled, StateText};
use cs_game::{Fact, Game, Role};
use cs_rep::history::Seen;
use cs_rep::lemma::{Arg, Lit, LitJson};
use cs_rep::statement::{Holder, Says, Standing, Statement};
use cs_tactics::ledger::Count;
use cs_tactics::meansends::{ask, ask_with, search, Answer, Ask, Goal, Question, Reply};
use cs_tactics::reach::{NARROWED, PLAIN};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

type R = Ruled<KernelGame>;

const BUDGETS: [u64; 3] = [5_000, 50_000, 500_000];
/// The depth in plies the goals are asked and the reference reads to: the side, the reply, the side.
const DEPTH: i32 = 5;
/// THE FLOORS (what the layer did when this test was written, not a claim about its strength): over the goals of the fixture, at each budget, the goals the layer says reached, the goals it says touched, and the goals it settles (reached, touched or out of reach).
/// PLAN 79.29 PART B: RESET from the run on the remade fixture (no goal names one unit twice, every negated literal goes with the literal that keeps its unit in place). Side by side with the floors of 79.28 on the old fixture (reached 131, 134, 134; touched 6 then 5 after part A, 96, 103; settled 159, 366, 400):
/// now reached 107, 110, 110; touched 1, 98, 110; settled 139, 384, 400.
const FLOOR_REACHED: [usize; 3] = [107, 110, 110];
const FLOOR_TOUCHED: [usize; 3] = [1, 98, 110];
const FLOOR_SETTLED: [usize; 3] = [139, 384, 400];
/// What a kind of goal must be at least, in the reference, in the cases of its kind (plan 79.28 part C).
const KIND_FLOOR: usize = 10;

// ------------------------------------------------------------------------------------------------------------------------------ the fixtures

struct GoalPos { id: String, text: String, kinds: Vec<String>, goals: Vec<Statement> }
struct GameFx { name: String, decl: String, positions: Vec<GoalPos> }

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn statement(id: &str, pattern: Vec<Lit>) -> Statement { Statement { id: id.to_string(), pattern, says: Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: vec![] } }

fn read_goals_from(name: &str) -> Vec<GameFx> {
    let mut out: Vec<GameFx> = vec![];
    for line in fixture(name).lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "note" => {}
            "game" => out.push(GameFx { name: f[1].into(), decl: f[2].into(), positions: vec![] }),
            "pos" => {
                let v: cs_rep::serde_json::Value = cs_rep::serde_json::from_str(f[3]).unwrap();
                let (mut kinds, mut goals) = (vec![], vec![]);
                for (i, g) in v.as_array().unwrap().iter().enumerate() {
                    kinds.push(g["kind"].as_str().unwrap().to_string());
                    let pat: Vec<Lit> = g["pattern"].as_array().unwrap().iter().map(|l| Lit::from_json(l).unwrap()).collect();
                    goals.push(statement(&format!("{}#{i}", f[1]), pat));
                }
                out.last_mut().unwrap().positions.push(GoalPos { id: f[1].into(), text: f[2].into(), kinds, goals });
            }
            other => panic!("unknown tag {other}"),
        }
    }
    out
}

fn read_goals() -> Vec<GameFx> { read_goals_from("goals.fixture") }

fn rules_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules") }
fn open(fx: &GameFx) -> R { Ruled::open(KernelGame::new(&fx.name, 518).unwrap(), &fx.name, &rules_root(), &fx.decl).unwrap() }

/// Every index of 0..n asked of `f` on all the cores, each thread with its own copy of the game and board; the answers in index order.
fn par<T: Send>(fx: &GameFx, n: usize, f: &(dyn Fn(&R, &Static, usize) -> T + Sync)) -> Vec<T> {
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1).min(16);
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(usize, T)>> = Mutex::new(vec![]);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let g = open(fx);
                let st = g.board().unwrap();
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n { break; }
                    let r = f(&g, &st, i);
                    out.lock().unwrap().push((i, r));
                }
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}

fn none() -> Arc<Seen> { Arc::new(Seen::new()) }

fn question<'a>(g: &'a R, st: &'a Static, s: &'a <R as Game>::State, goals: &'a [Goal], nodes: u64, judge: &'a Count) -> Question<'a, R, Count> {
    Question { game: g, board: st, state: s, judge, side: g.mover(s).expect("a position the game goes on from"), goals, nodes, seconds: None, max_depth: DEPTH, history: none() }
}

fn holds_goals(p: &GoalPos) -> Vec<Goal> { p.goals.iter().map(|s| Goal::Holds(s.clone())).collect() }

// ------------------------------------------------------------------------------------------------------------------------------ THE REFERENCE (written apart: every move, no memo, no order, the facts of every position made from nothing)

/// What the reference says of a goal asked alone: reached in n plies (the fewest within the depth), touched, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ref { Reached(i32), Touched, Out }

struct Reference<'a> { g: &'a R, roles: Vec<String>, kinds: Vec<String>, places: Vec<String>, statics: Vec<Fact> }

impl<'a> Reference<'a> {
    fn new(g: &'a R, st: &Static) -> Reference<'a> {
        let mut statics = vec![];
        for (li, l) in st.lines.iter().enumerate() {
            let id = st.nodes[st.line_node[li] as usize].id.clone();
            for p in &l.places { statics.push(Fact { relation: "on_line".into(), args: vec![st.names.places[*p as usize].clone(), id.clone()] }); }
        }
        for (d, name) in st.directions.iter().enumerate() {
            for p in 0..st.names.places.len() { if st.edge_distance[d][p] == 0 { statics.push(Fact { relation: "on_edge".into(), args: vec![st.names.places[p].clone(), name.clone()] }); } }
        }
        Reference { g, roles: g.roles(), kinds: g.kind_names(), places: g.place_names(), statics }
    }

    /// The facts of a position, from the game's own questions.
    fn facts(&self, s: &<R as Game>::State) -> Vec<Fact> {
        let mut out = self.statics.clone();
        if let Some(m) = self.g.mover(s) { out.push(Fact { relation: "tempo".into(), args: vec![self.roles[m].clone()] }); }
        for (o, k, p) in self.g.cells(s) {
            let (on, kn, pn) = (self.roles[o].clone(), self.kinds[k as usize].clone(), self.places[p as usize].clone());
            out.push(Fact { relation: "unit".into(), args: vec![on.clone(), kn.clone(), pn.clone()] });
            for q in self.g.bearing_from(s, o, k, p) { out.push(Fact { relation: "bear".into(), args: vec![on.clone(), kn.clone(), pn.clone(), self.places[q as usize].clone()] }); }
            for q in self.g.reach_from(s, o, k, p) { out.push(Fact { relation: "may_move".into(), args: vec![on.clone(), kn.clone(), pn.clone(), self.places[q as usize].clone()] }); }
        }
        out
    }

    /// Whether the pattern binds over the facts: the positive literals in the order written, then each `differ` literal must hold of the binding, then each negated literal must have no row.
    fn binds(pattern: &[Lit], facts: &[Fact]) -> bool {
        fn unify(l: &Lit, f: &Fact, b: &BTreeMap<String, String>) -> Option<BTreeMap<String, String>> {
            if l.relation != f.relation || l.args.len() != f.args.len() { return None; }
            let mut nb = b.clone();
            for (a, v) in l.args.iter().zip(&f.args) {
                match a { Arg::Const(c) => if c != v { return None; }, Arg::Var(x) => match nb.get(x) { Some(w) => if w != v { return None; }, None => { nb.insert(x.clone(), v.clone()); } } }
            }
            Some(nb)
        }
        fn val(a: &Arg, b: &BTreeMap<String, String>) -> String { match a { Arg::Const(c) => c.clone(), Arg::Var(v) => b[v].clone() } }
        fn go(i: usize, pos: &[&Lit], neg: &[&Lit], diffs: &[&Lit], b: BTreeMap<String, String>, facts: &[Fact]) -> bool {
            if i == pos.len() {
                if !diffs.iter().all(|d| (val(&d.args[0], &b) != val(&d.args[1], &b)) != d.neg) { return false; }
                return !neg.iter().any(|n| facts.iter().any(|f| unify(n, f, &b).is_some()));
            }
            facts.iter().any(|f| unify(pos[i], f, &b).is_some_and(|nb| go(i + 1, pos, neg, diffs, nb, facts)))
        }
        let pos: Vec<&Lit> = pattern.iter().filter(|l| !l.neg && l.relation != "differ").collect();
        let neg: Vec<&Lit> = pattern.iter().filter(|l| l.neg && l.relation != "differ").collect();
        let diffs: Vec<&Lit> = pattern.iter().filter(|l| l.relation == "differ").collect();
        go(0, &pos, &neg, &diffs, BTreeMap::new(), facts)
    }

    fn binds_at(&self, pattern: &[Lit], s: &<R as Game>::State) -> bool { Self::binds(pattern, &self.facts(s)) }

    /// HELD: the pattern binds and the side is to move, or the game is over.
    fn held(&self, pattern: &[Lit], s: &<R as Game>::State, me: Role) -> bool { (self.g.outcome(s).is_some() || self.g.mover(s) == Some(me)) && self.binds_at(pattern, s) }

    /// Reached within `d` plies by `me`? Every move of the side, every reply; a position where the pattern is held ends the line as reached; a game that is over, a position met again, the depth limit end it as not.
    fn reach(&self, pattern: &[Lit], s: &<R as Game>::State, me: Role, d: i32, path: &mut Vec<u64>) -> bool {
        if self.held(pattern, s, me) { return true; }
        if self.g.outcome(s).is_some() || d == 0 { return false; }
        let key = self.g.key(s);
        if !path.is_empty() && path.contains(&key) { return false; }
        path.push(key);
        let moves = self.g.moves(s);
        let me_moves = self.g.mover(s) == Some(me);
        let ok = if me_moves { moves.iter().any(|m| self.reach(pattern, &self.g.apply(s, m), me, d - 1, path)) } else { !moves.is_empty() && moves.iter().all(|m| self.reach(pattern, &self.g.apply(s, m), me, d - 1, path)) };
        path.pop();
        ok
    }

    /// The meaning of plan 79.27, kept only to count what changed: reached where the pattern binds at ANY position of the line, either side to move.
    fn reach_old(&self, pattern: &[Lit], s: &<R as Game>::State, me: Role, d: i32, path: &mut Vec<u64>) -> bool {
        if self.binds_at(pattern, s) { return true; }
        if self.g.outcome(s).is_some() || d == 0 { return false; }
        let key = self.g.key(s);
        if !path.is_empty() && path.contains(&key) { return false; }
        path.push(key);
        let moves = self.g.moves(s);
        let ok = if self.g.mover(s) == Some(me) { moves.iter().any(|m| self.reach_old(pattern, &self.g.apply(s, m), me, d - 1, path)) } else { !moves.is_empty() && moves.iter().all(|m| self.reach_old(pattern, &self.g.apply(s, m), me, d - 1, path)) };
        path.pop();
        ok
    }

    /// Reached or touched within `d` plies: the lines of `reach`, with one more kind of end: the pattern binds right after a move of the side (`after_mine`, at least one ply in, the other side to move, the game not over, a ply left to read) and some reply leaves it unbound.
    fn reach_or_touch(&self, pattern: &[Lit], s: &<R as Game>::State, me: Role, d: i32, path: &mut Vec<u64>, after_mine: bool, ply: i32) -> bool {
        if self.held(pattern, s, me) { return true; }
        if self.g.outcome(s).is_some() || d == 0 { return false; }
        let key = self.g.key(s);
        if !path.is_empty() && path.contains(&key) { return false; }
        let moves = self.g.moves(s);
        if after_mine && ply >= 1 && self.binds_at(pattern, s) && moves.iter().any(|m| !self.binds_at(pattern, &self.g.apply(s, m))) { return true; }
        path.push(key);
        let me_moves = self.g.mover(s) == Some(me);
        let ok = if me_moves { moves.iter().any(|m| self.reach_or_touch(pattern, &self.g.apply(s, m), me, d - 1, path, true, ply + 1)) } else { !moves.is_empty() && moves.iter().all(|m| self.reach_or_touch(pattern, &self.g.apply(s, m), me, d - 1, path, false, ply + 1)) };
        path.pop();
        ok
    }

    /// The reference's answer for one goal alone: the fewest plies it is reached in within the depth; else touched; else neither.
    fn status(&self, pattern: &[Lit], s: &<R as Game>::State, me: Role) -> Ref {
        if let Some(n) = (0..=DEPTH).find(|&d| self.reach(pattern, s, me, d, &mut vec![])) { return Ref::Reached(n); }
        if self.reach_or_touch(pattern, s, me, DEPTH, &mut vec![], false, 0) { Ref::Touched } else { Ref::Out }
    }

    /// The meaning of plan 79.27: the fewest plies within the depth, or None.
    fn plies_old(&self, pattern: &[Lit], s: &<R as Game>::State, me: Role) -> Option<i32> { (0..=DEPTH).find(|&d| self.reach_old(pattern, s, me, d, &mut vec![])) }

    /// Both patterns held along ONE line within `d` plies (the flags accumulate down the line; the other side's replies are all met).
    #[allow(clippy::too_many_arguments)]
    fn both(&self, a: &[Lit], b: &[Lit], s: &<R as Game>::State, me: Role, d: i32, fa: bool, fb: bool, path: &mut Vec<u64>) -> bool {
        let (fa, fb) = (fa || self.held(a, s, me), fb || self.held(b, s, me));
        if fa && fb { return true; }
        if self.g.outcome(s).is_some() || d == 0 { return false; }
        let key = self.g.key(s);
        if path.contains(&key) { return false; }
        path.push(key);
        let moves = self.g.moves(s);
        let ok = if self.g.mover(s) == Some(me) { moves.iter().any(|m| self.both(a, b, &self.g.apply(s, m), me, d - 1, fa, fb, path)) } else { !moves.is_empty() && moves.iter().all(|m| self.both(a, b, &self.g.apply(s, m), me, d - 1, fa, fb, path)) };
        path.pop();
        ok
    }
}

fn replay<'a>(g: &R, s: &<R as Game>::State, line: &'a [String], id: &str) -> Vec<<R as Game>::State> {
    let mut cur = s.clone();
    let mut states = vec![cur.clone()];
    for name in line {
        let m = g.find_move(&cur, name).unwrap_or_else(|| panic!("{id}: the line's move {name} is legal in its place"));
        cur = g.apply(&cur, &m);
        states.push(cur.clone());
    }
    states
}

fn grounded(rf: &Reference, pattern: &[Lit], at: &<R as Game>::State, bindings: &[cs_rep::matcher::Binding], id: &str) {
    assert!(!bindings.is_empty(), "{id}: the bindings are given");
    let facts: BTreeSet<Fact> = rf.facts(at).into_iter().collect();
    for b in bindings {
        for l in pattern.iter().filter(|l| !l.neg && l.relation != "differ") {
            let args: Vec<String> = l.args.iter().map(|a| match a { Arg::Const(c) => c.clone(), Arg::Var(v) => b[v].clone() }).collect();
            assert!(facts.contains(&Fact { relation: l.relation.clone(), args }), "{id}: a binding grounds a literal onto a fact of the position where the pattern binds");
        }
        for d in pattern.iter().filter(|l| l.relation == "differ") {
            let v = |a: &Arg| match a { Arg::Const(c) => c.clone(), Arg::Var(x) => b[x].clone() };
            assert!((v(&d.args[0]) != v(&d.args[1])) != d.neg, "{id}: a binding meets the differ literal");
        }
    }
}

/// What a reached answer must be: a line the rules replay, ending where the pattern is HELD (read by the reference), within the depth, with bindings that ground the pattern; and, where the line is a move of the side and one reply, the pattern binds after every legal reply.
fn confirm_reached(rf: &Reference, g: &R, s: &<R as Game>::State, pattern: &[Lit], r: &cs_tactics::meansends::Reached, id: &str) {
    assert!(r.plies as usize == r.line.len() && r.plies <= DEPTH, "{id}: the line is within the depth: {:?}", r.line);
    let me = g.mover(s).unwrap();
    let states = replay(g, s, &r.line, id);
    let end = states.last().unwrap();
    assert!(rf.held(pattern, end, me), "{id}: the pattern is held where the line ends (binds, the side to move or the game over)");
    grounded(rf, pattern, end, &r.bindings, id);
    if r.line.len() == 2 {
        let after_first = &states[1];
        for m in g.moves(after_first) {
            let q = g.apply(after_first, &m);
            assert!(rf.binds_at(pattern, &q), "{id}: a line of a move and a reply holds the pattern after every legal reply");
        }
    }
}

/// What a touched answer must be: a line the rules replay, within the depth less the reply, whose last move is the side's and after which the pattern binds with the other side to move and the game not over, and a reply, legal there, after which it does not.
fn confirm_touched(rf: &Reference, g: &R, s: &<R as Game>::State, pattern: &[Lit], t: &cs_tactics::meansends::Touched, id: &str) {
    assert!(t.plies as usize == t.line.len() && t.plies >= 1 && t.plies < DEPTH, "{id}: the line is within the depth less the reply: {:?}", t.line);
    let me = g.mover(s).unwrap();
    let states = replay(g, s, &t.line, id);
    let (end, before) = (states.last().unwrap(), &states[states.len() - 2]);
    assert!(g.mover(before) == Some(me), "{id}: the line's last move is the side's");
    assert!(g.mover(end) != Some(me) && g.outcome(end).is_none(), "{id}: the other side is to move and the game is not over");
    assert!(rf.binds_at(pattern, end), "{id}: the pattern binds where the line ends");
    grounded(rf, pattern, end, &t.bindings, id);
    let m = g.find_move(end, &t.reply).unwrap_or_else(|| panic!("{id}: the reply {} is legal there", t.reply));
    assert!(!rf.binds_at(pattern, &g.apply(end, &m)), "{id}: the reply leaves the pattern unbound");
}

// ------------------------------------------------------------------------------------------------------------------------------ the answers, read once

/// One goal alone: the reference's answer, and each budget's answer from the list and from the goal asked alone.
struct Row { game: usize, pos: usize, goal: usize, kind: String, reference: Ref, list: Vec<Answer>, alone: Vec<Answer> }

/// 0 reached, 1 touched, 2 out of reach, 3 not settled.
fn status(a: &Answer) -> u8 { match a { Answer::Reached(_) => 0, Answer::Touched(_) => 1, Answer::OutOfReach { .. } => 2, Answer::NotSettled { .. } => 3 } }
fn ref_status(r: &Ref) -> u8 { match r { Ref::Reached(_) => 0, Ref::Touched => 1, Ref::Out => 2 } }

type Table = (Vec<GameFx>, Vec<Row>);

/// The reference on every goal of a fixture, on all the cores, in the order of the games, positions and goals.
fn references(games: &[GameFx]) -> Vec<Vec<Ref>> {
    games.iter().map(|fx| {
        let flat: Vec<(usize, usize)> = (0..fx.positions.len()).flat_map(|p| (0..fx.positions[p].goals.len()).map(move |g| (p, g))).collect();
        par(fx, flat.len(), &|g, st, i| {
            let (p, k) = flat[i];
            let s = g.read_state_text(&fx.positions[p].text).unwrap();
            Reference::new(g, st).status(&fx.positions[p].goals[k].pattern, &s, g.mover(&s).unwrap())
        })
    }).collect()
}

fn refs() -> &'static (Vec<GameFx>, Vec<Vec<Ref>>) {
    static CELL: std::sync::OnceLock<(Vec<GameFx>, Vec<Vec<Ref>>)> = std::sync::OnceLock::new();
    CELL.get_or_init(|| {
        let games = read_goals();
        let t0 = Instant::now();
        let r = references(&games);
        println!("OBS 79.28 reference: {} goals read by the slow plain reference at depth {DEPTH} plies on all the cores in {:.1} s", r.iter().map(|x| x.len()).sum::<usize>(), t0.elapsed().as_secs_f64());
        (games, r)
    })
}

fn table() -> &'static Table {
    static CELL: std::sync::OnceLock<Table> = std::sync::OnceLock::new();
    CELL.get_or_init(|| {
        let (games0, refs) = refs();
        let games = read_goals();
        let _ = games0;
        let mut rows = vec![];
        let t0 = Instant::now();
        let judge = Count;
        for (gi, fx) in games.iter().enumerate() {
            let n = fx.positions.len();
            // the layer, the list at each budget and each goal alone at each budget
            let per: Vec<(Vec<Reply>, Vec<Vec<Reply>>)> = par(fx, n, &|g, st, p| {
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                let goals = holds_goals(&fx.positions[p]);
                let list: Vec<Reply> = BUDGETS.iter().map(|&b| ask(&question(g, st, &s, &goals, b, &judge))).collect();
                let alone: Vec<Vec<Reply>> = (0..goals.len()).map(|k| BUDGETS.iter().map(|&b| ask(&question(g, st, &s, &goals[k..k + 1], b, &judge))).collect()).collect();
                (list, alone)
            });
            let mut flat_i = 0;
            for (p, (list, alone)) in per.iter().enumerate() {
                for k in 0..fx.positions[p].goals.len() {
                    rows.push(Row { game: gi, pos: p, goal: k, kind: fx.positions[p].kinds[k].clone(), reference: refs[gi][flat_i], list: list.iter().map(|r| r.answers[k].clone()).collect(), alone: alone[k].iter().map(|r| r.answers[0].clone()).collect() });
                    flat_i += 1;
                }
            }
        }
        println!("OBS 79.28 table: {} goals, the layer (list, and each goal alone) at budgets {:?} nodes, depth {DEPTH} plies, read on all the cores in {:.1} s", rows.len(), BUDGETS, t0.elapsed().as_secs_f64());
        (games, rows)
    })
}

// ------------------------------------------------------------------------------------------------------------------------------ the tests

#[test]
fn every_kind_of_goal_is_reached_by_the_reference_in_at_least_ten_of_its_cases() {
    let (games, refs) = refs();
    let mut by_kind: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    let mut by_game: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for (gi, fx) in games.iter().enumerate() {
        let mut i = 0;
        for p in &fx.positions { for k in 0..p.goals.len() {
            let c = ref_status(&refs[gi][i]) as usize;
            by_kind.entry(p.kinds[k].clone()).or_default()[c] += 1;
            by_game.entry(fx.name.clone()).or_default()[c] += 1;
            i += 1;
        } }
    }
    for (k, c) in &by_kind { println!("OBS 79.28 reference kind {k}: {} goals, reached {}, touched {}, neither {}", c[0] + c[1] + c[2], c[0], c[1], c[2]); }
    for (g, c) in &by_game { println!("OBS 79.28 reference game {g}: {} goals, reached {}, touched {}, neither {}", c[0] + c[1] + c[2], c[0], c[1], c[2]); }
    for (k, c) in &by_kind { assert!(c[0] >= KIND_FLOOR, "kind {k}: reached in {} of {} cases, floor {KIND_FLOOR}", c[0], c[0] + c[1] + c[2]); }
    assert!(by_kind.len() >= 5, "five kinds of goal");
}

#[test]
fn the_layer_agrees_with_the_reference_on_every_goal_at_three_budgets() {
    let (games, rows) = table();
    let mut kinds: BTreeSet<&str> = BTreeSet::new();
    for r in rows { kinds.insert(&r.kind); }
    let mut seen: Vec<(usize, usize, usize)> = vec![];
    for b in 0..BUDGETS.len() {
        let (mut reached, mut touched, mut out, mut unsettled, mut contradictions, mut exact, mut line_gap) = (0, 0, 0, 0, 0, 0, 0);
        let mut per_kind: BTreeMap<&str, [usize; 5]> = BTreeMap::new();
        for r in rows {
            let a = &r.list[b];
            let s = status(a);
            let e = per_kind.entry(&r.kind).or_default();
            e[s as usize] += 1;
            match a { Answer::Reached(x) => { reached += 1; if let Ref::Reached(n) = r.reference { if x.plies < n { line_gap += 1; } } } Answer::Touched(_) => touched += 1, Answer::OutOfReach { .. } => out += 1, Answer::NotSettled { .. } => unsettled += 1 }
            if s != 3 && s != ref_status(&r.reference) { contradictions += 1; e[4] += 1; }
            if s == ref_status(&r.reference) { exact += 1; }
        }
        println!("OBS 79.28 list at {} nodes: of {} goals reached {reached}, touched {touched}, out of reach {out}, not settled {unsettled}; contradictions of the reference {contradictions}; answers equal to the reference {exact}; a line shorter than the reference's fewest {line_gap}", BUDGETS[b], rows.len());
        for (k, c) in &per_kind { println!("OBS 79.28 list at {} nodes, kind {k}: reached {}, touched {}, out of reach {}, not settled {}, contradictions {}", BUDGETS[b], c[0], c[1], c[2], c[3], c[4]); }
        assert_eq!(contradictions, 0, "every settled answer is the reference's: reached only where it reaches, touched only where it touches, out of reach only where it does neither");
        assert_eq!(line_gap, 0, "a line is never shorter than the reference's fewest plies");
        seen.push((reached, touched, reached + touched + out));
        if b == BUDGETS.len() - 1 { assert_eq!(unsettled, 0, "at the top budget every goal of this depth is settled"); assert_eq!(exact, rows.len(), "at the top budget the layer says exactly what the reference does"); }
    }
    println!("OBS 79.29 observed (reached, touched, settled) at the three budgets: {:?}; floors: reached {FLOOR_REACHED:?}, touched {FLOOR_TOUCHED:?}, settled {FLOOR_SETTLED:?}", seen);
    for (b, &(reached, touched, settled)) in seen.iter().enumerate() {
        assert!(reached >= FLOOR_REACHED[b] && touched >= FLOOR_TOUCHED[b] && settled >= FLOOR_SETTLED[b], "the floors at {} nodes: reached {reached} touched {touched} settled {settled}", BUDGETS[b]);
    }
    println!("OBS 79.28 kinds: {:?} on {} games", kinds, games.len());
}

#[test]
fn a_reached_or_touched_answer_is_a_line_the_rules_replay_to_the_position_the_meaning_asks() {
    let (games, rows) = table();
    let (mut confirmed_reached, mut confirmed_touched) = (0, 0);
    for (gi, fx) in games.iter().enumerate() {
        let g = open(fx);
        let st = g.board().unwrap();
        let rf = Reference::new(&g, &st);
        for r in rows.iter().filter(|r| r.game == gi) {
            for b in 0..BUDGETS.len() {
                let p = &fx.positions[r.pos];
                let s = g.read_state_text(&p.text).unwrap();
                let id = format!("{}#{} at {}", p.id, r.goal, BUDGETS[b]);
                match &r.list[b] {
                    Answer::Reached(x) => { confirm_reached(&rf, &g, &s, &p.goals[r.goal].pattern, x, &id); confirmed_reached += 1; }
                    Answer::Touched(t) => { confirm_touched(&rf, &g, &s, &p.goals[r.goal].pattern, t, &id); confirmed_touched += 1; }
                    _ => {}
                }
            }
        }
    }
    println!("OBS 79.28 replay: {confirmed_reached} reached answers replayed to a position where the pattern is held, {confirmed_touched} touched answers replayed to a position where it binds after a move of the side and to a reply that unbinds it (every goal, every budget)");
    assert!(confirmed_reached > 0 && confirmed_touched > 0);
}

#[test]
fn the_list_never_contradicts_a_goal_asked_alone_and_the_nodes_of_one_tree_are_counted_beside_k_searches() {
    let (games, rows) = table();
    let mut contradictions = 0;
    let mut differ = 0;
    for r in rows { for b in 0..BUDGETS.len() {
        let (l, a) = (status(&r.list[b]), status(&r.alone[b]));
        if l != 3 && a != 3 && l != a { contradictions += 1; }
        if l != a { differ += 1; }
    } }
    println!("OBS 79.28 list against alone: {} goals x {} budgets; answers settled in both that differ {contradictions}; answers whose status differs at all (one not settled) {differ}", rows.len(), BUDGETS.len());
    assert_eq!(contradictions, 0, "a goal settled both ways is settled the same way");
    // the nodes of one tree beside the nodes of k single searches, over the positions, at each budget
    let t0 = Instant::now();
    for &budget in BUDGETS.iter() {
        let mut totals = (0u64, 0u64, 0u64, 0u64);
        for fx in games {
            let sums: Vec<(u64, u64, u64, u64)> = par(fx, fx.positions.len(), &|g, st, p| {
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                let goals = holds_goals(&fx.positions[p]);
                let one = ask(&question(g, st, &s, &goals, budget, &Count));
                let (mut n, mut mk) = (0u64, 0u64);
                for k in 0..goals.len() { let a = ask(&question(g, st, &s, &goals[k..k + 1], budget, &Count)); n += a.nodes; mk += a.makes; }
                (one.nodes, n, one.makes, mk)
            });
            for s in sums { totals.0 += s.0; totals.1 += s.1; totals.2 += s.2; totals.3 += s.3; }
        }
        println!("OBS 79.28 one tree beside k searches at {budget} nodes: nodes one tree {} against the sum of k single searches {}; positions made on the live graph {} against {}", totals.0, totals.1, totals.2, totals.3);
    }
    println!("OBS 79.28 one tree beside k searches read in {:.1} s", t0.elapsed().as_secs_f64());
}

#[test]
fn the_order_of_the_list_changes_how_the_budget_is_spent_and_never_a_settled_answer() {
    let (games, _rows) = table();
    let mut changed = [0usize; 3];
    let mut total = 0;
    let mut contradictions = 0;
    for fx in games {
        let res: Vec<(Vec<[u8; 3]>, Vec<[u8; 3]>)> = par(fx, fx.positions.len(), &|g, st, p| {
            let s = g.read_state_text(&fx.positions[p].text).unwrap();
            let goals = holds_goals(&fx.positions[p]);
            let rev: Vec<Goal> = goals.iter().rev().map(|g| match g { Goal::Holds(s) => Goal::Holds(s.clone()), _ => unreachable!() }).collect();
            let (mut fw, mut bw) = (vec![], vec![]);
            let n = goals.len();
            let rs: Vec<(Reply, Reply)> = BUDGETS.iter().map(|&b| (ask(&question(g, st, &s, &goals, b, &Count)), ask(&question(g, st, &s, &rev, b, &Count)))).collect();
            for k in 0..n { fw.push([status(&rs[0].0.answers[k]), status(&rs[1].0.answers[k]), status(&rs[2].0.answers[k])]); bw.push([status(&rs[0].1.answers[n - 1 - k]), status(&rs[1].1.answers[n - 1 - k]), status(&rs[2].1.answers[n - 1 - k])]); }
            (fw, bw)
        });
        for (fw, bw) in res { for k in 0..fw.len() { total += 1; for b in 0..3 { if fw[k][b] != bw[k][b] { changed[b] += 1; if fw[k][b] != 3 && bw[k][b] != 3 { contradictions += 1; } } } } }
    }
    println!("OBS 79.28 order: of {total} goals the answer changes when the list is reversed, at {} nodes {}, at {} nodes {}, at {} nodes {}; settled both ways and different {contradictions}", BUDGETS[0], changed[0], BUDGETS[1], changed[1], BUDGETS[2], changed[2]);
    assert_eq!(contradictions, 0);
}

#[test]
fn the_same_answer_asked_twice_and_the_narrowing_answers_as_the_plain_tree_where_both_settle() {
    let (games, _rows) = table();
    let budget = 50_000;
    let (mut nodes_plain, mut nodes_narrow, mut makes_plain, mut makes_narrow, mut matches_plain, mut matches_narrow) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut diff, mut both_settled, mut twice) = (0, 0, 0);
    let (mut t_plain, mut t_narrow) = (0f64, 0f64);
    for fx in games {
        let res: Vec<(Reply, Reply, Reply, f64, f64)> = par(fx, fx.positions.len(), &|g, st, p| {
            let s = g.read_state_text(&fx.positions[p].text).unwrap();
            let goals = holds_goals(&fx.positions[p]);
            let q = question(g, st, &s, &goals, budget, &Count);
            let t0 = Instant::now();
            let a = ask_with(&q, PLAIN);
            let t1 = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let b = ask_with(&q, NARROWED);
            let t2 = t0.elapsed().as_secs_f64();
            let c = ask(&q);
            (a, b, c, t1, t2)
        });
        for (a, b, c, t1, t2) in res {
            nodes_plain += a.nodes; nodes_narrow += b.nodes; makes_plain += a.makes; makes_narrow += b.makes; matches_plain += a.matches; matches_narrow += b.matches; t_plain += t1; t_narrow += t2;
            if b == c { twice += 1; }
            for k in 0..a.answers.len() {
                let (x, y) = (status(&a.answers[k]), status(&b.answers[k]));
                if x != 3 && y != 3 { both_settled += 1; if x != y { diff += 1; } }
            }
        }
    }
    println!("OBS 79.28 narrowing at {budget} nodes (one tree per position): nodes plain {nodes_plain} narrowed {nodes_narrow}; positions made plain {makes_plain} narrowed {makes_narrow}; rows matched plain {matches_plain} narrowed {matches_narrow}; seconds (summed over threads) plain {t_plain:.2} narrowed {t_narrow:.2}; goals settled by both {both_settled}, answers that differ {diff}; the narrowed tree asked twice equal {twice} of {}", games.iter().map(|g| g.positions.len()).sum::<usize>());
    assert_eq!(diff, 0, "the narrowing answers as the plain tree wherever both settle");
    assert_eq!(twice, games.iter().map(|g| g.positions.len()).sum::<usize>(), "the same answer asked twice");
}

#[test]
fn each_part_of_the_narrowing_alone_is_measured_beside_the_plain_tree_and_answers_as_it_does() {
    let (games, _rows) = table();
    let budget = 50_000;
    let variants = [("plain", PLAIN), ("table only", cs_tactics::reach::Narrow { table: true, order: false }), ("order only", cs_tactics::reach::Narrow { table: false, order: true }), ("both", NARROWED)];
    let mut totals = [(0u64, 0u64, 0u64, 0f64); 4];
    let mut diff = 0;
    for fx in games {
        let res: Vec<Vec<(Reply, f64)>> = par(fx, fx.positions.len(), &|g, st, p| {
            let s = g.read_state_text(&fx.positions[p].text).unwrap();
            let goals = holds_goals(&fx.positions[p]);
            let q = question(g, st, &s, &goals, budget, &Count);
            variants.iter().map(|(_, n)| { let t0 = Instant::now(); let r = ask_with(&q, *n); (r, t0.elapsed().as_secs_f64()) }).collect()
        });
        for v in res {
            for (i, (r, t)) in v.iter().enumerate() { totals[i].0 += r.nodes; totals[i].1 += r.makes; totals[i].2 += r.matches; totals[i].3 += t; }
            for k in 0..v[0].0.answers.len() { for i in 1..4 { let (x, y) = (status(&v[0].0.answers[k]), status(&v[i].0.answers[k])); if x != 3 && y != 3 && x != y { diff += 1; } } }
        }
    }
    for (i, (name, _)) in variants.iter().enumerate() {
        println!("OBS 79.28 narrowing parts at {budget} nodes, {name}: nodes {} positions made {} rows matched {} seconds (summed over threads) {:.2}", totals[i].0, totals[i].1, totals[i].2, totals[i].3);
    }
    assert_eq!(diff, 0, "every variant answers as the plain tree where both settle");
}

#[test]
fn where_the_nodes_go_between_the_calls_asking_the_goals_held_the_goals_touched_and_both() {
    let (games, _rows) = table();
    for &budget in BUDGETS.iter() {
        let (mut held, mut touched, mut both, mut total, mut makes, mut matches) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for fx in games {
            let res: Vec<Reply> = par(fx, fx.positions.len(), &|g, st, p| {
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                ask(&question(g, st, &s, &holds_goals(&fx.positions[p]), budget, &Count))
            });
            for r in res { held += r.diag.calls_held; touched += r.diag.calls_touched; both += r.diag.calls_both; total += r.nodes; makes += r.makes; matches += r.matches; }
        }
        println!("OBS 79.29 where the nodes go at {budget} nodes (the list of each position, one reading for both answers): nodes {total}, of the calls of the tree: asking the goals held only {held}, touched only {touched}, both {both}; positions made on the live graph {makes}, rows matched {matches}");
        println!("OBS 79.29 the rest of the nodes at {budget} nodes: {} (the reading of whether one reply meets both goals of a pair)", total - held - touched - both);
        assert!(held + touched + both <= total);
    }
}

/// THE MEANING AGAINST THE ONE OF PLAN 79.27, on the questions of plan 79.27 (tests/fixtures/goals_79_27.fixture, the 400 goals of that plan's fixture, kept): the nodes of one tree per position at the three budgets under the new meaning, and, where the reference of each meaning is read (LAYER4_TRANSITIONS=1: the slow part,
/// two references over 400 goals), how many goals the old meaning reaches that the new one reaches, touches, or neither.
#[test]
fn the_questions_of_plan_79_27_are_asked_again_under_the_new_meaning() {
    let games = read_goals_from("goals_79_27.fixture");
    let t0 = Instant::now();
    for &budget in BUDGETS.iter() {
        let (mut nodes, mut makes, mut matches, mut held, mut touched) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let mut stat = [0usize; 4];
        for fx in &games {
            let res: Vec<Reply> = par(fx, fx.positions.len(), &|g, st, p| {
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                ask(&question(g, st, &s, &holds_goals(&fx.positions[p]), budget, &Count))
            });
            for r in res { nodes += r.nodes; makes += r.makes; matches += r.matches; held += r.diag.calls_held; touched += r.diag.calls_touched; for a in &r.answers { stat[status(a) as usize] += 1; } }
        }
        println!("OBS 79.29 the questions of 79.27 under the new meaning at {budget} nodes: nodes one tree {nodes} (calls asking held only {held}, touched only {touched}); positions made {makes}; rows matched {matches}; reached {}, touched {}, out of reach {}, not settled {}", stat[0], stat[1], stat[2], stat[3]);
    }
    println!("OBS 79.28 the questions of 79.27 read under the new meaning in {:.1} s", t0.elapsed().as_secs_f64());
    if std::env::var("LAYER4_TRANSITIONS").is_err() { return; }
    // the transitions: the old meaning's reference, the new meaning's reference and the layer at the top budget, goal by goal
    let mut table = [[0usize; 3]; 2];
    let (mut old_reached, mut now_reached, mut now_touched, mut now_neither, mut agree) = (0, 0, 0, 0, 0);
    let top = *BUDGETS.last().unwrap();
    for fx in &games {
        let res: Vec<(Vec<(Option<i32>, Ref, u8)>, usize)> = par(fx, fx.positions.len(), &|g, st, p| {
            let s = g.read_state_text(&fx.positions[p].text).unwrap();
            let me = g.mover(&s).unwrap();
            let rf = Reference::new(g, st);
            let goals = holds_goals(&fx.positions[p]);
            let rep = ask(&question(g, st, &s, &goals, top, &Count));
            let v = (0..goals.len()).map(|k| { let pat = &fx.positions[p].goals[k].pattern; (rf.plies_old(pat, &s, me), rf.status(pat, &s, me), status(&rep.answers[k])) }).collect();
            (v, p)
        });
        for (v, _) in res { for (old, new, layer) in v {
            if layer == ref_status(&new) { agree += 1; }
            if old.is_some() { old_reached += 1; match new { Ref::Reached(_) => now_reached += 1, Ref::Touched => now_touched += 1, Ref::Out => now_neither += 1 } }
            table[old.is_some() as usize][ref_status(&new) as usize] += 1;
        } }
    }
    println!("OBS 79.28 transitions on the 400 goals of 79.27 (the reference of each meaning): reached under the old meaning {old_reached}; of those, reached now {now_reached}, touched now {now_touched}, neither now {now_neither}; not reached under the old meaning: reached now {}, touched now {}, neither {}; the layer at {top} nodes equals the new reference on {agree} of {} goals", table[0][0], table[0][1], table[0][2], games.iter().map(|g| g.positions.iter().map(|p| p.goals.len()).sum::<usize>()).sum::<usize>());
    assert_eq!(table[0][0], 0, "a goal the old meaning does not reach is not reached by the stricter one");
}

#[test]
fn the_two_old_goals_answer_as_the_ledger_search_did_when_asked_in_a_list_of_one() {
    let text = fixture("puzzle.fixture");
    let mut fx = GameFx { name: String::new(), decl: String::new(), positions: vec![] };
    let mut texts = vec![];
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] { "game" => { fx.name = f[1].into(); fx.decl = f[2].into(); } "puzzle" => if texts.len() < 120 { texts.push(f[6].to_string()) }, _ => {} }
    }
    let budget = 50_000;
    let res: Vec<(bool, bool, u64, u64)> = par(&fx, texts.len(), &|g, st, i| {
        let s = g.read_state_text(&texts[i]).unwrap();
        let me = g.mover(&s).unwrap();
        let direct = search(&Ask { game: g, state: &s, judge: &Count, side: me, nodes: budget, seconds: None, max_depth: 64, history: none() });
        let goals = [Goal::Gain];
        let via = ask(&Question { game: g, board: st, state: &s, judge: &Count, side: me, goals: &goals, nodes: budget, seconds: None, max_depth: 64, history: none() });
        let gain_same = match &via.answers[0] { Answer::Reached(r) => r.found.as_ref() == Some(&direct), Answer::NotSettled { best, .. } => best.as_ref() == Some(&direct), _ => false };
        let goals = [Goal::End];
        let via_end = ask(&Question { game: g, board: st, state: &s, judge: &Count, side: me, goals: &goals, nodes: budget, seconds: None, max_depth: 64, history: none() });
        let end_same = match &via_end.answers[0] { Answer::Reached(r) => r.found.as_ref() == Some(&direct) && direct.ending == cs_tactics::meansends::Ending::Won, Answer::NotSettled { best, .. } => best.as_ref() == Some(&direct) && direct.ending != cs_tactics::meansends::Ending::Won, _ => false };
        (gain_same, end_same, direct.nodes, via.nodes)
    });
    let (g_ok, e_ok) = (res.iter().filter(|r| r.0).count(), res.iter().filter(|r| r.1).count());
    println!("OBS 79.28 old goals: of {} positions asked as a list of one, the gain answer is the ledger search's own {g_ok} times, the end answer {e_ok} times; nodes direct {} through the entry {}", res.len(), res.iter().map(|r| r.2).sum::<u64>(), res.iter().map(|r| r.3).sum::<u64>());
    assert_eq!(g_ok, res.len());
    assert_eq!(e_ok, res.len());
}

#[test]
fn positions_where_one_line_reaches_two_goals_are_read_from_the_fixture_the_reference_made() {
    let games = read_goals();
    if std::env::var("LAYER4_MAKE_PAIRS").is_ok() {
        let mut lines = vec!["note\tplan 79.29 part B (first made in plan 79.28 part D): positions where ONE line reaches two goals of the goals fixture within the depth, in the meaning of plan 79.28 (each pattern held along one line against every reply), found by the reference, each goal also reached alone; made by running layer4_goals.rs with LAYER4_MAKE_PAIRS=1".to_string()];
        for fx in &games {
            lines.push(format!("game\t{}\t{}", fx.name, fx.decl));
            let found: Vec<Vec<(usize, usize, usize)>> = par(fx, fx.positions.len(), &|g, st, p| {
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                let me = g.mover(&s).unwrap();
                let rf = Reference::new(g, st);
                let gs = &fx.positions[p].goals;
                let mut out = vec![];
                let alone: Vec<Ref> = gs.iter().map(|x| rf.status(&x.pattern, &s, me)).collect();
                for a in 0..gs.len() { for b in a + 1..gs.len() {
                    if matches!(alone[a], Ref::Reached(_)) && matches!(alone[b], Ref::Reached(_)) && rf.both(&gs[a].pattern, &gs[b].pattern, &s, me, DEPTH - 1, false, false, &mut vec![]) { out.push((p, a, b)); }
                } }
                out
            });
            for (p, a, b) in found.into_iter().flatten() { lines.push(format!("pair\t{}\t{}\t{}", fx.positions[p].id, a, b)); }
        }
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("pairs.fixture");
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        println!("OBS 79.28 pairs fixture written: {} pairs", lines.iter().filter(|l| l.starts_with("pair")).count());
        return;
    }
    let text = fixture("pairs.fixture");
    let mut pairs: Vec<(String, usize, usize)> = vec![];
    for l in text.lines() { let f: Vec<&str> = l.split('\t').collect(); if f[0] == "pair" { pairs.push((f[1].into(), f[2].parse().unwrap(), f[3].parse().unwrap())); } }
    assert!(pairs.len() >= 10, "the fixture holds positions: {}", pairs.len());
    let (mut said_both, mut reply_read, mut reply_true, mut each_reached, mut replay_ok) = (0, 0, 0, 0, 0);
    let top = *BUDGETS.last().unwrap();
    for fx in &games {
        let mine: Vec<&(String, usize, usize)> = pairs.iter().filter(|p| fx.positions.iter().any(|q| q.id == p.0)).collect();
        let res: Vec<(bool, bool, Option<bool>, bool)> = par(fx, mine.len(), &|g, st, i| {
            let (id, a, b) = mine[i];
            let p = fx.positions.iter().find(|q| &q.id == id).unwrap();
            let s = g.read_state_text(&p.text).unwrap();
            let me = g.mover(&s).unwrap();
            let goals = holds_goals(p);
            let rep = ask(&question(g, st, &s, &goals, top, &Count));
            let rf = Reference::new(g, st);
            let reached = rep.answers[*a].reached().is_some() && rep.answers[*b].reached().is_some();
            let both = rep.both.iter().find(|x| x.goals == (*a, *b));
            let replay_both = both.is_some_and(|x| {
                let mut cur = s.clone();
                let (mut fa, mut fb) = (rf.held(&p.goals[*a].pattern, &cur, me), rf.held(&p.goals[*b].pattern, &cur, me));
                for name in &x.line { let m = g.find_move(&cur, name).unwrap(); cur = g.apply(&cur, &m); fa |= rf.held(&p.goals[*a].pattern, &cur, me); fb |= rf.held(&p.goals[*b].pattern, &cur, me); }
                fa && fb
            });
            (reached, both.is_some(), both.and_then(|x| x.one_reply_meets_both), replay_both)
        });
        for (reached, both, one, replay_both) in res { if reached { each_reached += 1; } if both { said_both += 1; } if replay_both { replay_ok += 1; } if let Some(v) = one { reply_read += 1; if v { reply_true += 1; } } }
    }
    println!("OBS 79.28 pairs: {} positions where the reference finds one line reaching two goals; the layer reaches each of the two alone in {each_reached}; it says one line reaches both in {said_both}, and the line replays with both patterns held along it in {replay_ok}; one reply meeting both was read in full in {reply_read} and is true in {reply_true}", pairs.len());
    assert_eq!(each_reached, pairs.len(), "each goal of a pair is reached at the top budget");
    assert_eq!(replay_ok, said_both, "every line said to reach two goals does");
}

/// For the reader (plan 79.28 part E): with LAYER4_ANSWERS_OUT=<file> the answers at the top budget are written, one line of JSON each, for discovery/layer4_lines_print.py to put into words. No other test reads it.
#[test]
fn the_answers_at_the_top_budget_are_written_for_the_reader_when_asked() {
    let Ok(out) = std::env::var("LAYER4_ANSWERS_OUT") else { return };
    let (games, rows) = table();
    let top = BUDGETS.len() - 1;
    let mut lines: Vec<String> = vec![];
    let opened: Vec<R> = games.iter().map(open).collect();
    for r in rows {
        let (fx, p) = (&games[r.game], &games[r.game].positions[r.pos]);
        let me = { let g = &opened[r.game]; g.mover(&g.read_state_text(&p.text).unwrap()).unwrap() };
        // what the line took from the other side and gave up of the side's own (owner of the unit taken against the side asking), and the judge's value of the line for the side
        let split = |t: &Vec<(Role, String)>| (t.iter().filter(|(o, _)| *o != me).map(|(_, k)| k.clone()).collect::<Vec<_>>(), t.iter().filter(|(o, _)| *o == me).map(|(_, k)| k.clone()).collect::<Vec<_>>());
        let mut o = cs_rep::serde_json::json!({"game": fx.name, "position": p.id, "text": p.text, "goal": r.goal, "kind": r.kind, "pattern": p.goals[r.goal].pattern.iter().map(|l| l.to_json()).collect::<Vec<_>>(), "reference": format!("{:?}", r.reference), "budget": BUDGETS[top]});
        let bind = |b: &Vec<cs_rep::matcher::Binding>| b.iter().take(3).map(|x| cs_rep::serde_json::json!(x)).collect::<Vec<_>>();
        match &r.list[top] {
            Answer::Reached(x) => { o["status"] = "reached".into(); o["line"] = x.line.clone().into(); o["plies"] = x.plies.into(); o["bindings"] = bind(&x.bindings).into(); let (took, gave) = split(&x.taken); o["took"] = took.into(); o["gave"] = gave.into(); o["value"] = x.value.into(); }
            Answer::Touched(x) => { o["status"] = "touched".into(); o["line"] = x.line.clone().into(); o["plies"] = x.plies.into(); o["reply"] = x.reply.clone().into(); o["bindings"] = bind(&x.bindings).into(); let (took, gave) = split(&x.taken); o["took"] = took.into(); o["gave"] = gave.into(); o["value"] = x.value.into(); }
            Answer::OutOfReach { depth } => { o["status"] = "out".into(); o["depth"] = (*depth).into(); }
            Answer::NotSettled { depth, .. } => { o["status"] = "not_settled".into(); o["depth"] = (*depth).into(); }
        }
        lines.push(o.to_string());
    }
    std::fs::write(&out, lines.join("\n") + "\n").unwrap();
    println!("OBS 79.28 answers written: {} lines to {out}", lines.len());
}

/// PLAN 79.29 PART A: the cost of the second reading, on the questions of 79.28 (goals.fixture) and of 79.27 (goals_79_27.fixture), one tree per position (the list of each position), at the three budgets: nodes in all, nodes of each reading, the goals settled and what the first reading knew where the second expanded.
/// With LAYER4_DUMP=<file> every answer of every goal is written (the status, the plies, the line, the reply, the bindings) one line each, so that the answers of two builds are compared by a program and not by eye.
#[test]
fn the_cost_of_the_second_reading_is_counted_where_it_goes_and_the_answers_are_dumped_when_asked() {
    let mut dump: Vec<String> = vec![];
    for name in ["goals.fixture", "goals_79_27.fixture"] {
        let games = read_goals_from(name);
        for &budget in BUDGETS.iter() {
            let mut nodes = 0u64;
            let mut diag = cs_tactics::reach::Diag::default();
            let mut stat = [0usize; 4];
            for fx in &games {
                let res: Vec<Reply> = par(fx, fx.positions.len(), &|g, st, p| {
                    let s = g.read_state_text(&fx.positions[p].text).unwrap();
                    ask(&question(g, st, &s, &holds_goals(&fx.positions[p]), budget, &Count))
                });
                for (p, r) in res.iter().enumerate() {
                    nodes += r.nodes;
                    let d = r.diag;
                    diag.calls_held += d.calls_held; diag.calls_touched += d.calls_touched; diag.calls_both += d.calls_both;
                    for (k, a) in r.answers.iter().enumerate() {
                        stat[status(a) as usize] += 1;
                        let body = match a {
                            Answer::Reached(x) => format!("reached\t{}\t{}\t{}\t{}", x.plies, x.line.join(" "), "", x.bindings.len()),
                            Answer::Touched(x) => format!("touched\t{}\t{}\t{}\t{}", x.plies, x.line.join(" "), x.reply, x.bindings.len()),
                            Answer::OutOfReach { depth } => format!("out\t{depth}\t\t\t0"),
                            Answer::NotSettled { depth, .. } => format!("not_settled\t{depth}\t\t\t0"),
                        };
                        dump.push(format!("{name}\t{budget}\t{}\t{}\t{k}\t{body}", fx.name, fx.positions[p].id));
                    }
                }
            }
            println!("OBS 79.29 {name} at {budget} nodes: nodes {nodes}; reached {}, touched {}, out of reach {}, not settled {}; settled {}", stat[0], stat[1], stat[2], stat[3], stat[0] + stat[1] + stat[2]);
            println!("OBS 79.29 {name} at {budget} nodes, calls of the tree asking held only {}, touched only {}, both {}", diag.calls_held, diag.calls_touched, diag.calls_both);
        }
    }
    if let Ok(out) = std::env::var("LAYER4_DUMP") { std::fs::write(&out, dump.join("\n") + "\n").unwrap(); println!("OBS 79.29 answers dumped: {} lines to {out}", dump.len()); }
}
