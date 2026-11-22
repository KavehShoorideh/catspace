//! PLAN 79.27 PART C: THE TEST OF LAYER 4 WITH GOALS (SYSTEM_SPEC 11 "Layer 4 takes a goal as an input" and its amendment, "a layer's test uses only that layer and what is below it"; ledger rows 36, 14; piece: tactical reasoner).
//! It uses layer 1 (cs_rep: statements, the board), layer 2 (cs_game: the game through the text door, `Placed`, the board) and this layer's one entry (`meansends::ask`). Every run is bounded by NODES, never by a clock. The goals are DATA (tests/fixtures/goals.fixture, made by
//! discovery/layer4_goals_print.py: positions of two games with lists of patterns that are neither a gain nor the end) and tests/fixtures/pairs.fixture (positions where one line reaches two goals, MADE BY THE REFERENCE below: run this file with LAYER4_MAKE_PAIRS=1 to write it).
//! THE REFERENCE is written apart, here: a slow plain search of every move (no memo, no order, no live graph: the facts of each position are made from nothing out of the game's own questions) with a naive matcher of its own. It answers each goal ALONE: reached in n plies, or not within the depth.
//! THE TESTS: the layer never says reached where the reference does not, with a line the rules replay to a position where the pattern binds; never out of reach where the reference reaches; at the top budget it says reached exactly where the reference does; the same answer asked twice; the list never
//! contradicts a goal asked alone; the narrowing answers as the plain tree where both settle; the two old goals answer as the ledger search did (list of one); counts at three budgets are kept as floors. This file names no object of any game.
use cs_game::graph::Static;
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::rules::{Ruled, StateText};
use cs_game::{Fact, Game, Role};
use cs_rep::history::Seen;
use cs_rep::lemma::{Arg, Lit};
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
/// THE FLOORS (what the layer did when this test was written, not a claim about its strength): over the goals of the fixture, at each budget, the goals the layer says reached, and the goals it settles (reached or out of reach).
const FLOOR_REACHED: [usize; 3] = [183, 189, 189];
const FLOOR_SETTLED: [usize; 3] = [228, 390, 400];

// ------------------------------------------------------------------------------------------------------------------------------ the fixtures

struct GoalPos { id: String, text: String, kinds: Vec<String>, goals: Vec<Statement> }
struct GameFx { name: String, decl: String, positions: Vec<GoalPos> }

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn statement(id: &str, pattern: Vec<Lit>) -> Statement { Statement { id: id.to_string(), pattern, says: Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: vec![] } }

fn read_goals() -> Vec<GameFx> {
    let mut out: Vec<GameFx> = vec![];
    for line in fixture("goals.fixture").lines() {
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

    /// Whether the pattern binds over the facts: the positive literals in the order written, then each negated literal must have no row.
    fn binds(pattern: &[Lit], facts: &[Fact]) -> bool {
        fn unify(l: &Lit, f: &Fact, b: &BTreeMap<String, String>) -> Option<BTreeMap<String, String>> {
            if l.relation != f.relation || l.args.len() != f.args.len() { return None; }
            let mut nb = b.clone();
            for (a, v) in l.args.iter().zip(&f.args) {
                match a { Arg::Const(c) => if c != v { return None; }, Arg::Var(x) => match nb.get(x) { Some(w) => if w != v { return None; }, None => { nb.insert(x.clone(), v.clone()); } } }
            }
            Some(nb)
        }
        fn go(i: usize, pos: &[&Lit], neg: &[&Lit], b: BTreeMap<String, String>, facts: &[Fact]) -> bool {
            if i == pos.len() { return !neg.iter().any(|n| facts.iter().any(|f| unify(n, f, &b).is_some())); }
            facts.iter().any(|f| unify(pos[i], f, &b).is_some_and(|nb| go(i + 1, pos, neg, nb, facts)))
        }
        let pos: Vec<&Lit> = pattern.iter().filter(|l| !l.neg).collect();
        let neg: Vec<&Lit> = pattern.iter().filter(|l| l.neg).collect();
        go(0, &pos, &neg, BTreeMap::new(), facts)
    }

    fn holds(&self, pattern: &[Lit], s: &<R as Game>::State) -> bool { Self::binds(pattern, &self.facts(s)) }

    /// Reached within `d` plies by `me`? Every move of the side, every reply; the pattern read first; a game that is over, a position met again, the depth limit end a line.
    fn reach(&self, pattern: &[Lit], s: &<R as Game>::State, me: Role, d: i32, path: &mut Vec<u64>) -> bool {
        if self.holds(pattern, s) { return true; }
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

    /// The fewest plies within the depth, or None.
    fn plies(&self, pattern: &[Lit], s: &<R as Game>::State, me: Role) -> Option<i32> { (0..=DEPTH).find(|&d| self.reach(pattern, s, me, d, &mut vec![])) }

    /// Both patterns bound along ONE line within `d` plies (the flags accumulate down the line; the other side's replies are all met).
    fn both(&self, a: &[Lit], b: &[Lit], s: &<R as Game>::State, me: Role, d: i32, fa: bool, fb: bool, path: &mut Vec<u64>) -> bool {
        let (fa, fb) = (fa || self.holds(a, s), fb || self.holds(b, s));
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

/// What a reached answer must be: a line the rules replay, ending where the pattern binds (read by the reference), within the depth, with bindings that ground the pattern.
fn confirm_reached(rf: &Reference, g: &R, s: &<R as Game>::State, pattern: &[Lit], r: &cs_tactics::meansends::Reached, id: &str) {
    assert!(r.plies as usize == r.line.len() && r.plies <= DEPTH, "{id}: the line is within the depth: {:?}", r.line);
    let mut cur = s.clone();
    for name in &r.line {
        let m = g.find_move(&cur, name).unwrap_or_else(|| panic!("{id}: the line's move {name} is legal in its place"));
        cur = g.apply(&cur, &m);
    }
    assert!(rf.holds(pattern, &cur), "{id}: the pattern binds where the line ends");
    assert!(!r.bindings.is_empty(), "{id}: the bindings are given");
    let facts: BTreeSet<Fact> = rf.facts(&cur).into_iter().collect();
    for b in &r.bindings {
        for l in pattern.iter().filter(|l| !l.neg) {
            let args: Vec<String> = l.args.iter().map(|a| match a { Arg::Const(c) => c.clone(), Arg::Var(v) => b[v].clone() }).collect();
            assert!(facts.contains(&Fact { relation: l.relation.clone(), args }), "{id}: a binding grounds a literal onto a fact of the end position");
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ the answers, read once

/// One goal alone: the reference's plies, and each budget's answer from the list.
struct Row { game: usize, pos: usize, goal: usize, kind: String, reference: Option<i32>, list: Vec<Answer>, alone: Vec<Answer> }

fn status(a: &Answer) -> u8 { match a { Answer::Reached(_) => 0, Answer::OutOfReach { .. } => 1, Answer::NotSettled { .. } => 2 } }

type Table = (Vec<GameFx>, Vec<Row>, Vec<Vec<Reply>>);

fn table() -> &'static Table {
    static CELL: std::sync::OnceLock<Table> = std::sync::OnceLock::new();
    CELL.get_or_init(|| {
        let games = read_goals();
        let mut rows = vec![];
        let mut replies: Vec<Vec<Reply>> = vec![];
        let t0 = Instant::now();
        let judge = Count;
        for (gi, fx) in games.iter().enumerate() {
            let n = fx.positions.len();
            // the reference, per goal alone (the slow part), on all the cores, over the flattened goals
            let flat: Vec<(usize, usize)> = (0..n).flat_map(|p| (0..fx.positions[p].goals.len()).map(move |g| (p, g))).collect();
            let refs: Vec<Option<i32>> = par(fx, flat.len(), &|g, st, i| {
                let (p, k) = flat[i];
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                let rf = Reference::new(g, st);
                rf.plies(&fx.positions[p].goals[k].pattern, &s, g.mover(&s).unwrap())
            });
            // the layer, the list at each budget and each goal alone at each budget
            let per: Vec<(Vec<Reply>, Vec<Vec<Reply>>)> = par(fx, n, &|g, st, p| {
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                let goals = holds_goals(&fx.positions[p]);
                let list: Vec<Reply> = BUDGETS.iter().map(|&b| ask(&question(g, st, &s, &goals, b, &judge))).collect();
                let alone: Vec<Vec<Reply>> = (0..goals.len()).map(|k| BUDGETS.iter().map(|&b| ask(&question(g, st, &s, &goals[k..k + 1], b, &judge))).collect()).collect();
                (list, alone)
            });
            let mut flat_i = 0;
            let base: Vec<usize> = vec![0; BUDGETS.len()];
            let _ = base;
            let mut by_budget: Vec<Vec<Reply>> = vec![vec![]; BUDGETS.len()];
            for (p, (list, alone)) in per.iter().enumerate() {
                for (b, rep) in list.iter().enumerate() { by_budget[b].push(rep.clone()); }
                for k in 0..fx.positions[p].goals.len() {
                    // one row per goal and budget, folded into one row whose vectors are by budget
                    rows.push(Row { game: gi, pos: p, goal: k, kind: fx.positions[p].kinds[k].clone(), reference: refs[flat_i], list: list.iter().map(|r| r.answers[k].clone()).collect(), alone: alone[k].iter().map(|r| r.answers[0].clone()).collect() });
                    flat_i += 1;
                }
            }
            replies.extend(by_budget);
        }
        println!("OBS 79.27 table: {} games, {} goals, the reference and the layer (list, and each goal alone) at budgets {:?} nodes, depth {DEPTH} plies, read on all the cores in {:.1} s", games.len(), rows.len(), BUDGETS, t0.elapsed().as_secs_f64());
        (games, rows, replies)
    })
}

// ------------------------------------------------------------------------------------------------------------------------------ the tests

#[test]
fn the_layer_agrees_with_the_reference_on_every_goal_at_three_budgets() {
    let (games, rows, _) = table();
    let mut kinds: BTreeSet<&str> = BTreeSet::new();
    for r in rows { kinds.insert(&r.kind); }
    let ref_reached = rows.iter().filter(|r| r.reference.is_some()).count();
    println!("OBS 79.27 reference: {} goals of {} kinds ({:?}) on {} games; the reference reaches {} within {DEPTH} plies and not {}", rows.len(), kinds.len(), kinds, games.len(), ref_reached, rows.len() - ref_reached);
    let mut by_game: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for r in rows { let e = by_game.entry(r.game).or_default(); e.0 += 1; if r.reference.is_some() { e.1 += 1; } }
    for (gi, (n, k)) in &by_game { println!("OBS 79.27 reference game {}: {n} goals, {k} reached", games[*gi].name); }
    let mut by_kind: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for r in rows { let e = by_kind.entry(&r.kind).or_default(); e.0 += 1; if r.reference.is_some() { e.1 += 1; } }
    for (k, (n, c)) in &by_kind { println!("OBS 79.27 reference kind {k}: {n} goals, {c} reached"); }
    for b in 0..BUDGETS.len() {
        let (mut reached, mut out, mut unsettled, mut wrong_reach, mut wrong_out, mut exact) = (0, 0, 0, 0, 0, 0);
        let mut line_gap = 0;
        for r in rows {
            let a = &r.list[b];
            match a {
                Answer::Reached(x) => { reached += 1; if r.reference.is_none() { wrong_reach += 1; } else if x.plies < r.reference.unwrap() { line_gap += 1; } }
                Answer::OutOfReach { .. } => { out += 1; if r.reference.is_some() { wrong_out += 1; } }
                Answer::NotSettled { .. } => unsettled += 1,
            }
            if (status(a) == 0) == r.reference.is_some() && status(a) != 2 { exact += 1; }
        }
        println!("OBS 79.27 list at {} nodes: of {} goals reached {reached}, out of reach {out}, not settled {unsettled}; reached where the reference does not {wrong_reach}; out of reach where it reaches {wrong_out}; answers equal to the reference {exact}; a line shorter than the reference's fewest {line_gap}", BUDGETS[b], rows.len());
        assert_eq!(wrong_reach, 0, "reached only where the reference reaches");
        assert_eq!(wrong_out, 0, "never out of reach where the reference reaches");
        assert_eq!(line_gap, 0, "a line is never shorter than the reference's fewest plies");
        assert!(reached >= FLOOR_REACHED[b] && reached + out >= FLOOR_SETTLED[b], "the floors: reached {reached} settled {}", reached + out);
        if b == BUDGETS.len() - 1 { assert_eq!(unsettled, 0, "at the top budget every goal of this depth is settled"); assert_eq!(exact, rows.len(), "at the top budget the layer says reached exactly where the reference does"); }
    }
}

#[test]
fn a_reached_answer_is_a_line_the_rules_replay_to_a_position_where_the_pattern_binds() {
    let (games, rows, _) = table();
    let mut confirmed = 0;
    for (gi, fx) in games.iter().enumerate() {
        let g = open(fx);
        let st = g.board().unwrap();
        let rf = Reference::new(&g, &st);
        for r in rows.iter().filter(|r| r.game == gi) {
            for b in 0..BUDGETS.len() {
                if let Answer::Reached(x) = &r.list[b] {
                    let p = &fx.positions[r.pos];
                    let s = g.read_state_text(&p.text).unwrap();
                    confirm_reached(&rf, &g, &s, &p.goals[r.goal].pattern, x, &format!("{}#{} at {}", p.id, r.goal, BUDGETS[b]));
                    confirmed += 1;
                }
            }
        }
    }
    println!("OBS 79.27 replay: {confirmed} reached answers (every goal, every budget) replayed by the rules to a position where the reference binds the pattern, with bindings grounding it");
    assert!(confirmed > 0);
}

#[test]
fn the_list_never_contradicts_a_goal_asked_alone_and_the_nodes_of_one_tree_are_counted_beside_k_searches() {
    let (games, rows, _) = table();
    let mut contradictions = 0;
    let mut differ = 0;
    for r in rows { for b in 0..BUDGETS.len() {
        let (l, a) = (status(&r.list[b]), status(&r.alone[b]));
        if l != 2 && a != 2 && l != a { contradictions += 1; }
        if l != a { differ += 1; }
    } }
    println!("OBS 79.27 list against alone: {} goals x {} budgets; answers settled in both that differ {contradictions}; answers whose status differs at all (one not settled) {differ}", rows.len(), BUDGETS.len());
    assert_eq!(contradictions, 0, "a goal settled both ways is settled the same way");
    // the nodes of one tree beside the nodes of k single searches, over the positions, at each budget
    let t0 = Instant::now();
    for (b, &budget) in BUDGETS.iter().enumerate() {
        let mut totals = (0u64, 0u64, 0u64, 0u64);
        for fx in games {
            let sums: Vec<(u64, u64, u64, u64)> = par(fx, fx.positions.len(), &|g, st, p| {
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                let goals = holds_goals(&fx.positions[p]);
                let one = ask(&question(g, st, &s, &goals, budget, &Count));
                let (mut n, mut mk, mut mt) = (0u64, 0u64, 0u64);
                for k in 0..goals.len() { let a = ask(&question(g, st, &s, &goals[k..k + 1], budget, &Count)); n += a.nodes; mk += a.makes; mt += a.matches; }
                (one.nodes, n, one.makes + one.matches * 0, mk + mt * 0)
            });
            for s in sums { totals.0 += s.0; totals.1 += s.1; totals.2 += s.2; totals.3 += s.3; }
        }
        println!("OBS 79.27 one tree beside k searches at {budget} nodes: nodes one tree {} against the sum of k single searches {}; positions made on the live graph {} against {}", totals.0, totals.1, totals.2, totals.3);
        let _ = b;
    }
    println!("OBS 79.27 one tree beside k searches read in {:.1} s", t0.elapsed().as_secs_f64());
}

#[test]
fn the_order_of_the_list_changes_how_the_budget_is_spent_and_never_a_settled_answer() {
    let (games, _rows, _) = table();
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
        for (fw, bw) in res { for k in 0..fw.len() { total += 1; for b in 0..3 { if fw[k][b] != bw[k][b] { changed[b] += 1; if fw[k][b] != 2 && bw[k][b] != 2 { contradictions += 1; } } } } }
    }
    println!("OBS 79.27 order: of {total} goals the answer changes when the list is reversed, at {} nodes {}, at {} nodes {}, at {} nodes {}; settled both ways and different {contradictions}", BUDGETS[0], changed[0], BUDGETS[1], changed[1], BUDGETS[2], changed[2]);
    assert_eq!(contradictions, 0);
}

#[test]
fn the_same_answer_asked_twice_and_the_narrowing_answers_as_the_plain_tree_where_both_settle() {
    let (games, _rows, _) = table();
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
                if x != 2 && y != 2 { both_settled += 1; if x != y { diff += 1; } }
            }
        }
    }
    println!("OBS 79.27 narrowing at {budget} nodes (one tree per position): nodes plain {nodes_plain} narrowed {nodes_narrow}; positions made plain {makes_plain} narrowed {makes_narrow}; rows matched plain {matches_plain} narrowed {matches_narrow}; seconds (summed over threads) plain {t_plain:.2} narrowed {t_narrow:.2}; goals settled by both {both_settled}, answers that differ {diff}; the narrowed tree asked twice equal {twice} of {}", games.iter().map(|g| g.positions.len()).sum::<usize>());
    assert_eq!(diff, 0, "the narrowing answers as the plain tree wherever both settle");
    assert_eq!(twice, games.iter().map(|g| g.positions.len()).sum::<usize>(), "the same answer asked twice");
}

#[test]
fn each_part_of_the_narrowing_alone_is_measured_beside_the_plain_tree_and_answers_as_it_does() {
    let (games, _rows, _) = table();
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
            for k in 0..v[0].0.answers.len() { for i in 1..4 { let (x, y) = (status(&v[0].0.answers[k]), status(&v[i].0.answers[k])); if x != 2 && y != 2 && x != y { diff += 1; } } }
        }
    }
    for (i, (name, _)) in variants.iter().enumerate() {
        println!("OBS 79.27 narrowing parts at {budget} nodes, {name}: nodes {} positions made {} rows matched {} seconds (summed over threads) {:.2}", totals[i].0, totals[i].1, totals[i].2, totals[i].3);
    }
    assert_eq!(diff, 0, "every variant answers as the plain tree where both settle");
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
    println!("OBS 79.27 old goals: of {} positions asked as a list of one, the gain answer is the ledger search's own {g_ok} times, the end answer {e_ok} times; nodes direct {} through the entry {}", res.len(), res.iter().map(|r| r.2).sum::<u64>(), res.iter().map(|r| r.3).sum::<u64>());
    assert_eq!(g_ok, res.len());
    assert_eq!(e_ok, res.len());
}

#[test]
fn positions_where_one_line_reaches_two_goals_are_read_from_the_fixture_the_reference_made() {
    let games = read_goals();
    if std::env::var("LAYER4_MAKE_PAIRS").is_ok() {
        let mut lines = vec!["note\tplan 79.27 part C: positions where ONE line reaches two goals of the goals fixture within the depth, found by the reference (both patterns bound along one line against every reply), each goal also reached alone; made by running layer4_goals.rs with LAYER4_MAKE_PAIRS=1".to_string()];
        for fx in &games {
            lines.push(format!("game\t{}\t{}", fx.name, fx.decl));
            let found: Vec<Vec<(usize, usize, usize)>> = par(fx, fx.positions.len(), &|g, st, p| {
                let s = g.read_state_text(&fx.positions[p].text).unwrap();
                let me = g.mover(&s).unwrap();
                let rf = Reference::new(g, st);
                let gs = &fx.positions[p].goals;
                let mut out = vec![];
                let alone: Vec<Option<i32>> = gs.iter().map(|x| rf.plies(&x.pattern, &s, me)).collect();
                for a in 0..gs.len() { for b in a + 1..gs.len() {
                    if alone[a].is_some() && alone[b].is_some() && rf.both(&gs[a].pattern, &gs[b].pattern, &s, me, DEPTH - 1, false, false, &mut vec![]) { out.push((p, a, b)); }
                } }
                out
            });
            for (p, a, b) in found.into_iter().flatten() { lines.push(format!("pair\t{}\t{}\t{}", fx.positions[p].id, a, b)); }
        }
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("pairs.fixture");
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        println!("OBS 79.27 pairs fixture written: {} pairs", lines.iter().filter(|l| l.starts_with("pair")).count());
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
            let goals = holds_goals(p);
            let rep = ask(&question(g, st, &s, &goals, top, &Count));
            let rf = Reference::new(g, st);
            let reached = rep.answers[*a].reached().is_some() && rep.answers[*b].reached().is_some();
            let both = rep.both.iter().find(|x| x.goals == (*a, *b));
            let replay = both.is_some_and(|x| {
                let mut cur = s.clone();
                let (mut fa, mut fb) = (rf.holds(&p.goals[*a].pattern, &cur), rf.holds(&p.goals[*b].pattern, &cur));
                for name in &x.line { let m = g.find_move(&cur, name).unwrap(); cur = g.apply(&cur, &m); fa |= rf.holds(&p.goals[*a].pattern, &cur); fb |= rf.holds(&p.goals[*b].pattern, &cur); }
                fa && fb
            });
            (reached, both.is_some(), both.and_then(|x| x.one_reply_meets_both), replay)
        });
        for (reached, both, one, replay) in res { if reached { each_reached += 1; } if both { said_both += 1; } if replay { replay_ok += 1; } if let Some(v) = one { reply_read += 1; if v { reply_true += 1; } } }
    }
    println!("OBS 79.27 pairs: {} positions where the reference finds one line reaching two goals; the layer reaches each of the two alone in {each_reached}; it says one line reaches both in {said_both}, and the line replays with both patterns binding along it in {replay_ok}; one reply meeting both was read in full in {reply_read} and is true in {reply_true}", pairs.len());
    assert_eq!(each_reached, pairs.len(), "each goal of a pair is reached at the top budget");
    assert_eq!(replay_ok, said_both, "every line said to reach two goals does");
}
