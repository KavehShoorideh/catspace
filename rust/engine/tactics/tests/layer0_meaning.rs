//! PLAN 79.31 PART F: THE MEANING OF LAYER 0 HELD AGAINST THE SEARCH OF LAYER 4, ON THE GAME AT HAND (SYSTEM_SPEC 11 "Layer 0 is made right", "a layer's test uses only that layer and what is below it"; ledger rows 36, 37; piece: tactical reasoner against the base).
//! WHAT IT DOES. On the positions of tests/fixtures/goals.fixture (the game at hand, a few units, a shallow depth) the tree of choices is UNFOLDED BY THE RULES into cs_base's `Tree`: a position is a choice in turn by the chooser that is to move, its options are the legal replies, and a line ends where
//! the game is over, where the depth is spent, where a position is met again on the line (the rules' repeat) or where nothing can be played. The result at a line's end says which goals were HELD somewhere along the line. A goal's pattern is matched by cs_base's own matcher over the facts of the position.
//! THE MEANING (layer 4's, plan 79.28): a pattern is HELD at a position where it binds and the side asking is to move, or the game is over. REACHED: on every reply a held position is met within the depth, the root counting at nought plies. The layer 0 reading of the same: the set of one, the asking side, CAN FORCE
//! "the line met a held position" (`cs_base::choice::can_force_set`). It is compared with `meansends::ask`'s answer at the same depth: Reached against can-force; OutOfReach and Touched against cannot-force. TOUCHED is a separate answer and is not compared here. An answer the budget cut (NotSettled) is not compared and is counted.
//! Every disagreement is printed with the position and the goal. This file names no object of any game; the fixtures do.
use cs_base::choice::{can_force_set, party, Chooser, Tree};
use cs_base::logic::{bindings, Binding, Fact, FactIndex, Lit};
use cs_game::graph::Static;
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::rules::{Ruled, StateText};
use cs_game::{Game, Role};
use cs_rep::history::Seen;
use cs_rep::lemma::LitJson;
use cs_rep::statement::{Holder, Says, Standing, Statement};
use cs_tactics::ledger::Count;
use cs_tactics::meansends::{ask, Answer, Goal, Question};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

type R = Ruled<KernelGame>;

/// The depths in plies compared; the tree is made in full, so the deepest is a shallow one (the side, the reply, the side).
const DEPTHS: [i32; 4] = [0, 1, 2, 3];
const NODES: u64 = 3_000_000;

struct GoalPos { id: String, text: String, goals: Vec<Vec<Lit>> }
struct GameFx { name: String, decl: String, positions: Vec<GoalPos> }

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn read_goals(name: &str) -> Vec<GameFx> {
    let mut out: Vec<GameFx> = vec![];
    for line in fixture(name).lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "note" => {}
            "game" => out.push(GameFx { name: f[1].into(), decl: f[2].into(), positions: vec![] }),
            "pos" => {
                let v: cs_rep::serde_json::Value = cs_rep::serde_json::from_str(f[3]).unwrap();
                let goals = v.as_array().unwrap().iter().map(|g| g["pattern"].as_array().unwrap().iter().map(|l| Lit::from_json(l).unwrap()).collect()).collect();
                out.last_mut().unwrap().positions.push(GoalPos { id: f[1].into(), text: f[2].into(), goals });
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

/// The facts of a position, from the game's own questions (the board's lines and edges, whose turn it is, each unit with what it bears on and where it may move).
struct Facts<'a> { g: &'a R, roles: Vec<String>, kinds: Vec<String>, places: Vec<String>, statics: Vec<Fact> }

impl<'a> Facts<'a> {
    fn new(g: &'a R, st: &Static) -> Facts<'a> {
        let mut statics = vec![];
        for (li, l) in st.lines.iter().enumerate() {
            let id = st.nodes[st.line_node[li] as usize].id.clone();
            for p in &l.places { statics.push(Fact { relation: "on_line".into(), args: vec![st.names.places[*p as usize].clone(), id.clone()] }); }
        }
        for (d, name) in st.directions.iter().enumerate() {
            for p in 0..st.names.places.len() { if st.edge_distance[d][p] == 0 { statics.push(Fact { relation: "on_edge".into(), args: vec![st.names.places[p].clone(), name.clone()] }); } }
        }
        Facts { g, roles: g.roles(), kinds: g.kind_names(), places: g.place_names(), statics }
    }
    fn of(&self, s: &<R as Game>::State) -> Vec<Fact> {
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
}

/// What a line of the unfolded tree ends with: which goals were held somewhere along it (a bit for each goal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Held { mask: u64 }

/// The tree of choices from a position, by the rules, to `depth` plies. `strict` is the meaning (held where the asker is to move or the game is over); `strict = false` is a NEGATIVE CONTROL, the older meaning (bound anywhere along the line), which must disagree. A choice in turn by the chooser that is to move (the role's index is the chooser's identity).
#[allow(clippy::too_many_arguments)]
fn unfold(g: &R, facts: &Facts, s: &<R as Game>::State, me: Role, depth: i32, path: &mut Vec<u64>, goals: &[Vec<Lit>], so_far: u64, strict: bool) -> Tree<Held> {
    let over = g.outcome(s).is_some();
    let mut mask = so_far;
    let all = (1u64 << goals.len()) - 1;
    if mask != all && (!strict || over || g.mover(s) == Some(me)) {
        let idx = FactIndex::new(&facts.of(s));
        for (k, pat) in goals.iter().enumerate() {
            if mask >> k & 1 == 0 && !bindings(pat, &Binding::new(), &idx, 1_000_000, &mut 0).0.is_empty() { mask |= 1 << k; }
        }
    }
    if mask == all || over || depth == 0 { return Tree::Result(Held { mask }); }
    let key = g.key(s);
    if !path.is_empty() && path.contains(&key) { return Tree::Result(Held { mask }); }
    let moves = g.moves(s);
    if moves.is_empty() { return Tree::Result(Held { mask }); }
    path.push(key);
    let options: Vec<Tree<Held>> = moves.iter().map(|m| unfold(g, facts, &g.apply(s, m), me, depth - 1, path, goals, mask, strict)).collect();
    path.pop();
    Tree::turn(Chooser::new(g.mover(s).expect("a position the game goes on from") as u32), options).expect("a move was played")
}

/// PLAN 79.32 PART B: with LAYER0_BASELINE=<file> the whole reply of every question asked here is written (sorted), for the comparison of two builds by a program.
static DUMP: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Row { game: String, pos: String, goal: usize, depth: i32, layer0: bool, control: bool, with_control: bool, ask: Option<bool>, ask_plies: Option<i32> }

fn statement(id: &str, pattern: Vec<Lit>) -> Statement { Statement { id: id.to_string(), pattern, says: Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: vec![] } }

fn compare(fx: &GameFx, with_control: bool) -> Vec<Row> {
    let per: Vec<Vec<Row>> = par(fx, fx.positions.len(), &|g, st, i| {
        let p = &fx.positions[i];
        let s = g.read_state_text(&p.text).unwrap();
        let me = g.mover(&s).expect("a position the game goes on from");
        let facts = Facts::new(g, st);
        let goals: Vec<Goal> = p.goals.iter().enumerate().map(|(k, pat)| Goal::Holds(statement(&format!("{}#{k}", p.id), pat.clone()))).collect();
        let mut rows = vec![];
        for &d in &DEPTHS {
            let tree = unfold(g, &facts, &s, me, d, &mut vec![], &p.goals, 0, true);
            let wrong = if with_control { unfold(g, &facts, &s, me, d, &mut vec![], &p.goals, 0, false) } else { Tree::Result(Held { mask: 0 }) };
            let q = Question { game: g, board: st, state: &s, judge: &Count, side: me, goals: &goals, nodes: NODES, seconds: None, max_depth: d, history: Arc::new(Seen::new()) };
            let reply = ask(&q);
            if std::env::var("LAYER0_BASELINE").is_ok() { DUMP.lock().unwrap().push(format!("{}\t{}\t{d}\t{reply:?}", fx.name, p.id)); }
            for k in 0..p.goals.len() {
                let layer0 = can_force_set(&tree, &party(&[Chooser::new(me as u32)]), &|h: &Held| h.mask >> k & 1 == 1);
                let (a, plies) = match &reply.answers[k] { Answer::Reached(r) => (Some(true), Some(r.plies)), Answer::NotSettled { .. } => (None, None), _ => (Some(false), None) };
                let control = can_force_set(&wrong, &party(&[Chooser::new(me as u32)]), &|h: &Held| h.mask >> k & 1 == 1);
                rows.push(Row { game: fx.name.clone(), pos: p.id.clone(), goal: k, depth: d, layer0, control, with_control, ask: a, ask_plies: plies });
            }
        }
        rows
    });
    per.into_iter().flatten().collect()
}

#[test]
fn the_layer_0_reading_of_reached_is_the_answer_layer_4_gives_on_the_game_at_hand() {
    let t0 = Instant::now();
    let mut rows: Vec<Row> = vec![];
    for (file, with_control) in [("goals.fixture", true), ("goals_79_27.fixture", false)] {
        for fx in &read_goals(file) {
            let mut got = compare(fx, with_control);
            for r in got.iter_mut() { r.game = format!("{file}:{}", r.game); }
            rows.extend(got);
        }
    }
    let positions: std::collections::BTreeSet<(String, String)> = rows.iter().map(|r| (r.game.clone(), r.pos.clone())).collect();
    let goals: std::collections::BTreeSet<(String, String, usize)> = rows.iter().map(|r| (r.game.clone(), r.pos.clone(), r.goal)).collect();
    let mut disagree = vec![];
    let (mut compared, mut unsettled, mut forced, mut shorter_or_longer, mut control_apart, mut controlled) = (0, 0, 0, 0, 0, 0);
    for r in &rows {
        match r.ask {
            None => unsettled += 1,
            Some(a) => {
                compared += 1;
                if r.layer0 { forced += 1; }
                if r.with_control && a != r.control { control_apart += 1; controlled += 1; } else if r.with_control { controlled += 1; }
                if a != r.layer0 { disagree.push(format!("{} {} goal {} depth {}: layer 0 can force {}, layer 4 says reached {}", r.game, r.pos, r.goal, r.depth, r.layer0, a)); }
            }
        }
    }
    // the fewest depth at which each goal is forced, against the plies layer 4 gives at the deepest depth asked
    let deepest = *DEPTHS.last().unwrap();
    for r in rows.iter().filter(|r| r.depth == deepest && r.ask_plies.is_some()) {
        let fewest = DEPTHS.iter().copied().find(|&d| rows.iter().any(|x| x.game == r.game && x.pos == r.pos && x.goal == r.goal && x.depth == d && x.layer0));
        if fewest != r.ask_plies { shorter_or_longer += 1; println!("PLIES {} {} goal {}: layer 0 first forces at {:?} plies, layer 4 gives a line of {:?}", r.game, r.pos, r.goal, fewest, r.ask_plies); }
    }
    println!("OBS 79.31 layer 0 against layer 4: {} positions, {} goals, depths {:?}; {} answers compared ({} settled by layer 4 and not compared: budget), layer 0 can force in {}, disagreements {}, plies that differ {}, in {:.1} s",
        positions.len(), goals.len(), DEPTHS, compared, unsettled, forced, disagree.len(), shorter_or_longer, t0.elapsed().as_secs_f64());
    println!("OBS 79.31 negative control (a held position read where the pattern binds anywhere on the line, the meaning of plan 79.27): layer 4 and the control apart in {control_apart} of {controlled} answers");
    for d in &disagree { println!("DISAGREE {d}"); }
    if let Ok(out) = std::env::var("LAYER0_BASELINE") { let mut v = DUMP.lock().unwrap().clone(); v.sort(); std::fs::write(&out, v.join("\n") + "\n").unwrap(); println!("OBS 79.32 baseline written: {} questions to {out}", v.len()); }
    assert!(control_apart > 0, "the control must disagree: the test is able to tell the meanings apart");
    assert!(compared > 0 && forced > 0 && forced < compared, "the goals are of both kinds");
    assert!(disagree.is_empty(), "{} disagreements: {:?}", disagree.len(), &disagree[..disagree.len().min(5)]);
}
