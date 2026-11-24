//! PLAN 79.51 PART H (SYSTEM_SPEC 7 "Future prediction: one channel, two sub-layers"; piece: tactical reasoning, opportunity policy; ledger row 50): THE COUNTING over two lists of positions that exist as files, all cores. A MEASURE, not a gate: with the
//! environment variable FUTURE_LISTS unset the test says so and passes. With it set to the folder that holds `puzzles300.txt` (position, first move of the solution, id: the 300 puzzles of the measure) and `games200.txt` (position, game id, ply:
//! 200 positions of our own logged games), made by discovery/future_79_51_lists_print.py:
//!   cargo test --release -p cs_decision --test future_count -- --nocapture --test-threads=1
//! For each position the channel is read (FutureParams: the clock FUTURE_SECONDS, default 2.0 s a position) and counted: a goal on a line ALIVE, a THREAT NOW, a JOIN, a target KEPT and LEFT, a statement the clock did not reach. For the puzzles: the first
//! move of the solution against the moves the channel proposes (the first moves of its threats now) and against the moves of the list of today (the first moves of the lines of the threat list's rows against them and the moves of the fixes of our goals, read
//! from the graph alone), and the near read of every threat now (does the line read close it: a confirmed verdict). The first ten threats now that the near read does not close are printed with their line, to be read against the board.
//! FUTURE_OUT, when set, names a file the per-position statements are written to as JSON lines.
use cs_decision::future::{self, FutureParams, Stop};
use cs_decision::threat_verdict::Record;
use cs_decision::threats::{concrete_threats, ThreatParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }

/// What one position gave.
#[derive(Default, Clone)]
struct Row {
    ok: bool, legal: usize, targets: usize, kept: usize, left: usize, goals: usize, alive: usize, threat_now: usize, joins: usize, not_asked: usize, seconds: f64,
    closes: usize, not_closes: usize, unread_near: usize, first_takes: usize, multi: usize, opened: usize, split: [usize; 4], causes: Vec<&'static str>, proposed: BTreeSet<String>, today: BTreeSet<String>, solution: Option<String>, false_lines: Vec<String>, json: String,
}

thread_local! { static WORLD: (KernelGame, Static) = world(); }

fn one(fen: &str, solution: Option<&str>, seconds: f64) -> Row { WORLD.with(|(g, st)| one_in(g, st, fen, solution, seconds)) }

fn one_in(g: &KernelGame, st: &Static, fen: &str, solution: Option<&str>, seconds: f64) -> Row {
    let Ok(game) = KernelGame::from_fen("chess", fen) else { return Row::default() };
    let s = game.initial();
    if g.mover(&s).is_none() { return Row::default(); }
    let rec = Record { worth: table(), half: HashMap::new() };
    let fp = FutureParams { goal: Some(kind(g, "king")), seconds, ..FutureParams::default() };
    let r = future::read(st, g, &s, &rec, &fp);
    let mut row = Row { ok: true, legal: g.moves(&s).len(), targets: r.targets.len(), goals: r.goals.len(), alive: r.goals.iter().filter(|x| x.alive).count(), joins: r.joins.len(), not_asked: r.not_asked.len(), seconds: r.seconds, solution: solution.map(|x| x.to_string()), ..Row::default() };
    for t in &r.targets { match t.stop { Stop::Kept { .. } => row.kept += 1, Stop::Left { .. } => row.left += 1 } }
    for gl in &r.goals {
        if gl.opened_for_them.is_some() { row.opened += 1; }
        let Some(tn) = &gl.threat_now else { continue };
        if gl.facts.len() >= 2 { row.multi += 1; }
        row.threat_now += 1;
        row.proposed.insert(tn.first.mv.clone());
        if g.find_move(&s, &tn.first.mv).is_some_and(|m| !g.captured(&s, &m).is_empty()) { row.first_takes += 1; }
        let took = g.find_move(&s, &tn.first.mv).is_some_and(|m| !g.captured(&s, &m).is_empty());
        if let Some(n) = &gl.near { if !n.cut && n.verdict != "unread" { row.split[(if took { 0 } else { 2 }) + if n.closes { 0 } else { 1 }] += 1; } }
        match &gl.near { Some(n) if n.cut || n.verdict == "unread" => row.unread_near += 1, Some(n) if n.closes => row.closes += 1,
            Some(n) => { row.not_closes += 1; let cause = cause_of(g, &s, &tn.first.mv, &n.line, gl.target.r.at, tn.first.lands); row.causes.push(cause); row.false_lines.push(format!("[{cause}] goal {} takes {} (worth {}): first {} ({} {}, lands {}, taken there {}, held {}); rest {}; near read: line {}, net {}, {}", gl.bearer.text, gl.target.text, gl.stake, tn.first.mv, tn.first.mover.text, tn.first.how, tn.first.lands_text, tn.first.taken_there, tn.first.held, tn.rest.iter().map(|(_, m)| m.mv.clone()).collect::<Vec<_>>().join(","), n.line.join(" "), n.net, n.verdict)); }
            None => {} }
    }
    // the list of today: the first moves of the lines of the threat list's rows against them, and the moves of the fixes of our goals (the graph alone)
    let tp = ThreatParams { goal: fp.goal, ..ThreatParams::default() };
    let list = concrete_threats(st, g, &s, &rec, &tp);
    let mine = cs_decision::after::their_goals_with(st, g, &s, &rec, fp.goal, g.mover(&s).unwrap(), true);
    // the moves the wake names for the search today from its chains of goals of ours (the function the wake uses), and the first moves of the lines of the rows against them that are confirmed or an exchange
    for m in cs_decision::rule::names_of(&mine.chains, cs_decision::rule::NAMED_BUDGET) { row.today.insert(m); }
    row.today.retain(|_| true);
    let rows_named: BTreeSet<String> = list.other.iter().chain(list.exchange.iter()).filter(|r| matches!(r.verdict, cs_decision::threat_verdict::VerdictKind::Confirmed | cs_decision::threat_verdict::VerdictKind::Exchange)).filter_map(|r| r.line.first().cloned()).collect();
    row.today = row.today.iter().cloned().chain(rows_named).collect();
    row.json = serde_json::json!({"position": fen, "future": r.to_json()}).to_string();
    row
}

/// Why the near read does not close a threat now: what the other side's first reply in the printed line does.
fn cause_of(g: &KernelGame, s: &<KernelGame as Game>::State, first: &str, line: &[String], target_at: cs_game::placed::Place, lands: cs_game::placed::Place) -> &'static str {
    let Some(m1) = g.find_move(s, first) else { return "the first move is not legal" };
    let c1 = g.apply(s, &m1);
    if g.outcome(&c1).is_some() { return "the first move ends the game"; }
    let Some(reply) = line.get(1).and_then(|n| g.find_move(&c1, n)) else { return "no reply in the line" };
    let (from, to) = g.move_ends(&c1, &reply);
    let took = g.taken_at(&c1, &reply);
    if from == target_at { "the reply moves the target away" }
    else if took.is_empty() { "the reply takes nothing: the loss or the gain that fails comes later in the line" }
    else if to == lands { "the reply takes the unit that moved" }
    else { "the reply takes another unit of ours: a threat against us stands that the first move does not meet" }
}

fn read_list(path: &std::path::Path) -> Vec<(String, Option<String>)> {
    std::fs::read_to_string(path).unwrap_or_default().lines().filter(|l| !l.is_empty()).map(|l| { let p: Vec<&str> = l.split('\t').collect(); (p[0].to_string(), if path.ends_with("puzzles300.txt") { Some(p[1].to_string()) } else { None }) }).collect()
}

fn summarize(name: &str, rows: &[Row]) {
    let n = rows.iter().filter(|r| r.ok).count();
    let c = |f: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| r.ok && f(r)).count();
    println!("OBS future_79_51 count {name}: {n} positions read; targets a position {:.2}, goals on a line {:.2}, legal moves {:.1}, seconds {:.3}", rows.iter().map(|r| r.targets).sum::<usize>() as f64 / n.max(1) as f64, rows.iter().map(|r| r.goals).sum::<usize>() as f64 / n.max(1) as f64, rows.iter().map(|r| r.legal).sum::<usize>() as f64 / n.max(1) as f64, rows.iter().map(|r| r.seconds).sum::<f64>() / n.max(1) as f64);
    println!("OBS future_79_51 count {name}: a goal on a line alive in {} positions; a threat now in {}; a join in {}; a target kept in {}; a target left in {}; the clock left a target unread in {}", c(&|r| r.alive > 0), c(&|r| r.threat_now > 0), c(&|r| r.joins > 0), c(&|r| r.kept > 0), c(&|r| r.left > 0), c(&|r| r.not_asked > 0));
    let mut by: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for r in rows { for c in &r.causes { *by.entry(c).or_insert(0) += 1; } }
    println!("OBS future_79_51 count {name}: the first move is itself a taking in {} of the threats now; the false threats by what the first reply does: {:?}", rows.iter().map(|r| r.first_takes).sum::<usize>(), by);
    let sp = |i: usize| rows.iter().map(|r| r.split[i]).sum::<usize>();
    println!("OBS future_79_51 count {name}: threats now with two or more unmet facts {}; goals alive but the line opens for them too (not a threat now) {}; first move a taking: closed {} not closed {}; first move quiet: closed {} not closed {}", rows.iter().map(|r| r.multi).sum::<usize>(), rows.iter().map(|r| r.opened).sum::<usize>(), sp(0), sp(1), sp(2), sp(3));
    let tn: usize = rows.iter().map(|r| r.threat_now).sum();
    println!("OBS future_79_51 count {name}: {tn} goals are a threat now; the near read closes {} of them, does not close {} (false threats), cut or unread {}; goals alive {}, goals {}", rows.iter().map(|r| r.closes).sum::<usize>(), rows.iter().map(|r| r.not_closes).sum::<usize>(), rows.iter().map(|r| r.unread_near).sum::<usize>(), rows.iter().map(|r| r.alive).sum::<usize>(), rows.iter().map(|r| r.goals).sum::<usize>());
    if rows.iter().any(|r| r.solution.is_some()) {
        let has = |r: &Row| r.ok && r.solution.is_some();
        let n = rows.iter().filter(|r| has(r)).count();
        let hit_prop = rows.iter().filter(|r| has(r) && r.proposed.contains(r.solution.as_ref().unwrap())).count();
        let hit_today = rows.iter().filter(|r| has(r) && r.today.contains(r.solution.as_ref().unwrap())).count();
        let hit_new = rows.iter().filter(|r| has(r) && r.proposed.contains(r.solution.as_ref().unwrap()) && !r.today.contains(r.solution.as_ref().unwrap())).count();
        let with_prop = rows.iter().filter(|r| has(r) && !r.proposed.is_empty()).count();
        println!("OBS future_79_51 count {name}: of {n} puzzles the first move of the solution is proposed by the channel in {hit_prop}, named by the list of today in {hit_today}, proposed by the channel and not by the list of today in {hit_new}; the channel proposed some move in {with_prop} puzzles ({:.2} moves a puzzle where it proposed any), the list of today {:.1} moves a puzzle",
            rows.iter().filter(|r| has(r) && !r.proposed.is_empty()).map(|r| r.proposed.len()).sum::<usize>() as f64 / with_prop.max(1) as f64, rows.iter().filter(|r| has(r)).map(|r| r.today.len()).sum::<usize>() as f64 / n.max(1) as f64);
    }
    let mut shown = 0;
    for (i, r) in rows.iter().enumerate() { for l in &r.false_lines { if shown < 10 { println!("OBS future_79_51 false threat {} of {name} (position {i}): {l}", shown + 1); shown += 1; } } }
}

/// The board as text (upper case: the first side), for a reader to read the statements against.
fn board_text(fen: &str) -> String {
    let mut out = String::new();
    for (i, row) in fen.split(' ').next().unwrap_or("").split('/').enumerate() {
        out.push_str(&format!("      {} ", 8 - i));
        for c in row.chars() { if let Some(d) = c.to_digit(10) { for _ in 0..d { out.push('.'); } } else { out.push(c); } }
        out.push('\n');
    }
    out.push_str("        abcdefgh\n");
    out
}

/// FUTURE_SHOW=puzzles300.txt:3,17 prints the statements of those positions of a list (the board, the side to move, the solution, every statement).
fn show(dir: &str, spec: &str, seconds: f64) {
    let Some((name, ixs)) = spec.split_once(':') else { return };
    let list = read_list(&std::path::Path::new(dir).join(name));
    for ix in ixs.split(',').filter_map(|x| x.parse::<usize>().ok()) {
        let (fen, sol) = &list[ix];
        println!("OBS future_79_51 case {name} {ix}: position {fen}, the solution's first move {:?}\n{}", sol, board_text(fen));
        WORLD.with(|(g, st)| {
            let s = KernelGame::from_fen("chess", fen).unwrap().initial();
            let rec = Record { worth: table(), half: HashMap::new() };
            let r = future::read(st, g, &s, &rec, &FutureParams { goal: Some(kind(g, "king")), seconds, ..FutureParams::default() });
            for l in r.said() { println!("OBS future_79_51 case {name} {ix}: {l}"); }
        });
    }
}

#[test]
fn h_the_counting_over_the_puzzles_and_the_games() {
    let Ok(dir) = std::env::var("FUTURE_LISTS") else { println!("OBS future_79_51 count: FUTURE_LISTS is not set: the counting is a measure and is skipped"); return; };
    let seconds: f64 = std::env::var("FUTURE_SECONDS").ok().and_then(|x| x.parse().ok()).unwrap_or(2.0);
    if let Ok(spec) = std::env::var("FUTURE_SHOW") { show(&dir, &spec, seconds); return; }
    let mut dump = String::new();
    for name in ["puzzles300.txt", "games200.txt"] {
        let list = read_list(&std::path::Path::new(&dir).join(name));
        let t0 = std::time::Instant::now();
        let rows: Vec<Row> = list.par_iter().map(|(fen, sol)| one(fen, sol.as_deref(), seconds)).collect();
        println!("OBS future_79_51 count {name}: {} positions in {:.1} s of wall on all cores, {:.1} s the clock a position", list.len(), t0.elapsed().as_secs_f64(), seconds);
        summarize(name, &rows);
        for r in &rows { if r.ok { dump.push_str(&r.json); dump.push('\n'); } }
    }
    if let Ok(out) = std::env::var("FUTURE_OUT") { std::fs::write(out, dump).unwrap(); }
}

/// THE CHANNEL ADDS NOTHING, ON THE LISTED POSITIONS (the proof of the fixture file on more positions): at a fixed depth (the clock is far above what the depth needs, so the count of nodes is fixed) the decision with the channel asked and without it agree
/// in the move, value, depth, nodes, chain, the moves named and held equal and every field of the log but the channel's. The first FUTURE_SAME positions of each list (default 40), all cores.
#[test]
fn h_the_decisions_are_the_same_with_the_channel_on_and_off_on_the_listed_positions() {
    use cs_decision::memory::{Parts, WakeMemory};
    use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
    let Ok(dir) = std::env::var("FUTURE_LISTS") else { println!("OBS future_79_51 same: FUTURE_LISTS is not set: skipped"); return; };
    let n: usize = std::env::var("FUTURE_SAME").ok().and_then(|x| x.parse().ok()).unwrap_or(40);
    let mut all: Vec<String> = vec![];
    for name in ["puzzles300.txt", "games200.txt"] { all.extend(read_list(&std::path::Path::new(&dir).join(name)).into_iter().take(n).map(|x| x.0)); }
    let differ: Vec<String> = all.par_iter().filter_map(|fen| WORLD.with(|(g, st)| {
        let judge = cs_tactics::ledger::Worth { name: "human".into(), table: table(), ours_first: false };
        let s = KernelGame::from_fen("chess", fen).ok()?.initial();
        g.mover(&s)?;
        let params = |future: Option<FutureParams>| WakeParams { seconds: 600.0, depth: 3, pieces: Pieces { carried: true, frames: true, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: true, heads: false, after_move: true, direct_names: true },
            inputs: WakeInputs { memory: WakeMemory::given(Parts { record: Some(Record { worth: table(), half: HashMap::new() }), ..Default::default() }, None), goal: Some(kind(g, "king")), budget: 4, future, ..WakeInputs::default() }, ..WakeParams::default() };
        let off = decide(st, g, &s, &[], &judge, &params(None));
        let on = decide(st, g, &s, &[], &judge, &params(Some(FutureParams { goal: Some(kind(g, "king")), seconds: 20.0, ..FutureParams::default() })));
        let strip = |d: &cs_decision::wake::WakeDecision| { let mut v = d.log.to_json(); for k in ["step_seconds", "depth_reads", "future"] { v.as_object_mut().unwrap().remove(k); } v };
        let same = off.mv == on.mv && off.value == on.value && off.depth == on.depth && off.nodes == on.nodes && off.chain == on.chain && off.named_mine == on.named_mine && off.named_theirs == on.named_theirs && off.equal == on.equal && off.first == on.first && strip(&off) == strip(&on) && on.log.future.is_some() && off.log.future.is_none();
        if same { None } else { Some(fen.clone()) }
    })).collect();
    println!("OBS future_79_51 same: {} listed positions decided at depth 3 with the channel on and off; the decisions differ in {} ({:?})", all.len(), differ.len(), differ);
    assert!(differ.is_empty());
}
