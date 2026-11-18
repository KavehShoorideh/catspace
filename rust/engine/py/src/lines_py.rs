//! PLAN 78.72: the Python entry of the line read (cs_tactics::lines). Game-free: every name arrives as data (the kinds' worths of the caller's measured record, the places as numbers); the game is read
//! through `Placed`. `read_lines(variant, games, queries, worth, depth, fuse, end_plies, end_fuse, threads)`: a game is (start text or None, the moves played); each query is
//! (game index, mode, passed, me_first, a, b, c) and is asked at the position after ALL the game's moves, with the first mover of the query's position being the side to move there (`passed` = 0) or the
//! other side after the side to move passes once (`passed` = 1; a game that gives no pass answers {"ok": False}). `me_first` = 1: the line is read for the side that makes its first move; 0: for the other side.
//! Modes: 0 the takings on place `a`; 1 the move from `a` to `b`; 2 every legal move; 3 the quiet moves after which a taking on place `a` stands (the other side passing once); 4 the forced end of the game
//! for the mover of the position (the forcing read); 5 the moves of the mover after which the other side no longer forces the end. Returns, per query, {"ok", "reads": [...]} or {"ok", "forced": ...} or
//! {"ok", "answers": [...]}. Mode 9 (plan 78.77): the line that starts with the LAST move of the game's list, read from the position before it for the side that played it (the move's own events, then the best replies); `kind_change` False reads the judge with the event of a unit changing kind taken out. Modes 6 and 7 (a response, or every legal move, read two ways: the whole-board line and the taking lines of the unit on place  after it): {"afters": [...]}; mode 8: the best line of the mover. The judge of the lines is the caller's `worth` (the measured record of the kinds); mode 4 and 5 read the game's own end only.
use crate::base_graph_py::{bad, run_acc, GameWire};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::ledger::{Judge, NoChange, Worth};
use cs_tactics::lines::{self as L, Forced, LineParams, LineRead};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;

type Query = (usize, u8, u8, u8, u16, u16, u16);

enum Res { NoPass, Lines(Vec<LineRead>), Afters(Vec<L::After>), Forced(Forced), Answers(Vec<((u16, u16), String, u8)>) }

fn read_game<J: Judge>(variant: &str, game: &GameWire, qs: &[(usize, Query)], judge: &J, p: LineParams, end_plies: u32, end_fuse: u64, out: &mut Vec<(usize, Res)>) -> Result<(), String> {
    let g = match &game.0 { Some(text) => KernelGame::from_fen(variant, text)?, None => KernelGame::new(variant, 518)? };
    let mut s = g.initial();
    let mut last = None;                                          // (the position before the game's last move, and that move: mode 9 reads the line that starts with it)
    for (ply, name) in game.1.iter().enumerate() {
        let mv = g.find_move(&s, name).ok_or_else(|| format!("ply {ply}: {name} is not a legal move"))?;
        let next = g.apply(&s, &mv);
        last = Some((s, mv));
        s = next;
    }
    for (qi, (_gi, mode, passed, me_first, a, b, c)) in qs {
        if *mode == 9 {
            match &last {
                Some((before, mv)) => match g.mover(before) { Some(who) => out.push((*qi, Res::Lines(vec![L::read_line(&g, judge, before, mv, who, p)]))), None => out.push((*qi, Res::NoPass)) },
                None => out.push((*qi, Res::NoPass)),
            }
            continue;
        }
        let p0 = if *passed == 1 { match g.pass(&s) { Some(x) => x, None => { out.push((*qi, Res::NoPass)); continue; } } } else { s.clone() };
        let Some(mover) = g.mover(&p0) else { out.push((*qi, Res::NoPass)); continue };
        let me = if *me_first == 1 { mover } else { g.other(mover) };
        let r = match mode {
            0 => Res::Lines(L::read_takings(&g, judge, &p0, *a, me, p)),
            1 => Res::Lines(L::read_first(&g, judge, &p0, *a, *b, me, p)),
            2 => Res::Lines(L::read_all(&g, judge, &p0, me, p)),
            3 => Res::Lines(L::read_two_moves(&g, judge, &p0, *a, me, p)),
            6 => Res::Afters(g.moves(&p0).iter().filter(|m| g.move_ends(&p0, m) == (*a, *b)).map(|m| L::read_after(&g, judge, &p0, m, *c, me, p)).collect()),
            7 => Res::Afters(g.moves(&p0).iter().map(|m| L::read_after(&g, judge, &p0, m, *c, me, p)).collect()),
            8 => Res::Lines(vec![L::read_best(&g, judge, &p0, me, p)]),
            4 => Res::Forced(L::forced_end(&g, &p0, mover, end_plies, end_fuse)),
            5 => Res::Answers(L::end_answers(&g, &p0, g.other(mover), end_plies, end_fuse).into_iter().map(|(e, n, v)| ((e.0, e.1), n, v)).collect()),
            _ => return Err(format!("mode {mode}")),
        };
        out.push((*qi, r));
    }
    Ok(())
}

fn read_dict<'py>(py: Python<'py>, r: &LineRead) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("prefix", r.prefix.clone())?;
    d.set_item("lead", r.lead.map(|x| (x.0, x.1)))?;
    d.set_item("first", (r.first.0, r.first.1))?;
    d.set_item("names", r.names.clone())?;
    d.set_item("events", r.events.clone())?;
    d.set_item("changes", r.changes.clone())?;
    d.set_item("value", r.value)?;
    d.set_item("end", r.end)?;
    d.set_item("depth", r.depth)?;
    d.set_item("nodes", r.nodes)?;
    d.set_item("cut", r.cut)?;
    d.set_item("settled", r.settled)?;
    d.set_item("unsettled", r.unsettled)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (variant, games, queries, worth, depth=6, fuse=40000, end_plies=7, end_fuse=1500000, threads=0, kind_change=true))]
#[allow(clippy::too_many_arguments)]
pub fn read_lines(py: Python<'_>, variant: &str, games: Vec<GameWire>, queries: Vec<Query>, worth: HashMap<String, i64>, depth: i32, fuse: u64, end_plies: u32, end_fuse: u64, threads: usize, kind_change: bool) -> PyResult<PyObject> {
    let _ = pad_probe();
    let judge = Worth { name: "record".into(), table: worth, ours_first: false };
    let p = LineParams { depth, fuse };
    let mut by_game: Vec<Vec<(usize, Query)>> = vec![vec![]; games.len()];
    for (i, q) in queries.iter().enumerate() {
        if q.0 >= games.len() { return Err(bad(format!("query {i}: game {} of {}", q.0, games.len()))); }
        by_game[q.0].push((i, *q));
    }
    let variant = variant.to_string();
    let threads = if threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { threads };
    // PLAN 78.77: `kind_change` False takes the event "a unit changed kind" out of the ledger (the ablation); True (the default) is the ledger as it always was
    let off = NoChange(judge.clone());
    let accs = py.allow_threads(|| if kind_change {
        run_acc(games.len(), threads, Vec::<(usize, Res)>::new, |i, acc| read_game(&variant, &games[i], &by_game[i], &judge, p, end_plies, end_fuse, acc))
    } else {
        run_acc(games.len(), threads, Vec::<(usize, Res)>::new, |i, acc| read_game(&variant, &games[i], &by_game[i], &off, p, end_plies, end_fuse, acc))
    }).map_err(bad)?;
    let mut all: Vec<(usize, Res)> = accs.into_iter().flatten().collect();
    all.sort_by_key(|x| x.0);
    let out = PyList::empty(py);
    for (_qi, r) in all {
        let d = PyDict::new(py);
        match r {
            Res::NoPass => { d.set_item("ok", false)?; }
            Res::Lines(v) => {
                d.set_item("ok", true)?;
                let l = PyList::empty(py);
                for x in &v { l.append(read_dict(py, x)?)?; }
                d.set_item("reads", l)?;
            }
            Res::Afters(v) => {
                d.set_item("ok", true)?;
                let l = PyList::empty(py);
                for x in &v {
                    let e = PyDict::new(py);
                    e.set_item("whole", read_dict(py, &x.whole)?)?;
                    let tk = PyList::empty(py);
                    for y in &x.takes { tk.append(read_dict(py, y)?)?; }
                    e.set_item("takes", tk)?;
                    l.append(e)?;
                }
                d.set_item("afters", l)?;
            }
            Res::Forced(f) => {
                d.set_item("ok", true)?;
                match f {
                    Forced::Proved { depth, first, line } => { d.set_item("forced", "proved")?; d.set_item("depth", depth)?; d.set_item("first", (first.0, first.1))?; d.set_item("line", line)?; }
                    Forced::Suspected { depth, first, line } => { d.set_item("forced", "suspected")?; d.set_item("depth", depth)?; d.set_item("first", (first.0, first.1))?; d.set_item("line", line)?; }
                    Forced::None { depth } => { d.set_item("forced", "none")?; d.set_item("depth", depth)?; }
                    Forced::Gap { depth } => { d.set_item("forced", "gap")?; d.set_item("depth", depth)?; }
                }
            }
            Res::Answers(v) => { d.set_item("ok", true)?; d.set_item("answers", v)?; }
        }
        out.append(d)?;
    }
    out.into_py_any(py)
}

// (a linker detail: the loader of this host wants the string pool of the library 8-aligned, which depends on the parity of the library's imported symbols; one imported datum flips it)
extern "C" { static environ: *const *const u8; }
static PAD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn pad_probe() { PAD.store(unsafe { std::ptr::read_volatile(&raw const environ) } as usize, std::sync::atomic::Ordering::Relaxed); }
