//! PLAN 78.55: the Python entry of the frames (cs_tactics::sheet::frames). Game-free: every name arrives as data; the game is read through `Placed`.
//! `frame_read` reads one frame of a position; `frame_join` makes the frames of several seeds and joins those in contact; `frame_compose` composes two frames not in contact and checks it against
//! the joint read; `frames_measure` is stage G over the quiet rows of games.
use crate::base_graph_py::{bad, run_acc, GameWire};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::fates::{self, Cell, End};
use cs_tactics::sheet::frames::{self, Frame, Pair, Reader, Seed, Status};
use cs_tactics::sheet::patterns::{self, Board, FxHashMap};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;
use std::time::Instant;

/// The composition test reads only pairs of frames of at most this many members together, with this fuse: the joint read of a union frame is the costly one (named costs).
const COMPOSE_MEMBERS: usize = 9;
const COMPOSE_FUSE: u64 = 5_000;

type St = <KernelGame as Game>::State;

pub fn parse_unit(st: &Static, text: &str) -> Result<Cell, String> {
    let t = text.strip_prefix("unit:").unwrap_or(text);
    let mut it = t.splitn(3, ':');
    let (o, k, p) = (it.next().ok_or("no owner")?, it.next().ok_or("no kind")?, it.next().ok_or("no place")?);
    Ok((st.names.role(o)?, st.names.kind(k)?, st.names.place(p)?))
}

pub fn parse_seed(st: &Static, units: &[String], places: &[String]) -> Result<Seed, String> {
    let mut u: Vec<Cell> = units.iter().map(|t| parse_unit(st, t)).collect::<Result<_, _>>()?;
    u.sort();
    let mut p: Vec<u16> = places.iter().map(|t| st.names.place(t.strip_prefix("place:").unwrap_or(t))).collect::<Result<_, _>>()?;
    p.sort();
    Ok(Seed { units: u, places: p })
}

fn game_at(variant: &str, fen: &str) -> Result<(KernelGame, St), String> {
    let g = KernelGame::from_fen(variant, fen)?;
    let s = g.initial();
    Ok((g, s))
}

fn ids_of(st: &Static, f: &Frame, mask: u16) -> Vec<String> { (0..f.members.len()).filter(|i| mask & (1 << i) != 0).map(|i| frames::unit_id(st, &f.members[i])).collect() }

fn pairs_py(st: &Static, f: &Frame, pairs: &[Pair]) -> Vec<(Vec<String>, Vec<String>)> { pairs.iter().map(|p| (ids_of(st, f, p.lx), ids_of(st, f, p.lo))).collect() }

fn stats_dict<'py>(py: Python<'py>, s: &frames::Stats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("nodes", s.nodes)?; d.set_item("unread", s.unread)?; d.set_item("bounded", s.bounded)?; d.set_item("stuck", s.stuck)?; d.set_item("left", s.left)?;
    d.set_item("pass_unrealisable", s.pass_unrealisable)?; d.set_item("forced_pass", s.forced_pass)?; d.set_item("pass_refused", s.pass_refused)?; d.set_item("others", s.others)?;
    Ok(d)
}

fn frame_dict<'py>(py: Python<'py>, st: &Static, f: &Frame) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", f.id.clone())?;
    d.set_item("members", f.members.iter().map(|c| frames::unit_id(st, c)).collect::<Vec<_>>())?;
    d.set_item("places", f.places.iter().map(|p| format!("place:{}", st.names.places[*p as usize])).collect::<Vec<_>>())?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
pub fn frame_read(py: Python<'_>, variant: &str, st: &Static, fen: &str, seed: &Seed, conditional: bool, full_tempo: bool, table_passes: Option<u8>, fuse: u64) -> PyResult<PyObject> {
    let (g, s) = game_at(variant, fen).map_err(bad)?;
    let frame = Frame::new(st, &g, &s, seed).map_err(|e| bad(e.word()))?;
    let mut rep = frames::claims_of(&g, st, &frame, &s, conditional, fuse);
    if full_tempo {
        let mut rd = Reader::new(&g, st, &frame);
        rd.fuse = fuse;
        rep.tempo = frames::tempo_table(&mut rd, &s, &[0, 1, 2, frames::ALL]);
    }
    let d = PyDict::new(py);
    d.set_item("frame", frame_dict(py, st, &frame)?)?;
    d.set_item("mover", g.mover(&s).map(|m| st.names.roles[m].clone()))?;
    let claims = rep.claims.iter().map(|c| {
        let m = &frame.members[c.unit as usize];
        let cd = PyDict::new(py);
        cd.set_item("reading", st.names.roles[c.reading].clone())?;
        cd.set_item("kind", frames::claim_kind_word(c.kind))?;
        cd.set_item("unit", frames::unit_id(st, m))?;
        cd.set_item("status", c.status.word())?;
        if let Some(k) = &c.condition {
            let kd = PyDict::new(py);
            kd.set_item("kind", st.names.kinds[k.kind as usize].clone())?;
            kd.set_item("from", st.names.places[k.from as usize].clone())?;
            kd.set_item("to", st.names.places[k.to as usize].clone())?;
            kd.set_item("arrivals", k.arrivals)?;
            cd.set_item("condition", kd)?;
        }
        Ok::<_, PyErr>(cd)
    }).collect::<Result<Vec<_>, _>>()?;
    d.set_item("claims", claims)?;
    let tempo = rep.tempo.iter().map(|t| {
        let td = PyDict::new(py);
        td.set_item("flipped", t.flipped)?;
        td.set_item("budget", (t.bud[0], t.bud[1]))?;
        td.set_item("reading", st.names.roles[t.reading].clone())?;
        td.set_item("pairs", pairs_py(st, &frame, &t.pairs))?;
        td.set_item("stats", stats_dict(py, &t.stats)?)?;
        Ok::<_, PyErr>(td)
    }).collect::<Result<Vec<_>, _>>()?;
    d.set_item("tempo", tempo)?;
    d.set_item("stats", stats_dict(py, &rep.stats)?)?;
    d.set_item("arrivals_tried", rep.arrivals_tried)?;
    d.set_item("leaves", rep.leaves.iter().map(|i| frames::unit_id(st, &frame.members[*i as usize])).collect::<Vec<_>>())?;
    let l3 = frames::layer3(st, &rep);
    d.set_item("l3_nodes", l3.nodes.iter().map(|(t, i)| (t.to_string(), i.clone())).collect::<Vec<_>>())?;
    d.set_item("l3_links", l3.links.iter().map(|(t, a, b)| (t.to_string(), a.clone(), b.clone())).collect::<Vec<_>>())?;
    d.set_item("claim_status", l3.claim_status.iter().map(|(c, s)| (c.clone(), s.to_string())).collect::<Vec<_>>())?;
    if let Some(passes) = table_passes {
        let us = g.mover(&s).unwrap_or(0);
        let (rows, stats) = frames::outcomes_table(&g, st, &frame, &s, us, passes, fuse);
        let outs = rows.iter().map(|r| {
            let od = PyDict::new(py);
            od.set_item("lost_by_mover", ids_of(st, &frame, r.pair.lx))?;
            od.set_item("lost_by_other", ids_of(st, &frame, r.pair.lo))?;
            od.set_item("fates", r.fates.iter().enumerate().map(|(i, f)| match f {
                frames::Fate::Alive(p) => format!("{} alive on {}", frames::unit_id(st, &frame.members[i]), st.names.places[*p as usize]),
                frames::Fate::Taken { by, ply } => format!("{} taken by {} at ply {}", frames::unit_id(st, &frame.members[i]), if *by == 255 { "a unit outside".to_string() } else { frames::unit_id(st, &frame.members[*by as usize]) }, ply),
            }).collect::<Vec<_>>())?;
            od.set_item("line", r.line.clone())?;
            od.set_item("end", r.end)?;
            Ok::<_, PyErr>(od)
        }).collect::<Result<Vec<_>, _>>()?;
        d.set_item("outcomes", outs)?;
        d.set_item("outcomes_stats", stats_dict(py, &stats)?)?;
    }
    d.into_py_any(py)
}

/// The frames of several seeds on a position, those in contact joined: [(frame, how many seeds it came from)] or the refusal.
pub fn frame_join(py: Python<'_>, variant: &str, st: &Static, fen: &str, seeds: &[Seed]) -> PyResult<PyObject> {
    let (g, s) = game_at(variant, fen).map_err(bad)?;
    let part = st.position(&g, &s);
    let mut fs = vec![];
    let mut errs = vec![];
    for sd in seeds { match Frame::from_part(st, &part, sd) { Ok(f) => fs.push(f), Err(e) => errs.push(e.word()) } }
    let joined = frames::join_in_contact(st, &part, fs);
    let out = PyDict::new(py);
    let mut good = vec![];
    for r in joined { match r { Ok(f) => good.push(frame_dict(py, st, &f)?), Err(e) => errs.push(e.word()) } }
    out.set_item("frames", good)?;
    out.set_item("refused", errs)?;
    out.into_py_any(py)
}

/// Two frames not in contact: the composed set and the joint read, for both sides.
pub fn frame_compose(py: Python<'_>, variant: &str, st: &Static, fen: &str, a: &Seed, b: &Seed, fuse: u64) -> PyResult<PyObject> {
    let (g, s) = game_at(variant, fen).map_err(bad)?;
    let part = st.position(&g, &s);
    let fa = Frame::from_part(st, &part, a).map_err(|e| bad(e.word()))?;
    let fb = Frame::from_part(st, &part, b).map_err(|e| bad(e.word()))?;
    let contact = frames::in_contact(&fa, &fb, &part);
    let joint = Frame::from_part(st, &part, &frames::union_seed(a, b)).map_err(|e| bad(e.word()))?;
    let d = PyDict::new(py);
    d.set_item("in_contact", contact)?;
    d.set_item("a", frame_dict(py, st, &fa)?)?;
    d.set_item("b", frame_dict(py, st, &fb)?)?;
    d.set_item("joint", frame_dict(py, st, &joint)?)?;
    let mut rows = vec![];
    for reading in 0..2usize {
        let rd = PyDict::new(py);
        rd.set_item("reading", st.names.roles[reading].clone())?;
        let composed = frames::compose(&g, st, &fa, &fb, &s, reading, fuse);
        let mut rj = Reader::new(&g, st, &joint);
        rj.fuse = fuse;
        let jf = rj.forceable(&s, reading, false, frames::sound_budget(reading));
        let map = frames::joint_ids(&fa, &fb, &joint);
        // the names of the composed space: a's members then b's
        let names: Vec<String> = fa.members.iter().chain(fb.members.iter()).map(|c| frames::unit_id(st, c)).collect();
        let text = |ps: &[Pair]| -> Vec<(Vec<String>, Vec<String>)> {
            let ids = |m: u16| -> Vec<String> { (0..names.len()).filter(|i| m & (1 << i) != 0).map(|i| names[i].clone()).collect() };
            ps.iter().map(|p| (ids(p.lx), ids(p.lo))).collect()
        };
        if let Some(c) = &composed { rd.set_item("composed", text(c))?; }
        if let (Some(c), Some(j), Some(m)) = (&composed, &jf, &map) {
            let jr: Vec<Pair> = j.pairs.iter().map(|p| frames::reindex(*p, m)).collect();
            rd.set_item("joint", text(&jr))?;
            rd.set_item("cmp", frames::compare_sets(c, &jr).word())?;
            rd.set_item("joint_stats", stats_dict(py, &j.stats)?)?;
        }
        rows.push(rd);
    }
    d.set_item("readings", rows)?;
    d.into_py_any(py)
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ stage G

/// What stage G counts. Counters by name: [n, taken, taken for nothing, taken after taking].
#[derive(Default)]
struct Agg { obs: HashMap<String, [u64; 4]>, count: HashMap<String, u64>, secs: HashMap<String, f64>, dump: Vec<String> }

impl Agg {
    fn bump(&mut self, key: String, h: &fates::History) {
        if h.end == End::NoFate { return; }
        let e = self.obs.entry(key).or_default();
        e[0] += 1;
        if let End::Taken(_) = h.end { e[1] += 1; if h.took.is_empty() { e[2] += 1 } else { e[3] += 1 } }
    }
    fn n(&mut self, key: &str, by: u64) { *self.count.entry(key.to_string()).or_default() += by; }
    fn t(&mut self, key: &str, by: f64) { *self.secs.entry(key.to_string()).or_default() += by; }
    fn merge(&mut self, o: Agg) {
        for (k, v) in o.obs { let e = self.obs.entry(k).or_default(); for i in 0..4 { e[i] += v[i]; } }
        for (k, v) in o.count { *self.count.entry(k).or_default() += v; }
        for (k, v) in o.secs { *self.secs.entry(k).or_default() += v; }
        for d in o.dump { if self.dump.len() < 400 { self.dump.push(d); } }
    }
}

/// The parameters of stage G, all named: the fuse of one reading (positions), the fuse of one row (positions over all its frames), how often a row with two frames apart is composed.
pub struct MeasureParams { pub horizons: Vec<usize>, pub fuse: u64, pub row_fuse: u64, pub compose_every: usize, pub max_size: usize, pub conditional: bool }

#[allow(clippy::too_many_arguments)]
fn read_game(variant: &str, st: &Static, goal: Option<u8>, game: &GameWire, wanted: &[usize], want: &FxHashMap<u64, u32>, p: &MeasureParams, gid: usize, agg: &mut Agg) -> Result<(), String> {
    let g = match &game.0 { Some(fen) => KernelGame::from_fen(variant, fen)?, None => KernelGame::new(variant, 518)? };
    let mut s = g.initial();
    let n = game.1.len();
    let (mut plies, mut states) = (vec![], vec![]);
    for ply in 0..=n {
        let mut cells = g.cells(&s);
        cells.sort();
        plies.push(cells);
        states.push(s.clone());
        if ply < n {
            let mv = g.find_move(&s, &game.1[ply]).ok_or_else(|| format!("ply {ply}: {} is not a legal move", game.1[ply]))?;
            s = g.apply(&s, &mv);
        }
    }
    let tr = fates::track(&plies);
    let levels: Vec<u8> = (0..patterns::LEVELS).collect();
    let mut rowno = 0usize;
    for &t in wanted {
        if t >= states.len() { continue; }
        let s = &states[t];
        let Some(mover) = g.mover(s) else { continue };
        agg.n("rows", 1);
        // every unit on the row: the base of what became of it (stage 0), by kind, side to move and horizon
        for (i, c) in plies[t].iter().enumerate() {
            for &h in &p.horizons {
                let hist = fates::fate_history(&tr, &plies, t, i, h);
                agg.bump(format!("ALL|{}|{}|{}", c.1, (c.0 == mover) as u8, h), &hist);
            }
        }
        let t0 = Instant::now();
        let part = st.position(&g, s);
        let b = Board::new(st, &part, goal);
        let seeds = frames::seeds_of_board(&b, &part, p.max_size, &levels, want);
        agg.t("seconds_seeds", t0.elapsed().as_secs_f64());
        agg.n("seeds", seeds.len() as u64);
        if seeds.is_empty() { continue; }
        agg.n("rows_with_seeds", 1);
        let t1 = Instant::now();
        let mut fs: Vec<Frame> = vec![];
        for (_, sd) in &seeds {
            match Frame::from_part(st, &part, sd) {
                Ok(f) => { if !fs.iter().any(|x| x.id == f.id) { fs.push(f) } }
                Err(e) => agg.n(&format!("refused|{}", e.word().split(':').next().unwrap_or("")), 1),
            }
        }
        let before = fs.len();
        let joined: Vec<Frame> = frames::join_in_contact(st, &part, fs).into_iter().filter_map(|r| match r { Ok(f) => Some(f), Err(e) => { agg.n(&format!("refused|{}", e.word().split(':').next().unwrap_or("")), 1); None } }).collect();
        agg.n("frames_before_join", before as u64);
        agg.n("frames", joined.len() as u64);
        agg.t("seconds_frames_build", t1.elapsed().as_secs_f64());
        if joined.len() >= 2 { agg.n("rows_with_two_frames", 1); }
        let mut row_nodes = 0u64;
        for f in &joined {
            if row_nodes >= p.row_fuse { agg.n("frames_skipped_by_row_fuse", 1); continue; }
            let t2 = Instant::now();
            let rep = frames::claims_of(&g, st, f, s, p.conditional, p.fuse);
            let dt = t2.elapsed().as_secs_f64();
            row_nodes += rep.stats.nodes;
            agg.t("seconds_read", dt);
            agg.n("frames_read", 1);
            agg.n("members", f.members.len() as u64);
            agg.n("nodes", rep.stats.nodes);
            agg.n("bounded", rep.stats.bounded);
            agg.n("stuck", rep.stats.stuck);
            agg.n("left", rep.stats.left);
            agg.n("pass_unrealisable", rep.stats.pass_unrealisable);
            agg.n("forced_pass", rep.stats.forced_pass);
            agg.n("pass_refused", rep.stats.pass_refused);
            agg.n("others", rep.stats.others);
            if rep.stats.unread > 0 { agg.n("frames_with_unread_lines", 1); agg.n("unread_lines", rep.stats.unread); }
            agg.n("arrivals_tried", rep.arrivals_tried as u64);
            // the reasoned class of each member, against what became of it
            let status_of = |id: usize, reading: usize, kinds: &[frames::ClaimKind]| -> Status {
                rep.claims.iter().filter(|c| c.reading == reading && c.unit as usize == id && kinds.contains(&c.kind)).map(|c| c.status).fold(Status::Not, |a, b| rank(a, b))
            };
            for (id, c) in f.members.iter().enumerate() {
                let owner = c.0.min(1);
                let free = status_of(id, 1 - owner, &[frames::ClaimKind::TakeFree]);
                let safe = status_of(id, owner, &[frames::ClaimKind::Safe]);
                let class = if free == Status::Forced && safe == Status::Forced { "BOTH" } else if free == Status::Forced { "LOST" } else if safe == Status::Forced { "SAFE" }
                    else if rep.stats.unread > 0 { "UNREAD" } else if matches!(free, Status::Threat | Status::Waiting) { "THREAT" } else if free == Status::Conditional { "CONDITIONAL" } else { "CONTESTED" };
                let Ok(i) = plies[t].binary_search(c) else { continue };
                for &h in &p.horizons {
                    let hist = fates::fate_history(&tr, &plies, t, i, h);
                    agg.bump(format!("{}|{}|{}|{}", class, c.1, (c.0 == mover) as u8, h), &hist);
                    agg.bump(format!("MEMBER|{}|{}|{}", c.1, (c.0 == mover) as u8, h), &hist);
                }
            }
            // the conditional claims: was the named change played, and what followed
            for cr in rep.claims.iter().filter(|c| c.status == Status::Conditional && c.kind == frames::ClaimKind::TakeFree) {
                let Some(cd) = &cr.condition else { continue };
                let m = &f.members[cr.unit as usize];
                let played = |ply: usize| -> bool {
                    if ply >= n { return false; }
                    let st_ = &states[ply];
                    let Some(mv) = g.find_move(st_, &game.1[ply]) else { return false };
                    let (_, to) = g.move_ends(st_, &mv);
                    to == cd.to && g.move_info(st_, &mv).0 == cd.kind
                };
                let when = if played(t) { "now" } else if played(t + 2) { "next" } else { "no" };
                let Ok(i) = plies[t].binary_search(m) else { continue };
                for &h in &p.horizons {
                    let hist = fates::fate_history(&tr, &plies, t, i, h);
                    agg.bump(format!("CONDCLAIM|{}|{}", when, h), &hist);
                }
            }
        }
        // two frames apart: the composition against the joint read, now and then
        if joined.len() >= 2 {
            rowno += 1;
            if rowno % p.compose_every.max(1) == 0 {
                let (fa, fb) = (&joined[0], &joined[1]);
                if fa.members.len() + fb.members.len() > COMPOSE_MEMBERS { agg.n("F|frames too large to compose", 1); continue; }
                let joint = match Frame::from_part(st, &part, &frames::union_seed(&fa.seed, &fb.seed)) { Ok(j) => j, Err(_) => { agg.n("F|joint refused", 1); continue; } };
                let Some(map) = frames::joint_ids(fa, fb, &joint) else { agg.n("F|joint has members of neither frame", 1); continue; };
                for reading in 0..2usize {
                    let Some(c) = frames::compose(&g, st, fa, fb, s, reading, COMPOSE_FUSE) else { agg.n("F|too many members", 1); continue };
                    let mut rj = Reader::new(&g, st, &joint);
                    rj.fuse = COMPOSE_FUSE;
                    let Some(j) = rj.forceable(s, reading, false, frames::sound_budget(reading)) else { continue };
                    if j.stats.unread > 0 { agg.n("F|joint unread", 1); continue; }
                    let jr: Vec<Pair> = j.pairs.iter().map(|q| frames::reindex(*q, &map)).collect();
                    let cmp = frames::compare_sets(&c, &jr);
                    agg.n(&format!("F|sets {}", cmp.word()), 1);
                    let na = fa.members.len();
                    let owner_of = |i: usize| if i < na { fa.members[i].0 } else { fb.members[i - na].0 }.min(1);
                    for u in 0..na + fb.members.len() {
                        let kinds: &[frames::ClaimKind] = if owner_of(u) == reading { &[frames::ClaimKind::Safe] } else { &[frames::ClaimKind::TakeFree, frames::ClaimKind::TakeAny] };
                        for &k in kinds {
                            let (x, y) = (frames::holds(&c, k, u as u8), frames::holds(&jr, k, u as u8));
                            agg.n(&format!("F|claim {} {}", frames::claim_kind_word(k), if x == y { "agree" } else if x { "composed only" } else { "joint only" }), 1);
                        }
                    }
                    if cmp != frames::Cmp::Equal {
                        let names: Vec<String> = fa.members.iter().chain(fb.members.iter()).map(|c| frames::unit_id(st, c)).collect();
                        let txt = |ps: &[Pair]| ps.iter().map(|q| format!("lose{:?}/win{:?}", (0..names.len()).filter(|i| q.lx & (1 << i) != 0).map(|i| names[i].as_str()).collect::<Vec<_>>(), (0..names.len()).filter(|i| q.lo & (1 << i) != 0).map(|i| names[i].as_str()).collect::<Vec<_>>())).collect::<Vec<_>>().join(" | ");
                        agg.dump.push(format!("game {gid} ply {t} reading {} {}: composed {{{}}} joint {{{}}} frames {} + {}", st.names.roles[reading], cmp.word(), txt(&c), txt(&jr), fa.id, fb.id));
                    }
                }
            }
        }
    }
    Ok(())
}

/// The strongest status of two (forced beats threat beats waiting beats conditional beats no).
fn rank(a: Status, b: Status) -> Status {
    let r = |s: Status| match s { Status::Forced => 4, Status::Threat => 3, Status::Waiting => 2, Status::Conditional => 1, Status::Not => 0 };
    if r(b) > r(a) { b } else { a }
}

#[allow(clippy::too_many_arguments)]
pub fn frames_measure(py: Python<'_>, variant: &str, st: &Static, goal: Option<u8>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, params: MeasureParams, threads: usize) -> PyResult<PyObject> {
    let want: FxHashMap<u64, u32> = keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
    let st2 = st.clone();
    let variant = variant.to_string();
    let accs = py.allow_threads(|| run_acc(games.len(), threads, Agg::default, |i, acc| read_game(&variant, &st2, goal, &games[i], &rows[i], &want, &params, i, acc))).map_err(bad)?;
    let mut tot = Agg::default();
    for a in accs { tot.merge(a); }
    let d = PyDict::new(py);
    d.set_item("obs", tot.obs.iter().map(|(k, v)| (k.clone(), v.to_vec())).collect::<HashMap<_, _>>())?;
    d.set_item("count", tot.count)?;
    d.set_item("secs", tot.secs)?;
    d.set_item("dump", tot.dump)?;
    d.into_py_any(py)
}
