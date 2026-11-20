//! PLAN 78.57: the Python entry of the frame read by questions (cs_tactics::sheet::questions). Game-free: every name arrives as data; the game is read through `Placed`.
//! `question_read` reads one frame; `question_trap` asks the trap question of a unit; `question_compose` composes two frames with the tempo charged and reads the joint frame; `question_reopen` links pruned
//! root moves to claims of other frames; `questions_measure` is stage F over the quiet rows of games.
use crate::base_graph_py::{bad, run_acc, GameWire};
use crate::frames_py::{parse_seed, parse_unit};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::fates::{self, End};
use cs_tactics::sheet::frames::{unit_id, ClaimKind, Cmp, Seed, Status};
use cs_tactics::sheet::patterns::{self, Board, FxHashMap};
use cs_tactics::sheet::questions::{self as q, QFrame, QReader, QStats};
use cs_tactics::sheet::threat_frames::{rank, threat_frames, unit_class, FrameReadParams};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;
use std::time::Instant;

type St = <KernelGame as Game>::State;

fn game_at(variant: &str, text: &str) -> Result<(KernelGame, St), String> {
    let g = KernelGame::from_fen(variant, text)?;
    let s = g.initial();
    Ok((g, s))
}

fn stats_dict<'py>(py: Python<'py>, s: &QStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("questions", s.questions)?; d.set_item("positions", s.positions)?; d.set_item("candidates", s.candidates)?; d.set_item("not_asked", s.not_asked)?;
    d.set_item("pruned_moved", s.pruned_moved)?; d.set_item("pruned_other", s.pruned_other)?; d.set_item("stops", s.stops)?;
    d.set_item("bounded", s.bounded)?; d.set_item("unread", s.unread)?; d.set_item("stuck", s.stuck)?; d.set_item("forced_pass", s.forced_pass)?; d.set_item("pass_unrealisable", s.pass_unrealisable)?;
    d.set_item("pass_refused", s.pass_refused)?; d.set_item("est_pruned", s.est_pruned)?; d.set_item("est_read", s.est_read)?; d.set_item("est_cont", s.est_cont)?; d.set_item("est_unplayed", s.est_unplayed)?; d.set_item("by_class", s.by_class.to_vec())?; d.set_item("ask_class", s.ask_class.to_vec())?; d.set_item("gen_class", s.gen_class.to_vec())?; d.set_item("by_depth", s.by_depth.to_vec())?; d.set_item("too_big", s.too_big)?; d.set_item("gone", s.gone)?; for (k, v) in [("end_settled", s.end_settled), ("end_settled_est", s.end_settled_est), ("end_circle", s.end_circle), ("band_held", s.band_held), ("cause_guard", s.cause_guard), ("cause_line", s.cause_line), ("est_stop", s.est_stop), ("quiet_skipped", s.quiet_skipped), ("settle_trap_q", s.settle_trap_q), ("settle_trap_pos", s.settle_trap_pos), ("settle_trapped", s.settle_trapped)] { d.set_item(k, v)?; } d.set_item("follows", s.follows)?; d.set_item("others", s.others)?;
    Ok(d)
}

fn frame_dict<'py>(py: Python<'py>, st: &Static, f: &QFrame) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", f.id.clone())?;
    d.set_item("seed", f.seed.units.iter().map(|c| unit_id(st, c)).collect::<Vec<_>>())?;
    d.set_item("members", f.members.iter().map(|c| unit_id(st, c)).collect::<Vec<_>>())?;
    d.set_item("places", f.places.iter().map(|p| format!("place:{}", st.names.places[*p as usize])).collect::<Vec<_>>())?;
    d.set_item("split", f.split)?;
    Ok(d)
}

fn place_name(st: &Static, p: u16) -> String { st.names.places[p as usize].clone() }

fn report_dict<'py>(py: Python<'py>, st: &Static, g: &KernelGame, s: &St, rep: &q::QReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frame", frame_dict(py, st, &rep.frame)?)?;
    d.set_item("mover", g.mover(s).map(|m| st.names.roles[m].clone()))?;
    let claims = rep.claims.iter().map(|c| {
        let m = &rep.frame.members[c.unit as usize];
        let cd = PyDict::new(py);
        cd.set_item("reading", st.names.roles[c.reading].clone())?;
        cd.set_item("kind", cs_tactics::sheet::frames::claim_kind_word(c.kind))?;
        cd.set_item("unit", unit_id(st, m))?;
        cd.set_item("status", c.status.word())?;
        cd.set_item("status_read", c.status_word())?;
        cd.set_item("marks", q::marks_word(c.flags))?;
        cd.set_item("basis", if c.status == Status::Not { "none" } else { q::basis_word(q::basis_rank(c.flags)) })?;
        if let Some(k) = &c.condition {
            let kd = PyDict::new(py);
            kd.set_item("kind", st.names.kinds[k.kind as usize].clone())?;
            kd.set_item("owner", st.names.roles[k.owner].clone())?;
            kd.set_item("from", place_name(st, k.from))?;
            kd.set_item("to", place_name(st, k.to))?;
            kd.set_item("arrivals", k.arrivals)?;
            cd.set_item("condition", kd)?;
        }
        Ok::<_, PyErr>(cd)
    }).collect::<Result<Vec<_>, _>>()?;
    d.set_item("claims", claims)?;
    d.set_item("holds", rep.holds)?;
    d.set_item("gains", rep.gains)?;
    d.set_item("stats", stats_dict(py, &rep.stats)?)?;
    d.set_item("arrivals_tried", rep.arrivals_tried)?;
    let prunes = rep.prunes.iter().map(|p| {
        let pd = PyDict::new(py);
        pd.set_item("mover", st.names.roles[p.mover].clone())?;
        pd.set_item("move", (place_name(st, p.ends.0), place_name(st, p.ends.1)))?;
        pd.set_item("lost", unit_id(st, &p.lost))?;
        pd.set_item("moved_unit_lost", p.moved_lost)?;
        pd.set_item("reply", p.reply.map(|r| (place_name(st, r.0), place_name(st, r.1))))?;
        pd.set_item("id", q::prune_id(st, &rep.frame, p))?;
        Ok::<_, PyErr>(pd)
    }).collect::<Result<Vec<_>, _>>()?;
    d.set_item("prunes", prunes)?;
    // 78.60 C: the named causes for an exchange to go on below: (what, the unit named, the member concerned, the taking's ends)
    let causes = rep.causes.iter().map(|c| (if c.kind == 0 { "only_guard" } else { "uncovered_line" }, unit_id(st, &c.named), unit_id(st, &c.target), (place_name(st, c.ends.0), place_name(st, c.ends.1)))).collect::<Vec<_>>();
    d.set_item("causes", causes)?;
    let l3 = q::layer3_q(st, rep);
    d.set_item("l3_nodes", l3.nodes.iter().map(|(t, i)| (t.to_string(), i.clone())).collect::<Vec<_>>())?;
    d.set_item("l3_links", l3.links.iter().map(|(t, a, b)| (t.to_string(), a.clone(), b.clone())).collect::<Vec<_>>())?;
    d.set_item("claim_status", l3.claim_status.iter().map(|(c, s)| (c.clone(), s.to_string())).collect::<Vec<_>>())?;
    d.set_item("claim_marks", l3.claim_marks.iter().map(|(c, m)| (c.clone(), m.iter().map(|x| x.to_string()).collect::<Vec<_>>())).collect::<Vec<_>>())?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
pub fn question_read(py: Python<'_>, variant: &str, st: &Static, text: &str, seed: &Seed, conditional: bool, fuse: u64, kv: Option<Vec<i64>>, rules: q::Rules, bands: Option<Vec<(i64, i64)>>) -> PyResult<PyObject> {
    let (g, s) = game_at(variant, text).map_err(bad)?;
    let frame = QFrame::new_cap(st, &g, &s, seed, rules.root_cap).map_err(|e| bad(e.word()))?;
    let rep = q::claims_q_rules(&g, st, &frame, &s, conditional, fuse, kv.map(std::rc::Rc::new), rules, bands.map(std::rc::Rc::new));
    report_dict(py, st, &g, &s, &rep)?.into_py_any(py)
}

pub fn question_trap(py: Python<'_>, variant: &str, st: &Static, text: &str, unit: &str, fuse: u64) -> PyResult<PyObject> {
    let (g, s) = game_at(variant, text).map_err(bad)?;
    let cell = parse_unit(st, unit).map_err(bad)?;
    let a = q::trap(&g, st, &s, cell, fuse);
    let d = PyDict::new(py);
    d.set_item("word", a.word.word())?;
    d.set_item("free", a.free.iter().map(|p| place_name(st, *p)).collect::<Vec<_>>())?;
    d.set_item("bad", a.bad.iter().map(|p| place_name(st, *p)).collect::<Vec<_>>())?;
    d.set_item("stats", stats_dict(py, &a.stats)?)?;
    d.into_py_any(py)
}

fn pairs_text(st: &Static, cells: &[fates::Cell], ps: &[q::QP]) -> Vec<(Vec<String>, Vec<String>, u8, Vec<String>)> {
    let ids = |m: u64| -> Vec<String> { (0..cells.len()).filter(|i| m & (1u64 << i) != 0).map(|i| unit_id(st, &cells[i])).collect() };
    ps.iter().map(|p| (ids(p.p.lx), ids(p.p.lo), p.c, q::marks_word(p.fl).iter().map(|x| x.to_string()).collect())).collect()
}

/// Two frames: the tempo-charged composition and the joint read of the union frame, for the side to move.
pub fn question_compose(py: Python<'_>, variant: &str, st: &Static, text: &str, a: &Seed, b: &Seed, fuse: u64) -> PyResult<PyObject> {
    let (g, s) = game_at(variant, text).map_err(bad)?;
    let part = st.position(&g, &s);
    let fa = QFrame::new(st, &g, &s, a).map_err(|e| bad(e.word()))?;
    let fb = QFrame::new(st, &g, &s, b).map_err(|e| bad(e.word()))?;
    let mut seed = a.clone();
    for c in &b.units { if !seed.units.contains(c) { seed.units.push(*c); } }
    seed.units.sort();
    for p in &b.places { if !seed.places.contains(p) { seed.places.push(*p); } }
    let joint = QFrame::new(st, &g, &s, &seed).map_err(|e| bad(e.word()))?;
    let mut cells = g.cells(&s);
    cells.sort();
    let d = PyDict::new(py);
    d.set_item("in_contact", q::q_in_contact(&fa, &fb, &part))?;
    d.set_item("a", frame_dict(py, st, &fa)?)?;
    d.set_item("b", frame_dict(py, st, &fb)?)?;
    d.set_item("joint", frame_dict(py, st, &joint)?)?;
    let composed = q::compose_q(&g, st, &fa, &fb, &s, fuse);
    let mut rj = QReader::new(&g, st, &joint, &s);
    rj.fuse = fuse;
    let mover = g.mover(&s).unwrap_or(0);
    let jf = rj.forceable(&s, mover, false, 0, q::Goal::All);
    d.set_item("reading", st.names.roles[mover].clone())?;
    if let Some(c) = &composed { d.set_item("composed", pairs_text(st, &cells, c))?; }
    if let Some(j) = &jf { d.set_item("joint_pairs", pairs_text(st, &cells, &j.pairs))?; d.set_item("joint_stats", stats_dict(py, &j.stats)?)?; }
    if let (Some(c), Some(j)) = (&composed, &jf) { d.set_item("cmp", q::compare_q(c, &j.pairs).word())?; }
    d.into_py_any(py)
}

/// The frames of several seeds on a position (joined, split, none sharing a unit), their claims, and the pruned root moves reopened by claims of other frames.
pub fn question_reopen(py: Python<'_>, variant: &str, st: &Static, text: &str, seeds: &[Seed], fuse: u64) -> PyResult<PyObject> {
    let (g, s) = game_at(variant, text).map_err(bad)?;
    let built = q::build_frames(st, &g, &s, seeds);
    let reps: Vec<q::QReport> = built.frames.iter().map(|f| q::claims_q(&g, st, f, &s, true, fuse)).collect();
    let d = PyDict::new(py);
    d.set_item("reports", reps.iter().map(|r| report_dict(py, st, &g, &s, r)).collect::<Result<Vec<_>, _>>()?)?;
    let ro = q::reopen(&g, st, &reps, &s, fuse);
    d.set_item("reopen", ro.iter().map(|r| (r.prune.clone(), r.claim.clone(), r.frame_of_claim.clone(), r.reopened, r.why.to_string())).collect::<Vec<_>>())?;
    // layer 3: a reopened prune is joined to the claim it is reopened by (the claim node exists only as a claim made conditional on that move); a prune that stays pruned has no link
    let mut nodes: Vec<(String, String)> = vec![];
    let mut links: Vec<(String, String, String)> = vec![];
    for r in ro.iter().filter(|r| r.reopened) {
        nodes.push(("prune".to_string(), r.prune.clone()));
        nodes.push(("claim".to_string(), r.claim.clone()));
        links.push(("reopened_by".to_string(), r.prune.clone(), r.claim.clone()));
        links.push(("in_frame".to_string(), r.claim.clone(), r.frame_of_claim.clone()));
    }
    d.set_item("reopen_nodes", nodes)?;
    d.set_item("reopen_links", links)?;
    d.set_item("split_joins", built.split_joins)?;
    d.set_item("dropped_shared", built.dropped_shared)?;
    d.set_item("refused", built.refused)?;
    d.into_py_any(py)
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ stage F

/// Counters by name: [n, taken, taken for nothing, taken after taking].
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

pub struct MeasureParams { pub horizons: Vec<usize>, pub fuse: u64, pub row_fuse: u64, pub compose_every: usize, pub max_size: usize, pub conditional: bool, pub trap_fuse: u64, pub kv: Option<Vec<i64>>, pub rules: q::Rules, pub bands: Option<Vec<(i64, i64)>> }

fn hist_bucket(n: u64) -> &'static str {
    match n { 0..=1 => "q<=1", 2..=3 => "q<=3", 4..=10 => "q<=10", 11..=30 => "q<=30", 31..=100 => "q<=100", 101..=300 => "q<=300", 301..=1000 => "q<=1000", _ => "q>1000" }
}

#[allow(clippy::too_many_arguments)]
fn read_game(variant: &str, st: &Static, goal: Option<u8>, game: &GameWire, wanted: &[usize], want: &FxHashMap<u64, u32>, p: &MeasureParams, gid: usize, agg: &mut Agg) -> Result<(), String> {
    let g = match &game.0 { Some(text) => KernelGame::from_fen(variant, text)?, None => KernelGame::new(variant, 518)? };
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
        for (i, c) in plies[t].iter().enumerate() {
            for &h in &p.horizons {
                let hist = fates::fate_history(&tr, &plies, t, i, h);
                agg.bump(format!("ALL|{}|{}|{}", c.1, (c.0 == mover) as u8, h), &hist);
            }
        }
        let t0 = Instant::now();
        let part = st.position(&g, s);
        let b = Board::new(st, &part, goal);
        let seeds: Vec<Seed> = cs_tactics::sheet::frames::seeds_of_board(&b, &part, p.max_size, &levels, want).into_iter().map(|x| x.1).collect();
        agg.t("seconds_seeds", t0.elapsed().as_secs_f64());
        agg.n("seeds", seeds.len() as u64);
        if seeds.is_empty() { continue; }
        agg.n("rows_with_seeds", 1);
        let t1 = Instant::now();
        let built = q::build_frames_cap(st, &g, s, &seeds, p.rules.root_cap);
        for r in &built.refused { agg.n(&format!("refused|{}", r.split(':').next().unwrap_or("")), 1); }
        agg.n("frames", built.frames.len() as u64);
        agg.n("split_joins", built.split_joins as u64);
        agg.n("dropped_shared", built.dropped_shared as u64);
        agg.t("seconds_frames_build", t1.elapsed().as_secs_f64());
        let joined = &built.frames;
        if joined.len() >= 2 { agg.n("rows_with_two_frames", 1); }
        let mut row_q = 0u64;
        let mut reports: Vec<q::QReport> = vec![];
        for f in joined {
            if f.split { agg.n("frames_split_at_root", 1); }
            if row_q >= p.row_fuse { agg.n("frames_skipped_by_row_fuse", 1); continue; }
            let t2 = Instant::now();
            let rep = q::claims_q_rules(&g, st, f, s, p.conditional, p.fuse, p.kv.clone().map(std::rc::Rc::new), p.rules, p.bands.clone().map(std::rc::Rc::new));
            let dt = t2.elapsed().as_secs_f64();
            row_q += rep.stats.questions;
            agg.t("seconds_read", dt);
            agg.n("frames_read", 1);
            agg.n("members", f.members.len() as u64);
            let sx = &rep.stats;
            for (k, v) in [("questions", sx.questions), ("positions", sx.positions), ("candidates", sx.candidates), ("not_asked", sx.not_asked), ("pruned_moved", sx.pruned_moved), ("pruned_other", sx.pruned_other), ("stops", sx.stops), ("bounded", sx.bounded), ("stuck", sx.stuck), ("forced_pass", sx.forced_pass), ("pass_unrealisable", sx.pass_unrealisable), ("pass_refused", sx.pass_refused), ("too_big", sx.too_big),
                           ("gone", sx.gone), ("follows", sx.follows), ("others", sx.others), ("unread_lines", sx.unread), ("est_pruned", sx.est_pruned), ("est_read", sx.est_read), ("est_cont", sx.est_cont), ("est_unplayed", sx.est_unplayed),
                           ("end_settled", sx.end_settled), ("end_settled_est", sx.end_settled_est), ("end_circle", sx.end_circle), ("band_held", sx.band_held), ("cause_guard", sx.cause_guard), ("cause_line", sx.cause_line), ("est_stop", sx.est_stop), ("quiet_skipped", sx.quiet_skipped),
                           ("settle_trap_q", sx.settle_trap_q), ("settle_trap_pos", sx.settle_trap_pos), ("settle_trapped", sx.settle_trapped)] { agg.n(k, v); }
            if sx.unread > 0 { agg.n("frames_with_unread_lines", 1); }
            agg.n(&format!("hist|{}", hist_bucket(sx.questions)), 1);
            let bucket = if sx.questions <= 30 { "small" } else if sx.questions > 1000 { "big" } else { "mid" };
            for b in [bucket, "all"] {
                agg.n(&format!("split|{b}|frames"), 1);
                agg.n(&format!("split|{b}|questions"), sx.questions);
                agg.n(&format!("split|{b}|members"), f.members.len() as u64);
                agg.n(&format!("split|{b}|positions"), sx.positions);
                for i in 0..6 { agg.n(&format!("split|{b}|qclass|{i}"), sx.by_class[i]); }
                for i in 0..5 { agg.n(&format!("split|{b}|gen|{i}"), sx.gen_class[i]); agg.n(&format!("split|{b}|ask|{i}"), sx.ask_class[i]); }
                for d in 0..16 { agg.n(&format!("split|{b}|depth|{d}"), sx.by_depth[d]); }
            }
            agg.n(&format!("size|{}|frames", f.members.len()), 1);
            agg.n(&format!("size|{}|questions", f.members.len()), sx.questions);
            agg.n("arrivals_tried", rep.arrivals_tried as u64);
            agg.n("root_prunes", rep.prunes.len() as u64);
            let status_of = |id: usize, reading: usize, kinds: &[ClaimKind]| -> (Status, q::Fl) {
                let mine: Vec<&q::QClaim> = rep.claims.iter().filter(|c| c.reading == reading && c.unit as usize == id && kinds.contains(&c.kind)).collect();
                let best = mine.iter().map(|c| c.status).fold(Status::Not, rank);
                (best, mine.iter().filter(|c| c.status == best).fold(0xFFFFu16, |a, c| a & c.flags))
            };
            for (id, c) in f.members.iter().enumerate() {
                let owner = c.0.min(1);
                let (free, ffl) = status_of(id, 1 - owner, &[ClaimKind::TakeFree]);
                let (safe, sfl) = status_of(id, owner, &[ClaimKind::Safe]);
                // 78.58 D: a claim that holds but rests on a bound, a prune or an estimate is READ, never forced
                let (rl, rs) = (ffl & q::READ_MARKS != 0, sfl & q::READ_MARKS != 0);
                let class = if free == Status::Forced && safe == Status::Forced { if rl || rs { "READ_BOTH" } else { "BOTH" } } else if free == Status::Forced { if rl { "READ_LOST" } else { "LOST" } } else if safe == Status::Forced { if rs { "READ_SAFE" } else { "SAFE" } }
                    else if rep.stats.unread > 0 { "UNREAD" } else if matches!(free, Status::Threat | Status::Waiting) { "THREAT" } else if free == Status::Conditional { "CONDITIONAL" } else { "CONTESTED" };
                if class == "LOST" || class == "THREAT" || class == "READ_LOST" {
                    agg.n(&format!("marks|{class}|n"), 1);
                    for (bit, nm) in [(q::F_FOLLOW, "followed"), (q::F_BOUND, "bound"), (q::F_PRUNE_USED, "prune_used"), (q::F_EST_PRUNE, "est_prune"), (q::F_EST_READ, "est_read")] { if ffl & bit != 0 { agg.n(&format!("marks|{class}|{nm}"), 1); } }
                }
                // 78.60 E: the class by what the claim rests on: forced within the frame, by a prune (inclusion only), by estimate, or bound (a line was cut); a unit both lost and safe takes the lesser basis
                let basis = |status: Status, fl: q::Fl| if status == Status::Forced { Some(q::basis_rank(fl)) } else { None };
                let by_basis = match class {
                    "LOST" | "READ_LOST" => basis(free, ffl).map(|r| format!("{}_LOST", q::basis_word(r).to_uppercase())),
                    "SAFE" | "READ_SAFE" => basis(safe, sfl).map(|r| format!("{}_SAFE", q::basis_word(r).to_uppercase())),
                    "BOTH" | "READ_BOTH" => basis(free, ffl).zip(basis(safe, sfl)).map(|(a, b)| format!("{}_BOTH", q::basis_word(a.max(b)).to_uppercase())),
                    _ => None,
                };
                let Ok(i) = plies[t].binary_search(c) else { continue };
                if let Some(bc) = by_basis { for &h in &p.horizons { let hist = fates::fate_history(&tr, &plies, t, i, h); agg.bump(format!("{}|{}|{}|{}", bc, c.1, (c.0 == mover) as u8, h), &hist); } }
                for &h in &p.horizons {
                    let hist = fates::fate_history(&tr, &plies, t, i, h);
                    agg.bump(format!("{}|{}|{}|{}", class, c.1, (c.0 == mover) as u8, h), &hist);
                    agg.bump(format!("MEMBER|{}|{}|{}", c.1, (c.0 == mover) as u8, h), &hist);
                }
                // the trap question, once per member
                if p.trap_fuse > 0 {
                    let tt = Instant::now();
                    let ta = q::trap(&g, st, s, *c, p.trap_fuse);
                    agg.t("seconds_trap", tt.elapsed().as_secs_f64());
                    agg.n("trap_asked", 1);
                    agg.n(&format!("trap|{}", ta.word.word()), 1);
                    agg.n("trap_questions", ta.stats.questions);
                    agg.n("trap_positions", ta.stats.positions);
                    for &h in &p.horizons {
                        let hist = fates::fate_history(&tr, &plies, t, i, h);
                        agg.bump(format!("TRAP:{}|{}|{}|{}", ta.word.word(), c.1, (c.0 == mover) as u8, h), &hist);
                    }
                }
            }
            for cr in rep.claims.iter().filter(|c| c.status == Status::Conditional && c.kind == ClaimKind::TakeFree) {
                let Some(cd) = &cr.condition else { continue };
                let m = &f.members[cr.unit as usize];
                let played = |ply: usize| -> bool {
                    if ply >= n { return false; }
                    let st_ = &states[ply];
                    let Some(mv) = g.find_move(st_, &game.1[ply]) else { return false };
                    let (from, to) = g.move_ends(st_, &mv);
                    cd.moves.contains(&(from, to))
                };
                let who_moves = |ply: usize| g.mover(&states[ply]) == Some(cd.owner);
                let when = if t < n && who_moves(t) && played(t) { "now" } else if t + 1 < n && who_moves(t + 1) && played(t + 1) { "next" } else if t + 2 < n && who_moves(t + 2) && played(t + 2) { "next" } else { "no" };
                let Ok(i) = plies[t].binary_search(m) else { continue };
                for &h in &p.horizons {
                    let hist = fates::fate_history(&tr, &plies, t, i, h);
                    agg.bump(format!("CONDCLAIM|{}|{}", when, h), &hist);
                }
            }
            reports.push(rep);
        }
        // the pruned root moves reopened by claims of other frames
        if reports.len() >= 2 {
            let ro = q::reopen(&g, st, &reports, s, p.fuse);
            for r in &ro {
                agg.n("reopen|links", 1);
                agg.n(&format!("reopen|{}", if r.reopened { "reopened" } else { "not reopened" }), 1);
                agg.n(&format!("reopen|why|{}", r.why), 1);
                if agg.dump.len() < 400 { agg.dump.push(format!("REOPEN game {gid} ply {t}: {} <- {} ({}; {})", r.prune, r.claim, if r.reopened { "reopened" } else { "stays pruned" }, r.why)); }
            }
        }
        // two frames apart: the composition with the tempo charged against the joint read, now and then
        if joined.len() >= 2 {
            rowno += 1;
            if rowno % p.compose_every.max(1) == 0 {
                let (fa, fb) = (&joined[0], &joined[1]);
                if q::q_in_contact(fa, fb, &part) { agg.n("F|in contact after building", 1); continue; }
                let mut seed = fa.seed.clone();
                for c in &fb.seed.units { if !seed.units.contains(c) { seed.units.push(*c); } }
                seed.units.sort();
                for pl in &fb.seed.places { if !seed.places.contains(pl) { seed.places.push(*pl); } }
                let joint = match QFrame::new(st, &g, s, &seed) { Ok(j) => j, Err(_) => { agg.n("F|joint refused", 1); continue; } };
                let Some(c) = q::compose_q(&g, st, fa, fb, s, p.fuse) else { continue };
                let mut rj = QReader::new(&g, st, &joint, s);
                rj.fuse = p.fuse;
                let Some(j) = rj.forceable(s, mover, false, 0, q::Goal::All) else { continue };
                if j.stats.unread > 0 { agg.n("F|joint unread", 1); continue; }
                if c.iter().any(|x| x.fl & q::F_UNREAD != 0) { agg.n("F|composed unread", 1); continue; }
                let cmp = q::compare_q(&c, &j.pairs);
                agg.n(&format!("F|sets {}", cmp.word()), 1);
                let ra = QReader::new(&g, st, fa, s);
                let rb = QReader::new(&g, st, fb, s);
                for (fr, rd) in [(fa, &ra), (fb, &rb)] {
                    for (i, cell) in fr.members.iter().enumerate() {
                        let id = rd.member_id(i);
                        let kinds: &[ClaimKind] = if cell.0 == mover { &[ClaimKind::Safe] } else { &[ClaimKind::TakeFree, ClaimKind::TakeAny] };
                        for &k in kinds {
                            let (x, y) = (q::holds_q(&c, k, id), q::holds_q(&j.pairs, k, id));
                            agg.n(&format!("F|claim {} {}", cs_tactics::sheet::frames::claim_kind_word(k), if x == y { "agree" } else if x { "composed only" } else { "joint only" }), 1);
                        }
                    }
                }
                if cmp != Cmp::Equal {
                    let mut cells = g.cells(s);
                    cells.sort();
                    let txt = |ps: &[q::QP]| ps.iter().filter(|x| x.fl & (q::F_STUCK | q::F_PRUNED | q::F_UNREAD) == 0).map(|x| format!("lose{:?}/win{:?}/cost{}", (0..cells.len()).filter(|i| x.p.lx & (1u64 << i) != 0).map(|i| unit_id(st, &cells[i])).collect::<Vec<_>>(), (0..cells.len()).filter(|i| x.p.lo & (1u64 << i) != 0).map(|i| unit_id(st, &cells[i])).collect::<Vec<_>>(), x.c)).collect::<Vec<_>>().join(" | ");
                    agg.dump.push(format!("COMPOSE game {gid} ply {t} reading {} {}: composed {{{}}} joint {{{}}} frames {} + {}", st.names.roles[mover], cmp.word(), txt(&c), txt(&j.pairs), fa.id, fb.id));
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn questions_measure(py: Python<'_>, variant: &str, st: &Static, goal: Option<u8>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, params: MeasureParams, threads: usize) -> PyResult<PyObject> {
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

pub fn seeds_from(st: &Static, seeds: &[(Vec<String>, Vec<String>)]) -> Result<Vec<Seed>, String> { seeds.iter().map(|(u, p)| parse_seed(st, u, p)).collect() }

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ plan 78.62 stage A

/// One member unit of a frame read at a position: [mover's own (1) or the other's (0), kind index, class, basis, trap word, observed fate within the horizon]. Class: 0 lost for its owner (the other side
/// forces its taking), 1 safe, 2 both, 3 unread, 4 threat, 5 conditional, 6 contested. Basis (of a lost, safe or both class): 0 forced, 1 pruned, 2 estimate, 3 bound; 255 none. Trap: the index of the trap word
/// (free, trapped, immobile, not_quiet, ended, unread), 255 when not asked. Fate: 0 not taken, 1 taken for nothing, 2 taken after taking, 255 not known.
type StateRec = [u8; 6];

struct StateRow { game: usize, ply: usize, mover: u8, board: Vec<(u8, u8, u16)>, units: Vec<StateRec>, questions: u64, unread: u64, frames: u32 }

#[allow(clippy::too_many_arguments)]
fn state_game(variant: &str, st: &Static, goal: Option<u8>, game: &GameWire, wanted: &[usize], want: &FxHashMap<u64, u32>, p: &MeasureParams, horizon: usize, gid: usize, out: &mut Vec<StateRow>) -> Result<(), String> {
    let g = match &game.0 { Some(text) => KernelGame::from_fen(variant, text)?, None => KernelGame::new(variant, 518)? };
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
    for &t in wanted {
        if t >= states.len() { continue; }
        let s = &states[t];
        let Some(mover) = g.mover(s) else { continue };
        let part = st.position(&g, s);
        let b = Board::new(st, &part, goal);
        let seeds: Vec<Seed> = cs_tactics::sheet::frames::seeds_of_board(&b, &part, p.max_size, &levels, want).into_iter().map(|x| x.1).collect();
        let mut row = StateRow { game: gid, ply: t, mover: mover as u8, board: plies[t].iter().map(|c| (c.0 as u8, c.1 as u8, c.2 as u16)).collect(), units: vec![], questions: 0, unread: 0, frames: 0 };
        if !seeds.is_empty() {
            let built = q::build_frames_cap(st, &g, s, &seeds, p.rules.root_cap);
            let mut row_q = 0u64;
            for f in &built.frames {
                if row_q >= p.row_fuse { continue; }
                let rep = q::claims_q_rules(&g, st, f, s, p.conditional, p.fuse, p.kv.clone().map(std::rc::Rc::new), p.rules, p.bands.clone().map(std::rc::Rc::new));
                row_q += rep.stats.questions;
                row.questions += rep.stats.questions;
                row.unread += rep.stats.unread;
                row.frames += 1;
                for (id, c) in f.members.iter().enumerate() {
                    let owner = c.0.min(1);
                    let (class, basis) = unit_class(&rep, id, owner);
                    let trap: u8 = if p.trap_fuse > 0 {
                        match q::trap(&g, st, s, *c, p.trap_fuse).word { q::TrapWord::Free => 0, q::TrapWord::Trapped => 1, q::TrapWord::Immobile => 2, q::TrapWord::NotQuiet => 3, q::TrapWord::Ended => 4, q::TrapWord::Unread => 5 }
                    } else { 255 };
                    let fate: u8 = match plies[t].binary_search(c) {
                        Ok(i) => { let h = fates::fate_history(&tr, &plies, t, i, horizon); match h.end { End::NoFate => 255, End::Taken(_) => if h.took.is_empty() { 1 } else { 2 }, _ => 0 } }
                        Err(_) => 255,
                    };
                    row.units.push([(c.0 == mover) as u8, c.1 as u8, class, basis, trap, fate]);
                }
            }
        }
        out.push(row);
    }
    Ok(())
}

/// questions_state: the member units of every frame read at the wanted plies of each game, one record per unit (see `StateRec`; the dict also names the kinds, roles and places by index; each row also lists every unit on the board as (role, kind, place)), per row also the questions asked, the unread lines and the frames read.
#[allow(clippy::too_many_arguments)]
pub fn questions_state(py: Python<'_>, variant: &str, st: &Static, goal: Option<u8>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, params: MeasureParams, horizon: usize, threads: usize) -> PyResult<PyObject> {
    let want: FxHashMap<u64, u32> = keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
    let st2 = st.clone();
    let variant = variant.to_string();
    let accs = py.allow_threads(|| run_acc(games.len(), threads, Vec::<StateRow>::new, |i, acc| state_game(&variant, &st2, goal, &games[i], &rows[i], &want, &params, horizon, i, acc))).map_err(bad)?;
    let mut all: Vec<StateRow> = accs.into_iter().flatten().collect();
    all.sort_by_key(|r| (r.game, r.ply));
    let list = pyo3::types::PyList::empty(py);
    for r in all {
        let d = PyDict::new(py);
        d.set_item("game", r.game)?; d.set_item("ply", r.ply)?; d.set_item("mover", r.mover)?; d.set_item("board", r.board.iter().map(|b| vec![b.0 as u16, b.1 as u16, b.2]).collect::<Vec<_>>())?;
        d.set_item("questions", r.questions)?; d.set_item("unread", r.unread)?; d.set_item("frames", r.frames)?;
        d.set_item("units", r.units.iter().map(|u| u.iter().map(|x| *x as u16).collect::<Vec<u16>>()).collect::<Vec<_>>())?;
        list.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("kinds", st.names.kinds.clone())?; out.set_item("roles", st.names.roles.clone())?; out.set_item("places", st.names.places.clone())?;
    out.set_item("rows", list)?;
    out.into_py_any(py)
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ plan 78.70 stage A

/// One claim of the threat list: a member of a frame that the other side bears on (or that the read holds lost, both, a threat or conditional): [member index, class, basis] (the codes of `StateRec`), the units that
/// bear on it as (kind, place), the answers found as (from, to, how, the unit moves), the candidates tried and the ones the fuse left unread.
struct ThreatClaim { member: u16, class: u8, basis: u8, take_basis: u8, bearers: Vec<(u16, u16)>, answers: Vec<(u16, u16, u8, u8)>, tried: u32, unread: u32 }
/// One frame of a row: its identity, its members as (the mover's own, kind, place), its places (the read set with the members), the unread lines of its claims, and its claims.
struct ThreatFrame { id: String, members: Vec<(u8, u16, u16)>, places: Vec<u16>, unread: u64, claims: Vec<ThreatClaim> }
struct ThreatRow { game: usize, ply: usize, mover: u8, board: Vec<(u8, u8, u16)>, played: Option<(u16, u16)>, questions: u64, frames: Vec<ThreatFrame> }

#[allow(clippy::too_many_arguments)]
fn threat_game(variant: &str, st: &Static, goal: Option<u8>, game: &GameWire, wanted: &[usize], want: &FxHashMap<u64, u32>, p: &MeasureParams, answer_fuse: u64, gid: usize, out: &mut Vec<ThreatRow>) -> Result<(), String> {
    let g = match &game.0 { Some(text) => KernelGame::from_fen(variant, text)?, None => KernelGame::new(variant, 518)? };
    let mut s = g.initial();
    let n = game.1.len();
    let mut states = vec![];
    for ply in 0..=n {
        states.push(s.clone());
        if ply < n {
            let mv = g.find_move(&s, &game.1[ply]).ok_or_else(|| format!("ply {ply}: {} is not a legal move", game.1[ply]))?;
            s = g.apply(&s, &mv);
        }
    }
    // PLAN 79.08: the frame read of one position is cs_tactics::sheet::threat_frames (one piece, called by the wake too); this only turns it into the wire form
    let fp = FrameReadParams { max_size: p.max_size, fuse: p.fuse, row_fuse: p.row_fuse, answer_fuse, conditional: p.conditional, kv: p.kv.clone(), bands: p.bands.clone(), rules: p.rules };
    for &t in wanted {
        if t >= states.len() { continue; }
        let s = &states[t];
        let Some(mover) = g.mover(s) else { continue };
        let mut cells = g.cells(s);
        cells.sort();
        let read = threat_frames(st, &g, s, goal, want, &fp);
        let mut row = ThreatRow { game: gid, ply: t, mover: mover as u8, board: cells.iter().map(|c| (c.0 as u8, c.1 as u8, c.2 as u16)).collect(), played: game.1.get(t).and_then(|text| g.find_move(s, text)).map(|m| { let e = g.move_ends(s, &m); (e.0 as u16, e.1 as u16) }), questions: read.questions, frames: vec![] };
        for f in read.frames {
            let claims = f.claims.iter().map(|c| ThreatClaim { member: c.member, class: c.class, basis: c.basis, take_basis: c.take_basis, bearers: c.bearers.iter().map(|b| (b.0 as u16, b.1 as u16)).collect(), answers: c.answers.iter().map(|a| (a.0 as u16, a.1 as u16, a.2, a.3 as u8)).collect(), tried: c.tried, unread: c.unread }).collect();
            row.frames.push(ThreatFrame { id: f.id, members: f.members.iter().map(|m| (m.0 as u8, m.1 as u16, m.2 as u16)).collect(), places: f.places.iter().map(|x| *x as u16).collect(), unread: f.unread, claims });
        }
        out.push(row);
    }
    Ok(())
}

/// questions_threats: for every wanted row of every game, every frame read with, per member that the other side bears on (or that the read holds lost, both, a threat or conditional), the units that bear on it,
/// its class and basis, and the answers of its owner (see `QReader::answers`); per frame its members and places, which are the read set of its claims. The classes and bases are those of `questions_state`.
#[allow(clippy::too_many_arguments)]
pub fn questions_threats(py: Python<'_>, variant: &str, st: &Static, goal: Option<u8>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, params: MeasureParams, answer_fuse: u64, threads: usize) -> PyResult<PyObject> {
    let want: FxHashMap<u64, u32> = keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
    let st2 = st.clone();
    let variant = variant.to_string();
    let accs = py.allow_threads(|| run_acc(games.len(), threads, Vec::<ThreatRow>::new, |i, acc| threat_game(&variant, &st2, goal, &games[i], &rows[i], &want, &params, answer_fuse, i, acc))).map_err(bad)?;
    let mut all: Vec<ThreatRow> = accs.into_iter().flatten().collect();
    all.sort_by_key(|r| (r.game, r.ply));
    let list = pyo3::types::PyList::empty(py);
    for r in all {
        let d = PyDict::new(py);
        d.set_item("game", r.game)?; d.set_item("ply", r.ply)?; d.set_item("mover", r.mover)?; d.set_item("board", r.board.iter().map(|b| vec![b.0 as u16, b.1 as u16, b.2]).collect::<Vec<_>>())?;
        d.set_item("questions", r.questions)?; d.set_item("played", r.played)?;
        let fl = pyo3::types::PyList::empty(py);
        for f in r.frames {
            let fd = PyDict::new(py);
            fd.set_item("id", f.id)?; fd.set_item("members", f.members)?; fd.set_item("places", f.places)?; fd.set_item("unread", f.unread)?;
            let cl = pyo3::types::PyList::empty(py);
            for c in f.claims {
                let cd = PyDict::new(py);
                cd.set_item("member", c.member)?; cd.set_item("class", c.class)?; cd.set_item("basis", c.basis)?; cd.set_item("take_basis", c.take_basis)?; cd.set_item("bearers", c.bearers)?;
                cd.set_item("answers", c.answers)?; cd.set_item("tried", c.tried)?; cd.set_item("unread", c.unread)?;
                cl.append(cd)?;
            }
            fd.set_item("claims", cl)?;
            fl.append(fd)?;
        }
        d.set_item("frames", fl)?;
        list.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("kinds", st.names.kinds.clone())?; out.set_item("roles", st.names.roles.clone())?; out.set_item("places", st.names.places.clone())?;
    out.set_item("rows", list)?;
    out.into_py_any(py)
}
