//! PLAN 79.08 B, STEP 2 OF THE WAKE (SYSTEM_SPEC 8 "split the position into frames and drop those with no bearing", 3 frames; piece: the wake). The FRAME READ AS THE SOURCE OF THE THREAT LIST'S ROWS: the Rust form of catspace/incubator/analyst/discovery/threat_list.py `threat_rows` over the frame read of
//! cs_tactics::sheet::threat_frames (called in process; the binding's `questions_threats` runs the same code). The graph rows of threats.rs (`graph_rows`) remain the fallback when no learned seeds are given. Game-free: a unit, a kind, a place is a number the game gave.
//!
//! INTERFACE (written before the bodies).
//!   `FrameRead { want, params }`                      the learned seeds' keys and the frame read's parameters (fuses, kind values, bands, rules). Owned by the caller (it is learned data); None: no frame read, the graph rows.
//!   `frame_read(st, g, s, goal, &FrameRead, stake) -> FrameRows`   takes the static graph, the position, the goal kind, the read and the stake function of the caller; returns the rows of the threat list (`FrameRow`, in the order of stake then class then place: the same rows `threat_rows` makes, both sides) and
//!       the READ SET of each row (the members and places of the frame that explains it, with the side to move: what the read looked at, to the last unit), for the carry (step 1).
//!   Called by: the wake's loop (`wake::decide`) and the unit tests.
use crate::threats::FrameRow;
use crate::wake::UnitRef;
use cs_game::placed::{Kind, Place, Placed};
use cs_game::{Game, Role};
use cs_tactics::sheet::patterns::FxHashMap;
use cs_tactics::sheet::threat_frames::{threat_frames, FrameReadParams, ThreatFrames};
use std::collections::{BTreeMap, BTreeSet};

/// The class words of a unit in a frame, by the code of the frame read (tactical_state.CLASS_WORDS).
pub const CLASS: [&str; 7] = ["lost", "safe", "both", "unread", "threat", "conditional", "contested"];
/// The classes that make a row: lost, both, threat, conditional, contested.
pub const THREAT_CLASSES: [u8; 5] = [0, 2, 4, 5, 6];

/// The learned seeds and the parameters of the frame read.
#[derive(Debug, Clone, Default)]
pub struct FrameRead { pub want: Vec<u64>, pub params: FrameReadParams }

/// THE READ SET of a claim (threat_list.read_set_of): the side to move, the members of its frame as (side, kind, place), the places of the frame; what the read looked at, to the last unit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReadSet { pub mover: Role, pub members: Vec<(Role, Kind, Place)>, pub places: Vec<Place> }

/// The rows of the frame read and, for each row's unit, the read set of the frame that explains it, with the cost of the read.
#[derive(Debug, Clone, Default)]
pub struct FrameRows { pub rows: Vec<FrameRow>, pub read_sets: BTreeMap<UnitRef, ReadSet>, pub frames: usize, pub questions: u64, pub seconds: f64,
    /// PLAN 79.09 B: the frames the clock of the step did not reach (a gap, never a refutation)
    pub unread_frames: usize }

struct Flat { owner: Role, kind: Kind, place: Place, class: u8, bearers: Vec<(Kind, Place)>, answers: Vec<(Place, Place, u8, bool)>, n_members: usize, frame_id: String, members: BTreeSet<(Role, Kind, Place)>, read_set: ReadSet }

fn flat(raw: &ThreatFrames, mover: Role, other: Role) -> Vec<Flat> {
    let side = |own: bool| if own { mover } else { other };
    let mut out = vec![];
    for f in &raw.frames {
        let members: Vec<(Role, Kind, Place)> = { let mut m: Vec<_> = f.members.iter().map(|x| (side(x.0), x.1, x.2)).collect(); m.sort(); m };
        let mut places = f.places.clone();
        places.sort();
        let rs = ReadSet { mover, members: members.clone(), places };
        for c in &f.claims {
            let (own, kind, place) = f.members[c.member as usize];
            out.push(Flat { owner: side(own), kind, place, class: c.class, bearers: c.bearers.clone(), answers: c.answers.clone(), n_members: f.members.len(), frame_id: f.id.clone(), members: members.iter().copied().collect(), read_set: rs.clone() });
        }
    }
    out
}

/// THE THREAT LIST'S ROWS of the frame read (threat_list.threat_rows): per unit, the smallest frame of the worst class that makes a row explains it; the first answers found in that frame or a strict widening of it are the row's response; both sides.
pub fn rows_of(raw: &ThreatFrames, mover: Role, other: Role, kinds: usize, stake: Option<&dyn Fn(&str, bool) -> Option<f64>>, kind_words: &[String]) -> FrameRows {
    let cl = flat(raw, mover, other);
    let mut by_unit: BTreeMap<(Role, Kind, Place), Vec<&Flat>> = BTreeMap::new();
    for c in &cl { by_unit.entry((c.owner, c.kind, c.place)).or_default().push(c); }
    let _ = kinds;
    let mut out = FrameRows { frames: raw.frames.len(), questions: raw.questions, unread_frames: raw.unread_frames.len() + raw.unbuilt_seeds, ..Default::default() };
    for ((owner, kind, place), cs) in by_unit {
        let wins: Vec<&&Flat> = cs.iter().filter(|c| THREAT_CLASSES.contains(&c.class)).collect();
        if wins.is_empty() { continue; }
        let worst = wins.iter().map(|c| c.class).min().unwrap();
        let mut ws: Vec<&&Flat> = wins.iter().filter(|c| c.class == worst).copied().collect();
        ws.sort_by(|a, b| (a.n_members, &a.frame_id).cmp(&(b.n_members, &b.frame_id)));
        let win = **ws.first().unwrap();
        let mut wider: Vec<&Flat> = cs.iter().filter(|c| !std::ptr::eq(**c, win) && c.members.is_superset(&win.members) && c.members.len() > win.members.len()).copied().collect();
        wider.sort_by(|a, b| (a.n_members, &a.frame_id).cmp(&(b.n_members, &b.frame_id)));
        let mut scope: Vec<&Flat> = vec![win];
        scope.extend(wider);
        let resp: Vec<(Place, Place, u8, bool)> = scope.iter().find(|c| !c.answers.is_empty() && c.class != 0).map(|c| c.answers.clone()).unwrap_or_default();
        let unit = UnitRef { owner, kind, at: place };
        let by: Vec<UnitRef> = win.bearers.iter().map(|(k, p)| UnitRef { owner: if owner == mover { other } else { mover }, kind: *k, at: *p }).collect();
        let st = stake.and_then(|f| f(&kind_words[kind as usize], owner == mover));
        out.rows.push(FrameRow { unit, class: CLASS[win.class as usize].to_string(), by, response_ix: resp, stake: st });
        out.read_sets.insert(unit, win.read_set.clone());
    }
    out.rows.sort_by(|a, b| {
        let key = |r: &FrameRow| (-(r.stake.unwrap_or(-9.0)), CLASS.iter().position(|c| *c == r.class).unwrap_or(9), r.unit.at);
        let (x, y) = (key(a), key(b));
        x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2))
    });
    out
}

/// STEP 2: read the frames of the position with the learned seeds and make the rows of the threat list. Empty rows when the seeds name nothing on the board (the caller then falls back to the graph rows).
pub fn frame_read<G: Game + Placed>(st: &cs_tactics::sheet::base_graph::Static, g: &G, s: &G::State, goal: Option<Kind>, fr: &FrameRead, stake: Option<&dyn Fn(&str, bool) -> Option<f64>>) -> FrameRows {
    let t0 = std::time::Instant::now();
    let Some(mover) = g.mover(s) else { return FrameRows::default() };
    let want: FxHashMap<u64, u32> = fr.want.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
    let raw = threat_frames(st, g, s, goal, &want, &fr.params);
    let mut out = rows_of(&raw, mover, g.other(mover), g.kind_names().len(), stake, &g.kind_names());
    out.seconds = t0.elapsed().as_secs_f64();
    out
}
