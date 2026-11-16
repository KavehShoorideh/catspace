//! PLAN 78.72 A to D: A LINE READ TO A SETTLED END, game-free. SYSTEM_SPEC 4 ("concrete reasoning", how a line ends) and 8 (the threat list). One piece, one interface: it reads LINES of the game
//! for the threat list; it does not build frames, name a unit or a place, or hold a worth table (the judge is the caller's: the measured record of the kinds, `ledger::Worth`, or `ledger::Nothing`).
//!
//! INTERFACE (what it takes, what it returns, who calls it).
//!   `read_line(g, judge, s, m, me, params) -> LineRead`   the line that starts with the legal move `m` of the mover of `s`, then the best replies of both sides by the ledger search (`ledger::Search`: the
//!       minimax of the judge's sum over the units taken, deepened to `params.depth` plies while the node fuse lasts, the captures resolved at the horizon: each side may stop or go on capturing).
//!       Returns the move names of the line, its ledger (the units taken, as (was it `me`'s, kind)), its value for `me`, how it ended (QUIET at the settled horizon, or the end of the game won or lost) and
//!       whether the fuse cut it (`cut`: then the value is not a settled one and the caller marks the line UNREAD: no forced claim is made from a cut line, spec 4).
//!   `takings_on(g, s, place)`                the legal moves of the mover of `s` that take a unit standing on `place`.
//!   `read_takings`, `read_first`, `read_all` the lines of the takings on a place, of one move, of every legal move.
//!   `read_two_moves`                         the quiet moves of the mover after which, the other side passing once, a taking of a unit on `place` stands; each with the best taking line after it (a threat in
//!       two moves: the move that makes it is named).
//!   `forced_end(g, s, me, max_depth, fuse)`  the forcing read (`wins_in`, the engine's goal relation read through `Game::outcome`): does `me`, to move, force the end of the game within `max_depth` of its
//!       moves; with the line (the replies chosen to last longest). `end_answers(...)` the moves of the side to move at `s` after which the other side no longer forces it.
//!   Called by: the Python binding `read_lines` (py/src/lines_py.rs), for the threat list (catspace/incubator/analyst/discovery/threat_list.py). No player calls it.
use crate::ledger::{Judge, Search, WIN};
use crate::{won, wins_in, Budget};
use cs_game::placed::{Place, Placed};
use cs_game::{Game, Role};

/// The depth in plies after the first move and the node fuse of ONE line (a named fuse: the line it cuts is UNREAD).
#[derive(Clone, Copy, Debug)]
pub struct LineParams { pub depth: i32, pub fuse: u64 }

/// How a line ended: at the settled horizon (captures resolved), or the game's own end.
pub const END_QUIET: u8 = 0;
pub const END_WON: u8 = 1;
pub const END_LOST: u8 = 2;

#[derive(Clone, Debug)]
pub struct LineRead {
    /// the moves played before the read line (a threat in two moves: the quiet move that makes it), empty otherwise
    pub prefix: Vec<String>,
    pub lead: Option<(Place, Place)>,
    /// the ends of the first move of the read line
    pub first: (Place, Place),
    /// the move names of the line, the first move included
    pub names: Vec<String>,
    /// the ledger: every unit taken along the line, as (was it `me`'s, its kind as the game words it)
    pub events: Vec<(bool, String)>,
    /// the judge's sum for `me` along the line with the best replies (WIN-sized where the game ends)
    pub value: i64,
    pub end: u8,
    pub depth: i32,
    pub nodes: u64,
    pub cut: bool,
    /// how many moves of the line reach the last capture: the rest is play after the settled end (the search keeps a move there, it counts for nothing)
    pub settled: usize,
}

/// The number of moves of `names` (played from `s`) up to and including the last one that takes a unit (at least 1: the first move of a line is always shown).
fn settled_len<G: Game>(g: &G, s: &G::State, names: &[String]) -> usize {
    let (mut cur, mut last) = (s.clone(), 1);
    for (i, n) in names.iter().enumerate() {
        let Some(m) = g.find_move(&cur, n) else { break };
        if !g.captured(&cur, &m).is_empty() { last = i + 1; }
        cur = g.apply(&cur, &m);
    }
    last.min(names.len().max(1))
}

fn lost_kind(text: &str) -> Option<&str> { text.strip_prefix("lose ") }

pub fn read_line<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, m: &G::Move, me: Role, p: LineParams) -> LineRead {
    let mut sr = Search::with_fuse(g, judge, me, p.fuse);
    let d0 = sr.gain(s, m);
    let mut events: Vec<(bool, String)> = g.captured(s, m).into_iter().map(|(o, k)| (o == me, k)).collect();
    let mut names = vec![g.move_name(s, m)];
    let q = g.apply(s, m);
    let first = g.move_ends(s, m);
    let (value, depth, cut);
    if let Some(v) = g.outcome(&q) {
        value = d0 + if won(&v, me) { WIN } else if (0..v.len()).any(|i| i != me && won(&v, i)) { -WIN } else { 0 };
        depth = 0;
        cut = false;
    } else {
        let line = sr.run(&q, p.depth);
        value = d0 + line.value;
        depth = line.depth;
        cut = sr.cut() && line.depth < p.depth;
        names.extend(line.line);
        for (o, e) in line.lost { if let Some(k) = lost_kind(&e) { events.push((o == me, k.to_string())); } }
    }
    let end = if value >= WIN - 1000 { END_WON } else if value <= -WIN + 1000 { END_LOST } else { END_QUIET };
    let settled = settled_len(g, s, &names);
    LineRead { prefix: vec![], lead: None, first: (first.0, first.1), names, events, value, end, depth, nodes: sr.nodes, cut, settled }
}

pub fn takings_on<G: Game + Placed>(g: &G, s: &G::State, place: Place) -> Vec<G::Move> {
    g.moves(s).into_iter().filter(|m| g.move_ends(s, m).1 == place && !g.captured(s, m).is_empty()).collect()
}

pub fn read_takings<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, place: Place, me: Role, p: LineParams) -> Vec<LineRead> {
    takings_on(g, s, place).iter().map(|m| read_line(g, judge, s, m, me, p)).collect()
}

pub fn read_first<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, from: Place, to: Place, me: Role, p: LineParams) -> Vec<LineRead> {
    g.moves(s).into_iter().filter(|m| g.move_ends(s, m) == (from, to)).map(|m| read_line(g, judge, s, &m, me, p)).collect()
}

pub fn read_all<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, me: Role, p: LineParams) -> Vec<LineRead> {
    g.moves(s).iter().map(|m| read_line(g, judge, s, m, me, p)).collect()
}

/// A candidate response read two ways: the whole-board line of the move (what each side is left with), and the taking lines of a unit on `place` after it (the place follows the unit if the move is its own):
/// is the unit still won? A move that ends the game has no taking lines.
#[derive(Clone, Debug)]
pub struct After { pub whole: LineRead, pub takes: Vec<LineRead> }

pub fn read_after<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, m: &G::Move, place: Place, me: Role, p: LineParams) -> After {
    let whole = read_line(g, judge, s, m, me, p);
    let ends = g.move_ends(s, m);
    let c = g.apply(s, m);
    let now = if ends.0 == place { ends.1 } else { place };
    // the move's own taking counts in every taking line after it: the lines are read from the position after the move, so its gain and its ledger are added to each
    let d0 = Search::with_fuse(g, judge, me, 1).gain(s, m);
    let ev0: Vec<(bool, String)> = g.captured(s, m).into_iter().map(|(o, k)| (o == me, k)).collect();
    let name0 = g.move_name(s, m);
    let takes = if g.outcome(&c).is_some() { vec![] } else {
        read_takings(g, judge, &c, now, me, p).into_iter().map(|mut r| {
            r.value += d0;
            let mut ev = ev0.clone();
            ev.extend(r.events);
            r.events = ev;
            r.names.insert(0, name0.clone());
            r.settled += 1;
            r
        }).collect()
    };
    After { whole, takes }
}

/// The best line of the mover of `s` (the ledger search from `s` itself, one ply deeper than a line read after its first move): what `me` can count on if the position is its to play.
pub fn read_best<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, me: Role, p: LineParams) -> LineRead {
    let mut sr = Search::with_fuse(g, judge, me, p.fuse * 4);          // (the baseline of every response: read once, with more fuse than a line)
    let line = sr.run(s, p.depth + 1);
    let first = line.first_index.and_then(|i| g.moves(s).get(i).map(|m| g.move_ends(s, m))).unwrap_or((0, 0));
    let events = line.lost.iter().filter_map(|(o, e)| lost_kind(e).map(|k| (*o == me, k.to_string()))).collect();
    let end = if line.value >= WIN - 1000 { END_WON } else if line.value <= -WIN + 1000 { END_LOST } else { END_QUIET };
    let settled = settled_len(g, s, &line.line);
    LineRead { prefix: vec![], lead: None, first: (first.0, first.1), names: line.line, events, value: line.value, end, depth: line.depth, nodes: sr.nodes, cut: sr.cut() && line.depth < p.depth, settled }
}

/// The quiet moves of the mover of `s` that MAKE a threat on `place`: after the move and one pass of the other side a taking on `place` stands (the threat as the pass reads it). Each is then read as a whole-board
/// line with the other side's best replies (the move, the reply, ...): the threat is kept only if the line still ends ahead for `me` (the caller's verdict), so a move the other side simply answers by taking
/// the attacker is no threat. Returns the lines of the moves that make a threat, best first; `lead` is the move's ends.
pub fn read_two_moves<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, place: Place, me: Role, p: LineParams) -> Vec<LineRead> {
    let mut out: Vec<LineRead> = vec![];
    for m in g.moves(s) {
        if !g.captured(s, &m).is_empty() { continue; }
        let c = g.apply(s, &m);
        if g.outcome(&c).is_some() { continue; }
        let Some(cq) = g.pass(&c) else { continue };
        if takings_on(g, &cq, place).is_empty() { continue; }
        let mut r = read_line(g, judge, s, &m, me, p);
        let e = g.move_ends(s, &m);
        r.lead = Some((e.0, e.1));
        out.push(r);
    }
    out.sort_by_key(|r| -r.value);
    out
}

/// What `forced_end` found: `Proved` (the search finds a forced end of the game within `plies` plies of the position and the exact forcing read (`wins_in`: every reply of the other side) confirms its first
/// move: the number of moves of the side, the first move, the line), `Suspected` (the search finds it but the exact read could not confirm within its fuse: printed as suspected), `None` (the search finds no
/// forced end within the plies it read) or `Gap` (the fuse cut the search before it read the plies asked).
#[derive(Clone, Debug)]
pub enum Forced { Proved { depth: u32, first: (Place, Place), line: Vec<String> }, Suspected { depth: u32, first: (Place, Place), line: Vec<String> }, None { depth: u32 }, Gap { depth: u32 } }

/// Does `me`, to move at `s`, force the end of the game within `max_plies` plies? The search that counts nothing (`ledger::Nothing`: only the game's own end has a value, 78.61) is deepened ply by ply; a
/// win found is then confirmed by the exact forcing read on its first move.
pub fn forced_end<G: Game + Placed>(g: &G, s: &G::State, me: Role, max_plies: u32, fuse: u64) -> Forced {
    let judge = crate::ledger::Nothing;
    let mut sr = Search::with_fuse(g, &judge, me, fuse);
    let line = sr.run(s, max_plies as i32);
    if line.value >= WIN - 1000 {
        let plies = line.depth as u32;
        let n = plies.div_ceil(2).max(1);
        let ms = g.moves(s);
        let first = line.first_index.and_then(|i| ms.get(i).cloned());
        let Some(m) = first else { return Forced::Gap { depth: plies } };
        let ends = g.move_ends(s, &m);
        let c = g.apply(s, &m);
        let mut b = Budget::new(1e9, fuse);
        let sure = match g.outcome(&c) {
            Some(v) => Ok(won(&v, me)),
            None => (|| -> Result<bool, crate::Cut> {
                for r in g.moves(&c) {
                    b.spend()?;
                    let q2 = g.apply(&c, &r);
                    let ok = match g.outcome(&q2) { Some(v) => won(&v, me), None => n > 1 && wins_in(g, &q2, me, n - 1, &mut b)?.is_some() };
                    if !ok { return Ok(false); }
                }
                Ok(true)
            })(),
        };
        let names = line.line.clone();
        return match sure { Ok(true) => Forced::Proved { depth: n, first: (ends.0, ends.1), line: names }, _ => Forced::Suspected { depth: n, first: (ends.0, ends.1), line: names } };
    }
    if sr.cut() && line.depth < max_plies as i32 { Forced::Gap { depth: line.depth as u32 } } else { Forced::None { depth: line.depth as u32 } }
}

/// The moves of the mover of `s` after which the other side (`attacker`) no longer forces the end within `max_plies` plies (the same search): (ends, name, verdict) with verdict 0 = answers it, 1 = still
/// forced, 2 = the fuse cut the reading (unread), 3 = the move itself ends the game against the mover.
pub fn end_answers<G: Game + Placed>(g: &G, s: &G::State, attacker: Role, max_plies: u32, fuse: u64) -> Vec<((Place, Place), String, u8)> {
    let judge = crate::ledger::Nothing;
    let mut out = vec![];
    for r in g.moves(s) {
        let c = g.apply(s, &r);
        let ends = g.move_ends(s, &r);
        let name = g.move_name(s, &r);
        let v = if let Some(o) = g.outcome(&c) { if won(&o, attacker) { 3 } else { 0 } } else {
            let mut sr = Search::with_fuse(g, &judge, attacker, fuse);
            let line = sr.run(&c, max_plies.saturating_sub(1).max(1) as i32);
            if line.value >= WIN - 1000 { 1 } else if sr.cut() && line.depth < max_plies.saturating_sub(1).max(1) as i32 { 2 } else { 0 }
        };
        out.push(((ends.0, ends.1), name, v));
    }
    out
}
