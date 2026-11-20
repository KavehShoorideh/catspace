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
    /// PLAN 78.77: the kind changes along the line (the other event of the ledger, `Game::changed`): (was it `me`'s, the kind left -- "" when the unit arrived from nothing --, the kind arrived)
    pub changes: Vec<(bool, String, String)>,
    /// the judge's sum for `me` along the line with the best replies (WIN-sized where the game ends)
    pub value: i64,
    pub end: u8,
    pub depth: i32,
    pub nodes: u64,
    pub cut: bool,
    /// how many moves of the line reach the last capture: the rest is play after the settled end (the search keeps a move there, it counts for nothing)
    pub settled: usize,
    /// PLAN 78.85: the line, read on through its tail of captures, still ends in the middle of an exchange (a fuse stopped the tail): the caller marks it UNSETTLED, not REFUTED
    pub unsettled: bool,
    /// PLAN 78.87: how the TAIL ended (END_QUIET: nothing more is taken, or the game's own end won (END_WON) or lost (END_LOST) by `me` inside the tail: the judge counts the end of the game as an event of the ledger
    /// (the search's own `terminal`), so a taking that lets the other side end the game is not played, and a tail that does reach the end says so)
    pub tail_end: u8,
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

/// The node fuse of one tail (a named fuse: a tail it cuts leaves the line UNSETTLED).
pub const TAIL_NODES: u32 = 20_000;

fn lost_kind(text: &str) -> Option<&str> { text.strip_prefix("lose ") }

/// PLAN 78.85 PART TWO: THE TAIL OF A LINE. The ledger search resolves the captures at its horizon (each side may stand or go on capturing) but its printed line (the table's best moves) stops at the horizon,
/// so a line that ends inside an exchange shows the first capture and not the recapture the search counted. `settle_tail` reads the line on from its last position by the same rule: the side to move stands
/// (the ledger as it is) or takes on; returns the moves of the best run of captures (each side's best for `me`), whether the run met the named fuse `QUIET_FUSE` (then the line is UNSETTLED: it is still in the
/// middle of an exchange when a fuse stops it). PLAN 78.87: it is a READING by the judge, not a run of takings: the end of the game is counted (`WIN` less the plies), so a taking after which the other side ends the game is
/// worse than standing and is not played, and the side to move stands whenever every taking leaves it worse by the judge's own count (a move that takes nothing and changes no kind is not read, as in the search's own horizon).
pub fn settle_tail<G: Game, J: Judge>(g: &G, judge: &J, me: Role, s: &G::State) -> (Vec<G::Move>, bool) {
    fn go<G: Game, J: Judge>(g: &G, judge: &J, me: Role, s: &G::State, depth: u32, mut alpha: i64, mut beta: i64, hit: &mut bool, nodes: &mut u32) -> (i64, Vec<G::Move>) {
        if let Some(v) = g.outcome(s) {                                  // (PLAN 78.87: the end of the game is counted as the search counts it: above or below every ledger, the sooner the larger)
            return (if won(&v, me) { WIN - depth as i64 } else if (0..v.len()).any(|i| i != me && won(&v, i)) { -WIN + depth as i64 } else { 0 }, vec![]);
        }
        *nodes += 1;
        if depth >= crate::ledger::QUIET_FUSE || *nodes > TAIL_NODES { *hit = true; return (0, vec![]); }
        let maxing = g.mover(s) == Some(me);
        let probe = Search::with_fuse(g, judge, me, 1);
        let mut best: (i64, Vec<G::Move>) = (0, vec![]);                 // standing: nothing more is taken
        if maxing { if best.0 >= beta { return best; } alpha = alpha.max(best.0); } else { if best.0 <= alpha { return best; } beta = beta.min(best.0); }
        for m in g.moves(s) {
            let d = probe.gain(s, &m);
            if d == 0 { continue; }                                      // (captures and kind changes that add to the ledger only)
            let (v, mut rest) = go(g, judge, me, &g.apply(s, &m), depth + 1, alpha - d, beta - d, hit, nodes);
            let v = d + v;
            if (maxing && v > best.0) || (!maxing && v < best.0) { rest.insert(0, m); best = (v, rest); }
            if maxing { alpha = alpha.max(v); } else { beta = beta.min(v); }
            if alpha >= beta { break; }
        }
        best
    }
    let mut hit = false;
    let mut nodes = 0u32;
    let (_, tail) = go(g, judge, me, s, 0, i64::MIN / 2, i64::MAX / 2, &mut hit, &mut nodes);
    (tail, hit)
}

/// The line `names` (played from `s`) read on through the tail: the names of the tail moves, its captures and kind changes, and whether it is unsettled. The events of the line then agree with the
/// search's own value, which counts the tail.
fn read_on<G: Game, J: Judge>(g: &G, judge: &J, me: Role, s: &G::State, names: &[String]) -> (Vec<String>, Vec<(bool, String)>, Vec<(bool, String, String)>, bool, u8) {
    let mut cur = s.clone();
    for n in names {
        let Some(m) = g.find_move(&cur, n) else { return (vec![], vec![], vec![], false, END_QUIET) };
        cur = g.apply(&cur, &m);
    }
    if g.outcome(&cur).is_some() { return (vec![], vec![], vec![], false, END_QUIET); }
    let (tail, hit) = settle_tail(g, judge, me, &cur);
    let (mut tn, mut ev, mut ch) = (vec![], vec![], vec![]);
    for m in tail {
        ev.extend(g.captured(&cur, &m).into_iter().map(|(o, k)| (o == me, k)));
        ch.extend(changes_of(g, &cur, &m, me));
        tn.push(g.move_name(&cur, &m));
        cur = g.apply(&cur, &m);
    }
    let end = match g.outcome(&cur) { Some(v) if won(&v, me) => END_WON, Some(v) if (0..v.len()).any(|i| i != me && won(&v, i)) => END_LOST, _ => END_QUIET };
    (tn, ev, ch, hit, end)
}

/// The kind changes of one move, for `me`.
fn changes_of<G: Game>(g: &G, s: &G::State, m: &G::Move, me: Role) -> Vec<(bool, String, String)> {
    let mover = g.mover(s).unwrap_or(me);
    g.changed(s, m).into_iter().map(|(a, b)| (mover == me, a, b)).collect()
}

/// The kind changes in a search's line events.
fn changes_in(lost: &[(Role, String)], me: Role) -> Vec<(bool, String, String)> {
    lost.iter().filter_map(|(o, e)| crate::ledger::change_event_kinds(e).map(|(a, b)| (*o == me, a, b))).collect()
}

pub fn read_line<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, m: &G::Move, me: Role, p: LineParams) -> LineRead {
    let mut sr = Search::with_fuse(g, judge, me, p.fuse);
    read_line_in(&mut sr, s, m, me, p)
}

/// PLAN 79.08 (repair 2): `read_line` on a search the caller keeps, so that the lines read from one position share one table (the sub-positions they have in common are read once). The line has its own node fuse (`p.fuse` from here), its own count of nodes, and the
/// table kept; with a fresh search it is exactly `read_line`. The search must be for `me`.
pub fn read_line_in<G: Game + Placed, J: Judge>(sr: &mut Search<'_, G, J>, s: &G::State, m: &G::Move, me: Role, p: LineParams) -> LineRead {
    let (g, judge) = (sr.g, sr.judge);
    sr.rebudget(p.fuse);
    let n0 = sr.nodes;
    let d0 = sr.gain(s, m);
    let mut events: Vec<(bool, String)> = g.captured(s, m).into_iter().map(|(o, k)| (o == me, k)).collect();
    let mut changes = changes_of(g, s, m, me);
    let mut names = vec![g.move_name(s, m)];
    let q = g.apply(s, m);
    let first = g.move_ends(s, m);
    let (value, depth, cut);
    let (mut unsettled, mut tail_end) = (false, END_QUIET);
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
        changes.extend(changes_in(&line.lost, me));
        for (o, e) in line.lost { if let Some(k) = lost_kind(&e) { events.push((o == me, k.to_string())); } }
        let (tn, tev, tch, hit, te) = read_on(g, judge, me, s, &names);
        names.extend(tn);
        events.extend(tev);
        changes.extend(tch);
        unsettled = hit;
        tail_end = te;
    }
    // PLAN 78.87: `value` here carries the first move's own gain (a unit taken), which can be larger than the margin the search's own value is read with: a line that takes a unit and then loses the game was counted QUIET
    // (value -WIN plus the gain) and judged by its events alone. The end of the game is read against half of WIN, as the ledger's own numbers are (WIN less the plies, the sum of a ledger is far below it).
    let end = if value >= WIN / 2 { END_WON } else if value <= -WIN / 2 { END_LOST } else { END_QUIET };
    let settled = settled_len(g, s, &names);
    LineRead { prefix: vec![], lead: None, first: (first.0, first.1), names, events, changes, value, end, depth, nodes: sr.nodes - n0, cut, settled, unsettled, tail_end }
}

pub fn takings_on<G: Game + Placed>(g: &G, s: &G::State, place: Place) -> Vec<G::Move> {
    g.moves(s).into_iter().filter(|m| g.move_ends(s, m).1 == place && !g.captured(s, m).is_empty()).collect()
}

pub fn read_takings<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, place: Place, me: Role, p: LineParams) -> Vec<LineRead> {
    takings_on(g, s, place).iter().map(|m| read_line(g, judge, s, m, me, p)).collect()
}

/// PLAN 79.08 (repair 2): `read_takings` on a shared search (one table for the lines of one position).
pub fn read_takings_in<G: Game + Placed, J: Judge>(sr: &mut Search<'_, G, J>, s: &G::State, place: Place, me: Role, p: LineParams) -> Vec<LineRead> {
    let g = sr.g;
    takings_on(g, s, place).iter().map(|m| read_line_in(sr, s, m, me, p)).collect()
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
    let mut sr = Search::with_fuse(g, judge, me, p.fuse);
    read_after_in(&mut sr, s, m, place, me, p)
}

/// PLAN 79.08 (repair 2): `read_after` on a shared search.
pub fn read_after_in<G: Game + Placed, J: Judge>(sr: &mut Search<'_, G, J>, s: &G::State, m: &G::Move, place: Place, me: Role, p: LineParams) -> After {
    let (g, judge) = (sr.g, sr.judge);
    let whole = read_line_in(sr, s, m, me, p);
    let ends = g.move_ends(s, m);
    let c = g.apply(s, m);
    let now = if ends.0 == place { ends.1 } else { place };
    // the move's own taking counts in every taking line after it: the lines are read from the position after the move, so its gain and its ledger are added to each
    let d0 = Search::with_fuse(g, judge, me, 1).gain(s, m);
    let ev0: Vec<(bool, String)> = g.captured(s, m).into_iter().map(|(o, k)| (o == me, k)).collect();
    let ch0 = changes_of(g, s, m, me);
    let name0 = g.move_name(s, m);
    let takes = if g.outcome(&c).is_some() { vec![] } else {
        read_takings_in(sr, &c, now, me, p).into_iter().map(|mut r| {
            r.value += d0;
            let mut ev = ev0.clone();
            ev.extend(r.events);
            r.events = ev;
            let mut ch = ch0.clone();
            ch.extend(std::mem::take(&mut r.changes));
            r.changes = ch;
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
    let mut events: Vec<(bool, String)> = line.lost.iter().filter_map(|(o, e)| lost_kind(e).map(|k| (*o == me, k.to_string()))).collect();
    let mut changes = changes_in(&line.lost, me);
    let mut names = line.line.clone();
    let (tn, tev, tch, unsettled, tail_end) = read_on(g, judge, me, s, &names);
    names.extend(tn);
    events.extend(tev);
    changes.extend(tch);
    let end = if line.value >= WIN - 1000 { END_WON } else if line.value <= -WIN + 1000 { END_LOST } else { END_QUIET };
    let settled = settled_len(g, s, &names);
    LineRead { prefix: vec![], lead: None, first: (first.0, first.1), names, events, changes, value: line.value, end, depth: line.depth, nodes: sr.nodes, cut: sr.cut() && line.depth < p.depth, settled, unsettled, tail_end }
}

/// The quiet moves of the mover of `s` that MAKE a threat on `place`: after the move and one pass of the other side a taking on `place` stands (the threat as the pass reads it). Each is then read as a whole-board
/// line with the other side's best replies (the move, the reply, ...): the threat is kept only if the line still ends ahead for `me` (the caller's verdict), so a move the other side simply answers by taking
/// the attacker is no threat. Returns the lines of the moves that make a threat, best first; `lead` is the move's ends.
pub fn read_two_moves<G: Game + Placed, J: Judge>(g: &G, judge: &J, s: &G::State, place: Place, me: Role, p: LineParams) -> Vec<LineRead> {
    let mut sr = Search::with_fuse(g, judge, me, p.fuse);
    read_two_moves_in(&mut sr, s, place, me, p)
}

/// PLAN 79.08 (repair 2): `read_two_moves` on a shared search.
pub fn read_two_moves_in<G: Game + Placed, J: Judge>(sr: &mut Search<'_, G, J>, s: &G::State, place: Place, me: Role, p: LineParams) -> Vec<LineRead> {
    let g = sr.g;
    let mut out: Vec<LineRead> = vec![];
    for m in g.moves(s) {
        if !g.captured(s, &m).is_empty() { continue; }
        let c = g.apply(s, &m);
        if g.outcome(&c).is_some() { continue; }
        let Some(cq) = g.pass(&c) else { continue };
        if takings_on(g, &cq, place).is_empty() { continue; }
        let mut r = read_line_in(sr, s, &m, me, p);
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
