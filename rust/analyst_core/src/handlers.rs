//! PLAN 56 STAGE 2b: THE ROUTE READER, the first task body as a HANDLER (registered on the event "route popped"; its Lean words are the exchange law `Ideas.ExchangeLaw.seeXB`
//! (the static exchange), the mate in one of the rules, and `Ideas.Refuter.Refutes` for the standing kills). One call per route reads, for EVERY legal reply of theirs, what
//! agenda.py `_chase_route_body` computes before any inner agenda is opened:
//!   * the position after the route (`p0`, `p1` as `Pos`), the balance shift of the route itself;
//!   * the STANDING KILL (a reply whose position differs from an earlier route's only where that inner agenda's reads were not): the reply, its balance change and the index of the
//!     entry of `reply_kills[uci]` that matched (the caller prints the text from its own tuple);
//!   * the replies in the READING ORDER (recaptures on the route's square, checks, captures, the rest; the worst balance first inside a class; the uci last) as rows
//!     (uci, dbal, kind, best, is_check, fen, need, since0, exchange):
//!       kind "dead"   no legal move for us after the reply (is_check: they mate us; else stalemate)
//!       kind "mate1"  we have a mate in one: the rules close the reply at the mate value
//!       kind "exch"   dbal < 1, not in check, and the best static exchange `best` meets the route's need (1 - dbal) [and, when NEED_AWARE_CLOSE, dbal + best meets min_stake]: the rules
//!                     close it IF the goal's unmet fix (only when `unmet`) does not hold and THEIR pass finds nothing -- both are whole agendas, so they stay with the caller, which
//!                     resolves them in order, lazily, only for the replies it reaches (their agenda is memoised and files facts: the order of the reads is part of the answer)
//!       kind "open"   needs the inner agenda (`fen` is the position, `need` what the inner agenda must reach, `since0` / `exchange` the chain flags of the reply)
//!       kind "skip"   the caller already has this reply (an override, or closed by rule in the cheap pass): nothing is read
//! In promote mode only a mate in one closes a reply by rule. The caller keeps every effect on shared state (the counters, the log rows, the fact store's listener, their pass).
use super::*;
use crate::pos::{bal, pos_diff, pos_exchange, pos_mate_in_one, Pos};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};

struct Row { uci: String, dbal: i64, chk: bool, cap: bool, rec: bool, mate1: bool, q: Chess }

#[pyfunction]
#[pyo3(signature = (fen, move_uci, me_white, min_stake, need_aware, promote, target, goal_stake, mate_value, stake_promote, override_ucis, pre_ucis, reply_kills=None))]
#[allow(clippy::too_many_arguments)]
pub fn read_route<'py>(
    py: Python<'py>, fen: &str, move_uci: &str, me_white: bool, min_stake: i64, need_aware: bool, promote: bool, target: i64, goal_stake: Option<i64>, mate_value: i64,
    stake_promote: i64, override_ucis: Vec<String>, pre_ucis: Vec<String>, reply_kills: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let p0 = Pos::from_fen(fen)?;
    let p1 = p0.push_move(move_uci)?;
    let me = if me_white { Color::White } else { Color::Black };
    let shift = bal(&p1.pos, me) - bal(&p0.pos, me);
    let route_capture = {
        let m = shakmaty::uci::UciMove::from_ascii(move_uci.as_bytes()).ok().and_then(|u| u.to_move(&p0.pos).ok()).ok_or_else(|| PyValueError::new_err("illegal move"))?;
        m.is_capture()
    };
    let base = bal(&p1.pos, me);
    let mut rows: Vec<Row> = p1.pos.legal_moves().iter().map(|m| {
        let mut q = p1.pos.clone();
        q.play_unchecked(m);
        let mate1 = pos_mate_in_one(&q);
        Row { uci: m.to_uci(CastlingMode::Standard).to_string(), dbal: shift + bal(&q, me) - base, chk: q.is_check(), cap: m.is_capture(), rec: m.is_capture() && Some(m.to()) == p1.last_to, mate1, q }
    }).collect();
    let out = PyDict::new(py);
    out.set_item("p0", Py::new(py, p0)?)?;
    out.set_item("shift", shift)?;
    out.set_item("route_capture", route_capture)?;
    // the standing kill: the replies in the order of the position's move list, the first that an earlier entry of reply_kills covers
    let mut standing: Option<(String, i64, usize)> = None;
    if let Some(kills) = reply_kills {
        if !promote && !kills.is_empty() {
            'rows: for r in rows.iter() {
                if override_ucis.contains(&r.uci) { continue; }
                let Some(entries) = kills.get_item(&r.uci)? else { continue };
                for (k, e) in entries.try_iter()?.enumerate() {
                    let e = e?;
                    let reads = e.get_item(2)?;
                    let dbal0: i64 = e.get_item(1)?.extract()?;
                    if reads.is_none() || r.dbal >= 1 || dbal0 < r.dbal { continue; }
                    let old: PyRef<Pos> = e.get_item(0)?.extract()?;
                    let touched = pos_diff(&old.pos, &r.q).into_iter().any(|sq| reads.contains(sq as i64).unwrap_or(false));
                    if !touched { standing = Some((r.uci.clone(), r.dbal, k)); break 'rows; }
                }
            }
        }
    }
    out.set_item("standing", standing)?;
    out.set_item("p1", Py::new(py, p1)?)?;
    if out.get_item("standing")?.map(|s| !s.is_none()).unwrap_or(false) {
        out.set_item("rows", PyList::empty(py))?;
        return Ok(out);
    }
    rows.sort_by(|a, b| {
        let ka = if a.rec { 0 } else if a.chk { 1 } else if a.cap { 2 } else { 3 };
        let kb = if b.rec { 0 } else if b.chk { 1 } else if b.cap { 2 } else { 3 };
        ka.cmp(&kb).then(a.dbal.cmp(&b.dbal)).then_with(|| a.uci.cmp(&b.uci))
    });
    let list = PyList::empty(py);
    for r in rows.iter() {
        let skip = override_ucis.contains(&r.uci) || pre_ucis.contains(&r.uci);
        let since0 = route_capture && r.dbal >= 0;
        let (mut kind, mut best, mut fen_out, mut need) = ("open", 0i64, None::<String>, 0i64);
        if skip {
            kind = "skip";
        } else if r.q.legal_moves().is_empty() {
            kind = "dead";
        } else if r.mate1 {
            kind = "mate1";
        } else if !promote && r.dbal < 1 && !r.chk {
            let need_r = 1 - r.dbal;
            let b = pos_exchange(&r.q);
            if b >= need_r && !(need_aware && r.dbal + b < min_stake) { kind = "exch"; best = b; }
        }
        if kind == "exch" || kind == "open" {
            fen_out = Some(Fen::from_position(r.q.clone(), shakmaty::EnPassantMode::Legal).to_string());
            need = if promote && r.dbal <= 0 { stake_promote.min(1 - r.dbal) } else if promote { stake_promote } else if r.dbal < target { target - r.dbal } else { mate_value };
        }
        let unmet = kind == "exch" && goal_stake.map(|g| r.dbal + best < g).unwrap_or(false);
        let exch = route_capture && r.rec;
        list.append((r.uci.as_str(), r.dbal, kind, best, r.chk, fen_out, need, since0, exch, unmet))?;
    }
    out.set_item("rows", list)?;
    Ok(out)
}
