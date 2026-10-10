//! TARGET-FIRST PLANNING ON PATHS (Kaveh 2026-10-09, the castling position: "does the new rook attack anything? No ...
//! if I pass, all I'm going to lose is a pawn ... what attack can I cook up? ... if I castle and then I double, and then
//! I hop, I could set up a frame -- how do we see this frame without searching for it? And then how do we recognize
//! that every other frame is probably not going to give us anything? There is no more valuable piece to win"; "explain
//! it in a reasonable amount of calculations").
//!
//!  1 THEIR THREAT    on the null move: what we lose if we pass (their resolved captures) -- the TEMPO BUDGET: a plan
//!                    that needs a preparing move is allowed only if passing costs at most a pawn
//!  2 TARGETS         their pieces by value, the king first (mate); before each target, the STOP RULE: if what we
//!                    already have certified is worth at least the target, nothing below it can beat it
//!  3 SEEDS           only our moves on a PATH onto the target (paths.rs: arrivals and discoveries) -- each proved once
//!  4 ARMING          a seed that fails leaves the target escapes: our one-move paths onto those escapes or onto the
//!                    target (an arrival, or a battery loaded behind the seed's own piece) are the candidate preparing
//!                    moves -- each pair (preparing move, seed) proved once on the null move after it
//! Every proof is counted; the trace is the explanation.
use super::*;
use shakmaty::Bitboard;
use crate::paths::attack_paths;
use crate::waiting::{escapes, prove_n, MATE_GOAL};

pub struct Plan { pub trace: Vec<String>, pub best: Option<(String, String, i64, String)>, pub proofs: u32, pub nodes: u64 }

fn val(r: Role) -> i64 { if r == Role::King { 1000 } else { kernel_value(r) } }

fn san(pos: &Chess, uci: &str) -> String {
    uci.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok())
        .map(|m| shakmaty::san::San::from_move(pos, &m).to_string()).unwrap_or_else(|| uci.to_string())
}

pub fn plan(pos: &Chess) -> Plan {
    let us = pos.turn();
    let them = !us;
    let b = pos.board();
    let mut trace = Vec::new();
    let mut proofs = 0u32;
    let mut nodes = 0u64;
    let mut best: Option<(String, String, i64, String)> = None;    // (preparing move or "", seed, gain, how)
    // 1 THEIR THREAT
    let threat = match pass(pos) {
        Some(p) => { BUDGET.store(200_000, Ordering::Relaxed); let v = settle_d(&p, -2000, 2000, true, SETTLE_DEPTH).max(0); BUDGET.store(u64::MAX, Ordering::Relaxed); v }
        None => 0,
    };
    let checks: Vec<String> = match b.king_of(us) {
        Some(k) => attack_paths(pos, them, Bitboard::from_square(k)).into_iter().filter(|p| p.kind == "direct").filter_map(|p| p.mv).collect(),
        None => vec![],
    };
    let tempo = threat <= 1;
    trace.push(format!("their threat if we pass: captures worth {threat}; checks available to them: {}; a preparing move is {}",
        checks.len(), if tempo { "affordable" } else { "NOT affordable" }));
    // our checking moves: a check gives an already-attacked piece no time to escape (Kaveh: "a pin and also a cornering"
    // -- several frames from one move; 2026-10-09 misses: Qxf7+ winning the d1 bishop, f3+ the h3 rook, Qxf6+ the g2 bishop)
    let our_checks: Vec<String> = match b.king_of(them) {
        Some(k) => {
            let mut v: Vec<String> = Vec::new();
            for p in attack_paths(pos, us, Bitboard::from_square(k)) {
                match p.kind {
                    "direct" => if let Some(m) = p.mv { if !v.contains(&m) { v.push(m); } },
                    "discovery" => {
                        let line = shakmaty::attacks::between(p.target, p.piece);
                        for m in pos.legal_moves() {
                            if m.from() == p.via && !line.contains(m.to()) { let u = m.to_uci(CastlingMode::Standard).to_string(); if !v.contains(&u) { v.push(u); } }
                        }
                    }
                    _ => {}
                }
            }
            v
        }
        None => vec![],
    };
    // 2 TARGETS by value
    let mut armed: Vec<(i64, Square, Vec<String>)> = Vec::new();        // failing seeds kept for the arming phase
    let mut targets: Vec<(i64, Square)> = b.by_color(them).into_iter().filter_map(|sq| b.piece_at(sq).map(|p| (val(p.role), sq))).filter(|(v, _)| *v >= 3).collect();
    targets.sort_by(|a, b| b.0.cmp(&a.0));
    for (v, t) in targets {
        // the STOP RULE counts only IMMEDIATE wins (2026-10-09: an armed threat -- they get a move to parry it -- stopped
        // the search before Rf5 won a bishop, Rd4 a knight)
        let have = best.as_ref().filter(|x| x.0.is_empty()).map_or(0, |x| x.2);
        if have >= v.min(MATE_GOAL) {
            trace.push(format!("stop: {} on {t} is worth {}, not more than the {} already won", b.piece_at(t).unwrap().role.char(), v.min(MATE_GOAL), have));
            break;
        }
        let king = v == 1000;
        let goal = if king { MATE_GOAL } else { v };
        let name = format!("{:?} on {t}", b.piece_at(t).unwrap().role).to_lowercase();
        // 3 SEEDS: our moves on a path onto the target
        let paths = attack_paths(pos, us, Bitboard::from_square(t));
        let mut seeds: Vec<String> = Vec::new();
        for p in &paths {
            match p.kind {
                "direct" => if let Some(m) = &p.mv { if !seeds.contains(m) { seeds.push(m.clone()); } },
                "discovery" => {
                    let line = shakmaty::attacks::between(p.target, p.piece);
                    for m in pos.legal_moves() {
                        if m.from() == p.via && !line.contains(m.to()) {
                            let u = m.to_uci(CastlingMode::Standard).to_string();
                            if !seeds.contains(&u) { seeds.push(u); }
                        }
                    }
                }
                "open" => {
                    for m in pos.legal_moves() {                          // already attacked: the capture
                        if m.to() == t { let u = m.to_uci(CastlingMode::Standard).to_string(); if !seeds.contains(&u) { seeds.push(u); } }
                    }
                    if !king { for c in &our_checks { if !seeds.contains(c) { seeds.push(c.clone()); } } }   // ...or a check first
                }
                _ => {}
            }
        }
        if seeds.is_empty() {
            trace.push(format!("{name}: no path onto it this move -- nothing to prove (the path lemma)"));
            continue;
        }
        trace.push(format!("{name} (goal {goal}): {} moves on a path onto it: {}", seeds.len(), seeds.iter().map(|u| san(pos, u)).collect::<Vec<_>>().join(", ")));
        let mut failed: Vec<String> = Vec::new();
        let before = rules::threats_of(pos, us);
        for s in &seeds {
            // THE LONE-ATTACK LEMMA (2026-10-09, the castling position went from 4 proofs to 10 on attacks the piece just
            // steps away from): a move that attacks a piece with a safe escape wins nothing unless it also gives check or
            // attacks a second piece (escapes count pins). Not for the king: a check is the frame itself.
            if !king {
                if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                    let q = after(pos, &sm);
                    let now = rules::threats_of(&q, us);
                    let new = now.iter().filter(|(x, g)| **g > 0 && before.get(x).map_or(true, |g0| **g > *g0)).count();
                    let capture = sm.to() == t;
                    // ...and the king counts as a second target: a move that opens a new path onto their king (a check
                    // available next move) is a double threat (2026-10-09: Rf5 wins the c5 bishop because Black must
                    // answer the threat to the h5 king first)
                    let king_threat = b.king_of(them).map_or(false, |k| {
                        let n0 = attack_paths(pos, us, Bitboard::from_square(k)).iter().filter(|p| p.kind == "direct" || p.kind == "discovery").count();
                        let n1 = pass(&q).map_or(0, |p| attack_paths(&p, us, Bitboard::from_square(k)).iter().filter(|p| p.kind == "direct" || p.kind == "discovery").count());
                        n1 > n0
                    });
                    // (a capture elsewhere is a gain in hand: fxe5 attacking the f6 rook is a double threat)
                    if !capture && !sm.is_capture() && !q.is_check() && !king_threat && new < 2 && escapes(&q, us, t).any() {
                        trace.push(format!("    {}: a lone attack on a piece that can step away, no check, no second target, no new path onto the king -- no proof (the lone-attack lemma)", san(pos, s)));
                        failed.push(s.clone());
                        continue;
                    }
                }
            }
            // THE GOAL OF A SEED is what it can win, not the target's full value (2026-10-09: e5 against a pinned rook
            // gains 4 -- the pawn may fall -- and a fork wins the second piece): the static gain on the target after the
            // move (the exchange's first term), for two or more new targets the second best, for a capture the gain now
            let g = if king { goal } else if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                let q = after(pos, &sm);
                let now = rules::threats_of(&q, us);
                let mut new: Vec<i64> = now.iter().filter(|(x, gg)| **gg > 0 && before.get(x).map_or(true, |g0| **gg > *g0)).map(|(_, gg)| *gg).collect();
                new.sort_by(|a, b| b.cmp(a));
                if sm.to() == t { before.get(&(t as u8)).copied().unwrap_or(goal).max(1) }
                else if new.len() >= 2 { new[1].max(1) }
                else { now.get(&(t as u8)).copied().filter(|x| *x > 0).unwrap_or(goal).max(1) }
            } else { goal };
            let (ok, _, n, _) = prove_n(pos, s, &[t], g, None);
            proofs += 1; nodes += n;
            trace.push(format!("    prove {} (gain {g}): {} ({n} nodes)", san(pos, s), if ok { "HOLDS" } else { "fails" }));
            if ok { if best.as_ref().map_or(true, |x| g > x.2) { best = Some((String::new(), s.clone(), g, format!("{} wins {g} against the {name}", san(pos, s)))); } }
            else { failed.push(s.clone()); }
        }
        armed.push((goal, t, failed));
    }
    // 4 ARMING, only when no immediate win was found and a preparing move is affordable -- by value, stopping at the first
    // armed threat: the escapes each failing seed leaves, and our one-move paths onto them; a pair is proved only if the
    // preparing move, then the seed, leaves the target at most one escape (Kaveh: "how do we cover all three?")
    if best.is_none() && tempo {
      for (goal, t, failed) in armed {
        if best.is_some() { break; }
        let name = format!("{:?} on {t}", b.piece_at(t).unwrap().role).to_lowercase();
        for s in &failed {
            let Ok(u) = s.parse::<shakmaty::uci::UciMove>() else { continue };
            let Ok(sm) = u.to_move(pos) else { continue };
            let q = after(pos, &sm);
            let esc = escapes(&q, us, if sm.from() == Some(t) { sm.to() } else { t });
            let zone = esc.with(t);
            let mut arms: Vec<String> = Vec::new();
            for p in attack_paths(pos, us, zone) {
                let ok_kind = match p.kind {
                    "battery" => p.via == sm.from(),                       // loaded behind the seed's own piece
                    "direct" => p.piece != sm.from().unwrap_or(t),         // another piece covering an escape
                    _ => false,
                };
                if ok_kind { if let Some(m) = p.mv { if m != *s && !arms.contains(&m) { arms.push(m); } } }
            }
            if arms.is_empty() { continue; }
            trace.push(format!("    {} leaves escapes {:?}: preparing moves on a path onto them: {}", san(pos, s),
                esc.into_iter().map(|e| e.to_string()).collect::<Vec<_>>(), arms.iter().map(|u| san(pos, u)).collect::<Vec<_>>().join(", ")));
            for a in &arms {
                let Ok(au) = a.parse::<shakmaty::uci::UciMove>() else { continue };
                let Ok(am) = au.to_move(pos) else { continue };
                let q = after(pos, &am);
                if q.is_check() { continue; }
                let Some(p) = pass(&q) else { continue };
                let Some(sm2) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&p).ok()) else { continue };
                let q2 = after(&p, &sm2);
                let t2 = if sm2.from() == Some(t) { sm2.to() } else { t };
                // two or more escapes left open: no proof (one may still be chased -- Ne7+ Kb8 Rxd8#)
                if escapes(&q2, us, t2).count() >= 2 { continue; }
                let (ok, _, n, _) = prove_n(&p, s, &[t], goal, None);
                proofs += 1; nodes += n;
                trace.push(format!("        prove {} then {} (they pass): {} ({n} nodes)", san(pos, a), san(&p, s), if ok { "HOLDS -- an armed threat" } else { "fails" }));
                if ok && best.as_ref().map_or(true, |x| goal > x.2) {
                    best = Some((a.clone(), s.clone(), goal, format!("{} arms {} against the {name}", san(pos, a), san(&p, s))));
                }
            }
        }
      }
    }
    Plan { trace, best, proofs, nodes }
}

/// plan_trace(fen) -> (trace lines, (preparing move, seed, gain, summary) or None, proofs, nodes, ms)
#[pyfunction]
pub fn plan_trace(fen: String) -> PyResult<(Vec<String>, Option<(String, String, i64, String)>, u32, u64, f64)> {
    let Some(pos) = parse(&fen) else { return Ok((vec![], None, 0, 0, 0.0)) };
    let t = std::time::Instant::now();
    let p = plan(&pos);
    Ok((p.trace, p.best, p.proofs, p.nodes, t.elapsed().as_secs_f64() * 1000.0))
}
