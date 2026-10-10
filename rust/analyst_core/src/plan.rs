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

pub struct Plan { pub trace: Vec<String>, pub best: Option<(String, String, i64, String)>, pub proofs: u32, pub nodes: u64, pub target: Option<Square>,
                 pub immediate: Option<(String, String, i64, String)> }

fn val(r: Role) -> i64 { if r == Role::King { 1000 } else { kernel_value(r) } }

fn san(pos: &Chess, uci: &str) -> String {
    uci.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok())
        .map(|m| shakmaty::san::San::from_move(pos, &m).to_string()).unwrap_or_else(|| uci.to_string())
}

pub fn plan(pos: &Chess) -> Plan { plan_inner(pos, false) }

/// `immediate_only`: the immediate phase alone (targets, seeds, proofs) -- what the OPPONENT's threat is computed with,
/// on the null move.
pub fn plan_inner(pos: &Chess, immediate_only: bool) -> Plan {
    let us = pos.turn();
    let them = !us;
    let b = pos.board();
    let mut trace = Vec::new();
    let mut proofs = 0u32;
    let mut nodes = 0u64;
    let mut best: Option<(String, String, i64, String)> = None;    // (preparing move or "", seed, gain, how)
    let mut best_target: Option<Square> = None;
    // 1 THEIR THREAT: their resolved captures if we pass, AND their immediate wins found by the same reasoning on the
    // null move (2026-10-09, after O-O-O in the castling line: Black's planner said "nothing" -- the threat counted only
    // White's captures, not White's mate)
    let mut their: Option<(String, Square, i64)> = None;
    let threat = if immediate_only { 0 } else {
        let cap = match pass(pos) {
            Some(p) => { BUDGET.store(200_000, Ordering::Relaxed); let v = settle_d(&p, -2000, 2000, true, SETTLE_DEPTH).max(0); BUDGET.store(u64::MAX, Ordering::Relaxed); v }
            None => 0,
        };
        if let Some(p) = pass(pos) {
            let tp = plan_inner(&p, true);
            proofs += tp.proofs; nodes += tp.nodes;
            if let (Some(bb), Some(tt)) = (tp.best, tp.target) {
                trace.push(format!("their best immediate win if we pass: {} ({} proofs)", bb.3, tp.proofs));
                their = Some((bb.1, tt, bb.2));
            }
        }
        cap.max(their.as_ref().map_or(0, |x| x.2))
    };
    let tempo = threat <= 1;
    if !immediate_only {
        trace.push(format!("their threat if we pass: worth {threat}; a preparing move is {}", if tempo { "affordable" } else { "NOT affordable" }));
    }
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
            if ok { if best.as_ref().map_or(true, |x| g > x.2) { best = Some((String::new(), s.clone(), g, format!("{} wins {g} against the {name}", san(pos, s)))); best_target = Some(t); } }
            else { failed.push(s.clone()); }
        }
        armed.push((goal, t, failed));
    }
    if immediate_only { let im = best.clone(); return Plan { trace, best, proofs, nodes, target: best_target, immediate: im }; }
    // 3b DEFENCE: their immediate win is bigger than ours -- the candidate defences are read off their winning move's
    // geometry (the path lemma): take the attacking piece, move the target, guard the square it lands on, step onto a
    // line of theirs into the target's zone, free an escape next to the target, or check; one proof each: does their
    // move still win after ours?
    let ours_now = best.as_ref().map_or(0, |x| x.2);
    let immediate = best.clone();
    if let Some((tseed, tt, tg)) = their.clone() {
        // DOES OUR ATTACK ALREADY ANSWER THEIRS? (Kaveh 2026-10-09; the census: defending first lost 6% of the immediate
        // wins) -- one proof: their winning move after our best one
        let answered = match &immediate {
            Some(x) => {
                let q = x.1.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()).map(|m| after(pos, &m));
                match q {
                    Some(q) if q.is_checkmate() => true,
                    Some(q) => {
                        let legal = tseed.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&q).ok()).is_some();
                        if !legal { true } else { let (ok, _, n, _) = prove_n(&q, &tseed, &[tt], tg, None); proofs += 1; nodes += n; !ok }
                    }
                    None => false,
                }
            }
            None => false,
        };
        if answered { trace.push(format!("our {} also stops their threat", best.as_ref().unwrap().3)); }
        if tg >= 2 && tg > ours_now && !answered {
            let p0 = pass(pos).unwrap();
            let sm = tseed.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&p0).ok());
            if let Some(sm) = sm {
                let from = sm.from();
                let land = sm.to();
                let mut lines = Bitboard::EMPTY;                       // their lines into the target's zone
                let zone = shakmaty::attacks::king_attacks(tt).with(tt);
                for p in attack_paths(pos, them, zone) {
                    if matches!(p.kind, "open" | "discovery" | "battery" | "direct") { lines |= shakmaty::attacks::between(p.piece, p.target); if let Some(r) = p.rear { lines.add(r); } }
                }
                let mut cands: Vec<Move> = Vec::new();
                for m in pos.legal_moves() {
                    let q = after(pos, &m);
                    let hit = Some(m.to()) == from || m.from() == Some(tt) || q.board().attacks_to(land, us, q.board().occupied()).any()
                        && !b.attacks_to(land, us, b.occupied()).any()
                        || lines.contains(m.to()) || m.from().map_or(false, |f| shakmaty::attacks::king_attacks(tt).contains(f)) || q.is_check();
                    if hit { cands.push(m); }
                }
                trace.push(format!("their {} wins {tg}: {} candidate defences on its geometry", san(&p0, &tseed), cands.len()));
                for m in cands {
                    let q = after(pos, &m);
                    let mu = m.to_uci(CastlingMode::Standard).to_string();
                    let t2 = if m.from() == Some(tt) { m.to() } else { tt };
                    let legal = tseed.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&q).ok()).is_some();
                    let still = if legal { let (ok, _, n, _) = prove_n(&q, &tseed, &[t2], tg, None); proofs += 1; nodes += n; ok } else { false };
                    if !still {
                        trace.push(format!("    {} stops it", san(pos, &mu)));
                        if best.as_ref().map_or(true, |x| x.0 != "defend") {
                            best = Some(("defend".to_string(), mu.clone(), -tg, format!("{} stops their {}", san(pos, &mu), san(&p0, &tseed))));
                        }
                    }
                }
                return Plan { trace, best, proofs, nodes, target: best_target, immediate };
            }
        }
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
    Plan { trace, best, proofs, nodes, target: best_target, immediate }
}

/// plan_trace(fen) -> (trace lines, the decision (preparing move / "defend" / "", seed, gain, summary), proofs, nodes, ms,
/// the best immediate win found (whatever was decided))
#[pyfunction]
pub fn plan_trace(fen: String) -> PyResult<(Vec<String>, Option<(String, String, i64, String)>, u32, u64, f64, Option<(String, String, i64, String)>)> {
    let Some(pos) = parse(&fen) else { return Ok((vec![], None, 0, 0, 0.0, None)) };
    let t = std::time::Instant::now();
    let p = plan(&pos);
    Ok((p.trace, p.best, p.proofs, p.nodes, t.elapsed().as_secs_f64() * 1000.0, p.immediate))
}
