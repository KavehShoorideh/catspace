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
use crate::waiting::{escapes, prove_n, touched, MATE_GOAL};
use std::cell::RefCell;

// THE FRAME MEMORY (Kaveh 2026-10-09: "all of these frames ... have two or three outs, and each out is a value -- so we
// should have this value memorized, so that every time we see this frame we know the outcome, unless something; then we
// check the conditions -- that's the only part where we pay extra"). Every proof the planner asks -- a QUESTION: the seed,
// the focus, the goal, the side to move (where in the analysis it was asked does not matter; the board comparison
// decides) -- is remembered with the board it was asked on, its answer, its stored proof
// or refutation, and its SUPPORT (the squares the answer rests on: the squares its proof or refutation moved through,
// the focus and its neighbourhood, the seed's squares). Asked again on a later board:
//   same board                    -> the answer, no search
//   it HELD                       -> REPLAY the stored proof (cheap; sound -- a re-check, every reply searched)
//   it FAILED                     -> re-prove, the stored refutation tried first (reusing a "fails" whose support was
//                                    untouched is an option, OFF: audited unsafe -- see `mprove`)
#[derive(Clone)]
struct Entry { board: Vec<u8>, extra: String, holds: bool, hints: std::sync::Arc<Hints>, support: Bitboard }

#[derive(Default, Clone, Copy)]
pub struct Stats { pub fresh: u32, pub replay: u32, pub reuse_exact: u32, pub reuse_fail: u32, pub reprove: u32,
                   pub nodes: u64, pub lone: u32, pub frozen: u32, pub escape_skip: u32, pub cut_by_history: u32 }

#[derive(Default)]
pub struct Memo { entries: HashMap<String, Vec<Entry>>, pub history: HashMap<String, f32>, pub stats: Stats }
const SNAPSHOTS: usize = 8;
pub static REUSE_FAILS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_reuse_fails(flag: bool) -> PyResult<()> { REUSE_FAILS.store(flag, Ordering::Relaxed); Ok(()) }                                               // boards kept per question (the oldest dropped)

thread_local! {
    static MEMO: RefCell<Option<Memo>> = const { RefCell::new(None) };
    static TAG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

fn snapshot(p: &Chess) -> (Vec<u8>, String) {
    let b = p.board();
    let v = (0..64u32).map(|i| b.piece_at(Square::new(i)).map_or(0u8, |pc| 1 + pc.role as u8 + if pc.color == Color::White { 0 } else { 8 })).collect();
    let fen = shakmaty::fen::Fen::from_position(p.clone(), shakmaty::EnPassantMode::Legal).to_string();
    let parts: Vec<&str> = fen.split(' ').collect();
    (v, format!("{} {} {}", parts.get(1).unwrap_or(&""), parts.get(2).unwrap_or(&""), parts.get(3).unwrap_or(&"")))
}

fn with_tag<T>(tag: String, f: impl FnOnce() -> T) -> T {
    TAG.with(|t| t.borrow_mut().push(tag));
    let out = f();
    TAG.with(|t| { t.borrow_mut().pop(); });
    out
}

fn tally(f: impl FnOnce(&mut Stats)) { MEMO.with(|m| if let Some(m) = m.borrow_mut().as_mut() { f(&mut m.stats) }); }

/// A proof question through the memory (or straight to the search when no memory is installed): (holds, nodes searched).
fn mprove(p: &Chess, seed: &str, focus: &[Square], goal: i64) -> (bool, u64) {
    let installed = MEMO.with(|m| m.borrow().is_some());
    if !installed { let (ok, _, n, _) = prove_n(p, seed, focus, goal, None); return (ok, n); }
    // the QUESTION alone is the key -- where it was asked does not matter, the board comparison decides (2026-10-09: keyed
    // by where it was asked, White's null-move analysis of Black's threat never served Black's own analysis a move later)
    let key = format!("{}|{}|{}|{}|{}", seed, focus.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","), goal,
                      if p.turn() == Color::White { "w" } else { "b" }, crate::waiting::DEEP_CONDITIONS.load(Ordering::Relaxed));
    let (board, extra) = snapshot(p);
    let prior = MEMO.with(|m| m.borrow().as_ref().and_then(|m| m.entries.get(&key).and_then(|v| {
        v.iter().filter(|e| e.extra == extra)
            .min_by_key(|e| (0..64usize).filter(|i| e.board[*i] != board[*i]).count()).cloned()
            .or_else(|| v.last().cloned())
    })));
    let mut hints = None;
    if let Some(e) = prior {
        if e.extra == extra {
            let diff: Bitboard = (0..64u32).filter(|i| e.board[*i as usize] != board[*i as usize]).map(Square::new).collect();
            if diff.is_empty() { tally(|s| s.reuse_exact += 1); return (e.holds, 0); }
            // (OFF by default -- the audit of 2026-10-09: reusing a "fails" whose support no move touched missed two
            // immediate mates and misread their threat in several positions on 60 games: a refutation depends on more
            // than the squares it moved through, e.g. a piece guarding the landing square from afar)
            if REUSE_FAILS.load(Ordering::Relaxed) && !e.holds && (diff & e.support).is_empty() { tally(|s| s.reuse_fail += 1); return (false, 0); }
            if e.holds { tally(|s| s.replay += 1); } else { tally(|s| s.reprove += 1); }
        } else {
            tally(|s| s.reprove += 1);
        }
        hints = Some(e.hints.clone());
    } else {
        tally(|s| s.fresh += 1);
    }
    let (ok, rec, n, _) = prove_n(p, seed, focus, goal, hints);
    let mut support = touched(&rec);
    for f in focus { support.add(*f); support |= shakmaty::attacks::king_attacks(*f); }
    if let Some(m) = seed.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(p).ok()) {
        if let Some(fr) = m.from() { support.add(fr); }
        support.add(m.to());
    }
    MEMO.with(|m| if let Some(m) = m.borrow_mut().as_mut() {
        m.stats.nodes += n;
        let v = m.entries.entry(key).or_default();
        v.retain(|e| e.board != board || e.extra != extra);
        v.push(Entry { board, extra, holds: ok, hints: std::sync::Arc::new(rec), support });
        if v.len() > SNAPSHOTS { v.remove(0); }
    });
    (ok, n)
}

pub struct Plan { pub trace: Vec<String>, pub best: Option<(String, String, i64, String)>, pub proofs: u32, pub nodes: u64, pub target: Option<Square>,
                 pub immediate: Option<(String, String, i64, String)>, pub wins: Vec<(String, Square, i64)> }

fn val(r: Role) -> i64 { if r == Role::King { 1000 } else { kernel_value(r) } }

fn san(pos: &Chess, uci: &str) -> String {
    uci.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok())
        .map(|m| shakmaty::san::San::from_move(pos, &m).to_string()).unwrap_or_else(|| uci.to_string())
}

/// WHAT OUR MOVE EXPOSES (the vacated-square clause of the proved path lemma, Ideas.PathLemma.path_lemma, read for THEIR
/// attacks): our move m newly exposes a piece of ours that one of their sliders now sees through the square m vacated, or
/// leaves the moved piece itself hanging -> [(square, their static gain on it)] with a gain (Kaveh 2026-10-09: "the
/// running away of the bishop exposes the knight").
fn exposes(pos: &Chess, m: &Move) -> Vec<(Square, i64)> {
    let us = pos.turn();
    let them = !us;
    let q = after(pos, m);
    let gains = rules::threats_of(&q, them);
    let b0 = pos.board();
    let occ0 = b0.occupied();
    let vac: Vec<Square> = match m { Move::Castle { king, rook } => vec![*king, *rook], _ => m.from().into_iter().collect() };
    let mut out = Vec::new();
    for x in q.board().by_color(us) {
        let Some(&g) = gains.get(&(x as u8)) else { continue };
        if g <= 0 { continue; }
        let moved_here = x == m.to() || matches!(m, Move::Castle { .. });
        let newly = q.board().attacks_to(x, them, q.board().occupied()).into_iter().any(|a| {
            !b0.attacks_to(x, them, occ0).contains(a) && vac.iter().any(|v| shakmaty::attacks::between(a, x).contains(*v))
        });
        if newly || (moved_here && x == m.to()) { out.push((x, g)); }
    }
    out
}

/// THE LINE PAIR (Kaveh 2026-10-09): their slider `sl` attacks our front piece `f` and our piece `r` stands behind it on
/// the line. Can either protect the other from a new square? (Read off the legal moves -- for the knight and the bishop
/// the proved colour lemmas say why not.) -> an explanation when neither can: one of them is lost unless the front piece
/// is protected where it stands, the slider is taken or blocked, or a bigger threat comes first.
fn line_pair(pos: &Chess, sl: Square, f: Square) -> Option<String> {
    let b = pos.board();
    let us = pos.turn();
    let occ = b.occupied();
    let beyond = shakmaty::attacks::ray(sl, f) & !shakmaty::attacks::between(sl, f) & !Bitboard::from_square(sl) & !Bitboard::from_square(f);
    let r = beyond.into_iter().filter(|x| occ.contains(*x)).min_by_key(|x| shakmaty::attacks::between(f, *x).count())?;
    if b.piece_at(r)?.color != us { return None; }
    let protects = |from: Square, to: Square| pos.legal_moves().iter().any(|m| m.from() == Some(from) && {
        let q = after(pos, m);
        q.board().attacks_to(to, us, q.board().occupied()).contains(m.to())
    });
    if protects(f, r) || protects(r, f) { return None; }
    let (pf, pr) = (b.piece_at(f)?.role, b.piece_at(r)?.role);
    let orth = (f.file() as i32 - r.file() as i32).abs() + (f.rank() as i32 - r.rank() as i32).abs() == 1;
    let why = |p: Role, other: Square| -> String {
        match p {
            Role::Bishop if orth => format!(" (a bishop keeps its colour; {other} is the other colour)"),
            Role::Knight if orth => format!(" (a knight that hops never covers an orthogonal neighbour of where it stood)"),
            _ => String::new(),
        }
    };
    Some(format!("line pair: our {:?} on {f} and {:?} on {r} under their {:?} on {sl} -- the {:?} cannot protect {r} from a new square{}, nor the {:?} protect {f}{}: moving the front piece exposes the back one; one is lost unless {f} is protected where it stands, the {:?} is taken or blocked, or a bigger threat comes first",
        pf, pr, b.piece_at(sl).map(|x| x.role)?, pf, why(pf, r), pr, why(pr, f), b.piece_at(sl).map(|x| x.role)?).to_lowercase())
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
    let mut wins: Vec<(String, Square, i64)> = Vec::new();               // every immediate win found (for their threats)
    // 1 THEIR THREAT: their resolved captures if we pass, AND their immediate wins found by the same reasoning on the
    // null move (2026-10-09, after O-O-O in the castling line: Black's planner said "nothing" -- the threat counted only
    // White's captures, not White's mate)
    let mut their: Option<(String, Square, i64)> = None;
    let mut their_wins: Vec<(String, Square, i64)> = Vec::new();
    let threat = if immediate_only { 0 } else {
        let cap = match pass(pos) {
            Some(p) => { BUDGET.store(200_000, Ordering::Relaxed); let v = settle_d(&p, -2000, 2000, true, SETTLE_DEPTH).max(0); BUDGET.store(u64::MAX, Ordering::Relaxed); v }
            None => 0,
        };
        if let Some(p) = pass(pos) {
            let tp = with_tag("null".to_string(), || plan_inner(&p, true));
            proofs += tp.proofs; nodes += tp.nodes;
            their_wins = tp.wins.clone();
            if let (Some(bb), Some(tt)) = (tp.best, tp.target) {
                trace.push(format!("their best immediate win if we pass: {} ({} proofs)", bb.3, tp.proofs));
                their = Some((bb.1, tt, bb.2));
            }
        }
        cap.max(their.as_ref().map_or(0, |x| x.2))
    };
    let tempo = threat <= 1;
    // PINNED TO WHAT (Kaveh 2026-10-09: "the knight on d7 is pinned because the queen can give a mate on d8 if the knight
    // hops"): for each pin of theirs against our king zone (paths.rs), ONE analysis: lift our blocker and run their
    // immediate reasoning; if they then win more than they threaten now, the blocker is pinned to that loss, and its
    // moves that leave the line are refused without proof -- for a knight every move, by the proved lemma
    // Ideas.GeoLemmas.knight_leaves_every_line (taking the pinning piece, and a check, are still proved)
    let mut frozen: HashMap<Square, (Square, Square, i64, String)> = HashMap::new();   // blocker -> (slider, target, loss, why)
    if !immediate_only {
        if let Some(k) = b.king_of(us) {
            let zone = shakmaty::attacks::king_attacks(k).with(k);
            for p in attack_paths(pos, them, zone) {
                if p.kind != "pin" { continue; }
                let Some(bl) = p.via else { continue };
                if frozen.contains_key(&bl) { continue; }
                let mut setup = pos.clone().into_setup(shakmaty::EnPassantMode::Legal);
                setup.board.discard_piece_at(bl);
                setup.turn = them;
                setup.ep_square = None;                                  // (review: a kept ep square made the setup invalid)
                let lifted = Chess::from_setup(setup, CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights()).ok();
                let Some(lifted) = lifted else { continue };
                let lp = with_tag(format!("lift:{bl}"), || plan_inner(&lifted, true));
                proofs += lp.proofs; nodes += lp.nodes;
                if let Some(lb) = lp.best {
                    if lb.2 > threat.max(1) {
                        let role = b.piece_at(bl).map(|x| x.role);
                        trace.push(format!("pin: our {:?} on {bl} stands between their {:?} on {} and {}; lifted, they have {} -- {}",
                            role.unwrap_or(Role::Pawn), b.piece_at(p.piece).map(|x| x.role).unwrap_or(Role::Pawn), p.piece, p.target, lb.3,
                            if role == Some(Role::Knight) { "every knight move opens the line (knight_leaves_every_line): all frozen".to_string() } else { "moves off the line frozen".to_string() }));
                        frozen.insert(bl, (p.piece, p.target, lb.2, format!("{}", lb.3)));
                    }
                }
            }
        }
    }
    let refused = |m: &Move| -> Option<String> {
        let from = m.from()?;
        let (sl, tgt, loss, why) = frozen.get(&from)?;
        if m.to() == *sl { return None; }                                // taking the pinning piece
        if after(pos, m).is_check() { return None; }                       // a check is still proved
        if m.capture().map_or(false, |r| kernel_value(r) >= *loss) { return None; }   // a capture worth the loss: proved
        // still blocking: the move stays between the slider and the pinned-to square (review: the line to the KING was
        // used even when the pin points at another square, d8)
        let on_line = b.piece_at(from).map_or(false, |x| x.role != Role::Knight)
            && shakmaty::attacks::between(*sl, *tgt).contains(m.to());
        if on_line { return None; }
        Some(format!("pinned: it would allow {why} (worth {loss})"))
    };
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
        // HISTORY ORDER (Kaveh: "order the lemma check by the most recently used"): the moves that won most recently
        // first; once one wins the target outright, the rest of its moves are not proved
        let hist = MEMO.with(|m| m.borrow().as_ref().map(|m| m.history.clone()).unwrap_or_default());
        let side = if us == Color::White { "w" } else { "b" };
        let hk = |u: &String| format!("{side}:{u}");                      // (review: both sides' moves shared one history)
        seeds.sort_by(|a, b| hist.get(&hk(b)).unwrap_or(&0.0).partial_cmp(hist.get(&hk(a)).unwrap_or(&0.0)).unwrap());
        let mut won_outright = false;
        for s in &seeds {
            // (the cut only for our own moves: the opponent's threat collects EVERY win -- 2026-10-09: two mating threats,
            // each planner found a different one first and defended against only it)
            if won_outright && !immediate_only { tally(|st| st.cut_by_history += 1); continue; }
            if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                if let Some(why) = refused(&sm) {
                    trace.push(format!("    {}: refused without proof -- {why}", san(pos, s)));
                    tally(|st| st.frozen += 1);
                    continue;
                }
            }
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
                        tally(|st| st.lone += 1);
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
            let (ok, n) = with_tag("seed".to_string(), || mprove(pos, s, &[t], g));
            proofs += 1; nodes += n;
            trace.push(format!("    prove {} (gain {g}): {} ({n} nodes)", san(pos, s), if ok { "HOLDS" } else { "fails" }));
            if ok {
                wins.push((s.clone(), t, g));
                MEMO.with(|m| if let Some(m) = m.borrow_mut().as_mut() { *m.history.entry(hk(s)).or_insert(0.0) += 1.0; });
                if king { won_outright = true; }                         // (review: a fork's second target may be worth more than this target -- only a mate cannot be beaten)
            }
            if ok { if best.as_ref().map_or(true, |x| g > x.2) { best = Some((String::new(), s.clone(), g, format!("{} wins {g} against the {name}", san(pos, s)))); best_target = Some(t); } }
            else { failed.push(s.clone()); }
        }
        armed.push((goal, t, failed));
    }
    if immediate_only { let im = best.clone(); return Plan { trace, best, proofs, nodes, target: best_target, immediate: im, wins }; }
    // 3b DEFENCE: their immediate win is bigger than ours -- the candidate defences are read off their winning move's
    // geometry (the path lemma): take the attacking piece, move the target, guard the square it lands on, step onto a
    // line of theirs into the target's zone, free an escape next to the target, or check; one proof each: does their
    // move still win after ours?
    let ours_now = best.as_ref().map_or(0, |x| x.2);
    let immediate = best.clone();
    if let Some((tseed, tt, tg)) = their.clone() {
        // DOES OUR ATTACK ALREADY ANSWER THEIRS? (Kaveh 2026-10-09; the census: defending first lost 6% of the immediate
        // wins) -- one proof: their winning move after our best one
        // A CERTIFIED WIN ALREADY ANSWERS A MATERIAL THREAT: its proof searched every reply of theirs, their captures
        // included, and resolved the captures left at each leaf -- so their capture threat is inside its gain. Only a
        // MATE threat lies outside it (the leaf counts captures, not mates) and gets the check below. (2026-10-09:
        // re-proving their material threat after our win made the planner defend in 71 positions; Stockfish preferred
        // the win in 51.)
        let answered = match &immediate {
            Some(_) if tg < MATE_GOAL => true,
            Some(x) => {
                let q = x.1.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()).map(|m| after(pos, &m));
                match q {
                    Some(q) if q.is_checkmate() => true,
                    Some(q) => {
                        // every winning move of theirs worth as much must be stopped (2026-10-09 audit), and our move must
                        // expose nothing worth as much
                        let threats: Vec<(String, Square, i64)> = if their_wins.is_empty() { vec![(tseed.clone(), tt, tg)] }
                            else { their_wins.iter().filter(|w| w.2 >= tg).cloned().collect() };
                        let bm = x.1.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok());
                        // (the square our capturing piece lands on is left out: its recapture is already in our gain --
                        // 2026-10-09: "Nxd1 wins 2" was refused because the knight on d1 could be taken back)
                        let mut stops = bm.as_ref().map_or(true, |bm| exposes(pos, bm).iter().all(|(x, g)| *g < tg || *x == bm.to()));
                        if stops {
                            for (ws, wt, wg) in &threats {
                                let legal = ws.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&q).ok()).is_some();
                                // (review: their move illegal only because our move gives CHECK is not answered -- the
                                // threat returns after they answer the check)
                                if !legal { if q.is_check() { stops = false; break; } continue; }
                                let t2 = match &bm { Some(bm) if bm.from() == Some(*wt) => bm.to(), _ => *wt };
                                let (ok, n) = with_tag(format!("ans:{}", x.1), || mprove(&q, ws, &[t2], *wg)); proofs += 1; nodes += n;
                                if ok { stops = false; break; }
                            }
                        }
                        stops
                    }
                    None => false,
                }
            }
            None => false,
        };
        if answered { trace.push(format!("our {} also stops their threat", best.as_ref().unwrap().3)); }
        // TAKE OR DEFEND (2026-10-09, checked against Stockfish: of 99 positions where the planner defended instead of
        // taking an immediate win, the win was better in 81): if our win does not stop their threat, both happen -- we
        // defend only when what we win is worth less than what they win (net negative)
        let trade_ok = immediate.is_some() && ours_now >= tg;
        if !answered && trade_ok && immediate.is_some() {
            trace.push(format!("our {} is worth at least their threat ({tg}): take it", best.as_ref().unwrap().3));
        }
        if tg >= 2 && tg > ours_now && !answered && !trade_ok {
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
                    if refused(&m).is_some() { continue; }
                    let q = after(pos, &m);
                    let hit = Some(m.to()) == from || m.from() == Some(tt) || q.board().attacks_to(land, us, q.board().occupied()).any()
                        && !b.attacks_to(land, us, b.occupied()).any()
                        || lines.contains(m.to()) || m.from().map_or(false, |f| shakmaty::attacks::king_attacks(tt).contains(f)) || q.is_check();
                    if hit { cands.push(m); }
                }
                trace.push(format!("their {} wins {tg}: {} candidate defences on its geometry", san(&p0, &tseed), cands.len()));
                if let Some(slider) = sm.from() { if let Some(lp) = line_pair(pos, slider, tt) { trace.push(format!("    {lp}")); } }
                for m in cands {
                    let q = after(pos, &m);
                    let mu = m.to_uci(CastlingMode::Standard).to_string();
                    // a defence must stop EVERY winning move of theirs worth as much -- and must not EXPOSE something worth
                    // as much (the vacated-square clause: the bishop leaving d6 opens the rook's line onto the d7 knight);
                    // an exposure is confirmed with one proof of their capture
                    let mut still = false;
                    for (x, g) in exposes(pos, &m) {
                        if g < tg { continue; }
                        let cap = q.legal_moves().into_iter().find(|c| c.to() == x).map(|c| c.to_uci(CastlingMode::Standard).to_string());
                        if let Some(cap) = cap {
                            let (ok, n) = with_tag(format!("exp:{mu}"), || mprove(&q, &cap, &[x], g.min(tg))); proofs += 1; nodes += n;
                            if ok { trace.push(format!("    {}: exposes {x} (their {} wins it)", san(pos, &mu), san(&q, &cap))); still = true; break; }
                        }
                    }
                    let threats: Vec<(String, Square, i64)> = if their_wins.is_empty() { vec![(tseed.clone(), tt, tg)] }
                        else { their_wins.iter().filter(|w| w.2 >= tg).cloned().collect() };
                    for (ws, wt, wg) in &threats {
                        if still { break; }
                        let t2 = if m.from() == Some(*wt) { m.to() } else { *wt };
                        let legal = ws.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&q).ok()).is_some();
                        if !legal { continue; }
                        let (ok, n) = with_tag(format!("def:{mu}"), || mprove(&q, ws, &[t2], *wg)); proofs += 1; nodes += n;
                        if ok { still = true; break; }
                    }
                    if !still {
                        trace.push(format!("    {} stops it", san(pos, &mu)));
                        if best.as_ref().map_or(true, |x| x.0 != "defend") {
                            best = Some(("defend".to_string(), mu.clone(), -tg, format!("{} stops their {}", san(pos, &mu), san(&p0, &tseed))));
                        }
                    }
                }
                if best.as_ref().map_or(true, |x| x.0 != "defend") {
                    trace.push(format!("no defence stops their {}: the loss is {tg} -- what remains is the cheapest way to lose it (not yet chosen)", san(&p0, &tseed)));
                }
                return Plan { trace, best, proofs, nodes, target: best_target, immediate, wins };
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
                if refused(&am).is_some() { continue; }
                let q = after(pos, &am);
                if q.is_check() { continue; }
                let Some(p) = pass(&q) else { continue };
                let Some(sm2) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&p).ok()) else { continue };
                let q2 = after(&p, &sm2);
                let t2 = if sm2.from() == Some(t) { sm2.to() } else { t };
                // two or more escapes left open: no proof (one may still be chased -- Ne7+ Kb8 Rxd8#)
                if escapes(&q2, us, t2).count() >= 2 { tally(|st| st.escape_skip += 1); continue; }
                let (ok, n) = with_tag(format!("arm:{a}"), || mprove(&p, s, &[t], goal));
                proofs += 1; nodes += n;
                trace.push(format!("        prove {} then {} (they pass): {} ({n} nodes)", san(pos, a), san(&p, s), if ok { "HOLDS -- an armed threat" } else { "fails" }));
                if ok && best.as_ref().map_or(true, |x| goal > x.2) {
                    best = Some((a.clone(), s.clone(), goal, format!("{} arms {} against the {name}", san(pos, a), san(&p, s))));
                }
            }
        }
      }
    }
    Plan { trace, best, proofs, nodes, target: best_target, immediate, wins }
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


/// THE PLAN STATE OF A GAME: the frame memory and the move history, kept across moves. `observe(fen)` plans the side to
/// move with the memory installed -> (trace, decision, proofs asked, nodes searched, ms, best immediate win, and this
/// move's (fresh searches, replays, exact reuses, reused failures, re-proofs, lone-attack skips, frozen moves, escape
/// skips, moves cut by the history order)).
#[pyclass(unsendable)]
pub struct PlanState { memo: Option<Memo> }

type StatRow = (u32, u32, u32, u32, u32, u32, u32, u32, u32);

#[pymethods]
impl PlanState {
    #[new]
    fn new() -> Self { PlanState { memo: Some(Memo::default()) } }

    fn observe(&mut self, fen: String) -> PyResult<(Vec<String>, Option<(String, String, i64, String)>, u32, u64, f64, Option<(String, String, i64, String)>, StatRow)> {
        let Some(pos) = parse(&fen) else { return Ok((vec![], None, 0, 0, 0.0, None, (0, 0, 0, 0, 0, 0, 0, 0, 0))) };
        let mut memo = self.memo.take().unwrap_or_default();
        memo.stats = Stats::default();
        for v in memo.history.values_mut() { *v *= 0.8; }                    // recent wins weigh more
        MEMO.with(|m| *m.borrow_mut() = Some(memo));
        let t = std::time::Instant::now();
        let p = plan(&pos);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let memo = MEMO.with(|m| m.borrow_mut().take()).unwrap_or_default();
        let s = memo.stats;
        self.memo = Some(memo);
        Ok((p.trace, p.best, p.proofs, s.nodes, ms, p.immediate,
            (s.fresh, s.replay, s.reuse_exact, s.reuse_fail, s.reprove, s.lone, s.frozen, s.escape_skip, s.cut_by_history)))
    }

    /// Entries in the memory.
    fn size(&self) -> usize { self.memo.as_ref().map_or(0, |m| m.entries.values().map(|v| v.len()).sum()) }
}
