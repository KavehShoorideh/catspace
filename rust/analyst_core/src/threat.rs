//! THE THREAT (Kaveh 2026-10-10: "whenever we're protecting something, we should always protect against a threat ... a
//! threat should have a value with it ... one threat is losing a piece, the other is becoming mate -- cornering a king
//! or losing a king ... these generic concepts should be used in place of the very specific ones").
//!
//! A THREAT of side `by`: what `by` wins if the other side does nothing --
//!   MATERIAL  a piece of theirs `by` wins by the static exchange (rules::threats_of), worth that gain
//!   PROMOTION a pawn of `by` that queens next move, worth what the promotion keeps after their captures (the exchange
//!             law) -- material like any other (2026-10-10: Kd1 attacking the e1 knight while e7-e8=Q waits was refused
//!             as a single threat)
//!   TRAPPED   a piece of theirs we do not win where it stands but whose every move lands where we win it, with a
//!             move of ours onto it next (rules::trapped_unattacked: cornering a piece, as mate is cornering the king --
//!             Kaveh 2026-10-10: "the lemma that covers mate is basically covering trapping"), worth its value
//!   MATE      a forced mate by checks within MATE_CHECKS moves of `by` (every check, every reply -- spite blocks
//!             included: Kaveh's Rc1, "the rook lands on the back rank and gives mate", is Rc8+ Bf8 Rxf8#): their king
//!             is lost, worth MATE_GOAL
//! A JOB of a piece: a threat it parries -- lift it off the board and a threat appears (a piece it guards, a mate square
//! it covers, a line it blocks, the king it shields). The planner's piece-specific rules read these two concepts: the
//! single-threat lemma (a move making one threat against a piece that can step away, giving up no job, wins nothing),
//! overload (a piece with two jobs), the quiet test of the line-pair lemma.
use super::*;
use crate::waiting::MATE_GOAL;
use shakmaty::Bitboard;

#[derive(Clone, Debug, PartialEq)]
pub struct Threat {
    pub target: Square,
    pub value: i64,
    pub mate: bool,
    pub how: Option<String>,                  // the mating move (uci), for a mate
}

/// THE CHECKING MOVES of the side to move, read off the paths onto their king (Kaveh 2026-10-10: "I looked at the squares
/// the king has ... I reasoned through only the king"): arrivals and discoveries -- the proved path lemma
/// (Ideas.PathLemma.path_lemma: a new attack comes from the arriving piece or through the vacated square) makes the list
/// complete, so no other move is tried. -> (move, the board after it).
pub fn checking_moves(p: &Chess) -> Vec<(Move, Chess)> {
    let Some(k) = p.board().king_of(!p.turn()) else { return vec![] };
    let mut cand: Vec<Move> = Vec::new();
    let legal = p.legal_moves();
    for a in crate::paths::attack_paths(p, p.turn(), Bitboard::from_square(k)) {
        match a.kind {
            "direct" => for m in legal.iter().filter(|m| Some(m.to_uci(CastlingMode::Standard).to_string()) == a.mv) { if !cand.contains(m) { cand.push(m.clone()); } },
            "discovery" => for m in legal.iter().filter(|m| m.from() == a.via) { if !cand.contains(m) { cand.push(m.clone()); } },
            _ => {}
        }
    }
    // (castling, en passant and promotions are rare checkers: kept exact by trying the few legal ones)
    for m in legal.iter().filter(|m| matches!(m, Move::Castle { .. } | Move::EnPassant { .. }) || m.is_promotion()) { if !cand.contains(m) { cand.push(m.clone()); } }
    cand.into_iter().filter_map(|m| { let q = after(p, &m); q.is_check().then_some((m, q)) }).collect()
}

/// How many of the attacker's moves a mate threat may take, all of them checks.
pub const MATE_CHECKS: u32 = 3;

/// A FORCED MATE BY CHECKS for the side to move within `n` of its moves -- cornering the king: every move a check,
/// every reply searched, until no reply is left. -> its first move. (n = 1: a mate in one.)
pub fn mate_by_checks(p: &Chess, n: u32) -> Option<Move> {
    let checks = checking_moves(p);
    if let Some((m, _)) = checks.iter().find(|(_, q)| q.legal_moves().is_empty()) { return Some(m.clone()); }
    if n <= 1 { return None; }
    checks.into_iter().find(|(_, q)| q.legal_moves().iter().all(|r| mate_by_checks(&after(q, r), n - 1).is_some())).map(|(m, _)| m)
}

/// A mate in one for the side to move, if any.
pub fn mate_in_one(p: &Chess) -> Option<Move> { mate_by_checks(p, 1) }

/// Every threat of `by` in `pos` (`by` need not be to move: when not, the other side passes -- impossible when `by` is
/// giving check, then only the material is read).
pub fn threats(pos: &Chess, by: Color) -> Vec<Threat> {
    let mut v: Vec<Threat> = rules::threats_of(pos, by).into_iter().filter(|(_, g)| *g > 0)
        .map(|(x, g)| Threat { target: Square::new(x as u32), value: g, mate: false, how: None }).collect();
    let p = if pos.turn() == by { Some(pos.clone()) } else { pass(pos) };
    if let Some(p) = &p {
        for (x, val) in rules::trapped_unattacked(p, by) {
            let sq = Square::new(x as u32);
            let reach = crate::paths::attack_paths(p, by, Bitboard::from_square(sq)).iter().any(|a| a.kind == "direct" || a.kind == "discovery");
            if reach && v.iter().all(|t| t.target != sq) {
                v.push(Threat { target: sq, value: val, mate: false, how: None });
            }
        }
        for m in p.legal_moves() {
            if m.promotion() != Some(Role::Queen) { continue; }
            let q = after(p, &m);
            let bud = BUDGET.load(Ordering::Relaxed);
            BUDGET.store(200_000, Ordering::Relaxed);
            let lost = settle_d(&q, -2000, 2000, false, SETTLE_DEPTH).max(0);
            BUDGET.store(bud, Ordering::Relaxed);
            let gain = kernel_material(&q, by) - kernel_material(p, by) - lost;
            if gain > 0 && v.iter().all(|t| t.target != m.to() || t.value < gain) {
                v.retain(|t| t.target != m.to());
                v.push(Threat { target: m.to(), value: gain, mate: false, how: Some(m.to_uci(CastlingMode::Standard).to_string()) });
            }
        }
    }
    if let Some(p) = p {
        if let (Some(m), Some(k)) = (mate_by_checks(&p, MATE_CHECKS), p.board().king_of(!by)) {
            v.push(Threat { target: k, value: MATE_GOAL, mate: true, how: Some(m.to_uci(CastlingMode::Standard).to_string()) });
        }
    }
    // (ties broken by square: threats_of is a HashMap, its order differs per process -- 2026-10-10: two identical runs
    // decided 902 vs 904 of the same positions)
    v.sort_by(|a, b| b.value.cmp(&a.value).then(a.target.cmp(&b.target)).then(a.mate.cmp(&b.mate)));
    v
}

/// THE FORCING THREATS (Kaveh 2026-10-10, on the single-threat lemma's first surprise: "the king has only two squares
/// ... if I give the king a check, the king has one square to go to ... then my queen can swing over"): beyond the
/// static threats, what `by` wins NEXT MOVE by force --
///   BY CHECKS  a sequence of checks (every reply searched, up to GAIN_CHECKS of ours) ending where `by` has won material
///              (the captures along it, plus the best static capture at its end): the cornered king generalized from
///              mate to material -- Bf7+ Kd8 Nxc6+ bxc6 Qxc6
///   A FORK     a quiet move after which `by` wins two pieces by the static count (worth the smaller), the moving piece
///              not simply lost -- Nf7 against the queen and the h8 rook
/// Used where a claim of "nothing left" needs it (the single-threat lemma's escapes); the cheap static set elsewhere.
pub const GAIN_CHECKS: u32 = 2;

fn material_edge(p: &Chess, by: Color) -> i64 { kernel_material(p, by) - kernel_material(p, !by) }

fn best_capture(p: &Chess, by: Color) -> i64 { rules::threats_of(p, by).values().copied().max().unwrap_or(0).max(0) }

/// What the side to move wins by checks within `n` of its moves (beyond the material edge now): -> (first check, gain).
pub fn gain_by_checks(p: &Chess, n: u32) -> Option<(Move, i64)> {
    let by = p.turn();
    let base = material_edge(p, by);
    fn val(p: &Chess, by: Color, base: i64, n: u32) -> i64 {
        // `by` to move: the best of stopping here (the best capture now) and checking on
        let here = material_edge(p, by) - base + best_capture(p, by);
        if n == 0 { return here; }
        let mut best = here;
        for (_, q) in checking_moves(p) {
            let rs = q.legal_moves();
            if rs.is_empty() { return MATE_GOAL; }
            let v = rs.iter().map(|r| val(&after(&q, r), by, base, n - 1)).min().unwrap_or(0);
            best = best.max(v);
        }
        best
    }
    let here = best_capture(p, by);
    let mut out: Option<(Move, i64)> = None;
    for (m, q) in checking_moves(p) {
        let rs = q.legal_moves();
        let v = if rs.is_empty() { MATE_GOAL } else { rs.iter().map(|r| val(&after(&q, r), by, base, n - 1)).min().unwrap_or(0) };
        if v > here && out.as_ref().map_or(true, |o| v > o.1) { out = Some((m, v)); }
    }
    out
}

/// A fork for the side to move: a quiet move after which it wins two pieces (static count, beyond what it wins now),
/// the moving piece not lost for more than the smaller -> (the move, the smaller gain net of the moving piece's risk).
pub fn fork_next(p: &Chess) -> Option<(Move, i64)> {
    let by = p.turn();
    let now = rules::threats_of(p, by);
    // candidates: a move that arrives on a path onto two or more of their pieces (worth 3+), read off the paths
    let targets: Bitboard = p.board().by_color(!by) & !p.board().by_role(Role::King);
    let mut hits: HashMap<String, Bitboard> = HashMap::new();
    for a in crate::paths::attack_paths(p, by, targets) {
        if a.kind == "direct" { if let Some(u) = &a.mv { hits.entry(u.clone()).or_default().add(a.target); } }
    }
    let mut out: Option<(Move, i64)> = None;
    for m in p.legal_moves() {
        if m.is_capture() || m.is_promotion() { continue; }
        if hits.get(&m.to_uci(CastlingMode::Standard).to_string()).map_or(true, |h| h.count() < 2) { continue; }
        let q = after(p, &m);
        if q.is_check() { continue; }
        let mut new: Vec<i64> = rules::threats_of(&q, by).iter().filter(|(x, g)| **g > 0 && now.get(x).map_or(true, |g0| **g > *g0)).map(|(_, g)| *g).collect();
        if new.len() < 2 { continue; }
        new.sort_by(|a, b| b.cmp(a));
        let risk = rules::threats_of(&q, !by).get(&(m.to() as u8)).copied().unwrap_or(0).max(0);
        let v = new[1] - risk;
        if v > 0 && out.as_ref().map_or(true, |o| v > o.1) { out = Some((m, v)); }
    }
    out
}

/// The static threats of `by` plus the forcing ones (by checks, a fork) as threats on their king's square (by checks)
/// or the forking move's square.
pub fn forcing_threats(pos: &Chess, by: Color) -> Vec<Threat> {
    let mut v = threats(pos, by);
    let p = if pos.turn() == by { Some(pos.clone()) } else { pass(pos) };
    if let Some(p) = p {
        if let (Some((m, g)), Some(k)) = (gain_by_checks(&p, GAIN_CHECKS), p.board().king_of(!by)) {
            if g < MATE_GOAL && v.iter().all(|t| !(t.target == k && t.value >= g)) {
                v.push(Threat { target: k, value: g, mate: false, how: Some(m.to_uci(CastlingMode::Standard).to_string()) });
            }
        }
        if let Some((m, g)) = fork_next(&p) {
            v.push(Threat { target: m.to(), value: g, mate: false, how: Some(m.to_uci(CastlingMode::Standard).to_string()) });
        }
    }
    v
}

/// WHAT THE SIDE TO MOVE CAN FORCE (Kaveh 2026-10-10: "why does there need to be a numeric check? ... the more principled
/// version"): a proof search -- can it force a material gain of at least 1? Its moves are FORCING ones -- checks,
/// captures, and quiet moves that make a new threat worth 3 or more (read off the paths onto their pieces) -- each reply
/// of theirs searched; it may stop wherever it already stands better (the material won plus its best capture now). The
/// search ends where the forcing ends (no forcing move left) or a position repeats; the budget is only a safety net:
/// hitting it is UNKNOWN, never "nothing". (It replaces "material by checks within 2", which read its limit as "no
/// threat" and refused Be1 -- two threats: the c3 knight and Ba5+'s chain of rook checks, through a quiet c6.)
/// -> Some(true) a forced gain, Some(false) certainly none, None unknown.
pub const FORCING_BUDGET: u64 = 200;
/// (a second safety net, for the stack: a forcing line this long is UNKNOWN, like the budget -- 2026-10-10: unbounded
/// recursion overflowed the stack)
pub const FORCING_DEPTH: u32 = 24;

pub fn forcing_gain(p: &Chess, budget: u64) -> Option<bool> {
    let by = p.turn();
    let base = material_edge(p, by);
    let mut nodes = 0u64;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let key = |q: &Chess| shakmaty::fen::Fen::from_position(q.clone(), shakmaty::EnPassantMode::Legal).to_string();
    // the forcing moves of `by` (to move in q)
    fn forcing(q: &Chess, by: Color) -> Vec<(Move, Chess)> {
        let now = rules::threats_of(q, by);
        let targets: Bitboard = q.board().by_color(!by) & !q.board().by_role(Role::Pawn) & !q.board().by_role(Role::King);
        let paths = crate::paths::attack_paths(q, by, targets);
        let arriving: std::collections::HashSet<String> = paths.iter().filter(|a| a.kind == "direct").filter_map(|a| a.mv.clone()).collect();
        let disc_from: Vec<Square> = paths.iter().filter(|a| a.kind == "discovery").filter_map(|a| a.via).collect();
        let mut out = Vec::new();
        for m in q.legal_moves() {
            let r = after(q, &m);
            if r.is_check() || m.is_capture() || m.is_promotion() { out.push((m, r)); continue; }
            let u = m.to_uci(CastlingMode::Standard).to_string();
            let disc = m.from().map_or(false, |f| disc_from.contains(&f));
            if !arriving.contains(&u) && !disc { continue; }
            let new_big = rules::threats_of(&r, by).iter().any(|(x, g)| *g >= 3 && now.get(x).map_or(true, |g0| g > g0));
            if new_big { out.push((m, r)); }
        }
        out
    }
    // us to move in q: can we force >= 1 from here? (at the root only by a forcing move -- what stands already is the
    // static threats' business)
    fn ours(q: &Chess, by: Color, base: i64, root: bool, depth: u32, nodes: &mut u64, budget: u64, seen: &mut std::collections::HashSet<String>, key: &dyn Fn(&Chess) -> String) -> Option<bool> {
        *nodes += 1;
        if *nodes > budget || depth > FORCING_DEPTH { return None; }
        if !root && material_edge(q, by) - base + best_capture(q, by) >= 1 { return Some(true); }
        let k = key(q);
        if !seen.insert(k.clone()) { return Some(false); }        // a repetition gains nothing
        let mut unknown = false;
        for (_, r) in forcing(q, by) {
            match theirs(&r, by, base, depth + 1, nodes, budget, seen, key) { Some(true) => { seen.remove(&k); return Some(true); }, None => unknown = true, _ => {} }
        }
        seen.remove(&k);
        if unknown { None } else { Some(false) }
    }
    // them to move in r: does every reply leave us a forced gain?
    fn theirs(r: &Chess, by: Color, base: i64, depth: u32, nodes: &mut u64, budget: u64, seen: &mut std::collections::HashSet<String>, key: &dyn Fn(&Chess) -> String) -> Option<bool> {
        *nodes += 1;
        if *nodes > budget || depth > FORCING_DEPTH { return None; }
        let replies = r.legal_moves();
        if replies.is_empty() { return Some(r.is_check()); }      // mate is a gain; stalemate is not
        let mut unknown = false;
        for m in replies {
            match ours(&after(r, &m), by, base, false, depth + 1, nodes, budget, seen, key) { Some(false) => return Some(false), None => unknown = true, _ => {} }
        }
        if unknown { None } else { Some(true) }
    }
    ours(p, by, base, true, 0, &mut nodes, budget, &mut seen, &key)
}

/// IS THERE ANY NEW THREAT of `by` (static or forcing) against `before`? -- the cheapest first, stopping at the first
/// one found (material, promotion, trapped piece, mate by checks, gain by checks, a fork): a claim of "nothing left"
/// pays for everything, a refutation of it for the first fact that settles it.
/// -> Some(true) a new threat, Some(false) certainly none, None unknown (the forcing search's safety net was hit).
pub fn any_new_threat(pos: &Chess, by: Color, before: &[Threat]) -> Option<bool> {
    let is_new = |t: &Threat| before.iter().find(|b| b.target == t.target && b.mate == t.mate).map_or(true, |b| t.value > b.value);
    for (x, g) in rules::threats_of(pos, by) {
        if g > 0 && is_new(&Threat { target: Square::new(x as u32), value: g, mate: false, how: None }) { return Some(true); }
    }
    let Some(p) = (if pos.turn() == by { Some(pos.clone()) } else { pass(pos) }) else { return None };
    // promotion and trapped pieces, then the king: the full static set (cheap parts already settled above)
    let st = threats(&p, by);
    if st.iter().any(|t| is_new(t)) { return Some(true); }
    if let Some((m, g)) = fork_next(&p) { if is_new(&Threat { target: m.to(), value: g, mate: false, how: None }) { return Some(true); } }
    // what it can FORCE, to the end of the forcing (no count of checks)
    forcing_gain(&p, FORCING_BUDGET)
}

/// The threats in `now` that are NEW against `before`: a target not threatened before, or threatened for more.
pub fn new_threats<'a>(now: &'a [Threat], before: &[Threat]) -> Vec<&'a Threat> {
    now.iter().filter(|t| before.iter().find(|b| b.target == t.target && b.mate == t.mate).map_or(true, |b| t.value > b.value)).collect()
}

/// A board with the piece on `sq` lifted off it (the side to move unchanged), if it is a legal setup to read.
pub fn lifted(pos: &Chess, sq: Square) -> Option<Chess> {
    let mut st = pos.clone().into_setup(shakmaty::EnPassantMode::Legal);
    st.board.discard_piece_at(sq);
    st.ep_square = None;
    Chess::from_setup(st, CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights())
        .or_else(|e| e.ignore_impossible_check()).ok()
}

/// THE JOBS of the piece on `sq` (whoever's): the threats of its opponent that appear when it is lifted -- on another
/// target, or for more than now. Lifting a piece that shields its own king from a check counts as a job worth a mate.
pub fn jobs(pos: &Chess, sq: Square) -> Vec<Threat> {
    let Some(pc) = pos.board().piece_at(sq) else { return vec![] };
    jobs_with(pos, sq, &threats(pos, !pc.color))
}

/// `jobs` with the attacker's threats now (`now`) already computed.
pub fn jobs_with(pos: &Chess, sq: Square, now: &[Threat]) -> Vec<Threat> {
    let Some(pc) = pos.board().piece_at(sq) else { return vec![] };
    let atk = !pc.color;
    let Some(l) = lifted(pos, sq) else { return vec![] };
    let king = l.board().king_of(pc.color);
    if let Some(k) = king {
        if l.board().attacks_to(k, atk, l.board().occupied()).any() && !pos.board().attacks_to(k, atk, pos.board().occupied()).any() {
            return vec![Threat { target: k, value: MATE_GOAL, mate: true, how: None }];
        }
    }
    let after_lift = threats(&l, atk);
    new_threats(&after_lift, now).into_iter().filter(|t| t.target != sq).cloned().collect()
}

/// threat_check(fen) -> (the side to move's threats, the opponent's threats, and per piece of each side its jobs):
/// the generic threat's verification.
#[pyfunction]
pub fn threat_check(fen: String) -> PyResult<(Vec<(String, i64, bool)>, Vec<(String, i64, bool)>, Vec<(String, Vec<(String, i64, bool)>)>)> {
    let Some(pos) = parse(&fen) else { return Ok((vec![], vec![], vec![])) };
    let f = |v: Vec<Threat>| v.into_iter().map(|t| (t.target.to_string(), t.value, t.mate)).collect::<Vec<_>>();
    let us = pos.turn();
    let mut js = Vec::new();
    for sq in pos.board().occupied() {
        if pos.board().piece_at(sq).map_or(false, |p| p.role == Role::King) { continue; }
        let j = jobs(&pos, sq);
        if !j.is_empty() { js.push((sq.to_string(), f(j))); }
    }
    Ok((f(threats(&pos, us)), f(threats(&pos, !us)), js))
}

/// checks_audit(fen) -> (checking moves read off the paths, checking moves by brute force over the legal moves): the
/// completeness check of `checking_moves`.
#[pyfunction]
pub fn checks_audit(fen: String) -> PyResult<(Vec<String>, Vec<String>)> {
    let Some(p) = parse(&fen) else { return Ok((vec![], vec![])) };
    let u = |m: &Move| m.to_uci(CastlingMode::Standard).to_string();
    let mut a: Vec<String> = checking_moves(&p).iter().map(|(m, _)| u(m)).collect();
    let mut b: Vec<String> = p.legal_moves().iter().filter(|m| after(&p, m).is_check()).map(u).collect();
    a.sort(); b.sort();
    Ok((a, b))
}
