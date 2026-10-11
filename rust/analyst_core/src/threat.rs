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

/// (kernel_material is already ours minus theirs -- 2026-10-11, the bug search: this subtracted it twice, doubling every
/// material edge: a pawn won by checks counted 2, the forcing search's "standing 1 better" mixed doubled material with an
/// undoubled capture)
fn material_edge(p: &Chess, by: Color) -> i64 { kernel_material(p, by) }

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


/// THE LINE PRIOR (Kaveh 2026-10-10: "a prior to decide how far to follow each line ... unity for now; when it is
/// policy-based, a reinforcement-learning agent makes that decision"): the weight on a forcing move's proof number --
/// 1.0 for every move today; the hook a learned policy fills later (AlphaProof / HyperTree-style: it orders the work,
/// the proof still decides).
pub fn line_prior(_q: &Chess, _m: &Move) -> f64 { 1.0 }

/// Counters of the forcing search (pns::STATS): [calls, proven, disproven, unknown, expansions]; the budget settable.
pub static FBUDGET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(FORCING_BUDGET);
#[pyfunction]
pub fn forcing_stats(budget: Option<u64>) -> Vec<u64> {
    if let Some(b) = budget { FBUDGET.store(b, Ordering::Relaxed); }
    crate::pns::STATS.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect()
}

/// A node of the forcing search: the board and `by`'s threat map on it, carried down the tree and updated
/// INCREMENTALLY (rules::threats_of_inc -- only the pieces a move can affect are recomputed; exact by the path lemma,
/// audited equal on 175,502 move-side maps, 2026-10-11).
#[derive(Clone)]
pub struct FState { pos: Chess, thr: rules::ThrArr, them: rules::ThrArr, edge: i64, last: Option<Move> }

/// (the material edge carried too: it changes only on a capture or a promotion)
fn child_state(s: &FState, m: &Move, by: Color) -> FState {
    let q = after(&s.pos, m);
    let thr = rules::threats_arr_inc(&s.thr, &q, m, by);
    let them = rules::threats_arr_inc(&s.them, &q, m, !by);
    let mover_is_by = s.pos.turn() == by;
    let mut d = m.capture().map_or(0, kernel_value) + m.promotion().map_or(0, |r| kernel_value(r) - 1);
    if !mover_is_by { d = -d; }
    FState { pos: q, thr, them, edge: s.edge + d, last: Some(m.clone()) }
}

/// The squares of `by`'s pieces that stand ALONE between one of `by`'s sliders and a square of `targets` (moving one
/// uncovers the slider: a discovery) -- bitboards, no path lists.
fn single_blockers(q: &Chess, by: Color, targets: Bitboard) -> Bitboard {
    let b = q.board();
    let occ = b.occupied();
    let mut out = Bitboard::EMPTY;
    for s in b.by_color(by) & (b.by_role(Role::Bishop) | b.by_role(Role::Rook) | b.by_role(Role::Queen)) {
        let Some(p) = b.piece_at(s) else { continue };
        for t in targets {
            if !shakmaty::attacks::attacks(s, p, Bitboard::EMPTY).contains(t) { continue; }
            let btw = shakmaty::attacks::between(s, t) & occ;
            if btw.count() == 1 && (btw & b.by_color(by)).any() { out |= btw; }
        }
    }
    out
}

/// The forcing moves of `by` (to move in s): checks, captures, promotions, and quiet moves that make a new threat worth
/// 3+ -- the moved piece attacking a piece of theirs worth 3+ from its new square, or leaving a square where it alone
/// blocked our slider from one (a discovery) -- each with its child node. Bitboard tests per move (2026-10-11, the
/// profile: attack-path lists and move strings were the forcing search's largest costs).
fn forcing_moves(s: &FState, by: Color) -> Vec<(Move, FState)> {
    let q = &s.pos;
    let b = q.board();
    let occ = b.occupied();
    let targets: Bitboard = b.by_color(!by) & !b.by_role(Role::Pawn) & !b.by_role(Role::King);
    let king = b.king_of(!by);
    let disc_t = single_blockers(q, by, targets);
    let disc_k = king.map_or(Bitboard::EMPTY, |k| single_blockers(q, by, Bitboard::from_square(k)));
    let mut out = Vec::new();
    for m in q.legal_moves() {
        let special = matches!(m, Move::Castle { .. } | Move::EnPassant { .. });
        let forcing_now = m.is_capture() || m.is_promotion();
        let from = m.from();
        let to = m.to();
        let role = m.promotion().unwrap_or(m.role());
        let reach = shakmaty::attacks::attacks(to, shakmaty::Piece { color: by, role }, occ.without(from.unwrap_or(to)).with(to));
        let checks = special || king.map_or(false, |k| reach.contains(k)) || from.map_or(false, |f| disc_k.contains(f));
        // (castling and en passant move or remove a second piece: the child's threat map decides -- the audit's one miss
        // was O-O-O, the rook arriving on d1 against the d4 bishop)
        let quiet_cand = special || (reach & targets).any() || from.map_or(false, |f| disc_t.contains(f));
        if !forcing_now && !quiet_cand && !checks { continue; }
        let c = child_state(s, &m, by);
        if forcing_now || c.pos.is_check() || (quiet_cand && (0..64).any(|x| c.thr[x] >= 3 && c.thr[x] > s.thr[x])) { out.push((m, c)); }
    }
    out
}

/// The reference generator (attack-path lists), kept for the equivalence audit.
fn forcing_moves_ref(s: &FState, by: Color) -> Vec<(Move, FState)> {
    let q = &s.pos;
    let targets: Bitboard = q.board().by_color(!by) & !q.board().by_role(Role::Pawn) & !q.board().by_role(Role::King);
    let paths = crate::paths::attack_paths(q, by, targets);
    let arriving: std::collections::HashSet<String> = paths.iter().filter(|a| a.kind == "direct").filter_map(|a| a.mv.clone()).collect();
    let disc_from: Vec<Square> = paths.iter().filter(|a| a.kind == "discovery").filter_map(|a| a.via).collect();
    let mut out = Vec::new();
    for m in q.legal_moves() {
        let forcing_now = m.is_capture() || m.is_promotion();
        let u = m.to_uci(CastlingMode::Standard).to_string();
        let disc = m.from().map_or(false, |f| disc_from.contains(&f));
        let quiet_cand = arriving.contains(&u) || disc;
        if !forcing_now && !quiet_cand {
            // a check is forcing too: only the board after it can tell
            let r = after(q, &m);
            if r.is_check() { let c = child_state(s, &m, by); out.push((m, c)); }
            continue;
        }
        let c = child_state(s, &m, by);
        if forcing_now || c.pos.is_check() || (0..64).any(|x| c.thr[x] >= 3 && c.thr[x] > s.thr[x]) { out.push((m, c)); }
    }
    out
}

/// WHAT THE SIDE TO MOVE CAN FORCE, by PROOF-NUMBER SEARCH (pns.rs; the claim "we force a gain of 1+"): our nodes branch
/// on forcing moves only, theirs on every reply; proven at a node of ours already standing 1+ better (the material won
/// plus its best capture now; not at the root -- the standing threats are the static check's business) or at their
/// mate; disproven where our forcing ends, at a stalemate, at a repetition. A node of theirs starts with as many proof
/// steps as it has replies (a check, with few, looks cheap first), weighted by `line_prior`. The budget is only a safety
/// net: hitting it is UNKNOWN. -> Some(true), Some(false), None.
struct Forcing { by: Color, base: i64, need: i64 }

impl crate::pns::Problem for Forcing {
    type State = FState;
    fn judge(&self, s: &FState, or: bool, root: bool) -> Option<bool> {
        if or {
            // THE COMPOSED STANDING (the exchange law's composed tier, Lean Ideas.ExchangeLaw.exchangeLawComposed: the
            // intermezzo's price is their best capture elsewhere after ours): the material won plus our best capture
            // minus their best capture -- a bound (2026-10-11: Bb5+ c6 "won 9", Nxd1 then cxb5 takes the bishop back)
            if root { return None; }
            // (material already banked needs no capture: our free move parries their threat -- the intermezzo is priced
            // only when the standing rests on a capture of ours; 2026-10-11: Kxh3, Rxh2, Nxe2 ... seven plain captures
            // lost to a price charged with nothing to pay for)
            if s.edge - self.base >= self.need { return Some(true); }
            let best = s.thr.iter().copied().max().unwrap_or(0).max(0) as i64;
            if s.edge - self.base + best < self.need { return None; }
            if !COMPOSED.load(Ordering::Relaxed) { return Some(true); }
            // the crude bound first (their largest threat now); when it does not settle it, the EXACT intermezzo price:
            // play our best capture, then their best capture on any other square (exchange_law.intermezzo; 2026-10-11:
            // the crude bound cost 10 of 367 Stockfish tactics)
            let crude = s.them.iter().copied().max().unwrap_or(0).max(0) as i64;
            if s.edge - self.base + best - crude >= self.need { return Some(true); }
            let t = (0..64).max_by_key(|x| (s.thr[*x], *x)).map(|x| Square::new(x as u32)).unwrap();
            let cap = s.pos.legal_moves().into_iter().filter(|m| m.to() == t).min_by_key(|m| kernel_value(m.role()));
            let Some(cap) = cap else { return None };
            let q2 = after(&s.pos, &cap);
            // (a capture that gives CHECK leaves them no capture elsewhere but an answer to the check -- the tempo)
            let legal_caps: Bitboard = if q2.is_check() { q2.legal_moves().iter().filter(|m| m.is_capture()).map(|m| m.to()).collect() } else { Bitboard::FULL };
            let price = rules::threats_of(&q2, !self.by).iter().filter(|(x, _)| **x as u32 != t as u32 && legal_caps.contains(Square::new(**x as u32))).map(|(_, g)| *g).max().unwrap_or(0).max(0);
            if s.edge - self.base + best - price >= self.need { return Some(true); }
            None
        } else if s.pos.legal_moves().is_empty() { Some(s.pos.is_check()) } else { None }
    }
    fn expand(&self, s: &FState, or: bool) -> Vec<(FState, f64)> {
        if or { forcing_moves(s, self.by).into_iter().map(|(m, c)| { let w = line_prior(&s.pos, &m); (c, w) }).collect() }
        else { s.pos.legal_moves().iter().map(|m| (child_state(s, m, self.by), 1.0)).collect() }
    }
    fn and_size(&self, s: &FState) -> f64 { s.pos.legal_moves().len().max(1) as f64 }
    fn key(&self, s: &FState) -> u64 {
        use shakmaty::zobrist::{Zobrist64, ZobristHash};
        s.pos.zobrist_hash::<Zobrist64>(shakmaty::EnPassantMode::Legal).0
    }
}

/// forcing_moves_audit(fen) -> (positions' moves in the bitboard generator, in the reference, in both) for the side to move.
#[pyfunction]
pub fn forcing_moves_audit(fen: String) -> (u32, u32, u32) {
    let Some(p) = parse(&fen) else { return (0, 0, 0) };
    let by = p.turn();
    let s = FState { pos: p.clone(), thr: rules::threats_arr(&p, by), them: rules::threats_arr(&p, !by), edge: material_edge(&p, by), last: None };
    let a: std::collections::HashSet<String> = forcing_moves(&s, by).iter().map(|(m, _)| m.to_uci(CastlingMode::Standard).to_string()).collect();
    let r: std::collections::HashSet<String> = forcing_moves_ref(&s, by).iter().map(|(m, _)| m.to_uci(CastlingMode::Standard).to_string()).collect();
    (a.len() as u32, r.len() as u32, a.intersection(&r).count() as u32)
}

pub static COMPOSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_composed(flag: bool) { COMPOSED.store(flag, Ordering::Relaxed); }

/// Does the side to move force a gain of at least `need` (checks, captures, quiet threats; the proof-number search)?
/// None: the safety net was hit (unknown).
pub fn forcing_need(p: &Chess, need: i64, budget: u64) -> Option<bool> {
    let by = p.turn();
    let root = FState { pos: p.clone(), thr: rules::threats_arr(p, by), them: rules::threats_arr(p, !by), edge: material_edge(p, by), last: None };
    crate::pns::search(&Forcing { by, base: material_edge(p, by), need }, root, budget)
}

pub fn forcing_gain(p: &Chess, budget: u64) -> Option<bool> {
    let by = p.turn();
    let root = FState { pos: p.clone(), thr: rules::threats_arr(p, by), them: rules::threats_arr(p, !by), edge: material_edge(p, by), last: None };
    crate::pns::search(&Forcing { by, base: material_edge(p, by), need: 1 }, root, budget)
}

/// forcing_value_of(fen, budget) -> the forcing value and its first move (uci) for the side to move, and the search
/// statistics of the call -- for digging.
#[pyfunction]
pub fn forcing_value_of(fen: String, budget: u64) -> (Option<(String, i64)>, Vec<u64>) {
    let Some(p) = parse(&fen) else { return (None, vec![]) };
    crate::pns::STATS.iter().for_each(|a| { a.swap(0, Ordering::Relaxed); });
    let v = forcing_value(&p, budget).map(|(m, g)| (m.to_uci(CastlingMode::Standard).to_string(), g));
    (v, crate::pns::STATS.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect())
}

/// WHAT THE SIDE TO MOVE CAN FORCE, AS A VALUE (in place of "material by checks within 3" -- a count, and checks only:
/// 2026-10-11, after Bxc3 their Re7+ Kc8 Rxf7 ... Rf8+ Rxa8 mixes checks and captures): the largest gain of a material
/// ladder (1, 2, 3, 5, 9) the forcing search proves, and the first move of that proof. Mates are the mate search's.
pub fn forcing_value(p: &Chess, budget: u64) -> Option<(Move, i64)> {
    let by = p.turn();
    let base = material_edge(p, by);
    let root = FState { pos: p.clone(), thr: rules::threats_arr(p, by), them: rules::threats_arr(p, !by), edge: base, last: None };
    let mut best: Option<(Move, i64)> = None;
    for need in [1i64, 2, 3, 5, 9] {
        match crate::pns::search_with_proof(&Forcing { by, base, need }, root.clone(), budget) {
            (Some(true), Some(c)) => { if let Some(m) = c.last { best = Some((m, need)); } }
            _ => break,
        }
    }
    best
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
    forcing_gain(&p, FBUDGET.load(Ordering::Relaxed))
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
