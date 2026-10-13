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
    let key = (zkey(p), n);
    if let Some(v) = MATE_CACHE.with(|c| c.borrow().get(&key).cloned()) { return v; }
    let v = crate::prof::timed("mate_by_checks", || mate_by_checks_inner(p, n));
    MATE_CACHE.with(|c| c.borrow_mut().insert(key, v.clone()));
    v
}
fn mate_by_checks_inner(p: &Chess, n: u32) -> Option<Move> {
    let checks = checking_moves(p);
    if let Some((m, _)) = checks.iter().find(|(_, q)| q.legal_moves().is_empty()) { return Some(m.clone()); }
    if n <= 1 { return None; }
    // (a claimable draw on the way -- the game context -- is the defender's escape)
    let draws = crate::plan::GAME_DRAWS.load(Ordering::Relaxed);
    let open = |x: &Chess| !(draws && crate::game::claimable_draw(x).is_some());
    checks.into_iter().filter(|(_, q)| open(q))
        .find(|(_, q)| q.legal_moves().iter().all(|r| { let x = after(q, r); open(&x) && mate_by_checks(&x, n - 1).is_some() })).map(|(m, _)| m)
}

/// A mate in one for the side to move, if any.
pub fn mate_in_one(p: &Chess) -> Option<Move> { mate_by_checks(p, 1) }

/// Every threat of `by` in `pos` (`by` need not be to move: when not, the other side passes -- impossible when `by` is
/// giving check, then only the material is read).
// THE POSITION CACHES (2026-10-11, the game profile: the generic threat was computed 203 times a position and the mate
// search 2,400 times, mostly on the same positions after the same candidate moves): pure functions of the position,
// memoized by its Zobrist hash within one top-level plan() (plan.rs clears them). Exact: no semantic change.
thread_local! {
    static THREATS_CACHE: std::cell::RefCell<std::collections::HashMap<(u64, bool), Vec<Threat>>> = std::cell::RefCell::new(std::collections::HashMap::new());
    static MATE_CACHE: std::cell::RefCell<std::collections::HashMap<(u64, u32), Option<Move>>> = std::cell::RefCell::new(std::collections::HashMap::new());
    static FORCING_CACHE: std::cell::RefCell<std::collections::HashMap<(u64, u64), Option<bool>>> = std::cell::RefCell::new(std::collections::HashMap::new());
    static CORNER_CACHE: std::cell::RefCell<std::collections::HashMap<(u64, u64), Option<Move>>> = std::cell::RefCell::new(std::collections::HashMap::new());
    static GAIN_CACHE: std::cell::RefCell<std::collections::HashMap<(u64, u32), Option<(Move, i64)>>> = std::cell::RefCell::new(std::collections::HashMap::new());
}
pub fn clear_position_caches() {
    THREATS_CACHE.with(|c| c.borrow_mut().clear());
    MATE_CACHE.with(|c| c.borrow_mut().clear());
    FORCING_CACHE.with(|c| c.borrow_mut().clear());
    CORNER_CACHE.with(|c| c.borrow_mut().clear());
    GAIN_CACHE.with(|c| c.borrow_mut().clear());
}
pub(crate) fn zkey(p: &Chess) -> u64 {
    use shakmaty::zobrist::{Zobrist64, ZobristHash};
    p.zobrist_hash::<Zobrist64>(shakmaty::EnPassantMode::Legal).0
}
pub fn threats(pos: &Chess, by: Color) -> Vec<Threat> {
    let key = (zkey(pos), by == Color::White);
    if let Some(v) = THREATS_CACHE.with(|c| c.borrow().get(&key).cloned()) { return v; }
    let v = crate::prof::timed("threats (generic)", || threats_inner(pos, by));
    THREATS_CACHE.with(|c| c.borrow_mut().insert(key, v.clone()));
    v
}
fn threats_inner(pos: &Chess, by: Color) -> Vec<Threat> {
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
        } else if CORNERING.load(Ordering::Relaxed) {
            // THE MATING NET (the cornering lemma for the king): a mate by the forcing search -- checks, captures and the
            // quiet moves that take the king's flight squares -- the 12 catastrophic misses of the surprise loop's first turn
            if let Some(k) = p.board().king_of(!by) {
                if let Some(m) = forcing_mate(&p, FBUDGET.load(Ordering::Relaxed)) {
                    v.push(Threat { target: k, value: MATE_GOAL, mate: true, how: Some(m.to_uci(CastlingMode::Standard).to_string()) });
                }
            }
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
/// Cached per position within one plan() (the forcing ladder's searches judge the same leaves again).
pub fn gain_by_checks(p: &Chess, n: u32) -> Option<(Move, i64)> {
    let key = (zkey(p), n);
    if let Some(v) = GAIN_CACHE.with(|c| c.borrow().get(&key).cloned()) { return v; }
    let v = crate::prof::timed("gain_by_checks", || gain_by_checks_inner(p, n));
    GAIN_CACHE.with(|c| c.borrow_mut().insert(key, v.clone()));
    v
}
fn gain_by_checks_inner(p: &Chess, n: u32) -> Option<(Move, i64)> {
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
/// The king's FLIGHT SQUARES of `c` in `q`: next to it, not its own piece, not attacked by the other side -- the king's
/// escape set, as escape_set is a piece's (THE CORNERING LEMMA unified, Kaveh 2026-10-11: the king is a piece of
/// infinite value; taking a square from it is a threat as attacking a piece is).
pub fn king_flights(q: &Chess, c: Color) -> u32 {
    let b = q.board();
    let Some(k) = b.king_of(c) else { return 0 };
    let occ = b.occupied().without(k);
    (shakmaty::attacks::king_attacks(k) & !b.by_color(c)).into_iter().filter(|sq| !b.attacks_to(*sq, !c, occ).any()).count() as u32
}

fn forcing_moves(s: &FState, by: Color, need_left: i64, corner: bool) -> Vec<(Move, FState)> {
    let q = &s.pos;
    let qt = QUIET_THREATS.load(Ordering::Relaxed);
    // (the king's flights before, for the quiet moves that take one: CORNERING flag, or a mate search that asks for it)
    let flights0 = if corner || CORNERING.load(Ordering::Relaxed) { king_flights(q, !by) } else { 0 };
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
        // a quiet move that bears on the king's neighbourhood may take a flight square (the net is built before the check)
        let king_cand = flights0 > 0 && king.map_or(false, |k| (reach & shakmaty::attacks::king_attacks(k)).any() || from.map_or(false, |f| disc_k.contains(f)));
        let quiet_cand = special || (reach & targets).any() || from.map_or(false, |f| disc_t.contains(f));
        // (QUIET_THREATS: a quiet move bearing on their king's neighbourhood may threaten mate; any quiet move may fork)
        let mate_cand = qt && king.map_or(false, |k| (reach & shakmaty::attacks::king_attacks(k)).any());
        if !forcing_now && !quiet_cand && !checks && !king_cand && !mate_cand { continue; }
        // (a move that is only a mate candidate: the mate threat is tested on the bare position first -- the threat maps are
        // built only when it holds)
        // (THE MAGNITUDE of a mate threat: a pure one that they can parry wins nothing, so in a search for material it
        // counts only beside a second threat -- a fork with mate; a pure mate threat is searched only for a mate)
        let only_mate = !forcing_now && !quiet_cand && !checks && !king_cand;
        let for_mate = need_left >= crate::waiting::MATE_GOAL;
        if only_mate && !for_mate { continue; }
        let mate_test = |q0: &Chess| -> bool { let q1 = after(q0, &m); !q1.is_check() && pass(&q1).map_or(false, |p| mate_by_checks(&p, 1).is_some()) };
        if only_mate && !mate_test(q) { continue; }
        let c = child_state(s, &m, by);
        let takes_flight = king_cand && !c.pos.is_check() && king_flights(&c.pos, !by) < flights0;
        // THE QUIET THREATS (Kaveh 2026-10-11, the dig: their quiet Rdd1 built a mate threat no forcing search saw): a quiet
        // move after which we threaten MATE IN ONE, or make TWO new threats whose second (what one reply cannot parry, the
        // fork law) reaches what is still needed, is forcing like a check or a capture
        let quiet_threat = qt && !forcing_now && !c.pos.is_check() && {
            let mut new: Vec<i16> = (0..64).filter(|x| c.thr[*x] > 0 && c.thr[*x] > s.thr[*x]).map(|x| c.thr[x]).collect();
            new.sort_unstable_by(|a, b| b.cmp(a));
            let fork = new.get(1).map_or(false, |v| (*v as i64) >= need_left.clamp(1, 3));
            fork || (mate_cand && (only_mate || !new.is_empty()) && mate_test(q))
        };
        if forcing_now || c.pos.is_check() || takes_flight || quiet_threat || (quiet_cand && (0..64).any(|x| c.thr[x] >= 3 && c.thr[x] > s.thr[x])) { out.push((m, c)); }
    }
    out
}

// THE FORCED ENTRY INTO WHAT WE KNOW (Kaveh 2026-10-12: "the most likely trade is a rook for a rook ... can we prove that we
// can force one trade? ... that's the only trade we can bank on ... we should try to prove our way to achieving that
// goal"): our nodes branch on the forcing moves (checks, captures, threats on their pieces), theirs on every reply;
// PROVEN at a node of ours whose material we KNOW how to win (a frame for it: vocab.rs is_known) with their material
// smaller and our lead no lower than at the start -- a trade proves as a capture does: the known material is the goal,
// not a gain -- or at their mate; disproven at a stalemate, a claimable draw, a repetition, the forcing's end. The proof
// is the certificate on the board; Lean Ideas.Forced.toward_known composes it with the known material's own proof.
struct KnownEntry { by: Color, lead0: i64, theirs0: i64 }

fn known_here(q: &Chess, by: Color, lead0: i64, theirs0: i64) -> bool {
    let b = q.board();
    let (ours, theirs) = (crate::vocab::material(b, by), crate::vocab::material(b, !by));
    theirs < theirs0 && ours - theirs >= lead0 && crate::vocab::is_known(&crate::vocab::material_key(b, by))
}

impl crate::pns::Problem for KnownEntry {
    type State = FState;
    fn judge(&self, s: &FState, or: bool, root: bool) -> Option<bool> {
        if !root && crate::plan::GAME_DRAWS.load(Ordering::Relaxed) && crate::game::claimable_draw(&s.pos).is_some() { return Some(false); }
        let none = s.pos.legal_moves().is_empty();
        if or {
            if none { return Some(false); }                                   // mated or stalemated
            if !root && known_here(&s.pos, self.by, self.lead0, self.theirs0) { return Some(true); }
            None
        } else if none { Some(s.pos.is_check()) } else { None }               // their mate, or a stalemate
    }
    fn expand(&self, s: &FState, or: bool) -> Vec<(FState, f64)> {
        if or { forcing_moves(s, self.by, 5, false).into_iter().map(|(m, c)| { let w = line_prior(&s.pos, &m); (c, w) }).collect() }
        else { s.pos.legal_moves().iter().map(|m| (child_state(s, m, self.by), 1.0)).collect() }
    }
    fn and_size(&self, s: &FState) -> f64 { s.pos.legal_moves().len().max(1) as f64 }
    fn key(&self, s: &FState) -> u64 {
        use shakmaty::zobrist::{Zobrist64, ZobristHash};
        s.pos.zobrist_hash::<Zobrist64>(shakmaty::EnPassantMode::Legal).0
    }
}

/// The forced entry from `p` (the side to move): Some(the first move) when proven within the budget.
pub fn known_entry(p: &Chess, budget: u64) -> Option<Move> {
    let by = p.turn();
    let b = p.board();
    let (ours, theirs) = (crate::vocab::material(b, by), crate::vocab::material(b, !by));
    if theirs == 0 { return None; }
    let root = FState { pos: p.clone(), thr: rules::threats_arr(p, by), them: rules::threats_arr(p, !by), edge: ours - theirs, last: None };
    match crate::pns::search_with_proof(&KnownEntry { by, lead0: ours - theirs, theirs0: theirs }, root, budget) {
        (Some(true), Some(c)) => c.last,
        _ => None,
    }
}

/// THE REGRESSION, one level at a time (Kaveh 2026-10-12: "I want the machinery to search for this" -- a way into what we
/// know that has to be CREATED): level 1 is the forced entry; LEVEL n+1 is a move of ours -- quiet or not, nothing of ours
/// left to the static exchange -- after which EVERY reply leaves a position of level n (or the known itself). Each level
/// is a proof against every reply; Lean Ideas.Forced.toward_known composes them (the known region grows backward from
/// what we can prove). -> the move, when proven within the budgets (each forced-entry question gets `budget`).
pub fn known_level(p: &Chess, level: u32, budget: u64, cache: &mut std::collections::HashMap<u64, bool>) -> Option<Move> {
    if level <= 1 { return known_entry(p, budget); }
    let by = p.turn();
    let b = p.board();
    let (ours, theirs) = (crate::vocab::material(b, by), crate::vocab::material(b, !by));
    if theirs == 0 { return None; }
    let lead0 = ours - theirs;
    // our moves: no piece of ours left en prise, no stalemate; captures and checks first
    let mut moves: Vec<Move> = p.legal_moves().into_iter().filter(|m| {
        let q = after(p, m);
        !q.legal_moves().is_empty() && (q.board().by_color(by) & !q.board().kings()).into_iter().all(|s| rules::see(q.board(), s, !by) == 0)
    }).collect();
    moves.sort_by_key(|m| !(m.is_capture() || after(p, m).is_check()));
    'mv: for m in moves {
        let q = after(p, &m);
        if q.is_checkmate() { return Some(m); }
        // the replies most likely to refute first: checks and captures
        let mut replies: Vec<Move> = q.legal_moves().into_iter().collect();
        replies.sort_by_key(|r| !(r.is_capture() || after(&q, r).is_check()));
        for r in replies {
            let x = after(&q, &r);
            if x.legal_moves().is_empty() { if x.is_checkmate() { continue 'mv; } else { continue 'mv; } }   // they mate us / stalemate
            if known_here(&x, by, lead0, theirs) { continue; }
            if crate::vocab::material(x.board(), by) - crate::vocab::material(x.board(), !by) < lead0 { continue 'mv; }
            let key = crate::threat::zkey(&x).wrapping_add(level as u64);
            let ok = match cache.get(&key) { Some(v) => *v, None => { let v = known_level(&x, level - 1, budget, cache).is_some(); cache.insert(key, v); v } };
            if !ok { continue 'mv; }
        }
        return Some(m);
    }
    None
}

/// known_level_of(fen, level, budget) -> the first move of a proven way into what we know at that level, if any.
#[pyfunction]
pub fn known_level_of(fen: String, level: u32, budget: u64) -> Option<String> {
    let p = crate::parse(&fen)?;
    known_level(&p, level, budget, &mut std::collections::HashMap::new()).map(|m| m.to_uci(CastlingMode::Standard).to_string())
}

/// known_entry_of(fen, budget) -> the first move of a forced entry into a material we know, if proven.
#[pyfunction]
pub fn known_entry_of(fen: String, budget: u64) -> Option<String> {
    let p = crate::parse(&fen)?;
    known_entry(&p, budget).map(|m| m.to_uci(CastlingMode::Standard).to_string())
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
struct Forcing { by: Color, base: i64, need: i64, corner: bool }

/// THE STANDING WITH THE EXCHANGE LAW'S CONDITIONS (Kaveh 2026-10-11, the dig of the surprise loop: 7 false "wins by force"
/// -- their reply gave check, pinned our capturing queen, promoted, or our capture was counted while we were in check;
/// Lean Ideas.ExchangeLaw.honestComposedB states the conditions): an OR node (our move) is proven only when
///   BANKED   the material won, less what they still threaten that our one free move cannot parry: their SECOND-best
///            threat (their best when we are in check -- the move goes to the check), reaches the need; or
///   CAPTURE  our best LEGAL capture (a pinned capturer is no capturer; in check only a capture that answers it) by the
///            exchange law, less their best capture elsewhere after it (the intermezzo), reaches the need, AND after it they
///            have no CHECK (a tempo), no PROMOTION, and no move that brings a NEW attacker onto our capturing piece (the
///            pin that won the queen; the law's noArrival).
/// Otherwise the node stays open: the search goes on (sound -- nothing is proven that the conditions do not cover).
/// CAN A PIECE OF `them` BECOME A NEW ATTACKER OF `t` IN ONE MOVE (them to move in `q`)? Bitboards only, a SUPERSET of the
/// exact answer (every move tried): a piece not attacking `t` now that can step (attack-reach, or a pawn's push) onto a
/// square from which its kind attacks `t`, or a piece of theirs standing alone between their slider and `t` (a
/// discovery). False means certainly no arrival (the noArrival hypothesis holds); true may be a false alarm (the node
/// stays open -- sound).
fn may_arrive(q: &Chess, t: Square, them: Color) -> bool {
    let b = q.board();
    let occ = b.occupied();
    let att0 = b.attacks_to(t, them, occ);
    for p_sq in b.by_color(them) & !att0 {
        let Some(p) = b.piece_at(p_sq) else { continue };
        let occ_wo = occ.without(p_sq);
        let reach = if p.role == Role::Pawn {
            let fwd = if them == Color::White { 8i32 } else { -8 };
            let mut r = Bitboard::EMPTY;
            if let Some(y) = p_sq.offset(fwd) { if !occ.contains(y) { r.add(y); let home = if them == Color::White { 1 } else { 6 }; if p_sq.rank() as u32 == home { if let Some(y2) = y.offset(fwd) { if !occ.contains(y2) { r.add(y2); } } } } }
            r | (shakmaty::attacks::pawn_attacks(them, p_sq) & b.by_color(!them))
        } else { shakmaty::attacks::attacks(p_sq, p, occ) & !b.by_color(them) };
        // the squares from which this kind attacks t (a pawn's: the reverse of a pawn attack)
        let from_t = if p.role == Role::Pawn { shakmaty::attacks::pawn_attacks(!them, t) } else { shakmaty::attacks::attacks(t, shakmaty::Piece { color: them, role: p.role }, occ_wo) };
        if (reach & from_t).any() { return true; }
    }
    single_blockers(q, them, Bitboard::from_square(t)).any()
}

fn judge_conditions(f: &Forcing, s: &FState) -> Option<bool> {
    let in_check = s.pos.is_check();
    let banked = s.edge - f.base;
    let mut them: Vec<i64> = s.them.iter().map(|v| *v as i64).filter(|v| *v > 0).collect();
    them.sort_unstable_by(|a, b| b.cmp(a));
    if BANKED_FORCING.load(Ordering::Relaxed) {
        // (2026-10-12, the dig: "fxg5 wins 3 by force" stood on Qg7+ with the knight banked -- Ke8 Bb5+ ... Qxh8+ wins the
        // rook; fxe6 "wins 2" with Qh5's mate threat beside Nxe6's fork) IN CHECK the move goes to the check and they move
        // again: banked material stands only once the check is answered -- the node stays open, its few answers searched
        // (a check removes the tempo, Kaveh 2026-10-11). NOT IN CHECK their FORCING threats join the static ones -- a mate
        // by checks and material by checks: our one free move parries one threat, not a mate and a piece
        if !in_check && banked - them.get(1).copied().unwrap_or(0) >= f.need {
            // (lazily, the same verdict: with `margin` what the second threat may take, a forcing threat matters only as
            // the SECOND one beyond the margin -- beside a static threat beyond it, one forcing threat beyond it suffices;
            // otherwise two are needed, a mate and material by checks beyond it)
            let margin = banked - f.need;
            let open = BANKED_THREATS.load(Ordering::Relaxed) && pass(&s.pos).map_or(false, |p| {
                let gain = || gain_by_checks(&p, GAIN_CHECKS).map_or(false, |(_, g)| g > margin);
                let mate = || mate_by_checks(&p, MATE_CHECKS).is_some();
                if them.first().copied().unwrap_or(0) > margin { gain() || mate() } else { gain() && mate() }
            });
            if !open { return Some(true); }
        }
    } else {
        let parry_left = if in_check { them.first().copied().unwrap_or(0) } else { them.get(1).copied().unwrap_or(0) };
        if banked - parry_left >= f.need { return Some(true); }
    }
    // our captures, best first, the first LEGAL one
    let legal = s.pos.legal_moves();
    let mut sqs: Vec<(i16, usize)> = (0..64usize).filter(|x| s.thr[*x] > 0).map(|x| (s.thr[x], x)).collect();
    sqs.sort_unstable_by(|a, b| b.cmp(a));
    for (g, x) in sqs {
        let g = g as i64;
        if banked + g < f.need { return None; }                       // no capture left can reach the need
        let t = Square::new(x as u32);
        let Some(cap) = legal.iter().filter(|m| m.to() == t).min_by_key(|m| kernel_value(m.role())).cloned() else { continue };
        let q2 = after(&s.pos, &cap);
        // A CAPTURE THAT STALEMATES wins nothing (2026-10-12, the surprise loop, K+R+R v K+R: "Kf6 wins 5 by force",
        // Re6+ and Kxe6 stalemates -- the settle read the capture as +5); a capture that mates settles it
        if q2.legal_moves().is_empty() { if q2.is_check() { return Some(true); } continue; }
        if !checking_moves(&q2).is_empty() { return None; }            // their check: a tempo -- not settled
        let moves2 = q2.legal_moves();
        if moves2.iter().any(|m| m.is_promotion()) { return None; }      // their promotion -- not settled
        if may_arrive(&q2, t, !f.by) { return None; }                    // a new attacker on our capturer -- not settled
        let legal_caps: Bitboard = if q2.is_check() { moves2.iter().filter(|m| m.is_capture()).map(|m| m.to()).collect() } else { Bitboard::FULL };
        let price = rules::threats_of(&q2, !f.by).iter().filter(|(y, _)| **y as u32 != t as u32 && legal_caps.contains(Square::new(**y as u32))).map(|(_, v)| *v).max().unwrap_or(0).max(0);
        return if banked + g - price >= f.need { Some(true) } else { None };
    }
    None
}
pub static CONDITIONS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub static BANKED_FORCING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub static BANKED_THREATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_banked_threats(flag: bool) -> PyResult<()> { BANKED_THREATS.store(flag, Ordering::Relaxed); Ok(()) }
pub static EXACT_VALUE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_banked_forcing(flag: bool) -> PyResult<()> { BANKED_FORCING.store(flag, Ordering::Relaxed); Ok(()) }
#[pyfunction]
pub fn set_exact_value(flag: bool) -> PyResult<()> { EXACT_VALUE.store(flag, Ordering::Relaxed); Ok(()) }
pub static QUIET_THREATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_quiet_threats(flag: bool) -> PyResult<()> { QUIET_THREATS.store(flag, Ordering::Relaxed); Ok(()) }
#[pyfunction]
pub fn set_conditions(flag: bool) -> PyResult<()> { CONDITIONS.store(flag, Ordering::Relaxed); Ok(()) }

pub const POLICY_FEATURES: usize = 14;
fn policy_features(f: &Forcing, s: &FState, depth: u32) -> Vec<f32> {
    let b = s.pos.board();
    let banked = (s.edge - f.base) as f32;
    let replies = s.pos.legal_moves();
    let flights = replies.iter().filter(|m| m.role() == Role::King).count() as f32;
    let (cap, pv, promo) = match &s.last { Some(m) => (m.capture().map(kernel_value).unwrap_or(0) as f32, kernel_value(m.role()) as f32, m.is_promotion() as u8 as f32), None => (0.0, 0.0, 0.0) };
    // (a mate question's need is a sentinel beyond any material: the policy learned material needs, so it reads the
    // largest one -- 2026-10-11, the surprise loop: every line of every mate search was stopped at the root)
    let need = f.need.min(9) as f32;
    vec![depth as f32, banked, need - banked,
         s.thr.iter().copied().max().unwrap_or(0).max(0) as f32, s.them.iter().copied().max().unwrap_or(0).max(0) as f32,
         s.pos.is_check() as u8 as f32, replies.len() as f32, flights,
         (cap > 0.0) as u8 as f32, cap, pv, promo,
         s.them.iter().filter(|v| **v > 0).count() as f32, s.thr.iter().filter(|v| **v > 0).count() as f32]
        .into_iter().chain(std::iter::repeat(0.0)).take(POLICY_FEATURES).chain(std::iter::once(b.occupied().count() as f32)).take(POLICY_FEATURES).collect()
}

/// THE LEARNED STOP POLICY: a two-layer net on the features (trained in PyTorch, catspace/incubator/analyst/forcing_policy.py),
/// p(this line ends in a proof); a line below `tau` is dropped. Loaded once from JSON (set_forcing_policy).
pub struct Policy { w1: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: f32, mean: Vec<f32>, std: Vec<f32>, pub tau: f32,
                    // PHASE 2: a cost head on the same trunk (log expansions of the line's subtree) and the price of an expansion
                    wc: Vec<f32>, bc: f32, pub lam: f32 }
impl Policy {
    /// (p(the line proves), its expected cost in expansions)
    pub fn heads(&self, f: &[f32]) -> (f32, f32) {
        let x: Vec<f32> = f.iter().zip(self.mean.iter().zip(self.std.iter())).map(|(v, (m, s))| (v - m) / s.max(1e-6)).collect();
        let mut z = self.b2;
        let mut c = self.bc;
        for (k, (row, bw)) in self.w1.iter().zip(self.b1.iter()).enumerate() {
            let h = (row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>() + bw).max(0.0);
            z += self.w2[k] * h;
            if let Some(wc) = self.wc.get(k) { c += wc * h; }
        }
        (1.0 / (1.0 + (-z).exp()), c.exp())
    }
    pub fn prob(&self, f: &[f32]) -> f32 { self.heads(f).0 }
    /// THE DECISION (Kaveh's reward, per line): follow the line iff its chance of proving is worth its expected cost --
    /// p >= lam * cost; without a cost head (phase 1) the flat threshold tau.
    pub fn follow(&self, f: &[f32]) -> bool {
        let (p, c) = self.heads(f);
        if self.wc.is_empty() { p >= self.tau } else { p >= self.lam * c.max(1.0) }
    }
}
pub static POLICY: std::sync::OnceLock<Policy> = std::sync::OnceLock::new();
pub static POLICY_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static RL_SAMPLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
thread_local! {
    static RNG: std::cell::Cell<u64> = const { std::cell::Cell::new(0x9E37_79B9_7F4A_7C15) };
    static EPISODE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    // (episode, features, continued?, p) per decision; (episode, verdict, truth, expansions) per search
    static EP_LOG: std::cell::RefCell<Vec<(u64, Vec<f32>, bool, f32)>> = const { std::cell::RefCell::new(Vec::new()) };
    static EP_OUT: std::cell::RefCell<Vec<(u64, i8, i8, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
}
/// set_rl_sample(flag): the policy acts stochastically and logs its decisions; every forcing_gain also runs the full
/// search at budget 1,000 for the truth (collection only -- slow).
#[pyfunction]
pub fn set_rl_sample(flag: bool) -> PyResult<()> { RL_SAMPLE.store(flag, Ordering::Relaxed); Ok(()) }
/// rl_log() -> (decisions: (episode, features, continued, p), episodes: (episode, verdict, truth, expansions)); verdict/truth:
/// 1 proven, 0 disproven, -1 unknown.
#[pyfunction]
pub fn rl_log() -> (Vec<(u64, Vec<f32>, bool, f32)>, Vec<(u64, i8, i8, u64)>) {
    (EP_LOG.with(|l| std::mem::take(&mut *l.borrow_mut())), EP_OUT.with(|l| std::mem::take(&mut *l.borrow_mut())))
}
fn verdict_code(v: Option<bool>) -> i8 { match v { Some(true) => 1, Some(false) => 0, None => -1 } }
/// set_forcing_policy_on(flag): the loaded stop policy decides (off: every line followed, the search as before).
#[pyfunction]
pub fn set_forcing_policy_on(flag: bool) -> PyResult<()> { POLICY_ON.store(flag, Ordering::Relaxed); Ok(()) }
/// set_forcing_policy(json): load the stop policy {w1, b1, w2, b2, mean, std, tau} (once per process).
#[pyfunction]
pub fn set_forcing_policy(json: String) -> PyResult<bool> {
    let v: serde_json::Value = serde_json::from_str(&json).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let f32s = |k: &str| -> Vec<f32> { v[k].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default() };
    let w1: Vec<Vec<f32>> = v["w1"].as_array().map(|rows| rows.iter().map(|r| r.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default()).collect()).unwrap_or_default();
    let pol = Policy { w1, b1: f32s("b1"), w2: f32s("w2"), b2: v["b2"].as_f64().unwrap_or(0.0) as f32, mean: f32s("mean"), std: f32s("std"), tau: v["tau"].as_f64().unwrap_or(0.0) as f32,
                       wc: f32s("wc"), bc: v["bc"].as_f64().unwrap_or(0.0) as f32, lam: v["lam"].as_f64().unwrap_or(0.001) as f32 };
    let ok = POLICY.set(pol).is_ok();
    POLICY_ON.store(true, Ordering::Relaxed);
    Ok(ok)
}
/// forcing_log() -> the policy log since the last call: (features, the line ended proven, the root was proven).
#[pyfunction]
pub fn forcing_log() -> Vec<(Vec<f32>, bool, bool, u32)> { crate::pns::take_log() }
/// set_forcing_log(flag): record every line the forcing search creates.
#[pyfunction]
pub fn set_forcing_log(flag: bool) -> PyResult<()> { crate::pns::LOG.store(flag, Ordering::Relaxed); Ok(()) }
/// policy_stats() -> [lines created, lines stopped by the policy] since the last call.
#[pyfunction]
pub fn policy_stats() -> Vec<u64> { crate::pns::POLICY_N.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect() }

impl crate::pns::Problem for Forcing {
    type State = FState;
    fn judge(&self, s: &FState, or: bool, root: bool) -> Option<bool> {
        // a claimable draw on the way (the game context) ends the line: no gain, no mate
        if !root && crate::plan::GAME_DRAWS.load(Ordering::Relaxed) && crate::game::claimable_draw(&s.pos).is_some() { return Some(false); }
        if or {
            // (no move of ours: mated or stalemated -- no standing counts; 2026-10-11, the surprise loop: axb5 Qf7# was
            // "proven" at +3 because the standing read the material of a mated side)
            if s.pos.is_check() && s.pos.legal_moves().is_empty() { return Some(false); }
            if !root && !s.pos.is_check() && s.pos.legal_moves().is_empty() { return Some(false); }
            // THE COMPOSED STANDING (the exchange law's composed tier, Lean Ideas.ExchangeLaw.exchangeLawComposed: the
            // intermezzo's price is their best capture elsewhere after ours): the material won plus our best capture
            // minus their best capture -- a bound (2026-10-11: Bb5+ c6 "won 9", Nxd1 then cxb5 takes the bishop back)
            if root { return None; }
            // (material already banked needs no capture: our free move parries their threat -- the intermezzo is priced
            // only when the standing rests on a capture of ours; 2026-10-11: Kxh3, Rxh2, Nxe2 ... seven plain captures
            // lost to a price charged with nothing to pay for)
            if CONDITIONS.load(Ordering::Relaxed) { return judge_conditions(self, s); }
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
            if q2.legal_moves().is_empty() && !q2.is_check() { return None; }   // a capture that stalemates wins nothing
            // (a capture that gives CHECK leaves them no capture elsewhere but an answer to the check -- the tempo)
            let legal_caps: Bitboard = if q2.is_check() { q2.legal_moves().iter().filter(|m| m.is_capture()).map(|m| m.to()).collect() } else { Bitboard::FULL };
            let price = rules::threats_of(&q2, !self.by).iter().filter(|(x, _)| **x as u32 != t as u32 && legal_caps.contains(Square::new(**x as u32))).map(|(_, g)| *g).max().unwrap_or(0).max(0);
            if s.edge - self.base + best - price >= self.need { return Some(true); }
            None
        } else if s.pos.legal_moves().is_empty() { Some(s.pos.is_check()) } else { None }
    }
    fn expand(&self, s: &FState, or: bool) -> Vec<(FState, f64)> {
        if or { forcing_moves(s, self.by, self.need - (s.edge - self.base), self.corner).into_iter().map(|(m, c)| { let w = line_prior(&s.pos, &m); (c, w) }).collect() }
        else { s.pos.legal_moves().iter().map(|m| (child_state(s, m, self.by), 1.0)).collect() }
    }
    fn and_size(&self, s: &FState) -> f64 { s.pos.legal_moves().len().max(1) as f64 }
    /// THE STOP POLICY's features for a line of ours, read on the position after our move (they to move): generic scalars,
    /// no piece names -- the plies so far, the material banked, what is still needed, our best and their best static
    /// threat, whether they are in check, their replies, their king's flight squares, the move's kind and values, the
    /// counts of threatened pieces on each side.
    fn features(&self, s: &FState, depth: u32) -> Option<Vec<f32>> { Some(policy_features(self, s, depth)) }
    fn allow(&self, s: &FState, depth: u32) -> bool {
        if !POLICY_ON.load(Ordering::Relaxed) { return true; }
        let Some(pol) = POLICY.get() else { return true };
        let f = policy_features(self, s, depth);
        let p = pol.prob(&f);
        // PHASE 2 (RL): the policy acts STOCHASTICALLY while collecting -- continue with probability p -- and every
        // decision is logged with its episode (the search) for the policy-gradient update
        if RL_SAMPLE.load(Ordering::Relaxed) {
            let u = RNG.with(|r| { let mut x = r.get(); x ^= x << 13; x ^= x >> 7; x ^= x << 17; r.set(x); (x >> 11) as f32 / (1u64 << 53) as f32 });
            let act = u < p;
            EP_LOG.with(|l| l.borrow_mut().push((EPISODE.with(|e| e.get()), f, act, p)));
            return act;
        }
        let _ = p;
        pol.follow(&f)
    }
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
    let a: std::collections::HashSet<String> = forcing_moves(&s, by, 3, false).iter().map(|(m, _)| m.to_uci(CastlingMode::Standard).to_string()).collect();
    let r: std::collections::HashSet<String> = forcing_moves_ref(&s, by).iter().map(|(m, _)| m.to_uci(CastlingMode::Standard).to_string()).collect();
    (a.len() as u32, r.len() as u32, a.intersection(&r).count() as u32)
}

pub static COMPOSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// THE CORNERING LEMMA unified (Kaveh 2026-10-11): quiet moves that take a flight square from the king are forcing, and a
/// mate found by the forcing search (checks, captures, net-building quiet moves) is a threat beside the checks-only mate.
pub static CORNERING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_cornering(flag: bool) -> PyResult<()> { CORNERING.store(flag, Ordering::Relaxed); Ok(()) }
#[pyfunction]
pub fn set_composed(flag: bool) { COMPOSED.store(flag, Ordering::Relaxed); }

/// Does the side to move force a gain of at least `need` (checks, captures, quiet threats; the proof-number search)?
/// None: the safety net was hit (unknown).
pub fn forcing_need(p: &Chess, need: i64, budget: u64) -> Option<bool> {
    let by = p.turn();
    let root = FState { pos: p.clone(), thr: rules::threats_arr(p, by), them: rules::threats_arr(p, !by), edge: material_edge(p, by), last: None };
    crate::pns::search(&Forcing { by, base: material_edge(p, by), need, corner: false }, root, budget)
}

/// A MATE by the forcing search (need beyond any material: only a mate proves it); the first move of the proof.
pub fn forcing_mate(p: &Chess, budget: u64) -> Option<Move> { forcing_mate_with(p, budget, false) }
/// THE CORNERING MATE: the forcing search for a mate with the king's flight squares as threats (a quiet move that takes
/// one is forcing) whatever the CORNERING switch -- asked only where a mate is what matters (plan.rs: our side when only a
/// mate moves the expected result; their mate against a candidate's net). Cached per position.
pub fn forcing_mate_cornering(p: &Chess, budget: u64) -> Option<Move> {
    let key = (zkey(p), budget ^ 0xC0_C0);
    if let Some(v) = CORNER_CACHE.with(|c| c.borrow().get(&key).cloned()) { return v; }
    let v = crate::prof::timed("forcing_mate (cornering)", || forcing_mate_with(p, budget, true));
    CORNER_CACHE.with(|c| c.borrow_mut().insert(key, v.clone()));
    v
}
fn forcing_mate_with(p: &Chess, budget: u64, corner: bool) -> Option<Move> {
    let by = p.turn();
    let base = material_edge(p, by);
    let root = FState { pos: p.clone(), thr: rules::threats_arr(p, by), them: rules::threats_arr(p, !by), edge: base, last: None };
    match crate::pns::search_with_proof(&Forcing { by, base, need: 10_000, corner }, root, budget) {
        (Some(true), Some(c)) => c.last,
        _ => None,
    }
}

/// forcing_mate_of(fen, budget) -> (the mate's first move or None, [searches, proven, disproven, unknown, expansions]).
#[pyfunction]
pub fn forcing_mate_of(fen: String, budget: u64) -> (Option<String>, Vec<u64>) {
    let Some(p) = parse(&fen) else { return (None, vec![]) };
    crate::pns::STATS.iter().for_each(|a| { a.swap(0, Ordering::Relaxed); });
    let was = CORNERING.swap(true, Ordering::Relaxed);
    let m = forcing_mate(&p, budget).map(|m| m.to_uci(CastlingMode::Standard).to_string());
    CORNERING.store(was, Ordering::Relaxed);
    (m, crate::pns::STATS.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect())
}

pub fn forcing_gain(p: &Chess, budget: u64) -> Option<bool> {
    let key = (zkey(p), budget);
    if let Some(v) = FORCING_CACHE.with(|c| c.borrow().get(&key).copied()) { return v; }
    let v = crate::prof::timed("forcing_gain (PNS)", || forcing_gain_inner(p, budget));
    FORCING_CACHE.with(|c| c.borrow_mut().insert(key, v));
    if v.is_none() { UNKNOWNS.with(|u| { let mut u = u.borrow_mut(); if u.len() < 200 { u.push(shakmaty::fen::Fen::from_position(p.clone(), shakmaty::EnPassantMode::Legal).to_string()); } }); }
    v
}
// the positions where the forcing search hit its safety net (the first 200 since the last read) -- for digging
thread_local! { static UNKNOWNS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) }; }
/// forcing_unknowns() -> the FENs (side to move = the side whose forcing gain was asked) where the forcing search ran out.
#[pyfunction]
pub fn forcing_unknowns() -> Vec<String> { UNKNOWNS.with(|u| std::mem::take(&mut *u.borrow_mut())) }
fn forcing_gain_inner(p: &Chess, budget: u64) -> Option<bool> {
    if !RL_SAMPLE.load(Ordering::Relaxed) { return forcing_gain_plain(p, budget); }
    // an RL EPISODE: the stochastic policy's search (logged), then the truth from the full search
    EPISODE.with(|e| e.set(e.get() + 1));
    let e0 = crate::pns::STATS[4].load(Ordering::Relaxed);
    let v = forcing_gain_plain(p, budget);
    let used = crate::pns::STATS[4].load(Ordering::Relaxed) - e0;
    let was = POLICY_ON.swap(false, Ordering::Relaxed);
    let truth = forcing_gain_plain(p, 1000);
    POLICY_ON.store(was, Ordering::Relaxed);
    EP_OUT.with(|l| l.borrow_mut().push((EPISODE.with(|e| e.get()), verdict_code(v), verdict_code(truth), used)));
    v
}
fn forcing_gain_plain(p: &Chess, budget: u64) -> Option<bool> {
    let by = p.turn();
    let root = FState { pos: p.clone(), thr: rules::threats_arr(p, by), them: rules::threats_arr(p, !by), edge: material_edge(p, by), last: None };
    crate::pns::search(&Forcing { by, base: material_edge(p, by), need: 1, corner: false }, root, budget)
}

/// forcing_proof_of(fen, need, budget, max) -> (the verdict, the proof's lines in SAN, each ending with the leaf's standing:
/// the material banked and their static threats there) -- for digging a "wins by force".
#[pyfunction]
pub fn forcing_proof_of(fen: String, need: i64, budget: u64, max: usize) -> (Option<bool>, Vec<String>) {
    let Some(p) = parse(&fen) else { return (None, vec![]) };
    let by = p.turn();
    let base = material_edge(&p, by);
    let root = FState { pos: p.clone(), thr: rules::threats_arr(&p, by), them: rules::threats_arr(&p, !by), edge: base, last: None };
    let (r, lines) = crate::pns::search_proof_lines(&Forcing { by, base, need, corner: false }, root, budget, max);
    let out = lines.into_iter().map(|l| {
        let mut sans: Vec<String> = Vec::new();
        for w in l.windows(2) { if let Some(m) = &w[1].last { sans.push(crate::plan::san(&w[0].pos, &m.to_uci(CastlingMode::Standard).to_string())); } }
        let leaf = l.last().unwrap();
        let mut them: Vec<i16> = leaf.them.iter().copied().filter(|v| *v > 0).collect();
        them.sort_unstable_by(|a, b| b.cmp(a));
        format!("{}  [banked {}, their threats {:?}, our best capture {}{}]", sans.join(" "), leaf.edge - base, them,
            leaf.thr.iter().copied().max().unwrap_or(0), if leaf.pos.is_check() { ", in check" } else { "" })
    }).collect();
    (r, out)
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
pub fn forcing_value(p: &Chess, budget: u64) -> Option<(Move, i64)> { crate::prof::timed("forcing_value (PNS)", || forcing_value_inner(p, budget)) }
fn forcing_value_inner(p: &Chess, budget: u64) -> Option<(Move, i64)> {
    let by = p.turn();
    let base = material_edge(p, by);
    let root = FState { pos: p.clone(), thr: rules::threats_arr(p, by), them: rules::threats_arr(p, !by), edge: base, last: None };
    let mut best: Option<(Move, i64)> = None;
    let mut fail: Option<i64> = None;
    let prove = |need: i64| match crate::pns::search_with_proof(&Forcing { by, base, need, corner: false }, root.clone(), budget) {
        (Some(true), Some(c)) => c.last,
        _ => None,
    };
    for need in [1i64, 2, 3, 5, 9] {
        match prove(need) { Some(m) => best = Some((m, need)), None => { fail = Some(need); break; } }
    }
    // THE EXACT VALUE between the rungs (2026-10-12, the dig [16]: their d1=Q read as 5 -- the ladder's last proven rung --
    // where it wins 8): the gap to the first failing rung bisected (beyond 9: up to a queen and a rook)
    if EXACT_VALUE.load(Ordering::Relaxed) {
        if let Some((_, lo0)) = best.clone() {
            let (mut lo, mut hi) = (lo0, fail.unwrap_or(15));
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                match prove(mid) { Some(m) => { best = Some((m, mid)); lo = mid; } None => hi = mid }
            }
        }
    }
    best
}

/// IS THERE ANY NEW THREAT of `by` (static or forcing) against `before`? -- the cheapest first, stopping at the first
/// one found (material, promotion, trapped piece, mate by checks, gain by checks, a fork): a claim of "nothing left"
/// pays for everything, a refutation of it for the first fact that settles it.
/// -> Some(true) a new threat, Some(false) certainly none, None unknown (the forcing search's safety net was hit).
pub fn any_new_threat(pos: &Chess, by: Color, before: &[Threat]) -> Option<bool> { crate::prof::timed("any_new_threat", || any_new_threat_inner(pos, by, before)) }

/// THE ESCAPE TEST WITH ITS MAGNITUDE (Kaveh 2026-10-11: "a decision whether to reason about something should depend on
/// the gain in hand and the gain I am looking for ... the most this chain can win"): after the target's escape, can `by`
/// still force a gain of at least `need` -- a NEW static threat worth `need` or more (material, promotion, trapped, mate),
/// a fork worth it, or a forced line reaching it? A threat smaller than the need cannot carry the claim on its own; the
/// forcing search answers for the combinations. Some(false): certainly not; None: unknown (the safe side proves).
pub fn any_threat_reaching(pos: &Chess, by: Color, before: &[Threat], need: i64) -> Option<bool> {
    crate::prof::timed("any_threat_reaching", || {
        let need = need.max(1);
        let is_new = |t: &Threat| before.iter().find(|b| b.target == t.target && b.mate == t.mate).map_or(true, |b| t.value > b.value);
        for (x, g) in rules::threats_of(pos, by) {
            if g >= need && is_new(&Threat { target: Square::new(x as u32), value: g, mate: false, how: None }) { return Some(true); }
        }
        let Some(p) = (if pos.turn() == by { Some(pos.clone()) } else { pass(pos) }) else { return None };
        if threats(&p, by).iter().any(|t| t.value >= need && is_new(t)) { return Some(true); }
        if let Some((m, g)) = fork_next(&p) { if g >= need && is_new(&Threat { target: m.to(), value: g, mate: false, how: None }) { return Some(true); } }
        forcing_need(&p, need, FBUDGET.load(Ordering::Relaxed))
    })
}
pub static ESCAPE_NEED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_escape_need(flag: bool) -> PyResult<()> { ESCAPE_NEED.store(flag, Ordering::Relaxed); Ok(()) }
fn any_new_threat_inner(pos: &Chess, by: Color, before: &[Threat]) -> Option<bool> {
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


/// threat_list(fen, side) -> (the generic threats of `side` ("w"/"b"): [(target, value, mate)], the static exchange on
/// every piece of the other side: [(square, value)]) -- for digging.
#[pyfunction]
pub fn threat_list(fen: String, side: String) -> (Vec<(String, i64, bool)>, Vec<(String, i64)>) {
    let Some(pos) = parse(&fen) else { return (vec![], vec![]) };
    let by = if side == "w" { Color::White } else { Color::Black };
    let t = threats(&pos, by).into_iter().map(|t| (t.target.to_string(), t.value, t.mate)).collect();
    let e = pos.board().by_color(!by).into_iter().map(|s| (s.to_string(), rules::see(pos.board(), s, by))).collect();
    (t, e)
}
