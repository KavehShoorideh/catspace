//! PLAN 72.7 STEP 3: THE PYTHON WORDS IN RUST (docs/RUST_BOUNDARY.md section 4 step 3). Each word here is the port of a Python word of discovery/ (its docstring
//! there, with its Lean citation, is the specification and stays with the Python call-through); the Python body is kept as the TEST ORACLE in
//! tests/test_plan72_7_step3.py, compared on random positions and on the guard's. The words read the position on its handle (`Pos`, no FEN), in python-chess's
//! move order (`pos::py_order`, RUST_BOUNDARY risk 4) where a list order reaches an output, and return the structure Python formats (risk 3). Every pyo3 function
//! returns PyResult (risk 7); no rayon (risk 8); the dependency trackers are written exactly where the Python words wrote them (risk 2) -- by the vocabulary's
//! own reader (`vocab::read_word`, the word cache replaying what it recorded) and nowhere else.
use super::*;
use pyo3::exceptions::PyValueError;

fn err(s: &str) -> PyErr { PyValueError::new_err(s.to_string()) }
fn colour(white: bool) -> Color { if white { Color::White } else { Color::Black } }

/// A square named from the mover's side, as selfmine.view names it (black reads the ranks upside down).
fn view(s: Square, mover_white: bool) -> String { (if mover_white { s } else { s.flip_vertical() }).to_string() }

fn square_of(name: &str) -> PyResult<Square> { name.parse::<Square>().map_err(|_| err(&format!("no square {name}"))) }

// ------------------------------------------------------------------------------------------------------------------ 3a: the mate in two
// selfmine.bind_mate2 (plan 49; Lean `Ideas.Mined.Schemas.mate_in_two`): a move x -> y of the mover after which every reply is met by a mate, read as the PREMISE
// the binder writes for it -- the tail `(after x y (least (most (min check (eq movecount 0)))))` >= 1 and its three legality legs (selfmine.legs_of: the piece on x
// can legally go to y; a piece of ours stands on x; no pawn of ours on x about to promote, read <= 0) -- each term compiled from the s-expression Python renders
// for it (selfmine.render: squares named from the mover's side; the no-pawn leg's rank `a7` from either side) and read by `vocab::read_word` as `Pos.values`
// reads it (the word cache included). The candidates in python-chess's order, promotions and castling left out, stopping at the FIRST that binds.

/// (installation of the compositions, x, y, mover) -> the compiled ids [legal leg, ours leg, no-pawn leg, tail].
fn mate2_ids_cache() -> &'static Mutex<HashMap<(u64, u8, u8, bool), [u32; 4]>> {
    static C: OnceLock<Mutex<HashMap<(u64, u8, u8, bool), [u32; 4]>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn mate2_ids(x: Square, y: Square, mover_white: bool) -> PyResult<[u32; 4]> {
    let key = (crate::vocab::macro_gen(), x as u8, y as u8, mover_white);
    if let Some(ids) = mate2_ids_cache().lock().unwrap().get(&key) { return Ok(*ids); }
    let (vx, vy) = (view(x, mover_white), view(y, mover_white));
    let ids = [crate::vocab::compile_term(&format!("(count (and (moves {vx}) (at {vy})))"))?,
               crate::vocab::compile_term(&format!("(count (and ours (at {vx})))"))?,
               crate::vocab::compile_term(&format!("(count (and ourpawns (and (at {vx}) (rank a7))))"))?,
               crate::vocab::compile_term(&format!("(after {vx} {vy} (least (most (min check (eq movecount 0)))))"))?];
    mate2_ids_cache().lock().unwrap().insert(key, ids);
    Ok(ids)
}

/// Does the premise of `mate_in_two` at x -> y hold on `p` as it stands (`selfmine.premise_truth` of the binder's atoms)? Every atom is read (the legs, then the
/// tail), as `Pos.values` read them in one crossing.
fn mate2_holds(p: &Chess, x: Square, y: Square, mover_white: bool) -> PyResult<bool> {
    let ids = mate2_ids(x, y, mover_white)?;
    let mut v = [0u32; 4];
    for (i, id) in ids.iter().enumerate() {
        let c = crate::vocab::compiled(*id)?;
        v[i] = crate::vocab::read_word(p, *id, &c, false);
    }
    Ok(v[0] >= 1 && v[1] >= 1 && v[2] == 0 && v[3] >= 1)
}

/// The candidates `bind_mate2` generates: every legal move but promotions and castling, python-chess's order, each (from, to) once.
fn mate2_candidates(p: &Chess) -> Vec<(Square, Square)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = vec![];
    for m in crate::pos::py_order(p) {
        if m.promotion().is_some() || m.is_castle() { continue; }
        let pr = (m.from().unwrap_or(m.to()), m.to());
        if seen.insert(pr) { out.push(pr); }
    }
    out
}

/// The first candidate (its index in the list read, x, y) that binds `mate_in_two` on `p`.
fn mate2_first(p: &Chess, mover_white: bool, cands: &[(Square, Square)]) -> PyResult<Option<(usize, Square, Square)>> {
    for (i, &(x, y)) in cands.iter().enumerate() {
        if mate2_holds(p, x, y, mover_white)? { return Ok(Some((i, x, y))); }
    }
    Ok(None)
}

/// bind_mate2_pos(pos, mover_white, candidates=None) -> (index, x, y) of the FIRST candidate that binds `mate_in_two` on the handle as it stands (the squares named
/// as the binder names them, absolute), or None. `candidates`: [(x, y)] absolute names (default: the binder's own -- every legal move but promotions and castling,
/// python-chess's order). ValueError on a square name that is not one.
#[pyfunction]
#[pyo3(signature = (pos, mover_white, candidates=None))]
pub fn bind_mate2_pos(pos: &crate::pos::Pos, mover_white: bool, candidates: Option<Vec<(String, String)>>) -> PyResult<Option<(usize, String, String)>> {
    let cands = match candidates {
        Some(v) => v.iter().map(|(x, y)| Ok((square_of(x)?, square_of(y)?))).collect::<PyResult<Vec<_>>>()?,
        None => mate2_candidates(&pos.pos),
    };
    Ok(mate2_first(&pos.pos, mover_white, &cands)?.map(|(i, x, y)| (i, x.to_string(), y.to_string())))
}

/// regress.mate2_threat after our move: `c` = the board after it (their turn). False when it checks (no pass is legal), when we would have no legal move after a
/// pass, or when the board after the pass is no position shakmaty holds (the FEN reader read nothing there); else whether `mate_in_two` binds for us after the pass.
fn mate2_after(c: &Chess, me: bool) -> PyResult<bool> {
    if c.is_check() { return Ok(false); }
    let Some(w) = crate::pos::with_turn_strict(c, colour(me)) else { return Ok(false) };
    if w.legal_moves().is_empty() { return Ok(false); }
    let cands = mate2_candidates(&w);
    Ok(mate2_first(&*w, me, &cands)?.is_some())
}

/// mate2_threat_pos(pos, uci, me_white) -> regress.mate2_threat: after our legal move `uci` and a pass, a forced mate in two for us (`mate_in_two` binds).
/// ValueError on an illegal move.
#[pyfunction]
pub fn mate2_threat_pos(pos: &crate::pos::Pos, uci: &str, me_white: bool) -> PyResult<bool> {
    let c = pos.push_move(uci)?;
    mate2_after(&c.pos, me_white)
}

/// mate2_after_pos(pos, me_white) -> `mate2_threat` read on the board AFTER our move (the handle `pos`, their turn): for a board whose own handle shakmaty does
/// not hold before the move.
#[pyfunction]
pub fn mate2_after_pos(pos: &crate::pos::Pos, me_white: bool) -> PyResult<bool> { mate2_after(&pos.pos, me_white) }

// ------------------------------------------------------------------------------------------------------------------ 3b: the roll forward and the static walks
// regress.reach_map / reach_price (plan 72.1, Lean `Ideas.Reach.reachPrice`, `Ideas.Vocabulary.attackableIn` = its price-0 case by `Ideas.Reach.reachPrice_zero_iff`),
// race.stopper_walk (plan 62.1b, `Race.deny_sound`), graph.piece_moves_to (the `reach` of a slider / knight). The roll READS python-chess's boards as python-chess
// holds them: after a quiet move that gives check the roll goes on from `with_turn(a, own)`, a board whose side NOT to move is in check -- a position shakmaty
// refuses -- so the roll keeps a board as its placement and its side to move and generates the one unit's moves itself, by python-chess's rules
// (`generate_legal_moves`: `_slider_blockers`, `_generate_evasions`, `_is_safe`, the occupancy as it stands), in python-chess's order.

/// The eight bitboards of a python-chess board (pawns, knights, bishops, rooks, queens, kings, white, black) as a shakmaty Board; ValueError when inconsistent.
pub(crate) fn board_of(bbs: [u64; 8]) -> PyResult<shakmaty::Board> {
    let roles = &bbs[..6];
    let union = roles.iter().fold(0u64, |a, r| a | r);
    if roles.iter().map(|r| r.count_ones()).sum::<u32>() != union.count_ones() || union != (bbs[6] | bbs[7]) || bbs[6] & bbs[7] != 0 {
        return Err(err("inconsistent bitboards"));
    }
    let b = shakmaty::Bitboard;
    Ok(shakmaty::Board::from_bitboards(shakmaty::ByRole { pawn: b(bbs[0]), knight: b(bbs[1]), bishop: b(bbs[2]), rook: b(bbs[3]), queen: b(bbs[4]), king: b(bbs[5]) },
                                       shakmaty::ByColor { white: b(bbs[6]), black: b(bbs[7]) }))
}

fn sq_of(i: u8) -> PyResult<Square> { if i < 64 { Ok(Square::new(i as u32)) } else { Err(err(&format!("no square {i} (squares are 0..63, a1 = 0)"))) } }

/// python-chess `_slider_blockers(king)` for the side `us`: our pieces that are the only piece between our king and a slider of theirs aimed at it (the empty-board line).
fn slider_blockers(b: &shakmaty::Board, us: Color, king: Square) -> shakmaty::Bitboard {
    use shakmaty::{attacks, Bitboard};
    let snipers = (attacks::rook_attacks(king, Bitboard::EMPTY) & b.rooks_and_queens()) | (attacks::bishop_attacks(king, Bitboard::EMPTY) & b.bishops_and_queens());
    let mut blockers = Bitboard::EMPTY;
    for sniper in snipers & b.by_color(!us) {
        let bb = attacks::between(king, sniper) & b.occupied();
        if bb.count() == 1 { blockers |= bb; }
    }
    blockers & b.by_color(us)
}

const RANK_3: u64 = 0xFF << 16;
const RANK_4: u64 = 0xFF << 24;
const RANK_5: u64 = 0xFF << 32;
const RANK_6: u64 = 0xFF << 40;
const BACK_RANKS: u64 = 0xFF | (0xFF << 56);

/// The QUIET legal moves of the one unit on `at` (the side `us` to move on `b`), python-chess's order and legality (`generate_legal_moves(from_mask=at)`), the roll's
/// filter applied (no capture, no en passant, no castling, a promotion to a queen only): [(to, promotes)]. The board may hold the side not to move in check.
fn unit_quiet_moves(b: &shakmaty::Board, us: Color, at: Square) -> Vec<(Square, bool)> {
    use shakmaty::{attacks, Bitboard};
    let Some(pc) = b.piece_at(at) else { return vec![] };
    if pc.color != us { return vec![]; }
    let (our, occ) = (b.by_color(us), b.occupied());
    // generate_pseudo_legal_moves(from_mask=at, to_mask=target), quiet only: a non-pawn to-square descending; a pawn's single push then its double push (the double
    // computed from the single push BEFORE the target mask, as python-chess does)
    let pseudo = |target: Bitboard, from_ok: bool| -> Vec<(Square, bool)> {
        let mut v = vec![];
        if !from_ok { return v; }
        if pc.role != Role::Pawn {
            for to in (b.attacks_from(at) & !our & !occ & target).into_iter().rev() { v.push((to, false)); }
        } else {
            let sb = Bitboard::from_square(at).0;
            let (single, double) = if us == Color::White {
                let s = (sb << 8) & !occ.0;
                (s, (s << 8) & !occ.0 & (RANK_3 | RANK_4))
            } else {
                let s = (sb >> 8) & !occ.0;
                (s, (s >> 8) & !occ.0 & (RANK_6 | RANK_5))
            };
            for to in (Bitboard(single) & target).into_iter().rev() { v.push((to, BACK_RANKS & (1u64 << (to as u32)) != 0)); }
            for to in (Bitboard(double) & target).into_iter().rev() { v.push((to, false)); }
        }
        v
    };
    let kings = b.kings() & our;
    let Some(king) = kings.last() else { return pseudo(Bitboard::FULL, true) };
    let blockers = slider_blockers(b, us, king);
    let checkers = b.attacks_to(king, !us, occ);
    let gen = if checkers.any() {
        let mut v = vec![];
        let mut attacked = Bitboard::EMPTY;
        for c in checkers & (b.bishops() | b.rooks() | b.queens()) { attacked |= attacks::ray(king, c) & !Bitboard::from_square(c); }
        if at == king {
            for to in (attacks::king_attacks(king) & !our & !attacked & !occ).into_iter().rev() { v.push((to, false)); }
        }
        if checkers.count() == 1 {
            let checker = checkers.last().unwrap();
            v.extend(pseudo(attacks::between(king, checker) | checkers, !b.kings().contains(at)));
        }
        v
    } else {
        pseudo(Bitboard::FULL, true)
    };
    gen.into_iter().filter(|&(to, _)| {
        if at == king { b.attacks_to(to, !us, occ).is_empty() }                       // _is_safe: the king's square not attacked, the occupancy as it stands
        else { !blockers.contains(at) || attacks::ray(at, to).contains(king) }       // ... a pinned piece moves along its pin line only
    }).collect()
}

/// `graph.see_for(a, sq, side)` on a roll board (`side` to move after `with_turn`): the word `(see sq)` read on the handle of that board, 0 when shakmaty does not hold it
/// (`graph.pos_of` -> None -> `graph.see` reads 0). The castling rights are none (python-chess's clean rights or none: the same boards hold; `see` reads the board only).
fn see_on(b: &shakmaty::Board, side: Color, sq: Square) -> u32 {
    let mut s = shakmaty::Setup::empty();
    s.board = b.clone();
    s.turn = side;
    match Chess::from_setup(s, CastlingMode::Standard) {
        Ok(p) => {
            let named = if side == Color::White { sq } else { sq.flip_vertical() };
            crate::vocab::num(&crate::vocab::Term::See(Box::new(crate::vocab::Term::Named(named))), &p, !side)
        }
        Err(_) => 0,
    }
}

/// One landing of the roll: the square, (material, tempi), the path (UCI), the board after the landing (the other side to move).
#[derive(Clone)]
pub(crate) struct Landing { pub to: Square, pub material: u32, pub tempi: u32, pub path: Vec<String>, pub board: shakmaty::Board }

type RollKey = ([u64; 8], u8, u32, Option<(u8, bool)>);
fn roll_cache() -> &'static Mutex<HashMap<RollKey, std::sync::Arc<Vec<Landing>>>> {
    static C: OnceLock<Mutex<HashMap<RollKey, std::sync::Arc<Vec<Landing>>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// reach_cache_clear(): the roll's cache emptied -- registered in graph.CLEAR_HOOKS beside regress's `_REACH` (the same key: the placement, the square, k, the job).
#[pyfunction]
pub fn reach_cache_clear() -> PyResult<()> { roll_cache().lock().unwrap().clear(); Ok(()) }

/// reach_cache_size() -> how many rolls are kept.
#[pyfunction]
pub fn reach_cache_size() -> PyResult<usize> { Ok(roll_cache().lock().unwrap().len()) }

fn bbs_of(b: &shakmaty::Board) -> [u64; 8] {
    [b.pawns().0, b.knights().0, b.bishops().0, b.rooks().0, b.queens().0, b.kings().0, b.white().0, b.black().0]
}

/// regress.reach_map on a placement (see its docstring): the landings in the ORDER the Python dict held them (a landing's first entry fixes its place; a cheaper path
/// replaces the entry where it stands). Memoised by (the placement, the square, k, the job), as `_REACH` is.
pub(crate) fn roll(b: &shakmaty::Board, sq: Square, k: u32, job: Option<(Square, bool)>) -> std::sync::Arc<Vec<Landing>> {
    let key: RollKey = (bbs_of(b), sq as u8, k, job.map(|(t, x)| (t as u8, x)));
    if let Some(hit) = roll_cache().lock().unwrap().get(&key) { return hit.clone(); }
    let out = std::sync::Arc::new(roll_fresh(b, sq, k, job));
    roll_cache().lock().unwrap().insert(key, out.clone());
    out
}

fn roll_fresh(b0: &shakmaty::Board, sq: Square, k: u32, job: Option<(Square, bool)>) -> Vec<Landing> {
    let Some(pc) = b0.piece_at(sq) else { return vec![] };
    let own = pc.color;
    let mut out: Vec<Landing> = vec![];
    let mut place: [Option<usize>; 64] = [None; 64];
    let mut seen: [Option<u32>; 64] = [None; 64];
    seen[sq as usize] = Some(0);
    let mut frontier: Vec<(shakmaty::Board, Vec<String>, Square)> = vec![(b0.clone(), vec![], sq)];
    for depth in 1..=k {
        let mut nxt = vec![];
        for (b, path, at) in &frontier {
            let Some(mover) = b.piece_at(*at) else { continue };
            for (to, promotes) in unit_quiet_moves(b, own, *at) {
                let mut a = b.clone();
                a.discard_piece_at(*at);
                a.set_piece_at(to, if promotes { Piece { color: own, role: Role::Queen } } else { mover });
                if let Some((t, bearing)) = job {
                    if depth == k && !(if bearing { a.attacks_from(to).contains(t) } else { to == t }) { continue; }
                }
                let price = if pc.role == Role::King || a.attacks_to(to, !own, a.occupied()).is_empty() { 0 } else { see_on(&a, !own, to) };
                let mut p2 = path.clone();
                p2.push(format!("{}{}{}", at, to, if promotes { "q" } else { "" }));
                match place[to as usize] {
                    None => { place[to as usize] = Some(out.len()); out.push(Landing { to, material: price, tempi: depth, path: p2.clone(), board: a.clone() }); }
                    Some(i) => { if (price, depth) < (out[i].material, out[i].tempi) { out[i] = Landing { to, material: price, tempi: depth, path: p2.clone(), board: a.clone() }; } }
                }
                if price > 0 || depth == k { continue; }
                if let Some(d) = seen[to as usize] { if d <= depth { continue; } }
                seen[to as usize] = Some(depth);
                nxt.push((a, p2, to));
            }
        }
        frontier = nxt;
        if frontier.is_empty() { break; }
    }
    out
}

/// reach_map_bb(bbs, sq, k, job=None) -> [(landing, material, tempi, path [UCI], the unit's attacks from its landing as a mask)] in the order of regress.reach_map's dict.
/// `bbs` = the board's eight bitboards; `job` = (target, bearing). ValueError on inconsistent bitboards or a square out of range.
#[pyfunction]
#[pyo3(signature = (bbs, sq, k, job=None))]
pub fn reach_map_bb(bbs: [u64; 8], sq: u8, k: u32, job: Option<(u8, bool)>) -> PyResult<Vec<(u8, u32, u32, Vec<String>, u64)>> {
    let b = board_of(bbs)?;
    let job = match job { Some((t, x)) => Some((sq_of(t)?, x)), None => None };
    Ok(roll(&b, sq_of(sq)?, k, job).iter().map(|e| (e.to as u8, e.material, e.tempi, e.path.clone(), e.board.attacks_from(e.to).0)).collect())
}

/// regress.reach_price on a placement: the unit on `sq` comes to bear on `target` (bearing) or lands on it -> (material, tempi, first UCI or None, path, reads [squares])
/// or None (see its docstring). The cheapest landing by (material, tempi), the first in the roll's order on a tie.
pub(crate) fn reach_price_on(b: &shakmaty::Board, sq: Square, target: Square, k: u32, bearing: bool) -> Option<(u32, u32, Option<String>, Vec<String>, Vec<u8>)> {
    b.piece_at(sq)?;
    if target == sq { return None; }
    let mut reads: Vec<u8> = vec![sq as u8, target as u8];
    if bearing && b.attacks_from(sq).contains(target) {
        reads.sort_unstable();
        reads.dedup();
        return Some((0, 0, None, vec![], reads));
    }
    let mp = roll(b, sq, k, Some((target, bearing)));
    let mut best: Option<&Landing> = None;
    for e in mp.iter() {
        let does = if !bearing { e.to == target } else { e.board.attacks_from(e.to).contains(target) };
        if does && best.map_or(true, |x| (e.material, e.tempi) < (x.material, x.tempi)) { best = Some(e); }
    }
    let best = best?;
    for u in &best.path {
        let (f, t) = (u[0..2].parse::<Square>().ok()?, u[2..4].parse::<Square>().ok()?);
        reads.push(t as u8);
        reads.extend(shakmaty::attacks::between(f, t).into_iter().map(|s| s as u8));
    }
    if bearing {
        let t = best.path.last()?[2..4].parse::<Square>().ok()?;
        reads.extend(shakmaty::attacks::between(t, target).into_iter().map(|s| s as u8));
    }
    reads.sort_unstable();
    reads.dedup();
    Some((best.material, best.tempi, best.path.first().cloned(), best.path.clone(), reads))
}

/// reach_price_bb(bbs, sq, target, k, bearing=True) -> (material, tempi, first, path, reads) or None: regress.reach_price on the board's placement.
#[pyfunction]
#[pyo3(signature = (bbs, sq, target, k, bearing=true))]
pub fn reach_price_bb(bbs: [u64; 8], sq: u8, target: u8, k: u32, bearing: bool) -> PyResult<Option<(u32, u32, Option<String>, Vec<String>, Vec<u32>)>> {
    Ok(reach_price_on(&board_of(bbs)?, sq_of(sq)?, sq_of(target)?, k, bearing).map(|(m, t, f, p, r)| (m, t, f, p, r.into_iter().map(u32::from).collect())))
}

/// race.stopper_walk on a placement (see its docstring): the shortest walk of the piece on `s` to a square of `targets` within `limit` moves over the squares not in
/// `deny`, the other pieces standing; each step's squares in ascending order (python-chess's SquareSet order). -> (moves, [the squares it lands on]) or None.
pub(crate) fn stopper_walk_on(b: &shakmaty::Board, s: Square, targets: u64, limit: u32, deny: u64) -> Option<(u32, Vec<u8>)> {
    let pc = b.piece_at(s)?;
    let mut seen: u64 = 1 << (s as u32);
    let mut frontier: Vec<(Square, Vec<u8>)> = vec![(s, vec![])];
    for k in 1..=limit {
        let mut nxt = vec![];
        for (sq, path) in &frontier {
            let cand = if pc.role == Role::King { shakmaty::attacks::king_attacks(*sq) } else {
                shakmaty::attacks::attacks(*sq, pc, (b.occupied() & !shakmaty::Bitboard::from_square(s)) | shakmaty::Bitboard::from_square(*sq))
            };
            for t in cand {
                let bit = 1u64 << (t as u32);
                if seen & bit != 0 { continue; }
                if let Some(o) = b.piece_at(t) { if o.color == pc.color && t != s { continue; } }
                if deny & bit != 0 { continue; }
                let mut p2 = path.clone();
                p2.push(t as u8);
                if targets & bit != 0 { return Some((k, p2)); }
                seen |= bit;
                nxt.push((t, p2));
            }
        }
        frontier = nxt;
    }
    None
}

/// stopper_walk_bb(bbs, s, targets, limit, deny=0) -> (moves, squares) or None: race.stopper_walk on the board's placement (`targets` a mask).
#[pyfunction]
#[pyo3(signature = (bbs, s, targets, limit, deny=0))]
pub fn stopper_walk_bb(bbs: [u64; 8], s: u8, targets: u64, limit: u32, deny: u64) -> PyResult<Option<(u32, Vec<u32>)>> {
    Ok(stopper_walk_on(&board_of(bbs)?, sq_of(s)?, targets, limit, deny).map(|(k, p)| (k, p.into_iter().map(u32::from).collect())))      // (squares as a list: pyo3 hands a Vec<u8> over as bytes)
}

/// graph.piece_moves_to on a placement (see its docstring): the fewest moves the piece on `frm` needs to stand on a square of `targets` (a mask), the other pieces
/// standing, own-colour squares not entered, within `limit`; None beyond. Each step's squares ascending.
pub(crate) fn piece_moves_to_on(b: &shakmaty::Board, frm: Square, targets: u64, limit: u32) -> Option<u32> {
    let pc = b.piece_at(frm)?;
    let mut seen: u64 = 1 << (frm as u32);
    let mut frontier = vec![frm];
    for k in 1..=limit {
        let mut nxt = vec![];
        for s in &frontier {
            let occ = (b.occupied() & !shakmaty::Bitboard::from_square(frm)) | shakmaty::Bitboard::from_square(*s);
            for t in shakmaty::attacks::attacks(*s, pc, occ) {
                let bit = 1u64 << (t as u32);
                let own = b.piece_at(t).map_or(false, |o| o.color == pc.color && t != frm);
                if seen & bit != 0 || own { continue; }
                if targets & bit != 0 { return Some(k); }
                seen |= bit;
                nxt.push(t);
            }
        }
        frontier = nxt;
    }
    None
}

/// piece_moves_to_bb(bbs, frm, targets, limit) -> moves or None: graph.piece_moves_to on the board's placement (`targets` a mask).
#[pyfunction]
pub fn piece_moves_to_bb(bbs: [u64; 8], frm: u8, targets: u64, limit: u32) -> PyResult<Option<u32>> {
    Ok(piece_moves_to_on(&board_of(bbs)?, sq_of(frm)?, targets, limit))
}

/// python-chess's `push` on a placement (the board only, standard chess): the piece leaves `from` and stands on `to` (a capture removes what stood there; a promotion
/// puts the new piece); castling -- the king two files over -- brings the corner rook to the square the king crossed; a pawn's diagonal step to an empty square takes
/// the pawn beside it (en passant).
pub(crate) fn push_placement(b: &shakmaty::Board, from: Square, to: Square, promo: Option<Role>) -> shakmaty::Board {
    let mut a = b.clone();
    let Some(pc) = a.piece_at(from) else { return a };
    a.discard_piece_at(from);
    if pc.role == Role::King && (u32::from(from.file()) as i32 - u32::from(to.file()) as i32).abs() == 2 {
        let (rf, rt) = if to.file() > from.file() { (shakmaty::File::H, shakmaty::File::F) } else { (shakmaty::File::A, shakmaty::File::D) };
        let (rs, rd) = (Square::from_coords(rf, from.rank()), Square::from_coords(rt, from.rank()));
        if let Some(r) = a.piece_at(rs) { a.discard_piece_at(rs); a.set_piece_at(rd, r); }
    }
    if pc.role == Role::Pawn && from.file() != to.file() && a.piece_at(to).is_none() {
        a.discard_piece_at(Square::from_coords(to.file(), from.rank()));
    }
    a.set_piece_at(to, match promo { Some(r) => Piece { color: pc.color, role: r }, None => pc });
    a
}

/// A UCI move's squares and promotion (no legality: the caller hands python-chess's legal moves). ValueError when it does not parse.
fn uci_parts(u: &str) -> PyResult<(Square, Square, Option<Role>)> {
    if u.len() < 4 { return Err(err(&format!("no move {u}"))); }
    let (f, t) = (square_of(&u[0..2])?, square_of(&u[2..4])?);
    let p = match u.as_bytes().get(4) { None => None, Some(b'q') => Some(Role::Queen), Some(b'r') => Some(Role::Rook), Some(b'b') => Some(Role::Bishop), Some(b'n') => Some(Role::Knight),
                                        Some(_) => return Err(err(&format!("no move {u}"))) };
    Ok((f, t, p))
}

/// race.denial_mask on a placement (see its docstring): the squares the stopper on `stopper` cannot stand and survive on -- for a KING every square `me` attacks with the
/// king taken off (the passer `skip` not counted); for a knight, bishop, rook or queen the squares our pawns but the passer attack; else none.
fn denial_mask_on(a: &shakmaty::Board, me: Color, stopper: Square, skip: Square) -> u64 {
    let Some(pc) = a.piece_at(stopper) else { return 0 };
    let mut mask = 0u64;
    if pc.role == Role::King {
        let mut c = a.clone();
        c.discard_piece_at(stopper);
        for t in c.by_color(me) { if t != skip { mask |= c.attacks_from(t).0; } }
        return mask;
    }
    if matches!(pc.role, Role::Knight | Role::Bishop | Role::Rook | Role::Queen) {
        for t in a.pawns() & a.by_color(me) { if t != skip { mask |= a.attacks_from(t).0; } }
    }
    mask
}

/// denial_mask_bb(bbs, me_white, stopper, skip) -> race.denial_mask on the board's placement (a mask). ValueError on inconsistent bitboards or a square out of range.
#[pyfunction]
pub fn denial_mask_bb(bbs: [u64; 8], me_white: bool, stopper: u8, skip: u8) -> PyResult<u64> {
    Ok(denial_mask_on(&board_of(bbs)?, colour(me_white), sq_of(stopper)?, sq_of(skip)?))
}

/// deny_moves_bb(bbs, me_white, sq, stopper, targets, n, legal) -> [(index into `legal`, why)]: race.deny_moves on the board's placement (see its docstring) -- the moves
/// of ours (not the passer's own, `legal` = python-chess's legal moves as UCI in its order) after which the stopper's walk to `targets` (a mask) no longer exists within
/// n + 1 moves over the squares it can stand on. ValueError on inconsistent bitboards, a square out of range or a move that does not parse.
#[pyfunction]
pub fn deny_moves_bb(bbs: [u64; 8], me_white: bool, sq: u8, stopper: u8, targets: u64, n: u32, legal: Vec<String>) -> PyResult<Vec<(usize, String)>> {
    let b = board_of(bbs)?;
    let (me, sq, stopper) = (colour(me_white), sq_of(sq)?, sq_of(stopper)?);
    let Some((_, before)) = stopper_walk_on(&b, stopper, targets, n, 0) else { return Ok(vec![]) };
    let pc = b.piece_at(stopper).ok_or_else(|| err("no stopper"))?;
    let name = format!("{}{}", pc.role.upper_char(), stopper);
    let path = before.iter().map(|&x| Square::new(x as u32).to_string()).collect::<Vec<_>>().join(" ");
    let mut out = vec![];
    for (i, u) in legal.iter().enumerate() {
        let (f, t, p) = uci_parts(u)?;
        if f == sq || t == stopper { continue; }
        let a = push_placement(&b, f, t, p);
        let Some(sp) = a.piece_at(stopper) else { continue };
        if a.attacks_from(stopper).contains(t) && !(sp.role == Role::King && a.attacks_to(t, me, a.occupied()).any()) { continue; }
        if stopper_walk_on(&a, stopper, targets, n + 1, denial_mask_on(&a, me, stopper, sq)).is_none() {
            out.push((i, format!("denies the path of their {name} ({path}): it cannot stand on a square of it in {} moves", n + 1)));
        }
    }
    Ok(out)
}
