//! PLAN 54 STAGE B: THE FIX FILTER IN RUST. Ports of regress.py's `mate_defects` / `material_defects` (the counterfactual of a question), `_still` (the candidate-list
//! filter of `routes_of`) and `forcing_why` (check, mate threat, capture or threat worth the stake, `landsSafe`, the sacrifice that still threatens), with the SAME
//! results as the Python twins -- orders included. python-chess lists legal moves in its own order (king first when in check, then the non-pawns by from-square
//! descending, the pawn captures, the single pushes, the double pushes) and the defect lists inherit that order, so `py_key` reproduces it. What stays in Python:
//! the SAN in a note (shakmaty's SAN is not python-chess's: the caller gets the capture's UCI and writes it), `pass_claim` (the Lean passer prototype), and the
//! generation of the candidate list.
//! A position that shakmaty cannot hold even leniently (a missing king of ours, pawns on the back rank ...) raises ValueError: the Python wrapper keeps the old path.
use super::*;
use pyo3::exceptions::PyValueError;
use shakmaty::{attacks, Bitboard, Board, EnPassantMode, Setup};
use std::collections::BTreeMap;

const STAKE_MATE: i64 = 100;
const STAKE_PROMOTE: i64 = 8;

type Def = (String, Option<u8>, Vec<u8>, String);
type Placement = (char, u8, Option<u8>);

fn err(s: &str) -> PyErr { PyValueError::new_err(s.to_string()) }
fn nm(s: Square) -> String { s.to_string() }
fn letter(p: Piece) -> char { p.role.upper_char() }
fn col(white: bool) -> Color { if white { Color::White } else { Color::Black } }
fn worth(r: Role) -> i64 { crate::rules::val(r) }

/// A position from a board and a side to move: no castling, no en passant (a counterfactual has no history); strict = what the FEN reader would accept.
fn from_board(b: &Board, turn: Color, lenient: bool) -> Option<Chess> {
    let mut s = Setup::empty();
    s.board = b.clone();
    s.turn = turn;
    let r = Chess::from_setup(s, CastlingMode::Standard);
    if lenient {
        r.or_else(|e| e.ignore_impossible_check()).or_else(|e| e.ignore_too_much_material()).ok()
    } else {
        r.ok()
    }
}

/// `G.with_turn`: the same position with `turn` to move, no en passant square (castling rights kept). None when the result is not a legal position.
fn flip(p: &Chess, turn: Color) -> Option<Chess> {
    let mut s = p.clone().into_setup(EnPassantMode::Always);
    s.turn = turn;
    s.ep_square = None;
    Chess::from_setup(s, CastlingMode::Standard).ok()
}

fn apply(b: &mut Board, pl: &[Placement]) -> Result<(), PyErr> {
    for &(c, at, frm) in pl {
        let piece = Piece::from_char(c).ok_or_else(|| err("bad piece symbol"))?;
        if let Some(f) = frm { b.discard_piece_at(Square::new(f as u32)); }
        b.set_piece_at(Square::new(at as u32), piece);
    }
    Ok(())
}

fn sorted(bb: Bitboard) -> Vec<Square> { let mut v: Vec<Square> = bb.into_iter().collect(); v.sort_by_key(|s| *s as u32); v }

/// python-chess's order of the legal moves of a side (not in check: pseudo-legal order; in check: the evasions' order -- the same, with the king's moves first).
/// (king-first flag, category, k1, k2, promotion rank): category 0 = non-pawn, 1 = pawn capture, 2 = single push, 3 = double push, 4 = en passant.
fn py_key(m: &Move, board: &Board, evasion: bool) -> (u8, u8, i32, i32, u8) {
    let from = m.from().map(|s| s as i32).unwrap_or(0);
    let to = m.to() as i32;
    let king = evasion && m.role() == Role::King;
    let promo = match m.promotion() { Some(Role::Queen) => 0, Some(Role::Rook) => 1, Some(Role::Bishop) => 2, Some(Role::Knight) => 3, _ => 0 };
    let kf: u8 = if king { 0 } else { 1 };
    if m.role() != Role::Pawn { return (kf, 0, -from, -to, 0); }
    let _ = board;
    if m.is_en_passant() { (kf, 4, -from, -to, 0) }
    else if m.is_capture() { (kf, 1, -from, -to, promo) }
    else if (to - from).abs() == 8 { (kf, 2, -to, 0, promo) }
    else { (kf, 3, -to, 0, 0) }
}

fn legal_sorted(p: &Chess, evasion: bool) -> Vec<Move> {
    let mut v: Vec<Move> = p.legal_moves().into_iter().collect();
    let b = p.board();
    v.sort_by_key(|m| py_key(m, b, evasion));
    v
}

/// `mate_defects` on a board with the placements made and THEIR move: (kind, sq, by, why) in the Python order.
fn mate_defects_board(base: &Board, me: bool, pl: &[Placement]) -> Result<Vec<Def>, PyErr> {
    let mut b = base.clone();
    apply(&mut b, pl)?;
    let (me_c, them) = (col(me), col(!me));
    let Some(ksq) = b.king_of(them) else { return Ok(vec![("no check".into(), None, vec![], "their king is not on the board".into())]) };
    let checkers = b.attacks_to(ksq, me_c, b.occupied()) & b.by_color(me_c);
    if checkers.is_empty() {
        return Ok(vec![("no check".into(), Some(ksq as u8), vec![], format!("nothing of ours attacks {}", nm(ksq)))]);
    }
    let pos = from_board(&b, them, true).ok_or_else(|| err("mate_defects: the counterfactual is not a position"))?;
    let legal = legal_sorted(&pos, true);
    let mut out: Vec<Def> = vec![];
    for c in sorted(checkers) {
        for m in &legal {
            if m.to() == c && !m.is_castle() {
                let f = m.from().unwrap();
                let pc = b.piece_at(f).unwrap();
                out.push(("checker takeable".into(), Some(c as u8), vec![f as u8], format!("{} on {} can take the checker on {}", letter(pc), nm(f), nm(c))));
            }
        }
    }
    for m in &legal {
        if !m.is_castle() && m.from() == Some(ksq) && !checkers.contains(m.to()) {
            out.push(("flight".into(), Some(m.to() as u8), vec![ksq as u8], format!("the king can go to {}", nm(m.to()))));
        }
    }
    if checkers.count() == 1 {
        let c = checkers.first().unwrap();
        let between = attacks::between(c, ksq);
        let mut by_sq: BTreeMap<u8, Vec<u8>> = BTreeMap::new();
        for m in &legal {
            if !m.is_castle() && m.from() != Some(ksq) && between.contains(m.to()) { by_sq.entry(m.to() as u8).or_default().push(m.from().unwrap() as u8); }
        }
        for (sq, frm) in by_sq {
            out.push(("interposition".into(), Some(sq), frm, format!("a piece can step between on {}", nm(Square::new(sq as u32)))));
        }
    }
    Ok(out)
}

/// `G.see_for(board, t, me) > 0` on a (real) board: the word `see` read with `me` on move; 0 when with_turn(board, me) is not a legal position (the evaluator reads nothing).
fn see_for(p: &Chess, t: Square, me: Color) -> i64 {
    let b = p.board();
    if flip(p, me).is_none() { return 0; }
    match b.piece_at(t) { Some(pc) => crate::rules::see(b, t, !pc.color).max(0), None => 0 }
}

/// regress.material_defects on a board (a real position): [] when the exchange wins for us.
fn material_defects_pos(p: &Chess, me: bool, target: u8) -> Vec<Def> {
    let (b, me_c, t) = (p.board(), col(me), Square::new(target as u32));
    if see_for(p, t, me_c) > 0 { return vec![]; }
    let att = b.attacks_to(t, me_c, b.occupied()) & b.by_color(me_c);
    if att.is_empty() { return vec![("not attacked".into(), Some(target), vec![], format!("nothing of ours attacks {}", nm(t)))]; }
    let d: Vec<u8> = match b.piece_at(t) {
        Some(pc) => sorted(b.attacks_to(t, pc.color, b.occupied()) & b.by_color(pc.color)).into_iter().map(|s| s as u8).collect(),
        None => vec![],
    };
    vec![("defended".into(), Some(target), d.clone(), format!("attacked, guarded {}x, the exchange gives nothing", d.len()))]
}

// ------------------------------------------------------------------------------------------------------------------ the per-position cache (54 B3)
// The agenda asks the same board again and again (still_routes, forcing_whys, material_defects and mate_defects each parse the board of one routes_of): the parsed
// position of a FEN is kept for the whole root question; `clear_caches()` is called wherever graph.clear_caches() is (every fresh root).
static CACHE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
const CACHE_MAX: usize = 100_000;
fn cache() -> &'static Mutex<HashMap<String, Chess>> { static C: OnceLock<Mutex<HashMap<String, Chess>>> = OnceLock::new(); C.get_or_init(|| Mutex::new(HashMap::new())) }

/// Switch the per-position cache (on by default); returns the previous setting. Off: every call parses its FEN afresh (what the comparison harness measures against).
#[pyfunction]
pub fn set_caches(on: bool) -> bool { let old = CACHE_ON.swap(on, std::sync::atomic::Ordering::Relaxed); if !on { cache().lock().unwrap().clear(); } old }

/// Forget every cached position (a fresh root question).
#[pyfunction]
pub fn clear_caches() { cache().lock().unwrap().clear(); }

/// How many positions are cached.
#[pyfunction]
pub fn cache_size() -> usize { cache().lock().unwrap().len() }

fn parse_fen(fen: &str) -> PyResult<Chess> {
    if CACHE_ON.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(e) = cache().lock().unwrap().get(fen) { return Ok(e.clone()); }
    }
    let p = parse(fen).ok_or_else(|| err("illegal or unparsable fen"))?;
    if CACHE_ON.load(std::sync::atomic::Ordering::Relaxed) {
        let mut c = cache().lock().unwrap();
        if c.len() >= CACHE_MAX { c.clear(); }
        c.insert(fen.to_string(), p.clone());
    }
    Ok(p)
}

/// mate_defects(fen, me_white, placements) -> [(kind, sq or None, by, why)]: what stands between the pattern (placements made, THEIR move) and a mate.
#[pyfunction]
pub fn mate_defects(fen: &str, me_white: bool, placements: Vec<Placement>) -> PyResult<Vec<Def>> {
    let f: Fen = fen.parse().map_err(|_| err("unparsable fen"))?;
    let s = f.into_setup();
    mate_defects_board(&s.board, me_white, &placements)
}

/// material_defects(fen, me_white, target) -> [(kind, sq, by, why)]: why we do not win the piece on `target` now.
#[pyfunction]
pub fn material_defects(fen: &str, me_white: bool, target: u8) -> PyResult<Vec<Def>> {
    Ok(material_defects_pos(&parse_fen(fen)?, me_white, target))
}

// ------------------------------------------------------------------------------------------------------------------ _still

/// (root goal, q goal, defect key (kind, sq), placements, `before` keys, q.pattern target, q.pattern frm, q.pattern blocker, root.pattern target)
type Spec = (String, String, (String, Option<u8>), Vec<Placement>, Vec<(String, Option<u8>)>, Option<u8>, Option<u8>, Option<u8>, Option<u8>);

fn goal_done(q_goal: &str, cf: &Board, me: Color, target: Option<u8>, frm: Option<u8>, blocker: Option<u8>) -> Result<bool, PyErr> {
    let att = |t: u8| -> Bitboard { let t = Square::new(t as u32); cf.attacks_to(t, me, cf.occupied()) & cf.by_color(me) };
    match q_goal {
        "cover" => Ok(att(target.ok_or_else(|| err("cover without a target"))?).any()),
        "guard" => {
            let t = target.ok_or_else(|| err("guard without a target"))?;
            Ok(att(t).into_iter().any(|a| Some(a as u8) != frm && a as u8 != t))
        }
        "remove" if blocker.is_some() => Ok(cf.piece_at(Square::new(blocker.unwrap() as u32)).map_or(true, |p| p.color != me)),
        _ => Ok(true),
    }
}

/// still_routes(fen, us_white, ucis, spec) -> [(uci, still)]: regress._still for a whole candidate list. `fen` is the board (us to move), `ucis` the legal candidate
/// moves (castling excluded by the caller), `spec` the question (see Spec).
#[pyfunction]
pub fn still_routes(fen: &str, us_white: bool, ucis: Vec<String>, spec: Spec) -> PyResult<Vec<(String, bool)>> {
    let (root_goal, q_goal, dkey, pl, before, tgt, frm, blk, win_tgt) = spec;
    let p = parse_fen(fen)?;
    let (me, them) = (col(us_white), col(!us_white));
    let before: std::collections::HashSet<(String, Option<u8>)> = before.into_iter().collect();
    let mut out = Vec::with_capacity(ucis.len());
    for u in ucis {
        let m = shakmaty::uci::UciMove::from_ascii(u.as_bytes()).ok().and_then(|x| x.to_move(&p).ok()).ok_or_else(|| err("illegal candidate"))?;
        let mut c = p.clone();
        c.play_unchecked(&m);
        let still = if root_goal == "mate" {
            let after: Vec<(String, Option<u8>)> = mate_defects_board(c.board(), us_white, &pl)?.into_iter().map(|d| (d.0, d.1)).collect();
            if after.iter().any(|k| *k == dkey) || after.iter().any(|k| k.0 == "no check") || !after.iter().all(|k| before.contains(k)) {
                true
            } else {
                let mut cf = c.board().clone();
                apply(&mut cf, &pl)?;
                !goal_done(&q_goal, &cf, me, tgt, frm, blk)?
            }
        } else if root_goal == "win" {
            let t = win_tgt.ok_or_else(|| err("win without a target"))?;
            if c.is_check() {
                let b = c.board();
                (b.attacks_to(Square::new(t as u32), me, b.occupied()) & b.by_color(me)).is_empty()
            } else {
                let w = flip(&c, me).ok_or_else(|| err("still: with_turn is not a position"))?;
                !material_defects_pos(&w, us_white, t).is_empty()
            }
        } else { true };
        let _ = them;
        out.push((u, still));
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------------------------ forcing_why

/// regress.threat_value on the board after our move (their turn): the most we would win by the exchange on a piece of theirs, us on move. 0 when the board with us on
/// move is not a legal position (the evaluator reads nothing).
fn threat_value(c: &Chess, me: Color) -> i64 {
    let Some(d) = flip(c, me) else { return 0 };
    let b = d.board();
    let mut best = 0;
    for s in b.by_color(!me) {
        if b.role_at(s) == Some(Role::King) { continue; }
        if (b.attacks_to(s, me, b.occupied()) & b.by_color(me)).is_empty() { continue; }
        best = best.max(crate::rules::see(b, s, me).max(0));
    }
    best
}

fn mate_threat(c: &Chess, me: Color) -> Result<bool, PyErr> {
    let d = flip(c, me).ok_or_else(|| err("mate_threat: with_turn is not a position"))?;
    Ok(d.legal_moves().iter().any(|mv| { let mut r = d.clone(); r.play_unchecked(mv); r.is_checkmate() }))
}

/// forcing_whys(fen, ucis, stake, us_white, mate_threats) -> [(status, text, n1, n2, cap_uci)] per move, status:
///   "why"       text = the reason ("check", "threat of mate", "capture 5", "threat 3")
///   "none"      not forcing
///   "pass"      not forcing by the exchange; the caller asks pass_claim (stake <= 8, we have a pawn)
///   "survives"  a sacrifice that still threatens: text = the piece letter, n1 = what we still win after the worst capture, n2 = what they win on the square, cap = the worst capture
///   "hangs"     not a threat: text = the piece letter, n1 = what they win on the square; cap = the worst capture ("" when none is legal), n2 = what we win after it
/// The board is the one the moves are made on (us to move).
#[pyfunction]
pub fn forcing_whys(fen: &str, ucis: Vec<String>, stake: i64, us_white: bool, mate_threats: bool) -> PyResult<Vec<(String, String, i64, i64, String)>> {
    let p = parse_fen(fen)?;
    let me = col(us_white);
    let has_pawn = p.board().by_color(me).intersect(p.board().by_role(Role::Pawn)).any();
    let mut out = Vec::with_capacity(ucis.len());
    let r = |s: &str| (s.to_string(), String::new(), 0, 0, String::new());
    for u in ucis {
        let m = shakmaty::uci::UciMove::from_ascii(u.as_bytes()).ok().and_then(|x| x.to_move(&p).ok()).ok_or_else(|| err("illegal candidate"))?;
        let mut c = p.clone();
        c.play_unchecked(&m);
        if c.is_check() { out.push(("why".into(), "check".into(), 0, 0, String::new())); continue; }
        if mate_threats && mate_threat(&c, me)? { out.push(("why".into(), "threat of mate".into(), 0, 0, String::new())); continue; }
        if stake >= STAKE_MATE { out.push(r("none")); continue; }
        if m.is_capture() {
            let v = if m.is_en_passant() { 1 } else { worth(p.board().piece_at(m.to()).map_or(Role::King, |x| x.role)) };
            if v >= stake { out.push(("why".into(), format!("capture {v}"), 0, 0, String::new())); continue; }
        }
        let t = threat_value(&c, me);
        if t < stake {
            out.push(if stake <= STAKE_PROMOTE && has_pawn { r("pass") } else { r("none") });
            continue;
        }
        let to = m.to();
        let lost = crate::rules::see(c.board(), to, !me).max(0);
        if lost > 0 {
            let pc = c.board().piece_at(to).ok_or_else(|| err("forcing: nothing on the destination"))?;
            let mut caps: Vec<Move> = c.legal_moves().into_iter().filter(|x| x.to() == to && !x.is_castle()).collect();
            let b = c.board();
            caps.sort_by_key(|x| py_key(x, b, false));
            let mut worst: Option<(i64, String)> = None;
            for cap in caps {
                let mut d = c.clone();
                d.play_unchecked(&cap);
                let t2 = if d.is_check() { 0 } else { threat_value(&d, me) };
                if worst.as_ref().map_or(true, |w| t2 < w.0) { worst = Some((t2, cap.to_uci(CastlingMode::Standard).to_string())); }
            }
            let l = letter(pc).to_string();
            match worst {
                Some((t2, cap)) if t2 >= stake => out.push(("survives".into(), l, t2, lost, cap)),
                Some((t2, cap)) => out.push(("hangs".into(), l, lost, t2, cap)),
                None => out.push(("hangs".into(), l, lost, 0, String::new())),
            }
            continue;
        }
        out.push(("why".into(), format!("threat {t}"), 0, 0, String::new()));
    }
    Ok(out)
}
