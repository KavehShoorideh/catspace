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
// (plan 76.5: the promotion's stake is the valuation's -- crate::valuation::promotion_stake(), the queen less the pawn of the store's table)

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

/// Forget every cached position (a fresh root question) and every cached word (plan 72.7 step 1: vocab.rs, the word cache).
#[pyfunction]
pub fn clear_caches() { cache().lock().unwrap().clear(); crate::vocab::clear_word_cache(); }

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

// plan 72.7 step 1 (RUST_BOUNDARY 3.2): THE Pos-TAKING SIBLINGS of the FEN readers the agenda calls through regress.py / binder.py / reasoned.py -- the same inner
// functions, the position handed over as the node's handle (graph.pos_of) instead of a FEN string parsed per call. The FEN versions stay as shims.

/// mate_defects_pos(pos, me_white, placements): `mate_defects` on the handle's board.
#[pyfunction]
pub fn mate_defects_pos(pos: &crate::pos::Pos, me_white: bool, placements: Vec<Placement>) -> PyResult<Vec<Def>> {
    mate_defects_board(pos.pos.board(), me_white, &placements)
}

/// material_defects_pos(pos, me_white, target): `material_defects` on the handle.
#[pyfunction]
#[pyo3(name = "material_defects_pos")]
pub fn material_defects_at(pos: &crate::pos::Pos, me_white: bool, target: u8) -> PyResult<Vec<Def>> {
    Ok(material_defects_pos(&pos.pos, me_white, target))
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
    still_routes_on(&parse_fen(fen)?, us_white, ucis, spec)
}

/// still_routes_pos(pos, us_white, ucis, spec): `still_routes` on the handle.
#[pyfunction]
pub fn still_routes_pos(pos: &crate::pos::Pos, us_white: bool, ucis: Vec<String>, spec: Spec) -> PyResult<Vec<(String, bool)>> {
    still_routes_on(&pos.pos, us_white, ucis, spec)
}

fn still_routes_on(p: &Chess, us_white: bool, ucis: Vec<String>, spec: Spec) -> PyResult<Vec<(String, bool)>> {
    let (root_goal, q_goal, dkey, pl, before, tgt, frm, blk, win_tgt) = spec;
    let (me, them) = (col(us_white), col(!us_white));
    let before: std::collections::HashSet<(String, Option<u8>)> = before.into_iter().collect();
    let mut out = Vec::with_capacity(ucis.len());
    for u in ucis {
        let m = shakmaty::uci::UciMove::from_ascii(u.as_bytes()).ok().and_then(|x| x.to_move(p).ok()).ok_or_else(|| err("illegal candidate"))?;
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
    forcing_whys_on(&parse_fen(fen)?, ucis, stake, us_white, mate_threats)
}

/// forcing_whys_pos(pos, ucis, stake, us_white, mate_threats): `forcing_whys` on the handle.
#[pyfunction]
pub fn forcing_whys_pos(pos: &crate::pos::Pos, ucis: Vec<String>, stake: i64, us_white: bool, mate_threats: bool) -> PyResult<Vec<(String, String, i64, i64, String)>> {
    forcing_whys_on(&pos.pos, ucis, stake, us_white, mate_threats)
}

fn forcing_whys_on(p: &Chess, ucis: Vec<String>, stake: i64, us_white: bool, mate_threats: bool) -> PyResult<Vec<(String, String, i64, i64, String)>> {
    let me = col(us_white);
    let has_pawn = p.board().by_color(me).intersect(p.board().by_role(Role::Pawn)).any();
    let mut out = Vec::with_capacity(ucis.len());
    let r = |s: &str| (s.to_string(), String::new(), 0, 0, String::new());
    for u in ucis {
        let m = shakmaty::uci::UciMove::from_ascii(u.as_bytes()).ok().and_then(|x| x.to_move(p).ok()).ok_or_else(|| err("illegal candidate"))?;
        let mut c = p.clone();
        c.play_unchecked(&m);
        if c.is_check() { out.push(("why".into(), "check".into(), 0, 0, String::new())); continue; }
        if mate_threats && mate_threat(&c, me)? { out.push(("why".into(), "threat of mate".into(), 0, 0, String::new())); continue; }
        if stake >= STAKE_MATE { out.push(r("none")); continue; }
        if m.is_capture() {
            let v = if m.is_en_passant() { crate::valuation::pawn() } else { worth(p.board().piece_at(m.to()).map_or(Role::King, |x| x.role)) };
            if v >= stake { out.push(("why".into(), format!("capture {v}"), 0, 0, String::new())); continue; }
        }
        let t = threat_value(&c, me);
        if t < stake {
            out.push(if stake <= crate::valuation::promotion_stake() && has_pawn { r("pass") } else { r("none") });
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

// ------------------------------------------------------------------------------------------------------------------ plan 77.1 (a): the one definition's reading
// THE REPLY LOOP OF `regress.forcing_by_definition`, one call per node: for every legal move of the side to move, the claims of ours that bind after it and a pass,
// and after each of their captures of the unit that moved. Generic machinery over the rules kernel: WHICH claims are read is the caller's list (the store's GOAL
// entries, by the reader each is bound to); this module binds two readers, both rule level -- "terminal" (the kernel's mate in one: some legal move checkmates)
// and "exchange" (the exchange law on each unit of theirs we attack, the word `see`). Every other goal's claims (a lemma read in Python) are added by the caller
// on the positions this call names (where the other side's units fall inside a declared domain, `open`). The decision (new claim? survives? the text) stays in Python: one definition, read by two readers.

/// A claim read on a position (us to move): (reader, square or -1, value). Kept when its value reaches the stake.
type ClaimRow = (String, i64, i64);

fn claims_on(x: &Chess, me: Color, stake: i64, readers: &[String]) -> Vec<ClaimRow> {
    let mut out = vec![];
    for r in readers {
        match r.as_str() {
            "terminal" => if crate::pos::pos_mate_in_one(x) && STAKE_MATE >= stake { out.push((r.clone(), -1, STAKE_MATE)); },
            "exchange" => {
                let b = x.board();
                for sq in b.by_color(!me) {
                    if b.role_at(sq) == Some(crate::valuation::royal()) || (b.attacks_to(sq, me, b.occupied()) & b.by_color(me)).is_empty() { continue; }
                    let v = i64::from(crate::pos::see_word(x, sq));
                    if v >= stake { out.push((r.clone(), sq as i64, v)); }
                }
            }
            _ => {}
        }
    }
    out
}

fn fen_of(x: &Chess) -> String { Fen::from_position(x.clone(), EnPassantMode::Legal).to_string() }

/// The unit kinds the other side of `me` holds on `x`, as a mask of the kernel's kinds (bit k - 1 for kind k: the kernel's numbering, pawn 1 .. king 6).
fn their_kinds(x: &Chess, me: Color) -> u8 {
    let b = x.board();
    Role::ALL.iter().filter(|r| (b.by_role(**r) & b.by_color(!me)).any()).fold(0u8, |m, r| m | (1u8 << (*r as u8 - 1)))
}

/// The position is named to the caller (its FEN) when the other side's unit kinds fall inside one of the caller's masks (a lemma's declared domain: the caller's, by its store
/// entry; this module knows no lemma).
fn named(x: &Chess, me: Color, open: &[u8]) -> String { let k = their_kinds(x, me); if open.iter().any(|m| k & !m == 0) { fen_of(x) } else { String::new() } }

/// One capture of the moved unit: (uci, the threat ends -- it checks, or the position after it is a mate or a stalemate --, the claims then, that position's FEN or "").
type CapRow = (String, bool, Vec<ClaimRow>, String);
/// One move: (uci, "check" | "capture" | "ended" | "noturn" | "read", the capture's worth, the claims after the move and a pass, what they win by the exchange on the
/// moved unit's square, its captures in the kernel's order up to the first that ends the threat (read only when that exchange wins them something), the pass
/// position's FEN or "").
type DefRow = (String, String, i64, Vec<ClaimRow>, i64, Vec<CapRow>, String);

/// definition_rows_pos(pos, stake, us_white, readers, open) -> (the claims binding now, [one row per legal move]) -- see the section's comment. `open`: name the
/// positions read where the other side's units fall inside one of the `open` masks (the caller adds its own readers' claims there). ValueError when it is not `us`'s move.
#[pyfunction]
pub fn definition_rows_pos(pos: &crate::pos::Pos, stake: i64, us_white: bool, readers: Vec<String>, open: Vec<u8>) -> PyResult<(Vec<ClaimRow>, Vec<DefRow>)> {
    let p = &pos.pos;
    let me = col(us_white);
    if p.turn() != me { return Err(err("definition_rows: not our move")); }
    let before = claims_on(p, me, stake, &readers);
    let mut rows = Vec::new();
    for m in p.legal_moves() {
        let u = m.to_uci(CastlingMode::Standard).to_string();
        let mut c = p.clone();
        c.play_unchecked(&m);
        if c.is_check() { rows.push((u, "check".into(), 0, vec![], 0, vec![], String::new())); continue; }
        if m.is_capture() {
            let v = crate::valuation::worth(if m.is_en_passant() { m.role() } else { p.board().piece_at(m.to()).map_or(crate::valuation::royal(), |x| x.role) });   // (en passant takes a unit of the mover's kind)
            if v >= stake { rows.push((u, "capture".into(), v, vec![], 0, vec![], String::new())); continue; }
        }
        if c.is_checkmate() || c.is_stalemate() { rows.push((u, "ended".into(), 0, vec![], 0, vec![], String::new())); continue; }
        let Some(q) = crate::pos::with_turn_strict(&c, me) else { rows.push((u, "noturn".into(), 0, vec![], 0, vec![], String::new())); continue };
        let cl = claims_on(&q, me, stake, &readers);
        let qfen = named(&q, me, &open);
        let to = m.to();
        let lost = i64::from(crate::pos::see_word(&c, to));
        let mut caps = vec![];
        if lost >= crate::valuation::unit() {                       // (they win something: at least the valuation's unit)
            for r in c.legal_moves() {
                if !r.is_capture() || r.to() != to { continue; }
                let ru = r.to_uci(CastlingMode::Standard).to_string();
                let mut d = c.clone();
                d.play_unchecked(&r);
                if d.is_check() || d.is_checkmate() || d.is_stalemate() { caps.push((ru, true, vec![], String::new())); break; }
                let dc = claims_on(&d, me, stake, &readers);
                caps.push((ru, false, dc, named(&d, me, &open)));
            }
        }
        rows.push((u, "read".into(), 0, cl, lost, caps, qfen));
    }
    Ok((before, rows))
}

// ------------------------------------------------------------------------------------------------------------------ plan 77.1 (a): every legal reply
// THE EVERY-REPLY CLAUSE (Kaveh 2026-11-04): where the pass is no move of theirs -- after a check (a pass is illegal) and in a zugzwang (a pass flatters them) -- the
// question is whether a claim of ours binds after EVERY legal reply, the worst reply deciding. One call reads every reply of theirs to our move: the claims then (us to
// move) by the caller's readers, and our balance change by the reply (what it took or promoted). The decision (new claims, the worst, the text) stays in Python.

/// One reply of theirs: (uci, the threat ends -- the reply checks us, or the game is over after it --, our balance change by it, the claims then, that position's FEN or "").
type ReplyRow = (String, bool, i64, Vec<ClaimRow>, String);

/// reply_claims_pos(pos, uci, stake, us_white, readers, open) -> ("mate" | "stalemate" | "read", the claims binding now (before the move), [one row per legal reply to
/// our legal move `uci`, the kernel's order]).
/// ValueError when it is not `us`'s move or the move is not legal.
#[pyfunction]
pub fn reply_claims_pos(pos: &crate::pos::Pos, uci: &str, stake: i64, us_white: bool, readers: Vec<String>, open: Vec<u8>) -> PyResult<(String, Vec<ClaimRow>, Vec<ReplyRow>)> {
    let me = col(us_white);
    if pos.pos.turn() != me { return Err(err("reply_claims: not our move")); }
    let c = pos.push_move(uci)?.pos;
    if c.is_checkmate() { return Ok(("mate".into(), vec![], vec![])); }
    if c.is_stalemate() { return Ok(("stalemate".into(), vec![], vec![])); }
    let before = claims_on(&pos.pos, me, stake, &readers);
    let base = crate::pos::bal(&c, me);
    let mut rows = Vec::new();
    for r in c.legal_moves() {
        let ru = r.to_uci(CastlingMode::Standard).to_string();
        let mut d = c.clone();
        d.play_unchecked(&r);
        let dbal = crate::pos::bal(&d, me) - base;
        if d.is_check() || d.is_stalemate() { rows.push((ru, true, dbal, vec![], String::new())); continue; }
        let cl = claims_on(&d, me, stake, &readers);
        rows.push((ru, false, dbal, cl, named(&d, me, &open)));
    }
    Ok(("read".into(), before, rows))
}

/// THE TERMINAL GOAL IN `n` MOVES OF OURS, composed by the same clause: some legal move ends the game by the kernel's terminal (checkmate), or (n > 1) after it every
/// legal reply leaves the terminal goal in n - 1 (a reply-less position that is no mate is a stalemate: no). Mate in one is n = 1.
fn terminal_in(p: &Chess, n: u32) -> bool {
    for m in p.legal_moves() {
        let mut q = p.clone();
        q.play_unchecked(&m);
        if q.is_checkmate() { return true; }
        if n <= 1 { continue; }
        let rs = q.legal_moves();
        if rs.is_empty() { continue; }
        if rs.iter().all(|r| { let mut x = q.clone(); x.play_unchecked(r); terminal_in(&x, n - 1) }) { return true; }
    }
    false
}

/// terminal_after_pass_pos(pos, uci, us_white, n) -> after our legal move `uci` and a pass (none after a check: False), the terminal goal binds in `n` moves of ours.
#[pyfunction]
pub fn terminal_after_pass_pos(pos: &crate::pos::Pos, uci: &str, us_white: bool, n: u32) -> PyResult<bool> {
    let c = pos.push_move(uci)?.pos;
    if c.is_check() { return Ok(false); }
    let Some(q) = crate::pos::with_turn_strict(&c, col(us_white)) else { return Ok(false) };
    if q.legal_moves().is_empty() { return Ok(false); }
    Ok(terminal_in(&q, n))
}

// ------------------------------------------------------------------------------------------------------------------ 54 (B4): trap, fork, skewer, drive-onto, second moves
// Ports of regress.trap_routes / fork_routes / skewer_routes / drive_onto_routes and binder.second_moves with python-chess's candidate order (`legal_sorted`).
// The Python wrappers write the SAN (shakmaty's SAN is not python-chess's) and keep the old path behind regress.TRAP_RUST / SKEWER_RUST and binder.SECOND_RUST.

/// `G.with_turn(board, colour)`: the same position with `turn` to move and no en passant square (a copy is cloned only when something must change).
fn with_turn_pos(p: &Chess, turn: Color) -> PyResult<Chess> {
    if p.turn() == turn && p.maybe_ep_square().is_none() { return Ok(p.clone()); }
    flip(p, turn).ok_or_else(|| err("with_turn is not a position"))
}

/// The moves of `w` (us to move, no en passant square) in python-chess's order, castling dropped (`regress._moves`).
fn moves_of(w: &Chess) -> Vec<Move> {
    let mut v = legal_sorted(w, w.is_check());
    v.retain(|m| !m.is_castle());
    v
}

/// The squares the piece on `x` can go to safely (the word `mobility`, read with its owner to move, no en passant square: `G.mobility_of`).
fn mob_count(c: &Chess, x: Square) -> PyResult<usize> {
    if c.maybe_ep_square().is_some() {
        let c0 = crate::vocab::relaxed(c.clone().into_setup(EnPassantMode::Always)).ok_or_else(|| err("mobility: not a position"))?;
        return Ok(crate::vocab::mobility(&c0, x).count());
    }
    Ok(crate::vocab::mobility(c, x).count())
}

/// Our quiet moves after which the piece of theirs on `t` has no safe square: [(uci, why)]; `stake` is not read (the Python twin does not read it either).
fn trap_routes_pos(w: &Chess, me: Color, t: Square) -> PyResult<Vec<(String, String)>> {
    let mut out = vec![];
    for m in moves_of(w) {
        if m.to() == t { continue; }
        let mut c = w.clone();
        c.play_unchecked(&m);
        if c.is_check() { continue; }
        let Some(pc) = c.board().piece_at(t) else { continue };
        if pc.color == me { continue; }
        if mob_count(&c, t)? != 0 { continue; }
        let how = if (c.board().attacks_to(t, me, c.board().occupied()) & c.board().by_color(me)).any() { "also attacked" } else { "quiet" };
        out.push((m.to_uci(CastlingMode::Standard).to_string(), format!("traps the {} on {}: no safe square, {}", letter(pc), nm(t), how)));
    }
    Ok(out)
}

fn fork_routes_pos(w: &Chess, t: Square) -> Vec<(String, String)> {
    let mut out = vec![];
    for m in moves_of(w) {
        if m.to() == t { continue; }
        let mut c = w.clone();
        c.play_unchecked(&m);
        if !c.is_check() { continue; }
        let (Some(pc), Some(mv)) = (c.board().piece_at(t), c.board().piece_at(m.to())) else { continue };
        if pc.color == mv.color || !c.board().attacks_from(m.to()).contains(t) { continue; }
        out.push((m.to_uci(CastlingMode::Standard).to_string(), format!("fork: check and attack the {} on {}", letter(pc), nm(t))));
    }
    out
}

#[pyfunction]
pub fn trap_routes(fen: &str, us_white: bool, target: u8, stake: i64) -> PyResult<Vec<(String, String)>> {
    let _ = stake;
    let w = with_turn_pos(&parse_fen(fen)?, col(us_white))?;
    trap_routes_pos(&w, col(us_white), Square::new(target as u32))
}

#[pyfunction]
pub fn fork_routes(fen: &str, us_white: bool, target: u8) -> PyResult<Vec<(String, String)>> {
    let w = with_turn_pos(&parse_fen(fen)?, col(us_white))?;
    Ok(fork_routes_pos(&w, Square::new(target as u32)))
}

/// `trap_fix`'s three reads in one call: (n, trap routes, fork routes), n = the safe squares of the piece on `target` now (read with its owner to move); the trap
/// routes are listed only when n > 0, as the Python twin does. Err(ValueError) when there is no piece of theirs to read (the caller has checked).
#[pyfunction]
pub fn trap_fix_routes(fen: &str, us_white: bool, target: u8) -> PyResult<(usize, Vec<(String, String)>, Vec<(String, String)>)> {
    trap_fix_routes_on(&parse_fen(fen)?, us_white, target)
}

/// trap_fix_routes_pos(pos, us_white, target): `trap_fix_routes` on the handle.
#[pyfunction]
pub fn trap_fix_routes_pos(pos: &crate::pos::Pos, us_white: bool, target: u8) -> PyResult<(usize, Vec<(String, String)>, Vec<(String, String)>)> {
    trap_fix_routes_on(&pos.pos, us_white, target)
}

fn trap_fix_routes_on(p: &Chess, us_white: bool, target: u8) -> PyResult<(usize, Vec<(String, String)>, Vec<(String, String)>)> {
    let (me, t) = (col(us_white), Square::new(target as u32));
    let owner = p.board().color_at(t).ok_or_else(|| err("no piece on the target"))?;
    let _ = owner;
    let n = mob_count(&p, t)?;
    let w = with_turn_pos(&p, me)?;
    let traps = if n > 0 { trap_routes_pos(&w, me, t)? } else { vec![] };
    Ok((n, traps, fork_routes_pos(&w, t)))
}

/// `G.skewer_after`: on the board after our move (their turn) a slider of ours that gives check and whose line runs through their king to `t`, every other square
/// between empty -> the slider's square (the first in square order that qualifies).
fn skewer_after(c: &Chess, me: Color, t: Square) -> Option<Square> {
    let b = c.board();
    let k = b.king_of(!me)?;
    let pt = b.piece_at(t)?;
    if pt.color == me { return None; }
    let checkers = b.attacks_to(k, me, b.occupied()) & b.by_color(me);       // c.checkers(): attackers of the king of the side to move
    for s2 in sorted(checkers) {
        let pc = b.piece_at(s2)?;
        if !matches!(pc.role, Role::Rook | Role::Bishop | Role::Queen) { continue; }
        let line = attacks::between(s2, t);
        if !line.contains(k) { continue; }
        let straight = s2.file() == t.file() || s2.rank() == t.rank();
        if (pc.role == Role::Rook && !straight) || (pc.role == Role::Bishop && straight) { continue; }
        if (line & b.occupied()).without(k).is_empty() { return Some(s2); }
    }
    None
}

fn skewer_routes_pos(w: &Chess, me: Color, t: Square, first_only: bool) -> Vec<(String, String)> {
    let mut out = vec![];
    for m in moves_of(w) {
        if m.to() == t { continue; }
        let mut c = w.clone();
        c.play_unchecked(&m);
        if !c.is_check() { continue; }
        if skewer_after(&c, me, t) == Some(m.to()) {
            let pc = w.board().piece_at(t).unwrap();
            out.push((m.to_uci(CastlingMode::Standard).to_string(),
                      format!("check; skewer: through their K on {} to the {} on {} (Ideas.Skewer.skewer_attack)", nm(c.board().king_of(!me).unwrap()), letter(pc), nm(t))));
            if first_only { return out; }
        }
    }
    out
}

/// skewer_routes(fen, us_white, target) -> [(uci, why)]: checks by a slider of ours whose line runs through their king to `target` (52.23 (4)).
#[pyfunction]
pub fn skewer_routes(fen: &str, us_white: bool, target: u8) -> PyResult<Vec<(String, String)>> {
    let me = col(us_white);
    let w = with_turn_pos(&parse_fen(fen)?, me)?;
    Ok(skewer_routes_pos(&w, me, Square::new(target as u32), false))
}

/// skewer_routes_on_pos(pos, us_white, target): `skewer_routes` on the handle.
#[pyfunction]
#[pyo3(name = "skewer_routes_pos")]
pub fn skewer_routes_at(pos: &crate::pos::Pos, us_white: bool, target: u8) -> PyResult<Vec<(String, String)>> {
    let me = col(us_white);
    let w = with_turn_pos(&pos.pos, me)?;
    Ok(skewer_routes_pos(&w, me, Square::new(target as u32), false))
}

/// drive_onto_routes(fen, us_white, target, stake) -> [(uci, why)]: checks after which EVERY king reply allows a skewer route to `target` (52.23 (4b)).
#[pyfunction]
pub fn drive_onto_routes(fen: &str, us_white: bool, target: u8, stake: i64) -> PyResult<Vec<(String, String)>> {
    drive_onto_routes_on(&parse_fen(fen)?, us_white, target, stake)
}

/// drive_onto_routes_pos(pos, us_white, target, stake): `drive_onto_routes` on the handle.
#[pyfunction]
pub fn drive_onto_routes_pos(pos: &crate::pos::Pos, us_white: bool, target: u8, stake: i64) -> PyResult<Vec<(String, String)>> {
    drive_onto_routes_on(&pos.pos, us_white, target, stake)
}

fn drive_onto_routes_on(p: &Chess, us_white: bool, target: u8, stake: i64) -> PyResult<Vec<(String, String)>> {
    let _ = stake;
    let (me, t) = (col(us_white), Square::new(target as u32));
    let w = with_turn_pos(p, me)?;
    let mut out = vec![];
    for m in moves_of(&w) {
        if m.to() == t { continue; }
        let mut c = w.clone();
        c.play_unchecked(&m);
        if !c.is_check() || skewer_after(&c, me, t) == Some(m.to()) { continue; }
        let ksq = c.board().king_of(!me).unwrap();
        let kings: Vec<Move> = c.legal_moves().into_iter().filter(|r| r.from() == Some(ksq)).collect();
        if kings.is_empty() { continue; }
        let mut ok = true;
        for r in &kings {
            let mut d = c.clone();
            d.play_unchecked(r);
            let good = matches!(d.board().piece_at(t), Some(pc) if pc.color != me);
            // skewer_routes(d, us, t) reads d with `us` to move and no en passant square
            if !good || skewer_routes_pos(&with_turn_pos(&d, me)?, me, t, true).is_empty() { ok = false; break; }
        }
        if ok {
            out.push((m.to_uci(CastlingMode::Standard).to_string(),
                      format!("check; drives their king onto a skewer line to {} (every king reply, then a check through it: Ideas.Skewer.skewer_attack)", nm(t))));
        }
    }
    Ok(out)
}

/// second_moves(fen, route_uci, trap) -> [(from, to)]: binder.second_moves. `fen` = the board with us to move; `trap` = the square of the piece a QUIET route traps.
#[pyfunction]
#[pyo3(signature = (fen, route, trap=None))]
pub fn second_moves(fen: &str, route: &str, trap: Option<u8>) -> PyResult<Vec<(String, String)>> {
    second_moves_on(&parse_fen(fen)?, route, trap)
}

/// second_moves_pos(pos, route, trap=None): `second_moves` on the handle.
#[pyfunction]
#[pyo3(signature = (pos, route, trap=None))]
pub fn second_moves_pos(pos: &crate::pos::Pos, route: &str, trap: Option<u8>) -> PyResult<Vec<(String, String)>> {
    second_moves_on(&pos.pos, route, trap)
}

fn second_moves_on(p: &Chess, route: &str, trap: Option<u8>) -> PyResult<Vec<(String, String)>> {
    let m = shakmaty::uci::UciMove::from_ascii(route.as_bytes()).ok().and_then(|x| x.to_move(p).ok()).ok_or_else(|| err("illegal route"))?;
    let mut c = p.clone();
    c.play_unchecked(&m);
    let quiet = !c.is_check();
    let trap = trap.map(|t| Square::new(t as u32));
    if quiet {
        let Some(t) = trap else { return Ok(vec![]) };
        if c.board().piece_at(t).is_none() || mob_count(&c, t)? != 0 { return Ok(vec![]); }
    }
    let mut common: Option<std::collections::BTreeSet<(String, String)>> = None;
    for r in c.legal_moves() {
        let mut d = c.clone();
        d.play_unchecked(&r);
        let mut mine = std::collections::BTreeSet::new();
        for x in d.legal_moves() {
            if x.is_capture() && x.promotion().is_none() && !x.is_castle() && (!quiet || Some(x.to()) == trap) {
                mine.insert((nm(x.from().unwrap()), nm(x.to())));
            }
        }
        let next = match common { None => mine, Some(cm) => cm.intersection(&mine).cloned().collect() };
        if next.is_empty() { return Ok(vec![]); }
        common = Some(next);
    }
    Ok(common.map(|s| s.into_iter().collect()).unwrap_or_default())
}

/// see_after(fen, ucis, square) -> for each move (legal at `fen`), the word `(see square)` on the board after it: the static exchange the side to move there wins on that
/// square's piece (0 when it stands empty) -- `reasoned._see(c.fen(), sq)` on the board after the move, without the FEN round trip. 54 (B5.3).
#[pyfunction]
pub fn see_after(fen: &str, ucis: Vec<String>, square: u8) -> PyResult<Vec<i64>> {
    see_after_on(&parse_fen(fen)?, ucis, square)
}

/// see_after_pos(pos, ucis, square): `see_after` on the handle.
#[pyfunction]
pub fn see_after_pos(pos: &crate::pos::Pos, ucis: Vec<String>, square: u8) -> PyResult<Vec<i64>> {
    see_after_on(&pos.pos, ucis, square)
}

fn see_after_on(p: &Chess, ucis: Vec<String>, square: u8) -> PyResult<Vec<i64>> {
    let sq = Square::new(square as u32);
    let mut out = Vec::with_capacity(ucis.len());
    for u in ucis {
        let m = shakmaty::uci::UciMove::from_ascii(u.as_bytes()).ok().and_then(|x| x.to_move(p).ok()).ok_or_else(|| err("illegal move"))?;
        let mut c = p.clone();
        c.play_unchecked(&m);
        out.push(c.board().color_at(sq).map_or(0, |col| crate::rules::see(c.board(), sq, !col).max(0)));
    }
    Ok(out)
}
