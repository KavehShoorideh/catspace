//! ATTACK PATHS (Kaveh 2026-10-09: "if the line diagonally from the king is blocked, there's no piece that can attack
//! from there -- that's like a base lemma -- and so we should be able to see that I have accounted for all the threats,
//! because there is no path to attack my king").
//!
//! THE ONE-MOVE LEMMA: a move changes the attacks on a square z in only two ways -- the moved piece attacks from its
//! destination, and lines through the squares the move vacates open (castling vacates the rook's square too, en
//! passant the captured pawn's). So every attack the side S can make on z with its next move is one of
//!   DIRECT     a piece of S moves to a square from which it attacks z (along an open line, a knight's jump, a pawn's
//!              diagonal, the king's step) -- the ARRIVAL square is on the line, so the path is visible on the board
//!   DISCOVERY  a line from z whose first piece is S's and whose second is an S slider of the line's kind: the first
//!              piece moves off the line
//! and S already attacking z is a path of zero moves (OPEN). Two-move paths that matter for planning:
//!   BATTERY    a line from z whose first piece is S's (the discoverer) and beyond it an empty stretch: an S slider of
//!              the line's kind that can reach a square of that stretch in one move LOADS it (then a discovery)
//!   PIN        a line from z whose first piece is the target side's and second an S slider of the line's kind: the
//!              first piece is pinned to z (no move of it -- the relation, not an attack)
//! If S has no direct, discovery or open path to any square of a zone, S cannot attack the zone with its next move:
//! the safety certificate, read off the board without a search. `path_audit` checks the lemma against every legal move.
use super::*;
use shakmaty::Bitboard;

#[derive(Clone, Debug)]
pub struct Path {
    pub kind: &'static str,   // open / direct / discovery / battery / pin
    pub target: Square,       // the square attacked
    pub piece: Square,        // the piece that will attack (the slider, the arriving piece)
    pub via: Option<Square>,  // the arrival square (direct), the blocker (discovery, battery, pin)
    pub rear: Option<Square>, // battery: the square the slider loads onto
    pub mv: Option<String>,   // the move that makes it (direct: the arrival; battery: the load), in UCI
    pub moves: u8,            // moves needed: 0 open, 1 direct / discovery, 2 battery
}

const DIRS: [(i32, i32, bool); 8] = [(1, 0, false), (-1, 0, false), (0, 1, false), (0, -1, false),
                                     (1, 1, true), (1, -1, true), (-1, 1, true), (-1, -1, true)];

fn slides(role: Role, diagonal: bool) -> bool {
    match role { Role::Queen => true, Role::Rook => !diagonal, Role::Bishop => diagonal, _ => false }
}

fn walk(z: Square, df: i32, dr: i32) -> Vec<Square> {
    let (mut f, mut r) = (z.file() as i32 + df, z.rank() as i32 + dr);
    let mut out = Vec::new();
    while (0..8).contains(&f) && (0..8).contains(&r) {
        out.push(Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new(r as u32)));
        f += df; r += dr;
    }
    out
}

/// The legal moves of side `s` in `pos` (on the null move when it is not s's turn; empty if that is impossible).
fn moves_of(pos: &Chess, s: Color) -> Vec<Move> {
    if pos.turn() == s { pos.legal_moves().into_iter().collect() } else { pass(pos).map(|p| p.legal_moves().into_iter().collect()).unwrap_or_default() }
}

/// Every path of side `s` onto the squares `zone`.
pub fn attack_paths(pos: &Chess, s: Color, zone: Bitboard) -> Vec<Path> {
    let b = pos.board();
    let occ = b.occupied();
    let ms = moves_of(pos, s);
    let uci = |m: &Move| m.to_uci(CastlingMode::Standard).to_string();
    let mut out = Vec::new();
    for z in zone {
        // the eight lines from z
        for (df, dr, diag) in DIRS {
            let line = walk(z, df, dr);
            let mut first: Option<(usize, Square)> = None;
            let mut second: Option<(usize, Square)> = None;
            for (i, sq) in line.iter().enumerate() {
                if occ.contains(*sq) { if first.is_none() { first = Some((i, *sq)); } else { second = Some((i, *sq)); break; } }
            }
            let empty_to = first.map_or(line.len(), |(i, _)| i);         // the open stretch next to z
            let _ = empty_to;
            // DIRECT along the line: a slider of s (or a promotion to one) landing on the line with nothing between it
            // and z after the move (the mover's own square counts as empty; castling: the rook's landing square)
            for m in &ms {
                let (role, land, from) = match m {
                    Move::Castle { king, rook } => {
                        let file = if rook.file() > king.file() { shakmaty::File::F } else { shakmaty::File::D };
                        (Role::Rook, Square::from_coords(file, rook.rank()), *rook)
                    }
                    _ => (m.promotion().unwrap_or(m.role()), m.to(), m.from().unwrap_or(m.to())),
                };
                if !slides(role, diag) { continue; }
                if let Some(i) = line.iter().position(|q| *q == land) {
                    let king_from = if let Move::Castle { king, .. } = m { Some(*king) } else { None };
                    if line[..i].iter().all(|q| !occ.contains(*q) || *q == from || Some(*q) == king_from) {
                        out.push(Path { kind: "direct", target: z, piece: from, via: Some(land), rear: None, mv: Some(uci(m)), moves: 1 });
                    }
                }
            }
            let Some((_, f1)) = first else { continue };
            let p1 = b.piece_at(f1).unwrap();
            if p1.color == s && slides(p1.role, diag) {
                out.push(Path { kind: "open", target: z, piece: f1, via: None, rear: None, mv: None, moves: 0 });
                continue;
            }
            let p2 = second.map(|(_, q)| (q, b.piece_at(q).unwrap()));
            match p2 {
                Some((q2, pc2)) if pc2.color == s && slides(pc2.role, diag) => {
                    let kind = if p1.color == s { "discovery" } else { "pin" };
                    out.push(Path { kind, target: z, piece: q2, via: Some(f1), rear: None, mv: None, moves: if p1.color == s { 1 } else { 0 } });
                }
                _ => {
                    if p1.color == s {
                        // BATTERY: the stretch beyond our blocker, up to the next piece; a slider of s that can land on it
                        let i1 = first.unwrap().0;
                        let stop = second.map_or(line.len(), |(i, _)| i);
                        let stretch: Vec<Square> = line[i1 + 1..stop].to_vec();
                        for m in &ms {
                            let role = if matches!(m, Move::Castle { .. }) { Role::Rook } else { m.promotion().unwrap_or(m.role()) };
                            if m.from() == Some(f1) || !slides(role, diag) { continue; }
                            let land = match m { Move::Castle { king, rook } => {
                                let file = if rook.file() > king.file() { shakmaty::File::F } else { shakmaty::File::D };
                                Square::from_coords(file, rook.rank())
                            }, _ => m.to() };
                            if stretch.contains(&land) {
                                let piece = match m { Move::Castle { rook, .. } => *rook, _ => m.from().unwrap_or(land) };
                                out.push(Path { kind: "battery", target: z, piece, via: Some(f1), rear: Some(land), mv: Some(uci(m)), moves: 2 });
                            }
                        }
                    }
                }
            }
        }
        // jumps and steps: knights, pawns, the king
        for m in &ms {
            let role = m.promotion().unwrap_or(m.role());
            let hit = match role {
                Role::Knight => shakmaty::attacks::knight_attacks(m.to()).contains(z),
                Role::Pawn => shakmaty::attacks::pawn_attacks(s, m.to()).contains(z),
                Role::King => !matches!(m, Move::Castle { .. }) && shakmaty::attacks::king_attacks(m.to()).contains(z),
                _ => false,
            };
            if hit { out.push(Path { kind: "direct", target: z, piece: m.from().unwrap_or(m.to()), via: Some(m.to()), rear: None, mv: Some(uci(m)), moves: 1 }); }
        }
        for sq in b.by_color(s) {
            let p = b.piece_at(sq).unwrap();
            let open = match p.role {
                Role::Knight => shakmaty::attacks::knight_attacks(sq).contains(z),
                Role::Pawn => shakmaty::attacks::pawn_attacks(s, sq).contains(z),
                Role::King => shakmaty::attacks::king_attacks(sq).contains(z),
                _ => false,
            };
            if open { out.push(Path { kind: "open", target: z, piece: sq, via: None, rear: None, mv: None, moves: 0 }); }
        }
    }
    out
}

/// The moves of side s that the lemma says make a NEW attack on some square of `zone` (a square s does not attack now):
/// direct arrivals, and every move of a discovery blocker that leaves its line.
pub fn lemma_attacking_moves(pos: &Chess, s: Color, zone: Bitboard) -> std::collections::HashSet<String> {
    let occ = pos.board().occupied();
    let unattacked: Bitboard = zone.into_iter().filter(|z| pos.board().attacks_to(*z, s, occ).is_empty()).collect();
    let paths = attack_paths(pos, s, unattacked);
    let ms = moves_of(pos, s);
    let mut out = std::collections::HashSet::new();
    for p in &paths {
        match p.kind {
            "direct" => { if let Some(m) = &p.mv { out.insert(m.clone()); } }
            "discovery" => {
                let blocker = p.via.unwrap();
                let line = shakmaty::attacks::between(p.target, p.piece);
                for m in &ms {
                    if m.from() == Some(blocker) && !line.contains(m.to()) { out.insert(m.to_uci(CastlingMode::Standard).to_string()); }
                }
            }
            _ => {}
        }
    }
    // EN PASSANT vacates the captured pawn's square too (audit 2026-10-09: the only 5 misses in 836,694 moves): a
    // slider of s whose line to z runs through that square and is otherwise empty after the capture
    let b = pos.board();
    for m in &ms {
        if let Move::EnPassant { from, to } = m {
            let cap = Square::from_coords(to.file(), from.rank());
            let occ_after = (occ.without(*from).without(cap)).with(*to);
            for z in unattacked {
                for sl in b.by_color(s) & (b.by_role(Role::Rook) | b.by_role(Role::Bishop) | b.by_role(Role::Queen)) {
                    let between = shakmaty::attacks::between(z, sl);
                    if between.contains(cap) && (between & occ_after).is_empty()
                        && shakmaty::attacks::attacks(sl, b.piece_at(sl).unwrap(), occ_after).contains(z) {
                        out.insert(m.to_uci(CastlingMode::Standard).to_string());
                    }
                }
            }
        }
    }
    out
}

/// Brute force: the moves of side s after which s attacks a square of `zone` it did not attack before.
pub fn brute_attacking_moves(pos: &Chess, s: Color, zone: Bitboard) -> std::collections::HashSet<String> {
    let ms = moves_of(pos, s);
    let base = if pos.turn() == s { Some(pos.clone()) } else { pass(pos) };
    let Some(base) = base else { return Default::default() };
    let occ0 = pos.board().occupied();
    let unattacked: Vec<Square> = zone.into_iter().filter(|z| pos.board().attacks_to(*z, s, occ0).is_empty()).collect();
    ms.iter().filter(|m| {
        let q = after(&base, m);
        let occ = q.board().occupied();
        unattacked.iter().any(|z| q.board().attacks_to(*z, s, occ).any())
    }).map(|m| m.to_uci(CastlingMode::Standard).to_string()).collect()
}

fn king_zone(pos: &Chess, c: Color) -> Bitboard {
    match pos.board().king_of(c) { Some(k) => shakmaty::attacks::king_attacks(k).with(k), None => Bitboard::EMPTY }
}

/// path_audit_counts(fen) -> (zones checked, brute-force attacking moves in total, lemma moves in total)
#[pyfunction]
pub fn path_audit_counts(fen: String) -> PyResult<(usize, usize, usize)> {
    let Some(pos) = parse(&fen) else { return Ok((0, 0, 0)) };
    let (mut z, mut b, mut l) = (0, 0, 0);
    for s in [Color::White, Color::Black] {
        let mut zones = vec![king_zone(&pos, !s)];
        for sq in pos.board().by_color(!s) { if pos.board().piece_at(sq).map_or(false, |p| p.role != Role::King) { zones.push(Bitboard::from_square(sq)); } }
        for zz in zones { z += 1; b += brute_attacking_moves(&pos, s, zz).len(); l += lemma_attacking_moves(&pos, s, zz).len(); }
    }
    Ok((z, b, l))
}

/// path_audit(fen) -> [(side attacking, zone kind, moves the lemma missed, moves it over-claimed)] for both sides
/// against the other's king zone and against each of the other's pieces.
#[pyfunction]
pub fn path_audit(fen: String) -> PyResult<Vec<(String, String, Vec<String>, Vec<String>)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let mut out = Vec::new();
    for s in [Color::White, Color::Black] {
        let mut zones: Vec<(String, Bitboard)> = vec![("king zone".to_string(), king_zone(&pos, !s))];
        for sq in pos.board().by_color(!s) {
            if pos.board().piece_at(sq).map_or(false, |p| p.role != Role::King) { zones.push((format!("piece {sq}"), Bitboard::from_square(sq))); }
        }
        for (name, z) in zones {
            let lemma = lemma_attacking_moves(&pos, s, z);
            let brute = brute_attacking_moves(&pos, s, z);
            let missed: Vec<String> = brute.difference(&lemma).cloned().collect();
            let extra: Vec<String> = lemma.difference(&brute).cloned().collect();
            if !missed.is_empty() || !extra.is_empty() {
                out.push((if s == Color::White { "w" } else { "b" }.to_string(), name, missed, extra));
            }
        }
    }
    Ok(out)
}

/// attack_paths_of(fen, side "w"/"b", squares) -> [(kind, target, piece, via, rear, move, moves needed)]
#[pyfunction]
pub fn attack_paths_of(fen: String, side: String, squares: Vec<String>) -> PyResult<Vec<(String, String, String, Option<String>, Option<String>, Option<String>, u8)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let s = if side == "w" { Color::White } else { Color::Black };
    let zone: Bitboard = squares.iter().filter_map(|x| x.parse::<Square>().ok()).collect();
    Ok(attack_paths(&pos, s, zone).into_iter().map(|p| (p.kind.to_string(), p.target.to_string(), p.piece.to_string(),
        p.via.map(|x| x.to_string()), p.rear.map(|x| x.to_string()), p.mv, p.moves)).collect())
}
