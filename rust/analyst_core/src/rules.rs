//! THE RULE LAYER (Kaveh 2026-10-08: forward reasoning from basic facts, fast -- Datalog in Rust via Ascent).
//! Base facts of a position, read for one side ("us" = the side that may threaten): piece(square, ours, value) and
//! attacks(from, to). Derived without search:
//!   defended(t)     a piece of t's own side attacks t
//!   gain(t, g)      t (theirs, not the king) is attacked by ours and hangs (undefended: g = its value) or the
//!                   attacker is cheaper (g = the difference) -- the static exchange's first term, a PREFILTER
//!   best(t, g)      the largest such gain on t
//! A move's NEW threats are the targets whose best gain rises after it; the claims (fork, cornered piece, ...) are
//! named from them and decided by the kernel's certificate (lib.rs `local_check_pos`), so a certified claim is a fact.
use ascent::ascent;
use ascent::aggregators::max;
use shakmaty::{Chess, Color, Position, Role, Square};

ascent! {
    struct ThreatRules;
    relation piece(u8, bool, i64);
    relation attacks(u8, u8);
    relation defended(u8);
    relation gain(u8, i64);
    relation best(u8, i64);

    defended(t) <-- piece(t, false, _), attacks(d, t), piece(d, false, _);
    gain(t, vt) <-- piece(t, false, vt), if *vt > 0, attacks(a, t), piece(a, true, _), !defended(t);
    gain(t, vt - va) <-- piece(t, false, vt), if *vt > 0, attacks(a, t), piece(a, true, va), if *va < *vt;
    best(t, g) <-- gain(t, _), agg g = max(x) in gain(t, x);
}

fn val(role: Role) -> i64 {
    match role { Role::Pawn => 1, Role::Knight => 3, Role::Bishop => 3, Role::Rook => 5, Role::Queen => 9, Role::King => 0 }
}

/// The static exchange is the Rust MIRROR of the verified exchange law -- Lean: Ideas.ExchangeLaw.seeXB (the x-ray tier,
/// the capture game Ideas.Exchange.cheapestXF, nonnegative by Ideas.Exchange.cheapestXF_nonneg); its Python prototype
/// exchange_law.py `see_x`: equal to see_x on 3,028 of 3,028 attacked pieces (census sample, 2026-10-10). See
/// docs/LEAN_INDEX.md before writing another rule (Kaveh: Lean is the source of truth).
pub static SEE_THREATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// THE STATIC EXCHANGE on `t` for `us` capturing first (2026-10-10, Kaveh: "if a piece is attacked twice and defended
/// twice, it's stable ... we remove a defender ... that change should flag it"; game 10000007: two rooks on a pawn the
/// king alone defends were read as no threat -- the old rule counted a defender, never how many): the captures alternate,
/// the least valuable attacker first, pieces behind a slider joining as it leaves (the occupancy is updated), either
/// side may stop; a king may not capture onto a square the other side still attacks. -> what `us` wins (>= 0).
pub fn see(b: &shakmaty::Board, t: Square, us: Color) -> i64 {
    let Some(tp) = b.piece_at(t) else { return 0 };
    let mut occ = b.occupied();
    let mut gains: Vec<i64> = vec![val(tp.role)];
    let mut on_square = None::<i64>;
    let mut side = us;
    loop {
        let atk = b.attacks_to(t, side, occ) & occ & b.by_color(side);
        // the least valuable attacker; a king only if the other side has no attacker left on t
        let pick = atk.into_iter().filter_map(|s| b.piece_at(s).map(|p| (s, p.role))).min_by_key(|(s, r)| (if *r == Role::King { 100 } else { val(*r) }, *s as u32));
        let Some((s, r)) = pick else { break };
        if r == Role::King && (b.attacks_to(t, !side, occ.without(s)) & occ.without(s) & b.by_color(!side)).any() { break; }
        if let Some(v) = on_square { gains.push(v - gains.last().copied().unwrap_or(0)); }
        on_square = Some(if r == Role::King { 1000 } else { val(r) });
        occ = occ.without(s);
        side = !side;
    }
    if gains.len() == 1 && on_square.is_none() { return 0; }       // no attacker of ours
    // gains[i]: the material balance for the side making the i-th capture, before the reply; negamax back, each side may
    // stop instead of capturing
    for i in (1..gains.len()).rev() { gains[i - 1] = -(-gains[i - 1]).max(gains[i]); }
    gains[0].max(0)
}

/// {target square: best static gain} for `us` in `pos` (whoever is to move).
pub fn threats_of(pos: &Chess, us: Color) -> std::collections::HashMap<u8, i64> {
    if !SEE_THREATS.load(std::sync::atomic::Ordering::Relaxed) { return threats_of_prefilter(pos, us); }
    // the static exchange on every piece of theirs (the old Datalog prefilter -- undefended, or a cheaper attacker --
    // counted no attackers and defenders: kept as threats_of_prefilter)
    let b = pos.board();
    let mut out = std::collections::HashMap::new();
    for t in b.by_color(!us) {
        if b.piece_at(t).map_or(true, |p| p.role == Role::King) { continue; }
        let g = see(b, t, us);
        if g > 0 { out.insert(t as u8, g); }
    }
    out
}

/// The old Datalog prefilter (undefended, or a cheaper attacker).
pub fn threats_of_prefilter(pos: &Chess, us: Color) -> std::collections::HashMap<u8, i64> {
    let b = pos.board();
    let occ = b.occupied();
    let mut prog = ThreatRules::default();
    for sq in occ {
        if let Some(p) = b.piece_at(sq) {
            prog.piece.push((sq as u8, p.color == us, val(p.role)));
            for t in shakmaty::attacks::attacks(sq, p, occ) {
                prog.attacks.push((sq as u8, t as u8));
            }
        }
    }
    prog.run();
    prog.best.into_iter().collect()
}

pub fn square_name(s: u8) -> String { Square::new(s as u32).to_string() }

/// THE CORNERING FRAME (Kaveh 2026-10-08: "we should be able to mate any piece"; "mate is a specific type of
/// cornering"): their piece t is CORNERED in `q`
/// (them to move) when we win it where it stands (a gain g > 0 by the rules) and EVERY move it can make lands on a
/// square where we win it again. The king's version of this is mate. -> [(t, the smallest gain over staying and its
/// escapes)]. A trapped piece is usually cornered by a QUIET move that takes away its last escape, not by a new attack
/// (2026-10-08: 119 of 150 trapped-piece puzzles open with a quiet move; the piece falls two plies later in 104).
pub fn cornered(q: &Chess, us: Color) -> Vec<(u8, i64)> {
    let here = threats_of(q, us);
    let mut out = Vec::new();
    for (&t, &g) in here.iter() {
        let sq = Square::new(t as u32);
        let mut worst = g;
        let mut trapped = true;
        for mv in q.legal_moves() {
            if mv.from() != Some(sq) { continue; }
            let mut q2 = q.clone(); q2.play_unchecked(&mv);
            let to = mv.to() as u8;
            match threats_of(&q2, us).get(&to) {
                Some(&g2) if g2 > 0 => { if g2 < worst { worst = g2; } }
                _ => { trapped = false; break; }
            }
        }
        if trapped { out.push((t, worst)); }
    }
    // THE KING (Kaveh 2026-10-08: "mate is a specific type of cornering"): for the king "lost where it stands" is being
    // in check, and its escapes are its own moves -- a legal king move is never into attack, so the king is cornered
    // when it is in check and has no legal move of its own. Whether a block or a capture of the checker saves it is
    // the certificate's question (no rescue = mate). Valued at MATE_K (50: no material reaches it).
    let them = !us;
    if q.turn() == them && q.is_check() {
        if let Some(k) = q.board().king_of(them) {
            if !q.legal_moves().iter().any(|m| m.from() == Some(k)) {
                out.push((k as u8, 50));
            }
        }
    }
    out
}

/// A TRAPPED PIECE NOT YET ATTACKED (2026-10-08, puzzle 03Cdz: Bf3 takes the queen's last escape, Rh3 attacks it next):
/// their piece t (worth 3 or more) that we do not win where it stands, but every move of it lands on a square where we
/// win it. The attack comes next; the pursuit certificate follows it. -> [(t, its value)].
pub fn trapped_unattacked(q: &Chess, us: Color) -> Vec<(u8, i64)> {
    let here = threats_of(q, us);
    let them = !us;
    let mut out = Vec::new();
    for sq in q.board().by_color(them) {
        let Some(p) = q.board().piece_at(sq) else { continue };
        let v = val(p.role);
        if v < 3 || p.role == Role::King || here.contains_key(&(sq as u8)) { continue; }
        let mut any_move = false;
        let mut trapped = true;
        for mv in q.legal_moves() {
            if mv.from() != Some(sq) { continue; }
            any_move = true;
            let mut q2 = q.clone(); q2.play_unchecked(&mv);
            match threats_of(&q2, us).get(&(mv.to() as u8)) {
                Some(&g2) if g2 > 0 => {}
                _ => { trapped = false; break; }
            }
        }
        if any_move && trapped { out.push((sq as u8, v)); }
    }
    out
}
