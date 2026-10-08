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

/// {target square: best static gain} for `us` in `pos` (whoever is to move).
pub fn threats_of(pos: &Chess, us: Color) -> std::collections::HashMap<u8, i64> {
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
