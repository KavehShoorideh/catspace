//! names.rs -- the name of every dimension in the basis.
//!
//! These MUST equal `concepts_reference.SQ_NAMES` / `GLOB_NAMES` element-for-element. The whole
//! point of the basis is that a distilled concept prints as a weighted expression over named
//! primitives; a name that drifts from the column it labels turns every printed program into a
//! quiet lie. `test_probe_features.py` asserts the two lists are identical before it compares a
//! single number.

pub const SYM: [&str; 12] = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"];

/// Per-square feature names, in column order. 28 dims.
pub fn sq_names() -> Vec<String> {
    let mut v: Vec<String> = SYM.iter().map(|s| format!("occ_{}", s)).collect();
    for n in [
        "att_w", "att_b",
        "least_att_w", "least_att_b",
        "see",
        "pinned", "pinning",
        "xray_w", "xray_b",
        "pawn_att_w", "pawn_att_b",
        "hanging", "defenders",
        "attacks_count",
        "king_ring_w", "king_ring_b",
    ] {
        v.push(n.to_string());
    }
    v
}

/// Position-level feature names, in column order. 100 dims.
pub fn glob_names() -> Vec<String> {
    let mut v: Vec<String> = SYM.iter().map(|s| format!("n_{}", s)).collect();
    for n in ["material_diff", "phase_nonpawn", "piece_count"] {
        v.push(n.to_string());
    }
    for kind in ["passed", "isolated", "doubled", "backward"] {
        for c in ["w", "b"] {
            for f in 0..8 {
                v.push(format!("pawn_{}_{}_file{}", kind, c, f));
            }
        }
    }
    for kind in ["ring_pressure", "shield", "escape", "open_files"] {
        for c in ["w", "b"] {
            v.push(format!("king_{}_{}", kind, c));
        }
    }
    for n in [
        "stm_white", "castle_K", "castle_Q", "castle_k", "castle_q",
        "ep_file", "in_check", "checker_count",
        "tension", "legal_moves", "checks_available",
        "repetition", "halfmove",
    ] {
        v.push(n.to_string());
    }
    v
}

pub const N_SQ: usize = 28;
pub const N_GLOB: usize = 100;
