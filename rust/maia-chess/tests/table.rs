//! The move table and the encoding, without any network.
use maia_chess::{encode, move_index, policy_index, PLANES};
use shakmaty::{fen::Fen, CastlingMode, Chess, Position};

fn pos(fen: &str) -> Chess { fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap() }

#[test]
fn the_table_has_1858_distinct_moves() {
    let t = policy_index();
    assert_eq!(t.len(), 1858);
    let mut idx: Vec<usize> = t.values().copied().collect(); idx.sort(); idx.dedup();
    assert_eq!(idx.len(), 1858);
}

#[test]
fn every_legal_move_has_an_index_in_varied_positions() {
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/pppqppbp/2np1np1/8/3PP3/2N1BN2/PPPQ1PPP/R3KB1R b KQkq - 0 1",   // black castling both ways
                "1r5k/P7/8/8/8/8/6K1/8 w - - 0 1",                                      // promotions, all pieces
                "8/6K1/8/8/8/8/p6k/1R6 b - - 0 1",
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"] {      // en passant
        let p = pos(fen);
        for m in p.legal_moves() { assert!(move_index(&m, p.turn()).is_some(), "{fen}: {m:?}"); }
    }
}

#[test]
fn black_sees_the_mirror_of_white() {
    let w = encode(&pos("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"));
    let b = encode(&pos("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1"));
    assert_eq!(w.len(), PLANES * 64);
    for plane in 0..104 { assert_eq!(&w[plane * 64..plane * 64 + 64], &b[plane * 64..plane * 64 + 64], "plane {plane}"); }
}
