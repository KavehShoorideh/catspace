//! cargo run --release --example policy -- <maia-XXXX.onnx> "<fen>"
use maia_chess::Maia;
use shakmaty::{fen::Fen, CastlingMode, Chess};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let maia = Maia::load(&args[1]).expect("load the ONNX network");
    let pos: Chess = args[2].parse::<Fen>().expect("FEN").into_position(CastlingMode::Standard).expect("legal position");
    let t = std::time::Instant::now();
    let pol = maia.policy(&pos, 1.0).expect("policy");
    let wdl = maia.wdl(&pos).expect("wdl");
    println!("{:?} ({:.1} ms)", wdl, t.elapsed().as_secs_f64() * 1000.0);
    for (uci, p) in pol.iter().take(8) { println!("{uci:6} {:5.1}%", p * 100.0); }
}
