//! The bridge to lc0 itself (skipped unless `MAIA_BRIDGE_WEIGHTS` names a maia-XXXX.pb.gz and `MAIA_BRIDGE_ONNX` its
//! conversion, with lc0 on PATH): the probabilities lc0 prints (--verbose-move-stats, its default policy temperature
//! 1.359) against ours at the same temperature, on positions with castling, promotions and en passant.
use maia_chess::Maia;
use shakmaty::{fen::Fen, CastlingMode, Chess, Position};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[test]
fn matches_lc0() {
    let (Ok(w), Ok(onnx)) = (std::env::var("MAIA_BRIDGE_WEIGHTS"), std::env::var("MAIA_BRIDGE_ONNX")) else { return };
    let maia = Maia::load(&onnx).unwrap();
    let mut child = Command::new("lc0").args([format!("--weights={w}").as_str(), "--verbose-move-stats", "--threads=1", "--backend=eigen"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().expect("lc0 on PATH");
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, "uci\nisready").unwrap();
    let mut line = String::new();
    loop { line.clear(); out.read_line(&mut line).unwrap(); if line.trim() == "readyok" { break; } }
    let fens = ["r3k2r/pppqppbp/2np1np1/8/3PP3/2N1BN2/PPPQ1PPP/R3KB1R w KQkq - 0 1",
                "r1bq1rk1/pp3ppp/2n1pn2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 2 8",
                "8/1P4k1/8/8/8/8/6K1/8 w - - 0 1",
                "r1bqkbnr/ppp3pp/2n1p3/3pPp2/3P4/2P5/PP3PPP/RNBQKBNR w KQkq f6 0 5"];
    let mut agree = 0;
    for fen in fens {
        writeln!(stdin, "position fen {fen}\ngo nodes 1").unwrap();
        let mut lc: Vec<(String, f32)> = Vec::new();
        loop {
            line.clear(); out.read_line(&mut line).unwrap();
            if line.starts_with("bestmove") { break; }
            if let Some(rest) = line.strip_prefix("info string ") {
                let u = rest.split_whitespace().next().unwrap_or("").to_string();
                if let Some(i) = rest.find("(P:") {
                    let p: f32 = rest[i + 3..].trim_start().split('%').next().unwrap().trim().parse().unwrap_or(0.0);
                    if u.len() >= 4 && u != "node" { lc.push((u, p / 100.0)); }
                }
            }
        }
        let pos: Chess = fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap();
        let ours = maia.policy(&pos, 1.359).unwrap();
        let lc_top = lc.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0.clone();
        let castle = |u: &str| -> String { match u { "e1h1" => "e1g1", "e1a1" => "e1c1", "e8h8" => "e8g8", "e8a8" => "e8c8", x => x }.to_string() };
        if castle(&lc_top) == ours[0].0 { agree += 1; }
        let _ = pos.turn();
    }
    // with the real history: lc0 given "startpos moves ...", ours given the positions along the way
    let lines: [&[&str]; 3] = [&["g1f3"], &["e2e4", "c7c5"], &["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]];
    let mut hist_agree = 0; let mut worst = 0f32;
    for moves in lines {
        writeln!(stdin, "position startpos moves {}\ngo nodes 1", moves.join(" ")).unwrap();
        let mut lc: Vec<(String, f32)> = Vec::new();
        loop {
            line.clear(); out.read_line(&mut line).unwrap();
            if line.starts_with("bestmove") { break; }
            if let Some(rest) = line.strip_prefix("info string ") {
                let u = rest.split_whitespace().next().unwrap_or("").to_string();
                if let Some(i) = rest.find("(P:") {
                    let p: f32 = rest[i + 3..].trim_start().split('%').next().unwrap().trim().parse().unwrap_or(0.0);
                    if u.len() >= 4 && u != "node" { lc.push((u, p / 100.0)); }
                }
            }
        }
        let mut hist: Vec<Chess> = vec![Chess::default()];
        for u in moves {
            let p = hist.last().unwrap().clone();
            let m = u.parse::<shakmaty::uci::UciMove>().unwrap().to_move(&p).unwrap();
            hist.push(p.play(&m).unwrap());
        }
        let ours = maia.policy_history(&hist, 1.359).unwrap();
        let lc_top = lc.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0.clone();
        if lc_top == ours[0].0 { hist_agree += 1; }
        for (u, p) in &lc { if let Some((_, q)) = ours.iter().find(|(v, _)| v == u) { worst = worst.max((p - q).abs()); } }
    }
    writeln!(stdin, "quit").unwrap();
    eprintln!("bare positions: top move agrees {agree}/{}; with history: {hist_agree}/3, worst probability difference {worst:.4}", fens.len());
    assert_eq!(agree, fens.len(), "top move differs from lc0");
    assert_eq!(hist_agree, 3, "top move with history differs from lc0");
}
