# maia-chess

**Maia — human-like chess move prediction — in pure Rust.**

[Maia](https://github.com/CSSLab/maia-chess) (McIlroy-Young, Sen, Kleinberg & Anderson, *Aligning Superhuman AI with
Human Behavior: Chess as a Model System*, KDD 2020) predicts the move a human of a given rating would play. Its networks
are [Leela Chess Zero](https://github.com/LeelaChessZero/lc0) networks, normally run through the lc0 engine. This crate
runs them natively in Rust — no engine process, no UCI round trip:

- lc0's classical **112-plane input encoding**, with real game history or lc0's own history-fill rules for a bare
  position (none at the standard start; the position before the double step when en passant is possible);
- lc0's **1,858-move policy table**, read out of lc0's own `--verbose-move-stats` output (castling = king takes its own
  rook; a pawn's plain move to the last rank = knight promotion);
- inference with [tract](https://github.com/sonos/tract) (pure-Rust ONNX) on the networks converted by `lc0 leela2onnx`;
- the **policy over the legal moves** at any softmax temperature (Maia's own distribution at 1.0) and the **win/draw/loss** head;
- optional **Python bindings** (`maturin`).

## Accuracy and speed

Checked against lc0 itself (`tests/bridge.rs`, `MAIA_BRIDGE_WEIGHTS` / `MAIA_BRIDGE_ONNX`):

| | agreement with lc0 |
|---|---|
| bare positions (204 census positions, Maia-1100 and Maia-1900) | same top move 203/204 at both bands; the residual is a first-move opening given without history |
| with the real game history (startpos + moves) | same top move 3/3, largest probability difference 0.0003 |

About 0.4 ms per position on an Apple M-series CPU, single thread, batch 1.

## Use

```sh
scripts/convert.sh path/to/maia_weights out/        # maia-1100.pb.gz ... maia-1900.pb.gz -> .onnx (needs lc0)
cargo run --release --example policy -- out/maia-1500.onnx "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
```

```rust
let maia = maia_chess::Maia::load("out/maia-1500.onnx")?;
let probs = maia.policy(&position, 1.0)?;          // [(uci, p)], most likely first
let probs = maia.policy_history(&history, 1.0)?;   // with the game so far (oldest first)
let wdl = maia.wdl(&position)?;
```

```python
# maturin develop --release   (feature "python")
import maia_chess
m = maia_chess.Maia("out/maia-1500.onnx")
m.policy("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")   # [(uci, p), ...]
```

Note on temperature: lc0 displays its policy at `--policy-softmax-temp` 1.359 by default; Maia's own move
distribution is temperature 1.0. Use 1.359 only to reproduce lc0's numbers.

## License

GPL-3.0-only, like Maia and lc0 (the move table is read from lc0). The Maia weights are not included; they are
GPL-3.0 and available from the Maia repository.

## Credits

Maia: the CSSLab team (University of Toronto / Cornell / Microsoft Research). lc0: the Leela Chess Zero authors. tract:
Sonos. shakmaty: Niklas Fiekas.
