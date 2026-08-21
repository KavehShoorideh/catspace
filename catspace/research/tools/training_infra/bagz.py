#!/usr/bin/env python
"""bagz.py -- minimal vendored reader for DeepMind ChessBench .bag files (gko1 U11/H3).

Format (from google-deepmind/searchless_chess src/bagz.py, Apache-2.0; re-implemented
here rather than imported to avoid the apache_beam dependency):
    [record bytes...][index: one <q per record = END offset of that record][<q index_start]
Records for 'action_value' are Beam TupleCoder(StrUtf8, StrUtf8, FloatCoder) encodings:
    varint(len(fen)) fen  varint(len(move)) move  float64_BE(win_prob)
(the LAST tuple component is encoded un-nested = no length prefix; Beam's FloatCoder is
big-endian float64). 'state_value' drops the move. The self-test VERIFIES these framing
assumptions on the real file: FEN parses, move is legal in that FEN, prob in [0,1].

    .venv/bin/python -m catspace.research.tools.training_infra.bagz --selftest
"""
from __future__ import annotations

import struct


class BagReader:
    def __init__(self, path: str):
        self._f = open(path, "rb")
        self._f.seek(-8, 2)
        file_end = self._f.tell()
        (index_start,) = struct.unpack("<q", self._f.read(8))
        self._index_start = index_start
        self._n = (file_end - index_start) // 8

    def __len__(self):
        return self._n

    def _end(self, i: int) -> int:
        self._f.seek(self._index_start + 8 * i)
        (e,) = struct.unpack("<q", self._f.read(8))
        return e

    def __getitem__(self, i: int) -> bytes:
        if not 0 <= i < self._n:
            raise IndexError(i)
        start = self._end(i - 1) if i else 0
        end = self._end(i)
        self._f.seek(start)
        return self._f.read(end - start)


def _varint(b: bytes, o: int):
    r = s = 0
    while True:
        x = b[o]; o += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, o
        s += 7


def decode_action_value(rec: bytes):
    n, o = _varint(rec, 0)
    fen = rec[o:o + n].decode("utf-8"); o += n
    n, o = _varint(rec, o)
    move = rec[o:o + n].decode("utf-8"); o += n
    (wp,) = struct.unpack(">d", rec[o:o + 8])
    return fen, move, wp


def decode_state_value(rec: bytes):
    n, o = _varint(rec, 0)
    fen = rec[o:o + n].decode("utf-8"); o += n
    (wp,) = struct.unpack(">d", rec[o:o + 8])
    return fen, wp


def _selftest():
    import sys
    import chess
    ok = True
    r = BagReader("data/external/chessbench/test_action_value.bag")
    print(f"  action_value: {len(r):,} records")
    import random
    rng = random.Random(0)
    probes = list(range(200)) + [rng.randrange(len(r)) for _ in range(800)]
    bad = 0
    fens = set()
    for i in probes:
        try:
            fen, mv, wp = decode_action_value(r[i])
            b = chess.Board(fen)
            legal = chess.Move.from_uci(mv) in b.legal_moves
            if not (legal and 0.0 <= wp <= 1.0):
                bad += 1
            fens.add(fen)
        except Exception:
            bad += 1
    print(f"  probes 1000: bad {bad} | distinct FENs {len(fens)} | "
          f"sample wp {decode_action_value(r[0])[2]:.4f} move {decode_action_value(r[0])[1]}")
    ok &= bad == 0
    rs = BagReader("data/external/chessbench/test_state_value.bag")
    bad2 = 0
    for i in range(min(500, len(rs))):
        try:
            fen, wp = decode_state_value(rs[i])
            chess.Board(fen)
            if not 0.0 <= wp <= 1.0:
                bad2 += 1
        except Exception:
            bad2 += 1
    print(f"  state_value: {len(rs):,} records | bad {bad2}/500")
    ok &= bad2 == 0
    print(f"  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
