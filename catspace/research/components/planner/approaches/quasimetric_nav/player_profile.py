#!/usr/bin/env python
"""player_profile.py -- per-player memory for KittyChess (Kaveh 2026-08-13: "each game you
play against it should give it information on how to play against you").

STORAGE ABSTRACTION (Kaveh 2026-08-13: "make these stores sit on top of abstract unit
systems"): the domain logic never touches files. It composes two storage units:

    KVStore    get/put/delete/keys -- JSON values under string keys, namespaced
    AppendLog  append/scan         -- an append-only record stream, namespaced

LocalBackend implements both on the filesystem (JSON file per KV namespace, JSONL per log).
A cloud backend (S3/R2 object store for logs, Redis/DynamoDB for KV) implements the same
four+two methods and PlayerStore ports unchanged -- mirroring infra/'s vendor abstraction.

Per player (namespace prefix players/<name>/):
    log "games"     every ply: fen, move, expectation dist, surprisal bits, one-forward dE
    kv  "surprises" positions where the engine was surprised; re-met calmly = gotcha
    kv  "prep"      idle-time deep-search results (the between-sessions "study")
    kv  "profile"   aggregates incl. systematic weaknesses (LOW surprisal + HIGH dE loss)

Surprise is measured in BITS: the engine ponders your position, builds P(move) =
softmax(E_mover/tau) over your legal moves from its own child-E ranking, and your actual
move scores -log2 P(move). tau=0.03 E -- the same softness the analysis board's effective-
move count uses, until enough logged moves calibrate a per-player tau.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from abc import ABC, abstractmethod

import chess

from catspace.io import paths

TAU_E = 0.03          # softmax temperature in E units (matches the eff_moves scale)
EPS_UNSEEN = 0.02     # probability mass reserved for legal moves the search never ranked
SURPRISE_BITS = 3.0   # >= this -> the engine is surprised (P < ~1/8)
CALM_BITS = 1.5       # <= this at a previously-surprising position -> gotcha


# ---------------------------------------------------------------------------------------
# storage units
# ---------------------------------------------------------------------------------------
class KVStore(ABC):
    """JSON values under string keys. Namespaced by construction."""

    @abstractmethod
    def get(self, key: str, default=None): ...

    @abstractmethod
    def put(self, key: str, value) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def keys(self) -> list[str]: ...


class AppendLog(ABC):
    """append-only record stream. Namespaced by construction."""

    @abstractmethod
    def append(self, rec: dict) -> None: ...

    @abstractmethod
    def scan(self):
        """yield every record, oldest first."""
        ...


class Backend(ABC):
    """a storage vendor: hands out namespaced units."""

    @abstractmethod
    def kv(self, namespace: str) -> KVStore: ...

    @abstractmethod
    def log(self, namespace: str) -> AppendLog: ...


class _FileKV(KVStore):
    """one JSON file per namespace; atomic replace on write; write-through cache."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self._d = json.load(open(path))
        except Exception:
            self._d = {}

    def _flush(self):
        tmp = self.path + ".tmp"
        json.dump(self._d, open(tmp, "w"))
        os.replace(tmp, self.path)

    def get(self, key, default=None):
        return self._d.get(key, default)

    def put(self, key, value):
        self._d[key] = value
        self._flush()

    def delete(self, key):
        if key in self._d:
            del self._d[key]
            self._flush()

    def keys(self):
        return list(self._d.keys())


class _FileLog(AppendLog):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, rec):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def scan(self):
        try:
            with open(self.path) as f:
                for ln in f:
                    try:
                        yield json.loads(ln)
                    except Exception:
                        continue
        except FileNotFoundError:
            return


class LocalBackend(Backend):
    def __init__(self, root: str | None = None):
        self.root = root or str(paths.experiment("players"))

    def kv(self, namespace):
        return _FileKV(os.path.join(self.root, namespace + ".json"))

    def log(self, namespace):
        return _FileLog(os.path.join(self.root, namespace + ".jsonl"))


# ---------------------------------------------------------------------------------------
# surprise math
# ---------------------------------------------------------------------------------------
def fen_key(b: chess.Board) -> str:
    """position identity for memory: board + turn + castling + ep (counters stripped)."""
    return " ".join(b.fen().split(" ")[:4])


def expectation_dist(rows, human_white, legal_ucis):
    """rows (rank_by_child_E output, E = white-POV) -> P(move) over the human's legal moves.
    Unranked legal moves share EPS_UNSEEN so surprisal stays finite."""
    ranked = {}
    for r in rows:
        u = r["mv"].uci() if hasattr(r.get("mv"), "uci") else r.get("uci")
        e = r.get("E")
        if u is None or e is None or u in ranked:
            continue
        ranked[u] = float(e) if human_white else 1.0 - float(e)
    if not ranked:
        n = max(1, len(legal_ucis))
        return {u: 1.0 / n for u in legal_ucis}
    mx = max(ranked.values())
    ws = {u: math.exp((e - mx) / TAU_E) for u, e in ranked.items()}
    z = sum(ws.values())
    unseen = [u for u in legal_ucis if u not in ranked]
    scale = (1.0 - EPS_UNSEEN) if unseen else 1.0
    dist = {u: scale * w / z for u, w in ws.items()}
    for u in unseen:
        dist[u] = EPS_UNSEEN / len(unseen)
    return dist


def surprisal_bits(dist, uci) -> float:
    return -math.log2(max(dist.get(uci, EPS_UNSEEN / 20), 1e-9))


# ---------------------------------------------------------------------------------------
# the player store (pure domain logic over Backend units)
# ---------------------------------------------------------------------------------------
class PlayerStore:
    def __init__(self, name: str, backend: Backend | None = None):
        self.name = re.sub(r"[^A-Za-z0-9_-]", "_", name.strip())[:32] or "anon"
        be = backend or LocalBackend()
        ns = self.name + "/"
        self.games = be.log(ns + "games")
        self.surprises = be.kv(ns + "surprises")
        self.prep = be.kv(ns + "prep")
        self.profile = be.kv(ns + "profile")

    # ---- per-ply log --------------------------------------------------------------------
    def log_ply(self, rec: dict):
        self.games.append({"t": round(time.time(), 1), **rec})

    # ---- surprise memory (the emote substrate) ------------------------------------------
    def note_surprise(self, key: str, uci: str, bits: float, san: str):
        k = key + "|" + uci
        r = self.surprises.get(k) or {"n": 0, "bits": 0.0, "san": san, "avenged": 0}
        r["n"] += 1
        r["bits"] = max(float(r["bits"]), round(bits, 2))
        self.surprises.put(k, r)

    def check_gotcha(self, key: str, uci: str, bits_now: float):
        """the same move that once surprised the engine, now fully expected -> gotcha.
        Requires the surprise to be from a PRIOR visit (note_surprise happens after)."""
        r = self.surprises.get(key + "|" + uci)
        if r and bits_now <= CALM_BITS:
            r["avenged"] = int(r.get("avenged", 0)) + 1
            self.surprises.put(key + "|" + uci, r)
            return r
        return None

    def was_surprised_here(self, key: str, uci: str) -> bool:
        return self.surprises.get(key + "|" + uci) is not None

    # ---- prep cache (idle-time study) ---------------------------------------------------
    def get_prep(self, key: str):
        return self.prep.get(key)

    def put_prep(self, key: str, entry: dict):
        self.prep.put(key, {**entry, "t": round(time.time(), 1)})

    def pending_prep(self, limit=64):
        """positions from this player's logged games not yet studied, weakness-first:
        low-surprisal + high dE-loss plies (your SYSTEMATIC mistakes) get priority."""
        prepped = set(self.prep.keys())
        seen, rows = set(), []
        for r in self.games.scan():
            if r.get("type") == "ply" and r.get("fen_before"):
                k = " ".join(r["fen_before"].split(" ")[:4])
                if k in seen or k in prepped:
                    continue
                seen.add(k)
                dE = float(r.get("dE_mover", 0.0))
                bits = float(r.get("bits", 99.0))
                weak = (bits <= CALM_BITS and dE < -0.03)
                rows.append((0 if weak else 1, -abs(dE), r["fen_before"], r.get("mover")))
        rows.sort()
        return [(fen, mover) for _w, _d, fen, mover in rows[:limit]]

    # ---- aggregates ---------------------------------------------------------------------
    def aggregate(self):
        games, bits, weak, n_ply = set(), [], [], 0
        for r in self.games.scan():
            if r.get("type") != "ply":
                continue
            n_ply += 1
            games.add(r.get("game", 0))
            if r.get("bits") is not None:
                bits.append(float(r["bits"]))
            dE = float(r.get("dE_mover", 0.0))
            if r.get("bits", 99) <= CALM_BITS and dE < -0.03:
                weak.append({"fen": r["fen_before"], "san": r.get("san"),
                             "dE": round(dE, 3)})
        srec = [self.surprises.get(k) for k in self.surprises.keys()]
        prof = {"name": self.name, "games": len(games), "plies": n_ply,
                "mean_bits": round(sum(bits) / len(bits), 2) if bits else None,
                "n_surprises": len(srec),
                "n_avenged": sum(int(r.get("avenged", 0)) for r in srec if r),
                "n_prepped": len(self.prep.keys()),
                "weaknesses": sorted(weak, key=lambda w: w["dE"])[:20]}
        self.profile.put("current", prof)
        return prof


def _tests():
    ok = True
    b = chess.Board()
    legal = [m.uci() for m in b.legal_moves]
    rows = [{"uci": "e2e4", "E": 0.55}, {"uci": "d2d4", "E": 0.55},
            {"uci": "a2a3", "E": 0.45}]
    d = expectation_dist(rows, human_white=True, legal_ucis=legal)
    ok &= abs(sum(d.values()) - 1.0) < 1e-6
    ok &= d["e2e4"] > d["a2a3"]                       # better E -> more expected
    ok &= surprisal_bits(d, "a2a3") > surprisal_bits(d, "e2e4")
    ok &= surprisal_bits(d, "g2g4") > SURPRISE_BITS   # unranked = surprising
    # black POV flips: low white-E moves become the expected ones for black
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    legal2 = [m.uci() for m in b2.legal_moves]
    rows2 = [{"uci": "e7e5", "E": 0.45}, {"uci": "f7f5", "E": 0.60}]
    d2 = expectation_dist(rows2, human_white=False, legal_ucis=legal2)
    ok &= d2["e7e5"] > d2["f7f5"]
    print(f"[dist] sums to 1, orders by mover-E, unseen surprising, POV flips  "
          f"{'OK' if ok else 'FAIL'}")
    # store over an abstract backend: roundtrip, gotcha contract, weakness-first prep.
    # An in-memory backend proves the domain logic never touches the filesystem.
    class MemKV(KVStore):
        def __init__(self):
            self._d = {}

        def get(self, k, default=None):
            return self._d.get(k, default)

        def put(self, k, v):
            self._d[k] = v

        def delete(self, k):
            self._d.pop(k, None)

        def keys(self):
            return list(self._d)

    class MemLog(AppendLog):
        def __init__(self):
            self._r = []

        def append(self, rec):
            self._r.append(rec)

        def scan(self):
            yield from self._r

    class MemBackend(Backend):
        def __init__(self):
            self._u = {}

        def kv(self, ns):
            return self._u.setdefault(("kv", ns), MemKV())

        def log(self, ns):
            return self._u.setdefault(("log", ns), MemLog())

    for be in (MemBackend(),):
        st = PlayerStore("Test User!", backend=be)
        k = fen_key(b)
        ok &= st.check_gotcha(k, "g2g4", 0.5) is None          # never surprised yet
        st.note_surprise(k, "g2g4", 4.4, "g4")
        ok &= st.check_gotcha(k, "g2g4", 3.9) is None          # still surprised: no gotcha
        g = st.check_gotcha(k, "g2g4", 0.8)                    # calm now -> gotcha
        ok &= g is not None and g["avenged"] == 1
        st.log_ply({"type": "ply", "game": 1, "fen_before": b.fen(), "san": "g4",
                    "bits": 0.5, "dE_mover": -0.08, "mover": "human"})
        st.log_ply({"type": "ply", "game": 1, "fen_before": b2.fen(), "san": "e5",
                    "bits": 5.0, "dE_mover": 0.01, "mover": "human"})
        pend = st.pending_prep()
        ok &= len(pend) == 2 and pend[0][0] == b.fen()          # weakness prepped FIRST
        st.put_prep(fen_key(b), {"uci": "d2d4", "budget": 6.0})
        ok &= len(st.pending_prep()) == 1
        prof = st.aggregate()
        ok &= prof["games"] == 1 and prof["n_avenged"] == 1 and len(prof["weaknesses"]) == 1
        st2 = PlayerStore("Test User!", backend=be)             # persistence via the backend
        ok &= st2.was_surprised_here(k, "g2g4") and st2.get_prep(fen_key(b)) is not None
    print(f"[store] surprise->gotcha ordering, weakness-first prep, backend-portable  "
          f"{'OK' if ok else 'FAIL'}")
    # the file backend honors the same contract incl. cross-instance persistence
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        st = PlayerStore("kav", backend=LocalBackend(td))
        st.note_surprise("K", "e2e4", 5.0, "e4")
        st.log_ply({"type": "ply", "game": 2, "fen_before": b.fen(), "bits": 1.0,
                    "dE_mover": 0.0, "mover": "human"})
        st2 = PlayerStore("kav", backend=LocalBackend(td))
        ok &= st2.was_surprised_here("K", "e2e4")
        ok &= sum(1 for _ in st2.games.scan()) == 1
        ok &= os.path.exists(os.path.join(td, "kav", "surprises.json"))
    print(f"[file] LocalBackend same contract, files land under the player dir  "
          f"{'OK' if ok else 'FAIL'}")
    print("ALL PLAYER-PROFILE TESTS PASSED" if ok else "TESTS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _tests()
