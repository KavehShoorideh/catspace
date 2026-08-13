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

TAU_E = 0.05          # default softmax temperature in E units; per-player fit overrides
EPS_UNSEEN = 0.02     # probability mass reserved for legal moves the search never ranked
# EXCESS surprisal (2026-08-13, 'it gets surprised by everything'): the field's one-ply E
# is near-flat across sane moves (sibling-wall: top-2 gaps ~0.003 E), so ABSOLUTE bits sit
# at 3-4 for every move. Surprise is bits ABOVE the distribution's own entropy -- in a
# position where the engine spreads H bits of uncertainty, an H-bit move is average.
SURPRISE_BITS = 1.7   # excess bits >= this -> surprised
CALM_BITS = 0.5       # excess bits <= this at a previously-surprising move -> gotcha
TAU_GRID = (0.02, 0.03, 0.05, 0.08, 0.12, 0.20)   # per-player NLL fit search space
TAU_MIN_MOVES = 40    # logged moves needed before the per-player tau is trusted
# ANNEALED temperature (Kaveh 2026-08-13: "first times we play someone the temperature is
# really high and settles down as we get more data"): a stranger gets near-uniform
# expectations (hard to surprise -- the engine KNOWS it doesn't know you); the temperature
# glides toward the fitted/default value as logged plies accumulate.
TAU_HI = 0.30         # stranger temperature
TAU_HALF_PLIES = 60   # plies at which the anneal is halfway to the settled tau


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


def mover_E(r):
    """mover-POV expected score of a RAW search row. rank_by_child_E writes child_E
    (calibrated one-ply, mover POV) on the head; every row carries the searched value
    (mover POV, 1000*E scale; mate proofs +-1e6 clamp to certainty). The '\"E\" key'
    bug (2026-08-13, 'it keeps getting surprised by every move'): display rows have E,
    raw rows do NOT -- reading E here silently made every distribution uniform."""
    if r.get("child_E") is not None:
        return float(r["child_E"])
    v = r.get("value")
    if v is not None:
        return min(1.0, max(0.0, float(v) / 1000.0))
    return None


def ranked_E(rows):
    """-> {uci: mover-POV E} from raw search rows (first occurrence wins)."""
    out = {}
    for r in rows:
        u = r["mv"].uci() if hasattr(r.get("mv"), "uci") else r.get("uci")
        e = mover_E(r)
        if u is not None and e is not None and u not in out:
            out[u] = e
    return out


def dist_from_E(ranked, legal_ucis, tau=TAU_E):
    """{uci: mover E} -> P(move) over the mover's legal moves. Unranked legal moves share
    EPS_UNSEEN so surprisal stays finite."""
    if not ranked:
        n = max(1, len(legal_ucis))
        return {u: 1.0 / n for u in legal_ucis}
    mx = max(ranked.values())
    ws = {u: math.exp((e - mx) / tau) for u, e in ranked.items()}
    z = sum(ws.values())
    unseen = [u for u in legal_ucis if u not in ranked]
    scale = (1.0 - EPS_UNSEEN) if unseen else 1.0
    dist = {u: scale * w / z for u, w in ws.items()}
    for u in unseen:
        dist[u] = EPS_UNSEEN / len(unseen)
    return dist


def expectation_dist(rows, legal_ucis, tau=TAU_E):
    """raw search rows -> P(move) over the mover's legal moves (mover POV throughout)."""
    return dist_from_E(ranked_E(rows), legal_ucis, tau)


def surprisal_bits(dist, uci) -> float:
    return -math.log2(max(dist.get(uci, EPS_UNSEEN / 20), 1e-9))


def entropy_bits(dist) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def excess_bits(dist, uci) -> float:
    """surprisal RELATIVE to the position's expected surprisal (the surprise ruler)."""
    return surprisal_bits(dist, uci) - entropy_bits(dist)


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
                xb = float(r.get("xbits", 99.0))
                weak = (xb <= CALM_BITS and dE < -0.03)
                rows.append((0 if weak else 1, -abs(dE), r["fen_before"], r.get("mover")))
        rows.sort()
        return [(fen, mover) for _w, _d, fen, mover in rows[:limit]]

    # ---- per-player expectation temperature ---------------------------------------------
    def tau(self) -> float:
        """annealed: TAU_HI for a stranger -> fitted (or default) tau as data accumulates.
        n comes from the aggregated profile, so the anneal steps once per finished game."""
        prof = self.profile.get("current") or {}
        n = int(prof.get("plies") or 0)
        base = float(prof.get("tau") or TAU_E)
        w = n / (n + TAU_HALF_PLIES)
        return TAU_HI * (1.0 - w) + base * w

    def _fit_tau(self):
        """max-likelihood tau over this player's logged (e_top, played) pairs -- 'how far
        below the engine's best does THIS human actually play'. Needs TAU_MIN_MOVES."""
        obs = []
        for r in self.games.scan():
            if r.get("type") == "ply" and r.get("mover") == "human" and r.get("e_top"):
                obs.append((dict(r["e_top"]), r["uci"], int(r.get("n_legal", 0))))
        if len(obs) < TAU_MIN_MOVES:
            return None
        best_tau, best_nll = None, None
        for t in TAU_GRID:
            nll = 0.0
            for etop, uci, n_legal in obs:
                d = dist_from_E(etop, list(etop.keys()) +
                                ["_other"] * max(0, n_legal - len(etop)), tau=t)
                nll -= math.log2(max(d.get(uci, d.get("_other", 1e-9)), 1e-9))
            if best_nll is None or nll < best_nll:
                best_tau, best_nll = t, nll
        return best_tau

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
            if float(r.get("xbits", 99)) <= CALM_BITS and dE < -0.03:
                weak.append({"fen": r["fen_before"], "san": r.get("san"),
                             "dE": round(dE, 3)})
        srec = [self.surprises.get(k) for k in self.surprises.keys()]
        prof = {"name": self.name, "tau": self._fit_tau(),
                "games": len(games), "plies": n_ply,
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
    # RAW search-row shapes (the 2026-08-13 uniform-dist bug): child_E on the ranked head,
    # bare value (1000*E, mover POV) on the tail, mate at +-1e6 -- NEVER a bare "E" key
    rows = [{"uci": "e2e4", "child_E": 0.55}, {"uci": "d2d4", "child_E": 0.55},
            {"uci": "a2a3", "child_E": 0.45}, {"uci": "h2h4", "value": 430.0}]
    d = expectation_dist(rows, legal_ucis=legal)
    ok &= abs(sum(d.values()) - 1.0) < 1e-6
    ok &= d["e2e4"] > d["a2a3"] > d["h2h4"]           # better mover-E -> more expected
    ok &= surprisal_bits(d, "a2a3") > surprisal_bits(d, "e2e4")
    ok &= surprisal_bits(d, "g2g4") > SURPRISE_BITS   # unranked = surprising
    ok &= d["e2e4"] > 0.15                            # a top move is genuinely EXPECTED,
                                                      # not uniform-mush (the bug's signature)
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    ok &= mover_E({"value": 1e6}) == 1.0 and mover_E({"value": -1e6}) == 0.0   # mate clamps
    ok &= mover_E({"uci": "x"}) is None               # display-style rows refuse silently no
    # tau sharpens/softens: lower tau concentrates on the best move
    dsharp = expectation_dist(rows, legal, tau=0.02)
    dsoft = expectation_dist(rows, legal, tau=0.20)
    ok &= dsharp["e2e4"] > d["e2e4"] > dsoft["e2e4"]
    # EXCESS surprisal self-calibrates: in a flat dist the modal move is NOT surprising
    ok &= excess_bits(d, "e2e4") < 1.0                # engine-favored: calm
    ok &= excess_bits(d, "g2g4") > SURPRISE_BITS      # unranked: still a shock
    flat = {u: 1.0 / len(legal) for u in legal}       # totally flat position:
    ok &= abs(excess_bits(flat, "e2e4")) < 1e-6       # NOTHING is surprising
    print(f"[dist] raw-row keys (child_E/value), mate clamp, tau ordering  "
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
        g = st.check_gotcha(k, "g2g4", 0.3)                    # calm now -> gotcha
        ok &= g is not None and g["avenged"] == 1
        st.log_ply({"type": "ply", "game": 1, "fen_before": b.fen(), "san": "g4",
                    "bits": 3.0, "xbits": 0.1, "dE_mover": -0.08, "mover": "human"})
        st.log_ply({"type": "ply", "game": 1, "fen_before": b2.fen(), "san": "e5",
                    "bits": 7.0, "xbits": 4.0, "dE_mover": 0.01, "mover": "human"})
        pend = st.pending_prep()
        ok &= len(pend) == 2 and pend[0][0] == b.fen()          # weakness prepped FIRST
        st.put_prep(fen_key(b), {"uci": "d2d4", "budget": 6.0})
        ok &= len(st.pending_prep()) == 1
        # tau anneal: stranger = TAU_HI, softens toward the base as plies accumulate
        ok &= abs(PlayerStore("fresh", backend=be).tau() - TAU_HI) < 1e-9
        prof = st.aggregate()
        ok &= prof["games"] == 1 and prof["n_avenged"] == 1 and len(prof["weaknesses"]) == 1
        ok &= st.tau() < TAU_HI                                # 2 plies: barely settled
        st.profile.put("current", {**prof, "plies": 600})
        ok &= st.tau() < 0.09                                  # 600 plies: near the base
        st2 = PlayerStore("Test User!", backend=be)             # persistence via the backend
        ok &= st2.was_surprised_here(k, "g2g4") and st2.get_prep(fen_key(b)) is not None
    print(f"[store] surprise->gotcha ordering, weakness-first prep, backend-portable  "
          f"{'OK' if ok else 'FAIL'}")
    # the file backend honors the same contract incl. cross-instance persistence
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        st = PlayerStore("kav", backend=LocalBackend(td))
        st.note_surprise("K", "e2e4", 5.0, "e4")
        st.log_ply({"type": "ply", "game": 2, "fen_before": b.fen(), "bits": 1.0, "xbits": 0.2,
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
