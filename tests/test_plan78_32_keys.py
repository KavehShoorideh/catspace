"""Plan 78.32 (the hand step): the key-square words of rules/concepts/chess_keys.gdl -- each equals an independent python-chess oracle on small and full-board positions."""
import json
import random

import chess

import cs_engine as CE
from catspace.io import paths

RULES = paths.REPO_ROOT / "rules"
VJ = json.loads((RULES / "learned/chess/vocabulary.keys.json").read_text())
TEXT = (RULES / VJ["spec"]).read_text()
for _c in VJ["concepts"]:
    TEXT = CE.compose(TEXT, (RULES / _c).read_text())
NEW = ["key_gap", "key_guard", "king_in_front"]
VOC = json.dumps({"derived": NEW, "delta": ["cell"], "units": None})
SIDE = {chess.WHITE: "white", chess.BLACK: "black"}


def name(sq):
    return chess.FILE_NAMES[chess.square_file(sq)], str(chess.square_rank(sq) + 1)


def facts(fen):
    out = {k: set() for k in NEW}
    for r in CE.facts("chess", TEXT, VOC, fen, [], False):
        if r[0] in out:
            out[r[0]].add(tuple(r[1:]))
    return out


def key_squares(colour, p):
    f, r = chess.square_file(p), chess.square_rank(p)
    if f in (0, 7):
        return set()
    step = 1 if colour == chess.WHITE else -1
    early = r in ((1, 2, 3) if colour == chess.WHITE else (6, 5, 4))
    ranks = [r + 2 * step] + ([] if early else [r + step])
    return {chess.square(x, y) for x in (f - 1, f, f + 1) for y in ranks if 0 <= x < 8 and 0 <= y < 8}


def oracle(b):
    out = {k: set() for k in NEW}
    for colour in (chess.WHITE, chess.BLACK):
        s, own, other = SIDE[colour], b.king(colour), b.king(not colour)
        for p in b.pieces(chess.PAWN, colour):
            f, r = name(p)
            front = p + (8 if colour == chess.WHITE else -8)
            if 0 <= front < 64 and other == front:
                out["king_in_front"].add((s, f, r))
            ks = key_squares(colour, p)
            hanging = b.is_attacked_by(not colour, p) and not b.is_attacked_by(colour, p)
            if not ks or hanging:
                continue
            out["key_gap"].add((s, f, r, "d%d" % min(chess.square_distance(own, k) for k in ks)))
            out["key_guard"].add((s, f, r, "d%d" % min(chess.square_distance(other, k) for k in ks)))
    return out


def positions(n=150, seed=32):
    rng, out = random.Random(seed), []
    while len(out) < n:
        b = chess.Board(None)
        sqs = rng.sample(range(64), rng.choice([3, 3, 3, 4, 5, 6]))
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        for s in sqs[2:]:
            kind = rng.choice([chess.PAWN, chess.PAWN, chess.PAWN, chess.ROOK, chess.KNIGHT])
            if kind == chess.PAWN and chess.square_rank(s) in (0, 7):
                continue
            b.set_piece_at(s, chess.Piece(kind, rng.choice([chess.WHITE, chess.BLACK])))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if b.is_valid():
            out.append(b.fen())
    while len(out) < n + 40:                                             # (a king right in front of a pawn is rare by chance: build some)
        p, colour = rng.randrange(8, 56), rng.choice([chess.WHITE, chess.BLACK])
        front = p + (8 if colour == chess.WHITE else -8)
        own = rng.choice([s for s in range(64) if s not in (p, front) and chess.square_distance(s, front) > 1])
        b = chess.Board(None)
        b.set_piece_at(p, chess.Piece(chess.PAWN, colour))
        b.set_piece_at(front, chess.Piece(chess.KING, not colour))
        b.set_piece_at(own, chess.Piece(chess.KING, colour))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if b.is_valid():
            out.append(b.fen())
    b = chess.Board()
    for _ in range(40):
        if b.is_game_over():
            break
        b.push(rng.choice(list(b.legal_moves)))
        out.append(b.fen())
    return out


HAND = ["8/8/8/8/k2P4/8/3K4/8 w - - 0 1", "8/8/8/8/2Pk4/1K6/8/8 b - - 0 1", "4k3/8/8/8/8/8/P7/4K3 w - - 0 1", "8/3k4/8/3pK3/8/8/8/8 w - - 0 1", "8/4k3/4P3/4K3/8/8/8/8 b - - 0 1"]


def test_the_key_square_words_equal_the_oracle():
    seen = {k: 0 for k in NEW}
    for fen in HAND + positions():
        got, want = facts(fen), oracle(chess.Board(fen))
        for k in NEW:
            assert got[k] == want[k], (fen, k, sorted(got[k] ^ want[k]))
            seen[k] += len(want[k])
    assert all(v > 20 for v in seen.values()), seen                      # (every word was exercised)


def test_a_rook_pawn_and_a_hanging_pawn_have_no_key_distance():
    assert facts("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")["key_gap"] == set()
    assert facts("8/8/8/8/3k4/3P4/8/K7 w - - 0 1")["key_gap"] == set()          # (their king attacks the pawn, ours is far)
    assert facts("8/8/8/8/k2P4/8/3K4/8 w - - 0 1")["key_gap"] == {("white", "d", "4", "d4")}


# ----------------------------------------------------------------------------------------- the fixes and the hand step's plumbing (plan 78.32, after the first run)

from pathlib import Path

from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import induce as IND
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import targeted as TG

ROOT = Path(__file__).resolve().parents[1]
PROPOSED = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())


def test_a_snapshot_built_on_the_head_carries_the_heads_valuation_record(tmp_path):
    """The targeted miner's trial and published snapshots, and the lemma sleep's, once dropped the record the wake plays by: the next wake then read positions by another valuation."""
    root = IN.prepare_sandbox("chess", tmp_path / "store")
    want = GS.judge_of("chess", "material (mined)")
    lem = {"name": "cell-response-1", "body": [AB.lit("control", ["us"])], "response": {"removed": [], "added": [AB.lit("cell", ["?a", "?b", "us", "?c"])]}, "stats": {}, "meta": {}}
    with TG.sandbox(root):
        assert GS.head_judge("chess") == "store"                                     # (no head yet: the store's record in use)
        GS.publish("chess", vocabulary="relational", judge=want)
        assert GS.head_judge("chess") == want
        d = tmp_path / "t"
        d.mkdir()
        assert json.loads(Path(TG.trial_snapshot("chess", lem, d, "relational")).read_text())["judge"] == want
    src = (ROOT / "catspace/incubator/analyst/discovery/sleep.py").read_text()
    assert 'judge=GS.judge_of(game, s.judge) if s.judge else "store"' in src          # (the lemma sleep publishes with the record it learned by)
    tg = (ROOT / "catspace/incubator/analyst/discovery/targeted.py").read_text()
    assert "GS.publish(game, teacher_used=True, vocabulary=vocabulary, judge=GS.head_judge(game))" in tg


def test_the_audit_does_not_count_a_firing_that_names_no_move_as_refuted():
    src = (ROOT / "catspace/incubator/analyst/discovery/apply_audit.py").read_text()
    assert 'if opt and not item["trivial"] and all_named and not prop:' in src and "silent" in src


def test_the_proposed_rules_are_well_formed_and_say_they_are_by_hand():
    assert PROPOSED["vocabulary"] == "keys" and len(PROPOSED["rules"]) == 4
    words = set(GS.vocabulary("chess", "keys")["derived"])
    for r in PROPOSED["rules"]:
        GS.check_lemma(r)
        assert r["meta"]["origin"] == "hand" and r["stats"] == {"status": "HAND"}
        assert {l["relation"] for l in r["body"]} <= words
        assert "right" not in json.dumps(r) and "wrong" not in json.dumps(r)         # (no count of a drill is written into a rule)


def test_drill_admit_keeps_only_rules_with_no_wrong_firing(monkeypatch):
    class C:
        def cards(self, spec, vocab):
            self.built = True
    table = {"good": {"dev": (5, 3, 0, 2), "held": (4, 2, 0, 2)}, "bad": {"dev": (5, 3, 0, 2), "held": (4, 1, 1, 2)}}
    monkeypatch.setattr(IND.CP, "drill", lambda lem, corpus, part="dev", max_bindings=256: dict(zip(("binds", "right", "wrong", "trivial"), table[lem["name"]][part])))
    said = []
    kept, dropped = IND.drill_admit([{"name": "good"}, {"name": "bad"}], C(), {}, "{}", say=said.append)
    assert [l["name"] for l in kept] == ["good"] and [l["name"] for l in dropped] == ["bad"]
    assert said[-1] == "OBS drill admit total: rules 2, kept 1, dropped 1" and "DROPPED" in said[1] and "KEPT" in said[0]


def test_run_size_takes_seed_rules_and_the_drilled_sleep_and_refuses_seed_rules_without_a_corpus(tmp_path):
    import inspect
    sig = inspect.signature(IN.run_size).parameters
    assert sig["seed_rules"].default is None and sig["drill_sleep"].default is False
    src = (ROOT / "catspace/incubator/analyst/discovery/initialize.py").read_text()
    assert "seed_rules needs the exact corpus" in src and "--seed-rules" in src and "--drill-sleep" in src


def test_the_new_files_are_in_the_gate_and_in_repo_structure():
    assert "test_plan78_32_keys" in (ROOT / "scripts/gate.py").read_text()
    rs = (ROOT / "repo_structure.md").read_text()
    for f in ("chess_keys.gdl", "vocabulary.keys.json", "chess_keys.proposed.json"):
        assert f in rs
