"""PLAN 78.13 / 78.14 THE FACT LAYER, THE STORE'S SNAPSHOTS, THE WAKE AND THE SLEEP (docs/ANALYST_PLAN.md 78.13-78.15). The Rust side (the fact layer certified by co-play, the
matcher, the lemma operator, the card evaluation, the second game) is tested in rust/engine/*/tests and the crates' unit tests; here the Python entry and the loop:
the content address agrees with hashlib, a snapshot is immutable and the engine reads exactly the one it names, a motif planted in real positions is discovered and
verified on positions it never saw while a false lemma is refuted, surprises are read from the log, a small wake and sleep run end to end, and the teacher stays out
of the decision path."""
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.incubator.analyst.discovery import wake as WK

ROOT = Path(__file__).resolve().parents[1]
FILES = "abcdefgh"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The game store pointed at a temporary directory (the vocabulary copied in): a test never touches the real lemmas or snapshots."""
    monkeypatch.setattr(GS, "LEARNED_ROOT", tmp_path / "learned")
    monkeypatch.setattr(GS, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    shutil.copytree(ROOT / "rules" / "learned" / "chess", tmp_path / "learned" / "chess")
    for p in (tmp_path / "learned" / "chess" / "lemmas").glob("*.json"):
        p.unlink()
    return tmp_path


def fen_of(pieces, white_to_move=True):
    rows = [["."] * 8 for _ in range(8)]
    for f, r, ch in pieces:
        rows[8 - r][FILES.index(f)] = ch
    s = re.sub(r"\.+", lambda m: str(len(m.group())), "/".join("".join(row) for row in rows))
    return s + (" w" if white_to_move else " b") + " - - 0 1"


def motif(ranks):
    """A pawn of ours attacked by a pawn of theirs that it can take, kings far away: taking is the best move."""
    return [fen_of([("a", 1, "K"), ("a", 8, "k"), (FILES[f], r, "P"), (FILES[f + side], r + 1, "p")]) for r in ranks for f in range(2, 7) for side in (-1, 1)]


def decoys():
    """A rook that can take a pawn defended by a pawn (a bad capture), nothing of ours attacked."""
    out = []
    for d in range(2, 6):
        for r in (4, 5):
            out.append(fen_of([("h", 1, "K"), ("h", 8, "k"), ("a", r, "R"), (FILES[d + 1], r, "p"), (FILES[d], r + 1, "p")]))
            out.append(fen_of([("h", 1, "K"), ("h", 8, "k"), ("a", r, "R"), (FILES[d], r, "p"), (FILES[d - 1], r + 1, "p")]))
    return out


def cards_of(fens, depth=3):
    items = [("chess", f, [], f"p{i}", f"g{i}") for i, f in enumerate(fens)]
    return CE.build_cards(items, GS.spec_texts("chess"), json.dumps(GS.vocabulary("chess")), depth=depth)


# ------------------------------------------------------------------------------------------------------------------------------------ the store

def test_78_14_the_content_address_agrees_with_hashlib():
    for n in (0, 1, 3, 55, 56, 57, 63, 64, 65, 119, 120, 1000, 100_000):
        data = bytes((i * 31 + n) % 251 for i in range(n))
        assert CE.content_id(data) == hashlib.sha256(data).hexdigest()[:16], n


def lemma_named(name, rel="cell"):
    return {"name": name, "body": [{"relation": "cell", "args": ["?x", "?y", "us", "?k"], "neg": False}, {"relation": "attack", "args": ["them", "?a", "?b", "?x", "?y"], "neg": False}],
            "response": {"removed": [{"relation": rel, "args": ["?x", "?y", "us", "?k"], "neg": False}], "added": [{"relation": rel, "args": ["?u", "?w", "us", "?k"], "neg": False}]},
            "stats": {"status": "SAMPLED"}, "meta": {"origin": "test"}}


def test_78_14_a_snapshot_is_immutable_content_and_the_chain_grows_only_when_the_content_does(store):
    GS.write_lemma("chess", lemma_named("step-away"))
    a = GS.publish("chess")
    assert GS.publish("chess") == a                                      # the same content: the same snapshot, nothing new
    assert GS.info("chess")["parent"] is None and GS.head("chess") == a
    GS.write_lemma("chess", lemma_named("step-away-too"))
    b = GS.publish("chess")
    assert b != a and GS.info("chess", b)["parent"] == a and GS.head("chess") == b
    assert CE.snapshot_info(str(GS.path_of("chess", a)))["lemmas"] == ["step-away"]            # the older one is still what it was
    print(f"OBS 78.14 snapshots: {a} -> {b}")


def test_78_14_the_engine_refuses_a_snapshot_that_does_not_hash_to_its_name(store, tmp_path):
    GS.write_lemma("chess", lemma_named("step-away"))
    sid = GS.publish("chess")
    p = GS.path_of("chess", sid)
    bad = p.with_name("0123456789abcdef.json")
    bad.write_bytes(p.read_bytes())
    with pytest.raises(ValueError, match="named by its content"):
        CE.snapshot_info(str(bad))
    edited = p.read_bytes().replace(b"step-away", b"step-aside")
    p.write_bytes(edited)                                                 # edited in place: the bytes no longer hash to the name
    with pytest.raises(ValueError, match="named by its content"):
        CE.snapshot_info(str(p))
    with pytest.raises(ValueError):
        CE.solve_batch([("chess", None, ["e2e4"])], 0.05, store=str(p), specs=GS.spec_texts("chess"))


def test_78_14_a_malformed_lemma_never_reaches_the_store(store):
    bad = lemma_named("bad")
    bad["body"].append({"relation": "cell", "args": ["?never"], "neg": True})
    with pytest.raises(ValueError, match="binds"):
        GS.write_lemma("chess", bad)
    unsaid = lemma_named("unsaid")
    unsaid["response"] = {"removed": [], "added": []}
    with pytest.raises(ValueError):
        GS.write_lemma("chess", unsaid)
    wrong_rel = lemma_named("elsewhere", rel="control")
    GS.write_lemma("chess", wrong_rel)                                    # (well formed; the snapshot refuses a response outside the delta relations)
    with pytest.raises(ValueError, match="delta"):
        GS.publish("chess")


def test_78_14_the_engine_reads_the_snapshot_it_logs_and_the_ablation_removes_a_lemma(store, tmp_path):
    take = {"name": "take-the-attacker", "body": [{"relation": "cell", "args": ["?x", "?y", "us", "?k"], "neg": False}, {"relation": "attack", "args": ["them", "?a", "?b", "?x", "?y"], "neg": False},
                                                   {"relation": "cell", "args": ["?a", "?b", "them", "?j"], "neg": False}],
            "response": {"removed": [{"relation": "cell", "args": ["?a", "?b", "them", "?j"], "neg": False}], "added": [{"relation": "cell", "args": ["?a", "?b", "us", "?m"], "neg": False}]},
            "stats": {}, "meta": {}}
    GS.write_lemma("chess", take)
    sid = GS.publish("chess")
    specs, path = GS.spec_texts("chess"), str(GS.path_of("chess", sid))
    log = tmp_path / "t.jsonl"
    fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    CE.solve_batch([("chess", fen, [])], 0.3, store=path, specs=specs, trace_path=str(log))
    CE.solve_batch([("chess", fen, [])], 0.3, store=path, specs=specs, trace_path=str(log), ablate=["take-the-attacker"])
    plain = [json.loads(x) for x in log.read_text().splitlines()]
    assert all(r["snapshot"] == sid for r in plain)
    with_lemma, ablated = plain
    assert [s["operator"] for s in with_lemma["steps"]][:1] == ["lemma"] and with_lemma["steps"][0]["candidates"] == 1
    assert with_lemma["facts"] and with_lemma["facts"][0]["relation"] == "fired"
    ledger = next(s for s in with_lemma["steps"] if s["operator"] == "ledger")
    assert "prefer 1" in ledger["goal"]
    assert ablated["steps"][0]["candidates"] == 0 and "prefer" not in next(s for s in ablated["steps"] if s["operator"] == "ledger")["goal"]
    parts = {p["name"]: p for p in with_lemma["steps"][0]["parts"]}
    assert "facts" in parts and "match take-the-attacker" in parts and parts["match take-the-attacker"]["nodes"] > 0             # the cost in parts, measured
    assert with_lemma["move"] == ablated["move"] == "e4d5"                  # a prior: the same decision whether or not the lemma was read


# ------------------------------------------------------------------------------------------------------------------------------------ abstraction and verification

def climb(fens, gates, seed_index=0):
    cs = cards_of(fens)
    dev = [i for i in range(len(cs)) if i % 2 == 0]
    held = [i for i in range(len(cs)) if i % 2 == 1]
    card = json.loads(cs.card(seed_index))
    best = max(m["value"] for m in card["moves"])
    move = next(m for m in card["moves"] if m["value"] == best and m["lost"] == 0)
    seed = AB.seed_from(card, move, gates)
    return cs, dev, held, AB.generalize(seed, cs, dev, gates)


def test_78_14_a_planted_motif_is_generalized_and_holds_on_positions_it_never_saw():
    g = AB.Gates(min_fires=3, min_groups=3)
    cs, dev, held, c = climb(motif((3, 4)) + decoys(), g)
    assert c is not None and c.steps
    seed_constants = len(AB.constants(AB.seed_from(json.loads(cs.card(0)), next(m for m in json.loads(cs.card(0))["moves"] if m["value"] == max(x["value"] for x in json.loads(cs.card(0))["moves"])), g)))
    assert len(AB.constants(c.lemma)) < seed_constants, "constants were lifted to variables"
    body = {(l["relation"]) for l in c.lemma["body"]}
    assert "attack" in body, f"the attack is what excludes the decoys (a rook that can take but is not attacked): {c.lemma['body']}"
    h = AB.evaluate(cs, c.lemma, held, g)
    print(f"OBS 78.14 planted motif: {len(c.steps)} steps, dev fires {c.ev['fires']} precision {c.ev['precision']}; held-out fires {h['fires']} in {h['groups']} games, precision {h['precision']}, excess {h['excess']:.2f}")
    assert h["fires"] >= 5 and h["precision"] == 1.0 and g.admissible(h) and g.supported(h)
    assert not h["truncated"]


def test_78_14_a_false_lemma_is_refuted_and_a_vacuous_one_earns_no_excess():
    g = AB.Gates(min_fires=3, min_groups=3)
    cs = cards_of(motif((3, 4)) + decoys())
    # "when a pawn of mine is attacked, my king moves": never the best move in the motif (taking the attacker is)
    false = {"name": "king-steps", "body": [{"relation": "cell", "args": ["?x", "?y", "us", "pawn"], "neg": False}, {"relation": "attack", "args": ["them", "?a", "?b", "?x", "?y"], "neg": False}],
             "response": {"removed": [{"relation": "cell", "args": ["?kx", "?ky", "us", "king"], "neg": False}], "added": [{"relation": "cell", "args": ["?ku", "?kw", "us", "king"], "neg": False}]},
             "stats": {}, "meta": {}}
    ev = AB.evaluate(cs, false, None, g)
    assert ev["fires"] > 0 and ev["refuted"] > 0 and not g.admissible(ev)
    # and the true neighbour of it -- "a move of that pawn is among the best" (the capture is one) -- holds
    true = lemma_named("pawn-moves")
    true["body"][0]["args"][3] = "pawn"
    true["response"]["removed"][0]["args"][3] = "pawn"
    true["response"]["added"][0]["args"][3] = "pawn"
    ev_true = AB.evaluate(cs, true, None, g)
    assert ev_true["fires"] > 0 and ev_true["refuted"] == 0
    # "if I have any piece, a piece of mine moving is among the best": true often, but chance already makes it so -- and it matches nearly every move
    vacuous = {"name": "vac", "body": [{"relation": "control", "args": ["us"], "neg": False}],
               "response": {"removed": [{"relation": "cell", "args": ["?x", "?y", "us", "?k"], "neg": False}], "added": [{"relation": "cell", "args": ["?u", "?w", "us", "?k"], "neg": False}]}, "stats": {}, "meta": {}}
    ev2 = AB.evaluate(cs, vacuous, None, g)
    assert ev2["selectivity"] > g.max_selectivity or ev2["excess"] < g.min_excess
    assert not g.admissible(ev2)


def test_78_14_support_is_a_lower_confidence_bound_not_a_point_estimate():
    """Three clean firings say the precision is at least 0.53 at 95 % (one-sided Wilson), not 1.0; the support gate reads the bound. The eight lemmas the 96-game sleep
    passed on 3-11 clean development firings (and lost on held-out games) are the case: the default gate stops six of them."""
    g = AB.Gates()
    assert AB.lower_bound(0, 0) == 0.0 and AB.lower_bound(3, 3) < AB.lower_bound(10, 10) < AB.lower_bound(29, 29) < 1.0
    assert abs(AB._z(0.95) - 1.6449) < 1e-3 and abs(AB.lower_bound(10, 10) - 0.787) < 0.01
    assert AB.lower_bound(9, 14) < AB.lower_bound(14, 14)
    clean = lambda n: {"fires": n, "holds": n, "refuted": 0, "groups": 2}  # noqa: E731
    assert not g.supported(clean(3)) and not g.supported(clean(6)) and g.supported(clean(7)) and g.supported(clean(11))
    assert sum(not g.supported(clean(n)) for n in (3, 4, 4, 11, 7, 4, 6, 6)) == 6
    assert not g.supported({"fires": 20, "holds": 20, "refuted": 0, "groups": 1}), "independent games are still required"
    assert SL.spread([("a", 1), ("a", 2), ("b", 1), ("c", 1), ("a", 3)], lambda t: t[0])[:3] == [("a", 1), ("b", 1), ("c", 1)]


def test_78_14_cut_evaluations_are_never_admitted():
    g = AB.Gates()
    ev = {"fires": 5, "precision": 1.0, "excess": 0.9, "selectivity": 0.1, "truncated": True}
    assert not g.admissible(ev)


# ------------------------------------------------------------------------------------------------------------------------------------ the loop

def row(label, move, steps=None, proved=False):
    return {"kind": "position", "label": label, "move": move, "steps": steps or [], "error": None, "proved": proved}


def ledger_step(role, losses, depth=4, value=0):
    return {"operator": "ledger", "status": "Read", "depth": depth, "value": value, "events": [[role, f"lose {k}"] for k in losses], "line": []}


def test_78_14_a_surprise_is_a_loss_the_second_side_sees_that_the_first_did_not_foresee(tmp_path):
    log = tmp_path / "w.jsonl"
    rows = [
        row("r:0:4:0", "a2a3", [ledger_step(0, [], depth=4)]),                       # white foresaw no loss
        row("r:0:5:1", "b7b5", [ledger_step(0, ["p"], depth=4)]),                    # black sees white lose a pawn: a surprise
        row("r:0:6:0", "a3a4", [ledger_step(0, ["p"], depth=4)]),                    # white foresaw the loss of a pawn
        row("r:0:7:1", "b5a4", [ledger_step(0, ["p"], depth=4)]),                    # black sees the same: no surprise
        row("r:0:8:0", "h2h3", [{"operator": "forcing", "status": "Proved", "depth": 2, "events": []}], proved=True),   # white proves a forced win after black's move: a surprise for black
        row("r:0:9:1", "g7g6", [ledger_step(0, ["q"], depth=4)]),
    ]
    game = {"kind": "game", "run": "r", "game": 0, "variant": "chess", "moves": ["x"] * 4 + ["a2a3", "b7b5", "a3a4", "b5a4", "h2h3", "g7g6"], "outcome": None, "unfinished": True}
    log.write_text("\n".join(json.dumps(r) for r in rows + [game]) + "\n")
    d, g = SL.read_log(str(log))
    assert len(d) == 6 and (("r", 0) in g)
    sur = SL.surprises(d, g)
    assert [(s["ply"], s["predicted"], s["seen"]) for s in sur] == [(4, 0, 1), (7, 0, 1)]
    assert sur[0]["need"] == 5 and sur[0]["moves_before"] == ["x"] * 4 and sur[0]["played"] == "a2a3"
    assert sur[1]["need"] == 4, "a forced line of two of white's moves is read in four plies"
    assert [s["ply"] for s in SL.surprises(d, g, min_units=2)] == [7], "a higher threshold keeps only the proved win: it outweighs any count of units"


def test_78_14_wake_and_sleep_run_end_to_end_and_the_funnel_only_narrows(store, tmp_path):
    log = str(tmp_path / "wake.jsonl")
    games = WK.play_self("chess", 8, seconds=0.01, random_plies=4, max_plies=60, seed=3, log=log, run="e2e", threads=2)
    assert len(games) == 8 and all(len(g["moves"]) > 0 for g in games)
    rows = [json.loads(x) for x in open(log)]
    assert {r["kind"] for r in rows} == {"position", "game"} and all(r["label"].count(":") == 3 for r in rows if r["kind"] == "position")
    rep = SL.sleep("chess", log, settings=SL.Settings(depth=4, seed=3))
    f = rep["funnel"]
    order = ["decisions", "surprises", "replayable", "confirmed", "with_saving", "seeds", "climbed", "supported_dev", "verified_held"]
    vals = [f[k] for k in order]
    print("OBS 78.14 end to end:", ", ".join(f"{k} {f[k]}" for k in order))
    assert f["games"] == 8 and f["cards"] > 0 and f["cards_dev"] + f["cards_held"] == f["cards"]
    assert vals[1] >= vals[2] >= vals[3] >= vals[4] and f["seeds"] >= f["climbed"] >= f["supported_dev"] >= f["verified_held"]
    # whatever it admitted passes the engine's own reading, the held-out gates, and nothing was written (write=False)
    for l in rep["lemmas"]:
        GS.check_lemma(l)
        h = l["stats"]["held"]
        assert h["precision"] >= 1.0 and h["fires"] >= 3 and h["excess"] >= 0.2 and l["stats"]["status"] == "SAMPLED"
    assert not list(GS.lemma_dir("chess").glob("*.json")) and GS.head("chess") is None


def test_78_14_a_written_lemma_publishes_a_snapshot_the_engine_then_plays_with(store, tmp_path):
    take = {"name": "take-the-attacker", "body": [{"relation": "cell", "args": ["?x", "?y", "us", "?k"], "neg": False}, {"relation": "attack", "args": ["them", "?a", "?b", "?x", "?y"], "neg": False},
                                                   {"relation": "cell", "args": ["?a", "?b", "them", "?j"], "neg": False}],
            "response": {"removed": [{"relation": "cell", "args": ["?a", "?b", "them", "?j"], "neg": False}], "added": [{"relation": "cell", "args": ["?a", "?b", "us", "?m"], "neg": False}]},
            "stats": {}, "meta": {"teacher": False}}
    GS.write_lemma("chess", take)
    sid = GS.publish("chess")
    log = str(tmp_path / "w2.jsonl")
    games = WK.play_self("chess", 2, seconds=0.02, random_plies=2, max_plies=24, seed=4, store=str(GS.path_of("chess", sid)), specs=GS.spec_texts("chess"), log=log, run="w2", threads=2)
    assert all(g["snapshot"] == sid for g in games)
    pos = [json.loads(x) for x in open(log) if '"position"' in x]
    assert pos and all(r["snapshot"] == sid for r in pos)
    assert any(s["operator"] == "lemma" for r in pos for s in r["steps"])


# ------------------------------------------------------------------------------------------------------------------------------------ the teacher

def test_78_14_a_teacher_is_flagged_training_only_in_the_type_and_out_of_the_decision_path():
    engine = ROOT / "rust" / "engine"
    decision_path = ["operator", "tactics", "planning", "memory", "decision"]
    for crate in decision_path:
        manifest = (engine / crate / "Cargo.toml").read_text()
        assert "cs_learn" not in manifest, f"{crate} depends on the learning crate, where the teacher lives"
        for p in sorted((engine / crate / "src").glob("**/*.rs")):
            code = "\n".join(ln.split("//")[0] for ln in p.read_text().splitlines())
            assert "outside::" not in code and "Teacher" not in code, f"{p.relative_to(ROOT)} names a teacher"
    watch = (engine / "learn" / "src" / "watch.rs").read_text()
    assert "fn training_only(&self) -> bool { true }" in watch and "assert!(eng.training_only()" in watch


def test_78_14_whatever_a_teacher_touched_is_recorded_on_the_lemma_and_the_snapshot(store):
    taught = lemma_named("taught")
    taught["meta"] = {"origin": "sleep", "teacher": True}
    GS.write_lemma("chess", taught)
    sid = GS.publish("chess")
    assert GS.info("chess", sid)["teacher_used"] is True
    assert json.loads(GS.path_of("chess", sid).read_text())["teacher_used"] is True


def test_78_14_credit_by_removal_reports_the_positions_a_lemma_fired_on(store):
    take = {"name": "take-the-attacker", "body": [{"relation": "cell", "args": ["?x", "?y", "us", "?k"], "neg": False}, {"relation": "attack", "args": ["them", "?a", "?b", "?x", "?y"], "neg": False},
                                                   {"relation": "cell", "args": ["?a", "?b", "them", "?j"], "neg": False}],
            "response": {"removed": [{"relation": "cell", "args": ["?a", "?b", "them", "?j"], "neg": False}], "added": [{"relation": "cell", "args": ["?a", "?b", "us", "?m"], "neg": False}]},
            "stats": {}, "meta": {}}
    GS.write_lemma("chess", take)
    path = str(GS.path_of("chess", GS.publish("chess")))
    positions = [("chess", f, []) for f in motif((3, 4))[:6]] + [("chess", None, ["e2e4", "e7e5"]), ("chess", "4k3/8/8/8/8/8/8/4K2R w - - 0 1", [])]
    out = SL.credit("chess", path, positions, seconds=0.1, threads=2, say=lambda *_: None)
    r = out["take-the-attacker"]
    assert r["fired"] == 6 and r["positions_compared"] == 6, r             # the motif positions, not the opening or the bare rook
    assert r["moved"] == 0                                                  # the capture is the best move: the same decision with or without the lemma
    assert r["depth_with"] is not None and r["depth_without"] is not None
    print(f"OBS 78.14 credit by removal on {r['fired']} firing positions: depth {r['depth_with']:.2f} with, {r['depth_without']:.2f} without (se of the difference {r['depth_difference_se']})")
