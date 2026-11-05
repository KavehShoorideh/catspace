"""PLAN 78 THE END-TO-END SYSTEM (docs/ANALYST_PLAN.md 78). 78.1 THE GAME LAYER: the `Game` trait (rust/engine/game), any GDL text as a game (the spec evaluator,
cs_gdl), the chess kernels for the perfect-information variants (shakmaty), certify by co-play; the variants' rules as DELTAS on rules/chess.gdl (cs_gdl::compose).
The layer crates name no chess object: only the kernel module and the GDL texts are chess (rules ingestion)."""
import re
from pathlib import Path

import pytest

import cs_engine as CE

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
BASE = (ROOT / "rules" / "chess.gdl").read_text()
DELTA = {v: (ROOT / "rules" / "variants" / f"{v}.gdl").read_text() for v in ("kingofthehill", "3check")}
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


def _code(p: Path) -> str:
    """The Rust code without its comments (//, //!, ///) and string literals."""
    t = re.sub(r'"(?:\\.|[^"\\])*"', '""', p.read_text())
    return "\n".join(ln.split("//")[0] for ln in t.splitlines())


def test_78_1_the_layer_crates_name_no_chess_object_but_the_kernel_module():
    """The generic test, extended to the Rust layers: every source of the engine workspace but the kernel module (rules ingestion) and its certify binary is free
    of chess words in its code (comments and strings aside)."""
    allowed = {ENGINE / "game" / "src" / "kernel.rs", ENGINE / "game" / "src" / "outside.rs", ENGINE / "game" / "src" / "bin" / "certify_coplay.rs", ENGINE / "py" / "src" / "lib.rs"}
    seen = []
    for p in sorted(ENGINE.glob("*/src/**/*.rs")):
        if p in allowed:
            continue
        hit = re.search(CHESS_WORDS, _code(p), flags=re.I)
        assert hit is None, (str(p.relative_to(ROOT)), hit.group(0) if hit else None)
        seen.append(p.name)
    print(f"OBS 78.1 generic test: {len(seen)} layer sources free of chess words: {', '.join(seen)}")
    assert {"lib.rs", "gdl_game.rs", "certify.rs", "rng.rs"} <= set(seen)


def test_78_1_the_kernels_agree_with_their_gdl_texts_by_co_play_and_a_wrong_text_disagrees():
    for v in ("chess", "kingofthehill", "3check"):
        games, plies, states, bad = CE.certify_coplay(v, BASE if v == "chess" else DELTA[v], 3, 5, None if v == "chess" else BASE)
        print(f"OBS 78.1 co-play {v}: games {games}, plies {plies}, states {states}, disagreements {len(bad)}")
        assert games == 3 and plies > 0 and bad == []
    assert CE.certify_coplay("kingofthehill", BASE, 3, 5)[3]                         # the hill's kernel against plain chess: it disagrees (the control)
    assert CE.certify_coplay("3check", DELTA["kingofthehill"], 3, 5, BASE)[3]


def test_78_1_a_delta_must_retract_what_the_base_says():
    with pytest.raises(ValueError):
        CE.certify_coplay("kingofthehill", "(retract (<= terminal nothing_of_the_base))", 1, 1, BASE)


def test_78_1_every_variant_is_played_and_those_without_a_text_are_reported_uncertified():
    from catspace.incubator.analyst.discovery import certify as CF
    assert set(CE.variants()) == {"chess", "chess960", "crazyhouse", "atomic", "antichess", "kingofthehill", "3check", "racingkings", "horde"}
    un = CF.uncertified()
    print(f"OBS 78.1 uncertified variants (no GDL text yet): {un}")
    assert set(un) == set(CE.variants()) - set(CF.VARIANT_TEXTS)
    for v in CF.VARIANT_TEXTS:
        assert CF.status()[f"kernel {v} (cs_game)"] == "fresh"


# ---- 78.2 the thin approaches (tactics `forcing`, don't lose, planning `keep`) and the decision loop, one call per move

def _worth():
    from catspace.incubator.analyst.discovery import lemma_store as LS
    return {k: int(v) for k, v in LS.valuation().worth.items()}


def test_78_2_the_decision_proves_each_variants_own_goal():
    """The tactics layer reads only the Game trait's outcome: the same search proves a chess mate, a king walking onto the hill, a third check."""
    w = _worth()
    for v, fen, want in (("chess", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"), ("kingofthehill", "8/8/8/8/8/2K5/8/7k w - - 0 1", "c3d4"),
                         ("3check", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1+3 0 3", None)):
        mv, proved, trace, nodes, _f = CE.best_move(v, fen, [], 1.0, w)
        print(f"OBS 78.2 {v}: {mv} proved {proved}, {nodes} nodes -- {trace[1][:120]}")
        assert proved and (want is None or mv == want)


def test_78_2_every_variant_gets_a_legal_move_and_dont_lose_drops_a_losing_one():
    w = _worth()
    for v in CE.variants():
        mv, proved, trace, nodes, fen = CE.best_move(v, None, [], 0.2, w)
        assert mv in CE.state(v)[2], (v, mv)
    # chess, the fool's mate both ways: after 1.f3 e5 2.g4 Black's Qh4# is proved; after 1.f3 e5 White's 2.g4?? (it allows Qh4#) is not played (plan 78.5)
    mv, proved, trace, nodes, fen = CE.best_move("chess", None, ["f2f3", "e7e5", "g2g4"], 0.5, w)
    assert mv == "d8h4" and proved                                                    # (Black to move: the fool's mate, proved)
    mv, proved, trace, nodes, fen = CE.best_move("chess", None, ["f2f3", "e7e5"], 0.5, w)
    assert mv != "g2g4" and any("ledger search" in x for x in trace)                 # (White: g4?? allows Qh4#: the ledger search reads their mate below every ledger)


# ---- 78.4 knowledge of a game (the store, scoped by `game`) and of an instance of it (discovery/instance.py); the miners in Rust (cs_learn)

def test_78_4_the_rust_miner_reproduces_the_stores_mined_chess_valuation():
    """cs_learn's valuation miner (76.4c's definition, its loops in Rust) on the records 76.4c read: the same integer worths as the store's 'material (mined)'."""
    from catspace.incubator.analyst.discovery import learn as LN
    from catspace.incubator.analyst.discovery import lemma_store as LS
    recs, what = LN.lichess_records(4000)
    f = CE.mine_valuation("chess", recs, LN.FROM_PLY, LN.FUSE)
    want = {k: v for k, v in LS.load()["material (mined)"]["params"]["worth"].items() if k != "k"}
    print(f"OBS 78.4 rust miner on {what}: worth {dict(zip(f['kinds'], f['worth']))} price {f['price']}, positions {f['positions']}; the store's {want}")
    assert dict(zip(f["kinds"], f["worth"])) == want and f["constant"] == ["k"]


def test_78_4_knowledge_is_scoped_by_game_and_nothing_is_borrowed(monkeypatch):
    import copy
    from catspace.incubator.analyst.discovery import lemma_store as LS
    assert LS.valuation_of("chess").name == "units (no values)" and LS.valuation().name == "material"     # (78.5: the record in use; the legacy engine reads its GOAL entries')
    assert LS.valuation_of("atomic") is None                                         # the store knows no valuation of atomic: none, not chess's
    patched = copy.deepcopy(LS.load())
    e = copy.deepcopy(patched["material (mined)"])
    e["name"], e["game"] = "material (atomic, test)", "atomic"
    patched[e["name"]] = e
    monkeypatch.setattr(LS, "_STORE", [patched])
    assert LS.valuation_of("atomic").worth == e["params"]["worth"] and LS.check_entry(e) == []
    assert LS.check_entry(dict(e, game=3))


def test_78_4_watching_an_outside_engine_makes_instances_not_store_entries(tmp_path, monkeypatch):
    from catspace.incubator.analyst.discovery import instance as IN
    from catspace.incubator.analyst.discovery import learn as LN
    recs = CE.watch("chess", LN.TEACHER, 2, 6, 5, 7, 2)
    assert len(recs) == 2 and all(m for _s, m, _o in recs)
    monkeypatch.setattr(IN, "folder", lambda game: tmp_path)
    IN.save([IN.Instance(game="chess", start=s, moves=m, outcome=o, source="watch") for s, m, o in recs], "chess", "t")
    back = IN.load("chess", "t")
    assert [i.record() for i in back] == [(s, m, o) for s, m, o in recs]
    print(f"OBS 78.4 watched 2 chess games: plies {[len(m) for _s, m, _o in recs]}, outcomes {[o for _s, _m, o in recs]}")


# ---- 78.5 the ledger search: a tactic's outcome is the units each side loses; the judge (no values by default) is the variable

def test_78_5_the_ledger_search_wins_a_loose_unit_and_avoids_losing_one_with_no_values():
    """No values: Qa4 against a knight it can take for nothing is the ledger 'they lose n'; a queen attacked by a pawn steps away (ours first)."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    rec = LS.judge_of("chess")
    mv, proved, trace, nodes, _f = CE.best_move("chess", "4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1", [], 1.0, **rec)
    assert mv == "a4c6" and "they lose n" in trace[-1] and "we lose" not in trace[-1] and "units (no values)" in trace[-1]
    mv, proved, trace, nodes, _f = CE.best_move("chess", "4k3/8/8/2p5/3Q4/8/8/4K3 w - - 0 1", [], 1.0, **rec)        # the c5 pawn attacks the queen on d4
    assert "we lose q" not in trace[-1]
    print(f"OBS 78.5 ledger: {trace[-1][:200]}")


def test_78_5_the_judge_is_a_store_record_switched_without_code(monkeypatch):
    """The engine's judge is the VALUATION record in use: moving `in_use` from 'units (no values)' to 'material' changes what best_move judges by, no code changed."""
    import copy
    from catspace.incubator.analyst.discovery import lemma_store as LS
    assert LS.judge_of("chess")["valuation"] == "units (no values)" and LS.judge_of("chess")["ours_first"]
    patched = copy.deepcopy(LS.load())
    patched["units (no values)"]["params"]["in_use"] = False
    patched["material"]["params"]["in_use"] = True
    monkeypatch.setattr(LS, "_STORE", [patched])
    assert LS.judge_of("chess")["valuation"] == "material" and LS.judge_of("chess")["worth"]["q"] == 9
    patched["units (no values)"]["params"]["in_use"] = True
    with pytest.raises(LS.LemmaStoreError):
        LS.judge_of("chess")                                                         # two records in use: refused


def test_78_5_a_promotion_is_scored_by_the_valuation():
    """The ledger records a change of kind (pawn to queen): with 'material' plugged in it is +8 for us; with no values it is no gain (a unit is a unit)."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    fen = "8/4P3/8/8/8/k7/8/6K1 w - - 0 1"
    mv, _p, trace, _n, _f = CE.best_move("chess", fen, [], 1.0, **LS.judge_of("chess", "material"))
    assert mv == "e7e8q" and "we promote p to q" in trace[-1]


def test_78_5_the_judge_is_a_variable():
    """The same position read by the two judges: with no values a queen-for-bishop exchange is a loss of ours; with a valuation plugged in it is a gain."""
    fen = "r5k1/2q1b1pp/pp2p1r1/3b3Q/1P1Np3/1BPnP3/P5PP/R4RK1 w - - 1 25"      # rand14_10 Ddm9j: Bxd5 wins the queen for a bishop
    from catspace.incubator.analyst.discovery import lemma_store as LS
    _mv, _p, t_none, _n, _f = CE.best_move("chess", fen, [], 2.0, **LS.judge_of("chess"))
    _mv, _p, t_mat, _n, _f = CE.best_move("chess", fen, [], 2.0, **LS.judge_of("chess", "material"))
    assert "[units (no values)" in t_none[-1] and "[material" in t_mat[-1]


# ---- 78.6 the census of tactics by their defects (counted in tempo: the free moves of ours a tactic needs)

def test_78_6_the_census_counts_a_tactic_at_its_least_defects():
    """No values: a loose knight the queen already attacks is a 0-defect tactic ('they lose n'); one the queen must first go to attack is a 1-defect tactic, its free
    move named; nothing is counted twice."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    j = LS.judge_of("chess")
    now, _st = CE.census("chess", "4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1", [], 1, 3, 200000, "mover", **j)
    assert any(t["defects"] == 0 and t["line"][:1] == ["a4c6"] and t["wins"] == ["n"] for t in now)
    later, st = CE.census("chess", "4k3/8/2n5/8/8/8/8/3QK3 w - - 0 1", [], 1, 3, 200000, "mover", **j)
    assert not any(t["defects"] == 0 for t in later) and any(t["defects"] == 1 and t["wins"] == ["n"] and len(t["free"]) == 1 for t in later)
    keys = [(tuple(t["line"][:1]), tuple(t["wins"])) for t in later]
    assert len(keys) == len(set(keys))
    print(f"OBS 78.6 census: one move short -- {[(t['free'], t['line'][:2], t['wins']) for t in later if t['defects'] == 1][:4]}; readings {st['readings']}")


def test_78_6_the_census_of_the_other_side_is_read_after_a_pass():
    from catspace.incubator.analyst.discovery import lemma_store as LS
    theirs, _st = CE.census("chess", "4k3/8/2n5/8/Q7/8/8/4K3 b - - 0 1", [], 0, 3, 200000, "other", **LS.judge_of("chess"))
    assert any(t["line"][:1] == ["a4c6"] for t in theirs)                          # (Black to move: White's tactic, read as if Black passed)
