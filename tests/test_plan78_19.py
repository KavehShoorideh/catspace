"""PLAN 78.18 INTEGRATION (the main agent's tests over the subagent's pieces): the stronger rank source reaches Python -- the estimate's playout policy and leaf valuation
are chosen by keyword, the store's judge feeds the balance leaf, a snapshot's lemmas feed the lemma prior, sleep names a source `mcts:<rollout>/<leaf>`, and the rank
bench measures agreement and plays a match between two sources. The pieces alone are tested in rust/engine/planning/tests/rollout.rs and decision/tests/lemma_prior.rs."""
import hashlib
import json
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import rank_bench as RB
from catspace.incubator.analyst.discovery import sleep as SL

ROOT = Path(__file__).resolve().parents[1]
MATE = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]          # the win in one is h5f7 (plan 78.17's test position)
LEMMA = json.loads('''{"name": "cell-response-0265aee4", "body": [
 {"relation": "attack", "args": ["us", "?v2", "?v3", "?v0", "?v1"], "neg": false}, {"relation": "cell", "args": ["?v0", "?v3", "us", "pawn"], "neg": false},
 {"relation": "cell", "args": ["?v0", "?v1", "them", "pawn"], "neg": false}, {"relation": "cell", "args": ["?v0", "?v5", "them", "rook"], "neg": false},
 {"relation": "cell", "args": ["?v2", "?v3", "us", "pawn"], "neg": false}],
 "response": {"removed": [{"relation": "cell", "args": ["?v2", "?v3", "us", "pawn"], "neg": false}], "added": [{"relation": "cell", "args": ["?v2", "?v1", "us", "pawn"], "neg": false}]}}''')


def snapshot_with(tmp_path, lemmas: list, judge: dict) -> str:
    """A snapshot file of the chess store's vocabulary with `lemmas` and `judge`, named by its content id (the way the store names them)."""
    snap = {"format": 1, "game": "chess", "parent": None, "vocabulary": GS.vocabulary("chess"), "judge": judge, "teacher_used": False, "lemmas": lemmas}
    data = json.dumps(snap, sort_keys=True, separators=(",", ":")).encode()
    p = tmp_path / f"{hashlib.sha256(data).hexdigest()[:16]}.json"
    p.write_bytes(data)
    return str(p)


def test_78_18_the_estimate_names_its_policy_and_the_balance_leaf_reads_the_stores_judge(tmp_path):
    plain = CE.estimate("chess", None, MATE, seconds=0.1, seed=3)
    assert (plain["rollout"], plain["leaf"]) == ("uniform", "draw")
    store = snapshot_with(tmp_path, [], GS.judge_of("chess", "material"))
    r = CE.estimate("chess", None, MATE, seconds=0.2, seed=3, rollout="greedy", leaf="balance", store=store)
    assert r["rollout"] == "greedy" and r["leaf"].startswith("balance(material") and r["move"] == "h5f7"
    g = CE.estimate("chess", None, MATE, seconds=0.1, seed=3, rollout="gain", leaf="balance")
    assert g["rollout"].startswith("weighted(gain(count") and g["leaf"].startswith("balance(count")        # no store: every unit counts one
    with pytest.raises(ValueError):
        CE.estimate("chess", None, MATE, seconds=0.05, rollout="nosuch")
    with pytest.raises(ValueError):
        CE.estimate("chess", None, MATE, seconds=0.05, rollout="lemma")                                      # the lemma prior needs a store and the description


def test_78_18_the_lemma_prior_reads_a_snapshots_lemmas_through_the_variants_description(tmp_path):
    store = snapshot_with(tmp_path, [LEMMA], GS.judge_of("chess", "material"))
    assert CE.snapshot_info(store)["lemmas"] == ["cell-response-0265aee4"]
    src = SL.source_of("mcts:lemma/balance", 0.1, 1, store)
    assert src.source == "mcts:lemma/balance" and src.training_only is False and "chess" in src.specs
    r = CE.estimate("chess", None, MATE, seconds=0.3, seed=1, rollout="lemma", leaf="balance", store=store, spec=src.specs["chess"])
    assert r["rollout"].startswith("weighted(lemma(boost") and r["playouts"] >= 1 and r["status"] == "Read"
    # (the prior derives the position's facts at every playout ply: measured at about a millisecond a state, a thousand times a uniform ply -- see plan 78.18 --
    # so at this clock it finishes a handful of playouts; the policy is wired and named, its speed is the open item)


def test_78_18_sleep_names_a_rank_source_by_its_policy():
    assert SL.source_of("mcts").source == "mcts" and SL.source_of("mcts").options == {}
    s = SL.source_of("mcts:greedy/balance", store="x")
    assert s.source == "mcts:greedy/balance" and s.options == {"rollout": "greedy", "leaf": "balance", "store": "x"}
    assert RB.source_of("greedy+balance", 0.1, store="x").options == s.options
    assert RB.source_of("uniform", 0.1).source == "mcts"


class Fixed:
    """A rank source that always prefers the moves named first (a stand-in reference)."""
    def __init__(self, order): self.order = order
    def rank(self, variant, moves, cap=None):
        _fen, _mover, legal, _out, _key = CE.state(variant, None, list(moves))
        pref = [m for m in self.order if m in legal]
        rest = sorted(m for m in legal if m not in pref)
        return [(m, 0) for m in pref + rest]


def test_78_18_the_rank_bench_measures_agreement_and_samples_positions_spread_over_the_games(tmp_path):
    log = tmp_path / "wake.jsonl"
    rows = []
    for g in range(3):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "b1c3", "f8c5"]
        for ply in range(4, 8):
            rows.append({"kind": "position", "label": f"t:{g}:{ply}:{ply % 2}", "move": moves[ply], "steps": [{"operator": "ledger", "status": "Read", "events": [], "depth": 3}]})
        rows.append({"kind": "game", "run": "t", "game": g, "variant": "chess", "moves": moves, "outcome": None, "unfinished": True, "snapshot": ""})
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ps = RB.sample_positions(str(log), 6, seed=1)
    assert len(ps) == 6 and sorted(p["id"].split(":")[1] for p in ps) == ["0", "0", "1", "1", "2", "2"]      # two from each game
    ref = Fixed(["d2d4", "d7d5"])                                  # one preferred move a side (the sampled positions are of both sides)
    same = RB.agreement(ps, ref, Fixed(["d2d4", "d7d5"]))
    assert same["top1"] == 1.0 and same["mean_rank"] == 1.0 and same["top3"] == 1.0
    other = RB.agreement(ps, ref, Fixed(["a2a3", "h2h3", "b2b3", "d2d4", "a7a6", "h7h6", "b7b6", "d7d5"]))
    assert other["top1"] == 0.0 and other["ranks"] == [4] * 6 and other["top3"] == 0.0


def test_78_18_the_rank_bench_plays_a_match_by_the_games_own_outcome():
    a = Fixed(["d1h5", "h5f7", "e2e4", "f1c4"])                   # white: the scholar's mate order
    b = Fixed(["e7e5", "b8c6", "g8f6", "a7a6"])
    m = RB.play_match("chess", a, b, 2, random_plies=0, max_plies=40, seed=1)
    assert m["games"] == 2 and m["wins"] + m["draws"] + m["losses"] == 2 and 0.0 <= m["score"] <= 1.0
    assert m["unfinished"] <= 2 and m["plies"] <= 40
