"""PLAN 79.12 THE CHOOSER'S FIRST FIT (SYSTEM_SPEC 8 the rule for choosing a move and the wall, 7 "Calculation and estimate, decided together" points 4 to 6, 9 sleep; piece: the wake's last layer).
Rust holds the unit tests of the layer and of the reason (cargo test --release --offline -p cs_decision --test chooser). THIS file holds what belongs to Python: the wall for the sources (no game object named, no oracle, the fit never inside the wake), the training openings (never a judging opening),
the seed of a decision, the classes of the new reasons, the last layer through the binding, and the fit on a planted signal and on none. No match is played here."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import chooser_fit as CF
from catspace.incubator.analyst.discovery import whole_79_12 as W12
from catspace.incubator.analyst.discovery import whole_play as WP

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")


@pytest.mark.parametrize("name", ["whole_79_12.py", "chooser_fit.py"])
def test_the_wall_the_measure_scripts_name_no_game_object_and_reach_for_no_oracle(name):
    low = (DIR / name).read_text().lower()
    assert not re.search(r"\b[a-h][1-8]\b", low), re.search(r"\b[a-h][1-8]\b", low).group(0)
    assert "sys.path.insert" not in low
    for w in BANNED:
        assert not re.search(rf"\b{w}\b", low), (name, w)
    assert not re.search(r"ladder_oracle|leaf_ask|stockfish|analyse_pos|cs_learn|anthropic|\blo\.", low)


def test_the_fit_is_sleep_and_nothing_of_it_is_in_the_wake_or_the_binding():
    rust = "".join(p.read_text() for p in (ROOT / "rust" / "engine" / "decision" / "src").glob("*.rs"))
    assert "sklearn" not in rust and "LogisticRegression" not in rust and "bootstrap" not in rust.lower()
    play = (DIR / "whole_play.py").read_text()
    assert "chooser_fit" not in play and "sklearn" not in play, "the player never fits"
    binding = (ROOT / "rust" / "engine" / "py" / "src" / "base_graph_py.rs").read_text()
    assert "chooser" in binding and "cs_learn" not in (ROOT / "rust" / "engine" / "decision" / "Cargo.toml").read_text()


def test_no_training_opening_is_an_opening_that_judges_in_part_e():
    ops = W12.train_openings(130)                       # the sets C and D (100 pairs each) are walled first, so 130 reaches into C, D's start and no T needed; the assert inside also covers T
    judge = W12.judge_openings()
    assert len(ops) == 130 and len({tuple(o) for _s, o in ops}) == 130
    assert not ({tuple(o) for _s, o in ops} & judge)
    assert len(judge) == 200, "A and B: 100 distinct pairs each"


def test_a_game_replayed_gives_the_same_draws_and_a_move_of_another_ply_another_seed():
    spec = {"kind": WP.KIND_WAKE, "worth": {}, "chooser": W12.chooser_spec("explore", None, 7912)}
    m = WP.MoverWhole(dict(spec), 0.1, "game:7:0", "/tmp/none")
    n = WP.MoverWhole(dict(spec), 0.1, "game:7:0", "/tmp/none")
    assert m.chooser_arg(10) == n.chooser_arg(10)
    assert m.chooser_arg(10)[2] != m.chooser_arg(12)[2]
    assert m.chooser_arg(10)[0] == "explore"
    plain = WP.MoverWhole({"kind": WP.KIND_WAKE, "worth": {}}, 0.1, "game:7:0", "/tmp/none")
    assert plain.chooser_arg(10) is None


def test_the_new_reasons_are_classed_from_the_sentences_of_the_log():
    for why, want in (("the equal reading put the search's first move lower, 1216 below the best at that reading; this move is held equal to the best", "first_lowered"),
                      ("only its own line wins by its own reading: net 5 (x)", "only_own_line"), ("only it answers the threat of theirs on x, which stands; the search's first move does not", "only_answers"),
                      ("only it fixes the goal on x (bears); the search's first move does not", "only_fixes"),
                      ("nothing separates it from the search's first move but the estimates: both answer the threat of theirs that stands; the reasons' score picks it", "estimate"),
                      ("nothing separates it from the search's first move: no threat, goal or estimate decides", "search_only"),
                      ("explore: one of the 3 moves the search holds equal", "explore"), ("the chooser: the largest weighted sum of its inputs", "chooser"),
                      ("the end of the game is forced in its line: x", "forced_end")):
        assert WP.reason_of(why) == want, why


def _world():
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import odds_measure as OM
    w = WP.world()
    return CE, OM, w


def _decide(chooser, moves=(), seconds=0.15):
    CE, OM, w = _world()
    pos = CE.state(OM.VARIANT, None, list(moves))[0]
    history = [CE.state(OM.VARIANT, None, list(moves[:k]))[4] for k in range(len(moves))]
    from catspace.incubator.analyst.discovery import record_play as RP
    return json.loads(w.model.wake_decide(pos, history, seconds, 64, dict(RP.worth_of("human")), {}, w.goal, WP.PIECES, 0, 0, w.keys, WP.fuses(), w.chances, w.reasons, WP.TM.LEDGER_SHARE, None, chooser))


def test_the_binding_takes_the_last_layer_and_refuses_a_name_that_is_no_mode_or_input():
    out = _decide(("explore", {}, 5))
    assert out["move"] in out["equal"] or not out["equal"]
    c = out["log"]["chooser"]
    if c is not None:
        assert c["mode"] == "Explore" and len(c["inputs"]) == len(c["candidates"][0]["row"]) and set(x["move"] for x in c["candidates"]) <= set(out["equal"])
    assert "step_seconds" in out["log"] and "first_held" in out["log"]
    with pytest.raises(Exception, match="mode"):
        _decide(("nothing", {}, 0))
    with pytest.raises(Exception, match="input"):
        _decide(("weights", {"no_such_input": 1.0}, 0))
    plain = _decide(None)
    assert plain["log"]["chooser"] is None
    fixed = _decide(("fixed", {}, 0))
    assert fixed["move"] == plain["move"] or fixed["equal"] != plain["equal"]                # (a clock: equal sets can differ; with the same equal set the rule decides alike)


# ------------------------------------------------------------------------------------------------------------------------------ the fit

def _data(effect: float, games: int = 300, per: int = 6, seed: int = 1) -> "CF.Data":
    rng = np.random.default_rng(seed)
    X, seat, game, white = [], [], [], []
    for g in range(games):
        s = rng.integers(0, 2)
        score = 0.0
        rows = []
        for _ in range(per):
            x = np.array([rng.normal(), rng.normal(), rng.normal(), 0.0])
            rows.append(x)
            score += effect * x[0]
        p = 1 / (1 + np.exp(-score))
        w = float(rng.random() < p)
        white.append(w if s == 0 else 1 - w)
        for x in rows:
            X.append(x)
            seat.append(s)
            game.append(g)
    return CF.Data(np.array(X), np.array(seat), np.array(game), np.array(white), [str(g) for g in range(games)], ["signal", "noise_a", "noise_b", "never_varies"])


def test_the_fit_finds_a_planted_input_clear_on_held_out_and_stops_on_a_null():
    d = _data(0.5)
    r = CF.analyse(d, boot=60, shuffles=4)
    assert "never_varies" in r["dropped_nought"]
    k = r["names"].index("signal")
    assert r["clear"][k], "the planted input is clear of nought on held-out games"
    assert r["beta_fit"][k] > 0 and r["weights"]["signal"] > 0
    assert sum(int(x["clear"][k]) for x in r["shuffled"]) <= 1, "with the results shuffled the planted input is not clear"
    lines = CF.lines_of("t", ["r"], d, r)
    assert any("clear on held-out: YES" in x for x in lines) and not any("STOP" in x for x in lines)
    null = CF.analyse(_data(0.0, seed=3), boot=60, shuffles=4)
    lines = CF.lines_of("t", ["r"], _data(0.0, seed=3), null)
    if not null["clear"].any():
        assert any("STOP: no weight's band is clear of nought" in x for x in lines) and null["weights"] == {}


def test_an_input_identical_to_an_earlier_one_is_one_input():
    d = _data(0.3, games=120)
    d.X[:, 2] = d.X[:, 1]
    r = CF.analyse(d, boot=20, shuffles=1)
    assert r["dropped_duplicate"] == {"noise_b": "noise_a"} and "noise_b" not in r["names"]
