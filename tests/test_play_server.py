"""Plan 54 (5): the play server end to end through the HTTP API (no browser): 6 plies against the agenda with the random fallback."""
import json
import threading
import time
import urllib.error
import urllib.request

import chess
import pytest

from catspace.incubator.analyst import web
from catspace.incubator.analyst.discovery import play_server as PS


@pytest.fixture()
def server(tmp_path):
    box, ev = {}, threading.Event()

    def ready(s):
        box["s"] = s
        ev.set()
    pm = PS.PlayManager("random", seed=1, pgn=str(tmp_path / "games.pgn"))
    old = (web.Handler.jobs, web.Handler.play)
    threading.Thread(target=lambda: web.serve(0, 1, 1, 1, 1, play=pm, ready=ready), daemon=True).start()
    assert ev.wait(10)
    yield box["s"].server_address[1], tmp_path
    box["s"].shutdown()
    pm.game and setattr(pm.game, "live", False)
    web.Handler.jobs, web.Handler.play = old


def call(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body is not None else None)
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return {"http": e.code, **json.loads(e.read())}


def wait_idle(port, limit=40):
    t0 = time.time()
    while time.time() - t0 < limit:
        st = call(port, "/api/play/state")
        if not st["thinking"]:
            return st
        time.sleep(0.1)
    raise AssertionError("engine still thinking")


def test_six_plies_stream_and_name_the_fallback(server):
    port, tmp = server
    st = call(port, "/api/play/new", {"colour": "white"})
    assert st["turn"] == "white" and st["dests"] and st["fallback"] == "random"
    board = chess.Board()
    since, recs = 0, []
    for _ in range(3):
        mv = chess.Move.from_uci(sorted(f + t for f, ts in st["dests"].items() for t in ts)[0])
        assert mv in board.legal_moves
        board.push(mv)
        st = call(port, "/api/play/move", {"uci": mv.uci()})
        assert st["thinking"] or st["over"] or st["ply"] == len(board.move_stack)
        st = wait_idle(port)
        em = st["engine_move"]
        assert em and chess.Move.from_uci(em["uci"]) in board.legal_moves     # the engine's move is legal on our mirror board
        board.push(chess.Move.from_uci(em["uci"]))
        assert st["fen"] == board.fen()
        assert em["source"] == "agenda" or em["source"] == "fallback:random"
        t = call(port, f"/api/play/trace?since={since}")
        new = t["records"]
        assert new and all(r["n"] > since for r in new)
        assert all(set(r) >= {"n", "text", "kind", "indent", "moves", "squares"} and r["kind"] in PS.KINDS for r in new)
        since = t["n"]
        recs += new
        if em["source"].startswith("fallback"):
            assert any(r["text"].startswith("no decision: fallback random chose") for r in new)
        else:
            assert any(r["kind"] == "decision" for r in new)
    assert st["ply"] == 6
    assert [r["n"] for r in recs] == list(range(1, len(recs) + 1))            # contiguous, streamed in order
    assert call(port, "/api/play/move", {"uci": "a1a1"}).get("http") == 400      # illegal / out of turn is refused
    pgn = (tmp / "games.pgn").read_text()
    assert "[Fallback \"random\"]" in pgn and "1. " in pgn


def test_trace_hook_streams_every_append():
    seen = []
    tl = PS.TraceLines(seen.append)
    tl.append("a"); tl += ["b", "c\nd"]
    assert seen == ["a", "b", "c", "d"] and list(tl) == ["a", "b", "c\nd"]


def test_classify_and_parse_moves():
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    r = PS.make_record(1, "  -> CLAIM mate Qxf7# (value 100), lemma mate_in_one", b)
    assert r["kind"] == "claim" and r["indent"] == 2 and r["moves"][0]["from"] == "h5" and r["moves"][0]["to"] == "f7"
    assert PS.classify("Q3 win their R on a8 (worth 5) (stake 5): x") == "question"
    assert PS.classify("  -> KILLED Q1: tempo") == "kill" and PS.classify("STOP (a): done") == "stop"
    assert PS.make_record(2, "Q1 win their P on a7 (worth 1)", b)["moves"] == []     # a square is not a pawn push
    assert PS.make_record(3, "garbage ((( Zz9 O-O-O", b)["moves"] == []             # never a crash


def test_the_time_budget_is_set_on_new_and_changed_between_moves(server):
    port, tmp = server
    st = call(port, "/api/play/new", {"colour": "white", "seconds": 0.3})
    assert st["seconds"] == 0.3
    st = call(port, "/api/play/settings", {"seconds": 1.5})
    assert st["seconds"] == 1.5 and call(port, "/api/play/state")["seconds"] == 1.5
    assert call(port, "/api/play/settings", {"seconds": -1}).get("http") == 400
    assert call(port, "/api/play/settings", {"seconds": None})["seconds"] is None
    st = call(port, "/api/play/new", {"colour": "black", "seconds": 0.2})                    # the engine moves first: its trace names the budget
    st = wait_idle(port)
    lines = [r["text"] for r in call(port, "/api/play/trace?since=0")["records"]]
    assert any(t.startswith("budget: 0.2 s, used ") for t in lines)


def test_past_positions_the_whole_game_is_kept_per_ply_and_a_ply_can_be_asked_for(server):
    port, tmp = server
    st = call(port, "/api/play/new", {"colour": "white"})
    board = chess.Board()
    for _ in range(2):
        mv = chess.Move.from_uci(sorted(f + t for f, ts in st["dests"].items() for t in ts)[0])
        board.push(mv)
        call(port, "/api/play/move", {"uci": mv.uci()})
        st = wait_idle(port)
        board.push_san(st["san"][-1])
    assert len(st["san"]) == 4 and [m["san"] for m in st["moves"]] == st["san"] and [m["ply"] for m in st["moves"]] == [1, 2, 3, 4]
    assert all(m["fen"] for m in st["moves"]) and st["moves"][-1]["fen"] == board.fen()                        # the position after every ply, for the page's move list
    t2 = call(port, "/api/play/trace?ply=2")
    assert t2["ply"] == 2 and t2["records"] and all(r["ply"] == 2 for r in t2["records"])                      # only the engine's thinking before its move at ply 2
    assert all(r["fen"] == st["moves"][0]["fen"] for r in t2["records"])                                       # ... each parsed against the position before that move
    allr = call(port, "/api/play/trace?since=0")["records"]
    assert {r["ply"] for r in allr if r["kind"] != "background"} == {2, 4} and len(allr) >= len(t2["records"]) + len(call(port, "/api/play/trace?ply=4")["records"])
    assert call(port, "/api/play/trace?ply=1")["records"] == []                                                # a human ply has no engine trace


def test_the_trace_store_keeps_every_record_with_no_ring_buffer_loss():
    tb = PS.TraceBuffer()
    b = chess.Board()
    for i in range(7000):
        tb.add(f"line {i}", b, 1 + i % 3)
    assert len(tb.since(0)) == 7000 and len(tb.at_ply(2)) in (2333, 2334)


def test_52_19_3_the_server_thinks_on_the_humans_clock_and_the_lines_are_marked_background(server):
    port, tmp = server
    st = call(port, "/api/play/new", {"colour": "white"})
    call(port, "/api/play/move", {"uci": "e2e4"})
    st = wait_idle(port)
    t0, bg = time.time(), []
    while time.time() - t0 < 60 and not bg:                       # he has not moved: the worker is thinking, its lines stream to the trace
        bg = [r for r in call(port, "/api/play/trace?since=0")["records"] if r["kind"] == "background"]
        time.sleep(0.3)
    assert bg and all(r["text"].startswith("background:") for r in bg)
    assert all(r["ply"] == st["ply"] + 1 for r in bg)             # the ply the human is to play
    st = call(port, "/api/play/state")
    assert not st["thinking"] and st["dests"]                     # thinking in the background is not the engine's move: he can move at once
    mv = sorted(f + t for f, ts in st["dests"].items() for t in ts)[0]
    t1 = time.time()
    call(port, "/api/play/move", {"uci": mv})
    st = wait_idle(port, 60)
    assert st["ply"] == 4 and time.time() - t1 < 40               # his move stopped the worker, and the engine answered


def test_52_19_4_the_player_carries_the_fact_store_between_its_moves():
    pl = PS.AgendaPlayer("random", seed=3)
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    mv, first, _ = pl.move(b)
    store = pl.carry.facts
    b.push(mv)
    b.push_san("a6")                                               # the far wing
    mv2, second, _ = pl.move(b)
    assert pl.carry.facts is store                                 # the same store: move 2 is asked against what move 1 derived
    served = [ln for ln in second if " -- served from the store (fact #" in ln]
    assert served, "a fact derived on the first move whose reads are untouched must serve the second"
    print(f"OBS 52.19 (4): the player's second move was served {len(served)} root questions from the first move's facts")
