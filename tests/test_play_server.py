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
