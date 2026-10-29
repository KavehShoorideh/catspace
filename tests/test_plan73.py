"""PLAN 73 THE LEMMA STORE, part 1 (docs/ANALYST_PLAN.md 73.1-73.4, Kaveh 2026-10-25: the base concepts are movement and attack, tempo and material; every other word is
DERIVED and must be visibly so). The store loads and validates; every PROVED entry's theorems are proved theorems of docs/LEAN_INDEX.md; every word a run of the trace
guard's two fixed positions prints or sets is catalogued; the adapter returns its string unchanged and counts an unknown one; the K+Q technique has its measure; the report
prints its OBS lines; a renamed entry fails the validator and fires the counter (the negative control)."""
import json
import shutil

from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import words as WD


def test_the_store_loads_and_every_entry_keeps_the_schema():
    store = LS.load(force=True)
    print(f"OBS 73 store: {len(store)} entries")
    assert len(store) >= 60
    for n, e in store.items():
        assert LS.check_entry(e) == [], n
        assert e["reads"] and set(e["reads"]) <= set(LS.BASE), n
        assert (LS.DIR / (LS.slug(n) + ".json")).is_file(), n
    assert len({LS.slug(n) for n in store}) == len(store)                      # one file per name
    assert LS.implementation_problems(store) == []                             # every 'file:function' is defined where it says
    for n, e in store.items():
        for t in e["terms"]:
            assert LS.term_reads(t) <= set(e["reads"]), (n, t)                 # a term reads nothing its entry does not list


def test_every_proved_entry_cites_proved_theorems_of_the_index():
    store = LS.load()
    idx = LS.lean_index()
    proved = [e for e in store.values() if e["status"] == "PROVED"]
    print(f"OBS 73 PROVED entries {len(proved)}, theorems cited {len({t for e in proved for t in e['lean']})}")
    assert proved
    for e in proved:
        assert e["lean"], e["name"]
        for t in e["lean"]:
            assert idx.get(t) in ("theorem", "lemma"), (e["name"], t, idx.get(t))
    for e in store.values():
        if e["status"] != "PROVED":
            assert e["lean"] == [], e["name"]                                   # never a citation on an unproved word
    assert LS.citation_problems(store, idx) == []


def test_the_words_of_the_guards_fixed_positions_are_catalogued(tmp_path):
    """The trace guard's two fixed positions (the discovery position, pos2), run at the count budget and written as a small snapshot (the trace, the word list): every lemma,
    kill, GAP and judge cut the trace prints, every claim, kill and GAP of the root, every fact's lemma, and every string the agenda passed through the adapter
    (ctx.uncatalogued) is a word of the store."""
    runs = LS.fixed_run()
    total = 0
    for pid, trace, got, unc in runs:
        pairs = LS.extract_trace(trace) + got
        total += len(pairs)
        print(f"OBS 73 {pid}: words read {len(pairs)}, uncatalogued {len(LS.unresolved(pairs))}, ctx.uncatalogued {unc}")
        assert unc == 0, pid
        assert LS.unresolved(pairs) == [], (pid, LS.unresolved(pairs)[:5])
        assert "GAP: uncatalogued word" not in trace
        (tmp_path / f"{pid}.txt").write_text(trace)
        (tmp_path / f"{pid}{LS.SNAPSHOT_WORDS}").write_text("".join(f"{slot}\t{s}\n" for slot, s in sorted(set(got))))
    assert total > 0
    out = []
    assert LS.validate([str(tmp_path)], log=out.append, run=False) == 0, out
    print(out[0][:220])


def test_name_returns_its_string_and_counts_an_unknown_one():
    s = "a word no entry names (73 test)"
    n0 = LS.UNCATALOGUED[s]
    assert LS.name(s) is s
    assert LS.UNCATALOGUED[s] == n0 + 1
    k = "after Qxd5: no claim of ours and no material gain (-3)"
    u0 = LS.USES["no claim of ours and no material gain"]
    assert LS.name(k) is k
    assert LS.USES["no claim of ours and no material gain"] == u0 + 1
    assert LS.name(None) is None                                                # never raises in a run
    m = LS.mark()
    LS.name(s)
    n, new, uses = LS.since(m)
    assert n == 1 and new == [s] and not uses


def test_compositions_resolve_word_by_word():
    chase = "chase: step_after+step_least over chase: step_after+step_least over delayed_balance_alive_d1, mate_in_two"
    assert set(LS.resolve(chase)) == {"chase: step_after+step_least", "delayed_balance_alive_d1", "mate_in_two"}
    assert LS.resolve("chase: step_after+step_least over delayed_balance_alive_d1, no such word") is None          # an item of the list is a word or nothing (no detail swallows the rest)
    assert set(LS.resolve("chase: step_after+step_least over mate_in_two, stuck_reach (72.1): the priced reach, no reply read")) == \
        {"chase: step_after+step_least", "mate_in_two", "stuck_reach"}
    assert LS.resolve("chain stop: Kg1 not followed past Qd1+ (a second check of theirs in a row (not followed))") == ("a second check of theirs in a row",)
    assert LS.resolve("Q12 promote our P on b7 (2 pushes to b8): no stopper in time (distance 3 > pushes 1 + 1) and no capture binds") == ("no stopper in time",)
    assert LS.resolve("after Rf4: +0 stands only as a balance, their Rh8# wins 100 (lemma mate_in_one), no parry") == \
        ("... stands only as a balance, their ... wins ... (lemma ...), no parry", "mate_in_one")


def test_the_king_and_queen_technique_has_its_measure():
    e = LS.get("king and queen")
    assert e["kind"] == "TECHNIQUE" and e["status"] == "PROVED" and e["origin"] == "HAND"
    assert set(e["measure"]) == {"quantity", "floor", "terminal"}
    assert e["measure"]["terminal"] == "Ideas.QueenMate.kq_mate_on_edge" and "off the edge" in e["measure"]["quantity"]
    assert "Ideas.QueenMate.kq_mate_on_edge" in e["lean"]
    assert all(x["measure"] is None for x in LS.load().values() if x["kind"] != "TECHNIQUE")


def test_the_report_prints_its_obs_lines():
    lines = LS.report(log=lambda x: None)
    for x in lines:
        print(x[:220])
    heads = [x.split(":")[0] for x in lines]
    for h in ("OBS lemma store", "OBS lemma store by kind", "OBS lemma store by status", "OBS lemma store by origin"):
        assert h in heads
    assert any(x.startswith("OBS lemma store no citation") for x in lines)
    assert any(x.startswith("OBS lemma store entries with no reads (0)") for x in lines)
    assert any(x.startswith("OBS lemma store PROVED entries whose theorem is missing from LEAN_INDEX (0)") for x in lines)


def test_the_domain_words_hand_the_store():
    assert isinstance(WD.CHESS_WORDS, WD.Words)
    assert WD.CHESS_WORDS.lemmas() is LS.load()


def test_negative_control_a_renamed_entry_fails_validate_and_fires_the_counter(tmp_path, monkeypatch):
    """Rename the entry 'mate_in_one' in a copy of the store: the word the engine prints is no longer catalogued -- `resolve` is None, the adapter counts it, `validate` on a trace
    that prints it fails."""
    d = tmp_path / "lemmas"
    shutil.copytree(LS.DIR, d)
    e = json.loads((d / "mate_in_one.json").read_text())
    e["name"] = "mate_in_one (renamed)"
    (d / "mate_in_one.json").unlink()
    (d / (LS.slug(e["name"]) + ".json")).write_text(json.dumps(e))
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "x.txt").write_text("  -> CLAIM mate Qh7# (value 100), lemma mate_in_one, line Qh7#\nDECISION Qh7# 100 mate_in_one proved=True\n")
    monkeypatch.setattr(LS, "DIR", d)
    try:
        LS.load(force=True)
        assert LS.resolve("mate_in_one") is None
        n0 = LS.UNCATALOGUED["mate_in_one"]
        assert LS.name("mate_in_one") == "mate_in_one" and LS.UNCATALOGUED["mate_in_one"] == n0 + 1
        out = []
        assert LS.validate([str(snap)], log=out.append, run=False) == 1
        print("OBS 73 negative control:", out[0][:200])
        assert any("uncatalogued lemma" in x and "mate_in_one" in x for x in out)
    finally:
        monkeypatch.undo()
        LS.load(force=True)
    assert LS.resolve("mate_in_one") == ("mate_in_one",)
    assert LS.validate([str(snap)], log=lambda x: None, run=False) == 0


def test_76_3_e_use_counts_are_written_after_a_batch_and_the_unused_are_listed(tmp_path):
    """plan 76.3 (e): THE USE COUNTS -- a batch's tally (a guard snapshot directory: its traces and its word lists) is written into every entry's evidence.uses under
    the batch's name with its date, AFTER the batch (`lemma_store uses --write`); THE PRUNING RULE lists the entries never used in a batch tallied since they were
    written (the kinds read through their fields -- RULE, GOAL, VALUATION -- are no candidates); nothing is deleted."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "case1.txt").write_text("  -> KILLED Q3: verify: no route binds\n  -> KILLED Q4: verify: no route binds\nDECISION Qb7 3 standing kill proved=True\n")
    (snap / ("case1" + LS.SNAPSHOT_WORDS)).write_text("kill\ttempo arithmetic: the chain has spent its moves\n")
    c, unres = LS.tally(str(snap))
    assert c["verify: no route binds"] == 2 and c["tempo arithmetic"] == 1, (c, unres)
    store_dir = tmp_path / "lemmas"
    shutil.copytree(LS.DIR, store_dir, ignore=shutil.ignore_patterns("ontology", "surprises"))
    for f in store_dir.glob("*.json"):                                                                 # (a fresh store: the batches tallied so far left out)
        e = json.loads(f.read_text())
        e["evidence"]["uses"] = {}
        f.write_text(json.dumps(e, indent=1) + "\n")
    files = sorted(p.name for p in store_dir.glob("*.json"))
    n = LS.write_uses(c, "test batch", "2099-01-01", d=store_dir)
    assert n == len(files) and sorted(p.name for p in store_dir.glob("*.json")) == files               # every entry written, nothing deleted
    store, bad = LS._read_dir(store_dir)
    assert not bad, bad[:3]
    assert store["verify: no route binds"]["evidence"]["uses"]["test batch"] == {"count": 2, "date": "2099-01-01"}
    assert store["syzygy"]["evidence"]["uses"]["test batch"] == {"count": 0, "date": "2099-01-01"}
    cands = {name for name, *_ in LS.unused(store)}
    assert "syzygy" in cands and "verify: no route binds" not in cands and "tempo arithmetic" not in cands
    assert not any(store[x]["kind"] in LS.NOT_PRINTED for x in cands)
    assert LS.unused_lines(store)[0].startswith(f"OBS lemma store unused since written") and f": {len(cands)} of {len(store)} entries" in LS.unused_lines(store)[0]
    broken = dict(store["syzygy"], evidence=dict(store["syzygy"]["evidence"], uses={"test batch": 2}))
    assert any("evidence.uses" in x for x in LS.check_entry(broken))                                     # a use record is {count, date}
