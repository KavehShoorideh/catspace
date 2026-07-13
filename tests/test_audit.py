"""catspace/audit.py: the Stockfish-leakage safety gate. Fast, no torch/shard
dependency for the detector-logic tests; the "real codebase is clean" test
needs torch (imports TorchFB/FBBoardPolicy) so it's skipped without it."""
import pytest

from catspace.audit import (_scan, audit_checkpoint, checkpoint_provenance_check,
                            is_provenance_clean)


def test_scan_catches_forbidden_tokens():
    def leaky(batch, device):
        return batch.meta["eval_cp"]
    assert "eval_cp" in _scan(leaky)


def test_scan_catches_stockfish_word():
    def leaky():
        return "stockfish"
    assert "stockfish" in _scan(leaky)


def test_scan_clean_function_has_no_hits():
    def clean(x, y):
        return x @ y
    assert _scan(clean) == []


def test_scan_does_not_false_positive_on_compound_identifiers():
    """'stockfish_free' contains 'stockfish' as a substring but is the
    audit's OWN vocabulary, not a data leak -- caught for real once
    (train_lichess_fb.main used to dict-index this literal key)."""
    def uses_the_result(provenance):
        return provenance["stockfish_free"]
    # the naive substring scan WOULD flag this -- documented, not asserted
    # clean, since _scan is intentionally crude; is_provenance_clean() is
    # the actual API callers use to sidestep it (see next test).
    assert "stockfish" in _scan(uses_the_result)


def test_is_provenance_clean_reads_the_flag_without_reflecting_the_word():
    """The whole reason is_provenance_clean exists: callers check cleanliness
    without their own source containing the forbidden substring."""
    assert is_provenance_clean({"stockfish_free": True}) is True
    assert is_provenance_clean({"stockfish_free": False}) is False
    assert is_provenance_clean({}) is False


def test_checkpoint_provenance_check_missing_is_unknown_not_clean():
    r = checkpoint_provenance_check({})
    assert r["status"] == "unknown"


def test_checkpoint_provenance_check_explicit_clean():
    r = checkpoint_provenance_check({"provenance": {"stockfish_free": True}})
    assert r["status"] == "clean"


def test_checkpoint_provenance_check_explicit_dirty():
    r = checkpoint_provenance_check({"provenance": {"stockfish_free": False}})
    assert r["status"] == "contaminated"


def test_audit_checkpoint_dirty_provenance_fails_even_if_static_is_clean():
    torch = pytest.importorskip("torch")
    payload = {"provenance": {"stockfish_free": False}}
    result = audit_checkpoint(payload)
    assert result["clean"] is False
    assert result["provenance"]["status"] == "contaminated"


def test_audit_checkpoint_unknown_provenance_passes_on_clean_static():
    """Pre-audit-era checkpoints (no provenance stamp) are tolerated -- the
    static check is the fallback, not a second hard requirement."""
    torch = pytest.importorskip("torch")
    result = audit_checkpoint({})
    assert result["provenance"]["status"] == "unknown"
    assert result["clean"] == result["static"]["clean"]


def test_static_purity_check_passes_on_the_real_codebase():
    """The actual invariant this module exists to protect: right now, the FB
    training path and the planner's read path never touch Stockfish-derived
    identifiers. If this test starts failing, something added a real leak
    path (or an audit-vocabulary word landed somewhere it shouldn't have --
    see test_scan_does_not_false_positive_on_compound_identifiers)."""
    torch = pytest.importorskip("torch")
    from catspace.audit import static_purity_check
    r = static_purity_check()
    assert r["clean"], r["hits"]
