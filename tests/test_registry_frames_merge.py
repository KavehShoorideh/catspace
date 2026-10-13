"""The generated track's frames: a run replaces only the materials it owns (registry.save_frames)."""
from catspace.incubator.analyst.discovery import registry as R


def test_save_frames_keeps_other_materials(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FRAMES", tmp_path / "frames.json")
    R.save_frames({"KQvKR": ["(count edge)"], "KQvKR:check": ["(count corner)"]}, "run a")
    R.save_frames({"KBNvK": ["(count attacked)"]}, "run b", owned={"KBNvK"})
    assert sorted(R.load_frames()) == ["KBNvK", "KQvKR", "KQvKR:check"]
    R.save_frames({"KQvKR": ["(count free)"]}, "run c", owned={"KQvKR"})         # its old check frame goes with it
    assert R.load_frames() == {"KBNvK": ["(count attacked)"], "KQvKR": ["(count free)"]}
