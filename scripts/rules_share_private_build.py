"""PLAN 79.37 PART A: THE PRIVATE COUNTING BUILD of the Python binding (SYSTEM_SPEC 11; ledger rows 45, 44; piece: rules). It copies rust/engine into a folder OUTSIDE the repository (the session's scratchpad), adds one counter to each question of
`Game for KernelGame` (game/src/kernel.rs of the COPY) and one function `rules_counters()` to the binding's module (py/src/lib.rs of the COPY), builds the copy into a folder of its own and puts the module where PYTHONPATH can find it. The repository's
sources are not touched and the shared module (the one `import cs_engine` finds) is never rebuilt: scripts/rules_share_print.py prints the path of the module it imported.
  .venv/bin/python scripts/rules_share_private_build.py <folder outside the repository>
then:  PYTHONPATH=<folder>/pymod:<worktree> LCP_DATA=<main>/data .venv/bin/python scripts/rules_share_print.py profile --name counted
The build must use the python the shared module was built for (PYO3_PYTHON): a module built for another python crashes on import.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"mover": 0, "moves": 1, "apply": 2, "outcome": 3, "move_name": 4, "key": 5, "captured": 6, "taken_at": 7, "changed": 8, "pass": 9, "base_facts": 10, "units": 11}
COUNTERS = """/// PRIVATE COUNTING BUILD (plan 79.37 part A; never in the repository): how often each question of the `Game` trait is asked of the kernel.
pub static COUNTS: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
pub const COUNT_NAMES: [&str; 12] = ["mover", "moves", "apply", "outcome", "move_name", "key", "captured", "taken_at", "changed", "pass", "base_facts", "units"];
#[inline]
pub fn bump(i: usize) { COUNTS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
impl Game for KernelGame {"""
READER = """#[pyfunction]
fn rules_counters() -> Vec<(String, u64)> {
    cs_game::kernel::COUNT_NAMES.iter().zip(cs_game::kernel::COUNTS.iter()).map(|(n, c)| (n.to_string(), c.load(std::sync::atomic::Ordering::Relaxed))).collect()
}

#[pymodule]
fn cs_engine("""


def patch(engine: Path) -> None:
    kp = engine / "game" / "src" / "kernel.rs"
    lines = kp.read_text().split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("impl Game for KernelGame {"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    for i in range(start, end):
        m = re.match(r"^    fn (\w+)\(", lines[i])
        if m and m.group(1) in NAMES:
            j = lines[i].index("{")
            lines[i] = lines[i][:j + 1] + f" crate::kernel::bump({NAMES[m.group(1)]});" + lines[i][j + 1:]
    kp.write_text("\n".join(lines).replace("impl Game for KernelGame {", COUNTERS, 1))
    lp = engine / "py" / "src" / "lib.rs"
    t = lp.read_text().replace("#[pymodule]\nfn cs_engine(", READER, 1)
    t = t.replace("    m.add_function(wrap_pyfunction!(variants, m)?)?;", "    m.add_function(wrap_pyfunction!(variants, m)?)?;\n    m.add_function(wrap_pyfunction!(rules_counters, m)?)?;", 1)
    lp.write_text(t)


def main(argv) -> int:
    out = Path(argv[1]).resolve()
    if ROOT in out.parents or out == ROOT:
        raise SystemExit("the private build goes outside the repository")
    engine = out / "engine"
    shutil.copytree(ROOT / "rust" / "engine", engine, ignore=shutil.ignore_patterns("target"))
    patch(engine)
    env = dict(os.environ, CARGO_TARGET_DIR=str(out / "target"), RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup")
    env.setdefault("PYO3_PYTHON", sys.executable)
    subprocess.run(["cargo", "build", "--release", "--offline", "-p", "cs_engine"], cwd=engine, env=env, check=True)
    mod = out / "pymod" / "cs_engine"
    mod.mkdir(parents=True, exist_ok=True)
    import cs_engine
    shutil.copy(Path(cs_engine.__file__), mod / "__init__.py")
    shutil.copy(out / "target" / "release" / "libcs_engine.dylib", mod / f"cs_engine.cpython-{sys.version_info.major}{sys.version_info.minor}-darwin.so")
    print(f"OBS 79.37 A private build: {mod}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
