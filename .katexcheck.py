import json
import pathlib
import re

t = pathlib.Path("README.md").read_text()
theory = t[t.index("## The formal frame"):t.index("## The repo, at a glance")]
blocks = re.findall(r"\$\$(.+?)\$\$", theory, re.S)
inline = re.findall(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", theory)
print("display blocks:", len(blocks), "inline:", len(inline))
out = [{"m": b, "d": True} for b in blocks] + [{"m": i, "d": False} for i in inline]
pathlib.Path("/tmp/katex_in.json").write_text(json.dumps(out))
