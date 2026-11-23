#!/usr/bin/env python
"""PLAN 79.36 PART A: THE INVENTORY OF THE UNITS OF CODE OF LAYERS 0, 1 AND 2 AGAINST THEIR TESTS (SYSTEM_SPEC 11, last block; pieces: base, representation, rules; ledger row 43).

Reads the Rust sources of rust/engine/base (cs_base, layer 0), rust/engine/rep (cs_rep, layer 1) and rust/engine/game (cs_game, layer 2) and prints, per crate, the public items, the
items a test calls, the items no test calls, and the items on the list of exceptions (scripts/layer_units_exceptions.txt, each with a cause).

    python scripts/layer_units_print.py            OBS lines: counts per crate
    python scripts/layer_units_print.py --list     also the untested items (file:line)
    python scripts/layer_units_print.py --table    also every item with the tests that call it
    python scripts/layer_units_print.py --crate cs_rep   one crate only

WHAT IS AN ITEM (found by this program from the text, comments and string literals blanked first):
  - `pub fn`, `pub const`, `pub static` at module level; the function or constant is named by its module path.
  - `pub fn` and `pub const` inside an inherent `impl Type {` where Type is a `pub struct`/`pub enum`/`pub union` of the same crate: named `Type::name`.
  - every `fn` of a `pub trait` (declared with or without a default body): named `Trait::name`.
  - every `fn` of an `impl Trait for Type` whose trait is not one of the standard library's (Display, Default, From, Iterator, ...): named `Type::<Trait>::name`; it is public behaviour a caller reaches
    through the trait.
  NOT items: `pub(crate)`, `pub(super)`, `pub(in ..)` and private items; a `pub use` re-export (the item is counted where it is defined); derived impls (`#[derive]`) and impls of the standard
  library's traits (the program cannot see a derived body, and a Display or Default impl is exercised wherever the value is printed or made); `pub struct`/`enum` types and their fields and variants
  (a type is used wherever its methods are called); a binary under src/bin; generic items are counted once (not once per type parameter); the items of `#[cfg(test)]` modules; items inside macro_rules.

WHAT IS A TEST: a `fn` marked `#[test]` and not `#[ignore]`d, in a `#[cfg(test)]` module of a source or in a file under the crate's own tests/ folder (the crate's own tests only: a test of another crate
does not count for this crate), and whose call closure holds an assertion (`assert*!`, `panic!`, `unwrap_err`, `should_panic`): a test that asserts nothing does not count. The call closure is the test's body
plus the bodies of the helper functions it calls by name (helpers are the non-test `fn`s of the test sources of the crate), taken to the end of the chain.

WHAT IS A CALL (an item is TESTED when a test's closure calls it, by name, on a path that resolves to it):
  STRONG    a path call `Type::name(` or `Trait::name(` to the owner; a call `module::name(` or a bare `name(` to a free function (a bare call inside a `#[cfg(test)]` module resolves to a function of the same
            file first); an associated constant by `Type::NAME`; a constant by its word; a method call `.name(` when exactly one item of the crate has that method name and the name is not one a std type also has.
  WEAK      a method call `.name(` when the name belongs to several items of the crate (or is also a std method name, `get`, `len`, `insert`, `push` ...): credited to each owner Type whose name appears as a word in
            the closure. The count of items credited only this way is printed (the gate counts them tested; they are the first ones to read by hand).
KNOWN BLIND SPOTS (a text scan is not a type checker): (1) a method call on a receiver of unknown type is credited by the name rule above; two types that both appear in a test and share a method name are both
credited when only one is called (an over-credit); (2) a function used as a value (`.map(f)`) and a method used as a path value are not seen as calls (an under-credit); (3) tests written by a macro are not seen;
(4) a call inside a closure of a helper is seen, but a call made by a function the test reaches in the crate's own sources is not credited to that function (only the test's own text counts); (5) `Self::name(` is read
as a call to the owner of the enclosing impl only inside a test module's own impl; (6) a free function of the same name in two modules is credited to both when the call is bare and the file is not a source of one of them;
(7) the program does not see whether the asserted value is the right one: that is read by hand against the board.
Prints OBS lines only; exits 0 (the gate, tests/test_plan79_36.py, decides).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
CRATES = {"cs_base": ENGINE / "base", "cs_rep": ENGINE / "rep", "cs_game": ENGINE / "game"}
EXCEPTIONS = Path(__file__).with_name("layer_units_exceptions.txt")

STD_TRAITS = {
    "Display", "Debug", "Default", "Clone", "Copy", "From", "Into", "TryFrom", "TryInto", "PartialEq", "Eq", "Hash", "PartialOrd", "Ord", "Iterator", "IntoIterator", "FromIterator",
    "Extend", "Deref", "DerefMut", "Drop", "Add", "Sub", "Mul", "Div", "Neg", "Not", "Rem", "AddAssign", "SubAssign", "MulAssign", "Index", "IndexMut", "FromStr", "AsRef", "AsMut", "Borrow",
    "Error", "Write", "Read", "Send", "Sync", "Fn", "FnMut", "FnOnce", "ToString", "Sum", "Product", "BitAnd", "BitOr", "BitXor", "Shl", "Shr",
}
# a method name a std type also has: a call `.name(` on an unknown receiver may be the std one
STD_COMMON = {
    "new", "get", "get_mut", "len", "is_empty", "insert", "remove", "push", "pop", "iter", "iter_mut", "contains", "contains_key", "clear", "extend", "first", "last", "map", "min", "max", "sum",
    "clone", "next", "take", "entry", "keys", "values", "sort", "join", "split", "from", "into", "as_str", "as_ref", "write", "read", "load", "save", "hash", "fmt", "cmp", "eq", "ne", "lt", "le",
    "gt", "ge", "add", "sub", "mul", "div", "neg", "not", "index", "default", "drop", "to_string", "parse", "unwrap", "expect", "filter", "fold", "count", "rev", "zip", "chain", "find", "any", "all",
    "position", "retain", "drain", "append", "extend_from_slice", "truncate", "swap", "reverse", "dedup", "binary_search", "abs", "pow", "floor", "ceil", "round", "sqrt", "signum", "is_some", "is_none",
    "is_ok", "is_err", "ok", "err", "or", "and", "then", "flatten", "copied", "cloned", "bytes", "chars", "lines", "trim", "starts_with", "ends_with", "replace", "format", "display", "borrow",
    "borrow_mut", "set", "step", "apply", "run", "build", "open", "close", "start", "finish", "update", "reset", "name", "id", "key", "value", "size", "kind", "rows", "cols",
}
KEYWORDS = {"if", "while", "for", "match", "return", "loop", "fn", "let", "in", "as", "move", "else", "mut", "ref", "where", "impl", "dyn", "pub", "use", "mod", "struct", "enum", "trait", "type",
            "const", "static", "unsafe", "async", "await", "Some", "None", "Ok", "Err", "Box", "Vec", "Self", "self", "super", "crate"}


# ---------------------------------------------------------------------------------------- blanking
def blank(text: str) -> str:
    """The source without comments, string literals and character literals (each replaced by spaces / an empty literal), line structure kept."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    out.append("\n" if text[i] == "\n" else " ")
                    i += 1
        elif c == "r" and re.match(r'r#*"', text[i:i + 6]) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'r(#*)"', text[i:])
            end = '"' + m.group(1)
            j = text.find(end, i + m.end())
            j = n if j < 0 else j + len(end)
            out.append('""' + "".join("\n" for ch in text[i:j] if ch == "\n"))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append('""' + "".join("\n" for ch in text[i:j + 1] if ch == "\n"))
            i = j + 1
        elif c == "'":
            if text.startswith("'\\", i):
                j = text.find("'", i + 3)
                j = n if j < 0 else j + 1
                out.append("' '")
                i = j
            elif i + 2 < n and text[i + 2] == "'":
                out.append("' '")
                i += 3
            else:
                out.append(c)  # a lifetime
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------------------- parsing
@dataclass
class Item:
    crate: str
    kind: str            # fn | method | trait_method | impl_method | const | assoc_const
    name: str            # as printed
    base: str            # the bare name looked up in calls
    owner: str           # the type or trait, or ""
    mod: str             # module path inside the crate
    file: str
    line: int
    trait: str = ""
    ret: tuple = ()      # the type names of the return type (the result of a call is of one of them)
    tests: set = field(default_factory=set)
    weak: set = field(default_factory=set)


@dataclass
class FnRec:
    name: str
    owner: str
    body: str
    attrs: list
    file: str
    line: int
    in_test_mod: bool
    mod: str


def split_attrs(header: str) -> tuple[list[str], str]:
    attrs, i, n = [], 0, len(header)
    header = header.lstrip()
    n = len(header)
    while header.startswith("#"):
        j = header.find("[")
        if j < 0:
            break
        depth, k = 0, j
        while k < n:
            if header[k] == "[":
                depth += 1
            elif header[k] == "]":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        attrs.append(header[j + 1:k].strip())
        header = header[k + 1:].lstrip()
        n = len(header)
    return attrs, header


def return_names(body: str) -> tuple:
    """The capitalised names in the return type of the `fn` text `body` (after its argument list), `Self` left out."""
    i = body.find("(")
    if i < 0:
        return ()
    depth, rest = 0, ""
    for j in range(i, len(body)):
        depth += body[j] in "(["
        depth -= body[j] in ")]"
        if depth == 0:
            rest = body[j + 1:]
            break
    m = re.match(r"\s*->\s*(.*)", rest, flags=re.S)
    if not m:
        return ()
    ty = re.split(r"\bwhere\b", m.group(1))[0]
    return tuple(sorted({n for n in re.findall(r"\b[A-Z]\w*", ty) if n != "Self"}))


def strip_generics(s: str) -> str:
    s = s.replace("->", "")
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def base_type(t: str) -> str:
    t = strip_generics(t)
    t = re.sub(r"\bwhere\b.*", "", t, flags=re.S).strip()
    t = re.sub(r"^(&\s*(mut\s+)?|dyn\s+|\*\s*(const|mut)\s+)+", "", t).strip()
    m = re.search(r"([A-Za-z_]\w*)\s*$", t)
    return m.group(1) if m else t


def parse_impl_header(rest: str) -> tuple[str, str]:
    """(trait, type) of `impl<..> [Trait for] Type<..>`; trait is '' for an inherent impl."""
    r = rest[len("impl"):].strip().replace("->", "  ")
    if r.startswith("<"):
        depth = 0
        for i, ch in enumerate(r):
            depth += ch == "<"
            depth -= ch == ">"
            if depth == 0:
                r = r[i + 1:].strip()
                break
    r = re.sub(r"\bwhere\b.*", "", r, flags=re.S)
    parts = re.split(r"\bfor\b", r, maxsplit=1)
    if len(parts) == 2:
        return base_type(parts[0]), base_type(parts[1])
    return "", base_type(parts[0])


@dataclass
class Ctx:
    kind: str              # mod | impl | trait | opaque
    name: str = ""
    trait: str = ""
    test: bool = False
    pub: bool = False
    start: int = 0         # index in the blanked text where the body starts
    hdr: str = ""
    attrs: list = field(default_factory=list)
    line: int = 0


class Source:
    def __init__(self, crate: str, path: Path, mod: str, is_test_file: bool):
        self.crate, self.path, self.mod, self.is_test_file = crate, path, mod, is_test_file
        self.rel = path.relative_to(ROOT).as_posix()
        self.text = blank(path.read_text())
        self.items: list[Item] = []
        self.fns: list[FnRec] = []
        self.pub_types: set[str] = set()
        self.local_traits: set[str] = set()
        self.pending_impl_items: list[tuple[Item, str]] = []   # (item, impl type) checked against pub types later
        self.parse()

    def line_of(self, idx: int) -> int:
        return self.text.count("\n", 0, idx) + 1

    def parse(self):
        t, n = self.text, len(self.text)
        stack = [Ctx("mod", name=self.mod, test=self.is_test_file, pub=True)]
        buf_start, paren, i = 0, 0, 0
        opaque_depth = 0
        while i < n:
            c = t[i]
            top = stack[-1]
            if top.kind == "opaque":
                if c == "{":
                    opaque_depth += 1
                elif c == "}":
                    opaque_depth -= 1
                    if opaque_depth == 0:
                        self.close(stack.pop(), i)
                        buf_start = i + 1
                i += 1
                continue
            if c in "([":
                paren += 1
            elif c in ")]":
                paren -= 1
            elif paren <= 0 and c == "{":
                paren = 0
                header = t[buf_start:i]
                kind = self.open_block(header, stack, buf_start, i)
                ctx = Ctx(kind["kind"], kind.get("name", ""), kind.get("trait", ""), kind.get("test", False), kind.get("pub", False), start=i + 1, hdr=header, line=self.line_of(i))
                ctx.attrs = kind.get("attrs", [])
                stack.append(ctx)
                if ctx.kind == "opaque":
                    opaque_depth = 1
                buf_start = i + 1
            elif paren <= 0 and c == "}":
                paren = 0
                if len(stack) > 1:
                    self.close(stack.pop(), i)
                buf_start = i + 1
            elif paren <= 0 and c == ";":
                paren = 0
                self.declaration(t[buf_start:i], stack, buf_start)
                buf_start = i + 1
            i += 1

    # -- what a closed block means
    def close(self, ctx: Ctx, end: int):
        if ctx.kind == "opaque" and ctx.name == "fn":
            body = self.text[ctx.start:end]
            owner = ""
            self.fns.append(FnRec(ctx.trait, ctx.hdr, body, ctx.attrs, self.rel, ctx.line, False, ""))  # patched in open_block via side table

    def open_block(self, header: str, stack: list, hstart: int, idx: int) -> dict:
        attrs, rest = split_attrs(header)
        top = stack[-1]
        in_test = top.test or any("cfg(test)" in a.replace(" ", "") for a in attrs)
        m = re.match(r"(pub(\s*\([^)]*\))?)\s+", rest)
        vis = m.group(1) if m else ""
        is_pub = vis == "pub"
        body = rest[m.end():] if m else rest
        body = re.sub(r"^((default|const|async|unsafe)\s+)*(?=fn\b)", "", body) if re.match(r"((default|const|async|unsafe|extern\s*\"?\"?)\s+)*fn\b", body) else body
        line = self.line_of(hstart + len(header) - len(header.lstrip()))
        mm = re.match(r"mod\s+(\w+)", body)
        if mm:
            return {"kind": "mod", "name": mm.group(1), "test": in_test, "pub": is_pub, "attrs": attrs}
        if re.match(r"impl\b", body):
            trait, typ = parse_impl_header(body)
            return {"kind": "impl", "name": typ, "trait": trait, "test": in_test, "attrs": attrs}
        mm = re.match(r"(unsafe\s+)?trait\s+(\w+)", body)
        if mm:
            tname = mm.group(2)
            if is_pub and not in_test:
                self.local_traits.add(tname)
            return {"kind": "trait", "name": tname, "test": in_test, "pub": is_pub, "attrs": attrs}
        mm = re.match(r"(struct|enum|union)\s+(\w+)", body)
        if mm:
            if is_pub and not in_test:
                self.pub_types.add(mm.group(2))
            return {"kind": "opaque", "name": "type", "test": in_test}
        mm = re.match(r"fn\s+(\w+)", body)
        if mm:
            self.record_fn(mm.group(1), is_pub, attrs, stack, in_test, idx, line, body_follows=True, ret=return_names(body))
            return {"kind": "opaque", "name": "fn", "trait": mm.group(1), "hdr": header, "test": in_test, "attrs": attrs, "owner": top.name if top.kind in ("impl", "trait") else ""}
        mm = re.match(r"(const|static)\s+(mut\s+)?(\w+)", body)
        if mm and not body.startswith("const fn"):
            self.record_const(mm.group(3), is_pub, stack, in_test, line)
        return {"kind": "opaque", "name": "other", "test": in_test}

    def declaration(self, header: str, stack: list, hstart: int):
        attrs, rest = split_attrs(header)
        top = stack[-1]
        if top.kind not in ("mod", "impl", "trait"):
            return
        in_test = top.test or any("cfg(test)" in a.replace(" ", "") for a in attrs)
        m = re.match(r"(pub(\s*\([^)]*\))?)\s+", rest)
        vis = m.group(1) if m else ""
        is_pub = vis == "pub"
        body = rest[m.end():] if m else rest
        line = self.line_of(hstart + len(header) - len(header.lstrip()))
        mm = re.match(r"((const|async|unsafe)\s+)*fn\s+(\w+)", body)
        if mm and top.kind == "trait":
            self.record_fn(mm.group(3), is_pub, attrs, stack, in_test, 0, line, body_follows=False, ret=return_names(body))
            return
        mm = re.match(r"(const|static)\s+(mut\s+)?(\w+)", body)
        if mm and not body.startswith("const fn"):
            self.record_const(mm.group(3), is_pub, stack, in_test, line)
        mm = re.match(r"(struct|enum|union)\s+(\w+)", body)
        if mm and is_pub and not in_test and top.kind == "mod":
            self.pub_types.add(mm.group(2))

    # -- records
    def mod_path(self, stack) -> str:
        parts = [self.mod] if self.mod else []
        parts += [c.name for c in stack[1:] if c.kind == "mod"]
        return "::".join(parts)

    def record_fn(self, name, is_pub, attrs, stack, in_test, idx, line, body_follows, ret=()):
        top = stack[-1]
        modp = self.mod_path(stack)
        if in_test:
            return  # helper/test bodies are collected by collect_fns
        if top.kind == "mod" and is_pub and not self.is_test_file:
            self.items.append(Item(self.crate, "fn", f"{modp}::{name}" if modp else name, name, "", modp, self.rel, line, ret=ret))
        elif top.kind == "impl" and not self.is_test_file:
            typ, trait = top.name, top.trait
            if trait:
                if trait in STD_TRAITS:
                    return
                self.pending_impl_items.append((Item(self.crate, "impl_method", f"{typ}::<{trait}>::{name}", name, typ, modp, self.rel, line, trait=trait, ret=ret), typ))
            elif is_pub:
                self.pending_impl_items.append((Item(self.crate, "method", f"{typ}::{name}", name, typ, modp, self.rel, line, ret=ret), typ))
        elif top.kind == "trait" and top.pub and not self.is_test_file:
            self.items.append(Item(self.crate, "trait_method", f"{top.name}::{name}", name, top.name, modp, self.rel, line, trait=top.name, ret=ret))

    def record_const(self, name, is_pub, stack, in_test, line):
        top = stack[-1]
        if in_test or self.is_test_file or not is_pub and top.kind != "trait":
            return
        modp = self.mod_path(stack)
        if top.kind == "mod":
            self.items.append(Item(self.crate, "const", f"{modp}::{name}" if modp else name, name, "", modp, self.rel, line))
        elif top.kind == "impl" and not top.trait and is_pub:
            self.pending_impl_items.append((Item(self.crate, "assoc_const", f"{top.name}::{name}", name, top.name, modp, self.rel, line), top.name))


def module_of(path: Path, src_root: Path) -> str:
    rel = path.relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] in ("mod", "lib"):
        parts = parts[:-1]
    return "::".join(parts)


# a second, simpler pass for function bodies: each `fn` with the span of its braces, found on the blanked text
FN_RE = re.compile(r"((?:#\[[^\]]*\]\s*)*)(?:pub(?:\s*\([^)]*\))?\s+)?(?:(?:const|async|unsafe|extern)\s+)*fn\s+(\w+)")


def collect_fns(src: Source, test_mod_spans: list[tuple[int, int]]) -> list[FnRec]:
    t = src.text
    out = []
    for m in FN_RE.finditer(t):
        # find the body: the first `{` after the signature at paren depth 0 before any `;`
        i, depth = m.end(), 0
        while i < len(t):
            c = t[i]
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
            elif depth <= 0 and c == ";":
                i = -1
                break
            elif depth <= 0 and c == "{":
                break
            i += 1
        if i < 0 or i >= len(t):
            continue
        j, d = i, 0
        while j < len(t):
            if t[j] == "{":
                d += 1
            elif t[j] == "}":
                d -= 1
                if d == 0:
                    break
            j += 1
        in_test_mod = src.is_test_file or any(a <= m.start() < b for a, b in test_mod_spans)
        attrs = re.findall(r"#\[([^\]]*)\]", m.group(1))
        out.append(FnRec(m.group(2), "", t[i:j + 1], attrs, src.rel, src.text.count("\n", 0, m.start()) + 1, in_test_mod, src.mod))
    return out


def test_mod_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for m in re.finditer(r"#\[cfg\(test\)\]\s*(?:pub\s+)?mod\s+\w+\s*\{", text):
        j, d = m.end() - 1, 0
        while j < len(text):
            if text[j] == "{":
                d += 1
            elif text[j] == "}":
                d -= 1
                if d == 0:
                    break
            j += 1
        spans.append((m.start(), j))
    return spans


# ---------------------------------------------------------------------------------------- the inventory
def sources_of(crate: str, root: Path) -> list[Source]:
    out = []
    src = root / "src"
    for p in sorted(src.rglob("*.rs")):
        if "bin" in p.relative_to(src).parts[:1]:
            continue
        out.append(Source(crate, p, module_of(p, src), False))
    tests = root / "tests"
    if tests.exists():
        for p in sorted(tests.rglob("*.rs")):
            out.append(Source(crate, p, module_of(p, tests), True))
    return out


CALL_RE = re.compile(r"((?:[A-Za-z_]\w*\s*(?:::\s*<[^<>(){};]*>\s*)?::\s*)*)([A-Za-z_]\w*)\s*(?:::\s*<[^>(){};]*>\s*)?\(")
METHOD_RE = re.compile(r"\.\s*([A-Za-z_]\w*)\s*(?:::\s*<[^>(){};]*>\s*)?\(")
ASSERT_RE = re.compile(r"\bassert\w*\s*!|\bpanic\s*!|unwrap_err|should_panic|debug_assert\w*\s*!")


def closure_text(fn: FnRec, helpers: dict[str, list[FnRec]]) -> str:
    seen, todo, texts = set(), [fn], []
    while todo:
        f = todo.pop()
        key = (f.file, f.name, f.line)
        if key in seen:
            continue
        seen.add(key)
        texts.append(f.body)
        for m in CALL_RE.finditer(f.body):
            if m.group(1).strip().endswith("::") or not m.group(1).strip() or True:
                for h in helpers.get(m.group(2), []):
                    if (h.file, h.name, h.line) not in seen:
                        todo.append(h)
    return "\n".join(texts)


def analyse(crate: str, root: Path):
    sources = sources_of(crate, root)
    items: list[Item] = []
    pub_types: set[str] = set()
    local_traits: set[str] = set()
    pending: list[tuple[Item, str]] = []
    for s in sources:
        pub_types |= s.pub_types
        local_traits |= s.local_traits
        pending += s.pending_impl_items
        items += s.items
    for it, typ in pending:
        if it.kind == "impl_method":
            if it.trait in local_traits or typ in pub_types or True:
                items.append(it)
        elif typ in pub_types:
            items.append(it)
    # tests and helpers
    tests: list[FnRec] = []
    helpers: dict[str, list[FnRec]] = defaultdict(list)
    for s in sources:
        spans = test_mod_spans(s.text)
        for f in collect_fns(s, spans):
            if not f.in_test_mod:
                continue
            f.mod = s.mod
            is_test = any(re.fullmatch(r"test|tokio::test", a.strip()) for a in f.attrs)
            ignored = any(a.strip().startswith("ignore") for a in f.attrs)
            if is_test:
                if not ignored:
                    tests.append(f)
            else:
                helpers[f.name].append(f)
    # index of items
    by_base: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_base[it.base].append(it)
    method_owners: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        if it.kind in ("method", "trait_method", "impl_method"):
            method_owners[it.base].append(it)
    type_names = {it.owner for it in items if it.owner}
    mod_names = {it.mod.split("::")[-1] for it in items if it.mod} | {p for it in items for p in it.mod.split("::") if p}
    asserting_tests = 0
    for ts in tests:
        text = closure_text(ts, helpers)
        if not ASSERT_RE.search(text) and not any("should_panic" in a for a in ts.attrs):
            continue
        asserting_tests += 1
        label = f"{ts.file}::{ts.name}"
        words = set(re.findall(r"[A-Za-z_]\w*", text))
        # the result of a call is of the type its signature names: `m.sleep().add(..)` names no `Sleep`, but `sleep` returns one
        for mm in list(CALL_RE.finditer(text)) + list(METHOD_RE.finditer(text)):
            for it in by_base.get(mm.group(mm.lastindex if mm.re is METHOD_RE else 2), []):
                words.update(it.ret)
        # path and bare calls
        for m in CALL_RE.finditer(text):
            quals = [q for q in re.split(r"\s*::\s*", re.sub(r"::\s*<[^<>]*>", "", m.group(1)).strip()) if q]
            name = m.group(2)
            if name in KEYWORDS:
                continue
            start = m.start()
            if start > 0 and text[start - 1] == ".":
                continue
            cands = by_base.get(name, [])
            if not cands:
                continue
            q = quals[-1] if quals else ""
            if q:
                for it in cands:
                    if it.kind in ("method", "trait_method", "impl_method", "assoc_const") and (it.owner == q or (q == "Self" and it.owner in words)):
                        it.tests.add(label)
                    elif it.kind == "impl_method" and it.trait == q and it.owner in words:
                        it.tests.add(label)
                    elif it.kind in ("fn", "const") and q in it.mod.split("::"):
                        it.tests.add(label)
            else:
                free = [it for it in cands if it.kind == "fn"]
                same = [it for it in free if it.file == ts.file]
                for it in (same or free):
                    it.tests.add(label)
        # method calls
        for m in METHOD_RE.finditer(text):
            name = m.group(1)
            owners = method_owners.get(name, [])
            if not owners:
                continue
            distinct = {(o.owner, o.trait) for o in owners}
            if len(distinct) == 1 and name not in STD_COMMON:
                for o in owners:
                    o.tests.add(label)
            else:
                for o in owners:
                    if o.owner in words:
                        o.weak.add(label)
        # constants by word
        for it in items:
            if it.kind == "const" and it.base in words:
                same_name = by_base[it.base]
                if len([x for x in same_name if x.kind == "const"]) == 1 or any(f"{p}::{it.base}" in text for p in it.mod.split("::")):
                    it.tests.add(label)
            elif it.kind == "assoc_const" and re.search(rf"\b{re.escape(it.owner)}\s*::\s*{re.escape(it.base)}\b", text):
                it.tests.add(label)
    # a trait's method is exercised when a method of an impl of that trait of the same name is
    for it in items:
        if it.kind == "trait_method":
            for o in items:
                if o.kind == "impl_method" and o.trait == it.owner and o.base == it.base:
                    it.tests |= o.tests
                    it.weak |= o.weak
    return items, len(tests), asserting_tests


def load_exceptions() -> dict[str, str]:
    out = {}
    if EXCEPTIONS.exists():
        for line in EXCEPTIONS.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, cause = line.partition("|")
            out[key.strip()] = cause.strip()
    return out


def inventory(only: str | None = None) -> dict[str, dict]:
    """{crate: {items, strong, weak_only, untested, excepted, tests, asserting_tests, stale}}; used by tests/test_plan79_36.py."""
    exc = load_exceptions()
    res = {}
    used = set()
    for crate, root in CRATES.items():
        if only and crate != only:
            continue
        items, ntests, nassert = analyse(crate, root)
        # an item of the same printed name twice (two impls) stays two items
        tested, weak_only, untested, excepted = [], [], [], []
        for it in items:
            key = f"{crate} {it.name}"
            if it.tests:
                tested.append(it)
            elif it.weak:
                weak_only.append(it)
            elif key in exc:
                excepted.append(it)
                used.add(key)
            else:
                untested.append(it)
        res[crate] = dict(items=items, tested=tested, weak_only=weak_only, untested=untested, excepted=excepted, tests=ntests, asserting=nassert)
    res["_stale_exceptions"] = {k: v for k, v in exc.items() if k not in used and (not only or k.startswith(only))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--crate")
    a = ap.parse_args()
    res = inventory(a.crate)
    tot = defaultdict(int)
    for crate, r in res.items():
        if crate.startswith("_"):
            continue
        n, s, w, u, e = len(r["items"]), len(r["tested"]), len(r["weak_only"]), len(r["untested"]), len(r["excepted"])
        cov = (s + w) / max(1, n)
        print(f"OBS 79.36 A {crate}: items {n} tested {s + w} (strong {s}, weak only {w}) untested {u} excepted {e}; tests {r['tests']} of which asserting {r['asserting']}; items tested {cov:.3f}")
        for k, v in (("items", n), ("tested", s + w), ("untested", u), ("excepted", e)):
            tot[k] += v
    print(f"OBS 79.36 A all three crates: items {tot['items']} tested {tot['tested']} untested {tot['untested']} excepted {tot['excepted']}")
    stale = res.get("_stale_exceptions", {})
    if stale:
        print(f"OBS 79.36 A stale exceptions (listed, but not an item or now tested): {len(stale)}")
        for k in stale:
            print(f"  stale: {k}")
    if a.list or a.table:
        for crate, r in res.items():
            if crate.startswith("_"):
                continue
            print(f"--- {crate}: untested")
            for it in sorted(r["untested"], key=lambda x: (x.file, x.line)):
                print(f"  {it.file}:{it.line} {it.name} [{it.kind}]")
            print(f"--- {crate}: excepted")
            for it in sorted(r["excepted"], key=lambda x: (x.file, x.line)):
                print(f"  {it.file}:{it.line} {it.name}")
    if a.table:
        for crate, r in res.items():
            if crate.startswith("_"):
                continue
            print(f"--- {crate}: table")
            for it in sorted(r["items"], key=lambda x: (x.file, x.line)):
                ts = sorted(it.tests) or sorted(it.weak)
                tag = "strong" if it.tests else ("weak" if it.weak else "none")
                print(f"  {it.file}:{it.line} {it.name} [{it.kind}] {tag} {len(ts)} {ts[0].split('::')[-1] if ts else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
