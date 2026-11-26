"""PLAN 79.59 (THE LEMMA MINER OF LAYER 2; SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules, and the lemma machinery of row 49; ledger rows 59, 58, 57, 49).

THIS TEXT IS THE INTERFACE, written before any body (docs/AGENT_BRIEF.md section 0). The module is game-free: it names no relation, sort, constant or unit of any description. Everything it says about a description it reads from the clauses (the reader's `Program`, the
translator's `Description` and the support finder of pass 26). No lemma is written by a person and no proposer searches: each proposer is a CONSTRUCTION on the clauses, with the reason it is valid.

WHAT IT TAKES AND RETURNS (who calls it: scripts/lemma_miner_run.py, the judge, and the printing script scripts/lemma_miner_print.py).
  `shapes(text) -> [Shape]`                     the tests of the shape of pass 26 found in the composed description `text` (a test R of a key on a hypothetical relation H made from a state relation B: the support finder's own dicts), and the reasons for those refused.
  `null_change(text, shape) -> Proposal`        PROPOSER proposer one, THE NULL CHANGE. Reads, from the clauses of R and of every relation made from H that R reaches, the same clauses with H read as B and the key taken out (R0, "the test as the state stands") and builds, as clauses of the
                                                description's own language, a delta that defines: R0 (and its helpers, each `<name>_null`), the DIFFERING ROWS of a key (`<R>_diff`: the rows where the world of the key differs from the state, read off the views of H), one relation for each READ of
                                                R0's proof (`<R>_hit_<i>`: the keys whose differing rows meet the row-pattern that read makes), their union `<R>_hits`, and `<R>_answer` (the answer of R0 at the class of the key). It also returns, for each read, where the clause came from
                                                (the root clause, the chain of clauses it was unfolded through, the literal) so that a statement can be written in the description's own words. Raises `Refused` naming what in the clauses stands in the way (it does not guess).
  `Proposal.document(game, variant) -> dict`    the lemma as the document of the form FILTER (`rule_lemma_filter.py`, beside `rule_lemma.py`) (the delta, the two relations, the filter's hits relation, the reads, the standing PROPOSED).
  `statement(proposal, counts) -> [str]`        PROPOSER proposer two: the readable statement, read off the explanations: "the keys that can change the answer are those that change the row read by literal j of clause c", one line for each read, with the counts the judge printed.
  `explanation_analysis(...)`                   what follows, mechanically, from an explained read (a read used for existence only; a literal whose free variable another literal of its clause binds and could be joined before): see `read_findings`.
THE SOUNDNESS OF proposer one (the reason, written once). Let a world be the facts of the state relation B. H(K, .) is B outside the differing rows D(K) (what the views of H add and what their negated literals take out: read off the clauses by pass 26's finder). R(K) is a test whose inputs are
  the facts of H(K, .) and the guards at K; with H read as B it is R0 on the state. If no differing row of K meets a pattern R0 read on the state, the worlds agree on every input the test asked, and the answer of K is the answer of R0 at the class of K. This is the Lean theorem
  `Ideas.Support.Test.run_of_disjoint` with the reference world the state itself (the reference's differing set void): lean/ideas/Ideas/NullFilter.lean states it so (`Test.run_of_null`). The reads R0 makes are not recorded; a SUPERSET of them is written from the clauses: for each literal of
  the state relation in a clause of R0 (unfolded through the relations that are not recursive), the pattern it reads under every valuation of the literals before it. A superset keeps the lemma valid and costs narrowness only.
WHAT IT REFUSES, with the reason printed: a guard literal that is not one template for every clause; a compound term; a clause of R0 that is not safe once the key is taken out; a callee whose head is not distinct variables; no guard at all.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import gdl_to_rust as T


class Refused(Exception):
    """The clauses do not allow the construction; the message says what stands in the way."""


# ------------------------------------------------------------------------------------------------------------------------------------------ small helpers on terms and clauses

def is_var(t) -> bool:
    return isinstance(t, str) and t.startswith("?")


def term_text(t) -> str:
    if isinstance(t, tuple):
        return "(" + " ".join(term_text(x) for x in t) + ")"
    return t


def lit_text(sign: str, rel: str, args) -> str:
    """A literal as written in a clause (a literal of a state fluent as `(true (f ...))`)."""
    body = " ".join(term_text(a) for a in args)
    inner = f"(true ({rel[len('true.'):]}{' ' + body if body else ''}))" if rel.startswith("true.") else f"({rel}{' ' + body if body else ''})"
    return f"(not {inner})" if sign == "neg" else inner


def clause_text(head: tuple, body: list) -> str:
    h = lit_text("pos", head[0], head[1])
    return "(<= " + h + "".join(" " + (lit_text(l[0], l[1], l[2]) if l[0] != "distinct" else f"(distinct {term_text(l[1])} {term_text(l[2])})") for l in body) + ")"


def vars_of(t, acc=None) -> set:
    acc = set() if acc is None else acc
    if isinstance(t, tuple):
        for x in t:
            vars_of(x, acc)
    elif is_var(t):
        acc.add(t)
    return acc


def subst(t, m: dict):
    if isinstance(t, tuple):
        return tuple(subst(x, m) for x in t)
    return m.get(t, t) if isinstance(t, str) else t


def lit_vars(l) -> set:
    if l[0] == "distinct":
        return vars_of(l[1]) | vars_of(l[2])
    acc: set = set()
    for a in l[2]:
        vars_of(a, acc)
    return acc


def subst_lit(l, m: dict):
    if l[0] == "distinct":
        return ("distinct", subst(l[1], m), subst(l[2], m))
    return (l[0], l[1], tuple(subst(a, m) for a in l[2]))


# ------------------------------------------------------------------------------------------------------------------------------------------ what the proposer builds

@dataclass
class Read:
    """One READ of R0's proof: the state relation, the pattern it reads (the terms of the literal) under the valuations of `prefix`, and where it stands in the clauses."""
    index: int
    relation: str                     # the hit relation made for it
    pattern: tuple                    # the terms of the literal of the state relation, in the variables of this read
    sign: str
    root: str                         # the text of the clause of R0 (or of a relation of R0 that is recursive) the read belongs to
    trail: list                       # [(clause text, literal text)] from the root down to the clause that holds the literal (the chain of unfoldings)
    literal: str                      # the text of the literal read
    clause: str                       # the text of the description's clause the literal comes from


@dataclass
class Proposal:
    name: str
    test: str                         # R
    places: int                       # the places of the key
    state_relation: str               # B
    hypothetical: str                 # H
    null_test: str                    # R0
    answer: str                       # the answer of R0 at the class of a key
    hits: str                         # the keys whose differing rows meet a read
    diff: str                         # the differing rows
    signatures: list = field(default_factory=list)
    clauses: list = field(default_factory=list)
    reads: list = field(default_factory=list)
    guard: str = ""                   # the template of the guard literal, as text
    guard_relation: str = ""          # the relation of the guard: the keys it holds are the domain of the test
    tested: str = ""                  # the copy of the test whose cone is made for the keys of the hits only (part E, the mask)
    masked: list = field(default_factory=list)   # the relations of that copy
    outside_readers: list = field(default_factory=list)   # relations outside the cone that read a relation of it: they keep reading the whole ones
    diff_views: list = field(default_factory=list)   # the relations the differing rows are read off (what a view of the hypothetical relation adds, what a negated literal of a copy view names), in the order of the clauses
    cone: list = field(default_factory=list)     # the test, the relations made from the hypothetical relation that it reaches, the hypothetical relation and its views: made only to answer the test
    classes: int = 0                  # the places of the class of a key (the variables the guard binds besides the key)
    reasons: list = field(default_factory=list)

    @property
    def delta(self) -> list:
        return list(self.signatures) + list(self.clauses)


# ------------------------------------------------------------------------------------------------------------------------------------------ the shapes

def shapes(text: str) -> tuple:
    """-> (description, finder): the translator's `Description` of the text and its support finder, run. `finder.found` are the shapes, `finder.refused` {relation: reason}."""
    d = T.Description(text)
    f = T.SupportFinder(d)
    f.run()
    return d, f


# ------------------------------------------------------------------------------------------------------------------------------------------ proposer one, the null change

def null_change(text: str, shape: dict, d: "T.Description | None" = None, finder: "T.SupportFinder | None" = None) -> Proposal:
    if d is None or finder is None:
        d, finder = shapes(text)
    prog = d.prog
    R, H, B, m, p = shape["R"], shape["H"], shape["B"], shape["m"], shape["p"]
    kp: dict = dict(shape["kp"])                                # relation -> the places of its key, for R and the relations made from H it reaches
    cl = finder.cl
    taken = set(prog.nodes) | set(prog.signatures)

    def fresh_name(base: str) -> str:
        n = base
        while n in taken:
            n += "_"
        taken.add(n)
        return n

    def sig(rel: str) -> tuple:
        s = prog.signature(rel)
        if s is None:
            raise Refused(f"{rel} has no declared signature of sorts")
        return tuple(s)

    # ---- the guard: the literals that hold a key variable and are not of H or of a relation made from it
    guards: list = []                                           # (sign, rel, args, key variables in order) found in the clauses
    for x in sorted(kp):
        for c in cl[x]:
            kv = [c.head[1][j] for j in kp[x]]
            for l in c.body:
                if l[0] == "distinct":
                    continue
                sign, rel, args = l
                if any(isinstance(a, tuple) for a in args):
                    raise Refused(f"a compound term in a literal of the cone: {c.text}")
                if rel == H or rel in kp:
                    continue
                if any(a in kv for a in args):
                    guards.append((sign, rel, tuple(args), kv, c))
    if not guards:
        raise Refused("no guard literal holds the key in the clauses of the test: the class of a key is not known")

    def template(g) -> tuple:
        sign, rel, args, kv, _c = g
        slot: dict = {}
        out = []
        for a in args:
            if a in kv:
                out.append(("k", kv.index(a)))
            elif is_var(a):
                out.append(("c", slot.setdefault(a, len(slot))))
            else:
                out.append(("const", a))
        return sign, rel, tuple(out)

    tpl = template(guards[0])
    for g in guards[1:]:
        if template(g) != tpl:
            raise Refused(f"two different guard literals ({tpl} and {template(g)}): the class of a key is not one template ({g[4].text})")
    gsign, grel, gplaces = tpl
    if gsign != "pos":
        raise Refused("the guard literal is negated")
    nclass = 1 + max([x[1] for x in gplaces if x[0] == "c"], default=-1)
    gsig = sig(grel)
    class_sorts = [None] * nclass
    for j, x in enumerate(gplaces):
        if x[0] == "c":
            class_sorts[x[1]] = gsig[j]

    def guard_literal(keyvars: list, classvars: list):
        args = []
        for x in gplaces:
            args.append(keyvars[x[1]] if x[0] == "k" else classvars[x[1]] if x[0] == "c" else x[1])
        return ("pos", grel, tuple(args))

    # ---- which relations carry the class of the key: those with a guard literal in a clause, and those that call one that carries it
    carries = {x: False for x in kp}
    changed = True
    while changed:
        changed = False
        for x in kp:
            if carries[x]:
                continue
            for c in cl[x]:
                if any(l[0] != "distinct" and (l[1] == grel and any(a in [c.head[1][j] for j in kp[x]] for a in l[2]) or (l[1] in kp and carries.get(l[1]))) for l in c.body):
                    carries[x] = True
                    changed = True
                    break
    if not carries[R]:
        raise Refused(f"the test {R} does not read the guard {grel}: its answer would not depend on the class of the key")
    null = {x: fresh_name(f"{x}_null") for x in kp}
    out_sigs: list = []
    for x in kp:
        s = sig(x)
        keep = [s[j] for j in range(len(s)) if j not in kp[x]]
        out_sigs.append(f"(signature {null[x]}" + "".join(" " + a for a in (class_sorts if carries[x] else []) + keep) + ")")

    # ---- R0 and its helpers: every clause of every relation, H read as B, the key taken out, the guard literal taken out and its class carried
    null_clauses: dict = {x: [] for x in kp}                    # relation -> [(head args, body, source clause)]
    for x in sorted(kp):
        for c in cl[x]:
            kv = [c.head[1][j] for j in kp[x]]
            if len(set(kv)) != len(kv) or not all(is_var(a) for a in kv):
                raise Refused(f"the key places of {x} are not distinct variables: {c.text}")
            used = set()
            for t in c.head[1]:
                vars_of(t, used)
            for l in c.body:
                used |= lit_vars(l)
            counter = itertools.count()

            def fresh_var() -> str:
                while True:
                    v = f"?q{next(counter)}"
                    if v not in used:
                        used.add(v)
                        return v
            gl = [l for l in c.body if l[0] != "distinct" and l[1] == grel and any(a in kv for a in l[2])]
            if len(gl) > 1:
                raise Refused(f"two guard literals in one clause: {c.text}")
            classvars: list = []
            if carries[x]:
                if gl:
                    t = template((gl[0][0], gl[0][1], tuple(gl[0][2]), kv, c))
                    slot_var: dict = {}
                    for a, pl in zip(gl[0][2], gplaces):
                        if pl[0] == "c":
                            slot_var[pl[1]] = a
                    classvars = [slot_var.get(i) or fresh_var() for i in range(nclass)]
                else:
                    classvars = [fresh_var() for _ in range(nclass)]
            head_rest = [c.head[1][j] for j in range(len(c.head[1])) if j not in kp[x]]
            if any(a in kv for a in head_rest):
                raise Refused(f"a key variable stands at a place outside the key of {x}: {c.text}")
            body: list = []
            origs: list = []                                    # the literal of the description each literal of `body` comes from, as written
            for l in c.body:
                if l[0] == "distinct":
                    body.append(l)
                    origs.append(f"(distinct {term_text(l[1])} {term_text(l[2])})")
                    continue
                sign, rel, args = l
                if l in gl:
                    continue
                origs.append(lit_text(sign, rel, args))
                if rel == H:
                    body.append((sign, B, tuple(args[m:])))
                elif rel in kp:
                    rest = [args[j] for j in range(len(args)) if j not in kp[rel]]
                    body.append((sign, null[rel], tuple((classvars if carries[rel] else []) + rest)))
                else:
                    body.append(l)
            null_clauses[x].append((tuple(classvars + head_rest), body, c, origs))
            # safety: every variable of the head, of a negated literal and of a test is bound by a positive literal
            bound = set()
            for l in body:
                if l[0] == "pos":
                    bound |= lit_vars(l)
            need = set()
            for t in classvars + head_rest:
                vars_of(t, need)
            for l in body:
                if l[0] != "pos":
                    need |= lit_vars(l)
            if need - bound:
                raise Refused(f"the clause of {x} is not safe once the key is taken out ({sorted(need - bound)} are bound by no positive literal): {c.text}")
            for l in body:
                if lit_vars(l) & set(kv):
                    raise Refused(f"a key variable remains in the clause once the key is taken out: {c.text}")

    clauses: list = []
    for x in sorted(kp):
        for hargs, body, _c, _o in null_clauses[x]:
            clauses.append(clause_text((null[x], hargs), body))

    # ---- the differing rows of a key: what the views of H add, and the rows of B that a negated literal of a copy view names
    diff = fresh_name(f"{R}_diff")
    out_sigs.append(f"(signature {diff}" + "".join(" " + a for a in sig(H)) + ")")
    n = m + p
    diff_views: list = []
    for v, form in shape["views"]:
        if form == "add":
            vs = [f"?d{i}" for i in range(n)]
            clauses.append(clause_text((diff, tuple(vs)), [("pos", v, tuple(vs))]))
            diff_views.append(v)
            continue
        for c in cl[v]:
            got = finder.copy_clause(c, m)
            if got is None:
                raise Refused(f"a view clause of {H} that is not a copy: {c.text}")
            Bv, _guards, negs = got
            row = c.head[1][m:]
            for rel, args in negs:
                clauses.append(clause_text((diff, tuple(c.head[1])), [("pos", Bv, tuple(row)), ("pos", rel, tuple(args))]))
                diff_views.append(rel)

    # ---- the reads of R0's proof, with their contexts
    recursive: set = set()
    for x in kp:
        visited: set = set()
        stack = [l[1] for c in cl[x] for l in c.body if l[0] != "distinct" and l[1] in kp]
        while stack:
            y = stack.pop()
            if y in visited:
                continue
            visited.add(y)
            stack += [l[1] for c in cl[y] for l in c.body if l[0] != "distinct" and l[1] in kp]
        if x in visited:
            recursive.add(x)
    by_null = {null[x]: x for x in kp}
    reads: list = []
    counter = itertools.count()

    def rename(hargs, body, tag: int):
        names: set = set()
        for t in hargs:
            vars_of(t, names)
        for l in body:
            names |= lit_vars(l)
        mp = {v: f"?{tag}_{v[1:]}" for v in names}
        return tuple(subst(a, mp) for a in hargs), [subst_lit(l, mp) for l in body]

    def res(t, eq: dict):
        """A term with the equalities found so far put in (a variable is followed until it is no longer equated to another term)."""
        if isinstance(t, tuple):
            return tuple(res(x, eq) for x in t)
        while isinstance(t, str) and t in eq:
            t = eq[t]
        return t

    def unify(x, y, eq: dict) -> bool:
        x, y = res(x, eq), res(y, eq)
        if x == y:
            return True
        if is_var(x):
            eq[x] = y
            return True
        if is_var(y):
            eq[y] = x
            return True
        return False

    def walk(hargs, body, origs: list, prefix: list, trail: list, root_text: str, root_class: list, src: "GDL.Clause", eq: dict):
        """Every read of a state relation in this clause body (and in what it calls that is not recursive), each with the positive literals before it. `eq`: the equalities between terms that the unfoldings made (a callee whose head holds a variable twice). `origs`: the literal of the description each
        literal of the body comes from, as written (a read is named by that literal in the statement)."""
        prefix = list(prefix)
        for i, l in enumerate(body):
            if l[0] == "distinct":
                continue
            sign, rel, args = l
            if rel.startswith("true."):
                reads.append((rel, tuple(args), sign, list(prefix), root_text, list(trail), origs[i], src.text, root_class, dict(eq)))
            elif rel in by_null and by_null[rel] not in recursive:
                y = by_null[rel]
                for k, (yh, yb, ysrc, yo) in enumerate(null_clauses[y]):
                    th, tb = rename(yh, yb, next(counter))
                    eq2 = dict(eq)
                    mp: dict = {}
                    ok = True
                    for h, a in zip(th, args):
                        if is_var(h) and h not in mp:
                            mp[h] = a
                        else:
                            ok = unify(mp.get(h, h), a, eq2) and ok
                    if not ok:
                        continue                                   # the callee's head cannot match this literal: no read
                    walk(th, [subst_lit(x, mp) for x in tb], yo, prefix, trail + [(src.text, origs[i])], root_text, root_class, ysrc, eq2)
            if sign == "pos":
                prefix.append(l)

    for x in sorted(kp):
        if x != R and x not in recursive:
            continue
        for hargs, body, c, co in null_clauses[x]:
            th, tb = rename(hargs, body, next(counter))
            walk(th, tb, co, [], [], c.text, list(th[:nclass]) if carries[x] else [], c, {})

    seen_ctx: set = set()
    hit_names: list = []
    reads_objs: list = []
    for rel, pattern, sign, prefix, root_text, trail, ltext, ctext, rclass, eq in reads:
        pattern = res(pattern, eq)
        prefix = [(l[0], l[1], tuple(res(a, eq) for a in l[2])) if l[0] != 'distinct' else l for l in prefix]
        rclass = [res(a, eq) for a in rclass]
        order: dict = {}

        def canon(t):
            if isinstance(t, tuple):
                return tuple(canon(x) for x in t)
            if is_var(t):
                return order.setdefault(t, f"?v{len(order)}")
            return t
        key = (rel, sign, tuple(canon(a) for a in rclass), tuple((l[0], l[1], tuple(canon(a) for a in l[2])) if l[0] != "distinct" else l for l in prefix), tuple(canon(a) for a in pattern))
        if key in seen_ctx:
            continue
        seen_ctx.add(key)
        i = len(hit_names)
        hn = fresh_name(f"{R}_hit_{i}")
        hit_names.append(hn)
        keyvars = [f"?k{j}" for j in range(m)]
        body = []
        if rclass:
            body.append(guard_literal(keyvars, list(rclass)))
        body += [l for l in prefix]
        body.append(("pos", diff, tuple(keyvars) + tuple(pattern)))
        clauses.append(clause_text((hn, tuple(keyvars)), body))
        out_sigs.append(f"(signature {hn}" + "".join(" " + a for a in sig(R)) + ")")
        reads_objs.append(Read(i, hn, pattern, sign, root_text, trail, ltext, ctext))
    hits = fresh_name(f"{R}_hits")
    out_sigs.append(f"(signature {hits}" + "".join(" " + a for a in sig(R)) + ")")
    kvs = [f"?k{j}" for j in range(m)]
    for hn in hit_names:
        clauses.append(clause_text((hits, tuple(kvs)), [("pos", hn, tuple(kvs))]))

    # ---- the answer of R0 at the class of a key
    answer = fresh_name(f"{R}_answer")
    out_sigs.append(f"(signature {answer}" + "".join(" " + a for a in sig(R)) + ")")
    cvs = [f"?c{j}" for j in range(nclass)]
    clauses.append(clause_text((answer, tuple(kvs)), [guard_literal(kvs, cvs), ("pos", null[R], tuple(cvs))]))

    # ---- PART E, THE MASK: a copy of the cone of the test with the hits relation joined into every copy view of the hypothetical relation, so that the bulk evaluation makes the hypothetical board, and what is made from it, for the keys of the hits only.
    # The copy is additive (new names, `_m`): the relations of the description are not changed. Add views are not copied (the hits are made from them). At a key that is a row of the hits the world of the key is intact, so the copy has the rows of the test there.
    copy_views = [v for v, form in shape["views"] if form == "copy"]
    masked_set = set(kp) | {H} | set(copy_views)
    masked = {x: fresh_name(f"{x}_m") for x in sorted(masked_set)}
    for x in sorted(masked_set):
        out_sigs.append(f"(signature {masked[x]}" + "".join(" " + a for a in sig(x)) + ")")
        for c in cl[x]:
            body = []
            if x in copy_views:
                body.append(("pos", hits, tuple(c.head[1][:m])))
            for l in c.body:
                if l[0] == "distinct" or l[1] not in masked_set:
                    body.append(l)
                else:
                    body.append((l[0], masked[l[1]], tuple(l[2])))
            clauses.append(clause_text((masked[x], tuple(c.head[1])), body))
    outside = sorted({c.head[0] for rel, cs in cl.items() for c in cs if rel not in masked_set and any(l[0] != "distinct" and l[1] in masked_set for l in c.body)})

    prop = Proposal(name=f"{R}_null_filter", test=R, places=m, state_relation=B, hypothetical=H, null_test=null[R], answer=answer, hits=hits, diff=diff,
                    signatures=out_sigs, clauses=clauses, reads=reads_objs, guard=lit_text("pos", grel, [f"?{x[0]}{x[1]}" if x[0] != "const" else x[1] for x in gplaces]), guard_relation=grel, classes=nclass, tested=masked[R], masked=sorted(masked.values()), outside_readers=outside, cone=sorted(masked_set), diff_views=diff_views)
    return prop


# ------------------------------------------------------------------------------------------------------------------------------------------ the document (part C) and the statement (part proposer two)

def document(prop: Proposal, game: str, variant: str, standing: "dict | None" = None, evidence: "dict | None" = None) -> dict:
    """The proposal as the document of the form FILTER (`rule_lemma_filter.py`, beside `rule_lemma.py`): the delta, the test and its answer, the hits relation, the reads, the standing PROPOSED unless one is given."""
    return {
        "name": prop.name, "game": game, "variant": variant, "form": "FILTER",
        "says": (f"At a state, a key of the test {prop.test} whose differing rows meet none of the {len(prop.reads)} patterns that the test as the state stands ({prop.null_test}) reads (the keys that are not rows of {prop.hits}) has the answer of {prop.null_test} at the class of its key ({prop.answer}). "
                 f"The null change: the hypothetical relation {prop.hypothetical} is read as the state relation {prop.state_relation}."),
        "left": {"relation": prop.test, "places": prop.places}, "right": {"relation": prop.answer, "places": prop.places},
        "condition": [], "delta": prop.delta, "origin": "MINED",
        "standing": standing or {"is": "PROPOSED", "detail": "made by the proposer proposer one (the null change), not judged"},
        "lean": {"module": "Ideas.NullFilter", "namespace": "Ideas.NullFilter", "position": "the state as a world: inputs to values", "uses": ["Ideas.NullFilter.Test.run_of_null"],
                 "proof": ["an instance of Ideas.NullFilter.Test.run_of_null: the test is R0 read on the state, its asked inputs are contained in the patterns of the reads of the hit relations, the world of a key agrees with the state outside its differing rows"]},
        "filter": {"hits": prop.hits, "tested": prop.tested, "classes": prop.classes,
                   "reads": [{"relation": r.relation, "literal": r.literal, "clause": r.clause, "trail": [list(t) for t in r.trail]} for r in prop.reads]},
        "evidence": evidence or {},
    }


def reads_text(prop: Proposal) -> list:
    """One line a read: where it stands in the description's clauses (a read made through the unfolding of a relation says through which)."""
    out = []
    for r in prop.reads:
        via = "".join(f" [through {t[1]} of {t[0]}]" for t in r.trail)
        out.append(f"read {r.index} ({r.relation}): the literal {r.literal} of the clause {r.clause}{via}")
    return out


def statement(prop: Proposal, counts: "dict | None" = None) -> list:
    """PROPOSER proposer two: the readable statement, read off the explanations: the keys that can change the answer of the test are those that change a row one of its reads makes, one line for each read, in the description's own words; with the counts the judge printed when they are given
    (`counts`: {read relation: (keys it meets a state on average, share of the keys of the hits)})."""
    lines = [f"The test {prop.test} of a key is, for a key that changes none of the rows read below, the answer of {prop.null_test}; the keys that can change the answer are those that change the row read by one of these literals:"]
    for r in prop.reads:
        via = "".join(f" (reached through {t[1]})" for t in r.trail)
        c = (counts or {}).get(r.relation)
        tail = f"  [meets {c[0]:.2f} keys a state; {c[1]:.3f} of the keys of the hits]" if c else ""
        lines.append(f"  - the row read by the literal {r.literal} of the clause {r.clause}{via}{tail}")
    return lines


# ------------------------------------------------------------------------------------------------------------------------------------------ what follows from an explained read (part B, the analysis of the explanations)

def parse_clause_text(text: str) -> "GDL.Clause":
    """A clause as the description writes it, read by the reader (the translator's clause numbers in the module tables are texts)."""
    return GDL.clauses(GDL.parse(text))[0]


def findings_at(clause_text_: str, head_mask: str, order: list, position: int) -> list:
    """What follows, mechanically, from one literal of an explained proof: the literal at `position` of the join (`order`: the places of the clause's literals, in the order written, in the order joined) of a clause, a call of which binds the head places of `head_mask` (1 bound).
    The variables bound when the literal is read are those of the bound head places and of the positive literals joined before it. A list of findings, each a sentence in the description's words:
      ONE ROW      every place of the literal is bound when it is read: one row is tested, its support is that row.
      EXISTENCE    the places the literal leaves free hold variables used nowhere else in the clause: it is read only to know whether a row exists.
      BOUND LATER  a variable the literal leaves free is bound by a literal the join reads later; that literal could be joined first (a conjunction does not depend on the order of its literals), and the first literal would then be read with that place bound; the sentence says what the literal joined first would itself leave free.
      SCAN         nothing else binds a place the literal leaves free: the read is a scan for them.
    No game, no rule of thumb: the clause and the order decide."""
    c = parse_clause_text(clause_text_)
    lit = c.body[order[position]]
    if lit[0] == "distinct":
        return []
    sign, rel, args = lit
    bound: set = set()
    for j, a in enumerate(c.head[1]):
        if j < len(head_mask) and head_mask[j] == "1":
            vars_of(a, bound)
    for k in order[:position]:
        l = c.body[k]
        if l[0] == "pos":
            bound |= lit_vars(l)
    free = [j for j, a in enumerate(args) if is_var(a) and a not in bound]
    head_vars: set = set()
    for a in c.head[1]:
        vars_of(a, head_vars)
    if not free:
        return [f"ONE ROW: every place of {lit_text(sign, rel, args)} is bound when it is read, so one row is tested and its support is that row"]
    fv = {args[j] for j in free}
    used_elsewhere: set = set()
    for j, l in enumerate(c.body):
        if j != order[position]:
            used_elsewhere |= lit_vars(l)
    used_elsewhere |= head_vars
    out = []
    if not (fv & used_elsewhere):
        return [f"EXISTENCE: the places {lit_text(sign, rel, args)} leaves free hold variables used nowhere else in the clause: it is read only to know whether a row exists"]
    for k in order[position + 1:]:
        l = c.body[k]
        if l[0] != "pos":
            continue
        got = lit_vars(l) & fv
        if got:
            others = lit_vars(l) - got - bound
            out.append(f"BOUND LATER: {lit_text(sign, rel, args)} leaves {sorted(got)} free; they are bound by {lit_text(l[0], l[1], l[2])}, which the join reads later. A conjunction does not depend on the order of its literals: joined first, that literal would leave "
                       + (f"{sorted(others)} free" if others else "nothing free") + f" and {lit_text(sign, rel, args)} would be read with {sorted(got)} bound")
    return out or [f"SCAN: the places {lit_text(sign, rel, args)} leaves free ({sorted(fv)}) are bound by this literal alone"]
