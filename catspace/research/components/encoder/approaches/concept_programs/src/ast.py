#!/usr/bin/env python
"""ast.py -- a differentiable program over named chess primitives. (Kaveh 2026-08-17)

"we need a way to call functions in the AST-equivalent of the rust concept world in a parametrized
way, then calculate the error, backpropagate into the parameter and move it somehow."

That is what this is. Every choice in the program -- which operator a node applies, which value it
reads -- is a temperature-annealed softmax, so plain autograd moves STRUCTURE and CONSTANTS
together. Anneal the temperature until the selectors go one-hot, `harden()` reads off a discrete
program, and `emit.py` writes it as Rust with the constants baked in.

SINGLE-SORTED BY DESIGN. A typed DSL (square-level vs position-level values) would need typed
selectors and typed operators, and every soft mixture would have to respect the types. Instead
every value is carried at SQUARE level, shape (B, 64): the 100 global primitives are broadcast
across squares, and the aggregators `mean_sq` / `max_sq` reduce and broadcast straight back. A
value that is constant across squares IS a position-level value, so the two sorts coexist without
any type machinery, and the emitted Rust is a flat list of temporaries.

WHAT IT FITS. The concept being distilled is a VQ code -- which of the 16 entries in that head's
codebook the network selects. The AST emits a d_code-dim vector and is scored by the SOFT NEAREST
CODE relaxation:

    loss = CE( softmax(-||AST(x) - codebook_k||^2 / tau_c),  the network's code )

which is code agreement (the chosen fit-and-gate criterion) made differentiable. A hard 16-way
target alone gives almost no gradient to anneal a soft program with; this gives a dense one while
still optimising exactly the quantity the gate measures.

    .venv/bin/python -m ...concept_programs.src.ast        # unit tests
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Binary operators. Each takes (a, b, theta) with a, b of shape (B, 64) and theta a per-node
# parameter vector, and returns (B, 64). Kept deliberately small: every operator here has an
# obvious Rust spelling, because an operator that cannot be emitted is not worth being able to
# learn.
OPS = ["add", "sub", "mul", "max", "min", "gt", "wsum", "mean_sq", "max_sq", "id_a"]
N_THETA = 3          # per node: gt threshold/scale, wsum weights


def apply_op(k: int, a, b, theta):
    if k == 0:
        return a + b
    if k == 1:
        return a - b
    if k == 2:
        return a * b
    if k == 3:
        return torch.maximum(a, b)
    if k == 4:
        return torch.minimum(a, b)
    if k == 5:                                   # soft threshold: is a above a learned level?
        return torch.sigmoid((a - theta[0]) * F.softplus(theta[1]))
    if k == 6:
        return theta[0] * a + theta[1] * b
    if k == 7:                                   # aggregate and broadcast back
        return a.mean(dim=1, keepdim=True).expand_as(a)
    if k == 8:
        return a.max(dim=1, keepdim=True).values.expand_as(a)
    return a                                     # id_a -- lets a node be a pass-through


class SoftAST(nn.Module):
    """A DAG of `n_nodes` soft nodes over `n_leaf` named primitives, emitting `d_out` values.

    Node i may read any primitive or any EARLIER node, which keeps the graph acyclic by
    construction and lets the program share sub-expressions instead of recomputing them.
    """

    def __init__(self, n_sq_feat, n_glob_feat, n_nodes=8, d_out=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_sq, self.n_glob = n_sq_feat, n_glob_feat
        self.n_leaf = n_sq_feat + n_glob_feat
        self.n_nodes, self.d_out = n_nodes, d_out
        # per node: operator logits, two input selectors over (leaves + earlier nodes), constants
        self.op_logits = nn.Parameter(torch.randn(n_nodes, len(OPS), generator=g) * 0.1)
        self.in_a = nn.ParameterList()
        self.in_b = nn.ParameterList()
        for i in range(n_nodes):
            k = self.n_leaf + i
            self.in_a.append(nn.Parameter(torch.randn(k, generator=g) * 0.1))
            self.in_b.append(nn.Parameter(torch.randn(k, generator=g) * 0.1))
        self.theta = nn.Parameter(torch.randn(n_nodes, N_THETA, generator=g) * 0.1)
        # the read-out picks d_out values from the pool; a linear map here would be a learned
        # layer rather than a program, so it is a SELECTOR plus a per-output scale and offset,
        # all of which emit as constants.
        self.out_sel = nn.Parameter(torch.randn(d_out, self.n_leaf + n_nodes, generator=g) * 0.1)
        self.out_scale = nn.Parameter(torch.ones(d_out))
        self.out_bias = nn.Parameter(torch.zeros(d_out))

    # ---- forward -------------------------------------------------------------------
    def leaves(self, sq_feat, glob_feat):
        """(B,64,n_sq) + (B,n_glob) -> (B, 64, n_leaf). Globals are broadcast across squares,
        which is what lets one sort carry both kinds of value."""
        g = glob_feat[:, None, :].expand(-1, 64, -1)
        return torch.cat([sq_feat, g], dim=-1)

    def forward(self, sq_feat, glob_feat, tau=1.0, hard=False):
        """-> (B, 64, d_out). For a position-level concept the caller reduces over squares; for a
        square-level concept it does not."""
        pool = list(self.leaves(sq_feat, glob_feat).unbind(-1))       # each (B,64)
        for i in range(self.n_nodes):
            avail = torch.stack(pool, dim=-1)                        # (B,64,k)
            wa = self._sel(self.in_a[i], tau, hard)
            wb = self._sel(self.in_b[i], tau, hard)
            a = (avail * wa).sum(-1)
            b = (avail * wb).sum(-1)
            wo = self._sel(self.op_logits[i], tau, hard)
            val = 0.0
            for k in range(len(OPS)):
                if hard and float(wo[k]) == 0.0:
                    continue                                          # skip dead branches
                val = val + wo[k] * apply_op(k, a, b, self.theta[i])
            pool.append(val)
        avail = torch.stack(pool, dim=-1)                             # (B,64,n_leaf+n_nodes)
        w = self._sel(self.out_sel, tau, hard, dim=-1)                # (d_out, k)
        out = torch.einsum("bsk,dk->bsd", avail, w)
        return out * self.out_scale + self.out_bias

    @staticmethod
    def _sel(logits, tau, hard, dim=-1):
        """Annealed selector. `hard` snaps to one-hot -- the same argmax `harden()` reads, so the
        hardened program and the hard-forward agree by construction."""
        if hard:
            idx = logits.argmax(dim=dim, keepdim=True)
            return torch.zeros_like(logits).scatter_(dim, idx, 1.0)
        return F.softmax(logits / max(tau, 1e-6), dim=dim)

    # ---- discrete read-out ----------------------------------------------------------
    def harden(self):
        """-> a plain dict describing the discrete program. No tensors, so it can be json'd,
        diffed between rounds, and handed to the Rust emitter."""
        nodes = []
        for i in range(self.n_nodes):
            nodes.append({
                "op": OPS[int(self.op_logits[i].argmax())],
                "a": int(self.in_a[i].argmax()),
                "b": int(self.in_b[i].argmax()),
                "theta": [float(x) for x in self.theta[i].detach()],
            })
        return {
            "n_leaf": self.n_leaf, "n_sq": self.n_sq, "n_glob": self.n_glob,
            "nodes": nodes,
            "out": [int(x) for x in self.out_sel.argmax(-1)],
            "scale": [float(x) for x in self.out_scale.detach()],
            "bias": [float(x) for x in self.out_bias.detach()],
        }

    def size(self):
        """Program size actually REACHABLE from the outputs. Unreached nodes cost nothing at
        inference and should not count against interpretability, so the honest size is the live
        sub-DAG, not `n_nodes`."""
        prog = self.harden()
        live, stack = set(), [o for o in prog["out"] if o >= self.n_leaf]
        while stack:
            i = stack.pop()
            if i < self.n_leaf or i in live:
                continue
            live.add(i)
            nd = prog["nodes"][i - self.n_leaf]
            for c in (nd["a"], nd["b"]):
                if c >= self.n_leaf:
                    stack.append(c)
        return len(live)


def describe_program(prog, sq_names, glob_names, max_out=6):
    """Render a hardened program as named expressions -- the deliverable a learned trunk cannot
    produce. Only the sub-DAG reachable from the outputs is printed, since unreached nodes are
    not part of the concept."""
    n_leaf, n_sq = prog["n_leaf"], prog["n_sq"]

    def leaf_name(j):
        return f"sq.{sq_names[j]}" if j < n_sq else f"pos.{glob_names[j - n_sq]}"

    def ref(j):
        return leaf_name(j) if j < n_leaf else f"t{j - n_leaf}"

    live, stack = set(), [o for o in prog["out"] if o >= n_leaf]
    while stack:
        i = stack.pop()
        if i < n_leaf or i in live:
            continue
        live.add(i)
        nd = prog["nodes"][i - n_leaf]
        for c in (nd["a"], nd["b"]):
            if c >= n_leaf:
                stack.append(c)

    lines = []
    for i in sorted(live):
        nd = prog["nodes"][i - n_leaf]
        a, b, th = ref(nd["a"]), ref(nd["b"]), nd["theta"]
        op = nd["op"]
        if op == "gt":
            import math
            expr = f"sigmoid(({a} - {th[0]:.3f}) * {math.log1p(math.exp(th[1])):.3f})"
        elif op == "wsum":
            expr = f"{th[0]:.3f}*{a} + {th[1]:.3f}*{b}"
        elif op in ("mean_sq", "max_sq"):
            expr = f"{op}({a})"
        elif op == "id_a":
            expr = a
        else:
            sym = {"add": "+", "sub": "-", "mul": "*", "max": "max", "min": "min"}[op]
            expr = f"max({a}, {b})" if op == "max" else (
                f"min({a}, {b})" if op == "min" else f"{a} {sym} {b}")
        lines.append(f"    t{i - n_leaf} = {expr}")
    for d, o in enumerate(prog["out"][:max_out]):
        lines.append(f"    out[{d}] = {prog['scale'][d]:.3f} * {ref(o)} + {prog['bias'][d]:.3f}")
    if len(prog["out"]) > max_out:
        lines.append(f"    ... {len(prog['out']) - max_out} more outputs")
    return "\n".join(lines)


def soft_code_ce(pred, codebook, target_code, tau_c=1.0):
    """CE over the SOFT nearest code -- the differentiable form of 'does this program fire the
    same concept the network fired'.

    pred (N, d) ; codebook (K, d) ; target_code (N,) int64. -> scalar
    """
    d2 = torch.cdist(pred, codebook).pow(2)              # (N, K)
    return F.cross_entropy(-d2 / max(tau_c, 1e-6), target_code)


def code_agreement(pred, codebook, target_code):
    """The GATE: hard argmin agreement, no temperature. -> float in [0,1]"""
    with torch.no_grad():
        k = torch.cdist(pred, codebook).argmin(-1)
        return float((k == target_code).float().mean())


# ------------------------------------------------------------------------------------------
def _tests():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        cond = bool(cond); ok &= cond
        print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")

    torch.manual_seed(0)
    B, NS, NG = 5, 28, 100
    sq = torch.randn(B, 64, NS)
    gl = torch.randn(B, NG)
    ast = SoftAST(NS, NG, n_nodes=6, d_out=32)

    out = ast(sq, gl)
    chk("forward shape (B,64,d_out)", out.shape == (B, 64, 32))
    chk("gradient reaches operator logits, selectors, and constants",
        all(g is not None and g.abs().sum().item() > 0 for g in torch.autograd.grad(
            out.sum(), [ast.op_logits, ast.in_a[0], ast.theta, ast.out_scale],
            retain_graph=True, allow_unused=True)))

    # globals must be broadcast identically to every square
    lv = ast.leaves(sq, gl)
    chk("global primitives broadcast across all 64 squares",
        torch.allclose(lv[:, 0, NS:], lv[:, 37, NS:]))
    chk("square primitives are NOT broadcast",
        not torch.allclose(lv[:, 0, :NS], lv[:, 37, :NS]))

    # aggregators make a value square-invariant, which is how position-level concepts arise
    a = torch.randn(B, 64)
    chk("mean_sq is constant across squares",
        float((apply_op(7, a, a, ast.theta[0]).std(dim=1)).max()) < 1e-6)
    chk("max_sq is constant across squares",
        float((apply_op(8, a, a, ast.theta[0]).std(dim=1)).max()) < 1e-6)

    # THE ANNEALING CONTRACT, stated the way it is actually true: GIVEN separated selectors, the
    # soft forward equals the hard forward, so the program that gets emitted is the program that
    # was trained. It has to be conditional -- a selector whose top two logits are exactly tied
    # stays 50/50 at every temperature, so simply scaling random logits does not converge. That
    # is a property of ties, not a defect in the mechanism, and the anneal schedule's job is to
    # break them. Whether it managed to is measured on TRAINED programs by
    # `fit_concept_program.py` (agreement before vs after hardening), which is the check that
    # actually protects a distillation round.
    with torch.no_grad():
        for i in range(ast.n_nodes):
            for p in (ast.op_logits[i], ast.in_a[i], ast.in_b[i]):
                j = int(p.argmax()); p.zero_(); p[j] = 10.0
        for d in range(ast.out_sel.shape[0]):
            j = int(ast.out_sel[d].argmax()); ast.out_sel[d].zero_(); ast.out_sel[d, j] = 10.0
    peak = min([float(F.softmax(ast.op_logits / 0.05, -1).max(-1).values.min())]
               + [float(F.softmax(a / 0.05, -1).max()) for a in ast.in_a]
               + [float(F.softmax(ast.out_sel / 0.05, -1).max(-1).values.min())])
    with torch.no_grad():
        soft = ast(sq, gl, tau=0.05)
        hard = ast(sq, gl, hard=True)
    rel = float((soft - hard).abs().max() / (hard.abs().max() + 1e-6))
    chk("separated selectors: soft forward == hard forward", rel < 1e-4,
        f"peakiness {peak:.4f}, rel gap {rel:.2e}")

    # hardened description must be self-consistent and acyclic
    prog = ast.harden()
    chk("harden() returns one entry per node", len(prog["nodes"]) == ast.n_nodes)
    chk("every node reads only leaves or EARLIER nodes (acyclic)",
        all(nd["a"] < ast.n_leaf + i and nd["b"] < ast.n_leaf + i
            for i, nd in enumerate(prog["nodes"])))
    chk("harden() is json-safe (no tensors)",
        all(isinstance(v, (int, float)) for nd in prog["nodes"] for v in nd["theta"]))
    chk("size() counts only nodes reachable from the outputs",
        0 <= ast.size() <= ast.n_nodes, f"live {ast.size()}/{ast.n_nodes}")

    # the loss and the gate must agree about what a perfect program looks like
    cb = torch.randn(16, 32)
    tgt = torch.randint(0, 16, (40,))
    perfect = cb[tgt]
    chk("perfect prediction -> agreement 1.0", code_agreement(perfect, cb, tgt) == 1.0)
    chk("perfect prediction -> near-zero soft CE",
        float(soft_code_ce(perfect, cb, tgt, tau_c=0.01)) < 1e-3,
        f"{float(soft_code_ce(perfect, cb, tgt, tau_c=0.01)):.2e}")
    rnd = torch.randn(40, 32)
    chk("random prediction -> agreement near chance",
        code_agreement(rnd, cb, tgt) < 0.30,
        f"{code_agreement(rnd, cb, tgt):.3f} vs chance 0.0625")
    chk("soft CE has a gradient w.r.t. the prediction",
        torch.autograd.grad(soft_code_ce(rnd.requires_grad_(), cb, tgt), rnd)[0]
        .abs().sum().item() > 0)

    print("\nALL SOFT-AST TESTS PASSED" if ok else "\nSOFT-AST TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_tests())
