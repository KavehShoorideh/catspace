#!/usr/bin/env python
"""koopman_ops.py -- per-move-code linear rollforward operators on a lifted concept state.

The gko1 operator module (docs/GKO1_PLAN.md sec 2). The design bet, stated plainly: after
lifting the concept state past a fixed learned nonlinearity, MOVES ACT LINEARLY --

    z_aug  = [z ; phi(z)]        phi = low-rank learned bilinear (pairwise interaction
                                 features, so the evolution of concept a can depend on
                                 concept b being present)
    z_aug' = (A + B_c) z_aug     A shared, init exactly identity; B_c = U_c V_c^T
                                 low-rank per MOVE-CODE c with V init 0, so a fresh
                                 module is exactly the identity map (U1) while the
                                 zero factor still receives gradient (the LoRA init
                                 argument: dL/dV = U^T delta z^T != 0).

Linearity is the point, not a concession: composition over a 6-ply line is a matrix
product (U3), spectra are inspectable and clampable (U2), the persistence rulebook
tabulates argmax-code survival per operator offline, and sibling moves get DIFFERENT
operators by construction -- the architectural answer to the twice-verified dyn
sibling-blindness (beam children cosine 1.00000: an MLP dyn trained on played moves
collapses its children; indexed B_c cannot).

Rollouts NEVER re-lift. z_aug rolls forward by operator action alone, and
`lift_consistency` measures how far the carried phi-part drifts from phi of the
carried state-part. U1b bounds that drift on a synthetic linear system with monomial
observables, where an exact linear operator on the lifted space provably exists
(x' = Gx  =>  x x^T rolls by G on both sides) -- if the machinery cannot hold
consistency THERE, it is broken, not merely approximate.

One-ply loss detail (U13 lineage, the circularity guard): the state-part target is
the real child state (stop-grad); the phi-part target is phi(child) with BOTH input
and output detached, so the lift cannot shrink itself to make its own targets easy.
A collapsed phi would anyway cost the operators the interaction features they need
for the state-part anchor -- the two parts hold each other up.

Self-test gates U1, U1b, U2, U3, U4 (incl. kooplearn EDMD cross-check as the
reference implementation) and EXITS NONZERO on any failure:

    .venv/bin/python -m catspace.research.tools.training_infra.koopman_ops --selftest
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionLift(nn.Module):
    """z -> [z ; (Uz) * (Vz)] : each phi dim a rank-1 bilinear form over the whole
    concept state -- cross-head pairwise products included, jointly trained."""

    def __init__(self, d_state: int, d_phi: int = 128):
        super().__init__()
        self.d_state, self.d_phi = d_state, d_phi
        self.u = nn.Linear(d_state, d_phi, bias=False)
        self.v = nn.Linear(d_state, d_phi, bias=False)

    def phi(self, z):
        return self.u(z) * self.v(z)

    def forward(self, z):
        return torch.cat([z, self.phi(z)], -1)

    @property
    def d_aug(self):
        return self.d_state + self.d_phi


class KoopmanOps(nn.Module):
    """z_aug' = (A + U_c V_c^T) z_aug, batched over rows (z' = z M^T)."""

    def __init__(self, d_aug: int, n_codes: int = 64, rank: int = 16,
                 rho_max: float = 1.02):
        super().__init__()
        self.d_aug, self.n_codes, self.rank, self.rho_max = d_aug, n_codes, rank, rho_max
        self.A = nn.Parameter(torch.eye(d_aug))
        self.U = nn.Parameter(torch.randn(n_codes, d_aug, rank) / math.sqrt(d_aug))
        self.V = nn.Parameter(torch.zeros(n_codes, rank, d_aug))

    def matrix(self, codes):
        """(k,) code ids -> (k, d_aug, d_aug) full operators A + U_c V_c."""
        return self.A.unsqueeze(0) + self.U[codes] @ self.V[codes]

    def forward(self, z_aug, codes=None):
        """codes None = null move (B == 0): pure shared dynamics."""
        out = z_aug @ self.A.T
        if codes is None:
            return out
        t = torch.einsum("bd,brd->br", z_aug, self.V[codes])     # V z
        return out + torch.einsum("br,bdr->bd", t, self.U[codes])  # + U (V z)

    def spectral_penalty(self, codes, k: int = 8, iters: int = 48, keep: int = 24):
        """max(0, rho - rho_max)^2 over <=k sampled distinct codes in the batch.
        MATRIX-FREE power iteration (Mv = Av + U(Vv)): materializing k full d_aug^2
        operators was a 1.4GB-per-step allocation at the 512-concept scale. Same
        Gelfand estimate, near-zero extra memory."""
        uc = torch.unique(codes)
        if uc.numel() > k:
            uc = uc[torch.randperm(uc.numel(), device=uc.device)[:k]]
        v = torch.randn(uc.numel(), self.d_aug, device=self.A.device, dtype=self.A.dtype)
        v = v / v.norm(dim=-1, keepdim=True)
        Uc, Vc = self.U[uc], self.V[uc]
        logs = []
        for _ in range(iters):
            w = v @ self.A.T + torch.einsum("br,bdr->bd",
                                            torch.einsum("bd,brd->br", v, Vc), Uc)
            r = w.norm(dim=-1, keepdim=True).clamp(min=1e-30)
            logs.append(r.log().squeeze(-1))
            v = w / r
        rho = torch.stack(logs[-keep:], -1).mean(-1).exp()
        return F.relu(rho - self.rho_max).pow(2).mean(), rho.detach()

    @torch.no_grad()
    def init_from_ridge(self, Ms):
        """Warm-start from closed-form per-code fits: A = mean(M_c), B_c = SVD-r of
        the residual. Exact when rank >= true residual rank (checked in U4)."""
        Ms = Ms.to(self.A.dtype)
        self.A.copy_(Ms.mean(0))
        R = Ms - self.A.unsqueeze(0)
        Us, Ss, Vhs = torch.linalg.svd(R.double())
        r = self.rank
        self.U.copy_((Us[:, :, :r] * Ss[:, None, :r].sqrt()).to(self.U.dtype))
        self.V.copy_((Ss[:, :r, None].sqrt() * Vhs[:, :r, :]).to(self.V.dtype))


def spectral_radius(M, iters: int = 96, keep: int = 48):
    """Gelfand power estimate of rho(M) for (..., d, d): geometric mean of per-step
    growth over the tail. Converges to the dominant |eigenvalue| for generic init,
    complex pairs included (per-step norms oscillate; their geometric mean does not).
    Differentiable through M."""
    d = M.shape[-1]
    v = torch.randn(*M.shape[:-2], d, 1, device=M.device, dtype=M.dtype)
    v = v / v.norm(dim=-2, keepdim=True)
    logs = []
    for _ in range(iters):
        w = M @ v
        r = w.norm(dim=-2, keepdim=True).clamp(min=1e-30)
        logs.append(r.log().reshape(*M.shape[:-2]))
        v = w / r
    tail = torch.stack(logs[-keep:], -1)
    return tail.mean(-1).exp()


@torch.no_grad()
def ridge_fit(Z, Zp, codes, n_codes: int, lam: float = 1e-4):
    """Closed-form per-code least squares, Zp ~= Z M_c^T -> (n_codes, d, d).
    Double precision internally; codes with no rows get identity."""
    d = Z.shape[-1]
    out = torch.eye(d, dtype=Z.dtype).repeat(n_codes, 1, 1)
    I = torch.eye(d, dtype=torch.float64)
    for c in range(n_codes):
        m = codes == c
        if not bool(m.any()):
            continue
        Zc, Zpc = Z[m].double(), Zp[m].double()
        out[c] = torch.linalg.solve(Zc.T @ Zc + lam * I, Zc.T @ Zpc).T.to(Z.dtype)
    return out


def rollout(ops: KoopmanOps, z_aug0, code_seq):
    """(B, d_aug) x (B, T) -> (B, T, d_aug). Pure operator composition, no re-lifting:
    the whole Koopman claim lives in this function staying linear."""
    outs, za = [], z_aug0
    for t in range(code_seq.shape[1]):
        za = ops(za, code_seq[:, t])
        outs.append(za)
    return torch.stack(outs, 1)


def lift_consistency(lift: InteractionLift, z_aug):
    """Relative drift between the carried phi-part and phi(carried state-part).
    The U1b quantity; tracked per-horizon during training."""
    z, ph = z_aug[..., : lift.d_state], z_aug[..., lift.d_state:]
    return (lift.phi(z) - ph).norm(dim=-1) / ph.norm(dim=-1).clamp(min=1e-8)


def one_ply_loss(lift: InteractionLift, ops: KoopmanOps, z, codes, z_child,
                 w_phi: float = 0.25):
    """State-part anchored on the real child (stop-grad); phi-part on phi(child)
    detached on BOTH sides (see module docstring). Returns (loss, pred_z_aug)."""
    pred = ops(lift(z), codes)
    d = lift.d_state
    ls = F.mse_loss(pred[..., :d], z_child.detach())
    lp = F.mse_loss(pred[..., d:], lift.phi(z_child.detach()).detach())
    return ls + w_phi * lp, pred


class BasisOps(nn.Module):
    """gcf5 operator: K basis matrices mixed by weights from a BARE move-identity
    embedding (Kaveh 2026-08-23: position-free lookup; context lives in the STATE).
    K(m) = A + sum_k w_k(m) U_k V_k^T, w = softmax(W e_m). Same linear-in-state
    algebra as KoopmanOps (U1/U3 apply); smooth conditioning, no quantization loss;
    mixture weights = soft move concepts (discrete readout via argmax on demand)."""

    def __init__(self, d_aug: int, n_basis: int = 64, rank: int = 16,
                 d_move: int = 64, rho_max: float = 1.02, d_phi: int = 0):
        super().__init__()
        self.d_aug, self.n_basis, self.rank, self.rho_max = d_aug, n_basis, rank, rho_max
        # STATE-CONDITIONED MIXING (Kaveh 2026-08-24: "the state absolutely must be in
        # the operator... post interaction concepts and the move"). Without this,
        # move_weights ignores z entirely, so K(m) is a FIXED matrix per move identity
        # -- the same transform in the opening and the endgame. But what Nf3 DOES
        # depends on the position (capture? block? defend?), so a position-independent
        # map cannot express it, and the rolled path emitted a near-constant across
        # siblings (4pct of true spread, ordering 0.00).
        # phi is the lift's POST-INTERACTION tail of z_aug, i.e. concepts after the
        # interaction transformer -- not the raw pre-interaction heads.
        # Zero-init so a fresh/resumed model starts EXACTLY at the old behaviour.
        self.d_phi = int(d_phi)
        if self.d_phi:
            self.state_proj = nn.Linear(self.d_phi, d_move)
            nn.init.zeros_(self.state_proj.weight)
            nn.init.zeros_(self.state_proj.bias)
        self.A = nn.Parameter(torch.eye(d_aug))
        self.U = nn.Parameter(torch.randn(n_basis, d_aug, rank) / math.sqrt(d_aug))
        self.V = nn.Parameter(torch.zeros(n_basis, rank, d_aug))
        # bare identity: from-square x to-square x promo-class lookup
        self.emb_from = nn.Embedding(64, d_move)
        self.emb_to = nn.Embedding(64, d_move)
        self.emb_promo = nn.Embedding(5, d_move)
        self.mix = nn.Linear(d_move, n_basis)

    def move_weights(self, frm, to, promo, z_aug=None):
        e = self.emb_from(frm) + self.emb_to(to) + self.emb_promo(promo)
        if self.d_phi and z_aug is not None:
            phi = z_aug[..., -self.d_phi:]
            if phi.dim() == 1:
                phi = phi[None]
            e = e + self.state_proj(phi)
        return torch.softmax(self.mix(e), -1)             # (B, K)

    def forward(self, z_aug, frm, to, promo):
        w = self.move_weights(frm, to, promo, z_aug)      # (B, K)
        out = z_aug @ self.A.T
        t = torch.einsum("bd,krd->bkr", z_aug, self.V)    # V_k z
        t = t * w[..., None]
        return out + torch.einsum("bkr,kdr->bd", t, self.U)

    def roll_shared(self, z, frm, to, promo):
        """Roll ONE state through MANY moves -- the search's exact access pattern.

        forward() recomputes `z @ A.T` and `V z` once per row, but in a search node
        every row is the SAME z, so both are constant across the batch. Hoisting
        them out drops the per-move cost from (d*d + 2*K*r*d) to a single
        (n, K*r) @ (K*r, d) GEMM: ~4x fewer MACs and one BLAS call instead of two
        einsums. Mathematically identical to forward() on an expanded z -- gated
        by U1e.

        z: (d_aug,) or (1, d_aug). Returns (n_moves, d_aug).
        """
        z = z.reshape(-1)
        # phi is shared across the batch here too, so state conditioning costs one
        # extra (1, d_phi) @ (d_phi, d_move) -- the shared-z speedup is preserved
        w = self.move_weights(frm, to, promo, z)               # (n, K)
        az = z @ self.A.T                                      # (d,)   once
        vz = torch.einsum("d,krd->kr", z, self.V)              # (K, r) once
        t = (w[..., None] * vz[None]).reshape(len(w), -1)      # (n, K*r)
        u = self.U.permute(0, 2, 1).reshape(-1, self.d_aug)    # (K*r, d)
        return az[None] + t @ u

    def matrix(self, frm, to, promo, z_aug=None):
        w = self.move_weights(frm, to, promo, z_aug)
        B = torch.einsum("bk,kdr,kre->bde", w, self.U, self.V)
        return self.A.unsqueeze(0) + B

    def spectral_penalty(self, frm, to, promo, k: int = 8, iters: int = 48,
                         keep: int = 24, z_aug=None):
        n = min(k, len(frm))
        idx = torch.randperm(len(frm), device=frm.device)[:n]
        zs = None if z_aug is None else (z_aug[idx] if z_aug.dim() > 1 else z_aug)
        w = self.move_weights(frm[idx], to[idx], promo[idx], zs)
        v = torch.randn(n, self.d_aug, device=self.A.device, dtype=self.A.dtype)
        v = v / v.norm(dim=-1, keepdim=True)
        logs = []
        for _ in range(iters):
            t = torch.einsum("bd,krd->bkr", v, self.V) * w[..., None]
            wv = v @ self.A.T + torch.einsum("bkr,kdr->bd", t, self.U)
            r = wv.norm(dim=-1, keepdim=True).clamp(min=1e-30)
            logs.append(r.log().squeeze(-1))
            v = wv / r
        rho = torch.stack(logs[-keep:], -1).mean(-1).exp()
        return F.relu(rho - self.rho_max).pow(2).mean(), rho.detach()

    def identity_prior(self, u_pen: float = 0.1):
        """Kaveh's I+delta: the delta earns its magnitude everywhere, ongoing.

        `u_pen` scales the penalty on U. CAUTION (2026-08-23): U carries the
        ENTIRE move-dependent part of K(m) = A + sum_k w_k(m) U_k V_k^T. Measured
        over 47 real positions, that term had shrunk to ~0.5-1pct of the shared A
        term -- ~99pct of the operator was identical whichever move was played,
        so no downstream head could separate siblings. Penalising U buys slowness
        at the direct cost of move-conditioning; lower it when a loss (--w-sib) is
        actively asking the operator to distinguish moves, or the two cancel."""
        return ((self.A - torch.eye(self.d_aug, device=self.A.device)).pow(2).mean()
                + self.U.pow(2).mean() * u_pen)


# ---------------------------------------------------------------------------------
# self-test: the U1-U4 gates. Every check prints its measured number; any failure
# exits nonzero (gates must gate).
# ---------------------------------------------------------------------------------

class _MonomialLift(nn.Module):
    """Exact quadratic observables z -> [z ; z_i z_j (i<=j)] for the U1b system."""

    def __init__(self, d):
        super().__init__()
        self.d_state = d
        idx = torch.triu_indices(d, d)
        self.register_buffer("i0", idx[0])
        self.register_buffer("i1", idx[1])

    def phi(self, z):
        return z[..., self.i0] * z[..., self.i1]

    def forward(self, z):
        return torch.cat([z, self.phi(z)], -1)


def _selftest():
    import sys
    torch.manual_seed(0)
    fails = []

    def gate(name, ok, detail):
        print(f"  {name:<6} {'PASS' if ok else 'FAIL':<5} {detail}")
        if not ok:
            fails.append(name)

    # U1: fresh module is exactly the identity on any input, for any code ------------
    lift = InteractionLift(512, 128)
    ops = KoopmanOps(640, n_codes=64, rank=16)
    za = torch.randn(256, 640)
    codes = torch.randint(0, 64, (256,))
    err = (ops(za, codes) - za).abs().max()
    gate("U1", err.item() == 0.0, f"fresh-op identity max|dz| = {err.item():.1e} (exact 0 required)")

    # U1c (extra): the LoRA-init gradient argument, verified not assumed -------------
    z, zc = torch.randn(64, 512), torch.randn(64, 512)
    loss, _ = one_ply_loss(lift, ops, z, codes[:64], zc)
    loss.backward()
    gv, gu = ops.V.grad.abs().max().item(), ops.U.grad.abs().max().item()
    ga, gl = ops.A.grad.abs().max().item(), lift.u.weight.grad.abs().max().item()
    gate("U1c", gv > 0 and ga > 0 and gl > 0 and gu == 0.0,
         f"grads at init: V {gv:.1e} A {ga:.1e} lift {gl:.1e} | U {gu:.1e} (exact 0: B=UV with V=0)")
    ops.zero_grad(); lift.zero_grad()

    # U1b: monomial lift on x' = Gx -- exact linear operator exists on z_aug;
    # ridge must find it and a 6-step NO-RELIFT rollout must hold consistency -------
    d = 6
    G = torch.randn(d, d, dtype=torch.float64)
    G = 0.90 * G / spectral_radius(G.unsqueeze(0), 256, 128)[0]
    ml = _MonomialLift(d).double()
    X = torch.randn(5000, d, dtype=torch.float64)
    Za, Zap = ml(X), ml(X @ G.T)
    M = ridge_fit(Za, Zap, torch.zeros(len(Za), dtype=torch.long), 1, lam=1e-8)
    d_aug = Za.shape[-1]
    op1 = KoopmanOps(d_aug, n_codes=1, rank=d_aug).double()
    op1.init_from_ridge(M)
    x = torch.randn(128, d, dtype=torch.float64)
    za6 = rollout(op1, ml(x), torch.zeros(128, 6, dtype=torch.long))
    drift = lift_consistency(ml, za6[:, -1]).max().item()
    xt = x
    for _ in range(6):
        xt = xt @ G.T
    serr = ((za6[:, -1, :d] - xt).norm() / xt.norm()).item()
    gate("U1b", drift < 1e-3 and serr < 1e-3,
         f"6-step no-relift: phi drift {drift:.1e}, state err {serr:.1e} (tol 1e-3)")

    # U1d: BasisOps identity at init + compose == matmul (the U1/U3 forms) ----------
    bo = BasisOps(96, n_basis=8, rank=8, d_move=16)
    zb = torch.randn(16, 96)
    frm = torch.randint(0, 64, (16,)); to = torch.randint(0, 64, (16,))
    pr = torch.zeros(16, dtype=torch.long)
    e1d = (bo(zb, frm, to, pr) - zb).abs().max().item()
    with torch.no_grad():
        bo.V.normal_(0, 0.1); bo.A.add_(torch.randn(96, 96) * 0.02)
    frm2 = torch.randint(0, 64, (16,)); to2 = torch.randint(0, 64, (16,))
    two_b = bo(bo(zb, frm, to, pr), frm2, to2, pr)
    M = torch.einsum("bde,bef->bdf", bo.matrix(frm2, to2, pr), bo.matrix(frm, to, pr))
    comp_b = torch.einsum("bd,bed->be", zb, M)
    rel_b = ((two_b - comp_b).norm() / comp_b.norm()).item()
    gate("U1d", e1d == 0.0 and rel_b < 1e-4,
         f"BasisOps: init identity {e1d:.1e} (exact) | compose rel err {rel_b:.1e}")

    # U2: Gelfand estimator vs torch.linalg.eigvals on random 64x64, 5 seeds --------
    worst = 0.0
    for s in range(5):
        g = torch.Generator().manual_seed(s)
        M2 = torch.randn(64, 64, generator=g, dtype=torch.float64)
        true = torch.linalg.eigvals(M2).abs().max().item()
        est = spectral_radius(M2.unsqueeze(0), iters=256, keep=128)[0].item()
        worst = max(worst, abs(est - true) / true)
    gate("U2", worst < 0.05, f"rho estimate vs eigvals, 5 seeds: worst rel err {worst:.3f} (tol 0.05)")

    # U2b: matrix-free penalty rho == dense-matrix rho on perturbed ops -------------
    ops2b = KoopmanOps(96, n_codes=8, rank=8)
    with torch.no_grad():
        ops2b.V.normal_(0, 0.2); ops2b.A.add_(torch.randn(96, 96) * 0.05)
    cds2b = torch.arange(8)
    torch.manual_seed(5)
    _, rho_mf = ops2b.spectral_penalty(cds2b, k=8, iters=256, keep=128)
    rho_dn = spectral_radius(ops2b.matrix(cds2b), iters=256, keep=128)
    d2b = float((rho_mf - rho_dn).abs().max() / rho_dn.abs().max())
    gate("U2b", d2b < 0.02, f"matrix-free vs dense rho rel err {d2b:.1e} (tol 0.02)")

    # U3: two-op application == matmul-composed operator ----------------------------
    with torch.no_grad():
        ops.V.normal_(0, 0.05); ops.A.add_(torch.randn(640, 640) * 0.01)
    c1, c2 = codes, torch.randint(0, 64, (256,))
    two = ops(ops(za, c1), c2)
    M21 = ops.matrix(c2) @ ops.matrix(c1)
    comp = torch.einsum("bd,bed->be", za, M21)
    rel = ((two - comp).norm() / comp.norm()).item()
    gate("U3", rel < 1e-4, f"apply-apply vs composed-matrix rel err {rel:.1e} (tol 1e-4)")

    # U4: ridge recovers per-code generators; kooplearn EDMD agrees on the same data
    d4, nc = 12, 4
    Gs = []
    for c in range(nc):
        Gc = torch.randn(d4, d4, dtype=torch.float64)
        Gs.append(0.9 * Gc / spectral_radius(Gc.unsqueeze(0), 256, 128)[0])
    Z = torch.randn(8000, d4, dtype=torch.float64)
    cds = torch.randint(0, nc, (8000,))
    Zp = torch.stack([z @ Gs[int(c)].T for z, c in zip(Z, cds)]) + 1e-4 * torch.randn(8000, d4, dtype=torch.float64)
    Mhat = ridge_fit(Z, Zp, cds, nc, lam=1e-6)
    gerr = max((Mhat[c].double() - Gs[c]).norm().item() / Gs[c].norm().item() for c in range(nc))
    opsr = KoopmanOps(d4, n_codes=nc, rank=d4).double()
    opsr.init_from_ridge(Mhat.double())
    rerr = max((opsr.matrix(torch.tensor([c]))[0] - Mhat[c].double()).norm().item()
               / Mhat[c].double().norm().item() for c in range(nc))
    try:
        import numpy as np
        from kooplearn.linear_model import Ridge as KLRidge
        # stationary-scale trajectory (noise 0.35 keeps states O(1): a decaying one is
        # ill-conditioned across rows and amplifies any convention mismatch); kooplearn
        # regularizes C_X = X^T X / n with alpha (read from _linalg.covariance +
        # add_diagonal_), so the Gram-space equivalent is lam = n * alpha -- matched
        # exactly, the two solve the same normal equations.
        x0 = torch.randn(d4, dtype=torch.float64)
        traj = [x0]
        for _ in range(4000):
            traj.append(traj[-1] @ Gs[0].T + 0.35 * torch.randn(d4, dtype=torch.float64))
        traj = torch.stack(traj)
        alpha = 1e-6
        kl = KLRidge(n_components=d4, reduced_rank=False, alpha=alpha).fit(traj.numpy())
        Xte = traj[:-1]
        mine = (Xte @ ridge_fit(Xte, traj[1:], torch.zeros(len(Xte), dtype=torch.long),
                                1, lam=alpha * len(Xte))[0].T.double())
        klp = torch.from_numpy(np.asarray(kl.predict(Xte.numpy(), n_steps=1))).reshape(mine.shape)
        xerr = ((klp - mine).norm() / mine.norm()).item()
    except Exception as e:
        xerr, gerr = float("inf"), gerr
        print(f"         kooplearn cross-check raised: {type(e).__name__}: {e}")
    gate("U4", gerr < 1e-2 and rerr < 1e-6 and xerr < 1e-2,
         f"ridge-vs-generator {gerr:.1e} | init_from_ridge reconstr {rerr:.1e} | kooplearn-vs-ridge {xerr:.1e}")

    print(f"\n  {'ALL GATES PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        print("nothing to do without --selftest")


class TransformerLift(nn.Module):
    """z -> [z ; phi(z)] where phi = attention over the CONCEPT TOKENS (Kaveh
    2026-08-21: interactions -- pin attends to check -- are computed ONCE, written
    onto the tokens 'as a little bit of a rotation', and the operator applies AFTER,
    carrying the interaction signal through rollouts with no re-encode; 'otherwise I
    am no better than MuZero'). Same interface as InteractionLift, so the operator
    algebra, losses, and the consistency meter apply unchanged.

    Token layout for the gcf4h-family state (d_state = heads*d_code + 64*d_sq + d_hist):
    one token per global VQ head, one per square code, one for the edge-histogram
    block; typed + positioned, n_layers of standard attention, all output tokens
    concatenated and projected to d_phi."""

    def __init__(self, d_state: int, g_heads: int = 32, d_code: int = 16,
                 d_sq: int = 32, d_phi: int = 256, d_tok: int = 64,
                 n_layers: int = 2, n_attn: int = 4, n_piece: int = 0,
                 d_piece: int = 4):
        super().__init__()
        self.d_state, self.d_phi = d_state, d_phi
        self.g_heads, self.d_code, self.d_sq = g_heads, d_code, d_sq
        self.n_piece, self.d_piece = n_piece, d_piece
        self.d_hist = d_state - g_heads * d_code - 64 * d_sq - n_piece * d_piece
        assert self.d_hist >= 0, (d_state, g_heads, d_code, d_sq, n_piece)
        self.n_tok = g_heads + 64 + n_piece + (1 if self.d_hist else 0)
        if n_piece:
            self.p_piece = nn.Linear(d_piece, d_tok)
        self.p_head = nn.Linear(d_code, d_tok)
        self.p_sq = nn.Linear(d_sq, d_tok)
        self.p_hist = nn.Linear(self.d_hist, d_tok) if self.d_hist else None
        self.tok_emb = nn.Parameter(torch.randn(self.n_tok, d_tok) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_tok, nhead=n_attn,
                                           dim_feedforward=2 * d_tok, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(self.n_tok * d_tok, d_phi)
        self.out_pool = nn.Linear(d_tok, d_phi)
        self.p_mv = nn.Linear(32, d_tok)
        self.mv_emb_tok = nn.Parameter(torch.randn(1, 1, d_tok) * 0.02)

    def _tokens(self, z, mv_q=None):
        B = z.shape[0]
        h = self.g_heads * self.d_code
        s = h + 64 * self.d_sq
        toks = [self.p_head(z[:, :h].reshape(B, self.g_heads, self.d_code)),
                self.p_sq(z[:, h:s].reshape(B, 64, self.d_sq))]
        if self.n_piece:
            pe = s + self.n_piece * self.d_piece
            toks.append(self.p_piece(z[:, s:pe].reshape(B, self.n_piece, self.d_piece)))
            s = pe
        if self.p_hist is not None:
            toks.append(self.p_hist(z[:, s:])[:, None, :])
        out = torch.cat(toks, 1) + self.tok_emb
        if mv_q is not None and self.p_mv is not None:
            # MOVE CONCEPTS AT THE TABLE (Kaveh 2026-08-21 late): the position's
            # available-move tokens join the interaction conversation ("the check is
            # available" participates in pre-operator interactions). Rolled states
            # carry the result; they cannot re-present moves (no re-encode), so the
            # consistency check runs move-blind -- approximate by design, noted.
            out = torch.cat([out, self.p_mv(mv_q) + self.mv_emb_tok], 1)
        return out

    def phi(self, z, mv_q=None):
        # force the unfused MATH attention path: the fused sdpa BACKWARD wedges the
        # autograd worker on MPS (main thread parks in ReadyQueue::pop forever --
        # wedge-taxonomy variant 5, 2026-08-21); math = matmuls + softmax, MPS-safe
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            ctx = sdpa_kernel([SDPBackend.MATH])
        except Exception:
            import contextlib
            ctx = contextlib.nullcontext()
        with ctx:
            t = self.enc(self._tokens(z, mv_q))
        # pool: mean over tokens (token count varies with mv presence), then project
        return self.out_pool(t.mean(1)) + self.out(
            t[:, :self.n_tok].reshape(z.shape[0], -1))

    def forward(self, z, mv_q=None):
        return torch.cat([z, self.phi(z, mv_q)], -1)

    @property
    def d_aug(self):
        return self.d_state + self.d_phi
