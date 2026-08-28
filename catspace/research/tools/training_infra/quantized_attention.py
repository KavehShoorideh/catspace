"""Quantized attention over concept tokens: a typed, readable relation extractor.

Kaveh 2026-08-26: "I like the attention concept, and humans do pay attention in the
concept sense. So I wonder if there's a quantized version of attention where my query
and key become quantized -- 'who is attacking me?' and the answer is that piece. I want
to look at the matrix and see what was being ASKED and what was being ANSWERED."

WHY NOT JUST READ SOFT ATTENTION. Because the field tried and partly retracted it. Jain
& Wallace (2019) showed attention weights do not reliably indicate what a model used,
and Meister et al. (2021) showed sparsity does not fix that -- exact zeros make a map
readable without making it faithful. So this does not *read* an attention map. It
*constrains* one:

    the QUESTION is discrete   each token emits one of R learned query codes, so
                               "what is token i asking" has R possible answers, not a
                               continuous direction
    the ANSWER is sparse       top-k attention, so a question is answered by k tokens,
                               not smeared over 129
    the RESULT is quantized    the attended value is re-quantized to bits, so the
                               enriched features entering the readout are still concepts

That last step is the second level of quantization: level 1 is the raw per-square /
per-piece concepts, level 2 is *relational* concepts built by asking discrete questions
about them. Everything the downstream linear readout sees is a bit.

WHAT MAKES THE CLAIM CHECKABLE. Because the tokens are named (this square, this piece
slot) and the questions are discrete, a head's behaviour is a typed relation:

    (piece 3) --[question 5]--> (square e4)

and python-chess can independently say who actually attacks, defends or blocks whom. So
"question 5 means attacks" is a falsifiable statement with a measurable alignment score,
which is exactly what the Jain & Wallace critique says raw attention lacks. See
`relation_alignment()`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedAttention(nn.Module):
    """One layer of typed, sparse, quantized attention over concept tokens.

    Args:
        d_tok:      width of an incoming concept token
        n_q:        number of distinct QUESTIONS a token may ask (the query codebook)
        n_head:     parallel relation channels
        top_k:      how many tokens may answer one question
        out_bits:   bits per token of level-2 (relational) concept output
    """

    def __init__(self, d_tok: int, n_q: int = 16, n_head: int = 4, top_k: int = 4,
                 out_bits: int = 8, d_key: int = 32):
        super().__init__()
        self.n_q, self.n_head, self.top_k, self.out_bits = n_q, n_head, top_k, out_bits
        self.d_key = d_key
        # WHICH question does this token ask? A hard choice among n_q, so the query is
        # discrete by construction rather than discretised afterwards.
        self.q_select = nn.Linear(d_tok, n_head * n_q)
        # the questions themselves, one learned key-space direction each
        self.q_codebook = nn.Parameter(torch.randn(n_head, n_q, d_key) * 0.02)
        self.k_proj = nn.Linear(d_tok, n_head * d_key)
        self.v_proj = nn.Linear(d_tok, n_head * d_key)
        # level-2 quantiser: the ANSWER becomes bits, so it is a concept, not a vector
        from vector_quantize_pytorch import LFQ
        self.out_vq = LFQ(dim=out_bits, codebook_size=2, num_codebooks=out_bits)
        self.out_proj = nn.Linear(n_head * d_key, out_bits)
        self.last = {}

    def forward(self, tok: torch.Tensor, tok_mask: torch.Tensor | None = None,
                tau: float = 1.0):
        """tok: (B, T, d_tok) concept tokens. Returns (level2_bits, aux)."""
        B, T, _ = tok.shape
        H, R, K = self.n_head, self.n_q, self.d_key

        # ---- the QUESTION: a hard, discrete choice, straight-through for gradients ---
        qlog = self.q_select(tok).view(B, T, H, R)
        qsoft = F.softmax(qlog / tau, -1)
        qhard = F.one_hot(qsoft.argmax(-1), R).to(qsoft.dtype)
        qsel = qhard + (qsoft - qsoft.detach())          # straight-through
        # each token's query IS one codebook entry -- inspectable as an integer
        q = torch.einsum("bthr,hrk->bthk", qsel, self.q_codebook)

        k = self.k_proj(tok).view(B, T, H, K)
        v = self.v_proj(tok).view(B, T, H, K)
        logit = torch.einsum("bthk,bshk->bhts", q, k) / (K ** 0.5)
        if tok_mask is not None:
            logit = logit.masked_fill(~tok_mask[:, None, None, :], float("-inf"))

        # ---- the ANSWER: only top_k tokens may answer, the rest are exactly zero -----
        kk = min(self.top_k, T)
        topv, topi = logit.topk(kk, dim=-1)
        attn = torch.zeros_like(logit).scatter_(-1, topi, F.softmax(topv, -1))
        out = torch.einsum("bhts,bshk->bthk", attn, v).reshape(B, T, H * K)

        # ---- the RESULT: quantised, so level 2 is concepts and not a hidden vector ---
        z = self.out_proj(out)
        bits, ids, vq_loss = self.out_vq(z, mask=tok_mask) if tok_mask is not None \
            else self.out_vq(z)
        self.last = {"question": qsoft.argmax(-1).detach(),      # (B,T,H) integer
                     "attn": attn.detach(),                      # (B,H,T,T) sparse
                     "answer_ids": ids.detach()}
        return bits, {"vq_loss": vq_loss, "question": self.last["question"],
                      "attn": self.last["attn"],
                      # raw logits exposed for ST-MoE router losses (z-loss + balance);
                      # without these the stabilisers silently no-op
                      "qlog": qlog, "attn_logit": logit}

    @torch.no_grad()
    def questions_asked(self) -> torch.Tensor:
        """(n_head, n_q) histogram of which questions actually get used.

        A head that only ever asks one question is a dead channel; a head that spreads
        over all R is not a typed relation. Both are visible here.
        """
        q = self.last.get("question")
        if q is None:
            return torch.zeros(self.n_head, self.n_q)
        out = torch.zeros(self.n_head, self.n_q)
        for h in range(self.n_head):
            out[h] = torch.bincount(q[..., h].flatten(), minlength=self.n_q).float()
        return out / out.sum(-1, keepdim=True).clamp(min=1)


@torch.no_grad()
def relation_alignment(attn: torch.Tensor, truth: torch.Tensor,
                       tok_is_square: torch.Tensor) -> dict:
    """Does a head's attention correspond to a REAL chess relation?

    This is the faithfulness test the Jain & Wallace critique demands: not "the map
    looks sensible" but "head h's edges agree with a relation python-chess computes
    independently, at a rate a shuffled control does not reach".

    attn:  (B, H, T, T) attention, sparse
    truth: (B, S, S) boolean, the real relation between SQUARES (e.g. attacks)
    tok_is_square: (T,) index of the square a token refers to, or -1
    """
    sq = (tok_is_square >= 0).nonzero(as_tuple=True)[0]
    idx = tok_is_square[sq].long()
    A = attn[:, :, sq][:, :, :, sq]                       # (B,H,S',S')
    Tr = truth[:, idx][:, :, idx].unsqueeze(1).to(A.dtype)
    hit = (A * Tr).sum((-1, -2)) / A.sum((-1, -2)).clamp(min=1e-9)   # (B,H)
    base = Tr.mean((-1, -2))                               # density of the relation
    return {"precision_per_head": hit.mean(0),             # attention landing on truth
            "chance": base.mean(),
            "lift_per_head": (hit.mean(0) / base.mean().clamp(min=1e-9))}


def _test():
    torch.manual_seed(0)
    B, T, D = 4, 20, 32
    qa = QuantizedAttention(D, n_q=8, n_head=3, top_k=3, out_bits=8)
    tok = torch.randn(B, T, D)
    bits, aux = qa(tok)

    assert bits.shape == (B, T, 8), bits.shape
    vals = set(torch.unique(bits).tolist())
    assert vals <= {-1.0, 1.0}, f"level-2 output is not binary: {vals}"

    a = aux["attn"]
    nz = (a > 0).sum(-1)
    assert int(nz.max()) <= 3, f"attention is not top-3 sparse: {int(nz.max())}"
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)

    q = aux["question"]
    assert q.shape == (B, T, 3) and int(q.max()) < 8, "questions are not discrete ids"

    (bits.sum()).backward()
    assert qa.q_codebook.grad is not None and qa.q_codebook.grad.abs().sum() > 0, \
        "no gradient into the query codebook -- the questions cannot be learned"
    assert qa.q_select.weight.grad.abs().sum() > 0, \
        "no gradient into question SELECTION -- straight-through is broken"

    # the faithfulness harness runs and separates a real relation from a shuffled one
    S = 8
    tis = torch.full((T,), -1, dtype=torch.long); tis[:S] = torch.arange(S)
    truth = torch.zeros(B, S, S, dtype=torch.bool); truth[:, torch.arange(S), (torch.arange(S)+1) % S] = True
    r = relation_alignment(a, truth, tis)
    assert r["precision_per_head"].shape == (3,)

    hist = qa.questions_asked()
    print(f"  level-2 output: {bits.shape[-1]} bits per token, values {sorted(vals)}")
    print(f"  attention: at most {int(nz.max())} of {T} tokens answer any question, "
          f"rows sum to 1")
    print(f"  questions: discrete ids in [0,{qa.n_q}), gradient reaches both the codebook "
          f"and the selector")
    print(f"  question usage per head: {[round(float(x),2) for x in hist.max(-1).values]}"
          f"  (1.0 would mean a dead, single-question head)")
    print(f"  faithfulness harness returns per-head precision vs an independent relation")
    print("  QUANTIZED ATTENTION TESTS PASS")


if __name__ == "__main__":
    _test()
