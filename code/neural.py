"""
neural.py — neural Forward-Backward embeddings with a strict generalization protocol.

The claim to test: the model generalizes to UNSEEN STATES within a familiar
concept family (it has seen KRK boxing; it has not seen THIS exact position).

Protocol:
  - hold out 15% of W states uniformly at random
  - training pairs (s, g) are sampled from random-play games with BOTH s and g
    outside the holdout (held-out states never enter the network in any role)
  - evaluation happens AT the held-out states: reach-ranking quality and
    engine performance starting from them

Architecture: two MLPs (numpy, manual backprop, Adam)
  F-net: one-hot(wk)+one-hot(wr)+one-hot(bk) [75] -> 256 -> 256 -> d
  B-net: same input space + 2 absorbing flags [77] -> 256 -> 256 -> d
Loss: InfoNCE with in-batch negatives on geometric-horizon future pairs,
which trains F(s)·B(g) toward log M(g|s)/rho0(g) up to constants — the
contrastive estimate of the discounted successor measure (the cone).
"""
import numpy as np

def one_hot_state(st, nsq=25):
    v = np.zeros(3 * nsq + 2, dtype=np.float32)
    v[st[0]] = 1; v[nsq + st[1]] = 1; v[2 * nsq + st[2]] = 1
    return v

def absorbing_vec(kind, nsq=25):  # kind: 0=MATE, 1=DRAW
    v = np.zeros(3 * nsq + 2, dtype=np.float32)
    v[3 * nsq + kind] = 1
    return v

class MLP:
    def __init__(self, din, dh, dout, seed):
        r = np.random.default_rng(seed)
        s1, s2, s3 = (2/din)**.5, (2/dh)**.5, (2/dh)**.5
        self.p = dict(
            W1=r.standard_normal((din, dh)).astype(np.float32)*s1, b1=np.zeros(dh, np.float32),
            W2=r.standard_normal((dh, dh)).astype(np.float32)*s2,  b2=np.zeros(dh, np.float32),
            W3=r.standard_normal((dh, dout)).astype(np.float32)*s3, b3=np.zeros(dout, np.float32))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        p = self.p
        z1 = X @ p['W1'] + p['b1']; a1 = np.maximum(z1, 0)
        z2 = a1 @ p['W2'] + p['b2']; a2 = np.maximum(z2, 0)
        out = a2 @ p['W3'] + p['b3']
        self.cache = (X, z1, a1, z2, a2)
        return out

    def backward(self, dout):
        X, z1, a1, z2, a2 = self.cache
        p, g = self.p, {}
        g['W3'] = a2.T @ dout; g['b3'] = dout.sum(0)
        da2 = dout @ p['W3'].T; dz2 = da2 * (z2 > 0)
        g['W2'] = a1.T @ dz2; g['b2'] = dz2.sum(0)
        da1 = dz2 @ p['W2'].T; dz1 = da1 * (z1 > 0)
        g['W1'] = X.T @ dz1; g['b1'] = dz1.sum(0)
        return g

    def adam(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1*self.m[k] + (1-b1)*g[k]
            self.v[k] = b2*self.v[k] + (1-b2)*g[k]**2
            mh = self.m[k]/(1-b1**self.t); vh = self.v[k]/(1-b2**self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

class NeuralFB:
    def __init__(self, d=32, dh=256, seed=0, tau=0.1):
        self.F = MLP(77, dh, d, seed)
        self.B = MLP(77, dh, d, seed + 1)
        self.tau = tau
        self.d = d

    def train_step(self, Xs, Xg, lr):
        """InfoNCE with in-batch negatives: rows = anchors F(s), cols = B(g)."""
        f = self.F.forward(Xs)                       # (n, d)
        b = self.B.forward(Xg)                       # (n, d)
        logits = (f @ b.T) / self.tau                # (n, n)
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits); probs = e / e.sum(1, keepdims=True)
        n = len(Xs)
        loss = -np.log(probs[np.arange(n), np.arange(n)] + 1e-12).mean()
        dlog = (probs - np.eye(n, dtype=np.float32)) / n / self.tau
        df = dlog @ b                                # (n, d)
        db = dlog.T @ f
        self.F.adam(self.F.backward(df), lr)
        self.B.adam(self.B.backward(db), lr)
        return loss

    def embed_F(self, X): return self.F.forward(X)
    def embed_B(self, X): return self.B.forward(X)

def build_pairs(ch, games_transitions, holdout_mask, gamma, rng):
    """From raw game episodes, make (s, future g) index pairs with geometric
    horizon; drop any pair touching holdout. Episodes are lists of state
    indices ending optionally at an absorbing index."""
    pairs = []
    for ep in games_transitions:
        L = len(ep)
        for i in range(L - 1):
            s = ep[i]
            if s >= ch.nW or holdout_mask[s]: continue
            k = 1 + rng.geometric(1 - gamma)
            j = min(i + k, L - 1)
            g = ep[j]
            if g < ch.nW and holdout_mask[g]: continue
            pairs.append((s, g))
    return pairs

def sample_episodes(ch, n_games, max_plies=200, seed=1):
    r = np.random.default_rng(seed)
    eps = []
    starts = r.integers(0, ch.nW, size=n_games)
    for g in range(n_games):
        s = int(starts[g]); ep = [s]
        for _ in range(max_plies):
            mv = r.integers(0, len(ch.moves[s]))
            outs = ch.moves[s][mv]
            nxt = int(outs[r.integers(0, len(outs))])
            ep.append(nxt)
            if nxt >= ch.nW: break
            s = nxt
        eps.append(ep)
    return eps
