"""
Agents for TextWorld-Lite.

* RandomAgent            - uniform over admissible commands
* HeuristicAgent         - hand-written priority rules (reference, not learned)
* TabularQ               - Q-learning on the exact (observation string, command) pair
* DRRN                   - Deep Reinforcement Relevance Network (He et al., 2016),
                           implemented in pure NumPy with manual backprop + Adam.

DRRN: state text and each command text are encoded by SEPARATE bag-of-features
encoders  h = tanh(W * mean(E[feats]) + b),  and  Q(s, a) = h_s . h_a.
Features are either whole words ("word") or hashed character trigrams ("ngram");
the latter gives sub-word sharing and a chance to generalise to unseen words.
"""
import re
import zlib
import random
from collections import deque, defaultdict
import numpy as np
import scipy.sparse as sp

TOKEN_RE = re.compile(r"[a-z]+")


# ------------------------------------------------------------------ simple agents
class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs, cmds, explore=False):
        return self.rng.choice(cmds)


class HeuristicAgent:
    """take > unlock > random 'go' (never 'look'). Memoryless, so it can wander."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs, cmds, explore=False):
        for prefix in ("take ", "unlock "):
            c = [x for x in cmds if x.startswith(prefix)]
            if c:
                return self.rng.choice(c)
        c = [x for x in cmds if x.startswith("go ")]
        return self.rng.choice(c)


class GreedyAgent:
    """One-step greedy: simulates every admissible command on a COPY of the game (cheating: it has the
    simulator) and picks the one with the best immediate reward. Myopic upper-ish reference for 'greedy'."""
    needs_env = True
    def __init__(self, seed=0):
        self.rng = random.Random(seed); self.env = None
    def bind(self, env):
        self.env = env
    def act(self, obs, cmds, explore=False):
        import copy
        best, bc = -1e9, []
        for c in cmds:
            e = copy.deepcopy(self.env)
            r = e.step(c)[1]
            if r > best + 1e-9: best, bc = r, [c]
            elif abs(r - best) <= 1e-9: bc.append(c)
        return self.rng.choice(bc)


# --------------------------------------------------------------------- tabular Q
class TabularQ:
    def __init__(self, seed=0, alpha=0.2, gamma=0.95):
        self.rng = random.Random(seed)
        self.q = defaultdict(float)
        self.alpha, self.gamma = alpha, gamma
        self.eps = 1.0

    def act(self, obs, cmds, explore=True):
        if explore and self.rng.random() < self.eps:
            return self.rng.choice(cmds)
        vals = [self.q[(obs, c)] for c in cmds]
        m = max(vals)
        return self.rng.choice([c for c, v in zip(cmds, vals) if v == m])

    def observe(self, s, a, r, s2, cmds2, done):
        tgt = r if done else r + self.gamma * max(self.q[(s2, c)] for c in cmds2)
        self.q[(s, a)] += self.alpha * (tgt - self.q[(s, a)])


# ------------------------------------------------------------------------- DRRN
class Featurizer:
    def __init__(self, mode, vocab_texts=None, n_buckets=4096):
        self.mode, self.cache = mode, {}
        if mode == "word":
            words = sorted({w for t in vocab_texts for w in TOKEN_RE.findall(t.lower())})
            self.vocab = {w: i + 2 for i, w in enumerate(words)}   # 0 = pad, 1 = unk
            self.size = len(self.vocab) + 2
        else:
            self.n_buckets = n_buckets
            self.size = n_buckets + 2

    def __call__(self, text):
        if text in self.cache:
            return self.cache[text]
        words = TOKEN_RE.findall(text.lower())
        if self.mode == "word":
            ids = [self.vocab.get(w, 1) for w in words]
        else:
            ids = []
            for w in words:
                p = f"<{w}>"
                grams = [p[i:i + 3] for i in range(max(1, len(p) - 2))]
                ids += [zlib.crc32(g.encode()) % self.n_buckets + 2 for g in grams]
        ids = np.array(ids or [1], dtype=np.int64)
        self.cache[text] = ids
        return ids


class Encoder:
    """h = tanh( mean_over_features(E[ids]) @ W + b ), computed with a sparse averaging matrix M
    so that  X = M @ E  and  dE = M^T @ dX  (no padding, no slow scatter-add)."""
    def __init__(self, vocab, d, h, rng):
        self.vocab = vocab
        self.p = {"E": rng.normal(0, 0.1, (vocab, d)),
                  "W": rng.normal(0, 1 / np.sqrt(d), (d, h)),
                  "b": np.zeros(h)}

    def forward(self, id_list, p=None):
        p = p or self.p
        lens = np.fromiter((len(x) for x in id_list), dtype=np.int64, count=len(id_list))
        indices = np.concatenate(id_list)
        indptr = np.concatenate([[0], np.cumsum(lens)])
        data = np.repeat(1.0 / lens, lens)
        X = np.add.reduceat(p["E"][indices] * data[:, None], indptr[:-1], axis=0)
        H = np.tanh(X @ p["W"] + p["b"])
        return H, (indices, indptr, data, X, H)

    def backward(self, dH, cache, grads):
        indices, indptr, data, X, H = cache
        M = sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, self.vocab))
        dZ = dH * (1 - H ** 2)
        grads["W"] += X.T @ dZ
        grads["b"] += dZ.sum(0)
        grads["E"] += M.T @ (dZ @ self.p["W"].T)


class Adam:
    """Adam; embedding matrices (keys ending in 'E') use the row-sparse 'lazy' variant:
    only rows that received gradient in this batch have their moments / weights updated."""
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, clip=5.0):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in params:
            g = grads[k]
            if k.endswith("E"):
                rows = np.nonzero(np.any(g != 0, axis=1))[0]
                g = np.clip(g[rows], -clip, clip)
                m = self.m[k][rows] = self.b1 * self.m[k][rows] + (1 - self.b1) * g
                v = self.v[k][rows] = self.b2 * self.v[k][rows] + (1 - self.b2) * g * g
                params[k][rows] -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
            else:
                g = np.clip(g, -clip, clip)
                self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
                self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
                params[k] -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


class DRRN:
    """DRRN with an optional exact-match 'copy / cross-attention' head (match=True).

    Q(s,a) = h_s . h_a  +  match(s,a)
    match(s,a): every argument token of the command (all tokens but the verb) is matched, by exact
      string identity, against all positions of the state text -- this works for ANY word, including
      words never seen in training.  Each matched position j contributes its *context vector*
      c_j = mean of E_c[neighbours within +-2] (neighbours are mostly known function words such as
      'locked', 'door', 'see', 'carrying'), so the model learns e.g. "the word that also sits next to
      'locked ... door' is the key to take".  phi = mean of contexts of matched positions;
      match = tanh(W_m phi + b_m) . V[verb]   (a verb-specific read-out).
    """
    def __init__(self, feat_mode, vocab_texts, seed=0, d=32, h=64, lr=5e-4, gamma=0.95,
                 buffer=20000, batch=32, target_sync=200, match=False, d_c=16, h_m=16, tok_drop=0.0):
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.feat = Featurizer(feat_mode, vocab_texts)
        self.match, self.tok_drop = match, tok_drop
        self._acache = {}
        self.enc_s = Encoder(self.feat.size, d, h, self.rng)
        self.enc_a = Encoder(self.feat.size, d, h, self.rng)
        self.params = {f"s_{k}": v for k, v in self.enc_s.p.items()}
        self.params.update({f"a_{k}": v for k, v in self.enc_a.p.items()})
        if match:
            assert feat_mode == "word", "match head needs word tokens"
            V = self.feat.size
            self.params["c_E"] = self.rng.normal(0, 0.1, (V, d_c))
            self.params["m_W"] = self.rng.normal(0, 1 / np.sqrt(d_c), (d_c, h_m))
            self.params["m_b"] = np.zeros(h_m)
            self.params["v_E"] = self.rng.normal(0, 0.1, (V, h_m))
            self._scache = {}
        self.target = {k: v.copy() for k, v in self.params.items()}
        self.opt = Adam(self.params, lr=lr)
        self.gamma, self.batch, self.target_sync = gamma, batch, target_sync
        self.buf = deque(maxlen=buffer)
        self.eps, self.updates = 1.0, 0

    # ----------------------------------------------------------------- base encoders
    def _enc(self, enc, prefix, texts, params, drop=0.0):
        p = {k: params[f"{prefix}_{k}"] for k in ("E", "W", "b")}
        ids = [self.feat(t) for t in texts]
        if drop > 0:        # token dropout (training only): replace tokens by <unk>, keep a command's verb
            out = []
            for t, x in zip(texts, ids):
                x = x.copy()
                m = self.rng.random(len(x)) < drop
                if prefix == "a":
                    m[0] = False
                x[m] = 1
                out.append(x)
            ids = out
        return enc.forward(ids, p)

    # ------------------------------------------------------------------- match head
    def _state_struct(self, text):
        if text in self._scache:
            return self._scache[text]
        toks = TOKEN_RE.findall(text.lower())
        ids = np.array([self.feat.vocab.get(w, 1) for w in toks], dtype=np.int64)
        n = len(toks)
        rows, cols, vals = [], [], []
        for j in range(n):
            nb = [k for k in range(max(0, j - 2), min(n, j + 3)) if k != j]
            for k in nb:
                rows.append(j); cols.append(ids[k]); vals.append(1.0 / len(nb))
        Mctx = sp.csr_matrix((vals, (rows, cols)), shape=(n, self.feat.size))
        pos = {}
        for j, w in enumerate(toks):
            pos.setdefault(w, []).append(j)
        if len(self._scache) > 30000:
            self._scache.clear()
        self._scache[text] = (Mctx, pos, n)
        return self._scache[text]

    def _parse_action(self, ac):
        r = self._acache.get(ac)
        if r is None:
            toks = TOKEN_RE.findall(ac.lower())
            r = (self.feat.vocab.get(toks[0], 1) if toks else 1, toks[1:])
            self._acache[ac] = r
        return r

    def _match_forward(self, S, A, params):
        B = len(S)
        phi = np.zeros((B, params["c_E"].shape[1]))
        verbs = np.ones(B, dtype=np.int64)
        qs = []
        C, structs = {}, {}                           # per unique state text
        for i, (st, ac) in enumerate(zip(S, A)):
            if st not in structs:
                structs[st] = self._state_struct(st)
            Mctx, pos, n = structs[st]
            verbs[i], args = self._parse_action(ac)
            q = np.zeros(n)
            for w in args:
                for j in pos.get(w, ()):
                    q[j] += 1.0 / len(args)
            if q.any():
                if st not in C:
                    C[st] = Mctx @ params["c_E"]
                phi[i] = q @ C[st]
            qs.append(q)
        T = np.tanh(phi @ params["m_W"] + params["m_b"])
        Vv = params["v_E"][verbs]
        return (T * Vv).sum(1), (phi, T, Vv, verbs, qs, S)

    def _match_backward(self, dq, cache, grads):
        phi, T, Vv, verbs, qs, S = cache
        dT = dq[:, None] * Vv
        np.add.at(grads["v_E"], verbs, dq[:, None] * T)
        dZ = dT * (1 - T ** 2)
        grads["m_W"] += phi.T @ dZ
        grads["m_b"] += dZ.sum(0)
        dphi = dZ @ self.params["m_W"].T
        Dc = {}                                       # accumulate q (x) dphi per unique state
        for i, (st, q) in enumerate(zip(S, qs)):
            if q.any():
                Dc[st] = Dc.get(st, 0) + q[:, None] * dphi[i][None, :]
        for st, D in Dc.items():
            grads["c_E"] += self._state_struct(st)[0].T @ D

    # ------------------------------------------------------------------ acting / learning
    def q_values(self, obs, cmds, params=None):
        params = params or self.params
        Hs, _ = self._enc(self.enc_s, "s", [obs], params)
        Ha, _ = self._enc(self.enc_a, "a", cmds, params)
        q = Ha @ Hs[0]
        if self.match:
            q = q + self._match_forward([obs] * len(cmds), cmds, params)[0]
        return q

    def act(self, obs, cmds, explore=True):
        if explore and self.pyrng.random() < self.eps:
            return self.pyrng.choice(cmds)
        return cmds[int(np.argmax(self.q_values(obs, cmds)))]

    def observe(self, s, a, r, s2, cmds2, done):
        self.buf.append((s, a, r, s2, cmds2, done))

    def update(self):
        if len(self.buf) < self.batch:
            return None
        idx = self.rng.integers(len(self.buf), size=self.batch)
        B = [self.buf[i] for i in idx]
        S = [b[0] for b in B]; A = [b[1] for b in B]
        R = np.array([b[2] for b in B])

        nd = [i for i, b in enumerate(B) if not b[5]]
        y = R.copy()
        if nd:
            ns = [B[i][3] for i in nd]
            flat_a, seg, offs = [], [], []
            for j, i in enumerate(nd):
                offs.append(len(flat_a)); flat_a += B[i][4]; seg += [j] * len(B[i][4])
            Hs2, _ = self._enc(self.enc_s, "s", ns, self.target)
            Ha2, _ = self._enc(self.enc_a, "a", flat_a, self.target)
            q2 = (Hs2[seg] * Ha2).sum(1)
            if self.match:
                q2 = q2 + self._match_forward([ns[j] for j in seg], flat_a, self.target)[0]
            y[nd] += self.gamma * np.maximum.reduceat(q2, offs)

        Hs, cs = self._enc(self.enc_s, "s", S, self.params, drop=self.tok_drop)
        Ha, ca = self._enc(self.enc_a, "a", A, self.params, drop=self.tok_drop)
        q = (Hs * Ha).sum(1)
        if self.match:
            qm, mcache = self._match_forward(S, A, self.params)
            q = q + qm
        err = q - y
        dq = np.clip(err, -1, 1) / self.batch
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        gs = {k[2:]: grads[k] for k in grads if k.startswith("s_")}
        ga = {k[2:]: grads[k] for k in grads if k.startswith("a_")}
        self.enc_s.backward(dq[:, None] * Ha, cs, gs)
        self.enc_a.backward(dq[:, None] * Hs, ca, ga)
        if self.match:
            self._match_backward(dq, mcache, grads)
        self.opt.step(self.params, grads)
        self.updates += 1
        if self.updates % self.target_sync == 0:
            for k in self.params:
                self.target[k][...] = self.params[k]
        return float((err ** 2).mean())

    def save(self, path):
        np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in self.params.items()})
