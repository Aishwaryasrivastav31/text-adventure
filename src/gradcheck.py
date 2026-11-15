"""Numerical gradient check of hand-written backprop, base DRRN and match head (python src/gradcheck.py)."""
import numpy as np, sys
sys.path.insert(0, "src")
from agents import DRRN

st = ["you are in the kitchen . a locked tin door blocks the way . you see a tin key here . journal locked tin door",
      "take iron key carrying tin key door"]
texts = st + ["take tin key", "drop tin key", "go north", "look", "take iron key"]
S = [st[0], st[0], st[1]]; A = ["take tin key", "take iron key", "drop tin key"]; y = np.array([0.3, -0.2, 0.5])

for match in (False, True):
    ag = DRRN("word", texts, seed=1, batch=3, match=match)
    for k in ag.params: ag.params[k] += np.random.default_rng(0).normal(0, 0.05, ag.params[k].shape)

    def loss():
        q = np.array([ag.q_values(s, [a])[0] for s, a in zip(S, A)])
        return 0.5 * ((q - y) ** 2).sum()

    def analytic():
        Hs, cs = ag._enc(ag.enc_s, "s", S, ag.params); Ha, ca = ag._enc(ag.enc_a, "a", A, ag.params)
        q = (Hs * Ha).sum(1)
        if match:
            qm, mc = ag._match_forward(S, A, ag.params); q = q + qm
        err = q - y
        g = {k: np.zeros_like(v) for k, v in ag.params.items()}
        ag.enc_s.backward(err[:, None] * Ha, cs, {k[2:]: g[k] for k in g if k[:2] == "s_"})
        ag.enc_a.backward(err[:, None] * Hs, ca, {k[2:]: g[k] for k in g if k[:2] == "a_"})
        if match: ag._match_backward(err, mc, g)
        return g

    g = analytic(); worst = 0
    for k in ag.params:
        flat = ag.params[k].ravel()
        for i in np.random.default_rng(2).choice(flat.size, min(8, flat.size), replace=False):
            old = flat[i]; flat[i] = old + 1e-5; lp = loss(); flat[i] = old - 1e-5; lm = loss(); flat[i] = old
            num = (lp - lm) / 2e-5; ana = g[k].ravel()[i]
            if abs(num) + abs(ana) > 1e-9:
                worst = max(worst, abs(num - ana) / (abs(num) + abs(ana)))
    print(f"match={match}: max relative error {worst:.2e}", "PASS" if worst < 1e-4 else "FAIL")
