"""Watch a trained DRRN play one episode:  python src/play.py results/drrn_ngram_s0 [split] [world_seed]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from textworld_lite import TextWorld, LEXICONS
from agents import DRRN
from train import run_episode  # noqa

path = sys.argv[1]; split = sys.argv[2] if len(sys.argv) > 2 else "train"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000
feat = "ngram" if "ngram" in path else "word"
match = "match" in path
tr = TextWorld(LEXICONS["train"]); vt = []
for s in range(300):
    o = tr.reset(s); vt += [o] + tr.admissible()
ag = DRRN(feat, vt, match=match)
w = np.load(path + ".npz")
for k in ag.params: ag.params[k][...] = w[k]
env = TextWorld(LEXICONS[split]); obs = env.reset(seed); total = 0
print(f"[{split}] world seed {seed}\n\n{obs}\n")
while True:
    cmds = env.admissible(); q = ag.q_values(obs, cmds)
    a = cmds[int(np.argmax(q))]
    obs, r, done, info = env.step(a); total += r
    print(f"> {a:32s} (Q={q.max():5.2f}, r={r:+.2f})\n  {obs}\n")
    if done: break
print("SUCCESS" if info["success"] else "FAILED", f"| return {total:.2f} | steps {env.steps}")
