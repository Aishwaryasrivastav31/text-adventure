"""Quantified failure analysis from trajectories (TEST worlds).
    python src/failure_analysis.py results/final [n_seeds] [n_worlds]
For every trained DRRN variant: why do episodes fail, and how often does the take/drop wrong-key loop occur?"""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from textworld_lite import TextWorld, LEXICONS
from agents import DRRN
from train import make_vocab_texts, OFFSET

D = sys.argv[1] if len(sys.argv) > 1 else "results/final"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 150
vt = make_vocab_texts()
KINDS = [("drrn_word", "word", False), ("drrn_ngram", "ngram", False), ("drrn_match_aug", "word", True)]

def load(kind, feat, match, seed):
    f = f"{D}/{kind}_s{seed}"
    if not os.path.exists(f + ".npz"): return None
    a = json.load(open(f + ".json"))["args"]
    ag = DRRN(feat, vt, seed=seed, match=match, tok_drop=0.0)
    w = np.load(f + ".npz")
    for k in ag.params: ag.params[k][...] = w[k]
    return ag

def classify(env, trace):
    """returns (category, n_take_drop_cycles, wrong_key_taken, ended_holding_wrong_key)"""
    cycles = sum(1 for i in range(len(trace) - 1) if trace[i].startswith("take ") and trace[i + 1] == "drop " + trace[i][5:])
    wrong = [i for i in env.inv if i.endswith(" key") and i != env.key]
    wrong_taken = any(t.startswith("take ") and t.endswith(" key") and t[5:] != env.key for t in trace)
    if env.treasure in env.inv: cat = "success"
    elif not env.got_key: cat = "never picked up the matching key"
    elif not env.unlocked: cat = "had matching key, never unlocked the door"
    else: cat = "door unlocked, treasure not taken"
    return cat, cycles, wrong_taken, bool(wrong)

out = []
for lex in ("train", "new_names", "paraphrase"):
    out.append(f"\n### Test split: {lex}\n")
    out.append("| Model | success | never took matching key | key but never unlocked | unlocked, no treasure | episodes with ≥1 take→drop loop | ended holding a wrong key | steps/episode |")
    out.append("|---|---|---|---|---|---|---|---|")
    for kind, feat, match in KINDS:
        cats, loops, held, steps, n = {}, 0, 0, 0, 0
        for s in range(NS):
            ag = load(kind, feat, match, s)
            if ag is None: continue
            env = TextWorld(LEXICONS[lex])
            for i in range(NW):
                obs = env.reset(OFFSET["test"] + 5000 + i); trace = []; done = False
                while not done:
                    cmds = env.admissible(); a = ag.act(obs, cmds, explore=False)
                    obs, r, done, _ = env.step(a); trace.append(a)
                cat, cyc, wt, hw = classify(env, trace)
                cats[cat] = cats.get(cat, 0) + 1; loops += cyc > 0; held += hw; steps += env.steps; n += 1
        if n == 0: continue
        pc = lambda c: f"{100*cats.get(c,0)/n:.0f}%"
        out.append(f"| {kind} | {pc('success')} | {pc('never picked up the matching key')} | {pc('had matching key, never unlocked the door')} | "
                   f"{pc('door unlocked, treasure not taken')} | {100*loops/n:.0f}% | {100*held/n:.0f}% | {steps/n:.1f} |")
open("results/failure_analysis.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
