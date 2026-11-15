"""
Train / evaluate one agent.

Evaluation protocol (three DISJOINT world-seed ranges):
    train       worlds 0 ... (infinite stream, training only)
    validation  worlds 5,000,000+   learning curves + ALL hyper-parameter tuning
    test        worlds 10,000,000+  touched only by --final_eval test, after hyper-parameters are frozen

    python src/train.py --agent drrn --feat word --match --seed 0 --final_eval test --out results/final
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from textworld_lite import TextWorld, LEXICONS
from agents import RandomAgent, HeuristicAgent, GreedyAgent, TabularQ, DRRN

OFFSET = {"train": 0, "val": 5_000_000, "test": 10_000_000}


def run_episode(env, agent, seed, explore, learn, update_every=2, trace=None):
    obs, ret, t = env.reset(seed), 0.0, 0
    if getattr(agent, 'needs_env', False): agent.bind(env)
    while True:
        cmds = env.admissible()
        a = agent.act(obs, cmds, explore=explore)
        obs2, r, done, info = env.step(a)
        ret += r
        if trace is not None:
            trace.append(a)
        if learn:
            agent.observe(obs, a, r, obs2, env.admissible() if not done else ["look"], done)
            t += 1
            if isinstance(agent, DRRN) and t % update_every == 0:
                agent.update()
        obs = obs2
        if done:
            return ret, env.steps, float(info["success"])


def evaluate(agent, lexicon, seed_set, n=200):
    env = TextWorld(LEXICONS[lexicon])
    res = [run_episode(env, agent, OFFSET[seed_set] + i, explore=False, learn=False) for i in range(n)]
    ret, steps, succ = np.mean(res, axis=0)
    return {"return": float(ret), "steps": float(steps), "success": float(succ)}


def make_vocab_texts():
    env = TextWorld(LEXICONS["train"]); vt = []
    for s in range(300):
        o = env.reset(s); vt += [o] + env.admissible()
    return vt


def build_agent(agent, feat, match, lr, seed, vocab_texts, tok_drop=0.0):
    if agent == "random": return RandomAgent(seed)
    if agent == "heuristic": return HeuristicAgent(seed)
    if agent == "greedy": return GreedyAgent(seed)
    if agent == "tabular": return TabularQ(seed)
    return DRRN(feat, vocab_texts, seed=seed, lr=lr, match=match, tok_drop=tok_drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="drrn", choices=["random", "heuristic", "greedy", "tabular", "drrn"])
    ap.add_argument("--feat", default="word", choices=["word", "ngram"])
    ap.add_argument("--match", action="store_true", help="add exact-match cross-attention head")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--aug", action="store_true", help="vocabulary randomisation during training")
    ap.add_argument("--tok_drop", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--final_eval", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="results/final")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    kind = args.agent if args.agent != "drrn" else ("drrn_match" if args.match else f"drrn_{args.feat}")
    name = f"{kind}{'_aug' if args.aug else ''}{args.tag}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)

    env = TextWorld(LEXICONS["train_aug" if args.aug else "train"])
    agent = build_agent(args.agent, args.feat, args.match, args.lr, args.seed, make_vocab_texts(), args.tok_drop)
    learnable = args.agent in ("tabular", "drrn")
    n_eps = args.episodes if learnable else 0

    curve, t0 = [], time.time()
    for ep in range(n_eps + 1):
        if learnable:
            agent.eps = max(0.05, 1.0 - 0.95 * ep / (0.6 * n_eps))
        if learnable and ep % args.eval_every == 0:
            e = evaluate(agent, "train", "val", n=100)     # validation seeds only
            curve.append({"episode": ep, **e})
            print(f"[{name}] ep {ep:5d}  val succ {e['success']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        if ep < n_eps:
            run_episode(env, agent, OFFSET["train"] + ep, explore=True, learn=True)

    splits = ("train", "train_aug", "new_names", "paraphrase") if args.final_eval == "test" else ("train", "train_aug", "val_names")
    final = {s: evaluate(agent, s, args.final_eval) for s in splits}
    print(f"[{name}] FINAL({args.final_eval})", json.dumps({k: round(v['success'], 3) for k, v in final.items()}))
    json.dump({"args": vars(args), "final_eval": args.final_eval, "curve": curve, "final": final},
              open(f"{args.out}/{name}.json", "w"), indent=1)
    if args.agent == "drrn":
        agent.save(f"{args.out}/{name}.npz")


if __name__ == "__main__":
    main()
