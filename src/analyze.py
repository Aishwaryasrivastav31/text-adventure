"""Aggregate results/final/*.json (TEST seeds) -> results/summary.md, figures.  Mean +- 95% CI (t-dist) over seeds,
plus Welch t-tests for the comparisons that matter."""
import glob, json, os, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "results/final"
R = {}
for f in sorted(glob.glob(f"{D}/*.json")):
    d = json.load(open(f)); a = d["args"]
    assert d["final_eval"] == "test", f"{f} was not evaluated on test seeds"
    kind = a["agent"] if a["agent"] != "drrn" else ("drrn_match" if a["match"] else f"drrn_{a['feat']}")
    if a.get("aug") and a["agent"] in ("tabular", "drrn") and not a.get("match"): kind += "_aug"
    R.setdefault(kind, []).append(d)

LABEL = {"random": "Random", "heuristic": "Scripted heuristic", "greedy": "One-step greedy (has simulator)", "tabular_aug": "Tabular Q-learning",
         "drrn_word": "DRRN (word), standard training", "drrn_ngram": "DRRN (char-trigram), standard training",
         "drrn_word_aug": "DRRN (word), augmented training", "drrn_match": "DRRN + match head (proposed), augmented"}
ORDER = [k for k in LABEL if k in R]
SPLITS = [("train", "Original train words"), ("train_aug", "Randomised-name worlds"), ("new_names", "Unseen names"), ("paraphrase", "Unseen names + paraphrase")]

def vals(k, s, m="success"): return np.array([d["final"][s][m] for d in R[k]])
def ci(x):
    n = len(x)
    return x.mean(), (stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan"))

L = ["| Agent | n seeds | " + " | ".join(f"{n} success" for _, n in SPLITS) + " | Return (orig. words) |", "|---|---|" + "---|" * (len(SPLITS) + 1)]
for k in ORDER:
    cells = [f"{100*vals(k,s).mean():.1f} ± {100*vals(k,s).std(ddof=1) if len(vals(k,s))>1 else float('nan'):.1f} (CI ±{100*ci(vals(k,s))[1]:.1f})" for s, _ in SPLITS]
    r = ci(vals(k, "train", "return"))
    L.append(f"| {LABEL[k]} | {len(R[k])} | " + " | ".join(cells) + f" | {r[0]:.2f} ± {r[1]:.2f} |")
L += ["", "Cells: mean success rate (%) ± std over seeds (95% CI half-width in brackets, t-distribution); 200 fresh TEST worlds per split per seed.", "",
      "Welch t-tests (two-sided, over per-seed success rates):", "", "| Comparison | Split | mean diff (pp) | p-value |", "|---|---|---|---|"]
for a, b in [("drrn_ngram", "drrn_word"), ("drrn_match", "drrn_word_aug"), ("drrn_match", "drrn_word"), ("drrn_match", "drrn_ngram")]:
    if a in R and b in R:
        for s, n in SPLITS:
            x, y = vals(a, s), vals(b, s)
            p = stats.ttest_ind(x, y, equal_var=False).pvalue
            L.append(f"| {LABEL[a]} vs {LABEL[b]} | {n} | {100*(x.mean()-y.mean()):+.1f} | {p:.4f} |")
os.makedirs("figures", exist_ok=True)
open("results/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

# learning curves (validation seeds)
plt.figure(figsize=(7, 4.2))
for k, c in [("drrn_word", "tab:blue"), ("drrn_ngram", "tab:green"), ("drrn_word_aug", "tab:purple"), ("drrn_match", "tab:red"), ("tabular_aug", "tab:orange")]:
    if k not in R: continue
    eps = [p["episode"] for p in R[k][0]["curve"]]
    Y = np.array([[p["success"] for p in d["curve"]] for d in R[k]])
    m = Y.mean(0); h = stats.t.ppf(0.975, len(Y) - 1) * Y.std(0, ddof=1) / np.sqrt(len(Y))
    plt.plot(eps, m, color=c, label=LABEL[k]); plt.fill_between(eps, m - h, m + h, color=c, alpha=.2)
plt.xlabel("Training episodes"); plt.ylabel("Success (100 validation worlds)"); plt.grid(alpha=.3)
plt.title("Learning curves, mean ± 95% CI"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/learning_curves.png", dpi=150); plt.close()

# generalisation with CI bars + per-seed dots
plt.figure(figsize=(9, 4.4)); w = 0.8 / len(ORDER)
for i, k in enumerate(ORDER):
    xs = np.arange(len(SPLITS)) + i * w
    plt.bar(xs, [100 * ci(vals(k, s))[0] for s, _ in SPLITS], w, yerr=[100 * ci(vals(k, s))[1] for s, _ in SPLITS], capsize=2, label=LABEL[k], alpha=.8)
    for j, (s, _) in enumerate(SPLITS):
        plt.scatter(np.full(len(R[k]), xs[j]), 100 * vals(k, s), s=6, color="k", zorder=3)
plt.xticks(np.arange(len(SPLITS)) + 0.4 - w / 2, [n for _, n in SPLITS], fontsize=8); plt.ylabel("Test success (%)")
plt.title("Generalisation to held-out language (test seeds)"); plt.legend(fontsize=7); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("figures/generalisation.png", dpi=150)


# ---- memorisation vs relation learning: in-distribution vs unseen-names success, per seed
plt.figure(figsize=(6.2, 4.6))
cols = {"drrn_word": "tab:blue", "drrn_ngram": "tab:green", "drrn_word_aug": "tab:purple", "drrn_match": "tab:red", "tabular_aug": "tab:orange"}
for k, c in cols.items():
    if k not in R: continue
    x = 100 * vals(k, "train", "success"); y = 100 * vals(k, "new_names", "success")
    plt.scatter(x, y, color=c, alpha=.6, s=22, label=LABEL[k])
    plt.scatter(x.mean(), y.mean(), color=c, marker="*", s=180, edgecolor="k", zorder=4)
plt.plot([0, 60], [0, 60], "k--", lw=1)
plt.text(32, 36, "relation learning\n(no gap)", fontsize=8, rotation=0)
plt.text(34, 4, "memorisation\n(good on seen words,\nfails on unseen)", fontsize=8)
plt.xlabel("Success on SEEN words (%)  [original training vocabulary]"); plt.ylabel("Success on UNSEEN names (%)")
plt.title("Memorisation vs relation learning (each dot = 1 seed, star = mean)"); plt.legend(fontsize=6.5, loc="upper left")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("figures/memorisation_vs_relation.png", dpi=150)

gap = ["", "Memorisation gap = success on seen words minus success on unseen names (pp, mean ± std over seeds):", ""]
for k in ORDER:
    g = 100 * (vals(k, "train", "success") - vals(k, "new_names", "success"))
    gap.append(f"- {LABEL[k]}: {g.mean():+.1f} ± {g.std(ddof=1) if len(g)>1 else float('nan'):.1f}")
open("results/summary.md", "a").write("\n".join(gap) + "\n"); print("\n".join(gap))
