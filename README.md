# Language Generalisation in RL for Text Adventures: Memorisation vs Relation Learning

**One-line summary.** A DRRN agent learns a procedurally generated text adventure only by *memorising words*; it scores
0% on unseen vocabulary. I diagnosed the cause, added an exact-match cross-attention ("match") head plus vocabulary
randomisation, and it reaches **29% on unseen names** (vs ≤0.4% for standard DRRNs, 10 seeds, test worlds) - about the
level of a one-step greedy agent that is allowed to see the simulator. It is **not** a complete fix: it is still brittle to
paraphrased templates (7%) and no agent solves the task reliably.

## 1. Question
Do RL agents that read text learn the *relation* in the text ("the key that matches the door"), or do they memorise
word-specific behaviour that breaks on new words?

## 2. Task: TextWorld-Lite (`src/textworld_lite.py`)
Each episode is a freshly generated world of 6-8 rooms. The agent sees only text (room, exits, items, inventory, its last
3 commands, a *Journal* line) and chooses from admissible text commands (`go`, `take`, `drop`, `unlock`, `look`).
It must read the door ("A locked **brass** door ..."), find the **matching** key among decoy keys, unlock the door and take
the treasure; it can carry 2 items, has 50 steps, rewards: +1 key, +1 unlock, +10 treasure, -0.05/step.

*Design note (a bug I found):* in my first version the door material was visible only in the door's room, so in a key room the
agent had no information about which key matches - a memory problem, not a language problem. I added a `Journal:` line that
repeats the door text once seen. All numbers below are for this corrected environment; earlier numbers were discarded.

**Evaluation splits (all at game level, disjoint world seeds):** train worlds `0..3999` · validation worlds `5,000,000+`
(learning curves + all tuning) · test worlds `10,000,000+` (touched only for the final numbers, after freezing hyper-parameters).
Language splits on the test worlds: *original train words*; *randomised-name worlds* (the augmentation lexicon);
*unseen names* (new room/item/key words, same templates); *unseen names + paraphrase* (new words **and** new sentence templates).
Test and validation vocabularies share only the word "key" with training (verified).

## 3. Methods
* **Baselines:** random; scripted heuristic (take > unlock > move); **one-step greedy** (simulates each command on a copy of the game -
  it *has the simulator*, which the learned agents do not); tabular Q-learning on exact (text, command) pairs.
* **DRRN** (He et al. 2016), pure NumPy with hand-derived backprop and Adam (gradient-checked vs finite differences, error ~1e-8):
  state and command are encoded separately, `Q(s,a) = h_s · h_a`; trained with DQN (replay, target net, Huber loss).
  Features: **word** or hashed **character trigrams**.
* **Match head (proposed fix):** `Q(s,a) = h_s·h_a + match(s,a)`. Every argument word of the command is matched by *exact string identity*
  against every position of the state text (so it works for any word, including unseen ones); each matched position contributes the
  context vector of its neighbours (±2 words, mostly known function words such as "locked", "door", "see"), and a verb-specific read-out
  turns this into a Q-value. Intuition: "the word that sits next to *locked … door* is the key to take."
* **Augmented training** (used for the match model): random pseudo-words replace names during training (vocabulary randomisation),
  plus 15% token dropout on the base encoders, so word identity cannot be memorised. Pseudo-words are disjoint from all held-out words.
* Standard DRRNs (word / trigram) are reported **both** with standard training and (word) with the augmented pipeline.

**Tuning (validation only; 1 agent seed per config, 4000 episodes; success %, "unseen" = validation-only lexicon):**

| Config | seen | unseen |
|---|---|---|
| Word DRRN lr 5e-4 / 1e-3 | 20 / 8 | 0.5 / 1.5 |
| Match head, dropout 0 / 0.15 / 0.3 | 46 / 41.5 / 4.5 | 3.5 / 9 / 7 |
| Match + randomisation, dropout 0 / 0.15 | 24.5 / 5.5 | 16.5 / **24** |

Selection rule fixed in advance: best unseen validation success → randomisation + dropout 0.15, lr 5e-4, 4000 episodes.

## 4. Results (test worlds, 10 seeds each, 200 worlds per split per seed)
Success rate %, mean ± std over seeds (95% t-CI half-width in brackets).

| Agent | Original train words | Randomised-name worlds | **Unseen names** | Unseen names + paraphrase |
|---|---|---|---|---|
| Random | 5.8 ± 1.9 (1.4) | 5.3 ± 1.3 (0.9) | 4.9 ± 1.2 (0.8) | 5.3 ± 1.4 (1.0) |
| Scripted heuristic | 3.6 ± 1.3 (1.0) | 1.4 ± 1.0 (0.7) | 2.7 ± 1.2 (0.8) | 2.8 ± 0.9 (0.6) |
| One-step greedy (has simulator) | 28.5 ± 2.9 (2.1) | 28.4 ± 2.7 (2.0) | 28.2 ± 2.9 (2.1) | 27.0 ± 2.0 (1.4) |
| Tabular Q-learning | 5.3 ± 1.0 (0.7) | 4.9 ± 1.8 (1.3) | 4.8 ± 2.2 (1.6) | 5.5 ± 1.4 (1.0) |
| DRRN word, standard | 22.7 ± 11.4 (8.1) | 0.1 ± 0.2 (0.2) | 0.4 ± 0.6 (0.4) | 1.0 ± 1.3 (0.9) |
| DRRN char-trigram, standard | **42.5 ± 8.9 (6.4)** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| DRRN word, augmented | 3.2 ± 4.4 (3.1) | 9.4 ± 12.8 (9.2) | 8.5 ± 10.8 (7.8) | 2.9 ± 2.8 (2.0) |
| **DRRN + match head, augmented** | 20.1 ± 7.8 (5.6) | **28.1 ± 6.8 (4.8)** | **29.2 ± 7.4 (5.3)** | 7.3 ± 4.6 (3.3) |

Figures: `figures/learning_curves.png` (validation curves), `figures/generalisation.png` (per-split success with CI bars and per-seed dots),
`figures/memorisation_vs_relation.png` (success on seen vs unseen words, one dot per seed). Full table + tests: `results/summary.md`.

**Findings (Welch t-tests over per-seed success; no multiple-comparison correction was applied):**
1. **Memorisation is real.** Standard word and trigram DRRNs lose 22 and 43 percentage points from seen to unseen words
   (word: 22.7 → 0.4; trigram: 42.5 → 0.0). Tabular Q-learning cannot learn at all (every world yields new strings).
2. **Trigram vs word features:** trigrams are much better on seen words (+19.9 pp, p = 0.0004) but give **no** benefit on unseen words
   (both ≈ 0). Sub-word sharing helps memorise the training vocabulary; it does not give relational reasoning.
3. **The match head generalises to unseen names:** 29.2% vs 0.4% (word, standard), 0.0% (trigram, standard) and 8.5% (word, same augmented pipeline;
   +20.7 pp, p = 0.0001). The comparison against the augmented word DRRN shows the gain is not just from augmentation. Its unseen-name success is about
   the same as its randomised-name success (28-29%): little memorisation gap.
4. **Cost:** on original training words the match model (20.1%) is on par with the standard word DRRN (22.7%; p = 0.57) but well below the trigram DRRN (42.5%).
5. **Not solved:** with new sentence templates the match model drops to 7.3%. Its context windows then contain unseen words ("bolted", "bars"), so the learned
   "role" signal disappears. No learned agent exceeds ~43% on its best split, and the simulator-cheating greedy agent (≈28%) matches the match model on unseen names.

## 5. Failure analysis (`src/failure_analysis.py`, 3 seeds × 80 test worlds per model; `results/failure_analysis.md`)
Why do episodes fail? (% of episodes)

| Test split | Model | success | never took matching key | had key, never unlocked | unlocked, no treasure | ≥1 take→drop loop | ended holding a wrong key |
|---|---|---|---|---|---|---|---|
| unseen names | word, standard | 0 | 84 | 14 | 2 | 54 | 15 |
| unseen names | trigram, standard | 0 | 82 | 15 | 3 | 16 | 18 |
| unseen names | match head | 29 | 40 | 31 | 0 | 6 | 25 |
| paraphrase | word, standard | 0 | 84 | 15 | 1 | 64 | 3 |
| paraphrase | trigram, standard | 0 | 70 | 18 | 12 | 36 | 24 |
| paraphrase | match head | 7 | 64 | 28 | 1 | 18 | 41 |

* **The wrong-key loop.** On unseen words the standard DRRNs cannot tell which key matches (unseen words collapse to `<unk>`), so they pick keys effectively at random,
  pick up and put down items (the word DRRN takes then drops an item in 54-64% of unseen episodes), hold decoy keys and run out of steps.
  Over 80% of their failures are "never took the matching key". Mechanism: a dot product of two *independent* bag-of-features encoders
  can store "brass door → take brass key" but cannot express "the same word occurs in the door text and in the key name".
* **The match head removes most of the loops** on unseen names (6% of episodes) and raises matching-key pickup from ~16% to ~60%.
* **Remaining failures are mostly navigation/memory.** Even with the match head, 40% of unseen-name episodes never pick the matching key and 31% hold it but never unlock
  the door: the agent knows *what* the door looks like (Journal) but not *where* it is, and its only memory is a 3-command history. This is a planning/memory limitation,
  not a language one, and is what a recurrent or map-building agent should address.
* Caveat: counts come from 3 seeds × 80 worlds per cell, so they are descriptive, not tested.

## 6. Limitations (please read)
* **Synthetic, templated environment**; it is not TextWorld/Jericho. I could not run Microsoft TextWorld (needs Inform7) or an LLM zero-shot baseline (no model weights available in my sandbox).
* **Tuning asymmetry and thin tuning:** augmentation/dropout were tuned for the match model on 1 seed per config; baselines were run with standard training (and the word DRRN also with the
  augmented pipeline) but were not given separate tuning. Selection rule was fixed in advance, but one seed per config is noisy.
* The augmented models' "original train words" column is nearly out-of-distribution for them (those 4 materials are rare in training), so negative "memorisation gaps" for them
  are not evidence of better-than-seen performance.
* **Procedure note:** an early aborted run of the augmented word DRRN showed test-seed numbers before I added the "standard training" baselines; this influenced run design,
  not any hyper-parameter selection.
* High seed variance (std up to 13 pp); 10 seeds give wide CIs for the augmented models. Many t-tests without correction.
* Greedy baseline uses privileged simulator access, so it is a reference, not a fair competitor.
* Paraphrase generalisation is only partially addressed; an earlier version of this project (3 seeds, tuning on test seeds, no Journal) is superseded and should not be cited.

## 7. Future work
Replace the hand-built context windows with learned token-level cross-attention; a pretrained LM encoder; recurrent/map memory (to fix the "has key, never unlocks" failures);
policy-gradient/PPO; a Jericho/TextWorld evaluation; a simplified RLHF/REINFORCE variant for language generation.

## 8. Reproduce
```
pip install numpy scipy matplotlib
python src/gradcheck.py                 # backprop vs finite differences (base DRRN and match head)
bash tune.sh; bash tune2.sh             # validation-only tuning (results/tune)
bash run_final.sh                       # 80 test runs (8 agents x 10 seeds, ~5 h on 1 CPU core); resumable
python src/analyze.py                   # tables, tests, figures  (results/summary.md, figures/)
python src/failure_analysis.py results/final 3 80
python src/play.py results/final/drrn_match_aug_s0 new_names 10000003   # watch an episode (see note)
```
Layout: `src/textworld_lite.py` environment · `src/agents.py` agents · `src/train.py` training/eval protocol · `src/analyze.py`, `src/failure_analysis.py` · `results/` · `figures/`.
