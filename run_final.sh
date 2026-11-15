#!/bin/bash
# FINAL experiments on TEST seeds (hyper-parameters frozen from results/tune, i.e. validation only).
#  - DRRN word / trigram with STANDARD training (their own best setup): lr 5e-4, no augmentation, no dropout
#  - DRRN word + DRRN+match with the AUGMENTED pipeline chosen on validation: --aug --tok_drop 0.15
#  - random / heuristic / tabular baselines.   10 seeds each, 4000 episodes.  Resumable.
cd "$(dirname "$0")"
OUT=results/final; C="--episodes 4000 --final_eval test --out $OUT --lr 0.0005"
run() { name=$1; shift; [ -f "$OUT/${name}.json" ] || python3 src/train.py "$@" $C; }
for s in 0 1 2 3 4 5 6 7 8 9; do
  run random_s$s          --agent random    --seed $s
  run heuristic_s$s       --agent heuristic --seed $s
  run greedy_s$s          --agent greedy    --seed $s
  run tabular_aug_s$s     --agent tabular   --seed $s --aug
  run drrn_match_aug_s$s  --agent drrn --feat word --match --seed $s --aug --tok_drop 0.15
  run drrn_word_s$s       --agent drrn --feat word  --seed $s
  run drrn_ngram_s$s      --agent drrn --feat ngram --seed $s
  run drrn_word_aug_s$s   --agent drrn --feat word  --seed $s --aug --tok_drop 0.15
done
echo FINALDONE
