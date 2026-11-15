#!/bin/bash
# Hyper-parameter tuning: VALIDATION worlds + VALIDATION-only unseen lexicon (val_names). Test seeds untouched.
cd "$(dirname "$0")"
EP=4000
for lr in 0.0005 0.001; do
  python3 src/train.py --agent drrn --feat word --lr $lr --seed 100 --episodes $EP --final_eval val --out results/tune --tag _lr$lr
done
for d in 0.0 0.15 0.3; do
  python3 src/train.py --agent drrn --feat word --match --lr 0.0005 --tok_drop $d --seed 100 --episodes $EP --final_eval val --out results/tune --tag _drop$d
done
echo TUNEDONE
