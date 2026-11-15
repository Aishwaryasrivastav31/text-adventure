#!/bin/bash
cd "$(dirname "$0")"
python3 src/train.py --agent drrn --feat word --match --tok_drop 0.3 --seed 100 --episodes 4000 --final_eval val --out results/tune --tag _drop0.3
python3 src/train.py --agent drrn --feat word --match --aug --tok_drop 0.0 --seed 100 --episodes 4000 --final_eval val --out results/tune --tag _aug_drop0.0
python3 src/train.py --agent drrn --feat word --match --aug --tok_drop 0.15 --seed 100 --episodes 4000 --final_eval val --out results/tune --tag _aug_drop0.15
echo TUNE2DONE
