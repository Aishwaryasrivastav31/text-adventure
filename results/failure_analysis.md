
### Test split: train

| Model | success | never took matching key | key but never unlocked | unlocked, no treasure | episodes with ≥1 take→drop loop | ended holding a wrong key | steps/episode |
|---|---|---|---|---|---|---|---|
| drrn_word | 23% | 47% | 21% | 9% | 40% | 40% | 42.6 |
| drrn_ngram | 40% | 40% | 11% | 8% | 29% | 39% | 37.5 |
| drrn_match_aug | 17% | 52% | 28% | 3% | 40% | 22% | 43.6 |

### Test split: new_names

| Model | success | never took matching key | key but never unlocked | unlocked, no treasure | episodes with ≥1 take→drop loop | ended holding a wrong key | steps/episode |
|---|---|---|---|---|---|---|---|
| drrn_word | 0% | 84% | 14% | 2% | 54% | 15% | 50.0 |
| drrn_ngram | 0% | 82% | 15% | 3% | 16% | 18% | 50.0 |
| drrn_match_aug | 29% | 40% | 31% | 0% | 6% | 25% | 40.1 |

### Test split: paraphrase

| Model | success | never took matching key | key but never unlocked | unlocked, no treasure | episodes with ≥1 take→drop loop | ended holding a wrong key | steps/episode |
|---|---|---|---|---|---|---|---|
| drrn_word | 0% | 84% | 15% | 1% | 64% | 3% | 49.9 |
| drrn_ngram | 0% | 70% | 18% | 12% | 36% | 24% | 49.9 |
| drrn_match_aug | 7% | 64% | 28% | 1% | 18% | 41% | 47.2 |
