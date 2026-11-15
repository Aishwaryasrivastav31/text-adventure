| Agent | n seeds | Original train words success | Randomised-name worlds success | Unseen names success | Unseen names + paraphrase success | Return (orig. words) |
|---|---|---|---|---|---|---|
| Random | 10 | 5.8 ± 1.9 (CI ±1.4) | 5.3 ± 1.3 (CI ±0.9) | 4.9 ± 1.2 (CI ±0.8) | 5.3 ± 1.4 (CI ±1.0) | -1.07 ± 0.16 |
| Scripted heuristic | 10 | 3.6 ± 1.3 (CI ±1.0) | 1.4 ± 1.0 (CI ±0.7) | 2.7 ± 1.2 (CI ±0.8) | 2.8 ± 0.9 (CI ±0.6) | -1.50 ± 0.14 |
| One-step greedy (has simulator) | 10 | 28.5 ± 2.9 (CI ±2.1) | 28.4 ± 2.7 (CI ±2.0) | 28.2 ± 2.9 (CI ±2.1) | 27.0 ± 2.0 (CI ±1.4) | 2.00 ± 0.26 |
| Tabular Q-learning | 10 | 5.3 ± 1.0 (CI ±0.7) | 4.9 ± 1.8 (CI ±1.3) | 4.8 ± 2.2 (CI ±1.6) | 5.5 ± 1.4 (CI ±1.0) | -1.12 ± 0.10 |
| DRRN (word), standard training | 10 | 22.7 ± 11.4 (CI ±8.1) | 0.1 ± 0.2 (CI ±0.2) | 0.4 ± 0.6 (CI ±0.4) | 1.0 ± 1.3 (CI ±0.9) | 0.98 ± 1.07 |
| DRRN (char-trigram), standard training | 10 | 42.5 ± 8.9 (CI ±6.4) | 0.0 ± 0.0 (CI ±0.0) | 0.0 ± 0.0 (CI ±0.0) | 0.0 ± 0.0 (CI ±0.0) | 3.55 ± 0.82 |
| DRRN (word), augmented training | 10 | 3.2 ± 4.4 (CI ±3.1) | 9.4 ± 12.8 (CI ±9.2) | 8.5 ± 10.8 (CI ±7.8) | 2.9 ± 2.8 (CI ±2.0) | -1.74 ± 0.49 |
| DRRN + match head (proposed), augmented | 10 | 20.1 ± 7.8 (CI ±5.6) | 28.1 ± 6.8 (CI ±4.8) | 29.2 ± 7.4 (CI ±5.3) | 7.3 ± 4.6 (CI ±3.3) | 0.62 ± 0.77 |

Cells: mean success rate (%) ± std over seeds (95% CI half-width in brackets, t-distribution); 200 fresh TEST worlds per split per seed.

Welch t-tests (two-sided, over per-seed success rates):

| Comparison | Split | mean diff (pp) | p-value |
|---|---|---|---|
| DRRN (char-trigram), standard training vs DRRN (word), standard training | Original train words | +19.9 | 0.0004 |
| DRRN (char-trigram), standard training vs DRRN (word), standard training | Randomised-name worlds | -0.1 | 0.1679 |
| DRRN (char-trigram), standard training vs DRRN (word), standard training | Unseen names | -0.4 | 0.0528 |
| DRRN (char-trigram), standard training vs DRRN (word), standard training | Unseen names + paraphrase | -1.0 | 0.0332 |
| DRRN + match head (proposed), augmented vs DRRN (word), augmented training | Original train words | +16.9 | 0.0000 |
| DRRN + match head (proposed), augmented vs DRRN (word), augmented training | Randomised-name worlds | +18.7 | 0.0012 |
| DRRN + match head (proposed), augmented vs DRRN (word), augmented training | Unseen names | +20.7 | 0.0001 |
| DRRN + match head (proposed), augmented vs DRRN (word), augmented training | Unseen names + paraphrase | +4.4 | 0.0214 |
| DRRN + match head (proposed), augmented vs DRRN (word), standard training | Original train words | -2.5 | 0.5668 |
| DRRN + match head (proposed), augmented vs DRRN (word), standard training | Randomised-name worlds | +28.0 | 0.0000 |
| DRRN + match head (proposed), augmented vs DRRN (word), standard training | Unseen names | +28.8 | 0.0000 |
| DRRN + match head (proposed), augmented vs DRRN (word), standard training | Unseen names + paraphrase | +6.2 | 0.0021 |
| DRRN + match head (proposed), augmented vs DRRN (char-trigram), standard training | Original train words | -22.4 | 0.0000 |
| DRRN + match head (proposed), augmented vs DRRN (char-trigram), standard training | Randomised-name worlds | +28.1 | 0.0000 |
| DRRN + match head (proposed), augmented vs DRRN (char-trigram), standard training | Unseen names | +29.2 | 0.0000 |
| DRRN + match head (proposed), augmented vs DRRN (char-trigram), standard training | Unseen names + paraphrase | +7.3 | 0.0008 |

Memorisation gap = success on seen words minus success on unseen names (pp, mean ± std over seeds):

- Random: +0.9 ± 2.3
- Scripted heuristic: +0.9 ± 1.5
- One-step greedy (has simulator): +0.2 ± 4.1
- Tabular Q-learning: +0.5 ± 1.9
- DRRN (word), standard training: +22.3 ± 11.3
- DRRN (char-trigram), standard training: +42.5 ± 8.9
- DRRN (word), augmented training: -5.2 ± 7.0
- DRRN + match head (proposed), augmented: -9.1 ± 6.1
