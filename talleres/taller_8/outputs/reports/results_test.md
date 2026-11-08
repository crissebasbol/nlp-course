## Resultados en test (327 textos)

### Detección (A y B)

| system                                                          |   flag_acc |   sentence_acc |   sentence_acc_on_errors |   n |
|:----------------------------------------------------------------|-----------:|---------------:|-------------------------:|----:|
| Baseline: siempre error en la última oración                    |     0.5596 |         0.3456 |                   0.6175 | 327 |
| BiomedBERT fine-tuneado (w = 1)                                 |     0.5657 |         0.5535 |                   0.9508 | 327 |
| Qwen2.5-7B + DSPy sin compilar                                  |     0.581  |         0.37   |                   0.5137 | 327 |
| Qwen2.5-7B + DSPy compilado (w = 0)                             |     0.5657 |         0.3547 |                   0.5574 | 327 |
| Combinación ponderada (la combinación elige la oración)         |     0.5199 |         0.5138 |                   0.6557 | 327 |
| Combinación decide A, BERT elige la oración                     |     0.5657 |         0.5474 |                   0.9016 | 327 |
| Pipeline v1 (combinación elige la oración + corrector)          |     0.5199 |         0.5138 |                   0.6557 | 327 |
| Pipeline final (combinación decide A, BERT elige B + corrector) |     0.5657 |         0.5474 |                   0.9016 | 327 |
| Referencia: Claude Opus 5.5                                     |     0.948  |         0.9419 |                   0.9508 | 327 |

### Corrección (C)

| system                                                          |   rouge1_official |   rouge1_pairs |   n_pairs |   bertscore_official |   bertscore_pairs |
|:----------------------------------------------------------------|------------------:|---------------:|----------:|---------------------:|------------------:|
| Pipeline v1 (combinación elige la oración + corrector)          |            0.3712 |         0.6015 |       122 |               0.4635 |            0.8489 |
| Pipeline final (combinación decide A, BERT elige B + corrector) |            0.3405 |         0.5693 |       171 |               0.4788 |            0.8336 |
| Referencia: Claude Opus 5.5                                     |            0.7986 |         0.7224 |       176 |               0.8836 |            0.8803 |
