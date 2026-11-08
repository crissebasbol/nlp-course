## Resultados en val (334 textos)

### Detección (A y B)

| system                                                          |   flag_acc |   sentence_acc |   sentence_acc_on_errors |   n |
|:----------------------------------------------------------------|-----------:|---------------:|-------------------------:|----:|
| Baseline: siempre error en la última oración                    |     0.5419 |         0.3174 |                   0.5856 | 334 |
| BiomedBERT fine-tuneado (w = 1)                                 |     0.5479 |         0.5269 |                   0.9392 | 334 |
| Qwen2.5-7B + DSPy sin compilar                                  |     0.5479 |         0.3683 |                   0.4917 | 334 |
| Qwen2.5-7B + DSPy compilado (w = 0)                             |     0.5749 |         0.4042 |                   0.6022 | 334 |
| Combinación ponderada (la combinación elige la oración)         |     0.5479 |         0.5419 |                   0.6575 | 334 |
| Combinación decide A, BERT elige la oración                     |     0.5749 |         0.5509 |                   0.8729 | 334 |
| Pipeline v1 (combinación elige la oración + corrector)          |     0.5479 |         0.5419 |                   0.6575 | 334 |
| Pipeline final (combinación decide A, BERT elige B + corrector) |     0.5749 |         0.5509 |                   0.8729 | 334 |

### Corrección (C)

| system                                                          |   rouge1_official |   rouge1_pairs |   n_pairs |   bertscore_official |   bertscore_pairs |
|:----------------------------------------------------------------|------------------:|---------------:|----------:|---------------------:|------------------:|
| Pipeline v1 (combinación elige la oración + corrector)          |            0.4232 |         0.6557 |       121 |               0.4963 |            0.8575 |
| Pipeline final (combinación decide A, BERT elige B + corrector) |            0.3868 |         0.6216 |       166 |               0.4983 |            0.846  |
