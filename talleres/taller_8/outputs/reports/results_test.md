## Resultados en test (327 textos)

### Detección (A y B)

| system                                       |   flag_acc |   sentence_acc |   sentence_acc_on_errors |   n |
|:---------------------------------------------|-----------:|---------------:|-------------------------:|----:|
| Baseline: siempre error en la última oración |     0.5596 |         0.3456 |                   0.6175 | 327 |
| BiomedBERT fine-tuneado (w = 1)              |     0.5657 |         0.5535 |                   0.9508 | 327 |
| Qwen2.5-7B + DSPy sin compilar               |     0.581  |         0.37   |                   0.5137 | 327 |
| Qwen2.5-7B + DSPy compilado (w = 0)          |     0.5657 |         0.3547 |                   0.5574 | 327 |
| Combinación ponderada                        |     0.5199 |         0.5138 |                   0.6557 | 327 |
| Pipeline completo (combinación + corrector)  |     0.5199 |         0.5138 |                   0.6557 | 327 |

### Corrección (C)

| system                                      |   rouge1_official |   rouge1_pairs |   n_pairs |   bertscore_official |   bertscore_pairs |
|:--------------------------------------------|------------------:|---------------:|----------:|---------------------:|------------------:|
| Pipeline completo (combinación + corrector) |            0.3712 |         0.6015 |       122 |               0.4635 |            0.8489 |
