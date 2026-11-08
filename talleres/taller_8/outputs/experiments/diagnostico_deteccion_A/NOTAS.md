# Diagnóstico de la subtarea A (¿el texto tiene error?), 2026-11-07

Notas de trabajo, todavía no documentadas como hallazgo en el README. Todo se midió con los artefactos de
`outputs/` (BERT entrenado, puntajes DSPy, split por caso). Los scripts están en `scripts/` (se corrieron desde
el scratchpad; las rutas de entrada/salida van por argumento) y los resultados crudos en `resultados/`.

## Punto de partida (test, 327 textos: 183 con error, 144 sin error)

| Sistema | A | B | B en textos con error | % marcados | Correctos reconocidos |
|---|---:|---:|---:|---:|---:|
| BiomedBERT solo | 0.566 | 0.554 | 0.951 | 96 % | 7 / 144 |
| Pipeline final (combinación decide A, BERT elige B) | 0.566 | 0.547 | 0.902 | 92 % | 14 / 144 |
| Claude Opus 5.5 | 0.948 | 0.942 | 0.951 | 57 % | 134 / 144 |

Techo de B para un sistema que marca todo como error: 183/327 = 0.56. BERT ya está pegado a ese techo.

## Prueba 1: ¿qué aprendió BERT? (`scripts/diag.py`, sin LLM)

- Las dos versiones de un caso NO están alineadas por posición: en la versión correcta la oración de la
  respuesta suele estar en otra posición y con otra redacción (ej. con error [7] "Use of gripe water was
  recommended." / correcta [4] "Parents were reassured after examining the patient.").
- Midiendo por contenido (la oración que solo existe en la versión correcta), en las parejas sin ambigüedad:
  **BERT elige esa oración en el texto correcto en 85/87 (98 %) en val y 83/87 (95 %) en test.**
- Probabilidad máxima de BERT, mediana: 0.966 con error y 0.966 sin error. **AUC de BERT para A: 0.48**
  (máximo p_bert) y 0.47 (diferencia 1.ª - 2.ª).
- Conclusión: BERT encuentra la oración clave (diagnóstico/tratamiento) haya o no error, pero no sabe si es
  correcta. Ningún umbral sobre BERT puede arreglar A.
- Observación sin medir: la posición de la respuesta cambia entre versiones; si fuera sistemática sería un
  atajo del dataset, no conocimiento médico.

## Prueba 2: ¿la confianza de Qwen separa? (`scripts/diag.py`, val)

| | DSPy compilado | DSPy sin compilar |
|---|---:|---:|
| Dice "hay error" | 88 % | 80 % |
| Detecta textos con error | 166/181 (92 %) | 149/181 (82 %) |
| Reconoce textos correctos | 26/153 (17 %) | 34/153 (22 %) |
| Confianza declarada | casi siempre 0.90 / 0.95 | casi siempre 0.95 |
| AUC para A | 0.57 | 0.53 |

- Los 3 demos que eligió BootstrapFewShot tienen `has_error=True` (sesgo hacia "sí hay error").
- BERT (máx) + Qwen (confianza con signo), sin ajustar: AUC 0.54.

## Prueba 3: verificación de la oración de BERT con Qwen (`scripts/verify_exp.py` + `verify_analyze.py`)

Muestra de val: 100 textos (50/50, seed 42). Puntaje = logprobs de "Yes"/"No" (llama-server `top_logprobs`).

| Variante | AUC | IC 95 % (bootstrap) | A (umbral por CV, azar = 0.50) |
|---|---:|---|---:|
| V1 directa: "¿la oración k es correcta? Yes/No" | 0.602 | 0.49 a 0.71 | 0.610 |
| V2 ciega (se oculta k, Qwen la escribe) + juez | 0.499 | 0.45 a 0.55 | 0.490 |
| V2b ciega + coseno de embeddings | 0.510 | 0.39 a 0.62 | 0.490 |

- V1 con regla fija 0.5: detecta 35/50 con error, reconoce 26/50 correctos, marca 59 %.
- La respuesta ciega contiene el cambio clave de la referencia en 9/47; el corrector actual en 9/43 (~20 %).
  Qwen 7B sabe la respuesta del caso en ~1 de cada 5.

## Opción A completa: V1 sobre todo val y test (`scripts/v1_full.py` + `v1_analyze.py`)

AUC: **0.593 en val, 0.538 en test**. 0.42 a 0.66 s por texto en el Mac (~300 tokens de prompt, 1 token de salida).
Umbral elegido en val (objetivo A + B): P(correcta) < 0.3 (val A=0.602, B=0.599).

| Test | A | B | B en textos con error | % marcados | Correctos reconocidos |
|---|---:|---:|---:|---:|---:|
| Opción A (umbral 0.3 de val) | 0.550 | 0.544 | 0.585 | 56 % | 71 / 144 |
| Opción A (umbral fijo 0.5) | 0.557 | 0.550 | 0.639 | 61 % | 63 / 144 |

- Corrección (reusando las correcciones guardadas; 7 de 182 marcados sin corrección): ROUGE-1 oficial
  0.417 (actual 0.341), BERTScore oficial 0.492 (actual 0.479). La subida viene de que la métrica oficial da 1
  cuando un texto correcto se marca "sin error", no de mejores correcciones.
- Conclusión: no mejora A en test; el equilibrio en % marcados es aparente (elige casi al azar cuáles marcar).
  Solo gana en velocidad.

## Resumen

1. BERT resuelve el *dónde* (95 a 98 %), con o sin error.
2. El *si* necesita conocimiento médico; Qwen 7B da AUC 0.50 a 0.60 en todas las formas probadas.
3. Claude sí lo tiene (134/144 correctos reconocidos), con la advertencia de posible memorización de MedQA.

## Pendiente / ideas

- Mejorar BERT para A (en discusión): ver si más datos de entrenamiento con conocimiento médico ayudan.
- Probar un LLM más grande o médico con `v1_full.py` (ahora es rápido: ~5 min val + test).

## Experimento MedQA "hoja de respuestas" sin LLM (`scripts/medqa_sin_llm.py`, test)

Reutiliza `medec.medqa_experiment` (recuperación por embeddings sobre MedQA train + test, 11.451 preguntas)
pero sin DSPy ni Qwen: BERT elige la oración k; si k contiene la respuesta correcta de la pregunta
recuperada -> sin error; si contiene otra opción -> error y corrección = reemplazarla por la respuesta
(`fallback_correction`); si no contiene ninguna opción -> se cuenta como error, sin corrección.

- Similitud con la pregunta recuperada: mediana 0.949 (p10 0.872): casi todos los casos están en MedQA.
- Cuando la regla encuentra una opción, acierta siempre: "contiene la respuesta" 91/91 sin error;
  "contiene otra opción" 70/70 con error. 166 textos (113 con error, 53 sin error) no contienen el texto
  literal de ninguna opción (paráfrasis) y quedan como "error" por defecto.

| Test | A | B | B en textos con error | % marcados | Correctos reconocidos |
|---|---:|---:|---:|---:|---:|
| BiomedBERT solo | 0.566 | 0.554 | 0.951 | 96 % | 7 / 144 |
| MedQA sin LLM (BERT + hoja de respuestas) | 0.838 | 0.820 | 0.967 | 72 % | 91 / 144 |
| Claude Opus 5.5 | 0.948 | 0.942 | 0.951 | 57 % | 134 / 144 |

C: ROUGE-1 oficial 0.635, BERTScore oficial 0.642 (pipeline actual 0.341 / 0.479; Claude 0.799 / 0.884).

Lectura: con la respuesta recuperada y una búsqueda de texto se llega a A 0.84 sin conocimiento médico.
Confirma que los casos de test están en MedQA (fuga) y que buena parte de la tarea MS se resuelve
encontrando la respuesta, no razonando. No es un resultado válido del pipeline.

## BERT preentrenado con MedQA filtrado (`outputs/experiments/bert_medqa/`, 2026-11-08)

Idea: dar a BERT conocimiento médico practicando con preguntas MedQA que NO están en MEDEC, y luego
afinarlo con MEDEC como siempre.

- Filtro (`scripts/1_filter_medqa.py`): los embeddings no sirven (todos los casos clínicos se parecen: la
  2.ª pregunta más parecida ya tiene coseno ~0.87). TF-IDF de 1 a 3 palabras separa limpio: 1.912 textos
  MEDEC con su fuente > 0.6, ~230 < 0.2 (su fuente no está en esta copia de MedQA, probablemente en dev).
  Umbral 0.2 sobre train + val + test: se excluyen 1.301 preguntas y quedan 10.150. Textos con su fuente en
  MedQA: train 1370/1528, val 292/334, test 296/327.
- Etapa 1 (`scripts/2_train_and_score.py`): 19.286 ejemplos (caso + "pregunta? opción." -> 0 si es la
  correcta, 1 si es una incorrecta al azar), 1 época, batch 8 (con 16 hubo OOM en MPS).
  **eval_loss 0.6937 = ln 2: no aprendió nada, responde 50/50.**
- Etapa 2: afinado con MEDEC desde la etapa 1 (igual que el BERT actual). eval_loss 0.2675 (actual 0.2244).

| Modelo | AUC A val | AUC A test | B en textos con error (val) | A test | B test | % marcados | Correctos |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT actual (solo MEDEC) | 0.481 | 0.547 | 0.956 | 0.566 | 0.554 | 96 % | 7 / 144 |
| Etapa 1: solo MedQA filtrado | 0.515 | 0.496 | 0.204 | 0.440 | 0.440 | 0 % | 144 / 144 |
| Etapa 2: MedQA filtrado + MEDEC | 0.442 | 0.569 | 0.961 | 0.563 | 0.554 | 98 % | 4 / 144 |

Lectura: no mejora A (AUC sigue en torno a 0.5 en val y test); B se mantiene. Un BERT base (110 M)
no logra aprender a distinguir la opción correcta de MedQA en este formato. Modelos guardados en
`outputs/models/bert/medqa_stage{1,2}` (ignorados por git), se pueden borrar.

## Claude Haiku 5.5 como verificador de la oración de BERT (2026-11-08)

Código (sí integrado, parametrizable): `src/medec/verifier.py`, `scripts/run_verifier.py --verifier haiku|sonnet|qwen
[--compile]`, sección `verifier:` de `configs/default.yaml`. DSPy sin compilar (firma `VerifySentenceSignature`:
texto numerado + oración k de BERT -> `is_correct`, `corrected_sentence`). Llamadas síncronas en paralelo (8 hilos),
~2.5 a 3 s por llamada; sin `temperature` (Claude 5.5 rechaza parámetros de muestreo no predeterminados).
Sonnet quedó configurado pero no se corrió (decisión del usuario).

### Test (327 textos)

| Sistema | A | B | ROUGE-1 of. | BERTScore of. | Detecta con error | Reconoce correctos | Cambio clave en la corrección |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pipeline actual (BERT + Qwen) | 0.566 | 0.547 | 0.341 | 0.479 | 171/183 | 14/144 | 25/165 (15 %) |
| BERT + Haiku decide, Qwen corrige | 0.826 | 0.823 | 0.652 | 0.759 | 146/183 | 124/144 | 25/145 (17 %) |
| **BERT + Haiku decide y corrige** | **0.826** | **0.823** | **0.692** | **0.773** | 146/183 | **124/144** | 87/145 (60 %) |
| Haiku solo (prompt de Opus, todo el texto) | 0.801 | 0.771 | 0.601 | 0.714 | 166/183 | 96/144 | 100/156 (64 %) |
| Opus 5.5 solo | 0.948 | 0.942 | 0.799 | 0.884 | 176/183 | 134/144 | 124/174 (71 %) |

- Costos: verificador Haiku en test ~centavos; Haiku solo $0.13 (síncrono, el lote Batch quedó 18 min en 0/327 y
  se canceló sin cargo; ahora `run_claude_reference.py` tiene `--sync`, `--model`, `--name`).
- Val (334): BERT + Haiku A 0.835 / B 0.835 / ROUGE-1 of. 0.704 -> estable entre val y test.

### Lectura
- Haiku resuelve el cuello de botella A: reconoce 124/144 textos correctos (antes 14).
- BERT + Haiku > Haiku solo: BERT ubica mejor (B en textos con error 0.95 vs 0.85 de Haiku solo) y la pregunta
  concreta reduce falsas alarmas (124 vs 96 correctos reconocidos). Haiku solo detecta más errores (166 vs 146).
  Diferencia en A ~8 textos (podría ser ruido); en B ~17 textos (más sólida).
- Qwen como corrector no aporta: 17 % de correcciones con el cambio clave vs 60 % de Haiku.
- De los 37 errores que Haiku dejó pasar: 32 son de conocimiento de Haiku, 5 porque BERT eligió otra oración (en 4 el
  error era la 2.ª opción de BERT). Volver a ponderar con BERT no ayudaría: su confianza no separa esos casos
  (AUC 0.57 entre los textos que Haiku da por correctos).
- Compilar el verificador (BootstrapFewShot, 3 demos balanceados [F, T, F], métrica con cambio clave) no mejora en
  val: A 0.826 vs 0.835, cambio clave 58 % en ambos; no se corrió en test. Programa en `outputs/models/verifier_haiku.json`.
  Nota técnica: BootstrapFewShot y `load()` devuelven el programa sin LM; hay que hacer `set_lm` otra vez.
- Advertencia: Haiku y Opus pudieron ver MedQA (fuente de los textos MS) en su entrenamiento.
