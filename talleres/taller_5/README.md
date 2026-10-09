# Taller 5: Clasificación multiclase con RNN (LSTM bidireccional)

## Objetivo

Entrenar una red neuronal recurrente para clasificar textos en español sobre salud mental, ajustar sus hiperparámetros y analizar de forma comparativa cómo cada uno afecta el desempeño del modelo.

**Multiclase** con 7 categorías:

`Anxiety`, `Normal`, `Depression`, `Suicidal`, `Stress`, `Bipolar`, `Personality disorder`.

## Datos

- Archivo: `Combined_Data_es-ES.csv` (53 043 registros).
- Columna de texto usada: `statement_es_es`. Columna de etiqueta: `status`.

## Arquitectura

```
TextVectorization(max_tokens=V)
  -> Embedding(V, d, mask_zero=True)
  -> Bidirectional(LSTM(u))
  -> Dense(u, relu)
  -> Dense(7, softmax)
```

- Pérdida: `SparseCategoricalCrossentropy` (etiquetas enteras, salida softmax).
- Optimizador: Adam, tasa de aprendizaje `1e-4`.
- Épocas: 5 en todos los modelos.
- Partición: subconjunto estratificado del dataset y luego 80 % entrenamiento / 20 % prueba (`random_state=42`).

## Hiperparámetros evaluados

| Hiperparámetro | Valores probados |
|---|---|
| Tamaño de vocabulario (`vocab_size`) | 5 000, 8 000, 12 000 |
| Dimensión del embedding (`embedding_dim`) | 32, 64, 128 |
| Neuronas LSTM y capa densa (`rnn_units`) | 32, 64, 128 |
| Tamaño de lote (`batch_size`) | 32, 64 |
| Fracción de datos (`subset_fraction`) | 15 %, 30 %, 100 % |

Los experimentos se automatizan con la clase `RNNExperiment` del notebook.

## Resultados

| Modelo | Datos | Vocab | Embed | LSTM | Batch | Acc. prueba | Pérdida prueba |
|---|---|---|---|---|---|---|---|
| M1 (base) | 15 % | 5 000 | 64 | 64 | 64 | 53,33 % | 1,2407 |
| M2 (ligero) | 15 % | 8 000 | 32 | 32 | 64 | 52,70 % | 1,3657 |
| M3 (profundo) | 15 % | 5 000 | 128 | 128 | 32 | 57,98 % | 1,0663 |
| M4 (más datos) | 30 % | 5 000 | 64 | 64 | 64 | 55,95 % | 1,1387 |
| M5 (híbrido) | 30 % | 8 000 | 128 | 128 | 32 | 60,92 % | 1,0066 |
| **Bonus** | **100 %** | **12 000** | **128** | **128** | **32** | **70,28 %** | **0,8060** |

### Hallazgos principales

- **Reducir la capacidad (M2) empeora** exactitud y pérdida frente al modelo base: es una dirección equivocada.
- **Aumentar embedding y neuronas a 128** fue el cambio de arquitectura más efectivo (+4,65 pp en M3).
- **El volumen de datos es el factor de mayor impacto**: pasar del 30 % al 100 % sumó +9,36 pp.
- **M5 empieza a sobreajustar** (la pérdida de validación repunta en la época 5); con todos los datos (Bonus) la pérdida de validación sigue bajando.
- El vocabulario debe crecer junto con la cantidad de textos.

El análisis completo, con comparación por pares, limitaciones y referencias, está en `Taller5_RNN_IEEE.docx`.