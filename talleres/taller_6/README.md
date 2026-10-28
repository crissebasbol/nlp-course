# Taller 6: Generación de texto y traducción automática con RNN

## Objetivo

Aplicar redes neuronales recurrentes (GRU) a dos tareas clásicas del PLN:

1. **Generación de texto:** entrenar un modelo de lenguaje a nivel de carácter con un libro leído por el grupo y generar texto nuevo con su estilo.
2. **Traducción automática:** entrenar un modelo seq2seq (codificador-decodificador) que traduzca del **español al inglés** con el dataset *Combined Data es-ES*, y evaluarlo con **BLEU**, **ROUGE** y **F1-score**.

## Archivos

| Archivo | Descripción |
|---|---|
| `taller_6.ipynb` | Notebook con el desarrollo completo de ambos puntos, resultados y conclusiones. |
| `requirements.txt` | Dependencias para ejecutar el notebook fuera de Colab. |

## Cómo ejecutarlo

El notebook está pensado para **Google Colab con GPU**:

1. Abrir `taller_6.ipynb` en Colab: [abrir en Colab](https://colab.research.google.com/github/crissebasbol/nlp-course/blob/master/talleres/taller_6/taller_6.ipynb).
2. Ir a **Entorno de ejecución > Cambiar tipo de entorno de ejecución** y seleccionar **GPU (T4)**.
3. **Punto 1:** no requiere archivos; el libro se descarga automáticamente de Project Gutenberg.
4. **Punto 2:** subir `Combined Data es-ES.csv`
5. Ejecutar las celdas en orden.

Para ejecutarlo en local:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook taller_6.ipynb
```

En local hay que cambiar la ruta del CSV en la celda de carga del Punto 2. Sin GPU, el entrenamiento es considerablemente más lento.

## Punto 1: Generación de texto con *Don Quijote de la Mancha*

### Datos

- **Libro:** *El ingenioso hidalgo don Quijote de la Mancha*, de Miguel de Cervantes ([Project Gutenberg #2000](https://www.gutenberg.org/ebooks/2000)).
- **Limpieza:** se eliminaron el encabezado y el pie de licencia de Gutenberg, se normalizaron los saltos de línea de Windows y se compactaron las líneas en blanco. Quedan **2 110 296 caracteres**.
- **Tokenización** por carácter: 91 caracteres + 2 tokens especiales = **93 tokens**.
- **Secuencias** de 100 caracteres; la etiqueta es la entrada desplazada un carácter. Se reservó el **10 % final** para validación (18 993 secuencias de entrenamiento y 2 110 de validación).

### Arquitectura

```
Input(100) -> Embedding(93, 256) -> GRU(1024, return_sequences) -> Dropout(0.1) -> Dense(93, softmax)
```

~4.06M parámetros. Entrenamiento con Adam, `EarlyStopping` (paciencia 2) y `ModelCheckpoint`.

### Resultados

| Variante | Parámetros | Mejor época | val_loss | Perplejidad | val_accuracy |
|---|---|---|---|---|---|
| Base: GRU 1024, dropout 0.1 | 4.06M | 8 | 1.2443 | 3.47 | 60.9 % |
| GRU 1024, dropout 0.3 | 4.06M | 9 | 1.2420 | 3.46 | 61.0 % |
| **2 GRU 512 apiladas, dropout 0.1** | **2.83M** | 14 | **1.2351** | **3.44** | **61.3 %** |

### Hallazgos principales

- El modelo aprendió **ortografía, acentuación y puntuación del español** (`¿?`, `¡!`, diálogos con raya) y el **vocabulario del Quijote** (*caballero andante*, *rucio*, *vuestra merced*, *cueva de Montesinos*), sin memorizar el texto.
- La **decodificación voraz** cae en ciclos repetitivos. Una **temperatura de 0.5 a 0.8** da el mejor equilibrio entre coherencia y variedad.
- **Dos GRU apiladas de 512** superan a una GRU de 1024 con 30 % menos parámetros y sobreajustan menos por época.
- Las tres variantes convergen a una pérdida de ~1.24: el límite parece estar en el tamaño del corpus más que en la arquitectura.

## Punto 2: Traducción español → inglés con seq2seq

### Datos

- **Dataset:** `Combined Data es-ES.csv`, con 53 043 publicaciones sobre salud mental. Se usan `statement_es_es` (español, entrada) y `statement` (inglés, salida).
- **Limpieza:**

| Paso | Pares |
|---|---|
| Sin valores faltantes | 52 680 |
| Sin URLs | 51 538 |
| Entre 1 y 20 palabras | 15 103 |
| Con traducción real (español distinto del inglés) | 14 233 |
| Sin duplicados | **14 080** |

- **División:** 70 / 15 / 15, es decir, 9 856 pares de entrenamiento, 2 112 de validación y 2 112 de prueba.
- **Tokenización** por palabras: vocabulario de 8 000 por idioma, 20 tokens para el español y 22 para el inglés (con `[start]` y `[end]`).

### Arquitectura

```
Codificador:   Embedding(8000, 256) -> Bidirectional(GRU(512), merge_mode="sum")
Decodificador: Embedding(8000, 256) -> GRU(512, initial_state=codificador) -> Dropout(0.5) -> Dense(8000, softmax)
```

~11.75M parámetros. Entrenamiento con *teacher forcing*, Adam y `EarlyStopping`. La mejor época fue la 10, con 30 % de precisión por palabra en validación.

### Resultados en el conjunto de prueba

| Métrica | seq2seq GRU | Línea base (copiar el español) |
|---|---|---|
| BLEU | **2.62** | 2.06 |
| ROUGE-1 F | **17.77** | 9.41 |
| ROUGE-2 F | **4.91** | 2.40 |
| ROUGE-L F | **17.08** | 9.31 |
| F1 por tokens | **17.77** | 9.49 |

- **BLEU:** `sacrebleu.corpus_bleu`.
- **ROUGE:** `rouge-score`.
- **F1:** solapamiento de tokens por frase, estilo SQuAD.

Las referencias se normalizaron igual que la salida del modelo: minúsculas y sin puntuación.

### Hallazgos principales

- El modelo **duplica a la línea base** en palabras sueltas (ROUGE-1, F1) y en bigramas (ROUGE-2), pero casi no acierta secuencias largas: la precisión de 4-gramas es de 0.5 %.
- Genera **inglés fluido pero poco fiel a la entrada**. Suele acertar las primeras palabras y luego se desvía por el **sesgo de exposición**.
- **Reproduce sesgos del dataset:** llegó a generar expresiones de ideación suicida que no estaban en la frase de entrada.
- **Causas principales:**
  - pocos datos (~10 000 pares);
  - referencias parafraseadas y ruidosas;
  - el cuello de botella de comprimir la frase en un único vector;
  - el vocabulario cerrado (12 % de palabras desconocidas en español).