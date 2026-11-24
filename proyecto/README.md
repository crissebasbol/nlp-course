# SECOPEDIA

Clasificador automático de segmentos UNSPSC para contratos públicos colombianos de SECOP II.

## Descripción breve

Cuando una entidad pública publica un proceso en SECOP II debe clasificar lo que compra con un código del catálogo UNSPSC, y lo hace a mano entre miles de opciones. SECOPEDIA recibe el texto del objeto del contrato (por ejemplo "compra de computadores portátiles para la secretaría de educación") y sugiere los 3 segmentos UNSPSC más probables con su confianza. También puede auditar contratos ya publicados y marcar los que tienen una categoría que no coincide con su descripción.

- **Usuario principal:** funcionarios de contratación de entidades públicas.
- **Usuarios secundarios:** proveedores y pymes, veedurías y entes de control.
- **Tarea de PLN:** clasificación de texto multiclase supervisada (28 segmentos más la clase "otros").

## Pipeline de PLN

| Etapa | Módulo | Qué hace |
|---|---|---|
| 1. Extracción | `src/secopedia/extraccion.py` | Descarga desde la API de datos.gov.co solo las columnas necesarias, sin columnas de datos personales |
| 2. Limpieza | `src/secopedia/limpieza.py` | Quita textos vacíos, códigos inválidos, el código por defecto 801116 (personal temporal) y duplicados |
| 3. Etiquetas | `src/secopedia/etiquetas.py` | Código UNSPSC a segmento, clase "otros", tope por clase, particiones y prueba temporal |
| 4. Normalización | `src/secopedia/normalizacion.py` | Minúsculas, sin tildes, años a `<AÑO>`, números a `<NUM>` y municipios o departamentos (DIVIPOLA) a `<LUGAR>` |
| 5. Tokenización | `src/secopedia/tokenizacion.py` | spaCy `es_core_news_sm`: tokens, stopwords del español y del dominio, lemas |
| 6. Representación | `src/secopedia/representacion.py`, `modelos/red_neuronal.py` | TF-IDF de unigramas y bigramas de lemas; embeddings fastText propios; subpalabras de BETO |
| 7. Modelos | `src/secopedia/modelos/` | Línea base TF-IDF + Regresión Logística / SVM lineal; red neuronal CNN sobre fastText; fine-tuning de BETO |
| 8. Evaluación | `src/secopedia/evaluacion.py` | Accuracy, F1 macro, F1 ponderado, top-3 accuracy y matriz de confusión |
| 9. Demo | `src/secopedia/prediccion.py`, `src/secopedia/web/` | `predecir(texto)` (top 3) y `auditar(contratos)`, expuestos en una API FastAPI con interfaz web |

## Estructura

| Carpeta | Contenido |
|---|---|
| `configs/` | `config.yaml`: rutas, filtros, umbrales, semilla y repositorio de Hugging Face |
| `src/secopedia/` | Código del pipeline (una etapa por módulo) y la demo web |
| `scripts/` | Programas que encadenan las etapas: descargar, preparar, preprocesar y entrenar |
| `notebooks/` | `01` exploración de datos, `02` ejemplos de cada transformación, `03` evaluación y análisis de errores |
| `reports/` | Métricas (`metrics/`) y figuras (`figures/`) de todos los modelos |
| `tests/` | Pruebas unitarias |

## Requerimientos

- Python 3.10 o superior (probado con 3.12).
- Conexión a internet para instalar las dependencias y descargar los modelos o los datos.
- Espacio en disco: unos 2,5 GB para la demo (entorno y modelos) y unos 5 GB para el pipeline completo.
- GPU opcional: solo se recomienda para entrenar BETO (CUDA o Apple Silicon).
- Librerías principales: pandas, spaCy, scikit-learn, gensim, PyTorch, Transformers y FastAPI. Las versiones exactas están en `requirements.txt`.

## Instalación paso a paso

```bash
cd proyecto
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m spacy download es_core_news_sm
```

## Uso

### Solo la demo (sin descargar datos ni entrenar)

Los cuatro modelos entrenados y el léxico de lugares están publicados en Hugging Face ([crissebasbol/secopedia-beto-unspsc](https://huggingface.co/crissebasbol/secopedia-beto-unspsc)). Después de la instalación basta con:

```bash
uvicorn secopedia.web.servidor:app --port 8000
```

La primera vez que se usa cada modelo, la demo lo descarga del Hub (unos 500 MB en total, la mayor parte de BETO).

### Pipeline completo

```bash
# 1. Descargar los contratos (unos 586 mil, tarda varios minutos)
python scripts/descargar_datos.py

# 2. Limpiar y construir etiquetas y particiones
python scripts/preparar_datos.py

# 3. Construir la lista de municipios y departamentos para la normalización
python scripts/construir_lugares.py

# 4. Normalizar y lematizar las particiones
python scripts/preprocesar.py

# 5. Entrenar y evaluar la línea base
python scripts/entrenar_linea_base.py

# 6. Entrenar y evaluar la red neuronal (CNN sobre fastText, unos 4 minutos)
python scripts/entrenar_red_neuronal.py

# 7. Entrenar y evaluar BETO (alrededor de 1 hora en un Apple M4 Pro)
python scripts/entrenar_beto.py

# 8. Abrir la demo: interfaz en http://localhost:8000 y documentación de la API en /docs
uvicorn secopedia.web.servidor:app --port 8000

# Pruebas
pytest
```

El entrenamiento de BETO usa la GPU si está disponible (CUDA o Apple Silicon); en CPU funciona, pero mucho más lento. La demo usa CUDA si existe y, si no, la CPU.

Los datos se guardan en `data/` y no se suben al repositorio.

## Datos

SECOP II - Contratos Electrónicos, publicado por Colombia Compra Eficiente en datos.gov.co (id `jbjy-vk9h`). Se usan los contratos firmados desde 2022, excluyendo los de prestación de servicios, cuyo código UNSPSC es casi siempre el mismo.

## Resultados

| Modelo | F1 macro (prueba) | Top-3 (prueba) | F1 macro (prueba temporal 2026) | Top-3 (prueba temporal 2026) | ms por contrato |
|---|---|---|---|---|---|
| TF-IDF + Regresión Logística | 0,651 | 0,855 | 0,536 | 0,822 | 0,01 |
| TF-IDF + SVM lineal | 0,659 | 0,845 | 0,544 | 0,820 | 0,01 |
| CNN 1D sobre fastText | 0,652 | 0,859 | 0,534 | 0,839 | 0,01 |
| BETO (fine-tuning, 3 épocas) | **0,667** | **0,871** | **0,556** | **0,853** | 6 |

Los cuatro modelos están publicados en Hugging Face: [crissebasbol/secopedia-beto-unspsc](https://huggingface.co/crissebasbol/secopedia-beto-unspsc) (BETO en la raíz y los modelos clásicos en `clasicos/`). La demo los descarga automáticamente si no se entrenaron localmente.

BETO es el mejor en todas las métricas, sobre todo en top-3 con contratos nuevos (2026): en 85 de cada 100 el segmento correcto está entre las 3 sugerencias. A cambio, es unas 600 veces más lento por contrato (6 ms frente a 0,01 ms), aunque sigue siendo instantáneo para un usuario. Se entrenó en 57 minutos en la GPU de un Apple M4 Pro.

### Ajuste por distribución de clases

Los modelos se entrenan con un tope de contratos por segmento (clases casi balanceadas), pero en la realidad el segmento 80 es mucho más frecuente que, por ejemplo, el 95. La demo ajusta las probabilidades de cada modelo a la frecuencia real de los segmentos (calculada solo con contratos anteriores a 2026, archivo `reports/metrics/ajuste_distribucion.json`). En los contratos de 2026:

| Modelo | F1 macro sin ajuste | F1 macro con ajuste | Top-3 sin ajuste | Top-3 con ajuste |
|---|---|---|---|---|
| TF-IDF + Regresión Logística | 0,536 | 0,568 | 0,822 | 0,873 |
| TF-IDF + SVM lineal | 0,544 | 0,571 | 0,820 | 0,855 |
| CNN 1D sobre fastText | 0,534 | 0,564 | 0,839 | 0,876 |
| BETO | 0,556 | **0,591** | 0,853 | **0,888** |

Se puede desactivar con `prediccion.ajustar_distribucion: false` en `configs/config.yaml`.

El análisis de errores completo está en `notebooks/03_analisis_errores.ipynb`. En una revisión preliminar de 100 errores de BETO (pendiente de validar por el grupo), solo el 11% fueron errores claros del modelo: el resto eran etiquetas mal puestas por la entidad (36%), contratos ambiguos entre dos segmentos (46%) o textos que no dicen qué se compra (7%).

### Ensamble y curva de aprendizaje

El promedio de las probabilidades de los cuatro modelos llega a F1 macro 0,680 y top-3 0,877 en prueba, un poco mejor que BETO solo. La demo usa los modelos por separado.

Con más datos de entrenamiento la mejora casi se detiene: al pasar de 105 mil a 284 mil contratos, la CNN gana menos de 1 punto de F1 y la regresión logística empeora (`scripts/curva_aprendizaje.py`, `reports/figures/curva_aprendizaje.png`).
