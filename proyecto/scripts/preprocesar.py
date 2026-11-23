"""Aplica la normalización (etapa 4) y la lematización (etapa 5) a todas las particiones.

Agrega a cada parquet de data/processed/ las columnas:
- `texto_normalizado`: entrada de BETO.
- `lemas`: lista de lemas, entrada de TF-IDF y de los embeddings.

Uso:
    python scripts/preprocesar.py [--procesos 4]
"""

import argparse
import time

import pandas as pd

from secopedia.configuracion import cargar_config, ruta
from secopedia.normalizacion import Normalizador
from secopedia.tokenizacion import cargar_spacy, lematizar

PARTICIONES = ["entrenamiento", "validacion", "prueba", "prueba_temporal"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procesos", type=int, default=4, help="procesos paralelos de spaCy")
    args = parser.parse_args()

    config = cargar_config()
    normalizador = Normalizador.desde_archivo(ruta(config["rutas"]["lugares"]))
    nlp = cargar_spacy()
    carpeta = ruta(config["rutas"]["procesados"] + "/x").parent

    for nombre in PARTICIONES:
        inicio = time.time()
        archivo = carpeta / f"{nombre}.parquet"
        datos = pd.read_parquet(archivo)
        datos["texto_normalizado"] = datos["objeto_del_contrato"].map(normalizador)
        datos["lemas"] = list(lematizar(datos["texto_normalizado"], nlp, procesos=args.procesos))
        datos.to_parquet(archivo, index=False)
        vacios = (datos["lemas"].str.len() == 0).sum()
        print(f"{nombre}: {len(datos):,} contratos en {time.time() - inicio:.0f} s ({vacios} sin lemas)")


if __name__ == "__main__":
    main()
