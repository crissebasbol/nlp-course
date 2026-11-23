"""Entrena y evalúa la red neuronal (modelo 7b): CNN 1D sobre embeddings fastText.

Pasos:
1. Corpus para fastText: lemas de todos los contratos limpios anteriores a 2026 que no están
   en validación ni prueba (no hace falta etiqueta: fastText es no supervisado). Se lematizan
   una sola vez y se guardan en data/interim/.
2. Embeddings fastText (gensim) y vecinos de algunas palabras, para el documento.
3. CNN con parada temprana según F1 macro en validación.
4. Evaluación en prueba y prueba temporal, igual que los otros modelos.

Uso:
    python scripts/entrenar_red_neuronal.py
"""

import json
import time

import numpy as np
import pandas as pd

from secopedia.configuracion import cargar_config, fijar_semilla, ruta
from secopedia.evaluacion import calcular_metricas, graficar_matriz_confusion, reporte_por_clase
from secopedia.modelos.red_neuronal import (ClasificadorCNN, codificar, construir_vocabulario, entrenar_cnn,
                                            entrenar_fasttext, matriz_embeddings)
from secopedia.normalizacion import Normalizador
from secopedia.tokenizacion import cargar_spacy, lematizar

PARTICIONES = ["entrenamiento", "validacion", "prueba", "prueba_temporal"]
PALABRAS_EJEMPLO = ["computador", "combustible", "mobiliario", "poliza", "vial", "medicamento", "refrigerio"]


def corpus_fasttext(config, datos: dict[str, pd.DataFrame]) -> list[list[str]]:
    archivo = ruta("data/interim/corpus_fasttext.parquet")
    if archivo.exists():
        return pd.read_parquet(archivo)["lemas"].map(list).tolist()

    limpios = pd.read_parquet(ruta(config["rutas"]["limpios"]))
    corte = pd.Timestamp(config["etiquetas"]["fecha_corte_temporal"])
    excluidos = set().union(*(set(datos[n]["id_contrato"]) for n in ["validacion", "prueba", "prueba_temporal"]))
    ya_lematizados = set(datos["entrenamiento"]["id_contrato"])
    pendientes = limpios[(limpios["fecha_de_firma"] < corte) & ~limpios["id_contrato"].isin(excluidos | ya_lematizados)]
    print(f"Lematizando {len(pendientes):,} contratos extra (sin etiqueta) para fastText...")

    normalizador = Normalizador.desde_archivo(ruta(config["rutas"]["lugares"]))
    nlp = cargar_spacy()
    extra = list(lematizar(pendientes["objeto_del_contrato"].map(normalizador), nlp, procesos=4))
    documentos = datos["entrenamiento"]["lemas"].map(list).tolist() + extra
    pd.DataFrame({"lemas": documentos}).to_parquet(archivo, index=False)
    return documentos


def main() -> None:
    config = cargar_config()
    semilla = config["semilla"]
    fijar_semilla(semilla)
    carpeta = ruta(config["rutas"]["procesados"] + "/x").parent
    metricas_dir = ruta(config["rutas"]["metricas"] + "/x").parent
    figuras_dir = ruta(config["rutas"]["figuras"] + "/x").parent
    modelos_dir = ruta(config["rutas"]["modelos"] + "/red_neuronal/x").parent
    clases = json.loads((metricas_dir / "clases.json").read_text(encoding="utf-8"))
    indice_clase = {c: i for i, c in enumerate(clases)}

    datos = {n: pd.read_parquet(carpeta / f"{n}.parquet") for n in PARTICIONES}
    documentos = {n: df["lemas"].map(list).tolist() for n, df in datos.items()}
    y = {n: df["etiqueta"].map(indice_clase).to_numpy() for n, df in datos.items()}

    # 1 y 2. Embeddings fastText.
    corpus = corpus_fasttext(config, datos)
    inicio = time.time()
    fasttext = entrenar_fasttext(corpus, semilla=semilla)
    print(f"fastText: {len(corpus):,} documentos, {len(fasttext.wv):,} palabras, {time.time() - inicio:.0f} s")
    fasttext.save(str(modelos_dir / "fasttext.model"))
    vecinos = {p: [(v, round(float(s), 3)) for v, s in fasttext.wv.most_similar(p, topn=6)] for p in PALABRAS_EJEMPLO}
    for palabra, lista in vecinos.items():
        print(f"  {palabra}: {[v for v, _ in lista]}")
    (metricas_dir / "fasttext_vecinos.json").write_text(json.dumps(vecinos, ensure_ascii=False, indent=1), encoding="utf-8")

    # 3. CNN.
    vocabulario = construir_vocabulario(documentos["entrenamiento"])
    embeddings = matriz_embeddings(vocabulario, fasttext)
    x = {n: codificar(docs, vocabulario) for n, docs in documentos.items()}
    print(f"Vocabulario de la CNN: {len(vocabulario):,} lemas")
    inicio = time.time()
    modelo, historial = entrenar_cnn(x["entrenamiento"], y["entrenamiento"], x["validacion"], y["validacion"],
                                     embeddings, len(clases), semilla=semilla)
    minutos = (time.time() - inicio) / 60
    clasificador = ClasificadorCNN(modelo, vocabulario, clases)
    clasificador.guardar(modelos_dir / "cnn.pt")

    # 4. Evaluación.
    resumen = {"minutos_entrenamiento": round(minutos, 1), "historial": historial}
    for particion in ["validacion", "prueba", "prueba_temporal"]:
        textos = datos[particion]["lemas"].map(" ".join)
        inicio = time.time()
        probabilidades = clasificador.predict_proba(textos)
        milisegundos = 1000 * (time.time() - inicio) / len(textos)
        resumen[particion] = {**calcular_metricas(datos[particion]["etiqueta"], probabilidades, clases),
                              "ms_por_contrato": round(milisegundos, 3)}
        if particion == "prueba":
            predichos = np.asarray(clases)[probabilidades.argmax(axis=1)]
            reporte_por_clase(datos[particion]["etiqueta"], predichos, clases).to_csv(
                metricas_dir / "red_neuronal_reporte_clases.csv")
            graficar_matriz_confusion(datos[particion]["etiqueta"], predichos, clases,
                                      "Matriz de confusión: CNN + fastText (prueba)",
                                      figuras_dir / "matriz_confusion_red_neuronal.png")
    (metricas_dir / "red_neuronal.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in resumen.items() if k != "historial"}, indent=1))


if __name__ == "__main__":
    main()
