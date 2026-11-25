"""Curva de aprendizaje: ¿mejoran los modelos con más datos de entrenamiento?

Se varía el tope de contratos por clase (500 ... sin tope) usando todos los contratos limpios
anteriores a 2026 que no están en validación ni prueba. Para cada tamaño se entrenan la regresión
logística (TF-IDF) y la CNN (fastText) con los mismos hiperparámetros, y se evalúan en la misma
prueba y prueba temporal.

Como al subir el tope cambia la distribución de clases del entrenamiento, las probabilidades se
ajustan a la distribución del conjunto evaluado (corrección por regla de Bayes), para que la
comparación entre tamaños sea justa.

Uso:
    python scripts/curva_aprendizaje.py
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from secopedia.configuracion import cargar_config, fijar_semilla, ruta
from secopedia.etiquetas import asignar_clase, extraer_segmento
from secopedia.evaluacion import calcular_metricas
from secopedia.modelos.linea_base import crear_regresion_logistica
from secopedia.modelos.red_neuronal import (codificar, construir_vocabulario, entrenar_cnn, matriz_embeddings,
                                            ClasificadorCNN)
from secopedia.normalizacion import Normalizador
from secopedia.tokenizacion import cargar_spacy, lematizar

TOPES = [500, 1000, 2000, 4000, 8000, 16000, None]


def distribucion(etiquetas: pd.Series, clases: list[str]) -> np.ndarray:
    return etiquetas.value_counts(normalize=True).reindex(clases).fillna(1e-9).to_numpy()


def ajustar(probabilidades: np.ndarray, p_entrenamiento: np.ndarray, p_objetivo: np.ndarray) -> np.ndarray:
    ajustadas = probabilidades * p_objetivo / p_entrenamiento
    return ajustadas / ajustadas.sum(axis=1, keepdims=True)


def conjunto_completo(config, datos, clases) -> pd.DataFrame:
    """Todos los contratos anteriores al corte que no están en validación ni prueba, con lemas."""
    archivo = ruta("data/interim/conjunto_completo.parquet")
    if archivo.exists():
        return pd.read_parquet(archivo)
    limpios = pd.read_parquet(ruta(config["rutas"]["limpios"]))
    corte = pd.Timestamp(config["etiquetas"]["fecha_corte_temporal"])
    excluidos = set().union(*(set(datos[n]["id_contrato"]) for n in ["validacion", "prueba", "prueba_temporal"]))
    conjunto = limpios[(limpios["fecha_de_firma"] < corte) & ~limpios["id_contrato"].isin(excluidos)].copy()
    otros = config["etiquetas"]["nombre_clase_otros"]
    segmentos = conjunto["codigo_de_categoria_principal"].map(extraer_segmento)
    conjunto["etiqueta"] = asignar_clase(segmentos, [c for c in clases if c != otros], otros)

    # Se reutilizan los lemas ya calculados del entrenamiento y se lematiza el resto.
    lemas = dict(zip(datos["entrenamiento"]["id_contrato"], datos["entrenamiento"]["lemas"].map(list)))
    pendientes = conjunto[~conjunto["id_contrato"].isin(lemas)]
    print(f"Lematizando {len(pendientes):,} contratos...")
    normalizador = Normalizador.desde_archivo(ruta(config["rutas"]["lugares"]))
    nuevos = lematizar(pendientes["objeto_del_contrato"].map(normalizador), cargar_spacy(), procesos=4)
    lemas.update(zip(pendientes["id_contrato"], nuevos))
    conjunto["lemas"] = conjunto["id_contrato"].map(lemas)
    conjunto = conjunto[["id_contrato", "etiqueta", "lemas"]].reset_index(drop=True)
    conjunto.to_parquet(archivo, index=False)
    return conjunto


def main() -> None:
    config = cargar_config()
    semilla = config["semilla"]
    fijar_semilla(semilla)
    carpeta = ruta(config["rutas"]["procesados"] + "/x").parent
    metricas_dir = ruta(config["rutas"]["metricas"] + "/x").parent
    clases = json.loads((metricas_dir / "clases.json").read_text(encoding="utf-8"))
    indice = {c: i for i, c in enumerate(clases)}
    datos = {n: pd.read_parquet(carpeta / f"{n}.parquet")
             for n in ["entrenamiento", "validacion", "prueba", "prueba_temporal"]}
    completo = conjunto_completo(config, datos, clases)
    print(f"Contratos disponibles para entrenar: {len(completo):,}")

    ajuste = json.loads(ruta(config["rutas"]["ajuste_distribucion"]).read_text(encoding="utf-8"))
    objetivos = {"prueba": distribucion(datos["prueba"]["etiqueta"], clases),
                 "prueba_temporal": np.asarray(ajuste["p_real"])}

    from gensim.models import FastText
    fasttext = FastText.load(str(ruta(config["rutas"]["modelos"]) / "red_neuronal" / "fasttext.model"))

    filas = []
    for tope in TOPES:
        if tope is None:
            entrenamiento = completo
        else:
            entrenamiento = pd.concat(
                [g.sample(n=min(len(g), tope), random_state=semilla) for _, g in completo.groupby("etiqueta")],
                ignore_index=True)
        p_entrenamiento = distribucion(entrenamiento["etiqueta"], clases)
        textos = entrenamiento["lemas"].map(" ".join)
        etiqueta_tope = "sin tope" if tope is None else f"{tope:,}"
        print(f"\n=== Tope {etiqueta_tope}: {len(entrenamiento):,} contratos")

        # Regresión logística (C=4, el mejor en validación con el tope actual).
        inicio = time.time()
        modelo = crear_regresion_logistica(c=4.0, semilla=semilla).fit(textos, entrenamiento["etiqueta"])
        resultados = {"TF-IDF + Reg. logística": (lambda df: modelo.predict_proba(df["lemas"].map(" ".join)),
                                                  time.time() - inicio)}

        # CNN con los mismos embeddings fastText.
        inicio = time.time()
        vocabulario = construir_vocabulario(entrenamiento["lemas"].tolist())
        x_val = codificar(datos["validacion"]["lemas"].tolist(), vocabulario)
        red, _ = entrenar_cnn(codificar(entrenamiento["lemas"].tolist(), vocabulario),
                              entrenamiento["etiqueta"].map(indice).to_numpy(), x_val,
                              datos["validacion"]["etiqueta"].map(indice).to_numpy(),
                              matriz_embeddings(vocabulario, fasttext), len(clases), semilla=semilla)
        cnn = ClasificadorCNN(red, vocabulario, clases)
        resultados["CNN + fastText"] = (lambda df: cnn.predict_proba(df["lemas"].map(" ".join)), time.time() - inicio)

        for nombre, (predecir, segundos) in resultados.items():
            for particion, objetivo in objetivos.items():
                probabilidades = ajustar(predecir(datos[particion]), p_entrenamiento, objetivo)
                m = calcular_metricas(datos[particion]["etiqueta"], probabilidades, clases)
                filas.append({"modelo": nombre, "tope": etiqueta_tope, "contratos": len(entrenamiento),
                              "particion": particion, "f1_macro": m["f1_macro"],
                              "top3_accuracy": m["top3_accuracy"], "accuracy": m["accuracy"],
                              "segundos": round(segundos, 1)})
                print(filas[-1])

    tabla = pd.DataFrame(filas)
    tabla.to_csv(metricas_dir / "curva_aprendizaje.csv", index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for eje, particion, titulo in zip(ejes, ["prueba", "prueba_temporal"], ["Prueba", "Prueba temporal (2026)"]):
        for nombre, grupo in tabla[tabla["particion"] == particion].groupby("modelo"):
            eje.plot(grupo["contratos"], grupo["f1_macro"], marker="o", label=nombre)
        eje.set_xscale("log")
        eje.set_xlabel("Contratos de entrenamiento (escala log)")
        eje.set_ylabel("F1 macro")
        eje.set_title(titulo)
        eje.grid(True, which="both", alpha=0.3)
    ejes[0].legend()
    plt.tight_layout()
    fig.savefig(ruta(config["rutas"]["figuras"]) / "curva_aprendizaje.png", dpi=150)
    print(tabla.pivot_table(index=["tope", "contratos"], columns=["particion", "modelo"], values="f1_macro"))


if __name__ == "__main__":
    main()
