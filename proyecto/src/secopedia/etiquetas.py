"""Etapa 3: construcción de etiquetas (segmento UNSPSC) y particiones de datos."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def extraer_segmento(codigo: str) -> str:
    """Toma los 2 primeros dígitos del código UNSPSC: 'V1.43211500' -> '43'."""
    return str(codigo).replace("V1.", "")[:2]


def elegir_clases(segmentos: pd.Series, min_ejemplos: int) -> list[str]:
    """Segmentos con al menos `min_ejemplos` contratos; el resto irá a la clase 'otros'."""
    conteo = segmentos.value_counts()
    return sorted(conteo[conteo >= min_ejemplos].index.tolist())


def asignar_clase(segmentos: pd.Series, clases: list[str], nombre_otros: str) -> pd.Series:
    return segmentos.where(segmentos.isin(clases), nombre_otros)


def construir_particiones(
    contratos: pd.DataFrame, config: dict[str, Any], semilla: int
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Devuelve las particiones {entrenamiento, validacion, prueba, prueba_temporal} y la lista de clases.

    1. Se separan primero los contratos firmados desde la fecha de corte (prueba temporal),
       para medir si el modelo generaliza a contratos nuevos.
    2. Las clases se eligen con los contratos anteriores al corte.
    3. Se aplica un tope por clase (muestreo estratificado) para controlar el desbalance.
    4. Se divide 80/10/10 de forma estratificada.
    """
    datos = contratos.copy()
    datos["segmento"] = datos["codigo_de_categoria_principal"].map(extraer_segmento)

    corte = pd.Timestamp(config["fecha_corte_temporal"])
    recientes = datos[datos["fecha_de_firma"] >= corte]
    historicos = datos[datos["fecha_de_firma"] < corte]

    clases = elegir_clases(historicos["segmento"], config["min_ejemplos_por_segmento"])
    nombre_otros = config["nombre_clase_otros"]
    clases_finales = clases + [nombre_otros]
    historicos = historicos.assign(etiqueta=asignar_clase(historicos["segmento"], clases, nombre_otros))
    recientes = recientes.assign(etiqueta=asignar_clase(recientes["segmento"], clases, nombre_otros))

    # Tope por clase: las clases pequeñas se conservan completas.
    tope = config["tope_por_clase"]
    muestra = pd.concat(
        [grupo.sample(n=min(len(grupo), tope), random_state=semilla) for _, grupo in historicos.groupby("etiqueta")],
        ignore_index=True,
    )

    proporciones = config["proporciones"]
    entrenamiento, resto = train_test_split(
        muestra,
        train_size=proporciones["entrenamiento"],
        stratify=muestra["etiqueta"],
        random_state=semilla,
    )
    fraccion_validacion = proporciones["validacion"] / (proporciones["validacion"] + proporciones["prueba"])
    validacion, prueba = train_test_split(
        resto, train_size=fraccion_validacion, stratify=resto["etiqueta"], random_state=semilla
    )

    # La prueba temporal conserva la distribución real (sin tope), como la vería la aplicación.
    n_temporal = min(config["tamano_prueba_temporal"], len(recientes))
    prueba_temporal = recientes.sample(n=n_temporal, random_state=semilla)

    particiones = {
        "entrenamiento": entrenamiento,
        "validacion": validacion,
        "prueba": prueba,
        "prueba_temporal": prueba_temporal,
    }
    return {nombre: df.reset_index(drop=True) for nombre, df in particiones.items()}, clases_finales


def tabla_distribucion(particiones: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Conteo de contratos por clase en cada partición (para el documento)."""
    conteos = {nombre: df["etiqueta"].value_counts() for nombre, df in particiones.items()}
    return pd.DataFrame(conteos).fillna(0).astype(int).sort_values("entrenamiento", ascending=False)


def distribuciones_para_ajuste(
    contratos: pd.DataFrame, entrenamiento: pd.DataFrame, clases: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    """Distribución real de clases (contratos limpios anteriores al corte temporal) y la del
    conjunto de entrenamiento (con tope por clase). Se usan para ajustar las probabilidades
    de los modelos a la frecuencia real de cada segmento (ver prediccion.AjusteDistribucion).
    """
    corte = pd.Timestamp(config["fecha_corte_temporal"])
    historicos = contratos[contratos["fecha_de_firma"] < corte]
    segmentos = historicos["codigo_de_categoria_principal"].map(extraer_segmento)
    etiquetas = asignar_clase(segmentos, [c for c in clases if c != config["nombre_clase_otros"]],
                              config["nombre_clase_otros"])
    real = etiquetas.value_counts(normalize=True).reindex(clases).fillna(0)
    en_entrenamiento = entrenamiento["etiqueta"].value_counts(normalize=True).reindex(clases).fillna(0)
    return {
        "clases": clases,
        "p_real": [round(float(v), 6) for v in real],
        "p_entrenamiento": [round(float(v), 6) for v in en_entrenamiento],
        "fecha_corte": config["fecha_corte_temporal"],
    }

