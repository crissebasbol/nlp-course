"""Pruebas de las etapas 2 y 3 con contratos de juguete."""

import pandas as pd

from secopedia.etiquetas import construir_particiones, extraer_segmento
from secopedia.limpieza import clave_texto, limpiar_contratos

CONFIG_LIMPIEZA = {
    "min_palabras": 3,
    "textos_vacios": ["no definido", "sin descripcion"],
    "patron_codigo": r"^V1\.\d{8}$",
}


def _contratos(filas):
    df = pd.DataFrame(filas, columns=["objeto_del_contrato", "codigo_de_categoria_principal", "fecha_de_firma"])
    df["fecha_de_firma"] = pd.to_datetime(df["fecha_de_firma"])
    return df


def test_extraer_segmento():
    assert extraer_segmento("V1.43211500") == "43"


def test_clave_texto_quita_tildes_y_signos():
    assert clave_texto("ADQUISICIÓN de equipos; TECNOLÓGICOS") == "adquisicion de equipos tecnologicos"


def test_limpieza_filtra_vacios_codigos_y_duplicados():
    contratos = _contratos([
        ["COMPRA DE PRODUCTOS DE CAFETERIA", "V1.50201700", "2023-01-01"],
        ["Compra de productos de cafetería", "V1.50201700", "2023-02-01"],  # duplicado
        ["No definido", "V1.50201700", "2023-01-01"],  # vacío
        ["compra de computadores portatiles", "UNSPECIFIED", "2023-01-01"],  # código inválido
        ["mantenimiento de vias urbanas", "V1.72141000", "2023-01-01"],
        ["mantenimiento de vias urbanas", "V1.72141000", "2023-03-01"],  # duplicado
        ["mantenimiento de vias urbanas", "V1.81101500", "2023-04-01"],  # conflicto 72 vs 81
    ])
    limpios, conflictos, reporte = limpiar_contratos(contratos, CONFIG_LIMPIEZA)
    assert len(limpios) == 2
    assert set(limpios["segmento"]) == {"50", "72"}  # gana el segmento más frecuente
    assert conflictos["clave_texto"].nunique() == 1
    assert reporte.como_tabla()["eliminados"].sum() == 5


def test_particiones_separa_prueba_temporal_y_agrupa_otros():
    filas = []
    for i in range(200):
        filas.append([f"compra de equipos numero {i}", "V1.43211500", "2023-01-01"])
        filas.append([f"obra de vias numero {i}", "V1.72141000", "2023-01-01"])
    for i in range(10):
        filas.append([f"compra de muebles numero {i}", "V1.56101500", "2023-01-01"])
    for i in range(30):
        filas.append([f"compra reciente numero {i}", "V1.43211500", "2025-06-01"])
    contratos = _contratos(filas)
    config = {
        "min_ejemplos_por_segmento": 100,
        "nombre_clase_otros": "otros",
        "tope_por_clase": 150,
        "fecha_corte_temporal": "2025-01-01",
        "tamano_prueba_temporal": 1000,
        "proporciones": {"entrenamiento": 0.8, "validacion": 0.1, "prueba": 0.1},
    }
    particiones, clases = construir_particiones(contratos, config, semilla=42)
    assert clases == ["43", "72", "otros"]
    assert len(particiones["prueba_temporal"]) == 30
    total = sum(len(particiones[p]) for p in ("entrenamiento", "validacion", "prueba"))
    assert total == 150 + 150 + 10  # tope en las clases grandes, "otros" completa
    assert (particiones["entrenamiento"]["fecha_de_firma"] < "2025-01-01").all()
