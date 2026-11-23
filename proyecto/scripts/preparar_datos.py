"""Limpieza y construcción de etiquetas y particiones.

Lee data/raw/, escribe data/interim/ y data/processed/, y guarda los reportes
de limpieza y distribución en reports/metrics/.

Uso:
    python scripts/preparar_datos.py [--entrada data/raw/muestra_5000.parquet]
"""

import argparse
import json

import pandas as pd

from secopedia.configuracion import cargar_config, fijar_semilla, ruta
from secopedia.etiquetas import construir_particiones, distribuciones_para_ajuste, tabla_distribucion
from secopedia.limpieza import limpiar_contratos
from secopedia.segmentos import nombre_segmento


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", default=None, help="parquet crudo (por defecto el de config.yaml)")
    args = parser.parse_args()

    config = cargar_config()
    fijar_semilla(config["semilla"])
    rutas = config["rutas"]

    contratos = pd.read_parquet(ruta(args.entrada or rutas["crudos"]))
    print(f"Contratos crudos: {len(contratos):,}")

    excluidos = config["limpieza"].get("tipos_excluidos", [])
    if excluidos:
        antes = len(contratos)
        contratos = contratos[~contratos["tipo_de_contrato"].isin(excluidos)]
        print(f"Excluidos por tipo {excluidos}: {antes - len(contratos):,}")

    limpios, conflictos, reporte = limpiar_contratos(contratos, config["limpieza"])
    limpios.to_parquet(ruta(rutas["limpios"]), index=False)
    conflictos.to_parquet(ruta(rutas["conflictos"]), index=False)
    tabla_limpieza = reporte.como_tabla()
    print("\nReporte de limpieza:\n", tabla_limpieza.to_string(index=False))
    print(f"\nTextos con segmentos distintos (evidencia de ruido): {conflictos['clave_texto'].nunique():,}")

    particiones, clases = construir_particiones(limpios, config["etiquetas"], config["semilla"])
    carpeta = ruta(rutas["procesados"] + "/x").parent
    for nombre, df in particiones.items():
        df.to_parquet(carpeta / f"{nombre}.parquet", index=False)

    distribucion = tabla_distribucion(particiones)
    distribucion.insert(0, "nombre", [nombre_segmento(c) for c in distribucion.index])
    print(f"\nClases ({len(clases)}):\n", distribucion.to_string())

    metricas = ruta(rutas["metricas"] + "/x").parent
    tabla_limpieza.to_csv(metricas / "reporte_limpieza.csv", index=False)
    distribucion.to_csv(metricas / "distribucion_clases.csv")
    ajuste = distribuciones_para_ajuste(limpios, particiones["entrenamiento"], clases, config["etiquetas"])
    with open(ruta(rutas["ajuste_distribucion"]), "w", encoding="utf-8") as archivo:
        json.dump(ajuste, archivo, ensure_ascii=False, indent=1)
    with open(metricas / "clases.json", "w", encoding="utf-8") as archivo:
        json.dump(clases, archivo, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
