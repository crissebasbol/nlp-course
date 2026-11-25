"""Entrena y evalúa BETO (modelo 7c) localmente.

Usa la GPU si está disponible (CUDA o Apple Silicon con MPS); si no, la CPU.
En un Apple M4 Pro tarda alrededor de una hora con 3 épocas.

Guarda:
- models/beto_secopedia/                  modelo y tokenizador afinados
- reports/metrics/beto.json               métricas en prueba y prueba temporal
- reports/metrics/beto_reporte_clases.csv métricas por clase
- reports/metrics/beto_probabilidades_*.parquet  probabilidades (para el análisis de errores)
- reports/figures/matriz_confusion_beto.png

Uso:
    python scripts/entrenar_beto.py [--epocas 3] [--limite 2000]
    python scripts/entrenar_beto.py --solo-evaluar   # evalúa el modelo ya guardado
"""

import argparse
import json
import time

import pandas as pd
import torch

from secopedia.configuracion import cargar_config, fijar_semilla, ruta
from secopedia.evaluacion import calcular_metricas, graficar_matriz_confusion, reporte_por_clase
from secopedia.modelos.beto import entrenar_beto, probabilidades_beto

PARTICIONES = ["entrenamiento", "validacion", "prueba", "prueba_temporal"]


def dispositivo_disponible() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps (GPU de Apple Silicon)"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=3)
    parser.add_argument("--tasa-aprendizaje", type=float, default=3e-5)
    parser.add_argument("--tamano-lote", type=int, default=32)
    parser.add_argument("--limite", type=int, default=None,
                        help="usar solo n contratos por partición (prueba rápida del script)")
    parser.add_argument("--solo-evaluar", action="store_true", help="no entrenar: evaluar el modelo guardado")
    args = parser.parse_args()

    config = cargar_config()
    semilla = config["semilla"]
    fijar_semilla(semilla)
    carpeta = ruta(config["rutas"]["procesados"] + "/x").parent
    metricas_dir = ruta(config["rutas"]["metricas"] + "/x").parent
    figuras_dir = ruta(config["rutas"]["figuras"] + "/x").parent
    sufijo = f"_prueba{args.limite}" if args.limite else ""
    salida = ruta(config["rutas"]["modelos"] + f"/beto_secopedia{sufijo}/x").parent
    clases = json.loads((metricas_dir / "clases.json").read_text(encoding="utf-8"))

    datos = {n: pd.read_parquet(carpeta / f"{n}.parquet") for n in PARTICIONES}
    if args.limite:
        datos = {n: df.sample(n=min(args.limite, len(df)), random_state=semilla) for n, df in datos.items()}
    print(f"Dispositivo: {dispositivo_disponible()}")
    for nombre, df in datos.items():
        print(f"{nombre}: {len(df):,} contratos")

    archivo_metricas = metricas_dir / f"beto{sufijo}.json"
    if args.solo_evaluar:
        anterior = json.loads(archivo_metricas.read_text(encoding="utf-8")) if archivo_metricas.exists() else {}
        resumen = {k: anterior[k] for k in ("epocas", "tasa_aprendizaje", "minutos_entrenamiento") if k in anterior}
    else:
        inicio = time.time()
        trainer, tokenizador = entrenar_beto(
            datos["entrenamiento"], datos["validacion"], clases,
            carpeta_salida=salida / "checkpoints",
            epocas=args.epocas, tasa_aprendizaje=args.tasa_aprendizaje,
            tamano_lote=args.tamano_lote, semilla=semilla,
        )
        minutos_entrenamiento = (time.time() - inicio) / 60
        print(f"Entrenamiento: {minutos_entrenamiento:.1f} minutos")
        trainer.save_model(str(salida))
        tokenizador.save_pretrained(str(salida))
        resumen = {"epocas": args.epocas, "tasa_aprendizaje": args.tasa_aprendizaje,
                   "minutos_entrenamiento": round(minutos_entrenamiento, 1)}
        del trainer  # libera memoria de la GPU antes de evaluar

    for particion in ["validacion", "prueba", "prueba_temporal"]:
        inicio = time.time()
        probabilidades = probabilidades_beto(str(salida), datos[particion], clases)
        milisegundos = 1000 * (time.time() - inicio) / len(datos[particion])
        resumen[particion] = {**calcular_metricas(datos[particion]["etiqueta"], probabilidades, clases),
                              "ms_por_contrato": round(milisegundos, 3)}
        tabla = pd.DataFrame(probabilidades, columns=clases)
        tabla.insert(0, "id_contrato", datos[particion]["id_contrato"].to_numpy())
        tabla.to_parquet(metricas_dir / f"beto_probabilidades_{particion}{sufijo}.parquet", index=False)
        if particion == "prueba":
            predichos = pd.Series(clases).iloc[probabilidades.argmax(axis=1)].to_numpy()
            reporte_por_clase(datos[particion]["etiqueta"], predichos, clases).to_csv(
                metricas_dir / f"beto_reporte_clases{sufijo}.csv")
            graficar_matriz_confusion(datos[particion]["etiqueta"], predichos, clases,
                                      "Matriz de confusión: BETO (prueba)",
                                      figuras_dir / f"matriz_confusion_beto{sufijo}.png")

    archivo_metricas.write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    print(json.dumps(resumen, indent=1))


if __name__ == "__main__":
    main()
