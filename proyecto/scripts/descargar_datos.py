"""Descarga los contratos de SECOP II y los guarda en data/raw/."""

import argparse
import logging

from secopedia.configuracion import cargar_config, ruta
from secopedia.extraccion import contar_registros, descargar_contratos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limite", type=int, default=None, help="máximo de contratos a descargar")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = cargar_config()
    print(f"Contratos que cumplen el filtro en la API: {contar_registros(config['extraccion']):,}")

    destino = ruta(config["rutas"]["crudos"])
    if args.limite:
        destino = destino.with_name(f"muestra_{args.limite}.parquet")
    paginas = destino.parent / f"paginas_{destino.stem}"

    contratos = descargar_contratos(config["extraccion"], paginas, limite=args.limite)
    contratos.to_parquet(destino, index=False)
    print(f"Guardados {len(contratos):,} contratos en {destino}")


if __name__ == "__main__":
    main()
