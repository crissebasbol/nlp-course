# Procesamiento de Lenguaje Natural. 
## Pontificia Universidad Javeriana

Repositorio público con los entregables del curso de **Procesamiento de Lenguaje Natural (PLN)** de la Pontificia Universidad Javeriana. El objetivo es documentar el proceso de aprendizaje de forma que otras personas puedan usarlo como referencia o punto de partida.

## Objetivo del curso

Explorar los fundamentos y técnicas modernas del PLN: desde preprocesamiento de texto y representaciones clásicas (bolsa de palabras, TF-IDF), hasta modelos de lenguaje neurales y transformers.

## Estructura del repositorio

Cada taller o proyecto vive en su propia carpeta con entorno aislado:

```
nlp-course/
├── README.md          ← este archivo
├── .gitignore
└── talleres/
    └── <nombre-taller>/
        ├── README.md          ← descripción, instrucciones y resultados
        ├── requirements.txt   ← dependencias del taller
        ├── .venv/             ← entorno virtual (ignorado por git)
        └── src/               ← código fuente en inglés
```

## Cómo usar este repositorio

Cada carpeta de taller tiene su propio `README.md` con instrucciones específicas. El flujo general es:

```bash
cd talleres/<nombre-taller>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Notas importantes

- Todo el **código fuente está en inglés**.
- La **documentación y READMEs están en español**.
- Este repositorio es **público**: no se suben credenciales, tokens de API ni datos sensibles.
- Los entornos virtuales (`.venv/`) y archivos de configuración local están excluidos por `.gitignore`.