from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "datos_finales.csv"
OUTPUTS_DIR = ROOT / "outputs"
DICT_DIR = ROOT / "dictionaries"

for _d in [OUTPUTS_DIR, DICT_DIR]:
    _d.mkdir(exist_ok=True)

SPACY_MODEL = "es_core_news_md"
SUBJECT_COL = "asignatura"
TOPIC_COL = "TEMA"

MIN_TOKEN_LEN = 2
TOP_N_KEYWORDS = 30
MIN_SUBJECTS_FOR_SHARED_KEYWORD = 2
MIN_SHARED_KEYWORDS_FOR_EDGE = 3
