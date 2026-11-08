"""Step 6: compare every system available in predictions/ and write the reports."""

import json

import pandas as pd

from _common import parse_args

from medec.config import output_path
from medec.data import load_split
from medec.evaluation import compare_correctors, compare_detectors, error_analysis
from medec.schemas import load_predictions

SYSTEMS = {
    "baseline_last_sentence": "Baseline: siempre error en la última oración",
    "bert": "BiomedBERT fine-tuneado (w = 1)",
    "dspy_zeroshot": "Qwen2.5-7B + DSPy sin compilar",
    "dspy": "Qwen2.5-7B + DSPy compilado (w = 0)",
    "combined": "Combinación ponderada",
    "pipeline": "Pipeline completo (combinación + corrector)",
    "claude": "Referencia: Claude Opus 5.5",
    "pipeline_medqa": "Experimento: pipeline + pista de MedQA",
}
CORRECTORS = ["pipeline", "claude", "pipeline_medqa"]


def extra(parser):
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore (faster)")


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True, extra=extra)
    splits_dir = output_path(cfg, "splits", "x").parent
    bs_model = None if args.no_bertscore else cfg["evaluation"]["bertscore_model"]
    for split in args.splits:
        gold = load_split(splits_dir, split)
        systems = {}
        for key, label in SYSTEMS.items():
            path = output_path(cfg, "predictions", f"{key}_{split}.jsonl")
            if path.exists():
                systems[label] = load_predictions(path)
        if not systems:
            print(f"{split}: no hay predicciones")
            continue
        det = compare_detectors(gold, systems)
        corr_systems = {SYSTEMS[k]: systems[SYSTEMS[k]] for k in CORRECTORS if SYSTEMS[k] in systems}
        corr = compare_correctors(gold, corr_systems, bs_model) if corr_systems else pd.DataFrame()

        report = {"split": split, "n_texts": len(gold), "detection": det.reset_index().to_dict("records"),
                  "correction": corr.reset_index().to_dict("records") if len(corr) else []}
        pipe_label = SYSTEMS["pipeline"] if SYSTEMS["pipeline"] in systems else SYSTEMS.get("combined")
        if pipe_label in systems:
            analysis = error_analysis(gold, systems[pipe_label])
            report["error_analysis"] = {k: v.reset_index().to_dict("records") for k, v in analysis.items()}
            report["doubtful_corrections"] = sum(p.doubtful for p in systems[pipe_label])

        out = output_path(cfg, "reports", f"metrics_{split}.json")
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
        md = [f"## Resultados en {split} ({len(gold)} textos)", "", "### Detección (A y B)", "", det.to_markdown()]
        if len(corr):
            md += ["", "### Corrección (C)", "", corr.to_markdown()]
        output_path(cfg, "reports", f"results_{split}.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))


if __name__ == "__main__":
    main()
