"""Steps 4 and 5: correct the flagged sentence with Qwen + few-shot, then quality check.

Reads the detection predictions of --system (default: hybrid) and writes
predictions/pipeline_<split>.jsonl with A, B and C. Needs the LLM server running.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from _common import parse_args

from medec.config import output_path
from medec.corrector import Corrector, build_example_index, load_encoder, load_example_index, save_example_index
from medec.data import load_split
from medec.llm_server import wait_until_ready
from medec.quality import quality_check_correction
from medec.schemas import Prediction, append_jsonl, load_jsonl, load_predictions, save_jsonl


def extra(parser):
    parser.add_argument("--system", default="hybrid", help="Detection predictions to correct")
    parser.add_argument("--output-name", default="pipeline")


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True, extra=extra)
    mcfg = cfg["models"]
    wait_until_ready(mcfg["llm_api_base"], timeout_s=60)
    splits_dir = output_path(cfg, "splits", "x").parent
    art_dir = output_path(cfg, "artifacts", "x").parent

    encoder = load_encoder(mcfg["embeddings"])
    if (art_dir / "fewshot_embeddings.npy").exists():
        index = load_example_index(art_dir)
    else:
        index = build_example_index(load_split(splits_dir, "train"), encoder)
        save_example_index(index, art_dir)
        print(f"Índice de ejemplos: {len(index['examples'])} correcciones de train")
    corrector = Corrector(index, encoder, mcfg["llm_api_base"], mcfg["llm_name"], cfg["corrector"])

    for split in args.splits:
        df = load_split(splits_dir, split).set_index("text_id")
        detections = load_predictions(output_path(cfg, "predictions", f"{args.system}_{split}.jsonl"))
        if args.limit:
            detections = detections[: args.limit]
        out_path = output_path(cfg, "predictions", f"{args.output_name}_{split}.jsonl")
        # Resume only texts whose saved detection (flag, sentence) still matches the current one;
        # if the detector changed, stale rows are dropped and those texts are corrected again.
        current = {d.text_id: (d.error_flag, d.error_sentence_id) for d in detections}
        saved = load_jsonl(out_path) if out_path.exists() else []
        valid = [r for r in saved if current.get(r["text_id"]) == (r["error_flag"], r["error_sentence_id"])]
        if len(valid) != len(saved):
            save_jsonl(valid, out_path)
            print(f"{split}: {len(saved) - len(valid)} correcciones viejas descartadas (cambió la detección)")
        done = {r["text_id"] for r in valid}
        todo = [d for d in detections if d.text_id not in done]

        def run(det: Prediction) -> None:
            if det.error_flag:
                sentences = list(df.loc[det.text_id, "sentences"])
                res = corrector.correct(sentences, det.error_sentence_id)
                qc = quality_check_correction(sentences[det.error_sentence_id], res["corrected_sentence"], cfg["quality"]["min_rouge"])
                det = Prediction(det.text_id, 1, det.error_sentence_id, res["corrected_sentence"], det.scores,
                                 qc["doubtful"], {"rouge_l": qc["rouge_l"], "reason": qc["reason"]})
            append_jsonl(asdict(det), out_path)

        with ThreadPoolExecutor(max_workers=cfg["dspy"]["num_threads"]) as pool:
            list(pool.map(run, todo))
        print(f"{split}: {len(todo)} textos procesados ({sum(d.error_flag for d in todo)} corregidos) -> {out_path}")


if __name__ == "__main__":
    main()
