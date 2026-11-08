"""Optional experiment: the LLM detector + corrector with the MedQA hint, on the test split.

Writes predictions/pipeline_medqa_test.jsonl; run_evaluate.py adds it to the tables.
Needs the LLM server running. Not part of the main pipeline (see README).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from _common import parse_args

from medec.config import output_path
from medec.corrector import (
    Corrector, build_correction_prompt, extract_corrected_sentence, generate_correction, load_encoder,
    load_example_index, select_few_shot_examples,
)
from medec.data import load_split, number_sentences
from medec.detector_dspy import configure_dspy_lm, load_compiled_program, predict_dspy_scores
from medec.llm_server import wait_until_ready
from medec.medqa_experiment import (
    add_medqa_context, build_medqa_index, fallback_correction, format_retrieved_options, load_medqa,
    retrieve_similar_questions,
)
from medec.quality import quality_check_correction
from medec.schemas import Prediction, append_jsonl, load_jsonl


def main() -> None:
    args, cfg = parse_args(__doc__)
    m, d = cfg["models"], cfg["dspy"]
    wait_until_ready(m["llm_api_base"], timeout_s=60)
    configure_dspy_lm(m["llm_api_base"], m["llm_name"], d["temperature"], d["max_tokens"], d["timeout"])
    program = load_compiled_program(output_path(cfg, "models", "dspy_detector.json"))
    encoder = load_encoder(m["embeddings"])

    index_path = output_path(cfg, "artifacts", "medqa_embeddings.npy")
    medqa = load_medqa("train")
    if index_path.exists():
        index = np.load(index_path)
    else:
        index = build_medqa_index(medqa["question"].tolist(), encoder)
        np.save(index_path, index)

    corrector = Corrector(load_example_index(output_path(cfg, "artifacts", "x").parent), encoder,
                          m["llm_api_base"], m["llm_name"], cfg["corrector"])
    df = load_split(output_path(cfg, "splits", "x").parent, "test")
    if args.limit:
        df = df.head(args.limit)
    out = output_path(cfg, "predictions", "pipeline_medqa_test.jsonl")
    done = {r["text_id"] for r in load_jsonl(out)} if out.exists() else set()

    def run(row) -> None:
        sentences = list(row.sentences)
        hits = retrieve_similar_questions(" ".join(sentences), index, medqa, encoder)
        hint = format_retrieved_options(hits)
        # The detector sees the hint appended as an extra, unnumbered line
        det = predict_dspy_scores(program, sentences + [f"(Reference, not part of the note) {hint}"])
        sid = det["error_sentence_id"]
        if not det["has_error"] or sid >= len(sentences):
            append_jsonl(asdict(Prediction(row.text_id, 0)), out)
            return
        examples = select_few_shot_examples(corrector.index, sentences[sid], encoder, cfg["corrector"]["k_examples"])
        messages = build_correction_prompt(add_medqa_context(number_sentences(sentences), hint), sentences[sid], examples)
        raw = generate_correction(messages, m["llm_api_base"], m["llm_name"], max_tokens=cfg["corrector"]["max_tokens"])
        correction = extract_corrected_sentence(raw) or sentences[sid]
        qc = quality_check_correction(sentences[sid], correction, cfg["quality"]["min_rouge"])
        if qc["doubtful"]:
            correction = fallback_correction(sentences[sid], hits) or correction
        append_jsonl(asdict(Prediction(row.text_id, 1, sid, correction, det["scores"][: len(sentences)], qc["doubtful"])), out)

    rows = [r for r in df.itertuples() if r.text_id not in done]
    with ThreadPoolExecutor(max_workers=d["num_threads"]) as pool:
        list(pool.map(run, rows))
    print(f"{len(rows)} textos procesados -> {out}")


if __name__ == "__main__":
    main()
