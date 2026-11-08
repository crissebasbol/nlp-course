"""Step 3b: compile the DSPy detector on train texts and score the val/test texts.

Needs the LLM server running (llama-server with Qwen2.5-7B, see README).
Results are appended line by line, so an interrupted run resumes where it stopped.
"""

import random

from _common import parse_args

from medec.config import output_path
from medec.data import load_split
from medec.detector_dspy import (
    build_dspy_detector, compile_dspy_detector, configure_dspy_lm, load_compiled_program,
    predict_split_dspy, save_compiled_program, to_dspy_example,
)
from medec.llm_server import wait_until_ready
from medec.schemas import append_jsonl, load_jsonl


def extra(parser):
    parser.add_argument("--no-compile", action="store_true", help="Zero-shot detector (no DSPy optimization)")
    parser.add_argument("--recompile", action="store_true", help="Compile again even if the program exists")


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True, extra=extra)
    dcfg, mcfg = cfg["dspy"], cfg["models"]
    wait_until_ready(mcfg["llm_api_base"], timeout_s=60)
    configure_dspy_lm(mcfg["llm_api_base"], mcfg["llm_name"], dcfg["temperature"], dcfg["max_tokens"], dcfg["timeout"])
    splits_dir = output_path(cfg, "splits", "x").parent

    name = "dspy_zeroshot" if args.no_compile else "dspy"
    program_path = output_path(cfg, "models", "dspy_detector.json")
    if args.no_compile:
        program = build_dspy_detector()
    elif program_path.exists() and not args.recompile:
        program = load_compiled_program(program_path)
        print("Programa compilado cargado de", program_path)
    else:
        train = load_split(splits_dir, "train")
        rows = list(train.itertuples())
        random.Random(cfg["seed"]).shuffle(rows)
        trainset = [to_dspy_example(r) for r in rows[: dcfg["compile_train_size"]]]
        program = compile_dspy_detector(trainset, dcfg["max_bootstrapped_demos"], dcfg["max_labeled_demos"])
        save_compiled_program(program, program_path)
        print(f"Compilado con {len(trainset)} textos de train -> {program_path}")

    for split in args.splits:
        df = load_split(splits_dir, split)
        if args.limit:
            df = df.head(args.limit)
        path = output_path(cfg, "scores", f"{name}_{split}.jsonl")
        done = {r["text_id"] for r in load_jsonl(path)} if path.exists() else set()
        new = predict_split_dspy(program, df, dcfg["num_threads"], done, on_result=lambda r: append_jsonl(r, path))
        failed = sum(r["failed"] for r in new)
        print(f"{split}: {len(new)} nuevos, {len(done)} ya hechos, {failed} fallidos -> {path}")


if __name__ == "__main__":
    main()
