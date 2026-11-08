"""External reference: Claude Opus 5.5 over a split with the Message Batches API.

Costs money. First run with --dry-run (counts tokens and estimates the cost),
then without it. The API key is read from taller_8/.env.
"""

from _common import parse_args

import anthropic

from medec.claude_reference import (
    build_claude_request, collect_claude_results, count_prompt_tokens, format_example, load_api_key,
    run_claude_sync, submit_claude_batch, to_prediction_format, wait_for_batch,
)
from medec.config import PROJECT_DIR, output_path
from medec.data import load_split
from medec.schemas import load_jsonl, save_jsonl, save_predictions

# USD per million tokens (input, output), before the 50 % Batch discount
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def approx_prompt_tokens(request) -> float:
    """Offline estimate: ~3.5 characters per token, plus the JSON schema and message overhead."""
    params = request["params"]
    chars = len(params["system"]) + sum(len(m["content"]) for m in params["messages"]) + len(str(params["output_config"]))
    return chars / 3.5 + 50


def extra(parser):
    parser.add_argument("--dry-run", action="store_true", help="Only count tokens and estimate the cost")
    parser.add_argument("--batch-id", default=None, help="Collect an already submitted batch")
    parser.add_argument("--est-output-tokens", type=int, default=1000, help="Output tokens per text (thinking included)")
    parser.add_argument("--sync", action="store_true", help="Normal parallel calls instead of the Batch API (2x price, minutes)")
    parser.add_argument("--model", default=None, help="Override claude.model (e.g. claude-haiku-5-5)")
    parser.add_argument("--name", default="claude", help="Output name: claude/<name>_... and predictions/<name>_<split>.jsonl")


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True, extra=extra)
    ccfg = {**cfg["claude"], **({"model": args.model} if args.model else {})}
    PRICE_IN, PRICE_OUT = PRICES[ccfg["model"]]
    print(f"Modelo: {ccfg['model']} | salida: {args.name}")
    try:
        client = anthropic.Anthropic(api_key=load_api_key(PROJECT_DIR / ".env"))
    except RuntimeError:
        if not args.dry_run:
            raise
        client = None  # dry run without a key: tokens are approximated offline
        print("Sin ANTHROPIC_API_KEY: los tokens se aproximan como caracteres / 3.5 (estimación gruesa)")
    splits_dir = output_path(cfg, "splits", "x").parent
    example = format_example(load_split(splits_dir, "train").query("error_flag == 1").iloc[0]) if ccfg["n_examples"] else None

    for split in args.splits:
        df = load_split(splits_dir, split)
        if args.limit:
            df = df.head(args.limit)
        # A --limit run is a sample: it gets its own file so the full run does not reuse it
        suffix = f"_limit{args.limit}" if args.limit else ""
        prefix = "results" if args.name == "claude" else f"{args.name}_results"
        raw_path = output_path(cfg, "claude", f"{prefix}_{split}{suffix}.jsonl")
        if args.batch_id:
            batch_id = args.batch_id
        elif raw_path.exists():
            print(f"{split}: ya existe {raw_path}, se reutiliza")
            batch_id = None
        else:
            requests = [build_claude_request(r.text_id, list(r.sentences), ccfg, example) for r in df.itertuples()]
            if client is not None:
                sample = requests[:5]
                avg_in = sum(count_prompt_tokens(client, r) for r in sample) / len(sample)
            else:
                avg_in = sum(approx_prompt_tokens(r) for r in requests) / len(requests)
            cost = len(requests) * (avg_in * PRICE_IN + args.est_output_tokens * PRICE_OUT) / 1e6 / 2
            print(f"{split}: {len(requests)} textos, ~{avg_in:.0f} tokens de entrada por texto, costo estimado con batch ~${cost:.2f}")
            if args.dry_run:
                continue
            if args.sync:
                save_jsonl(run_claude_sync(client, requests), raw_path)
                batch_id = None
            else:
                batch_id = submit_claude_batch(client, requests)
                print("Lote enviado:", batch_id)
        if batch_id:
            wait_for_batch(client, batch_id)
            save_jsonl(collect_claude_results(client, batch_id), raw_path)
        if not raw_path.exists():
            continue
        results = load_jsonl(raw_path)
        if not args.limit:
            save_predictions(to_prediction_format(results, df), output_path(cfg, "predictions", f"{args.name}_{split}.jsonl"))
        refused = sum(r.get("stop_reason") == "refusal" for r in results)
        tokens_in = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
        tokens_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
        real = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6 / (1 if args.sync else 2)
        print(f"{split}: {len(results)} resultados, {refused} rechazos, costo real ~${real:.2f}")
        if results:
            print(f"  promedio por texto: {tokens_in / len(results):.0f} tokens de entrada, {tokens_out / len(results):.0f} de salida")
            full = len(load_split(splits_dir, split))
            print(f"  proyección para los {full} textos de {split}: ~${real / len(results) * full:.2f}")
        if args.limit and results:
            gold = dict(zip(df["text_id"], df["error_flag"].astype(int)))
            hits = sum(p.error_flag == gold[p.text_id] for p in to_prediction_format(results, df))
            print(f"  muestra: {hits}/{len(results)} aciertos en A (orientativo)")


if __name__ == "__main__":
    main()
