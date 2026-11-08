"""External reference: Claude Opus 5.5 over a split with the Message Batches API.

Costs money. First run with --dry-run (counts tokens and estimates the cost),
then without it. The API key is read from taller_8/.env.
"""

from _common import parse_args

import anthropic

from medec.claude_reference import (
    build_claude_request, collect_claude_results, count_prompt_tokens, format_example, load_api_key,
    submit_claude_batch, to_prediction_format, wait_for_batch,
)
from medec.config import PROJECT_DIR, output_path
from medec.data import load_split
from medec.schemas import load_jsonl, save_jsonl, save_predictions

PRICE_IN, PRICE_OUT = 4.0, 20.0  # USD per million tokens, claude-opus-5-5


def approx_prompt_tokens(request) -> float:
    """Offline estimate: ~3.5 characters per token, plus the JSON schema and message overhead."""
    params = request["params"]
    chars = len(params["system"]) + sum(len(m["content"]) for m in params["messages"]) + len(str(params["output_config"]))
    return chars / 3.5 + 50


def extra(parser):
    parser.add_argument("--dry-run", action="store_true", help="Only count tokens and estimate the cost")
    parser.add_argument("--batch-id", default=None, help="Collect an already submitted batch")
    parser.add_argument("--est-output-tokens", type=int, default=1000, help="Output tokens per text (thinking included)")


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True, extra=extra)
    ccfg = cfg["claude"]
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
        raw_path = output_path(cfg, "claude", f"results_{split}.jsonl")
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
            batch_id = submit_claude_batch(client, requests)
            print("Lote enviado:", batch_id)
        if batch_id:
            wait_for_batch(client, batch_id)
            save_jsonl(collect_claude_results(client, batch_id), raw_path)
        if not raw_path.exists():
            continue
        results = load_jsonl(raw_path)
        save_predictions(to_prediction_format(results, df), output_path(cfg, "predictions", f"claude_{split}.jsonl"))
        refused = sum(r.get("stop_reason") == "refusal" for r in results)
        tokens_in = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
        tokens_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
        real = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6 / 2
        print(f"{split}: {len(results)} resultados, {refused} rechazos, costo real ~${real:.2f}")


if __name__ == "__main__":
    main()
