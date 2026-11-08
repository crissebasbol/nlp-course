"""Step 3b (alternative): verify the sentence picked by BERT with an LLM chosen in the config.

BERT finds the key sentence (diagnosis / management) very well but cannot tell whether it is correct.
Here an LLM answers only that question for sentence k, and proposes a correction in the same call.
The LLM is a parameter: Claude (Haiku / Sonnet, through the Anthropic API) or the local Qwen.
DSPy is used without compilation: it only gives the same signature to any provider.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import dspy

from medec.data import number_sentences

logger = logging.getLogger(__name__)


class VerifySentenceSignature(dspy.Signature):
    """You are a physician reviewing a clinical note. The note is either fully correct or contains
    exactly one medical error (a wrong diagnosis, management step, treatment, pharmacotherapy or
    causal organism). Sentences are numbered from 0. One sentence was selected because it states the
    key clinical decision. Decide whether that sentence is medically correct given the rest of the
    note. If it is not, rewrite it with the error fixed, keeping the same style."""

    numbered_text: str = dspy.InputField(desc="Clinical note, one numbered sentence per line")
    sentence_id: int = dspy.InputField(desc="Number of the selected sentence")
    sentence: str = dspy.InputField(desc="Text of the selected sentence")
    is_correct: bool = dspy.OutputField(desc="True if the selected sentence is medically correct")
    corrected_sentence: str = dspy.OutputField(desc="If incorrect, the sentence with the error fixed; if correct, the same sentence")


def make_verifier_lm(model: str, vcfg: dict[str, Any], cfg: dict[str, Any]) -> dspy.LM:
    """`anthropic/<model>` uses ANTHROPIC_API_KEY; `local` uses the llama-server of the config.

    No temperature is sent: Claude 5.5 models reject non-default sampling parameters.
    """
    if model == "local":
        m = cfg["models"]
        return dspy.LM(f"openai/{m['llm_name']}", api_base=m["llm_api_base"], api_key="local",
                       temperature=0.0, max_tokens=vcfg.get("max_tokens", 600), num_retries=1)
    if model.startswith("anthropic/"):
        from medec.claude_reference import load_api_key
        from medec.config import PROJECT_DIR

        load_api_key(PROJECT_DIR / ".env")  # puts the key in the environment; litellm takes it from there
    return dspy.LM(model, max_tokens=vcfg.get("max_tokens", 4000), num_retries=3)


def build_verifier(lm: dspy.LM, chain_of_thought: bool = False) -> dspy.Module:
    """Predict for models with their own internal reasoning (Claude); ChainOfThought optional for Qwen."""
    program = (dspy.ChainOfThought if chain_of_thought else dspy.Predict)(VerifySentenceSignature)
    program.set_lm(lm)
    return program


def verify_sentence(program: dspy.Module, sentences: list[str], k: int) -> dict[str, Any]:
    """{is_correct, corrected_sentence, failed, seconds}; a failed call is reported, not raised."""
    start = time.time()
    try:
        pred = program(numbered_text=number_sentences(sentences), sentence_id=k, sentence=sentences[k])
        out = {"is_correct": bool(pred.is_correct), "corrected_sentence": (pred.corrected_sentence or "").strip(), "failed": False}
    except Exception as exc:  # parse errors, refusals, network
        logger.warning("Verifier failed: %s", exc)
        out = {"is_correct": None, "corrected_sentence": "", "failed": True, "error": str(exc)[:300]}
    out["seconds"] = round(time.time() - start, 2)
    return out


def verify_split(program: dspy.Module, items: list[dict[str, Any]], num_threads: int = 8,
                 on_result: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """items: [{text_id, sentences, k}]. Results are passed to on_result as they arrive (to save them)."""

    def run(item: dict[str, Any]) -> dict[str, Any]:
        res = {"text_id": item["text_id"], "k": item["k"], **verify_sentence(program, item["sentences"], item["k"])}
        if on_result:
            on_result(res)
        return res

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(run, items))


# ------------------------------------------------------------------ optional compilation
def to_verifier_example(sentences: list[str], k: int, error_flag: int, error_sentence_id: int,
                        corrected_sentence: str | None) -> dspy.Example:
    """Gold answer for sentence k: it is wrong only if the text has an error and k is that sentence."""
    wrong = bool(error_flag) and k == int(error_sentence_id)
    return dspy.Example(
        numbered_text=number_sentences(sentences), sentence_id=k, sentence=sentences[k],
        is_correct=not wrong, corrected_sentence=corrected_sentence if wrong else sentences[k],
    ).with_inputs("numbered_text", "sentence_id", "sentence")


def verification_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """1 for the right verdict; for a wrong sentence, half of it depends on the correction containing the
    key change of the reference. While bootstrapping demos (trace) only fully right answers count."""
    from medec.quality import compare_with_reference

    if bool(pred.is_correct) != bool(example.is_correct):
        return 0.0
    if example.is_correct:
        return 1.0
    verdict = compare_with_reference(example.sentence, pred.corrected_sentence or "", example.corrected_sentence)["verdict"]
    good_fix = verdict in ("exact", "key_terms")
    if trace is not None:
        return float(good_fix)
    return 0.5 + 0.5 * good_fix


def compile_verifier(program: dspy.Module, trainset: list[dspy.Example], max_demos: int = 3) -> dspy.Module:
    """BootstrapFewShot keeps the first right answers as demos, so trainset should alternate
    correct / wrong sentences: demos all of one kind would bias the verdict (as happened with Qwen)."""
    optimizer = dspy.BootstrapFewShot(metric=verification_metric, max_bootstrapped_demos=max_demos, max_labeled_demos=0)
    return optimizer.compile(program, trainset=trainset)


def balanced_order(examples: list[dspy.Example]) -> list[dspy.Example]:
    """Interleave correct and wrong examples: wrong, correct, wrong, correct, ..."""
    wrong = [e for e in examples if not e.is_correct]
    right = [e for e in examples if e.is_correct]
    out = []
    for i in range(max(len(wrong), len(right))):
        out += [x[i] for x in (wrong, right) if i < len(x)]
    return out
