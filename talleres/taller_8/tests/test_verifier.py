"""Verifier helpers (no LLM calls)."""

import dspy

from medec.verifier import balanced_order, to_verifier_example, verification_metric

SENTS = ["A 30-year-old man has fever.", "Diagnosis is malaria.", "He is treated."]


def test_gold_answer_depends_on_bert_pick():
    # The text has an error in sentence 1
    right_pick = to_verifier_example(SENTS, 1, 1, 1, "Diagnosis is dengue.")
    wrong_pick = to_verifier_example(SENTS, 2, 1, 1, "Diagnosis is dengue.")
    assert right_pick.is_correct is False and right_pick.corrected_sentence == "Diagnosis is dengue."
    assert wrong_pick.is_correct is True and wrong_pick.corrected_sentence == SENTS[2]


def test_metric_rewards_verdict_and_key_change():
    ex = to_verifier_example(SENTS, 1, 1, 1, "Diagnosis is dengue.")
    good = dspy.Prediction(is_correct=False, corrected_sentence="Diagnosis is dengue fever.")
    bad_fix = dspy.Prediction(is_correct=False, corrected_sentence="Diagnosis is typhoid.")
    wrong = dspy.Prediction(is_correct=True, corrected_sentence=SENTS[1])
    assert verification_metric(ex, good) == 1.0
    assert verification_metric(ex, bad_fix) == 0.5
    assert verification_metric(ex, bad_fix, trace=[]) == 0.0  # demos need a right fix
    assert verification_metric(ex, wrong) == 0.0


def test_balanced_order_alternates():
    exs = [to_verifier_example(SENTS, k, 1, 1, "Diagnosis is dengue.") for k in (0, 1, 2)]
    exs.append(to_verifier_example(SENTS, 1, 1, 1, "Diagnosis is dengue."))
    order = [e.is_correct for e in balanced_order(exs)]
    assert order[:4] == [False, True, False, True]
