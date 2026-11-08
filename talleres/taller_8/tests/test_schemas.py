from medec.schemas import Prediction, load_predictions, save_predictions


def test_no_error_prediction_is_normalized():
    p = Prediction("x", 0, 3, "something")
    assert p.error_sentence_id == -1 and p.corrected_sentence is None


def test_roundtrip(tmp_path):
    preds = [Prediction("a", 1, 2, "fixed", [0.1, 0.9]), Prediction("b", 0)]
    path = save_predictions(preds, tmp_path / "p.jsonl")
    assert load_predictions(path) == preds
