from medec.corrector import extract_corrected_sentence
from medec.quality import format_prediction, quality_check_correction


def test_identical_correction_is_doubtful():
    qc = quality_check_correction("The pathogen was H. influenzae.", "The pathogen was H. influenzae.")
    assert qc["doubtful"] and qc["reason"] == "identical"


def test_small_change_passes():
    qc = quality_check_correction(
        "After imaging the causal pathogen was determined to be Haemophilus influenzae.",
        "After imaging the causal pathogen was determined to be Streptococcus pneumoniae.",
    )
    assert not qc["doubtful"]


def test_format_prediction():
    assert format_prediction("ms-1", 0, -1, None) == "ms-1 0 -1 NA"
    assert format_prediction("ms-2", 1, 3, "Fixed  text.") == 'ms-2 1 3 "Fixed text."'


def test_extract_corrected_sentence():
    assert extract_corrected_sentence('Corrected sentence: "He takes aspirin."\nBecause...') == "He takes aspirin."
    assert extract_corrected_sentence("6 The pathogen was S. pneumoniae.") == "The pathogen was S. pneumoniae."


def test_compare_with_reference():
    from medec.quality import compare_with_reference

    orig = "Suspected of infection with clostridium botulinum."
    ref = "Suspected of infection with Streptococcus pyogenes."
    right = compare_with_reference(orig, "Suspected infection with Streptococcus pyogenes (group A).", ref)
    wrong = compare_with_reference(orig, "Suspected of infection with clostridium perfringens.", ref)
    assert right["key_terms"] == ["streptococcus", "pyogenes"] and right["verdict"] == "key_terms"
    assert wrong["verdict"] == "different" and wrong["key_found"] == []
    assert compare_with_reference(orig, ref, ref)["verdict"] == "exact"
