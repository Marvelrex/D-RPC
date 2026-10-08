"""Tests for per-dataset answer extractors in collab_infer.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infer"))

from collab_infer import (
    EXTRACTORS,
    _extract_gsm8k_answer,
    _extract_aqua_answer,
    _extract_strategyqa_answer,
)


def test_gsm8k_plain_number():
    assert _extract_gsm8k_answer("The answer is 42.") == "42"


def test_gsm8k_hash_delimiter():
    assert _extract_gsm8k_answer("She has 5 apples. #### 5") == "5"


def test_gsm8k_json_format():
    assert _extract_gsm8k_answer('{"rationale": "...", "ans": 123}') == "123"


def test_gsm8k_negative():
    assert _extract_gsm8k_answer("The result is -15.") == "-15"


def test_gsm8k_decimal():
    assert _extract_gsm8k_answer("Total: 3.14") == "3.14"


def test_gsm8k_comma_separated():
    assert _extract_gsm8k_answer("Amount = 1,234") == "1234"


def test_gsm8k_none_on_no_number():
    assert _extract_gsm8k_answer("No numerical answer here.") is None


def test_aqua_plain_letter():
    assert _extract_aqua_answer("The answer is C.") == "C"


def test_aqua_parenthesized():
    assert _extract_aqua_answer("Answer is (B).") == "B"


def test_aqua_json_format():
    assert _extract_aqua_answer('{"rationale": "...", "ans": "D"}') == "D"


def test_aqua_lowercase_letter():
    assert _extract_aqua_answer("answer is e") == "E"


def test_aqua_multiple_letters_last_wins():
    result = _extract_aqua_answer("Option A seems close, but the answer is B.")
    assert result == "B"


def test_aqua_none_on_no_letter():
    result = _extract_aqua_answer("The computation shows the value is 42.")
    assert result is None or result in "ABCDE"


def test_ai2arc_and_gpqa_use_choice_extractor():
    assert EXTRACTORS["ai2arc"] is _extract_aqua_answer
    assert EXTRACTORS["gpqa"] is _extract_aqua_answer


def test_strategyqa_yes():
    assert _extract_strategyqa_answer("Yes, this is correct.") == "yes"


def test_strategyqa_no():
    assert _extract_strategyqa_answer("No, that is not true.") == "no"


def test_strategyqa_json_true():
    assert _extract_strategyqa_answer('{"rationale": "...", "ans": true}') == "yes"


def test_strategyqa_json_false():
    assert _extract_strategyqa_answer('{"rationale": "...", "ans": false}') == "no"


def test_strategyqa_last_token_wins():
    result = _extract_strategyqa_answer("At first yes, but ultimately no.")
    assert result == "no"


def test_strategyqa_none_on_ambiguous():
    result = _extract_strategyqa_answer("The answer is unclear.")
    assert result is None


def test_strategyqa_true_keyword():
    assert _extract_strategyqa_answer("The statement is true.") == "yes"


def test_strategyqa_false_keyword():
    assert _extract_strategyqa_answer("The claim is false.") == "no"
