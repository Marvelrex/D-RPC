"""Tests for SG cleaning/validation (clean_sg.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))

from clean_sg import validate_sg, count_steps, RejectReason


def test_count_numbered_steps():
    sg = "1. Identify the variables.\n2. Set up the equation.\n3. Verify the result."
    assert count_steps(sg) == 3


def test_count_bulleted_steps():
    sg = "- Read the problem.\n- Identify unknowns.\n- Formulate a plan."
    assert count_steps(sg) == 3


def test_count_step_keyword():
    sg = "Step 1: Understand the problem.\nStep 2: Break it down.\nStep 3: Plan a solution."
    assert count_steps(sg) >= 3


def test_count_fallback_nonbulleted():
    sg = "Understand what is asked.\nIdentify relevant information.\nPlan a strategy."
    assert count_steps(sg) == 3


def test_valid_sg_passes():
    sg = (
        "1. Identify the number of items and their unit cost.\n"
        "2. Determine the total expenditure constraint.\n"
        "3. Check whether the purchase fits within the budget."
    )
    valid, reasons = validate_sg(sg)
    assert valid, f"Expected valid but got reasons: {[r.value for r in reasons]}"
    assert reasons == []


def test_valid_strategyqa_style():
    sg = (
        "- Recall the relevant historical facts about the topic.\n"
        "- Consider whether the claim aligns with those facts.\n"
        "- Determine if additional reasoning steps are needed."
    )
    valid, reasons = validate_sg(sg)
    assert valid, reasons


def test_rejects_one_step():
    sg = "1. Solve the problem directly."
    valid, reasons = validate_sg(sg)
    assert not valid
    assert RejectReason.TOO_FEW_STEPS in reasons


def test_rejects_seven_steps():
    sg = "\n".join(f"{i+1}. Step {i+1}" for i in range(7))
    valid, reasons = validate_sg(sg)
    assert not valid
    assert RejectReason.TOO_MANY_STEPS in reasons


def test_rejects_empty():
    valid, reasons = validate_sg("")
    assert not valid
    assert RejectReason.EMPTY in reasons


def test_rejects_addition():
    sg = (
        "1. Note that 3 + 4 = 7.\n"
        "2. Multiply by the rate.\n"
        "3. Sum all values."
    )
    valid, reasons = validate_sg(sg)
    assert not valid
    assert RejectReason.CONTAINS_ARITHMETIC in reasons


def test_rejects_multiplication():
    sg = "1. Compute 12 * 5.\n2. Subtract from total.\n3. Verify."
    valid, reasons = validate_sg(sg)
    assert not valid
    assert RejectReason.CONTAINS_ARITHMETIC in reasons


def test_rejects_equation_with_numbers():
    sg = "1. Note x = 42.\n2. Substitute back.\n3. Verify constraints."
    valid, reasons = validate_sg(sg)
    assert not valid
    assert RejectReason.CONTAINS_EQUATION in reasons


def test_rejects_final_answer_number():
    sg = (
        "1. Identify the variables.\n"
        "2. Set up equations.\n"
        "Therefore the answer is 42."
    )
    valid, reasons = validate_sg(sg)
    assert not valid
    assert RejectReason.CONTAINS_FINAL_ANSWER_NUMBER in reasons


def test_rejects_choice_answer():
    sg = (
        "1. Analyze the options.\n"
        "2. Compare with constraints.\n"
        "The answer is (C)."
    )
    valid, reasons = validate_sg(sg)
    assert not valid
    assert RejectReason.CONTAINS_CHOICE_ANSWER in reasons


def test_rejects_long_numbers():
    sg = (
        "1. Identify the large value 123456.\n"
        "2. Check the constraints.\n"
        "3. Verify the answer."
    )
    valid, reasons = validate_sg(sg)
    assert not valid
    assert RejectReason.CONTAINS_LONG_NUMBERS in reasons


def test_allows_variable_names_with_numbers():
    sg = (
        "1. Identify variable x2 and variable y.\n"
        "2. Determine the relationship between them.\n"
        "3. Formulate the approach to compare them."
    )
    valid, reasons = validate_sg(sg)
    assert RejectReason.CONTAINS_ARITHMETIC not in reasons
    assert RejectReason.CONTAINS_EQUATION not in reasons
