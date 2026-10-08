"""Tests for count_steps in clean_sg.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))

from clean_sg import count_steps


def test_two_numbered():
    sg = "1. First step.\n2. Second step."
    assert count_steps(sg) == 2


def test_six_bulleted():
    sg = "\n".join(f"- Step {i}" for i in range(1, 7))
    assert count_steps(sg) == 6


def test_mixed_markers():
    sg = "1. Do A.\n- Do B.\n2. Do C."
    assert count_steps(sg) >= 2


def test_empty_returns_zero():
    assert count_steps("") == 0
    assert count_steps("   ") == 0


def test_step_keyword_format():
    sg = "Step 1: Understand.\nStep 2: Plan.\nStep 3: Execute.\nStep 4: Verify."
    assert count_steps(sg) == 4


def test_fallback_non_bulleted():
    sg = "Understand the problem.\nIdentify key values.\nPlan the solution."
    assert count_steps(sg) == 3


def test_single_line_is_one():
    sg = "1. Only step."
    assert count_steps(sg) == 1


def test_parenthesis_marker():
    sg = "1) Read the problem.\n2) Identify unknowns.\n3) Apply formula."
    assert count_steps(sg) == 3
