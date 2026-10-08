"""
Tests that SGFT prompt strings match Table 1 wording exactly (arXiv:2412.09906v1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))

from generate_sg import (
    PROMPT_I_TEMPLATE,
    PROMPT_II,
    PROMPT_III_TEMPLATE,
    PROMPT_IV,
    PROMPT_V_TEMPLATE,
    RETRY_CONSTRAINT,
)


def test_prompt_i_contains_required_fields():
    rendered = PROMPT_I_TEMPLATE.format(question="How much is 2+2?")
    assert "[Problem Type]" in rendered
    assert "[Solution Objective]" in rendered
    assert "[Constraints]" in rendered
    assert "[Priorities and Considerations]" in rendered
    assert 'How much is 2+2?' in rendered


def test_prompt_ii_wording():
    assert "step-by-step solution" in PROMPT_II
    assert "don't need to solve it" in PROMPT_II
    assert ";" in PROMPT_II
    assert "2 to 6 steps" in PROMPT_II


def test_prompt_iii_wording():
    rendered = PROMPT_III_TEMPLATE.format(examples="[Q: foo\nSG: bar]")
    assert "step-by-step solution" in rendered
    assert "don't need to solve it" in rendered
    assert "," in rendered
    assert "Examples for the SG data are as follows" in rendered
    assert "[Q: foo\nSG: bar]" in rendered


def test_prompt_iv_no_calculations_phrase():
    assert "with no calculations" in PROMPT_IV
    assert "step-by-step solution" in PROMPT_IV
    assert "2 to 6 steps" in PROMPT_IV
    assert "following problem" in PROMPT_IV


def test_prompt_v_contains_examples_placeholder():
    rendered = PROMPT_V_TEMPLATE.format(examples="[Q: q1\nSG: s1]")
    assert "with no calculations" in rendered
    assert "Examples for the SG data are as follows" in rendered
    assert "[Q: q1\nSG: s1]" in rendered


def test_retry_constraint_no_calculations():
    assert "calculations" in RETRY_CONSTRAINT.lower()
    assert "numeric operations" in RETRY_CONSTRAINT.lower()
    assert "high-level steps" in RETRY_CONSTRAINT.lower()


def test_prompt_ii_vs_iii_differ_in_semicolon_comma():
    """prompt_ii uses ';' while prompt_iii uses ',' before 'just output'."""
    assert "; just output" in PROMPT_II or ";\n just output" in PROMPT_II
    sample_iii = PROMPT_III_TEMPLATE.format(examples="")
    assert ", just output" in sample_iii or ",\n just output" in sample_iii
