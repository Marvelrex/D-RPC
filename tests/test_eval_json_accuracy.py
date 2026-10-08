from drpc.eval.eval_json_accuracy import compute_parse_stats, evaluate


def test_compute_parse_stats_counts_json_ans_outputs():
    rows = [
        {"gold_ans": "B", "model_response": '{"rationale": "...", "ans": "B"}'},
        {"gold_ans": "7", "response_payload": '{"ans": 7}'},
        {"gold_ans": "3", "model_response": "no json here"},
    ]
    parsed_total, total, ratio = compute_parse_stats(rows)
    assert parsed_total == 2
    assert total == 3
    assert ratio == 2 / 3


def test_evaluate_numeric_answers():
    rows = [
        {"gold_ans": "42", "model_response": '{"rationale": {"Step": "Step1: ..."}, "ans": 42}'},
        {"gold_ans": "0.5", "model_response": '{"rationale": {}, "ans": "1/2"}'},
        {"gold_ans": "10", "model_response": '{"rationale": {}, "ans": 11}'},
        {"gold_ans": "10", "model_response": "truncated output without a closing brace"},
    ]
    correct, total, misses = evaluate(rows)
    assert total == 4
    assert correct == 2
    assert [m[0] for m in misses] == [2, 3]


def test_evaluate_multiple_choice_answers():
    rows = [
        {"gold_ans": "C", "model_response": '{"rationale": {}, "ans": "C"}'},
        {"gold_ans": "A", "model_response": '{"rationale": {}, "ans": "B"}'},
    ]
    correct, total, _ = evaluate(rows)
    assert (correct, total) == (1, 2)
