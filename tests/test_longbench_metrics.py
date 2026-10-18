from __future__ import annotations

import pytest

from jetfire_vllm.bench.longbench_metrics import (
    classification_score,
    normalize_answer,
    qa_f1_score,
    rouge_score,
)


def test_normalize_answer_matches_longbench_rules() -> None:
    assert normalize_answer(" The, QUICK fox! ") == "quick fox"


def test_qa_f1_ignores_articles_and_punctuation() -> None:
    assert qa_f1_score("A quick fox.", "the quick fox") == 1.0


def test_qa_f1_partial_overlap() -> None:
    assert qa_f1_score("red green", "green blue") == pytest.approx(0.5)


def test_rouge_identical_text() -> None:
    assert rouge_score("one two three", "one two three") == pytest.approx(1.0)


def test_rouge_invalid_empty_input_is_zero() -> None:
    assert rouge_score("", "reference") == 0.0


def test_classification_score_accepts_single_matching_class() -> None:
    assert classification_score(
        "Individual",
        "Individual",
        all_classes=["Individual", "Date"],
    ) == 1.0


def test_classification_score_penalizes_multiple_predicted_classes() -> None:
    assert classification_score(
        "Individual, not Date",
        "Individual",
        all_classes=["Individual", "Date"],
    ) == pytest.approx(0.5)


def test_classification_score_rejects_wrong_class() -> None:
    assert classification_score(
        "Date",
        "Individual",
        all_classes=["Individual", "Date"],
    ) == 0.0
