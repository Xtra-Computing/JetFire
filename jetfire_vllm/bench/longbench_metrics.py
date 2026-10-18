"""English LongBench v1 evaluation metrics.

Vendored from THUDM/LongBench so the release evaluator does not depend on a
sibling checkout or ambient ``sys.path`` contents.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from rouge import Rouge

_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    text = text.lower()
    text = "".join(char for char in text if char not in _PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: list[str], ground_truth: list[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: object) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return _token_f1(prediction_tokens, ground_truth_tokens)


def classification_score(
    prediction: str,
    ground_truth: str,
    *,
    all_classes: list[str] | None = None,
    **_: object,
) -> float:
    """Apply LongBench's multi-label penalty for classification outputs."""

    matched = [
        class_name
        for class_name in (all_classes or [])
        if class_name in prediction
    ]
    matched = [
        class_name
        for class_name in matched
        if class_name == ground_truth or class_name not in ground_truth
    ]
    if ground_truth not in matched:
        return 0.0
    return 1.0 / len(matched)


def rouge_score(prediction: str, ground_truth: str, **_: object) -> float:
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return float(scores["rouge-l"]["f"])
