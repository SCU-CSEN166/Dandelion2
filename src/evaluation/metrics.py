"""Counting and scoring shared by the evaluation scripts.

Binary decisions (e.g. relevant or not) count TP/FP/FN/TN as usual. Extracted fields
are scored like slot filling, since most labels are empty and accuracy alone would
reward extracting nothing:
    TP = right value, FP = value given but wrong or not expected,
    FN = expected value missed or wrong, TN = both empty.
A wrong value is both an FP and an FN.
"""

from collections.abc import Iterable
from typing import Any

NULL = "null"  # stands in for None as a confusion-matrix key (JSON keys must be strings)


def scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    """The four counts plus accuracy, precision, recall and F1 (None when undefined)."""
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def binary_counts(decisions: Iterable[tuple[bool, bool]]) -> tuple[int, int, int, int]:
    """(TP, FP, FN, TN) over (predicted, actual) pairs."""
    tp = fp = fn = tn = 0
    for predicted, actual in decisions:
        tp += predicted and actual
        fp += predicted and not actual
        fn += actual and not predicted
        tn += not predicted and not actual
    return tp, fp, fn, tn


def slot_counts(pairs: Iterable[tuple[Any, Any]]) -> tuple[int, int, int, int]:
    """(TP, FP, FN, TN) over (predicted, expected) field values; see the module docstring."""
    tp = fp = fn = tn = 0
    for predicted, expected in pairs:
        if predicted is not None and predicted == expected:
            tp += 1
        elif predicted is None and expected is None:
            tn += 1
        else:
            fp += predicted is not None
            fn += expected is not None
    return tp, fp, fn, tn


def slot_scores(pairs: list[tuple[Any, Any]]) -> dict:
    """scores() for a field. Accuracy is the share of exact matches; the standard formula
    would count a wrong value twice."""
    exact = sum(predicted == expected for predicted, expected in pairs)
    return {**scores(*slot_counts(pairs)), "accuracy": _ratio(exact, len(pairs))}


def require_options(labels: Iterable[Any], options: list[str], source: str) -> None:
    """Stop before any model runs if a label is not one of the template options: no model
    could ever predict it, so it would silently count as a miss."""
    unknown = sorted({label for label in labels if label is not None and label not in options})
    if unknown:
        raise SystemExit(f"{source}: labels that are not template options: {unknown}. "
                         "Map them to an option in ACTIVITY_LABELS.")


def confusion_matrix(pairs: Iterable[tuple[Any, Any]]) -> dict[str, dict[str, int]]:
    """{expected value: {predicted value: count}}."""
    matrix: dict[str, dict[str, int]] = {}
    for predicted, expected in pairs:
        row = matrix.setdefault(_key(expected), {})
        row[_key(predicted)] = row.get(_key(predicted), 0) + 1
    return matrix


def _key(value: Any) -> str:
    return NULL if value is None else str(value)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
