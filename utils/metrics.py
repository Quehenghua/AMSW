"""
Evaluation Metrics
==================
Utilities for computing Precision, Recall, and macro-F1 consistent with the
ToxiCN MM benchmark evaluation protocol (Lu et al., NeurIPS 2024).

The benchmark reports:
  • Harmful Meme Detection:     P / R / F1 (macro) + F1 of the Harmful class
  • Harmful Type Identification: P / R / F1 (macro) + per-type F1
                                  (Tg. / Off. / Sex. / Disp.)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    f1_score,
)


# ---------------------------------------------------------------------------
# Label name maps
# ---------------------------------------------------------------------------
DETECTION_CLASS_NAMES = ["Non-Harmful", "Harmful"]
TYPE_CLASS_NAMES = ["Non-Harmful", "Targeted", "Offense", "Sexual", "Dispirited"]


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------
def compute_macro_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict[str, float]:
    """
    Compute macro-averaged Precision, Recall, and F1.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.

    Returns:
        Dict with keys "precision", "recall", "f1_macro".
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1_macro": float(f1)}


def compute_per_class_f1(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    num_classes: int,
) -> list[float]:
    """
    Compute per-class F1 scores.

    Returns:
        List of F1 scores, one per class in ascending label order.
    """
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)),
        average=None, zero_division=0
    )
    return [float(v) for v in f1_per_class]


# ---------------------------------------------------------------------------
# Task-specific evaluation
# ---------------------------------------------------------------------------
def evaluate_detection(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate Harmful Meme Detection (binary task).

    Metrics reported:
      P, R, F1 (macro), F1_Harmful

    Args:
        y_true:  Ground-truth binary labels (0 = Non-Harmful, 1 = Harmful).
        y_pred:  Predicted binary labels.
        verbose: Print a classification report.

    Returns:
        Dict with keys: "P", "R", "F1", "F1_Harmful".
    """
    macro = compute_macro_metrics(y_true, y_pred)
    per_cls = compute_per_class_f1(y_true, y_pred, num_classes=2)

    results = {
        "P":          round(macro["precision"] * 100, 2),
        "R":          round(macro["recall"]    * 100, 2),
        "F1":         round(macro["f1_macro"]  * 100, 2),
        "F1_Harmful": round(per_cls[1]         * 100, 2),
    }

    if verbose:
        print("\n=== Harmful Meme Detection ===")
        print(classification_report(
            y_true, y_pred,
            target_names=DETECTION_CLASS_NAMES,
            digits=4, zero_division=0,
        ))
        print(f"Macro F1:   {results['F1']:.2f}%")
        print(f"F1_Harmful: {results['F1_Harmful']:.2f}%\n")

    return results


def evaluate_type(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate Harmful Type Identification (5-class task).

    Metrics reported:
      P, R, F1 (macro), F1_Tg, F1_Off, F1_Sex, F1_Disp

    Note: class 0 = Non-Harmful, classes 1-4 = the four harmful types.

    Args:
        y_true:  Ground-truth type labels.
        y_pred:  Predicted type labels.
        verbose: Print a classification report.

    Returns:
        Dict with keys: "P", "R", "F1", "F1_Tg", "F1_Off", "F1_Sex", "F1_Disp".
    """
    macro = compute_macro_metrics(y_true, y_pred)
    per_cls = compute_per_class_f1(y_true, y_pred, num_classes=5)

    results = {
        "P":        round(macro["precision"] * 100, 2),
        "R":        round(macro["recall"]    * 100, 2),
        "F1":       round(macro["f1_macro"]  * 100, 2),
        "F1_Tg":    round(per_cls[1]         * 100, 2),
        "F1_Off":   round(per_cls[2]         * 100, 2),
        "F1_Sex":   round(per_cls[3]         * 100, 2),
        "F1_Disp":  round(per_cls[4]         * 100, 2),
    }

    if verbose:
        print("\n=== Harmful Type Identification ===")
        print(classification_report(
            y_true, y_pred,
            target_names=TYPE_CLASS_NAMES,
            digits=4, zero_division=0,
        ))
        print(f"Macro F1: {results['F1']:.2f}%")
        for k in ["F1_Tg", "F1_Off", "F1_Sex", "F1_Disp"]:
            print(f"  {k}: {results[k]:.2f}%")
        print()

    return results


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------
def evaluate(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    task: str,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Dispatcher that calls the appropriate task-specific evaluator.

    Args:
        y_true:  Ground-truth labels.
        y_pred:  Predicted labels.
        task:    "detection" or "type".
        verbose: Print report.

    Returns:
        Metrics dict.
    """
    if task == "detection":
        return evaluate_detection(y_true, y_pred, verbose=verbose)
    elif task == "type":
        return evaluate_type(y_true, y_pred, verbose=verbose)
    else:
        raise ValueError(f"Unknown task '{task}'. Expected 'detection' or 'type'.")
