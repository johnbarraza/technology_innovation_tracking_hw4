from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, ensure_parent_dir, read_csv_if_exists, write_csv


TEST_INPUT = ROOT / "data" / "splits" / "test.csv"
METRICS_OUTPUT = ROOT / "output" / "metrics" / "metrics.json"
CONFUSION_OUTPUT = ROOT / "output" / "metrics" / "confusion_matrix.csv"
REPORT_OUTPUT = ROOT / "output" / "metrics" / "classification_report.csv"
EXAMPLES_OUTPUT = ROOT / "output" / "tables" / "prediction_examples.csv"


def predict_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """Load the trained model and add a predicted_label column.

    Persona 2 should implement real model inference here.
    """
    raise NotImplementedError("Implement model inference here.")


def evaluate_predictions(df: pd.DataFrame) -> None:
    y_true = df["label"]
    y_pred = df["predicted_label"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "model_name": "pending",
    }

    labels = sorted(set(y_true).union(set(y_pred)))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_df.insert(0, "label", labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index(names="label")

    examples_df = df.copy()
    examples_df["is_correct"] = examples_df["label"] == examples_df["predicted_label"]

    ensure_parent_dir(METRICS_OUTPUT)
    with METRICS_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    write_csv(confusion_df, CONFUSION_OUTPUT)
    write_csv(report_df, REPORT_OUTPUT)
    write_csv(examples_df, EXAMPLES_OUTPUT)


def main() -> None:
    test_df = read_csv_if_exists(TEST_INPUT)
    if test_df.empty:
        print(f"No test data found at {TEST_INPUT.relative_to(ROOT)}")
        return

    predictions_df = predict_test_set(test_df)
    evaluate_predictions(predictions_df)
    print("Saved evaluation outputs.")


if __name__ == "__main__":
    main()
