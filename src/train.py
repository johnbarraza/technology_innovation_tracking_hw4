from __future__ import annotations

from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, read_csv_if_exists, write_csv


LABELED_INPUT = ROOT / "data" / "labeled" / "labeled_repositories.csv"
TRAIN_OUTPUT = ROOT / "data" / "splits" / "train.csv"
VALIDATION_OUTPUT = ROOT / "data" / "splits" / "validation.csv"
TEST_OUTPUT = ROOT / "data" / "splits" / "test.csv"
MODEL_OUTPUT_DIR = ROOT / "models" / "trained_models"


def create_splits(test_size: float = 0.15, validation_size: float = 0.15) -> None:
    df = read_csv_if_exists(LABELED_INPUT)
    if df.empty:
        write_csv(df, TRAIN_OUTPUT)
        write_csv(df, VALIDATION_OUTPUT)
        write_csv(df, TEST_OUTPUT)
        return

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"] if "label" in df.columns and df["label"].nunique() > 1 else None,
    )
    validation_ratio = validation_size / (1 - test_size)
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=validation_ratio,
        random_state=42,
        stratify=train_val_df["label"]
        if "label" in train_val_df.columns and train_val_df["label"].nunique() > 1
        else None,
    )

    write_csv(train_df, TRAIN_OUTPUT)
    write_csv(validation_df, VALIDATION_OUTPUT)
    write_csv(test_df, TEST_OUTPUT)


def fine_tune_model() -> None:
    """Fine-tune a lightweight BERT classifier and save it.

    Persona 2 should implement this with HuggingFace Transformers. Recommended
    model: distilbert-base-uncased or sentence-transformers/all-MiniLM-L6-v2.
    """
    raise NotImplementedError("Implement BERT fine-tuning here.")


def main() -> None:
    create_splits()
    fine_tune_model()
    print(f"Model artifacts should be saved to {MODEL_OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
