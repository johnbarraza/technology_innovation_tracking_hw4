from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, normalize_label, read_csv_if_exists, write_csv


SUMMARY_INPUT = ROOT / "data" / "processed" / "repository_summaries.csv"
LABELED_OUTPUT = ROOT / "data" / "labeled" / "labeled_repositories.csv"


CATEGORY_DEFINITIONS = {
    "emerging_technology": "High recent activity and growing interest in a newer technical area.",
    "mature_ecosystem": "Stable, broad adoption with sustained maintenance and community activity.",
    "declining_technology": "Older area with weak recent activity or decreasing maintenance signals.",
    "experimental_niche": "Small, specialized, exploratory, or early-stage technical area.",
}


def build_labeling_prompt(repository_summary: str) -> str:
    categories = "\n".join(
        f"- {label}: {definition}" for label, definition in CATEGORY_DEFINITIONS.items()
    )
    return f"""Classify this GitHub repository into exactly one category.

Categories:
{categories}

Repository summary:
{repository_summary}

Return JSON with keys: label, confidence, rationale.
"""


def label_with_llm(repository_summary: str) -> dict:
    """Call an LLM provider and return label metadata.

    Persona 2 should implement the API call here. Keep the returned keys:
    label, confidence, rationale.
    """
    raise NotImplementedError("Implement LLM weak labeling here.")


def label_repositories(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        output = df.copy()
        for column in ["weak_label", "llm_rationale", "llm_confidence", "label"]:
            output[column] = []
        return output

    rows = []
    for _, row in df.iterrows():
        result = label_with_llm(row["repository_summary"])
        labeled_row = row.to_dict()
        labeled_row["weak_label"] = result.get("label")
        labeled_row["llm_rationale"] = result.get("rationale")
        labeled_row["llm_confidence"] = result.get("confidence")
        labeled_row["label"] = normalize_label(result.get("label", ""))
        rows.append(labeled_row)

    return pd.DataFrame(rows)


def main() -> None:
    df = read_csv_if_exists(SUMMARY_INPUT)
    labeled_df = label_repositories(df)
    write_csv(labeled_df, LABELED_OUTPUT)
    print(f"Saved {len(labeled_df)} rows to {LABELED_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
