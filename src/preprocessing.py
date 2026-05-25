from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, read_csv_if_exists, write_csv


RAW_INPUT = ROOT / "data" / "raw" / "repositories_raw.csv"
PROCESSED_OUTPUT = ROOT / "data" / "processed" / "repositories.csv"


REQUIRED_COLUMNS = [
    "repo_id",
    "full_name",
    "owner",
    "name",
    "html_url",
    "description",
    "language",
    "topics",
    "stars",
    "forks",
    "open_issues",
    "contributors_count",
    "commits_last_30d",
    "pull_requests_count",
    "releases_count",
    "has_ci",
    "readme_length",
    "created_at",
    "updated_at",
    "pushed_at",
    "repo_age_days",
    "last_activity_days",
    "primary_topic",
]


def preprocess_repositories(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw GitHub data into the shared processed schema."""
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    processed = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in processed.columns:
            processed[column] = None

    return processed[REQUIRED_COLUMNS]


def main() -> None:
    raw_df = read_csv_if_exists(RAW_INPUT)
    processed_df = preprocess_repositories(raw_df)
    write_csv(processed_df, PROCESSED_OUTPUT)
    print(f"Saved {len(processed_df)} rows to {PROCESSED_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
