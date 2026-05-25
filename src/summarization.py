from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, read_csv_if_exists, write_csv


PROCESSED_INPUT = ROOT / "data" / "processed" / "repositories.csv"
SUMMARY_OUTPUT = ROOT / "data" / "processed" / "repository_summaries.csv"


def build_repository_summary(row: pd.Series) -> str:
    topics = row.get("topics", "")
    return (
        f"Repository {row.get('full_name', 'unknown')} focuses on {row.get('primary_topic', 'unknown topic')}. "
        f"It has {row.get('stars', 0)} stars, {row.get('forks', 0)} forks, "
        f"{row.get('contributors_count', 0)} contributors, {row.get('commits_last_30d', 0)} commits in the last 30 days, "
        f"{row.get('open_issues', 0)} open issues, {row.get('releases_count', 0)} releases, "
        f"CI presence set to {row.get('has_ci', False)}, and topics: {topics}."
    )


def add_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        output = df.copy()
        output["repository_summary"] = []
        return output

    output = df.copy()
    output["repository_summary"] = output.apply(build_repository_summary, axis=1)
    return output


def main() -> None:
    df = read_csv_if_exists(PROCESSED_INPUT)
    output = add_summaries(df)
    write_csv(output, SUMMARY_OUTPUT)
    print(f"Saved {len(output)} rows to {SUMMARY_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
