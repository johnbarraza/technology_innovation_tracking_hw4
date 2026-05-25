from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, write_csv


RAW_OUTPUT = ROOT / "data" / "raw" / "repositories_raw.csv"


TECH_QUERIES = [
    "ai agents",
    "vector database",
    "cybersecurity tooling",
    "blockchain infrastructure",
    "robotics framework",
    "mlops platform",
]


def collect_repositories() -> pd.DataFrame:
    """Collect repository metadata from the GitHub API.

    Persona 1 should implement this function. The final output should follow
    the raw-to-processed pipeline documented in DATA_CONTRACT.md.
    """
    raise NotImplementedError("Implement GitHub API collection here.")


def main() -> None:
    df = collect_repositories()
    write_csv(df, RAW_OUTPUT)
    print(f"Saved {len(df)} rows to {RAW_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
