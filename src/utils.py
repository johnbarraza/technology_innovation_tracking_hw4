from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# Carga las variables de entorno desde el .env del proyecto
load_dotenv(ROOT / ".env")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def normalize_label(label: str) -> str:
    cleaned = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "emerging": "emerging_technology",
        "emerging_tech": "emerging_technology",
        "mature": "mature_ecosystem",
        "declining": "declining_technology",
        "experimental": "experimental_niche",
        "niche": "experimental_niche",
        "experimental_or_niche": "experimental_niche",
    }
    return aliases.get(cleaned, cleaned)
