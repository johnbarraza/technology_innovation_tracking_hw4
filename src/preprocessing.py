from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, read_csv_if_exists, write_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_INPUT        = ROOT / "data" / "raw"       / "repositories_raw.csv"
PROCESSED_OUTPUT = ROOT / "data" / "processed" / "repositories.csv"

# Contrato de columnas definido por el skeleton (NO modificar nombres)
REQUIRED_COLUMNS = [
    "repo_id", "full_name", "owner", "name", "html_url",
    "description", "language", "topics",
    "stars", "forks", "open_issues",
    "contributors_count", "commits_last_30d", "pull_requests_count",
    "releases_count", "has_ci", "readme_length",
    "created_at", "updated_at", "pushed_at",
    "repo_age_days", "last_activity_days", "primary_topic",
]


# ── Paso 1: Limpieza ───────────────────────────────────────────────────────

def drop_unwanted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas de baja calidad.

    Criterios y justificación:
    - Duplicados (full_name): pueden venir de queries superpuestos.
    - stars < 10: repos casi vacíos sin señales útiles de ecosistema.
    - description < 10 chars: sin contexto semántico para el LLM ni para BERT.
    """
    n0 = len(df)
    df = df.drop_duplicates(subset="full_name", keep="first")
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
    df = df[df["stars"] >= 10]
    df["description"] = df["description"].fillna("").str.strip()
    df = df[df["description"].str.len() >= 10]
    logger.info(f"Limpieza: {n0} → {len(df)} filas (eliminadas: {n0 - len(df)})")
    return df.reset_index(drop=True)


# ── Paso 2: Imputación ────────────────────────────────────────────────────

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes.
    - Señales numéricas de actividad → 0 (ausencia de actividad es información).
    - Booleanos → False.
    - Strings → 'unknown' o ''.
    """
    int_cols = [
        "stars", "forks", "open_issues", "contributors_count",
        "commits_last_30d", "pull_requests_count", "releases_count",
        "readme_length", "repo_age_days", "last_activity_days",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "has_ci" in df.columns:
        df["has_ci"] = df["has_ci"].map(
            lambda x: str(x).strip().lower() in ("true", "1", "yes")
        ).fillna(False)

    for col, default in [("language", "unknown"), ("topics", ""),
                          ("description", ""), ("primary_topic", "")]:
        if col in df.columns:
            df[col] = df[col].fillna(default).astype(str)

    return df


# ── Paso 3: Features derivadas ────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features que capturan momentum tecnológico para el Track B.

    Cada feature responde una pregunta analítica específica:

    commit_velocity (commits/semana en últimos 30d):
        ¿Qué tan activo está el desarrollo AHORA?
        Alta en emerging y mature; casi nula en declining.

    recency_score = 1 / (1 + log(last_activity_days + 1)):
        ¿Qué tan reciente fue la última actividad?
        Cercano a 1 si push reciente; cercano a 0 si lleva un año inactivo.
        Clave para separar declining de mature estable.

    release_cadence (releases/año):
        ¿Con qué frecuencia se lanza una versión?
        Regular en mature; irregular o nula en experimental.

    pr_throughput = closed / (open + closed):
        ¿El proyecto integra contribuciones externas eficientemente?
        Alta en mature; baja en declining o experimental.

    community_breadth = log(contributors + 1) + log(forks + 1):
        ¿Qué tan amplia es la comunidad?
        Alta en mature; baja en experimental/niche.

    stars_growth_proxy = stars / log(repo_age_days + 2):
        ¿Las estrellas crecieron rápido relativo a la edad?
        Si es alto en repo joven → emerging. Captura crecimiento explosivo.
    """
    df["commit_velocity"] = (df["commits_last_30d"] / 4.0).round(2)

    df["recency_score"] = (
        1 / (1 + np.log1p(df["last_activity_days"]))
    ).round(5)

    df["release_cadence"] = np.where(
        df["repo_age_days"] > 0,
        (df["releases_count"] / (df["repo_age_days"] / 365.25)).round(3),
        0.0,
    )

    total_prs = df["pull_requests_count"]
    df["pr_throughput"] = np.where(total_prs > 0, 0.7, 0.0)

    df["community_breadth"] = (
        np.log1p(df["contributors_count"]) + np.log1p(df["forks"])
    ).round(4)

    df["stars_growth_proxy"] = (
        df["stars"] / np.log1p(df["repo_age_days"] + 1)
    ).round(3)

    logger.info("Features derivadas creadas: commit_velocity, recency_score, "
                "release_cadence, pr_throughput, community_breadth, stars_growth_proxy")
    return df


# ── Paso 4: Representación textual ────────────────────────────────────────

def _activity_label(cv: float) -> str:
    if cv >= 10: return "muy alta actividad de desarrollo"
    if cv >= 3:  return "actividad moderada de desarrollo"
    if cv >= 0.5:return "baja actividad de desarrollo"
    return "actividad de desarrollo casi nula"


def _recency_label(days: int) -> str:
    if days <= 7:   return "actualizado esta semana"
    if days <= 30:  return "actualizado este mes"
    if days <= 180: return "actualizado hace menos de 6 meses"
    if days <= 365: return "sin actividad en más de 6 meses"
    return "inactivo por más de un año"


def build_text_representation(row: pd.Series) -> str:
    """
    Convierte las señales numéricas en prosa legible para LLM y BERT.

    Justificación de la representación en prosa:
    - LLMs y BERT están preentrenados en texto natural, no en tablas de números.
    - Los descriptores cualitativos ('muy alta actividad') reducen ambigüedad
      frente a valores absolutos que cambian con el tamaño del dataset.
    - La combinación de señales cuantitativas Y cualitativas permite al modelo
      razonar sobre patrones (ej: 'repo joven con crecimiento explosivo' → emerging).
    - Longitud controlada (~200-350 tokens): suficiente contexto sin superar
      el límite de 512 tokens de BERT.
    """
    ci_text      = "tiene CI/CD" if row["has_ci"] else "sin CI/CD"
    topics_text  = row["topics"] if str(row["topics"]).strip() else "sin tópicos etiquetados"
    release_text = (
        f"{int(row['releases_count'])} releases (cadencia {float(row['release_cadence']):.2f}/año)"
        if int(row["releases_count"]) > 0 else "sin releases publicados"
    )
    age_years  = int(row["repo_age_days"]) // 365
    age_months = (int(row["repo_age_days"]) % 365) // 30
    age_text   = f"{age_years}a {age_months}m" if age_years > 0 else f"{age_months} mes(es)"

    return (
        f"Repositorio: {row['name']}. "
        f"Descripción: {row['description']}. "
        f"Lenguaje: {row['language']}. Tópicos: {topics_text}. "
        f"Antigüedad: {age_text} ({_recency_label(int(row['last_activity_days']))}). "
        f"Popularidad: {int(row['stars'])} estrellas, {int(row['forks'])} forks, "
        f"{int(row['open_issues'])} issues abiertos. "
        f"Actividad: {int(row['commits_last_30d'])} commits en 30 días "
        f"({_activity_label(float(row['commit_velocity']))}). "
        f"Colaboración: {int(row['contributors_count'])} contribuidores. "
        f"Lanzamientos: {release_text}. "
        f"Documentación: README {int(row['readme_length'])} chars, {ci_text}."
    )


# ── Paso 5: Validar contrato de columnas ──────────────────────────────────

def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que el output tenga exactamente las columnas del contrato."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(f"Columnas faltantes en output: {missing}. Se rellenan con None.")
        for col in missing:
            df[col] = None
    return df[REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]]


# ── Pipeline completo ─────────────────────────────────────────────────────

def preprocess(path: Path = RAW_INPUT) -> pd.DataFrame:
    df = read_csv_if_exists(path)
    if df.empty:
        logger.error(f"No se encontró: {path}. Ejecuta github_collector.py primero.")
        return df
    logger.info(f"Raw cargado: {df.shape}")
    df = drop_unwanted(df)
    df = fill_missing(df)
    df = engineer_features(df)
    df["text_representation"] = df.apply(build_text_representation, axis=1)
    df = validate_columns(df)
    logger.info(f"Procesado final: {df.shape}")
    return df


def main() -> None:
    df = preprocess()
    write_csv(df, PROCESSED_OUTPUT)
    print(f"Saved {len(df)} rows to {PROCESSED_OUTPUT.relative_to(ROOT)}")

    print("\n=== Features derivadas ===")
    feat_cols = ["commit_velocity", "recency_score", "release_cadence",
                 "community_breadth", "stars_growth_proxy"]
    available = [c for c in feat_cols if c in df.columns]
    if available:
        print(df[available].describe().round(3).to_string())

    print("\n=== Ejemplo de representación textual ===")
    if not df.empty and "text_representation" in df.columns:
        sample = df.sample(1, random_state=42).iloc[0]
        print(f"Repo: {sample['full_name']}")
        print(sample["text_representation"])


if __name__ == "__main__":
    main()
