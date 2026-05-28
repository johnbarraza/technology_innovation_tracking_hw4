from __future__ import annotations

import json
import re
import time
import logging
from pathlib import Path
import sys

import pandas as pd
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, normalize_label, read_csv_if_exists, write_csv, get_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Rutas ──────────────────────────────────────────────────────────────────
SUMMARY_INPUT  = ROOT / "data" / "processed" / "repository_summaries.csv"
PROCESSED_INPUT= ROOT / "data" / "processed" / "repositories.csv"
LABELED_OUTPUT = ROOT / "data" / "labeled"   / "labeled_repositories.csv"

# ── Categorías (etiquetas exactas del skeleton) ────────────────────────────
CATEGORY_DEFINITIONS = {
    "emerging_technology":  "High recent activity and growing interest in a newer technical area.",
    "mature_ecosystem":     "Stable, broad adoption with sustained maintenance and community activity.",
    "declining_technology": "Older area with weak recent activity or decreasing maintenance signals.",
    "experimental_niche":   "Small, specialized, exploratory, or early-stage technical area.",
}


# ── Cliente DeepSeek ───────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    """
    DeepSeek usa la misma interfaz que OpenAI.
    Solo cambia la base_url y el api_key.
    """
    api_key  = get_env("DEEPSEEK_API_KEY")
    base_url = get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY no está en .env\n"
            "Regístrate en platform.deepseek.com y crea una API key."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


# ── Prompt ─────────────────────────────────────────────────────────────────

def build_labeling_prompt(repository_summary: str) -> str:
    """
    Construye el prompt para clasificar un repositorio.

    Decisiones de diseño justificadas:
    - Definiciones explícitas por categoría + guía de señales:
      reduce ambigüedad en casos limítrofes (mature vs declining,
      emerging vs experimental).
    - JSON obligatorio con label + confidence + rationale:
        · label:      etiqueta normalizada para el entrenamiento de BERT
        · confidence: permite filtrar etiquetas de baja calidad (< 0.6)
        · rationale:  útil para el análisis de errores del Stage 6
    - Instrucción de responder SOLO JSON: evita texto extra que
      complicaría el parsing automático.
    - temperature=0 en la llamada: máxima consistencia entre ejecuciones,
      lo que hace el etiquetado reproducible.

    Limitaciones del etiquetado débil:
    - El LLM puede tener sesgo hacia tecnologías que conoce mejor.
    - No ve el historial completo del repo, solo el resumen textual.
    - Las etiquetas son weak labels, no ground truth verificado por humanos.
    """
    categories = "\n".join(
        f"- {label}: {definition}"
        for label, definition in CATEGORY_DEFINITIONS.items()
    )
    return f"""Classify this GitHub repository into exactly one category.

Categories:
{categories}

Signal guidance:
- emerging_technology: repo < 4 years, high recent commits, fast star growth relative to age.
- mature_ecosystem: repo > 5 years, many contributors, regular releases, high PR activity.
- declining_technology: was active before, now low commits, stale issues, infrequent pushes.
- experimental_niche: small community, irregular activity, academic or research context.

Repository summary:
{repository_summary}

Return ONLY a JSON object with these exact keys: label, confidence, rationale.
No text outside the JSON. No markdown code blocks.
"""


# ── Parsing ────────────────────────────────────────────────────────────────

def _parse_response(text: str) -> dict | None:
    """Extrae y valida el JSON de la respuesta del LLM."""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    if not {"label", "confidence", "rationale"}.issubset(data.keys()):
        return None

    raw_label = normalize_label(str(data["label"]))
    if raw_label not in CATEGORY_DEFINITIONS:
        return None

    try:
        conf = float(data["confidence"])
        conf = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        conf = 0.5

    return {
        "label":      raw_label,
        "confidence": round(conf, 4),
        "rationale":  str(data["rationale"]).strip(),
    }


# ── Llamada al LLM ─────────────────────────────────────────────────────────

def label_with_llm(repository_summary: str) -> dict:
    """
    Llama a DeepSeek para etiquetar un repositorio.

    DeepSeek es compatible con la API de OpenAI, solo cambia
    base_url y model. Usamos deepseek-chat que es el modelo
    estándar de chat, suficiente para clasificación de texto.

    Retorna dict con claves: label, confidence, rationale.
    En caso de fallo total retorna label='unknown' para
    identificar estos casos en el análisis de errores.
    """
    client = _get_client()
    model  = get_env("LLM_MODEL", "deepseek-chat")
    prompt = build_labeling_prompt(repository_summary)

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0,   # determinismo para reproducibilidad
            )
            raw_text = response.choices[0].message.content
            parsed   = _parse_response(raw_text)

            if parsed:
                return parsed

            logger.warning(f"  Parse fallido intento {attempt+1}. "
                           f"Raw: {raw_text[:80]!r}")
            time.sleep(1.0)

        except Exception as e:
            logger.error(f"  Error intento {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    logger.error("  Todos los intentos fallaron. Registrando como 'unknown'.")
    return {
        "label":      "unknown",
        "confidence": 0.0,
        "rationale":  "labeling failed after all retries",
    }


# ── Loop de etiquetado ─────────────────────────────────────────────────────

def label_repositories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etiqueta todos los repositorios del DataFrame.

    Características:
    - Soporte de reanudación: si el output ya existe, omite los ya
      etiquetados. Permite continuar si el script se interrumpe.
    - Guarda cada 20 repos para no perder trabajo.
    - Respeta las claves del skeleton: weak_label, llm_rationale,
      llm_confidence, label.
    """
    if df.empty:
        output = df.copy()
        for col in ["weak_label", "llm_rationale", "llm_confidence", "label"]:
            output[col] = []
        return output

    # Reanudación
    existing   = read_csv_if_exists(LABELED_OUTPUT)
    done_names: set[str] = set()
    if not existing.empty and "full_name" in existing.columns:
        done_names = set(existing["full_name"].tolist())
        logger.info(f"Reanudando: {len(done_names)} ya etiquetados, "
                    f"{len(df) - len(done_names)} pendientes.")

    rows       = []
    save_every = 20

    for i, (_, row) in enumerate(df.iterrows(), 1):
        fn = row.get("full_name", "")
        if fn in done_names:
            continue

        # Usa repository_summary si existe, si no usa text_representation
        summary = row.get("repository_summary") or row.get("text_representation", "")
        logger.info(f"[{i}/{len(df)}] {fn}")

        result = label_with_llm(summary)

        labeled_row = row.to_dict()
        labeled_row["weak_label"]     = result["label"]
        labeled_row["llm_rationale"]  = result["rationale"]
        labeled_row["llm_confidence"] = result["confidence"]
        labeled_row["label"]          = normalize_label(result["label"])
        rows.append(labeled_row)

        logger.info(f"  → {result['label']} (conf={result['confidence']:.2f})")

        # Guardado incremental
        if len(rows) % save_every == 0:
            _save_incremental(existing, rows, df)
            logger.info(f"  Guardado incremental ({len(rows)} nuevos).")

        time.sleep(0.5)

    return _save_incremental(existing, rows, df)


def _save_incremental(existing: pd.DataFrame,
                      new_rows: list[dict],
                      original_df: pd.DataFrame) -> pd.DataFrame:
    """Fusiona nuevos resultados con los existentes y guarda."""
    if not new_rows:
        return existing

    new_df = pd.DataFrame(new_rows)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset="full_name", keep="last")
    write_csv(combined, LABELED_OUTPUT)
    return combined


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Intenta leer summaries de Persona 2; si no existen, usa processed
    df = read_csv_if_exists(SUMMARY_INPUT)
    if df.empty:
        logger.info("repository_summaries.csv no encontrado. "
                    "Usando repositories.csv (text_representation).")
        df = read_csv_if_exists(PROCESSED_INPUT)

    if df.empty:
        logger.error("No hay datos. Ejecuta github_collector.py y preprocessing.py primero.")
        return

    labeled_df = label_repositories(df)
    write_csv(labeled_df, LABELED_OUTPUT)
    print(f"Saved {len(labeled_df)} rows to {LABELED_OUTPUT.relative_to(ROOT)}")

    if not labeled_df.empty and "label" in labeled_df.columns:
        print("\nDistribución de etiquetas:")
        print(labeled_df["label"].value_counts().to_string())
        unknown = (labeled_df["label"] == "unknown").sum()
        if unknown > 0:
            print(f"\n⚠ {unknown} repos con label 'unknown' (falló el etiquetado).")
        low_conf = (labeled_df["llm_confidence"] < 0.6).sum()
        if low_conf > 0:
            print(f"⚠ {low_conf} repos con confianza < 0.6 (revisar manualmente).")


if __name__ == "__main__":
    main()
