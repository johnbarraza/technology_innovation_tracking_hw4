from __future__ import annotations

import base64
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

load_dotenv()

from src.utils import ROOT, write_csv, get_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_OUTPUT = ROOT / "data" / "raw" / "repositories_raw.csv"

GITHUB_TOKEN = get_env("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
BASE_URL = "https://api.github.com"

# ── Queries por categoría ──────────────────────────────────────────────────
# Cada entrada es (query_de_búsqueda, category_hint).
#
# Justificación del muestreo:
#   - Buscamos por query de tópico en lugar de por lenguaje o rango de estrellas
#     para obtener diversidad semántica alineada con las 4 categorías objetivo.
#   - Los tópicos son auto-asignados por los autores del repo, lo que los hace
#     señales directas de la intención del proyecto.
#   - category_hint es SOLO metadata de recolección, no es la etiqueta final.
#     Las etiquetas finales las genera el LLM en llm_labeling.py.
#   - Limitación principal: introduce sesgo de selección (repos sin tópicos quedan fuera)
#     y sobre-representa tecnologías de comunidades anglófonas en GitHub.

TECH_QUERIES: list[tuple[str, str]] = [
    # emerging_technology
    ("ai agents",                      "emerging_technology"),
    ("vector database",                "emerging_technology"),
    ("retrieval augmented generation", "emerging_technology"),
    ("llm agents",                     "emerging_technology"),
    ("multimodal ai",                  "emerging_technology"),
    # mature_ecosystem
    ("kubernetes",                     "mature_ecosystem"),
    ("react framework",                "mature_ecosystem"),
    ("mlops platform",                 "mature_ecosystem"),
    ("docker container",               "mature_ecosystem"),
    ("django web framework",           "mature_ecosystem"),
    # declining_technology
    ("jquery plugin",                  "declining_technology"),
    ("angularjs",                      "declining_technology"),
    ("grunt build tool",               "declining_technology"),
    ("coffeescript",                   "declining_technology"),
    ("bower package manager",          "declining_technology"),
    # experimental_niche
    ("quantum computing",              "experimental_niche"),
    ("robotics framework",             "experimental_niche"),
    ("neuromorphic computing",         "experimental_niche"),
    ("cybersecurity tooling",          "experimental_niche"),
    ("blockchain infrastructure",      "experimental_niche"),
]

# Queries adicionales — solo se ejecutan con collect_additional()
ADDITIONAL_QUERIES: list[tuple[str, str]] = [
    # declining_technology — nuevos
    ("backbone js",                 "declining_technology"),
    ("requirejs",                   "declining_technology"),
    ("gulp build",                  "declining_technology"),
    ("knockout js",                 "declining_technology"),
    ("flash actionscript",          "declining_technology"),
    # experimental_niche — nuevos
    ("homomorphic encryption",      "experimental_niche"),
    ("dna computing",               "experimental_niche"),
    ("swarm robotics",              "experimental_niche"),
    ("neuromorphic chip",           "experimental_niche"),
    ("photonic computing",          "experimental_niche"),
]

REPOS_PER_QUERY = 20
MIN_STARS       = 10


def _github_get(url: str, params: dict | None = None, retries: int = 5) -> Optional[dict | list]:
    """GET con manejo de rate-limit y backoff exponencial."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 403:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait  = max(reset - int(time.time()), 1) + 5
                logger.warning(f"Rate limit. Esperando {wait}s…")
                time.sleep(wait)
            elif r.status_code in (404, 409):
                return None
            else:
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            logger.error(f"Error de red: {e}")
            time.sleep(2 ** attempt)
    return None


def _contributors_count(full_name: str) -> int:
    data = _github_get(f"{BASE_URL}/repos/{full_name}/contributors",
                       {"per_page": 100, "anon": "true"})
    return len(data) if isinstance(data, list) else 0


def _commits_last_30d(full_name: str) -> int:
    """Endpoint /stats/participation devuelve 52 semanas. Toma las últimas 4 (~30 días)."""
    data = _github_get(f"{BASE_URL}/repos/{full_name}/stats/participation")
    if data and isinstance(data, dict) and "all" in data:
        return sum(data["all"][-4:])
    return 0


def _pull_requests_count(full_name: str) -> int:
    open_  = _github_get(f"{BASE_URL}/repos/{full_name}/pulls",
                         {"state": "open",   "per_page": 100})
    closed = _github_get(f"{BASE_URL}/repos/{full_name}/pulls",
                         {"state": "closed", "per_page": 100})
    return (len(open_) if isinstance(open_, list) else 0) + \
           (len(closed) if isinstance(closed, list) else 0)


def _releases_count(full_name: str) -> int:
    data = _github_get(f"{BASE_URL}/repos/{full_name}/releases", {"per_page": 100})
    return len(data) if isinstance(data, list) else 0


def _has_ci(full_name: str) -> bool:
    data = _github_get(f"{BASE_URL}/repos/{full_name}/contents/.github/workflows")
    return isinstance(data, list) and len(data) > 0


def _readme_length(full_name: str) -> int:
    data = _github_get(f"{BASE_URL}/repos/{full_name}/readme")
    if data and isinstance(data, dict) and "content" in data:
        try:
            return len(base64.b64decode(data["content"]).decode("utf-8", errors="ignore"))
        except Exception:
            return 0
    return 0


def _extract_signals(repo: dict, query: str, category_hint: str) -> dict:
    """
    Extrae las columnas definidas en REQUIRED_COLUMNS de preprocessing.py.

    Señales recolectadas (cumple el mínimo de 6 requerido por la tarea):
      1. stars              — popularidad acumulada
      2. forks              — adopción y reutilización
      3. open_issues        — actividad de la comunidad
      4. contributors_count — amplitud del ecosistema
      5. commits_last_30d   — momentum de desarrollo reciente
      6. pull_requests_count— colaboración externa
      7. releases_count     — cadencia de lanzamientos
      8. has_ci             — madurez de DevOps
      9. readme_length      — calidad de documentación
     10. repo_age_days      — madurez temporal
     11. last_activity_days — actividad reciente
    """
    fn  = repo["full_name"]
    now = datetime.now(timezone.utc)
    created = datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00"))
    pushed  = datetime.fromisoformat(repo["pushed_at"].replace("Z",  "+00:00"))

    topics_list = repo.get("topics") or []

    return {
        "repo_id":             repo["id"],
        "full_name":           fn,
        "owner":               repo["owner"]["login"],
        "name":                repo["name"],
        "html_url":            repo["html_url"],
        "description":         (repo.get("description") or "").strip(),
        "language":            repo.get("language") or "unknown",
        "topics":              ", ".join(topics_list),
        "primary_topic":       query,
        "stars":               repo["stargazers_count"],
        "forks":               repo["forks_count"],
        "open_issues":         repo["open_issues_count"],
        "contributors_count":  _contributors_count(fn),
        "commits_last_30d":    _commits_last_30d(fn),
        "pull_requests_count": _pull_requests_count(fn),
        "releases_count":      _releases_count(fn),
        "has_ci":              _has_ci(fn),
        "readme_length":       _readme_length(fn),
        "created_at":          repo["created_at"],
        "updated_at":          repo["updated_at"],
        "pushed_at":           repo["pushed_at"],
        "repo_age_days":       (now - created).days,
        "last_activity_days":  (now - pushed).days,
        "category_hint":       category_hint,
    }


def _search_repos(query: str, n: int = REPOS_PER_QUERY) -> list[dict]:
    data = _github_get(
        f"{BASE_URL}/search/repositories",
        {"q": f"{query} stars:>{MIN_STARS}", "sort": "updated",
         "order": "desc", "per_page": n},
    )
    return data.get("items", []) if isinstance(data, dict) else []


def collect_repositories() -> pd.DataFrame:
    """
    Recolecta repositorios de GitHub para Track B — Technology Innovation Tracking.

    Estrategia de muestreo:
    - Búsqueda por query de tópico, agrupada en 4 categorías de ecosistema.
    - Se excluyen forks (copias semánticas, no proyectos independientes).
    - Se deduplica por full_name para evitar repos repetidos entre queries.
    - category_hint se guarda como metadata, NO como etiqueta final.

    Limitaciones (requeridas por la tarea):
    - Los tópicos son auto-asignados: sesgo de selección posible.
    - Sobre-representa tecnologías de comunidades anglófonas en GitHub.
    - Rate-limit de la API: ~5.000 req/hora con token de acceso personal.
    """
    all_records: list[dict] = []
    seen: set[str] = set()

    for query, category_hint in TECH_QUERIES:
        logger.info(f"Query: '{query}' [{category_hint}]")
        repos = _search_repos(query)
        time.sleep(1.0)

        for repo in repos:
            fn = repo["full_name"]
            if fn in seen or repo.get("fork"):
                continue
            seen.add(fn)
            logger.info(f"  → {fn}")
            record = _extract_signals(repo, query, category_hint)
            all_records.append(record)
            time.sleep(0.4)

    df = pd.DataFrame(all_records)
    logger.info(f"Total recolectados: {len(df)} repos")
    if not df.empty and "category_hint" in df.columns:
        logger.info(f"Por categoría:\n{df['category_hint'].value_counts().to_string()}")
    return df


def collect_additional() -> pd.DataFrame:
    """
    Ejecuta solo ADDITIONAL_QUERIES y anexa los resultados al CSV existente.

    - Carga los full_name ya presentes en RAW_OUTPUT para deduplicar.
    - Usa mode='a' (append) sin sobrescribir el archivo existente.
    - Retorna solo las filas nuevas agregadas.
    """
    seen: set[str] = set()
    if RAW_OUTPUT.exists():
        existing = pd.read_csv(RAW_OUTPUT, usecols=["full_name"])
        seen = set(existing["full_name"].tolist())
        logger.info(f"CSV existente: {len(seen)} repos ya registrados")

    all_records: list[dict] = []

    for query, category_hint in ADDITIONAL_QUERIES:
        logger.info(f"Query adicional: '{query}' [{category_hint}]")
        repos = _search_repos(query)
        time.sleep(1.0)

        for repo in repos:
            fn = repo["full_name"]
            if fn in seen or repo.get("fork"):
                continue
            seen.add(fn)
            logger.info(f"  → {fn}")
            record = _extract_signals(repo, query, category_hint)
            all_records.append(record)
            time.sleep(0.4)

    df_new = pd.DataFrame(all_records)
    if df_new.empty:
        logger.info("No se encontraron repos nuevos para agregar.")
        return df_new

    write_header = not RAW_OUTPUT.exists()
    df_new.to_csv(RAW_OUTPUT, mode="a", header=write_header, index=False)
    logger.info(f"Agregados {len(df_new)} repos nuevos a {RAW_OUTPUT.relative_to(ROOT)}")
    if "category_hint" in df_new.columns:
        logger.info(f"Por categoría:\n{df_new['category_hint'].value_counts().to_string()}")
    return df_new


def main() -> None:
    import sys as _sys
    if "--additional" in _sys.argv:
        df = collect_additional()
        print(f"Appended {len(df)} new rows to {RAW_OUTPUT.relative_to(ROOT)}")
    else:
        df = collect_repositories()
        write_csv(df, RAW_OUTPUT)
        print(f"Saved {len(df)} rows to {RAW_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
