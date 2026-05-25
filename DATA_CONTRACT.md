# Data Contract

Este documento define las rutas y columnas que todo el equipo debe respetar para que las partes de datos, LLM, BERT, evaluacion y Streamlit encajen sin cambios manuales.

## Rutas Principales

Todas las rutas son relativas a la raiz del repositorio. No usar rutas absolutas de una computadora local.

```text
data/raw/repositories_raw.csv
data/processed/repositories.csv
data/labeled/labeled_repositories.csv
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
output/metrics/metrics.json
output/metrics/confusion_matrix.csv
output/metrics/classification_report.csv
output/tables/prediction_examples.csv
models/trained_models/
```

En Python, construir rutas con `ROOT` desde `src/utils.py`:

```python
from src.utils import ROOT

processed_path = ROOT / "data" / "processed" / "repositories.csv"
```

## `data/processed/repositories.csv`

Archivo producido por la Persona 1.

Columnas minimas:

```text
repo_id
full_name
owner
name
html_url
description
language
topics
stars
forks
open_issues
contributors_count
commits_last_30d
pull_requests_count
releases_count
has_ci
readme_length
created_at
updated_at
pushed_at
repo_age_days
last_activity_days
primary_topic
```

Notas:

- `topics` debe guardarse como texto separado por coma, por ejemplo `ai-agents,llm,automation`.
- `has_ci` debe ser booleano o 0/1.
- Fechas en formato ISO, por ejemplo `2026-05-24T20:10:00Z`.

## `data/labeled/labeled_repositories.csv`

Archivo producido por la Persona 2 despues de weak labeling.

Debe incluir todas las columnas de `repositories.csv` mas:

```text
repository_summary
weak_label
llm_rationale
llm_confidence
label
```

Categorias validas para `label`:

```text
emerging_technology
mature_ecosystem
declining_technology
experimental_niche
```

Regla:

- `weak_label` puede guardar la respuesta cruda del LLM.
- `label` debe guardar la categoria normalizada que usara BERT.

## Splits

Archivos producidos por la Persona 2.

```text
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
```

Columnas obligatorias:

```text
full_name
repository_summary
label
```

Pueden conservar columnas extra para analisis.

## `output/metrics/metrics.json`

Archivo producido por entrenamiento/evaluacion.

Formato esperado:

```json
{
  "accuracy": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "model_name": "distilbert-base-uncased",
  "baseline": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "alternative": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  }
}
```

## `output/metrics/confusion_matrix.csv`

Formato esperado:

```text
label,emerging_technology,mature_ecosystem,declining_technology,experimental_niche
emerging_technology,0,0,0,0
mature_ecosystem,0,0,0,0
declining_technology,0,0,0,0
experimental_niche,0,0,0,0
```

## `output/metrics/classification_report.csv`

Formato esperado:

```text
label,precision,recall,f1-score,support
emerging_technology,0.0,0.0,0.0,0
mature_ecosystem,0.0,0.0,0.0,0
declining_technology,0.0,0.0,0.0,0
experimental_niche,0.0,0.0,0.0,0
```

## `output/tables/prediction_examples.csv`

Formato esperado:

```text
full_name,repository_summary,true_label,predicted_label,is_correct,notes
```

Debe incluir ejemplos correctos e incorrectos para la app y el video.
