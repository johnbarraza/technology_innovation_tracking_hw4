# Project Guide

## Track Seleccionado

Track B: Technology Innovation & Ecosystem Tracking.

Objetivo: clasificar repositorios de GitHub segun momentum tecnologico:

- `emerging_technology`
- `mature_ecosystem`
- `declining_technology`
- `experimental_niche`

## Division Del Trabajo

### Persona 1: GitHub Data Collection

Archivos:

```text
src/github_collector.py
src/preprocessing.py
src/utils.py
data/raw/
data/processed/
```

Entregables:

- Usar GitHub API REST o GraphQL.
- Recolectar repositorios por areas tecnologicas.
- Extraer minimo 6 senales.
- Guardar `data/processed/repositories.csv`.
- Documentar sesgo de seleccion.

Branch sugerida:

```text
feature/github-data-collection
```

### Persona 2: LLM Labeling, BERT Training, Evaluation

Archivos:

```text
src/summarization.py
src/llm_labeling.py
src/train.py
src/evaluation.py
data/labeled/
data/splits/
models/trained_models/
output/metrics/
output/tables/
```

Entregables:

- Crear summaries de repositorios.
- Disenar prompt para weak labeling.
- Generar etiquetas con LLM.
- Crear splits train/validation/test.
- Fine-tunear DistilBERT, MiniLM u otro modelo liviano.
- Guardar metricas, matriz de confusion y ejemplos.

Branch sugerida:

```text
feature/llm-bert-pipeline
```

### Persona 3: App, Visualizaciones, README, Video

Archivos:

```text
app.py
src/visualization.py
README.md
requirements.txt
video/link.txt
```

Entregables:

- Streamlit con exactamente 4 tabs.
- Visualizaciones EDA y resultados.
- README completo.
- Guion/pitch del video.
- Link final en `video/link.txt`.

Branch actual:

```text
feature/app-presentation
```

## Reglas De Integracion

- No trabajar directamente en `main`.
- Cada persona trabaja en su branch.
- No cambiar rutas ni columnas definidas en `DATA_CONTRACT.md` sin avisar al equipo.
- Usar rutas relativas al repo. No usar rutas absolutas como `C:\Users\...` dentro del codigo.
- Para rutas en Python, usar `pathlib` y el `ROOT` definido en `src/utils.py`.
- La app debe leer archivos ya generados, no recalcular el pipeline completo.
- Las API keys van en `.env`, nunca en Git. El archivo `.env.example` muestra los nombres esperados.
- Los datasets finales deben ser pequenos o medianos para que el repo sea reproducible.
- Si un archivo todavia no existe, el codigo debe mostrar un mensaje claro y no romperse.

Ejemplo correcto para rutas:

```python
from src.utils import ROOT

path = ROOT / "data" / "processed" / "repositories.csv"
```

Ejemplo incorrecto:

```python
path = "C:\\Users\\johnb\\Documents\\Github\\technology_innovation_tracking_hw4\\data\\processed\\repositories.csv"
```

## Variables De Entorno

Copiar `.env.example` a `.env` localmente y completar solo las claves necesarias.

```text
GITHUB_TOKEN=
OPENAI_API_KEY=
GOOGLE_API_KEY=
DEEPSEEK_API_KEY=

MODEL_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Reglas:

- No commitear `.env`.
- No escribir tokens directamente en notebooks, scripts o README.
- Si se cambia el nombre de una variable, actualizar `.env.example` y avisar al equipo.
- Para DeepSeek, completar `DEEPSEEK_API_KEY` y dejar `MODEL_PROVIDER=deepseek`.

## Orden Recomendado

1. Persona 1 genera `data/processed/repositories.csv`.
2. Persona 2 genera `data/labeled/labeled_repositories.csv`.
3. Persona 2 genera `data/splits/`.
4. Persona 2 entrena y genera `output/metrics/`.
5. Persona 3 actualiza README, app y video con resultados finales.

## Comandos Utiles

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar app:

```bash
streamlit run app.py
```

Ejecutar recoleccion:

```bash
python src/github_collector.py
```

Ejecutar preprocessing:

```bash
python src/preprocessing.py
```

Ejecutar summaries:

```bash
python src/summarization.py
```

Ejecutar labeling:

```bash
python src/llm_labeling.py
```

Ejecutar entrenamiento:

```bash
python src/train.py
```

Ejecutar evaluacion:

```bash
python src/evaluation.py
```
