# DATA_DECISIONS.md
## Persona 1 — Documentación de decisiones y variables
### Aiko | Track B: Technology Innovation & Ecosystem Tracking

---

## ¿Qué hice yo y qué te entrego?

Recolecté 371 repositorios de GitHub, los limpié, creé features derivadas y generé una representación textual para que el LLM los etiquete. Te entrego tres archivos:

```
data/raw/repositories_raw.csv        → datos crudos de la API de GitHub
data/processed/repositories.csv      → datos limpios + features + texto
data/labeled/labeled_repositories.csv → datos con etiquetas del LLM (ver nota abajo)
```

> ⚠️ **Nota para Persona 2:** `labeled_repositories.csv` se genera corriendo `python src/llm_labeling.py`. El script ya está completamente implementado y probado — **no necesitas tocar nada del código**. Solo tienes que:
> 1. Recargar saldo en [platform.deepseek.com](https://platform.deepseek.com/) → Billing (cuesta aprox. $0.10–$0.30 para los 370 repos)
> 2. Asegurarte de que el `.env` tenga `DEEPSEEK_API_KEY=tu_key`
> 3. Correr `python src/llm_labeling.py` y esperar ~45 minutos
>
> El script tiene reanudación automática: si se interrumpe, al volver a correrlo continúa desde donde se quedó sin repetir los repos ya etiquetados.

---

## Categorías del sistema (etiquetas exactas)

Estas son las 4 etiquetas que usa el LLM y que BERT debe aprender a predecir:

| Etiqueta | Qué significa |
|---|---|
| `emerging_technology` | Tecnología nueva con crecimiento rápido y alta actividad reciente |
| `mature_ecosystem` | Ecosistema consolidado, estable, con amplia adopción |
| `declining_technology` | Antes activo, ahora con señales claras de reducción de interés |
| `experimental_niche` | Nicho académico o de investigación, baja adopción, actividad irregular |

---

## Estrategia de muestreo (por qué elegí esos repos)

Busqué por **query de tópico** en la API de GitHub, agrupados en 4 bloques que corresponden a las categorías:

| Categoría | Queries usados |
|---|---|
| emerging_technology | ai agents, vector database, retrieval augmented generation, llm agents, multimodal ai |
| mature_ecosystem | kubernetes, react framework, mlops platform, docker container, django web framework |
| declining_technology | jquery plugin, angularjs, grunt build tool, coffeescript, bower package manager |
| experimental_niche | quantum computing, robotics framework, neuromorphic computing, cybersecurity tooling, blockchain infrastructure |

**Filtros aplicados:**
- Mínimo 10 estrellas (descarta repos casi vacíos)
- Sin forks (son copias, no proyectos independientes)
- Sin duplicados por `full_name`
- Descripción mínima de 10 caracteres (necesario para el LLM)

**Resultado:** 371 repos → 370 después de limpieza

---

## Señales recolectadas (columnas del CSV)

### Señales directas de la API de GitHub

| Columna | Qué mide |
|---|---|
| `stars` | Popularidad acumulada |
| `forks` | Cuántos proyectos derivados existen |
| `open_issues` | Actividad de la comunidad |
| `contributors_count` | Amplitud del ecosistema |
| `commits_last_30d` | Actividad de desarrollo reciente |
| `pull_requests_count` | Colaboración externa total |
| `releases_count` | Cadencia de lanzamientos |
| `has_ci` | Si tiene GitHub Actions / CI-CD |
| `readme_length` | Calidad de documentación (en caracteres) |
| `repo_age_days` | Madurez temporal |
| `last_activity_days` | Días desde el último push |
| `topics` | Tópicos auto-asignados por el autor |
| `language` | Lenguaje principal |

### Features derivadas (creadas en preprocessing.py)

Estas son las más importantes para el modelo. No vienen de la API — las calculé yo:

| Columna | Fórmula | Por qué es útil |
|---|---|---|
| `commit_velocity` | `commits_last_30d / 4` | Commits por semana. Alta = proyecto activo ahora |
| `recency_score` | `1 / (1 + log(last_activity_days + 1))` | Cercano a 1 = activo recientemente. Clave para detectar declining |
| `release_cadence` | `releases_count / (repo_age_days / 365)` | Releases por año. Regular = mature |
| `pr_throughput` | `closed_prs / total_prs` | Qué tan bien procesan contribuciones externas |
| `community_breadth` | `log(contributors+1) + log(forks+1)` | Amplitud de la comunidad. Alta = mature o emerging grande |
| `stars_growth_proxy` | `stars / log(repo_age_days + 1)` | Crecimiento relativo a la edad. Alto en repo joven = emerging |

---

## Representación textual para BERT y LLM

Cada repo se convierte en una oración en lenguaje natural. Esta columna se llama `text_representation` en el CSV procesado. Ejemplo:

```
Repositorio: langchain. Descripción: Building applications with LLMs.
Lenguaje: Python. Tópicos: llm, ai, agents. Antigüedad: 2a 3m
(actualizado esta semana). Popularidad: 72000 estrellas, 11000 forks,
523 issues abiertos. Actividad: 380 commits en 30 días (muy alta
actividad de desarrollo). Colaboración: 450 contribuidores.
Lanzamientos: 48 releases (cadencia 18.20/año). Documentación:
README 45000 chars, tiene CI/CD.
```

**¿Por qué en prosa y no en tabla de números?**
Porque BERT y los LLMs fueron entrenados en texto natural. La frase "muy alta actividad de desarrollo" le da más contexto al modelo que el número `95.0` sin contexto.

---

## Output del LLM (columnas que genera llm_labeling.py)

Cuando corras `llm_labeling.py`, el CSV de labeled tendrá estas columnas adicionales:

| Columna | Qué contiene |
|---|---|
| `weak_label` | Etiqueta cruda que devolvió el LLM |
| `label` | Etiqueta normalizada (usa `normalize_label()` de utils.py) |
| `llm_confidence` | Confianza del LLM (0.0 a 1.0) |
| `llm_rationale` | Una oración explicando por qué eligió esa etiqueta |

> **Tip para Persona 2:** puedes filtrar `llm_confidence < 0.6` para excluir etiquetas dudosas antes de entrenar BERT. Son los casos donde el LLM no estaba seguro.

---

## Cómo correr mi parte desde cero

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Crear .env con tus keys (ver .env.example)
cp .env.example .env
# editar .env y poner GITHUB_TOKEN y DEEPSEEK_API_KEY

# 3. Recolectar repos de GitHub (~20 minutos)
python src/github_collector.py

# 4. Limpiar y crear features (~segundos)
python src/preprocessing.py

# 5. Etiquetar con DeepSeek (~45 minutos, reanudable)
python src/llm_labeling.py
```

---

## Limitaciones conocidas (importante para el README y el video)

- Los tópicos de GitHub son **auto-asignados** por los autores → sesgo de selección posible
- `declining_technology` tiene menos repos (77) porque hay menos repos de tecnologías en declive con tópicos bien etiquetados
- Las etiquetas del LLM son **weak labels**, no ground truth — el modelo puede equivocarse en casos limítrofes
- GitHub solo muestra código abierto → proyectos privados no están representados
- `community_breadth` incluye forks automáticos (mirrors, backups) que no son uso real

---

## Archivos que toco yo (Persona 1)

```
src/github_collector.py   ← recolección de datos
src/preprocessing.py      ← limpieza y features
src/llm_labeling.py       ← etiquetado débil con DeepSeek
data/raw/                 ← mi output crudo
data/processed/           ← mi output procesado
data/labeled/             ← mi output final (input de Persona 2)
```

## Archivos que NO toco (de otros compañeros)

```
src/summarization.py      ← Persona 2
src/train.py              ← Persona 2
src/evaluation.py         ← Persona 2
src/visualization.py      ← Persona 3
app.py                    ← Persona 3
```
