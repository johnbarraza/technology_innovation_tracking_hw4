# TODO

## Persona 1 — Data & Labels (Aiko)
- [ ] Crear token de GitHub y guardarlo en `.env`
- [ ] Implementar `src/github_collector.py` → output: `data/raw/repositories_raw.csv`
- [ ] Implementar `src/preprocessing.py`   → output: `data/processed/repositories.csv`
- [ ] Implementar `src/llm_labeling.py`    → output: `data/labeled/labeled_repositories.csv`
- [ ] Escribir notas sobre sesgo de selección

## Persona 2 — Model (nombre)
- [ ] Implementar `src/summarization.py`   → output: `data/processed/repository_summaries.csv`
- [ ] Implementar `src/train.py`
- [ ] Implementar `src/evaluation.py`

## Persona 3: App + Presentacion

- [x] Crear Streamlit con 4 tabs.
- [x] Crear funciones base de visualizacion.
- [x] Crear README inicial.
- [x] Crear `.env.example`.
- [x] Documentar regla de rutas relativas.
- [ ] Actualizar app con datos finales.
- [ ] Completar metricas reales en README.
- [ ] Preparar guion del video.
- [ ] Grabar video.
- [ ] Poner link final en `video/link.txt`.

## Antes De Entregar

- [ ] El repo tiene el nombre correcto.
- [ ] Se uso GitHub API.
- [ ] Hay al menos 6 senales de repositorio.
- [ ] Se implemento weak labeling con LLM.
- [ ] Existen splits train/validation/test.
- [ ] Se hizo fine-tuning de BERT.
- [ ] Hay accuracy, precision, recall y F1.
- [ ] Streamlit tiene exactamente 4 tabs.
- [ ] README explica metodologia y hallazgos.
- [ ] `video/link.txt` tiene link real.
- [ ] El trabajo fue hecho con branches y PRs.
