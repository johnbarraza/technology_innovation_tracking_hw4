# TODO

## Persona 1: Datos

- [ ] Crear token de GitHub y guardarlo en `.env`.
- [ ] Usar solo rutas relativas con `ROOT` de `src/utils.py`.
- [ ] Definir queries de busqueda por tecnologia.
- [ ] Implementar `src/github_collector.py`.
- [ ] Extraer minimo 6 senales por repositorio.
- [ ] Guardar `data/raw/repositories_raw.csv`.
- [ ] Implementar `src/preprocessing.py`.
- [ ] Guardar `data/processed/repositories.csv` con el contrato de columnas.
- [ ] Escribir notas sobre sesgo de seleccion.

## Persona 2: LLM + BERT

- [ ] Copiar `.env.example` a `.env` y completar la API key usada.
- [ ] Usar solo rutas relativas con `ROOT` de `src/utils.py`.
- [ ] Implementar `src/summarization.py`.
- [ ] Crear `repository_summary`.
- [ ] Implementar `src/llm_labeling.py`.
- [ ] Definir prompt y categorias.
- [ ] Guardar `data/labeled/labeled_repositories.csv`.
- [ ] Crear train/validation/test.
- [ ] Implementar `src/train.py`.
- [ ] Fine-tunear modelo BERT liviano.
- [ ] Implementar `src/evaluation.py`.
- [ ] Guardar metricas, matriz de confusion y ejemplos.

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
