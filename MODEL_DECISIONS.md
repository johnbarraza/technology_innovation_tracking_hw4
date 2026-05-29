# Model Decisions — Persona 2
## Track B: Technology Innovation & Ecosystem Tracking

## Modelo utilizado
DistilBERT (distilbert-base-uncased) — modelo ligero basado en BERT con 66M parámetros. Se eligió por su eficiencia en Colab con GPU T4 y su buen balance entre rendimiento y velocidad de entrenamiento.

## Etiquetado débil (Weak Supervision)
Se usó DeepSeek API para etiquetar 370 repositorios de GitHub con 4 categorías:
- **emerging** → tecnología emergente con crecimiento activo
- **mature** → ecosistema maduro y estable
- **declining** → tecnología en declive o abandono
- **experimental** → nicho experimental o investigación

Las etiquetas originales de DeepSeek (`emerging_technology`, `mature_ecosystem`, etc.) fueron mapeadas a nombres cortos antes del entrenamiento.

## Variables de entrada
La columna `text_representation` generada por `summarization.py` incluye:
- Nombre y descripción del repositorio
- Lenguaje principal y tópicos
- Estrellas, forks y contribuidores
- Commits últimos 30 días, PRs, releases
- Presencia de CI/CD
- Antigüedad y última actividad
- Métricas derivadas: commit_velocity, recency_score, release_cadence, pr_throughput, community_breadth, stars_growth_proxy

## Hiperparámetros
- Épocas: 3
- Batch size: 16
- Learning rate: 2e-5
- Weight decay: 0.01
- Warmup ratio: 0.1
- Max length: 512 tokens
- Split: 70% train / 15% val / 15% test

## Limitaciones
- Dataset pequeño (370 repos) — puede afectar la generalización del modelo
- Las etiquetas de DeepSeek son "débiles" — pueden contener ruido
- La clase `declining` tiene solo 31 muestras vs ~115 de las otras — desbalance de clases
- El modelo no fue entrenado con datos en tiempo real — puede no reflejar tendencias actuales
- `text_representation` es en español pero DistilBERT fue entrenado en inglés — puede reducir precisión

## Archivos generados
- `models/trained_models/` → modelo entrenado (excluido de GitHub por tamaño)
- `models/trained_models/test_set.csv` → conjunto de prueba
- `models/trained_models/label_meta.json` → metadatos de etiquetas
- `output/metrics/confusion_matrix.png` → matriz de confusión
- `output/metrics/confusion_matrix_normalized.png` → matriz normalizada
- `output/metrics/per_class_metrics.png` → métricas por clase
- `output/metrics/metrics.csv` → métricas en CSV
- `output/metrics/classification_report.txt` → reporte completo
