# GitHub Technology Innovation Tracking

This project builds a weak-supervision NLP pipeline to classify GitHub repositories by technology momentum. The selected track is **Track B: Technology Innovation & Ecosystem Tracking**.

## What The Project Does

The system collects GitHub repository metadata, converts each repository into a textual representation, uses an LLM to generate weak labels, fine-tunes a lightweight BERT-based classifier, evaluates model performance, and presents the results through a Streamlit application.

## Repository Selection

Repositories are selected from technology-focused GitHub searches around AI agents, vector databases, cybersecurity tooling, blockchain infrastructure, robotics frameworks, and MLOps platforms.

This sampling strategy is useful for analyzing visible open-source technology ecosystems, but it may underrepresent private, enterprise, or very early-stage innovation.

## GitHub Signals

The planned repository-level signals include:

- Stars and forks
- Number of contributors
- Issue activity
- Pull request activity
- Release frequency
- Repository topics/tags
- Repository age
- Last activity date
- README characteristics
- Workflow/CI presence
- Recent commit activity

## Repository Summaries

Each repository is converted into a structured textual summary that combines metadata and activity signals. Example:

```text
Repository focused on vector databases. It has high recent activity, multiple contributors, active issues, regular releases, and topics related to embeddings and retrieval.
```

This representation is useful because both LLMs and BERT-based classifiers can consume text while still preserving structured repository signals.

## Prompt Design

The LLM prompt includes:

- The repository summary
- Definitions of the target categories
- Instructions to return one label
- A short rationale for the label

The target categories are:

- Emerging technology
- Mature ecosystem
- Declining technology
- Experimental or niche area

LLM labels are treated as weak labels rather than ground truth.

## Dataset Split

- **Train:** 368 repositories (70%)
- **Validation:** 79 repositories (15%)
- **Test:** 79 repositories (15%)

The test set remained unseen during training and was used only for final evaluation.

## BERT Model

DistilBERT (`distilbert-base-uncased`) fine-tuned on 526 repository text representations. Training: 70% train / 15% validation / 15% test, 3 epochs, batch size 16, learning rate 2e-5, max length 512 tokens. GPU: Google Colab T4.

## Final Metrics

Test set (79 repositories):

| Metric | Weighted | Macro |
|---|---|---|
| Accuracy | **58.23%** | — |
| Precision | 0.6378 | 0.6504 |
| Recall | 0.5823 | 0.5808 |
| F1-Score | 0.5765 | 0.5835 |

Per-class performance:

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| declining | 0.6842 | 0.7647 | **0.7222** | 17 |
| emerging | 0.6923 | 0.5000 | 0.5806 | 18 |
| experimental | 0.4474 | 0.7083 | 0.5484 | 24 |
| mature | 0.7778 | 0.3500 | 0.4828 | 20 |

Best predicted: declining (F1=0.72). Hardest: mature (F1=0.48), heavily confused with experimental (10 of 20 mature repos misclassified as experimental). The baseline model (370 repos, no class balancing) achieved 42.86% accuracy and could not predict declining or experimental at all. Adding 156 repos with more declining/experimental samples improved accuracy by 15.4pp and enabled all-class prediction.

## Main Limitations

GitHub activity is an imperfect proxy for innovation. Stars can reflect visibility rather than technical value, recent activity may be seasonal, and LLM-generated labels can contain prompt bias or category ambiguity.

## Business Applications

This system could help investors, consulting firms, governments, and technology researchers monitor open-source technology momentum, identify emerging ecosystems, and compare technical areas using repeatable repository-level signals.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Environment Variables

Copy `.env.example` to `.env` and fill only the keys needed for your part of the pipeline. Do not commit `.env`.

The project should use paths relative to the repository root. Python scripts should build paths with `ROOT` from `src/utils.py`, not with absolute local paths.

Default LLM provider is DeepSeek:

```text
MODEL_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_key_here
```

## Expected Pipeline Outputs

The Streamlit app reads these files when they are available:

```text
data/processed/repositories.csv
data/labeled/labeled_repositories.csv
data/splits/test.csv
output/metrics/metrics.json
output/metrics/confusion_matrix.csv
output/metrics/classification_report.csv
output/tables/prediction_examples.csv
```

If files are missing, the app shows a clear placeholder instead of crashing.
