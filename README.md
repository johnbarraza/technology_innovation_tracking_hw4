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

The planned split is:

- 70% train
- 15% validation
- 15% test

The test set remains unseen during model training.

## BERT Model

The planned model is a lightweight BERT-based classifier such as DistilBERT or MiniLM. The input is the repository representation, and the output is the predicted technology momentum category.

## Final Metrics

Final metrics will be added after training:

- Accuracy: pending
- Precision: pending
- Recall: pending
- F1-score: pending

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
