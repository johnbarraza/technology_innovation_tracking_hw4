from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.visualization import (
    plot_category_distribution,
    plot_confusion_matrix,
    plot_numeric_signal_by_category,
    plot_repository_activity,
)


ROOT = Path(__file__).parent
PROCESSED_DATA = ROOT / "data" / "processed" / "repositories.csv"
LABELED_DATA = ROOT / "data" / "labeled" / "labeled_repositories.csv"
TEST_DATA = ROOT / "data" / "splits" / "test.csv"
METRICS_JSON = ROOT / "output" / "metrics" / "metrics.json"
CONFUSION_MATRIX = ROOT / "output" / "metrics" / "confusion_matrix.csv"
CLASSIFICATION_REPORT = ROOT / "output" / "metrics" / "classification_report.csv"
PREDICTION_EXAMPLES = ROOT / "output" / "tables" / "prediction_examples.csv"


st.set_page_config(
    page_title="GitHub Technology Innovation Tracking",
    page_icon="",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def show_missing_data_notice(path: Path, purpose: str) -> None:
    st.info(f"Waiting for `{path.as_posix()}` to show {purpose}.")


repos_df = load_csv(PROCESSED_DATA)
labeled_df = load_csv(LABELED_DATA)
test_df = load_csv(TEST_DATA)
metrics = load_json(METRICS_JSON)
confusion_df = load_csv(CONFUSION_MATRIX)
report_df = load_csv(CLASSIFICATION_REPORT)
examples_df = load_csv(PREDICTION_EXAMPLES)

st.title("GitHub Technology Innovation Tracking")
st.caption("Weak-supervision NLP pipeline for repository-level technology momentum analysis.")

tab_problem, tab_eda, tab_results, tab_explorer = st.tabs(
    [
        "Problem & Methodology",
        "Exploratory Analysis",
        "Model Results",
        "Interactive Repository Exploration",
    ]
)


with tab_problem:
    st.header("Problem & Methodology")
    st.write(
        """
        This project analyzes GitHub repositories to classify technology areas as
        emerging, mature, declining, or experimental/niche. The system combines
        repository metadata, textual summaries, LLM-generated weak labels, and a
        fine-tuned lightweight BERT-based classifier.
        """
    )

    st.subheader("Repository Selection")
    st.write(
        """
        Repositories are selected from technology-focused GitHub searches around
        AI agents, vector databases, cybersecurity tooling, blockchain
        infrastructure, robotics frameworks, and MLOps platforms. This strategy
        favors visible open-source projects and may underrepresent private,
        early-stage, or enterprise-only innovation.
        """
    )

    st.subheader("GitHub Signals")
    st.write(
        """
        The pipeline is designed to use repository-level signals such as stars,
        forks, contributors, issue activity, pull request activity, releases,
        repository age, last activity date, README length, workflow/CI presence,
        topics, and recent commit activity.
        """
    )

    st.subheader("Prompt Strategy")
    st.write(
        """
        Each repository is converted into a structured textual representation.
        The LLM receives the summary plus category definitions and returns a weak
        label with a short rationale. These labels are treated as training
        signals, not as ground truth.
        """
    )

    st.subheader("Dataset Construction")
    st.write(
        """
        Labeled examples are split into train, validation, and test sets. The
        suggested split is 70% train, 15% validation, and 15% test, with the test
        set kept unseen until final evaluation.
        """
    )

    st.subheader("Limitations")
    st.write(
        """
        GitHub activity is only a proxy for technology momentum. Popularity can
        reflect marketing, documentation, or community effects rather than
        technical depth. LLM weak labels may also inherit prompt bias, recency
        bias, and ambiguity in category definitions.
        """
    )


with tab_eda:
    st.header("Exploratory Analysis")

    analysis_df = labeled_df if not labeled_df.empty else repos_df
    if analysis_df.empty:
        show_missing_data_notice(PROCESSED_DATA, "repository statistics and exploratory plots")
    else:
        st.subheader("Repository Statistics")
        col_count, col_stars, col_forks, col_categories = st.columns(4)

        label_col = first_existing_column(
            analysis_df, ["label", "category", "weak_label", "predicted_label"]
        )
        stars_col = first_existing_column(analysis_df, ["stars", "stargazers_count"])
        forks_col = first_existing_column(analysis_df, ["forks", "forks_count"])

        col_count.metric("Repositories", f"{len(analysis_df):,}")
        col_stars.metric(
            "Median Stars",
            "N/A" if stars_col is None else f"{analysis_df[stars_col].median():,.0f}",
        )
        col_forks.metric(
            "Median Forks",
            "N/A" if forks_col is None else f"{analysis_df[forks_col].median():,.0f}",
        )
        col_categories.metric(
            "Categories",
            "N/A" if label_col is None else f"{analysis_df[label_col].nunique():,}",
        )

        st.subheader("Category Distribution")
        if label_col is None:
            st.warning("No category column found yet.")
        else:
            st.pyplot(plot_category_distribution(analysis_df, label_col))
            st.write(
                "This visualization checks whether weak labels are balanced or dominated by a few categories."
            )

        st.subheader("Signal Comparisons")
        numeric_cols = analysis_df.select_dtypes(include="number").columns.tolist()
        if label_col and numeric_cols:
            default_index = numeric_cols.index(stars_col) if stars_col in numeric_cols else 0
            selected_signal = st.selectbox(
                "Numeric signal",
                numeric_cols,
                index=default_index,
            )
            st.pyplot(plot_numeric_signal_by_category(analysis_df, selected_signal, label_col))
            st.write(
                "Signal comparisons help identify which repository behaviors separate technology momentum categories."
            )
        elif numeric_cols:
            activity_col = st.selectbox("Numeric signal", numeric_cols)
            st.pyplot(plot_repository_activity(analysis_df, activity_col))
        else:
            st.warning("No numeric repository signals found yet.")

        with st.expander("Data Preview"):
            st.dataframe(analysis_df, use_container_width=True)


with tab_results:
    st.header("Model Results")

    st.subheader("Evaluation Metrics")
    if not metrics:
        show_missing_data_notice(METRICS_JSON, "accuracy, precision, recall, and F1-score")
    else:
        metric_cols = st.columns(4)
        for index, key in enumerate(["accuracy", "precision", "recall", "f1"]):
            value = metrics.get(key)
            metric_cols[index].metric(key.title(), "N/A" if value is None else f"{value:.3f}")

    st.subheader("Confusion Matrix")
    if confusion_df.empty:
        show_missing_data_notice(CONFUSION_MATRIX, "the confusion matrix")
    else:
        st.pyplot(plot_confusion_matrix(confusion_df))

    st.subheader("Category Performance")
    if report_df.empty:
        show_missing_data_notice(CLASSIFICATION_REPORT, "per-category precision, recall, and F1")
    else:
        st.dataframe(report_df, use_container_width=True)

    st.subheader("Baseline vs Alternative")
    baseline = metrics.get("baseline") if metrics else None
    alternative = metrics.get("alternative") if metrics else None
    if not baseline and not alternative:
        st.info("Waiting for baseline and alternative comparison from the training pipeline.")
    else:
        comparison_df = pd.DataFrame(
            [
                {"approach": "baseline", **(baseline or {})},
                {"approach": "alternative", **(alternative or {})},
            ]
        )
        st.dataframe(comparison_df, use_container_width=True)


with tab_explorer:
    st.header("Interactive Repository Exploration")

    explorer_df = labeled_df if not labeled_df.empty else repos_df
    if explorer_df.empty:
        show_missing_data_notice(LABELED_DATA, "repository search, filters, and predictions")
    else:
        name_col = first_existing_column(
            explorer_df, ["full_name", "repo_name", "name", "repository"]
        )
        label_col = first_existing_column(
            explorer_df, ["predicted_label", "label", "category", "weak_label"]
        )
        topic_col = first_existing_column(explorer_df, ["topics", "topic", "primary_topic"])

        search = st.text_input("Search repositories", "")
        filtered_df = explorer_df.copy()

        if label_col:
            selected_categories = st.multiselect(
                "Category",
                sorted(filtered_df[label_col].dropna().astype(str).unique().tolist()),
            )
            if selected_categories:
                filtered_df = filtered_df[
                    filtered_df[label_col].astype(str).isin(selected_categories)
                ]

        if topic_col:
            topic_query = st.text_input("Topic contains", "")
            if topic_query:
                filtered_df = filtered_df[
                    filtered_df[topic_col].astype(str).str.contains(
                        topic_query, case=False, na=False
                    )
                ]

        if search and name_col:
            filtered_df = filtered_df[
                filtered_df[name_col].astype(str).str.contains(search, case=False, na=False)
            ]

        st.dataframe(filtered_df, use_container_width=True)

        if not filtered_df.empty:
            selected_index = st.selectbox(
                "Repository record",
                filtered_df.index.tolist(),
                format_func=lambda idx: str(filtered_df.loc[idx, name_col])
                if name_col
                else f"Row {idx}",
            )
            selected_row = filtered_df.loc[selected_index]

            st.subheader("Metadata")
            st.json(selected_row.dropna().to_dict())

    st.subheader("Model Prediction Examples")
    if examples_df.empty:
        show_missing_data_notice(PREDICTION_EXAMPLES, "correct and incorrect prediction examples")
    else:
        st.dataframe(examples_df, use_container_width=True)
