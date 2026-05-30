import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")


def _figure(width: float = 9, height: float = 5):
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def plot_category_distribution(df: pd.DataFrame, label_col: str):
    fig, ax = _figure()
    counts = df[label_col].fillna("unknown").astype(str).value_counts()
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#2F6B5F")
    ax.set_xlabel("Repositories")
    ax.set_ylabel("Category")
    ax.set_title("Repository Category Distribution")
    fig.tight_layout()
    return fig


def plot_numeric_signal_by_category(df: pd.DataFrame, signal_col: str, label_col: str):
    fig, ax = _figure()
    plot_df = df[[signal_col, label_col]].dropna().copy()
    plot_df[label_col] = plot_df[label_col].astype(str)

    # Clip at p99 to prevent outliers from squashing the boxplot
    p99 = plot_df[signal_col].quantile(0.99)
    original_max = plot_df[signal_col].max()
    n_clipped = int((plot_df[signal_col] > p99).sum())
    plot_df[signal_col] = plot_df[signal_col].clip(upper=p99)

    sns.boxplot(data=plot_df, x=signal_col, y=label_col, ax=ax, color="#86B6A8")
    ax.set_xlabel(
        signal_col.replace("_", " ").title()
        + (f"  (clipped at p99={p99:.1f}, {n_clipped} outlier(s) hidden)"
           if n_clipped else "")
    )
    ax.set_ylabel("Category")
    ax.set_title(f"{signal_col.replace('_', ' ').title()} by Category")
    fig.tight_layout()
    return fig


def plot_repository_activity(df: pd.DataFrame, signal_col: str):
    fig, ax = _figure()
    values = pd.to_numeric(df[signal_col], errors="coerce").dropna()
    sns.histplot(values, bins=20, ax=ax, color="#3B6EA8")
    ax.set_xlabel(signal_col.replace("_", " ").title())
    ax.set_ylabel("Repositories")
    ax.set_title(f"Distribution of {signal_col.replace('_', ' ').title()}")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion_df: pd.DataFrame):
    fig, ax = _figure(7, 6)

    matrix = confusion_df.copy()
    if matrix.columns[0].lower() in {"label", "category", "actual", "true_label"}:
        matrix = matrix.set_index(matrix.columns[0])

    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0)
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig
