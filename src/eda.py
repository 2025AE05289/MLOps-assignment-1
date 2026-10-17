"""
Exploratory Data Analysis (EDA) for Heart Disease Dataset.

Generates professional visualizations:
1. Dataset overview & statistics
2. Target class distribution (bar chart)
3. Histograms of numerical features
4. Correlation heatmap
5. Feature distributions by target class (boxplots)
6. Categorical feature distributions by target (count plots)
7. Pairplot of key numerical features

All plots are saved to output/figures/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import preprocess_data, identify_feature_types

FIGURES_DIR = os.path.join(PROJECT_ROOT, "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "sans-serif"

COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "no_disease": "#10B981",
    "disease": "#EF4444",
    "palette": ["#10B981", "#EF4444"],
}


def save_fig(fig, name):
    """Save figure to output/figures/ directory."""
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_target_distribution(df):
    """Plot target class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    counts = df["target"].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease (1)"]
    bars = axes[0].bar(labels, counts.values, color=COLORS["palette"], edgecolor="white", linewidth=1.5)
    axes[0].set_title("Target Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=12)
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     str(count), ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=COLORS["palette"],
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11},
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Target Class Proportion", fontsize=14, fontweight="bold")
    
    fig.suptitle("Class Balance Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "01_target_distribution")


def plot_numerical_histograms(df, numerical_features):
    """Plot histograms for all numerical features."""
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, col in enumerate(numerical_features):
        ax = axes[i]
        # Histogram with KDE
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=ax,
                     palette=COLORS["palette"], alpha=0.6, edgecolor="white")
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(title="Target", labels=["Disease", "No Disease"])
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Numerical Feature Distributions by Target", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "02_numerical_histograms")


def plot_correlation_heatmap(df, numerical_features):
    """Plot correlation heatmap."""
    cols_for_corr = numerical_features + ["target"]
    corr_matrix = df[cols_for_corr].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                center=0, square=True, linewidths=1, linecolor="white",
                ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    save_fig(fig, "03_correlation_heatmap")


def plot_boxplots(df, numerical_features):
    """Plot boxplots of numerical features grouped by target."""
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, col in enumerate(numerical_features):
        ax = axes[i]
        sns.boxplot(data=df, x="target", y=col, palette=COLORS["palette"],
                    ax=ax, width=0.5, flierprops={"marker": "o", "markersize": 4})
        ax.set_xticklabels(["No Disease", "Disease"])
        ax.set_title(f"{col} by Target", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Numerical Features by Target Class (Boxplots)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "04_boxplots_by_target")


def plot_categorical_distributions(df, categorical_features):
    """Plot count plots for categorical features by target."""
    n_features = len(categorical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, col in enumerate(categorical_features):
        ax = axes[i]
        sns.countplot(data=df, x=col, hue="target", palette=COLORS["palette"],
                      ax=ax, edgecolor="white")
        ax.set_title(f"{col} Distribution", fontsize=12, fontweight="bold")
        ax.legend(title="Target", labels=["No Disease", "Disease"])
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Categorical Features by Target Class", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "05_categorical_distributions")


def plot_pairplot(df, numerical_features):
    """Plot pairplot of key numerical features."""
    # Select top features to keep the plot readable
    key_features = numerical_features[:4] + ["target"]
    
    g = sns.pairplot(df[key_features], hue="target", palette=COLORS["palette"],
                     diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20},
                     height=2.5)
    g.figure.suptitle("Pairplot of Key Numerical Features", fontsize=16,
                      fontweight="bold", y=1.02)
    save_fig(g.figure, "06_pairplot")


def print_dataset_overview(df):
    """Print comprehensive dataset overview."""
    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n{'Column':<15} {'Type':<10} {'Non-Null':<10} {'Unique':<8} {'Missing':<8}")
    print("-" * 55)
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        unique = df[col].nunique()
        missing = df[col].isnull().sum()
        print(f"{col:<15} {dtype:<10} {non_null:<10} {unique:<8} {missing:<8}")
    
    print(f"\n{'─' * 70}")
    print("DESCRIPTIVE STATISTICS")
    print(f"{'─' * 70}")
    print(df.describe().round(2).to_string())
    
    print(f"\n{'─' * 70}")
    print("TARGET DISTRIBUTION")
    print(f"{'─' * 70}")
    target_counts = df["target"].value_counts()
    for val, count in target_counts.items():
        pct = count / len(df) * 100
        label = "No Disease" if val == 0 else "Disease"
        print(f"  {label} ({val}): {count} ({pct:.1f}%)")


def run_eda():
    """Run the complete EDA pipeline."""
    print("\n" + "=" * 70)
    print("  EXPLORATORY DATA ANALYSIS - Heart Disease UCI Dataset")
    print("=" * 70)
    
    # Load and preprocess data
    df = preprocess_data()
    
    # Identify feature types
    categorical_features, numerical_features = identify_feature_types(df)
    print(f"\n[INFO] Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"[INFO] Numerical features ({len(numerical_features)}): {numerical_features}")
    
    # Print overview
    print_dataset_overview(df)
    
    # Generate all plots
    print(f"\n{'─' * 70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'─' * 70}")
    
    print("\n[1/6] Target class distribution...")
    plot_target_distribution(df)
    
    print("[2/6] Numerical feature histograms...")
    plot_numerical_histograms(df, numerical_features)
    
    print("[3/6] Correlation heatmap...")
    plot_correlation_heatmap(df, numerical_features)
    
    print("[4/6] Boxplots by target...")
    plot_boxplots(df, numerical_features)
    
    print("[5/6] Categorical distributions...")
    plot_categorical_distributions(df, categorical_features)
    
    print("[6/6] Pairplot of key features...")
    plot_pairplot(df, numerical_features)
    
    print(f"\n{'=' * 70}")
    print(f"  EDA COMPLETE — All figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_eda()
