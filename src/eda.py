"""Exploratory data analysis for the student performance dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def save_table(table: pd.DataFrame, name: str, directory: Path) -> None:
    """Save a table to disk."""
    directory.mkdir(exist_ok=True)
    table.to_csv(directory / name)


def save_figure(ax: plt.Axes, name: str, directory: Path) -> None:
    """Tighten layout, save and close a figure given an Axes."""
    directory.mkdir(exist_ok=True)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(directory / name)
    plt.close(fig)


# Load dataset
DATA_PATH = Path("student-mat.csv")
df = pd.read_csv(DATA_PATH)

# Output directories
fig_dir = Path("figures")
table_dir = Path("tables")

# Summary statistics and additional tables
summary = df.describe(include="all")
print(summary)
save_table(summary, "summary.csv", table_dir)

group_tables = {
    "grade_by_sex.csv": df.groupby("sex")["G3"].agg(["count", "mean"]),
    "grade_by_studytime.csv": df.groupby("studytime")["G3"].agg(["count", "mean"]),
}

numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()
group_tables["correlation_matrix.csv"] = corr

for name, table in group_tables.items():
    save_table(table, name, table_dir)


# Plots
ax = sns.histplot(df["G3"], bins=20, kde=True)
ax.set(xlabel="Final Grade (G3)", ylabel="Count")
save_figure(ax, "g3_distribution.png", fig_dir)

ax = sns.boxplot(data=df, x="sex", y="G3")
ax.set(xlabel="Sex", ylabel="Final Grade (G3)")
save_figure(ax, "g3_by_sex.png", fig_dir)

ax = sns.regplot(data=df, x="studytime", y="G3", scatter_kws={"s": 20})
ax.set(xlabel="Weekly Study Time", ylabel="Final Grade (G3)")
save_figure(ax, "studytime_vs_g3.png", fig_dir)

ax = sns.regplot(data=df, x="absences", y="G3", scatter_kws={"s": 20})
ax.set(xlabel="Number of Absences", ylabel="Final Grade (G3)")
save_figure(ax, "absences_vs_g3.png", fig_dir)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.5})
save_figure(ax, "correlation_heatmap.png", fig_dir)

g = sns.pairplot(
    df[["G1", "G2", "G3"]],
    kind="reg",
    diag_kind="kde",
    plot_kws={"scatter_kws": {"s": 20}},
)
g.fig.tight_layout()
g.fig.savefig(fig_dir / "grades_pairplot.png")
plt.close(g.fig)
