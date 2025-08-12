import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# High-resolution settings for publication-ready figures
sns.set_theme(context='paper', style='whitegrid', font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Load dataset
DATA_PATH = Path('student-mat.csv')
df = pd.read_csv(DATA_PATH)

# Ensure output directory exists
fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

# Summary statistics printed to console
summary = df.describe(include='all')
print(summary)

# Distribution of final grade
plt.figure(figsize=(6,4))
sns.histplot(df['G3'], bins=20, kde=True)
plt.xlabel('Final Grade (G3)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(fig_dir / 'g3_distribution.png')
plt.close()

# Boxplot of final grade by sex
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='sex', y='G3')
plt.xlabel('Sex')
plt.ylabel('Final Grade (G3)')
plt.tight_layout()
plt.savefig(fig_dir / 'g3_by_sex.png')
plt.close()

# Study time vs final grade
plt.figure(figsize=(6,4))
sns.regplot(data=df, x='studytime', y='G3', scatter_kws={'s':20})
plt.xlabel('Weekly Study Time')
plt.ylabel('Final Grade (G3)')
plt.tight_layout()
plt.savefig(fig_dir / 'studytime_vs_g3.png')
plt.close()

# Absences vs final grade
plt.figure(figsize=(6,4))
sns.regplot(data=df, x='absences', y='G3', scatter_kws={'s':20})
plt.xlabel('Number of Absences')
plt.ylabel('Final Grade (G3)')
plt.tight_layout()
plt.savefig(fig_dir / 'absences_vs_g3.png')
plt.close()

# Correlation heatmap of numeric features
numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(10,8))
corr = numeric_df.corr()
sns.heatmap(corr, cmap='vlag', center=0, square=True, cbar_kws={'shrink': .5})
plt.tight_layout()
plt.savefig(fig_dir / 'correlation_heatmap.png')
plt.close()

# Pairplot of grades
g = sns.pairplot(
    df[['G1', 'G2', 'G3']],
    kind='reg',
    diag_kind='kde',
    plot_kws={'scatter_kws': {'s': 20}}
)
g.fig.tight_layout()
g.fig.savefig(fig_dir / 'grades_pairplot.png')
plt.close(g.fig)
