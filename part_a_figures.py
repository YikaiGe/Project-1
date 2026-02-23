import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel(r"C:\Users\46473\OneDrive\Desktop\CSE5104\Project\1\concretecompressivestrength\Concrete_Data.xls")

short_names = {
    'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Blast Furnace Slag',
    'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly Ash',
    'Water  (component 4)(kg in a m^3 mixture)': 'Water',
    'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Superplasticizer',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'Coarse Aggregate',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'Fine Aggregate',
    'Age (day)': 'Age',
    'Concrete compressive strength(MPa, megapascals) ': 'Compressive Strength'
}
df = df.rename(columns=short_names)

# Table 1: Summary Statistics
print("=" * 60)
print("TABLE 1: Summary Statistics")
print("=" * 60)
stats = df.describe().round(2)
print(stats.loc[['mean', 'std', 'min', '50%', 'max']].to_string())

# Figure 1: Distribution Histograms
fig, axes = plt.subplots(9, 1, figsize=(14, 30))
axes = axes.flatten()
for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='bisque')
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Distribution of All Variables', fontsize=14, fontweight='bold', y=1.02)
plt.subplots_adjust(hspace=0.5)
plt.show()

# Table 2: Correlation with Compressive Strength
print("\n" + "=" * 60)
print("TABLE 2: Pearson Correlation with Compressive Strength")
print("=" * 60)
corr_target = df.corr()['Compressive Strength'].drop('Compressive Strength')
corr_target = corr_target.sort_values(ascending=False).round(4)
for name, val in corr_target.items():
    print(f"  {name:25s}: {val:+.4f}")

# Figure 2: Correlation Matrix Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 3: Scatter Plots (top 4 predictors vs target)
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
top_predictors = ['Cement', 'Superplasticizer', 'Age', 'Water']
for i, pred in enumerate(top_predictors):
    ax = axes[i]
    ax.scatter(df[pred], df['Compressive Strength'], alpha=0.4, s=15, color='bisque')
    ax.set_xlabel(pred)
    ax.set_ylabel('Compressive Strength (MPa)')
    ax.set_title(f'{pred} vs Compressive Strength')
plt.suptitle('Top Predictor Variables vs Compressive Strength',fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
