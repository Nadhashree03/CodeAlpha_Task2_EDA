"""
run_eda.py
Executes all EDA analysis and saves all visualizations.
This is the execution script — the notebook contains the same code
and is the primary deliverable.
"""

import os
import warnings
warnings.filterwarnings('ignore')

# ── Core imports ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr, shapiro

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
VIZ_DIR = os.path.join('..', 'visualizations')
DATA_PATH = os.path.join('..', 'data', 'titanic.csv')
os.makedirs(VIZ_DIR, exist_ok=True)


def save(fname):
    path = os.path.join(VIZ_DIR, fname)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close('all')
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 6 — LOAD DATASET')
print('='*65)
df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()
print(f'Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns')
print(f'Columns: {list(df.columns)}')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 7 — INITIAL DATA INSPECTION')
print('='*65)
print('\nFirst 5 rows:')
print(df.head().to_string())
print('\nLast 5 rows:')
print(df.tail().to_string())
print('\nRandom sample (n=8, seed=42):')
print(df.sample(8, random_state=42).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 8 — DATA STRUCTURE AND TYPES')
print('='*65)
print('\nColumn types and non-null counts:')
df.info()

print('\nUnique values per column:')
unique_counts = df.nunique().reset_index()
unique_counts.columns = ['Column', 'Unique Values']
unique_counts['% of Total'] = (unique_counts['Unique Values'] / len(df) * 100).round(1)
print(unique_counts.to_string(index=False))

print('\nCategorical column value counts:')
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in categorical_cols:
    print(f'\n--- {col} ---')
    print(df[col].value_counts(dropna=False).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 9 — DATA QUALITY ANALYSIS')
print('='*65)

print('\nRedundant / derived columns:')
print("  'class'       = categorical duplicate of 'pclass'")
print("  'alive'       = string duplicate of 'survived'")
print("  'embark_town' = full-text duplicate of 'embarked'")
print("  'who'         = derived from 'sex' + 'age'")
print("  'adult_male'  = derived from 'sex' + 'age'")

print('\nMissing values summary:')
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
print(missing_df.to_string())

print(f'\nTotal cells with missing data: {df.isnull().sum().sum():,}')
print(f'Overall missing rate: {df.isnull().sum().sum() / df.size * 100:.1f}%')

print('\nDuplicate rows:')
dupe_count = df.duplicated().sum()
print(f'Exact duplicates: {dupe_count}')

print('\nSuspicious values:')
age_anom = df[(df['age'] < 0) | (df['age'] > 100)]
print(f'Age outside [0,100]: {len(age_anom)}')
zero_fare = df[df['fare'] == 0]
print(f'Zero fare values: {len(zero_fare)}')
if len(zero_fare) > 0:
    print(zero_fare[['pclass', 'sex', 'age', 'fare', 'embarked', 'survived']].to_string())
print(f'Max SibSp: {df["sibsp"].max()}  |  Max Parch: {df["parch"].max()}')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 10 — MISSING VALUE VISUALIZATION')
print('='*65)

missing_cols = df.isnull().sum()
missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bar_colors = ['#e74c3c', '#e67e22', '#f1c40f'][:len(missing_cols)]
axes[0].bar(missing_cols.index, missing_cols.values / len(df) * 100,
            color=bar_colors, edgecolor='white', linewidth=0.8)
axes[0].set_title('Missing Value Percentage by Column', fontweight='bold')
axes[0].set_ylabel('Missing (%)')
axes[0].set_xlabel('Column')
for i, v in enumerate(missing_cols.values / len(df) * 100):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

missing_pattern = df[missing_cols.index].isnull()
sns.heatmap(missing_pattern.T, cbar=False, yticklabels=True,
            xticklabels=False, cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Missing Value Pattern Across Rows', fontweight='bold')
axes[1].set_xlabel('Passenger Records (891 rows)')
axes[1].set_ylabel('Column')

plt.suptitle('Figure 5 — Missing Value Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save('05_missing_values_heatmap.png')

# Age missingness analysis
df['age_missing'] = df['age'].isnull().astype(int)
print('\nIs age missingness systematic?')
print('Survival rate — age missing vs present:')
print(df.groupby('age_missing')['survived'].agg(['mean', 'count']).rename(
    columns={'mean': 'Survival Rate', 'count': 'Count'}).to_string())
print('\nPclass distribution — age missing vs present:')
print(df.groupby('age_missing')['pclass'].value_counts(normalize=True).unstack().round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 11 — DUPLICATE ANALYSIS')
print('='*65)
key_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
n_key_dupes = df.duplicated(subset=key_cols).sum()
print(f'Near-duplicate rows (same key fields): {n_key_dupes}')
if n_key_dupes > 0:
    dupes = df[df.duplicated(subset=key_cols, keep=False)].sort_values(key_cols)
    print('\nSample near-duplicates (may be family members):')
    print(dupes[key_cols + ['survived']].head(10).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 12 — DATA CLEANING')
print('='*65)

# Decision 1: Drop redundant columns
cols_to_drop = ['class', 'alive', 'embark_town', 'who', 'adult_male', 'age_missing']
df_clean = df.drop(columns=cols_to_drop)
print(f'DECISION 1: Dropped redundant columns: {cols_to_drop}')
print(f'  Shape after: {df_clean.shape}')

# Decision 2: Impute age
age_medians = df_clean.groupby(['pclass', 'sex'])['age'].transform('median')
df_clean['age'] = df_clean['age'].fillna(age_medians)
print(f'\nDECISION 2: Age imputed with pclass+sex group median.')
print('  Group medians used:')
print(df.groupby(['pclass', 'sex'])['age'].median().to_string())
remaining = df_clean['age'].isnull().sum()
print(f'  Remaining missing age values: {remaining}')

# Decision 3: Drop 2 rows with missing embarked
n_before = len(df_clean)
df_clean = df_clean.dropna(subset=['embarked'])
print(f'\nDECISION 3: Dropped {n_before - len(df_clean)} rows with missing Embarked.')
print(f'  Shape after: {df_clean.shape}')

# Decision 4: Drop deck
df_clean = df_clean.drop(columns=['deck'])
print(f'\nDECISION 4: Dropped "deck" column (77.2% missing).')
print(f'  Shape after: {df_clean.shape}')

# Decision 5: Flag zero fares
zero_fare_count = (df_clean['fare'] == 0).sum()
print(f'\nDECISION 5: Retained {zero_fare_count} zero-fare passengers (flagged as anomalous).')

# Decision 6: Feature engineering
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)
print(f'\nDECISION 6: Engineered family_size and is_alone.')
print(f'  Family size range: {df_clean["family_size"].min()} – {df_clean["family_size"].max()}')

print(f'\nFinal clean dataset: {df_clean.shape}')
print(f'Columns: {list(df_clean.columns)}')
remaining_missing = df_clean.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
if len(remaining_missing) == 0:
    print('No remaining missing values.')
else:
    print('Remaining missing values:')
    print(remaining_missing.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 13 — DESCRIPTIVE STATISTICS')
print('='*65)

num_cols = ['age', 'fare', 'sibsp', 'parch', 'family_size']
desc_num = df_clean[num_cols].describe().T
desc_num['skewness'] = df_clean[num_cols].skew().round(2)
desc_num['kurtosis'] = df_clean[num_cols].kurt().round(2)
print('\nNumerical descriptive statistics:')
print(desc_num.round(2).to_string())

print('\nCategorical distributions:')
cat_summary_cols = ['survived', 'pclass', 'sex', 'embarked']
for col in cat_summary_cols:
    vc = df_clean[col].value_counts()
    pct = df_clean[col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({'Count': vc, 'Percentage': pct.round(1)})
    print(f'\n--- {col.upper()} ---')
    print(summary.to_string())

survival_rate = df_clean['survived'].mean() * 100
survived_n = df_clean['survived'].sum()
died_n = len(df_clean) - survived_n
print(f'\nOverall survival: {survived_n} survived ({survival_rate:.1f}%), {died_n} died ({100-survival_rate:.1f}%)')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 14 — UNIVARIATE ANALYSIS')
print('='*65)

# Figure 1: Overall Survival
fig, ax = plt.subplots(figsize=(6, 5))
counts = df_clean['survived'].value_counts().sort_index()
bars = ax.bar(['Did Not Survive', 'Survived'], counts.values,
               color=['#e74c3c', '#2ecc71'], edgecolor='white', linewidth=1.2, width=0.5)
for bar, val in zip(bars, counts.values):
    pct = val / len(df_clean) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
ax.set_title('Figure 1 — Overall Survival Distribution', fontweight='bold', pad=12)
ax.set_ylabel('Number of Passengers')
ax.set_ylim(0, max(counts.values) * 1.2)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
save('01_survival_rate.png')

# Figure 2: Passenger Class
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
class_counts = df_clean['pclass'].value_counts().sort_index()
class_labels = ['1st Class', '2nd Class', '3rd Class']
colors = ['#3498db', '#9b59b6', '#e67e22']
bars = axes[0].bar(class_labels, class_counts.values, color=colors,
                    edgecolor='white', linewidth=1.2, width=0.5)
for bar, val in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val}\n({val/len(df_clean)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')
axes[0].set_title('Passenger Class Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.4)
axes[1].pie(class_counts.values, labels=class_labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Passenger Class Share', fontweight='bold')
plt.suptitle('Figure 2 — Passenger Class Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
save('02_passenger_class.png')

# Figure 3: Gender
fig, ax = plt.subplots(figsize=(6, 5))
sex_counts = df_clean['sex'].value_counts()
ax.bar(['Female', 'Male'], sex_counts.values,
       color=['#e91e8c', '#3498db'], edgecolor='white', linewidth=1.2, width=0.4)
for i, (idx, val) in enumerate(sex_counts.items()):
    ax.text(i, val + 3, f'{val}\n({val/len(df_clean)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
ax.set_title('Figure 3 — Gender Distribution', fontweight='bold', pad=12)
ax.set_ylabel('Number of Passengers')
ax.set_ylim(0, sex_counts.max() * 1.2)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
save('03_gender_distribution.png')

# Figure 4: Age Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_clean['age'], bins=30, color='#3498db', edgecolor='white', linewidth=0.8, alpha=0.85)
axes[0].axvline(df_clean['age'].median(), color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Median: {df_clean["age"].median():.1f}')
axes[0].axvline(df_clean['age'].mean(), color='#2ecc71', linestyle='-.', linewidth=2,
                label=f'Mean: {df_clean["age"].mean():.1f}')
axes[0].set_title('Age Distribution (Histogram)', fontweight='bold')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.4)
for survived, label, color in [(0, 'Did Not Survive', '#e74c3c'), (1, 'Survived', '#2ecc71')]:
    subset = df_clean[df_clean['survived'] == survived]['age']
    axes[1].hist(subset, bins=25, alpha=0.55, color=color,
                 edgecolor='white', linewidth=0.5, density=True, label=label)
    subset.plot.kde(ax=axes[1], color=color, linewidth=2.5)
axes[1].set_title('Age Distribution by Survival Status', fontweight='bold')
axes[1].set_xlabel('Age (years)')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.4)
plt.suptitle('Figure 4 — Age Distribution Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
save('04_age_distribution.png')

# Figure 6: Fare Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_clean['fare'], bins=50, color='#9b59b6', edgecolor='white', linewidth=0.8, alpha=0.85)
axes[0].axvline(df_clean['fare'].median(), color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Median: £{df_clean["fare"].median():.2f}')
axes[0].axvline(df_clean['fare'].mean(), color='#3498db', linestyle='-.', linewidth=2,
                label=f'Mean: £{df_clean["fare"].mean():.2f}')
axes[0].set_title('Fare Distribution (Raw)', fontweight='bold')
axes[0].set_xlabel('Fare (£)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.4)
log_fare = np.log1p(df_clean['fare'])
axes[1].hist(log_fare, bins=40, color='#e67e22', edgecolor='white', linewidth=0.8, alpha=0.85)
axes[1].axvline(log_fare.median(), color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Median: {log_fare.median():.2f}')
axes[1].set_title('Fare Distribution (Log-Transformed)', fontweight='bold')
axes[1].set_xlabel('log(1 + Fare)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.4)
axes[1].annotate('Log transform reveals bi-modal\nstructure masked by extreme outliers',
                 xy=(0.98, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
plt.suptitle('Figure 6 — Fare Distribution Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
save('06_fare_distribution.png')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 15 — BIVARIATE / MULTIVARIATE ANALYSIS')
print('='*65)

# Survival by class
survival_class = df_clean.groupby(['pclass', 'survived']).size().unstack()
survival_class.index = ['1st Class', '2nd Class', '3rd Class']
survival_class.columns = ['Did Not Survive', 'Survived']
survival_rate_class = df_clean.groupby('pclass')['survived'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
survival_class.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'],
                    edgecolor='white', linewidth=0.8, rot=0)
axes[0].set_title('Survival Count by Passenger Class', fontweight='bold')
axes[0].set_ylabel('Number of Passengers')
axes[0].legend(title='Outcome')
axes[0].grid(axis='y', alpha=0.4)
colors_class = ['#3498db', '#9b59b6', '#e67e22']
bars = axes[1].bar(['1st Class', '2nd Class', '3rd Class'],
                    survival_rate_class.values, color=colors_class,
                    edgecolor='white', linewidth=1.2, width=0.5)
for bar, val in zip(bars, survival_rate_class.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[1].axhline(df_clean['survived'].mean() * 100, color='black', linestyle='--',
                linewidth=1.5, label=f'Overall avg: {df_clean["survived"].mean()*100:.1f}%')
axes[1].set_title('Survival Rate by Passenger Class', fontweight='bold')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_ylim(0, 85)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.4)
plt.suptitle('Figure — Survival by Passenger Class (Q1)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('02_survival_by_class.png')

print('Survival rates by class:')
for cls, rate in zip(['1st', '2nd', '3rd'], survival_rate_class.values):
    print(f'  {cls} Class: {rate:.1f}%')

# Survival by gender
survival_sex = df_clean.groupby(['sex', 'survived']).size().unstack()
survival_sex.index = ['Female', 'Male']
survival_sex.columns = ['Did Not Survive', 'Survived']
survival_rate_sex = df_clean.groupby('sex')['survived'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
survival_sex.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'],
                  edgecolor='white', linewidth=0.8, rot=0)
axes[0].set_title('Survival Count by Gender', fontweight='bold')
axes[0].set_ylabel('Number of Passengers')
axes[0].legend(title='Outcome')
axes[0].grid(axis='y', alpha=0.4)
axes[1].bar(['Female', 'Male'], survival_rate_sex.values,
             color=['#e91e8c', '#3498db'], edgecolor='white', linewidth=1.2, width=0.4)
for i, (_, val) in enumerate(zip(['Female', 'Male'], survival_rate_sex.values)):
    axes[1].text(i, val + 0.8, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)
axes[1].axhline(df_clean['survived'].mean() * 100, color='black', linestyle='--',
                linewidth=1.5, label=f'Overall avg: {df_clean["survived"].mean()*100:.1f}%')
axes[1].set_title('Survival Rate by Gender', fontweight='bold')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_ylim(0, 100)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.4)
plt.suptitle('Figure — Survival by Gender (Q2)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('03_survival_by_gender.png')

print('\nSurvival rates by gender:')
for sex_label, rate in zip(['Female', 'Male'], survival_rate_sex.values):
    print(f'  {sex_label}: {rate:.1f}%')

# Embarkation analysis
port_labels = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
df_clean['port'] = df_clean['embarked'].map(port_labels)
port_counts = df_clean['port'].value_counts()
surv_port = df_clean.groupby('port')['survived'].mean() * 100
surv_port = surv_port.reindex(port_counts.index)
colors_port = ['#3498db', '#2ecc71', '#e67e22']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].bar(port_counts.index, port_counts.values, color=colors_port,
             edgecolor='white', linewidth=1.2, width=0.5)
for i, val in enumerate(port_counts.values):
    axes[0].text(i, val + 2, f'{val}', ha='center', va='bottom', fontweight='bold')
axes[0].set_title('Passenger Count by Port', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.4)
axes[1].bar(surv_port.index, surv_port.values, color=colors_port,
             edgecolor='white', linewidth=1.2, width=0.5)
for i, val in enumerate(surv_port.values):
    axes[1].text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[1].axhline(df_clean['survived'].mean() * 100, color='black', linestyle='--', linewidth=1.5)
axes[1].set_title('Survival Rate by Port', fontweight='bold')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_ylim(0, 80)
axes[1].grid(axis='y', alpha=0.4)
port_class = df_clean.groupby(['port', 'pclass']).size().unstack().reindex(port_counts.index)
port_class.columns = ['1st Class', '2nd Class', '3rd Class']
port_class_pct = port_class.div(port_class.sum(axis=1), axis=0) * 100
port_class_pct.plot(kind='bar', ax=axes[2], color=['#3498db', '#9b59b6', '#e67e22'],
                     stacked=True, edgecolor='white', linewidth=0.8, rot=0)
axes[2].set_title('Class Mix by Port (Stacked %)', fontweight='bold')
axes[2].set_ylabel('Percentage')
axes[2].legend(title='Class', bbox_to_anchor=(1, 1))
axes[2].grid(axis='y', alpha=0.4)
plt.suptitle('Figure 8 — Embarkation Port Analysis (Q7)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('08_embarkation_analysis.png')

# Family size vs survival
fam_surv = df_clean.groupby('family_size')['survived'].agg(['mean', 'count'])
fam_surv.columns = ['Survival Rate', 'Count']
fam_surv['Survival Rate'] = fam_surv['Survival Rate'] * 100
fam_surv_filtered = fam_surv[fam_surv['Count'] >= 5]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(fam_surv_filtered.index.astype(str), fam_surv_filtered['Survival Rate'].values,
             color='#3498db', edgecolor='white', linewidth=0.8)
for i, (idx, row) in enumerate(fam_surv_filtered.iterrows()):
    axes[0].text(i, row['Survival Rate'] + 0.8,
                 f"{row['Survival Rate']:.0f}%\n(n={int(row['Count'])})",
                 ha='center', va='bottom', fontsize=8.5)
axes[0].axhline(df_clean['survived'].mean() * 100, color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'Overall: {df_clean["survived"].mean()*100:.1f}%')
axes[0].set_title('Survival Rate by Family Size', fontweight='bold')
axes[0].set_xlabel('Family Size (self + relatives)')
axes[0].set_ylabel('Survival Rate (%)')
axes[0].set_ylim(0, 100)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.4)
alone_surv = df_clean.groupby('is_alone')['survived'].mean() * 100
axes[1].bar(['With Family', 'Alone'], alone_surv.values,
             color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.2, width=0.4)
for i, val in enumerate(alone_surv.values):
    axes[1].text(i, val + 0.8, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)
axes[1].axhline(df_clean['survived'].mean() * 100, color='black', linestyle='--', linewidth=1.5)
axes[1].set_title('Survival: Alone vs With Family', fontweight='bold')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_ylim(0, 80)
axes[1].grid(axis='y', alpha=0.4)
plt.suptitle('Figure 9 — Family Size and Survival (Q8)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('09_family_size_survival.png')

print('\nSurvival by family size:')
print(fam_surv_filtered.to_string())

# Gender × Class heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pivot = df_clean.pivot_table(values='survived', index='sex', columns='pclass', aggfunc='mean') * 100
pivot.index = ['Female', 'Male']
pivot.columns = ['1st Class', '2nd Class', '3rd Class']
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
            linewidths=0.5, linecolor='white', annot_kws={'size': 13, 'weight': 'bold'},
            cbar_kws={'label': 'Survival Rate (%)'}, ax=axes[0])
axes[0].set_title('Survival Rate (%) by Gender × Class', fontweight='bold')
axes[0].set_ylabel('Gender')
pivot_count = df_clean.pivot_table(values='survived', index='sex', columns='pclass', aggfunc='count')
pivot_count.index = ['Female', 'Male']
pivot_count.columns = ['1st Class', '2nd Class', '3rd Class']
sns.heatmap(pivot_count, annot=True, fmt='d', cmap='Blues',
            linewidths=0.5, linecolor='white', annot_kws={'size': 13, 'weight': 'bold'},
            cbar_kws={'label': 'Passenger Count'}, ax=axes[1])
axes[1].set_title('Passenger Count by Gender × Class', fontweight='bold')
axes[1].set_ylabel('Gender')
plt.suptitle('Figure 11 — Gender × Class Interaction (Q10)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('11_gender_class_survival.png')

print('\nSurvival rate (%) — Gender × Class:')
print(pivot.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 16 — TREND ANALYSIS')
print('='*65)

age_bins = [0, 12, 18, 30, 45, 60, 80]
age_labels = ['0-12\n(Child)', '13-18\n(Teen)', '19-30\n(Young Adult)',
              '31-45\n(Adult)', '46-60\n(Mature)', '61+\n(Senior)']
df_clean['age_group'] = pd.cut(df_clean['age'], bins=age_bins, labels=age_labels, right=True)
age_surv = df_clean.groupby('age_group', observed=True)['survived'].agg(['mean', 'count'])
age_surv.columns = ['Survival Rate', 'Count']
age_surv['Survival Rate'] = age_surv['Survival Rate'] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(age_surv.index, age_surv['Survival Rate'],
                    color='#3498db', edgecolor='white', linewidth=0.8, alpha=0.85)
for bar, (idx, row) in zip(bars, age_surv.iterrows()):
    axes[0].text(bar.get_x() + bar.get_width()/2, row['Survival Rate'] + 0.8,
                 f"{row['Survival Rate']:.0f}%\nn={int(row['Count'])}",
                 ha='center', va='bottom', fontsize=8.5)
axes[0].axhline(df_clean['survived'].mean() * 100, color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'Overall: {df_clean["survived"].mean()*100:.1f}%')
axes[0].set_title('Survival Rate by Age Group', fontweight='bold')
axes[0].set_ylabel('Survival Rate (%)')
axes[0].set_ylim(0, 80)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.4)
sns.violinplot(data=df_clean, x='survived', y='age', hue='survived',
               palette={0: '#e74c3c', 1: '#2ecc71'}, inner='box',
               cut=0, ax=axes[1], legend=False)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Did Not Survive', 'Survived'])
axes[1].set_title('Age Distribution by Survival (Violin)', fontweight='bold')
axes[1].set_ylabel('Age (years)')
axes[1].grid(axis='y', alpha=0.4)
plt.suptitle('Figure — Age–Survival Trend Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
save('04_age_survival_trend.png')

print('Survival rate by age group:')
print(age_surv.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 17 — OUTLIER / ANOMALY ANALYSIS')
print('='*65)

Q1 = df_clean['fare'].quantile(0.25)
Q3 = df_clean['fare'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
outliers_fare = df_clean[df_clean['fare'] > upper_fence]

df_clean['class_label'] = df_clean['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df_clean, x='class_label', y='fare',
            palette=['#3498db', '#9b59b6', '#e67e22'],
            flierprops={'marker': 'o', 'markerfacecolor': '#e74c3c',
                        'markersize': 5, 'alpha': 0.6}, ax=axes[0])
axes[0].set_title('Fare Distribution by Class (with Outliers)', fontweight='bold')
axes[0].set_xlabel('Passenger Class')
axes[0].set_ylabel('Fare (£)')
axes[0].grid(axis='y', alpha=0.4)
axes[0].annotate('Red dots = outliers\n(legitimate high-fare\n1st class passengers)',
                 xy=(0.02, 0.97), xycoords='axes fraction', va='top',
                 fontsize=9, bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
normal = df_clean[df_clean['fare'] <= upper_fence]
axes[1].scatter(range(len(normal)), normal['fare'].values,
                alpha=0.4, s=12, color='#3498db', label='Normal')
axes[1].scatter(range(len(outliers_fare)), outliers_fare['fare'].values,
                alpha=0.7, s=30, color='#e74c3c', label=f'Outliers (n={len(outliers_fare)})')
axes[1].axhline(upper_fence, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=f'Upper fence: £{upper_fence:.1f}')
axes[1].set_title('Fare Outlier Detection (IQR Method)', fontweight='bold')
axes[1].set_xlabel('Passenger Index')
axes[1].set_ylabel('Fare (£)')
axes[1].legend()
axes[1].grid(alpha=0.4)
plt.suptitle('Figure 7 — Fare Outlier Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
save('07_fare_boxplot_outliers.png')

print(f'Fare outlier summary:')
print(f'  Q1: £{Q1:.2f}  |  Q3: £{Q3:.2f}  |  IQR: £{IQR:.2f}')
print(f'  Upper fence: £{upper_fence:.2f}')
print(f'  Outlier count: {len(outliers_fare)} ({len(outliers_fare)/len(df_clean)*100:.1f}%)')
print(f'  Max fare: £{df_clean["fare"].max():.2f}')
print('Top 5 highest fares:')
print(df_clean.nlargest(5, 'fare')[['pclass', 'sex', 'age', 'fare', 'embarked', 'survived']].to_string())
print('Interpretation: Outliers are legitimate 1st class passengers — NOT data errors. Retained.')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 18 — CORRELATION ANALYSIS')
print('='*65)

num_cols_corr = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'is_alone']
corr_matrix = df_clean[num_cols_corr].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, mask=mask, linewidths=0.5, linecolor='white',
            annot_kws={'size': 10}, cbar_kws={'label': 'Pearson r', 'shrink': 0.8}, ax=ax)
ax.set_title('Figure 10 — Correlation Matrix (Numerical Variables)', fontweight='bold', pad=15)
plt.tight_layout()
save('10_correlation_heatmap.png')

print('Key correlations with survival:')
surv_corr = corr_matrix['survived'].drop('survived').sort_values(key=abs, ascending=False)
for col, val in surv_corr.items():
    direction = 'positive' if val > 0 else 'negative'
    print(f'  {col:15s}: r = {val:+.3f}  ({direction})')

# Pairplot
g = sns.pairplot(
    df_clean[['survived', 'age', 'fare', 'pclass', 'family_size']].assign(
        survived=df_clean['survived'].map({0: 'Died', 1: 'Survived'})
    ),
    hue='survived',
    palette={'Died': '#e74c3c', 'Survived': '#2ecc71'},
    plot_kws={'alpha': 0.4, 's': 20},
    diag_kind='kde', corner=True
)
g.figure.suptitle('Pairplot — Key Numerical Variables by Survival', y=1.02, fontweight='bold')
g.figure.savefig(os.path.join(VIZ_DIR, '10b_pairplot.png'), bbox_inches='tight', dpi=120)
plt.close('all')
print('  Saved: 10b_pairplot.png')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' SECTION 19 — HYPOTHESIS TESTING')
print('='*65)

# Test 1: Gender and Survival (Chi-Square)
print('\n--- TEST 1: Gender and Survival (Chi-Square) ---')
print('H₀: No significant association between gender and survival.')
contingency_sex = pd.crosstab(df_clean['sex'], df_clean['survived'])
chi2, p_val, dof, expected = chi2_contingency(contingency_sex)
print(f'Contingency Table:\n{contingency_sex.rename(columns={0:"Died",1:"Survived"}).to_string()}')
print(f'Chi2={chi2:.4f}, p={p_val:.2e}, df={dof}')
if p_val < 0.05:
    female_surv = df_clean[df_clean['sex']=='female']['survived'].mean()*100
    male_surv = df_clean[df_clean['sex']=='male']['survived'].mean()*100
    print(f'RESULT: REJECT H₀ — gender is significantly associated with survival (p={p_val:.2e})')
    print(f'  Female survival: {female_surv:.1f}% | Male survival: {male_surv:.1f}%')

# Test 2: Class and Survival (Chi-Square)
print('\n--- TEST 2: Class and Survival (Chi-Square) ---')
print('H₀: No significant association between passenger class and survival.')
contingency_class = pd.crosstab(df_clean['pclass'], df_clean['survived'])
chi2_c, p_c, dof_c, _ = chi2_contingency(contingency_class)
print(f'Chi2={chi2_c:.4f}, p={p_c:.2e}, df={dof_c}')
if p_c < 0.05:
    print(f'RESULT: REJECT H₀ — class is significantly associated with survival (p={p_c:.2e})')

# Test 3: Fare vs Survival (Mann-Whitney U)
print('\n--- TEST 3: Fare vs Survival (Mann-Whitney U) ---')
print('H₀: Fare distributions of survivors and non-survivors are identical.')
fares_survived = df_clean[df_clean['survived'] == 1]['fare']
fares_died = df_clean[df_clean['survived'] == 0]['fare']
u_stat, p_mw = mannwhitneyu(fares_survived, fares_died, alternative='greater')
print(f'Median fare — Survived: £{fares_survived.median():.2f} | Died: £{fares_died.median():.2f}')
print(f'U={u_stat:.0f}, p={p_mw:.2e}')
if p_mw < 0.05:
    print(f'RESULT: REJECT H₀ — survivors paid significantly higher fares (p={p_mw:.2e})')

# Test 4: Point-Biserial Correlation
print('\n--- TEST 4: Fare-Survival Correlation (Point-Biserial) ---')
print('H₀: No linear correlation between fare and survival probability.')
corr_pb, p_pb = pointbiserialr(df_clean['survived'], df_clean['fare'])
print(f'Point-biserial r={corr_pb:.4f}, p={p_pb:.2e}')
if p_pb < 0.05:
    strength = 'weak' if abs(corr_pb) < 0.3 else 'moderate'
    print(f'RESULT: REJECT H₀ — {strength} positive correlation (r={corr_pb:.3f}, p={p_pb:.2e})')

# Test 5: Age Normality (Shapiro-Wilk)
print('\n--- TEST 5: Age Normality (Shapiro-Wilk, n=200 sample) ---')
print('H₀: Age is normally distributed.')
age_sample = df_clean['age'].dropna().sample(200, random_state=42)
stat_sw, p_sw = shapiro(age_sample)
print(f'W={stat_sw:.4f}, p={p_sw:.4f}')
if p_sw < 0.05:
    print(f'RESULT: REJECT H₀ — age is NOT normally distributed (p={p_sw:.4f})')
else:
    print(f'RESULT: FAIL TO REJECT H₀ — age approximately normal in sample (p={p_sw:.4f})')

# Figure 12: Hypothesis test visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df_clean, x='survived', y='fare', hue='survived',
            palette={0: '#e74c3c', 1: '#2ecc71'}, legend=False,
            flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5}, ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Did Not Survive', 'Survived'])
axes[0].set_title('Fare by Survival (Test 3 — Mann-Whitney U)', fontweight='bold')
axes[0].set_ylabel('Fare (£)')
axes[0].annotate(f'p < 0.001\nReject H₀', xy=(0.97, 0.97), xycoords='axes fraction',
                  ha='right', va='top', fontsize=10, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.9))
axes[0].grid(axis='y', alpha=0.4)
pivot_sexclass = df_clean.pivot_table(values='survived', index='sex', columns='pclass', aggfunc='mean') * 100
pivot_sexclass.index = ['Female', 'Male']
pivot_sexclass.columns = ['1st', '2nd', '3rd']
pivot_sexclass.T.plot(kind='bar', ax=axes[1], color=['#e91e8c', '#3498db'],
                       edgecolor='white', linewidth=0.8, rot=0)
axes[1].axhline(df_clean['survived'].mean() * 100, color='black', linestyle='--',
                linewidth=1.5, label='Overall avg')
axes[1].set_title('Survival % — Gender × Class (Tests 1 & 2)', fontweight='bold')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_xlabel('Passenger Class')
axes[1].legend(title='Gender')
axes[1].grid(axis='y', alpha=0.4)
plt.suptitle('Figure 12 — Hypothesis Test Visualizations', fontsize=13, fontweight='bold')
plt.tight_layout()
save('12_hypothesis_test_summary.png')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print(' ALL VISUALIZATIONS GENERATED SUCCESSFULLY')
print('='*65)

saved_files = sorted(os.listdir(VIZ_DIR))
print(f'\nFiles in visualizations/:')
for f in saved_files:
    fpath = os.path.join(VIZ_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f'  {f}  ({size_kb:.1f} KB)')

print(f'\nTotal: {len(saved_files)} visualization files')
print('\nEDA execution complete.')
