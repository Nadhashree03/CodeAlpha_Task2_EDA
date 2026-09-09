# 🚢 Titanic Passenger Dataset — Exploratory Data Analysis

> **CodeAlpha Data Analytics Internship — Task 2**

---

## Project Overview

A complete, structured Exploratory Data Analysis (EDA) of the Titanic passenger dataset, investigating which demographic and socioeconomic factors influenced survival during the April 1912 disaster.

This project addresses all five CodeAlpha Task 2 requirements:
- Meaningful analytical questions defined upfront
- Data structure, variables, and types thoroughly explored
- Trends, patterns, and anomalies identified and visualized
- Hypotheses statistically tested and validated
- Data quality issues detected, documented, and handled with explicit decisions

---

## Project Structure

```
CODEALPHA_TASK2_EDA/
│
├── data/
│   └── titanic.csv                  # Raw dataset (891 passengers × 15 variables)
│
├── notebook/
│   ├── titanic_eda.ipynb            # Main Jupyter Notebook (23 sections, 64 cells)
│   ├── run_eda.py                   # Standalone execution script
│   └── build_notebook.py           # Notebook generation script
│
├── visualizations/                  # 16 saved chart files (PNG, 150 DPI)
│   ├── 01_survival_rate.png
│   ├── 02_passenger_class.png
│   ├── 02_survival_by_class.png
│   ├── 03_gender_distribution.png
│   ├── 03_survival_by_gender.png
│   ├── 04_age_distribution.png
│   ├── 04_age_survival_trend.png
│   ├── 05_missing_values_heatmap.png
│   ├── 06_fare_distribution.png
│   ├── 07_fare_boxplot_outliers.png
│   ├── 08_embarkation_analysis.png
│   ├── 09_family_size_survival.png
│   ├── 10_correlation_heatmap.png
│   ├── 10b_pairplot.png
│   ├── 11_gender_class_survival.png
│   └── 12_hypothesis_test_summary.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

| Attribute | Value |
|---|---|
| Name | Titanic Passenger Survival Dataset |
| Source | Seaborn built-in / Kaggle / Vanderbilt Biostatistics |
| Rows | 891 passengers |
| Columns | 15 variables |
| License | Public domain |

**Key variables:** `survived`, `pclass`, `sex`, `age`, `fare`, `embarked`, `sibsp`, `parch`

---

## Analytical Questions

This EDA was structured around 10 pre-defined questions:

| # | Question | Analysis Type |
|---|---|---|
| Q1 | What is the overall survival rate by passenger class? | Univariate + Bivariate |
| Q2 | Did gender significantly influence survival? | Hypothesis Testing |
| Q3 | How is age distributed and how does it relate to survival? | Univariate + Trend |
| Q4 | What are the data quality issues in this dataset? | Data Quality |
| Q5 | How is fare distributed? Are there extreme outliers? | Outlier Analysis |
| Q6 | Is there a statistically significant relationship between fare and survival? | Hypothesis Testing |
| Q7 | How do embarkation port, class, and fare interact? | Multivariate |
| Q8 | Did traveling with family improve survival odds? | Bivariate |
| Q9 | What does the correlation structure reveal? | Correlation Analysis |
| Q10 | Does gender × class interaction reveal hidden survival patterns? | Multivariate |

---

## Key Findings

All findings derived from actual data — no fabricated results.

### Survival Overview
- **38.2%** of passengers survived (340 of 889 after cleaning)
- Overall male-to-female ratio: 64.9% male, 35.1% female

### Class Effect (Q1)
| Class | Survival Rate |
|---|---|
| 1st Class | **62.6%** |
| 2nd Class | **47.3%** |
| 3rd Class | **24.2%** |

Chi-square test: χ²=100.98, **p=1.18×10⁻²²** — highly significant

### Gender Effect (Q2)
| Gender | Survival Rate |
|---|---|
| Female | **74.0%** |
| Male | **18.9%** |

Chi-square test: χ²=258.43, **p=3.78×10⁻⁵⁸** — strongest predictor in dataset

### Gender × Class Intersection (Q10)
| | 1st Class | 2nd Class | 3rd Class |
|---|---|---|---|
| Female | **96.7%** | **92.1%** | **50.0%** |
| Male | **36.9%** | **15.7%** | **13.5%** |

### Fare vs Survival (Q5, Q6)
- Median fare, survivors: **£26.00** vs. non-survivors: **£10.50**
- Mann-Whitney U: **p=5.96×10⁻²²** — survivors paid significantly higher fares
- Point-biserial r=0.255 — moderate positive correlation
- Max fare in dataset: **£512.33** (first-class cabin group — retained, not errors)

### Family Size (Q8)
- Solo travelers: **30.1%** survival
- Family of 2–4: **55–72%** survival
- Family of 7+: **0–13%** survival (large lower-class families)

### Embarkation (Q7)
- Cherbourg: **55%** survival (disproportionately 1st class)
- Southampton: **34%** survival (majority of passengers, mixed class)
- Queenstown: **39%** survival (mostly 3rd class)

---

## Hypothesis Tests

| Test | Method | Statistic | P-Value | Result |
|---|---|---|---|---|
| Gender → Survival | Chi-Square | χ²=258.43 | 3.78×10⁻⁵⁸ | Reject H₀ ✅ |
| Class → Survival | Chi-Square | χ²=100.98 | 1.18×10⁻²² | Reject H₀ ✅ |
| Fare → Survival | Mann-Whitney U | U=128,888 | 5.96×10⁻²² | Reject H₀ ✅ |
| Fare–Survival Correlation | Point-Biserial | r=0.255 | 1.08×10⁻¹⁴ | Reject H₀ ✅ |
| Age Normality | Shapiro-Wilk | W=0.9573 | <0.0001 | Reject H₀ ✅ |

All tests at α = 0.05 significance level.

---

## Data Cleaning Decisions

| Decision | Action | Justification |
|---|---|---|
| D1 | Dropped 5 redundant columns (class, alive, embark_town, who, adult_male) | Derived from retained columns — carry no independent information |
| D2 | Imputed age with group median (pclass × sex) | 20% missing; group median is more accurate than global mean; dropping 177 rows loses too much data |
| D3 | Dropped 2 rows with missing Embarked | Only 2 rows — negligible impact; imputation would be unreliable |
| D4 | Dropped Deck column | 77.2% missing — analytically unrecoverable |
| D5 | Retained 15 zero-fare passengers | Flagged as anomalous but valid; likely staff/officer records |
| D6 | Engineered family_size = sibsp + parch + 1 | More interpretable than separate components |

---

## Tool Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12 | Runtime |
| Pandas | 2.2.2 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| Matplotlib | 3.8.2 | Base visualization |
| Seaborn | 0.13.2 | Statistical visualization |
| SciPy | 1.11.4 | Hypothesis testing |
| Jupyter Notebook | 7.x | Interactive analysis |

---

## Setup and Usage

### Clone and install dependencies

```bash
git clone <your-repo-url>
cd CODEALPHA_TASK2_EDA
pip install -r requirements.txt
```

### Open the notebook

```bash
cd notebook
jupyter notebook titanic_eda.ipynb
```

### Or run the analysis script directly

```bash
cd notebook
python run_eda.py
```

This re-executes all analysis and regenerates all charts in `visualizations/`.

---

## Notebook Structure

The notebook contains 23 clearly labeled sections:

1. Project Overview
2. Objective
3. Dataset Information
4. Analytical Questions
5. Import Libraries
6. Load Dataset
7. Initial Data Inspection
8. Data Structure and Data Types
9. Data Quality Analysis
10. Missing Value Analysis
11. Duplicate Analysis
12. Data Cleaning
13. Descriptive Statistics
14. Univariate Analysis
15. Bivariate / Multivariate Analysis
16. Trend Analysis
17. Outlier / Anomaly Analysis
18. Correlation Analysis
19. Hypothesis Testing
20. Key Findings
21. Business / Practical Insights
22. Limitations
23. Conclusion

---

## Limitations

- Age imputation (20% missing) introduces estimation uncertainty in age-based analysis
- Deck column (77% missing) prevents cabin-location analysis
- Dataset covers passengers only — crew data is excluded
- Findings are specific to this disaster and cannot be directly generalized
- All associations are correlational — causal mechanisms require domain knowledge beyond the data

---

## License

Dataset: Public domain (Vanderbilt University Biostatistics / Kaggle).  
Analysis code: MIT License.

---

*CodeAlpha Data Analytics Internship — Task 2*  
*Exploratory Data Analysis | Python · Pandas · Seaborn · SciPy*
