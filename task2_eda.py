"""
============================================================
  CodeAlpha Internship — TASK 2: Exploratory Data Analysis
  Dataset: Titanic (famous beginner-friendly public dataset)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── STYLE ─────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── LOAD DATASET ──────────────────────────────────────────
print("=" * 55)
print("  CodeAlpha Internship — Task 2: EDA (Titanic)")
print("=" * 55)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(f"\n[✓] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── STEP 1: ASK MEANINGFUL QUESTIONS ─────────────────────
print("\n" + "─" * 55)
print("  STEP 1: Questions We Will Answer")
print("─" * 55)
questions = [
    "1. What is the overall survival rate?",
    "2. Did gender affect survival?",
    "3. Did passenger class affect survival?",
    "4. What was the age distribution of passengers?",
    "5. Are there missing values / data issues?",
]
for q in questions:
    print("  ", q)

# ── STEP 2: EXPLORE DATA STRUCTURE ───────────────────────
print("\n" + "─" * 55)
print("  STEP 2: Data Structure")
print("─" * 55)
print("\nColumn Info:")
print(df.dtypes.to_string())
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nBasic Stats:")
print(df.describe().to_string())

# ── STEP 3: IDENTIFY TRENDS & PATTERNS ───────────────────
print("\n" + "─" * 55)
print("  STEP 3: Trends & Patterns")
print("─" * 55)

survival_rate = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate : {survival_rate:.1f}%")

gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print("\nSurvival by Gender:")
print(gender_survival.to_string())

class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print("\nSurvival by Passenger Class:")
print(class_survival.to_string())

# ── STEP 4: TEST HYPOTHESES WITH VISUALIZATION ───────────
print("\n" + "─" * 55)
print("  STEP 4: Visualizations")
print("─" * 55)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic EDA — CodeAlpha Internship Task 2",
             fontsize=16, fontweight="bold", y=1.01)

# Plot 1 — Survival Count
sns.countplot(x="Survived", data=df, ax=axes[0, 0],
              palette=["#e74c3c", "#2ecc71"])
axes[0, 0].set_title("Survival Count")
axes[0, 0].set_xticklabels(["Did Not Survive", "Survived"])
axes[0, 0].set_xlabel("")

# Plot 2 — Survival by Gender
sns.barplot(x="Sex", y="Survived", data=df, ax=axes[0, 1],
            palette=["#3498db", "#e91e63"], ci=None)
axes[0, 1].set_title("Survival Rate by Gender")
axes[0, 1].set_ylabel("Survival Rate")

# Plot 3 — Survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=df, ax=axes[0, 2],
            palette="Blues_d", ci=None)
axes[0, 2].set_title("Survival Rate by Passenger Class")
axes[0, 2].set_ylabel("Survival Rate")
axes[0, 2].set_xlabel("Class (1=First, 3=Third)")

# Plot 4 — Age Distribution
df["Age"].dropna().plot(kind="hist", bins=30, ax=axes[1, 0],
                         color="#9b59b6", edgecolor="white")
axes[1, 0].set_title("Age Distribution of Passengers")
axes[1, 0].set_xlabel("Age")
axes[1, 0].set_ylabel("Count")

# Plot 5 — Fare Distribution
df["Fare"].plot(kind="hist", bins=40, ax=axes[1, 1],
                color="#1abc9c", edgecolor="white")
axes[1, 1].set_title("Fare Distribution")
axes[1, 1].set_xlabel("Fare (£)")

# Plot 6 — Missing Values Heatmap
missing = df.isnull().sum().reset_index()
missing.columns = ["Column", "Missing"]
missing = missing[missing["Missing"] > 0]
sns.barplot(x="Missing", y="Column", data=missing,
            ax=axes[1, 2], palette="Reds_r")
axes[1, 2].set_title("Missing Values per Column")
axes[1, 2].set_xlabel("Count of Missing")

plt.tight_layout()
plt.savefig("eda_titanic_analysis.png", bbox_inches="tight")
print("[✓] Chart saved: eda_titanic_analysis.png")
plt.show()

# ── STEP 5: DATA ISSUES ───────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 5: Data Issues Detected")
print("─" * 55)
missing_vals = df.isnull().sum()
missing_pct  = (missing_vals / len(df) * 100).round(1)
issues_df    = pd.DataFrame({"Missing Count": missing_vals,
                              "Missing %": missing_pct})
issues_df    = issues_df[issues_df["Missing Count"] > 0]
print("\nColumns with Missing Data:")
print(issues_df.to_string())
print("\nSuggested Fixes:")
print("  Age     → Fill with median age (or by Pclass median)")
print("  Cabin   → Drop column (77% missing — not useful)")
print("  Embarked→ Fill with mode (most common port)")

# ── SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EDA SUMMARY")
print("=" * 55)
print(f"  • Dataset : Titanic ({df.shape[0]} passengers)")
print(f"  • Survival Rate : {survival_rate:.1f}%")
print(f"  • Women survived {gender_survival['female']:.1f}% vs Men {gender_survival['male']:.1f}%")
print(f"  • 1st Class survival: {class_survival[1]:.1f}%  |  3rd Class: {class_survival[3]:.1f}%")
print(f"  • Missing: Age ({missing_vals.get('Age',0)}), Cabin ({missing_vals.get('Cabin',0)})")
print("\n[✓] Task 2 Complete! Upload .py + .png to GitHub.")
print("=" * 55)
