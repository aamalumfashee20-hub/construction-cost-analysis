"""
Construction Cost Analysis & Data Quality Review
Dataset: illustrative/simulated BOQ and actual cost data.
Run from project root: python src/analyze_costs.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "construction_boq_cost_data.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
REPORTS = OUT / "reports"
CHARTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
required = ["item_id", "work_section", "description", "unit",
            "budget_quantity", "budget_rate_ngn", "actual_quantity",
            "actual_rate_ngn"]
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Recalculate amounts from source quantities and rates.
df["budget_amount_ngn"] = df["budget_quantity"] * df["budget_rate_ngn"]
df["actual_amount_ngn"] = df["actual_quantity"] * df["actual_rate_ngn"]
df["variance_ngn"] = df["actual_amount_ngn"] - df["budget_amount_ngn"]
df["variance_pct"] = np.where(
    df["budget_amount_ngn"] != 0,
    df["variance_ngn"] / df["budget_amount_ngn"] * 100,
    np.nan
)
df["variance_status"] = np.select(
    [df["variance_ngn"] > 0, df["variance_ngn"] < 0],
    ["Over Budget", "Under Budget"],
    default="On Budget"
)

budget_total = df["budget_amount_ngn"].sum()
actual_total = df["actual_amount_ngn"].sum()
total_variance = actual_total - budget_total
variance_pct = (total_variance / budget_total * 100) if budget_total else np.nan

section = (df.groupby("work_section", as_index=False)
           .agg(budget_ngn=("budget_amount_ngn", "sum"),
                actual_ngn=("actual_amount_ngn", "sum")))
section["variance_ngn"] = section["actual_ngn"] - section["budget_ngn"]
section["variance_pct"] = section["variance_ngn"] / section["budget_ngn"] * 100

# Data quality checks on the raw quality-check sample.
raw = pd.read_csv(ROOT / "data" / "construction_boq_quality_check_sample.csv")
quality_issues = []
for idx, row in raw.iterrows():
    if pd.isna(row["actual_rate_ngn"]):
        quality_issues.append((row.get("item_id", ""), "Missing actual rate", idx + 2))
    if pd.notna(row["actual_quantity"]) and row["actual_quantity"] < 0:
        quality_issues.append((row.get("item_id", ""), "Negative actual quantity", idx + 2))
    if isinstance(row["description"], str) and row["description"] != row["description"].strip():
        quality_issues.append((row.get("item_id", ""), "Leading/trailing whitespace", idx + 2))
duplicate_mask = raw["item_id"].duplicated(keep=False)
for idx in raw.index[duplicate_mask]:
    quality_issues.append((raw.loc[idx, "item_id"], "Duplicate item ID", idx + 2))

quality_df = pd.DataFrame(quality_issues, columns=["item_id", "issue", "csv_row"])
quality_df.to_csv(REPORTS / "data_quality_issues.csv", index=False)

df.to_csv(REPORTS / "analysed_cost_data.csv", index=False)
section.to_csv(REPORTS / "cost_summary_by_section.csv", index=False)

# Chart 1: budget vs actual by section.
x = np.arange(len(section))
width = 0.38
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - width/2, section["budget_ngn"] / 1e6, width, label="Budget")
ax.bar(x + width/2, section["actual_ngn"] / 1e6, width, label="Actual")
ax.set_xticks(x)
ax.set_xticklabels(section["work_section"], rotation=40, ha="right")
ax.set_ylabel("Cost (₦ million)")
ax.set_title("Budget vs Actual Cost by Work Section")
ax.legend()
fig.tight_layout()
fig.savefig(CHARTS / "budget_vs_actual_by_section.png", dpi=180)
plt.close(fig)

# Chart 2: largest cost variances.
top = df.sort_values("variance_ngn", ascending=False).head(10).sort_values("variance_ngn")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top["description"], top["variance_ngn"] / 1e6)
ax.set_xlabel("Variance (₦ million)")
ax.set_title("10 Largest Positive Cost Variances")
fig.tight_layout()
fig.savefig(CHARTS / "largest_cost_variances.png", dpi=180)
plt.close(fig)

# Chart 3: distribution of item status.
counts = df["variance_status"].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(counts.index, counts.values)
ax.set_ylabel("Number of BOQ items")
ax.set_title("BOQ Items by Budget Status")
fig.tight_layout()
fig.savefig(CHARTS / "variance_status_count.png", dpi=180)
plt.close(fig)

# Plain-text executive summary.
over_count = int((df["variance_status"] == "Over Budget").sum())
under_count = int((df["variance_status"] == "Under Budget").sum())
summary = f"""CONSTRUCTION COST ANALYSIS – EXECUTIVE SUMMARY
Dataset: Illustrative simulated BOQ dataset; not a live client project.

Total budget: ₦{budget_total:,.2f}
Total actual cost: ₦{actual_total:,.2f}
Net variance: ₦{total_variance:,.2f} ({variance_pct:.2f}%)
Items over budget: {over_count}
Items under budget: {under_count}
Data-quality issues identified in deliberately flawed QA sample: {len(quality_df)}

Interpretation:
A positive variance indicates actual cost above budget. A negative variance indicates
actual cost below budget. Section-level and item-level outputs should be reviewed
alongside scope, measurement basis, procurement records and approved variations.
This analysis is a portfolio demonstration, not a valuation or audit certificate.
"""
(REPORTS / "executive_summary.txt").write_text(summary, encoding="utf-8")
print(summary)
print(f"Outputs saved under: {OUT}")
