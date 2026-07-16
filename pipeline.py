"""
pipeline.py
-----------
Runs the full ChurnRadar pipeline end-to-end:
  1. Generate/refresh raw customer data
  2. Load into SQLite + run SQL feature engineering
  3. Train XGBoost + evaluate
  4. Run SHAP explainability
  5. Export the Power BI-ready dataset

    python src/pipeline.py
"""
import time
from generate_data import generate_customers
from config import RAW_CSV
import db_utils
import train_model
import explainability
import powerbi_export


def main():
    t0 = time.time()

    print("=" * 60)
    print("STEP 1/5 — Generating raw customer data")
    print("=" * 60)
    df = generate_customers()
    df.to_csv(RAW_CSV, index=False)
    print(f"{len(df):,} records -> {RAW_CSV}  (churn rate: {df['churned'].mean():.2%})\n")

    print("=" * 60)
    print("STEP 2/5 — SQL feature engineering (SQLite)")
    print("=" * 60)
    db_utils.build_feature_table()
    print()

    print("=" * 60)
    print("STEP 3/5 — Training XGBoost model")
    print("=" * 60)
    _, _, _, metrics = train_model.train()
    print()

    print("=" * 60)
    print("STEP 4/5 — SHAP explainability")
    print("=" * 60)
    explainability.run_shap_analysis()
    print()

    print("=" * 60)
    print("STEP 5/5 — Power BI export")
    print("=" * 60)
    powerbi_export.build_powerbi_dataset()
    print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"DONE in {elapsed:.1f}s — AUC-ROC: {metrics['auc_roc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
