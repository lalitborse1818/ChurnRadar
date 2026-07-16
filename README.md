# data/

## Files in this folder

| File | Committed? | Rows | Description |
|---|---|---|---|
| `sample_raw_customers.csv` | ✅ Yes | 200 | Preview of the raw customer extract — lets you browse the schema on GitHub without running anything |
| `sample_features.csv` | ✅ Yes | 200 | Preview of the SQL-engineered feature table (post `sql/02_feature_engineering.sql`) |
| `raw_customers.csv` | ❌ Generated | 52,000 | Full raw customer extract, produced by `src/generate_data.py` |
| `features.csv` | ❌ Generated | 52,000 | Full feature table, produced by `src/db_utils.py` running the SQL pipeline |
| `churnradar.db` | ❌ Generated | — | SQLite database backing the SQL feature pipeline (`customers` table + `customer_features` table + the two aggregation views) |

The full files are excluded via `.gitignore` to keep the repo lightweight and
avoid committing large regenerable artifacts. Run this from the repo root to
produce them locally:

```bash
python src/pipeline.py
```

## Column reference (`features.csv` / `sample_features.csv`)

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Unique customer key |
| `age` | int | Customer age |
| `region` | str | North / South / East / West / Central |
| `tenure_months` | int | Months since signup |
| `contract_type` | str | Month-to-Month / One-Year / Two-Year |
| `plan_tier` | str | Basic / Standard / Premium |
| `monthly_charges` | float | Current monthly bill ($) |
| `total_charges` | float | Lifetime billed amount ($) |
| `payment_method` | str | Credit Card / Bank Transfer / Electronic Check / Mailed Check |
| `autopay_enrolled` | 0/1 | Enrolled in automatic payment |
| `paperless_billing` | 0/1 | Enrolled in paperless billing |
| `has_dependents` | 0/1 | Has dependents on the account |
| `num_products` | int | Number of bundled products (1–4) |
| `support_calls_last_90d` | int | Support contacts in the last 90 days |
| `late_payments_last_12m` | int | Late payments in the last 12 months |
| `usage_trend_pct` | float | % change in usage over the last 90 days |
| `satisfaction_score` | float | Self-reported satisfaction (0–10) |
| `discount_pct` | int | Active discount applied (%) |
| `tenure_segment` | str | *SQL-engineered:* New / Growing / Established / Loyal |
| `value_tier` | str | *SQL-engineered:* Low / Mid / High value, by total charges |
| `contract_risk_tier` | str | *SQL-engineered:* risk tier derived from contract type |
| `high_support_flag` | 0/1 | *SQL-engineered:* 3+ support calls in last 90 days |
| `late_payment_flag` | 0/1 | *SQL-engineered:* 2+ late payments in last 12 months |
| `low_engagement_billing_flag` | 0/1 | *SQL-engineered:* no autopay and no paperless billing |
| `bundled_customer_flag` | 0/1 | *SQL-engineered:* 3+ bundled products |
| `usage_trend_bucket` | str | *SQL-engineered:* Sharp Decline / Mild Decline / Stable-Growing / Sharp Growth |
| `price_per_tenure_month` | float | *SQL-engineered:* monthly charges ÷ tenure months |
| `satisfaction_vs_cohort` | float | *SQL-engineered:* satisfaction score vs. same-tier cohort average |
| `churned` | 0/1 | **Target** — whether the customer churned |

The `*SQL-engineered*` columns are built entirely in
[`sql/02_feature_engineering.sql`](../sql/02_feature_engineering.sql) — see
the main [README](../README.md#feature-engineering-sql) for the three
feature families (segmentation, behavioral flags, trend features).

`churnradar_powerbi_dataset.csv` (in [`powerbi/`](../powerbi/)) adds three
more columns on top of this table: `churn_probability`, `risk_band`, and
`predicted_churn` — the model's output, not raw/engineered data, so it lives
separately.
