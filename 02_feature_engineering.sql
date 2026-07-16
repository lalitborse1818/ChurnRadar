-- ============================================================
-- ChurnRadar :: Feature Engineering
-- Builds the model-ready feature table directly in SQL so the same
-- logic can run against a warehouse (Snowflake/BigQuery/Postgres)
-- with minimal changes. Three feature families, matching the
-- resume-level description of the pipeline:
--   1. Segmentation      (tenure bands, value tier, contract risk tier)
--   2. Behavioral flags   (high support usage, late-payment flag, autopay,
--                          multi-product bundling)
--   3. Trend features     (usage trend bucket, price-per-tenure-month,
--                          satisfaction deviation from cohort mean)
-- ============================================================

DROP TABLE IF EXISTS customer_features;

CREATE TABLE customer_features AS
WITH cohort AS (
    SELECT
        plan_tier,
        AVG(satisfaction_score) AS cohort_avg_satisfaction
    FROM customers
    GROUP BY plan_tier
)
SELECT
    c.customer_id,
    c.age,
    c.region,
    c.tenure_months,
    c.contract_type,
    c.plan_tier,
    c.monthly_charges,
    c.total_charges,
    c.payment_method,
    c.autopay_enrolled,
    c.paperless_billing,
    c.has_dependents,
    c.num_products,
    c.support_calls_last_90d,
    c.late_payments_last_12m,
    c.usage_trend_pct,
    c.satisfaction_score,
    c.discount_pct,

    -- ---- 1. Segmentation ----
    CASE
        WHEN c.tenure_months < 6  THEN 'New (0-6mo)'
        WHEN c.tenure_months < 18 THEN 'Growing (6-18mo)'
        WHEN c.tenure_months < 36 THEN 'Established (18-36mo)'
        ELSE 'Loyal (36mo+)'
    END AS tenure_segment,

    CASE
        WHEN c.total_charges >= 3000 THEN 'High Value'
        WHEN c.total_charges >= 1000 THEN 'Mid Value'
        ELSE 'Low Value'
    END AS value_tier,

    CASE
        WHEN c.contract_type = 'Month-to-Month' THEN 'High Risk Contract'
        WHEN c.contract_type = 'One-Year' THEN 'Medium Risk Contract'
        ELSE 'Low Risk Contract'
    END AS contract_risk_tier,

    -- ---- 2. Behavioral flags ----
    CASE WHEN c.support_calls_last_90d >= 3 THEN 1 ELSE 0 END AS high_support_flag,
    CASE WHEN c.late_payments_last_12m >= 2 THEN 1 ELSE 0 END AS late_payment_flag,
    CASE WHEN c.autopay_enrolled = 0 AND c.paperless_billing = 0 THEN 1 ELSE 0 END AS low_engagement_billing_flag,
    CASE WHEN c.num_products >= 3 THEN 1 ELSE 0 END AS bundled_customer_flag,

    -- ---- 3. Trend / derived features ----
    CASE
        WHEN c.usage_trend_pct <= -15 THEN 'Sharp Decline'
        WHEN c.usage_trend_pct < 0   THEN 'Mild Decline'
        WHEN c.usage_trend_pct < 15  THEN 'Stable/Growing'
        ELSE 'Sharp Growth'
    END AS usage_trend_bucket,

    ROUND(c.monthly_charges / NULLIF(c.tenure_months, 0), 2) AS price_per_tenure_month,
    ROUND(c.satisfaction_score - coh.cohort_avg_satisfaction, 2) AS satisfaction_vs_cohort,

    c.churned
FROM customers c
JOIN cohort coh ON coh.plan_tier = c.plan_tier;

-- Quick sanity checks / example analytical queries the pipeline also exposes
-- to Power BI as pre-aggregated views.

DROP VIEW IF EXISTS v_churn_rate_by_segment;
CREATE VIEW v_churn_rate_by_segment AS
SELECT
    tenure_segment,
    contract_risk_tier,
    COUNT(*)                                   AS customers,
    SUM(churned)                                AS churned_customers,
    ROUND(1.0 * SUM(churned) / COUNT(*), 4)     AS churn_rate
FROM customer_features
GROUP BY tenure_segment, contract_risk_tier
ORDER BY churn_rate DESC;

DROP VIEW IF EXISTS v_churn_rate_by_flags;
CREATE VIEW v_churn_rate_by_flags AS
SELECT
    high_support_flag,
    late_payment_flag,
    low_engagement_billing_flag,
    bundled_customer_flag,
    COUNT(*)                                AS customers,
    ROUND(1.0 * SUM(churned) / COUNT(*), 4) AS churn_rate
FROM customer_features
GROUP BY high_support_flag, late_payment_flag, low_engagement_billing_flag, bundled_customer_flag
ORDER BY churn_rate DESC;
