-- ============================================================
-- ChurnRadar :: Schema
-- Raw customer table as it would land from a CRM / billing extract.
-- ============================================================

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id            INTEGER PRIMARY KEY,
    age                     INTEGER,
    region                  TEXT,
    tenure_months           INTEGER,
    contract_type           TEXT,
    plan_tier               TEXT,
    monthly_charges         REAL,
    total_charges           REAL,
    payment_method           TEXT,
    autopay_enrolled         INTEGER,
    paperless_billing        INTEGER,
    has_dependents            INTEGER,
    num_products              INTEGER,
    support_calls_last_90d    INTEGER,
    late_payments_last_12m    INTEGER,
    usage_trend_pct           REAL,
    satisfaction_score        REAL,
    discount_pct               INTEGER,
    churned                    INTEGER
);

CREATE INDEX idx_customers_contract ON customers(contract_type);
CREATE INDEX idx_customers_region ON customers(region);
