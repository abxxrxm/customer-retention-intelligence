-- churn_analysis.sql
-- Churn analysis queries for data/customer_churn_dataset.csv
-- Dialect: SQLite 3 (also valid in PostgreSQL; see the note on rounding below).
-- Tested: all queries ran in SQLite 3.37 against the repo CSV (5,000 rows; overall churn 37.92%).
--
-- Load the CSV into a table called customers (SQLite shell):
--   sqlite3 churn.db
--   .mode csv
--   .import --skip 1 data/customer_churn_dataset.csv customers_raw
-- Then create the typed table below and fill it from customers_raw.
-- In PostgreSQL, create the same table and use \copy customers FROM 'data/customer_churn_dataset.csv' CSV HEADER
--
-- Columns: customer_id, age, gender, subscription_type, monthly_fee, tenure_months,
--          monthly_usage_hours, support_tickets, payment_failures, discount,
--          satisfaction_score, competitor_usage, churn (1 = churned, 0 = retained)

CREATE TABLE IF NOT EXISTS customers (
    customer_id         TEXT PRIMARY KEY,
    age                 INTEGER,
    gender              TEXT,
    subscription_type   TEXT,
    monthly_fee         REAL,
    tenure_months       INTEGER,
    monthly_usage_hours REAL,
    support_tickets     INTEGER,
    payment_failures    INTEGER,
    discount            REAL,
    satisfaction_score  INTEGER,
    competitor_usage    INTEGER,
    churn               INTEGER
);

-- SQLite only: fill the typed table from the raw import.
-- INSERT INTO customers SELECT * FROM customers_raw;


-- 1. Overall churn rate
SELECT
    COUNT(*)                          AS customers,
    SUM(churn)                        AS churned,
    ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
FROM customers;


-- 2. Churn rate by subscription type
SELECT
    subscription_type,
    COUNT(*)                          AS customers,
    SUM(churn)                        AS churned,
    ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct,
    ROUND(AVG(monthly_fee), 2)        AS avg_monthly_fee
FROM customers
GROUP BY subscription_type
ORDER BY churn_rate_pct DESC;


-- 3. Churn rate by tenure band
SELECT
    CASE
        WHEN tenure_months <= 12 THEN '1. 0-12 months'
        WHEN tenure_months <= 24 THEN '2. 13-24 months'
        WHEN tenure_months <= 48 THEN '3. 25-48 months'
        ELSE                          '4. 49+ months'
    END                               AS tenure_band,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY tenure_band;


-- 4. Churn rate by satisfaction score
SELECT
    satisfaction_score,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
FROM customers
GROUP BY satisfaction_score
ORDER BY satisfaction_score;


-- 5. Churn rate by payment failures and support tickets
-- (counts capped at 3+ so small groups stay readable)
SELECT
    MIN(payment_failures, 3)          AS payment_failures_capped,
    MIN(support_tickets, 3)           AS support_tickets_capped,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
FROM customers
GROUP BY payment_failures_capped, support_tickets_capped
ORDER BY payment_failures_capped, support_tickets_capped;
-- PostgreSQL: replace MIN(a, 3) with LEAST(a, 3) and group by the expressions or ordinals (GROUP BY 1, 2).


-- 6. Competitor usage vs churn, split by subscription type
SELECT
    subscription_type,
    competitor_usage,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
FROM customers
GROUP BY subscription_type, competitor_usage
ORDER BY subscription_type, competitor_usage;


-- 7. Churned vs retained: average profile
SELECT
    churn,
    COUNT(*)                          AS customers,
    ROUND(AVG(age), 1)                AS avg_age,
    ROUND(AVG(tenure_months), 1)      AS avg_tenure_months,
    ROUND(AVG(monthly_usage_hours), 1) AS avg_usage_hours,
    ROUND(AVG(support_tickets), 2)    AS avg_support_tickets,
    ROUND(AVG(payment_failures), 2)   AS avg_payment_failures,
    ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction
FROM customers
GROUP BY churn;


-- 8. Churn rate by age group and gender
SELECT
    CASE
        WHEN age < 30 THEN '18-29'
        WHEN age < 45 THEN '30-44'
        WHEN age < 60 THEN '45-59'
        ELSE               '60+'
    END                               AS age_group,
    gender,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
FROM customers
GROUP BY age_group, gender
ORDER BY age_group, gender;


-- 9. Highest-risk segments: low satisfaction, payment failures and competitor usage together
-- Segment definition is an analyst choice, not a model output.
SELECT
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
FROM customers
WHERE satisfaction_score <= 2
  AND payment_failures >= 1
  AND competitor_usage = 1;
