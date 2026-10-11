import pandas as pd


EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "gender",
    "subscription_type",
    "monthly_fee",
    "tenure_months",
    "monthly_usage_hours",
    "support_tickets",
    "payment_failures",
    "discount",
    "satisfaction_score",
    "competitor_usage",
    "churn"
]


def validate_data(df):

    errors = []

    # Check columns
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Check missing values
    if df.isnull().sum().sum() > 0:
        errors.append("Dataset contains missing values.")

    # Check duplicate rows
    if df.duplicated().sum() > 0:
        errors.append("Dataset contains duplicate rows.")

    # Check unique customer IDs
    if df["customer_id"].nunique() != len(df):
        errors.append("Customer IDs are not unique.")

    # Check age
    if not df["age"].between(18, 100).all():
        errors.append("Age contains invalid values.")

    # Check satisfaction score
    if not df["satisfaction_score"].between(1, 5).all():
        errors.append("Satisfaction score contains invalid values.")

    # Check churn
    if not df["churn"].isin([0, 1]).all():
        errors.append("Churn contains values other than 0 and 1.")

    if errors:
        return False, errors

    return True, ["Data validation passed."]