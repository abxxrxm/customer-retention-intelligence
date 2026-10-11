import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind


def overall_metrics(df):
    """
    Calculate high-level customer and churn metrics.
    """

    total_customers = len(df)
    churned_customers = df["churn"].sum()
    churn_rate = df["churn"].mean()

    return {
        "total_customers": total_customers,
        "churned_customers": int(churned_customers),
        "retained_customers": int(total_customers - churned_customers),
        "churn_rate": churn_rate
    }


def churn_by_category(df, column):
    """
    Calculate customer count and churn rate
    for a categorical variable.
    """

    result = (
        df.groupby(column)["churn"]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean"
        )
        .reset_index()
    )

    result["churn_rate"] = result["churn_rate"] * 100

    return result


def churn_by_numeric_group(df, column, bins):
    """
    Divide a numerical variable into groups
    and calculate churn rate for each group.
    """

    grouped_column = f"{column}_group"

    temp = df.copy()

    temp[grouped_column] = pd.cut(
        temp[column],
        bins=bins,
        include_lowest=True
    )

    result = (
        temp.groupby(grouped_column, observed=True)["churn"]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean"
        )
        .reset_index()
    )

    result["churn_rate"] = result["churn_rate"] * 100

    return result


def chi_square_test(df, variable1, variable2):
    """
    Perform a Chi-square test of independence
    between two categorical variables.
    """

    contingency_table = pd.crosstab(
        df[variable1],
        df[variable2]
    )

    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    n = contingency_table.to_numpy().sum()

    phi = np.sqrt(chi2 / n)

    return {
        "chi_square": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "effect_size": phi,
        "contingency_table": contingency_table
    }


def t_test_by_churn(df, numeric_variable):
    """
    Compare the mean of a numerical variable
    between retained and churned customers.
    """

    retained = df.loc[
        df["churn"] == 0,
        numeric_variable
    ]

    churned = df.loc[
        df["churn"] == 1,
        numeric_variable
    ]

    statistic, p_value = ttest_ind(
        retained,
        churned,
        equal_var=False
    )

    return {
        "retained_mean": retained.mean(),
        "churned_mean": churned.mean(),
        "t_statistic": statistic,
        "p_value": p_value
    }