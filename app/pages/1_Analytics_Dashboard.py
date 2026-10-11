import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(APP_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "customer_churn_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


try:

    df = load_data()

except Exception as e:

    st.error(
        "Unable to load the customer churn dataset."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📊 Churn Analytics Dashboard")

st.write(
    """
    Explore customer churn patterns and identify the
    characteristics associated with customer retention
    and churn.
    """
)

st.divider()


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

total_customers = len(df)

churned_customers = int(
    df["churn"].sum()
)

retained_customers = (
    total_customers - churned_customers
)

churn_rate = (
    churned_customers / total_customers
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )


with col3:

    st.metric(
        "Retained Customers",
        f"{retained_customers:,}"
    )


with col4:

    st.metric(
        "Overall Churn Rate",
        f"{churn_rate:.1%}"
    )


st.divider()


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

st.header("📌 Customer Churn Overview")


overview_col1, overview_col2 = st.columns(2)


with overview_col1:

    churn_counts = (
        df["churn"]
        .value_counts()
        .sort_index()
    )

    churn_labels = [
        "Retained",
        "Churned"
    ]

    fig, ax = plt.subplots()

    ax.bar(
        churn_labels,
        [
            churn_counts.get(0, 0),
            churn_counts.get(1, 0)
        ]
    )

    ax.set_ylabel("Number of Customers")

    ax.set_title(
        "Customer Retention vs Churn"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with overview_col2:

    churn_summary = pd.DataFrame({
        "Status": [
            "Retained",
            "Churned"
        ],

        "Customers": [
            retained_customers,
            churned_customers
        ],

        "Percentage": [
            retained_customers / total_customers * 100,
            churned_customers / total_customers * 100
        ]
    })

    churn_summary["Percentage"] = (
        churn_summary["Percentage"]
        .round(2)
    )

    st.dataframe(
        churn_summary,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# SUBSCRIPTION ANALYSIS
# ============================================================

st.header("💳 Churn by Subscription Type")


subscription_summary = (
    df.groupby("subscription_type")
    .agg(
        customers=("churn", "count"),
        churned=("churn", "sum")
    )
    .reset_index()
)

subscription_summary["churn_rate"] = (
    subscription_summary["churned"]
    /
    subscription_summary["customers"]
    * 100
)


sub_col1, sub_col2 = st.columns(2)


with sub_col1:

    fig, ax = plt.subplots()

    ax.bar(
        subscription_summary["subscription_type"],
        subscription_summary["churn_rate"]
    )

    ax.set_ylabel("Churn Rate (%)")

    ax.set_xlabel(
        "Subscription Type"
    )

    ax.set_title(
        "Churn Rate by Subscription Type"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with sub_col2:

    display_subscription = (
        subscription_summary
        .copy()
    )

    display_subscription["churn_rate"] = (
        display_subscription["churn_rate"]
        .round(2)
    )

    st.dataframe(
        display_subscription,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# USAGE ANALYSIS
# ============================================================

st.header("⏱️ Churn by Monthly Usage")


usage_bins = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    float("inf")
]

usage_labels = [
    "0–10",
    "10–20",
    "20–30",
    "30–40",
    "40–50",
    "50–60",
    "60+"
]


df["usage_group"] = pd.cut(
    df["monthly_usage_hours"],
    bins=usage_bins,
    labels=usage_labels,
    include_lowest=True
)


usage_summary = (
    df.groupby(
        "usage_group",
        observed=False
    )
    .agg(
        customers=("churn", "count"),
        churned=("churn", "sum")
    )
    .reset_index()
)

usage_summary["churn_rate"] = (
    usage_summary["churned"]
    /
    usage_summary["customers"]
    * 100
)


fig, ax = plt.subplots()

ax.plot(
    usage_summary["usage_group"].astype(str),
    usage_summary["churn_rate"],
    marker="o"
)

ax.set_xlabel(
    "Monthly Usage Hours"
)

ax.set_ylabel(
    "Churn Rate (%)"
)

ax.set_title(
    "Churn Rate Across Usage Groups"
)

plt.xticks(rotation=30)

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# TENURE ANALYSIS
# ============================================================

st.header("📅 Churn by Customer Tenure")


tenure_bins = [
    0,
    12,
    24,
    36,
    48,
    60,
    float("inf")
]

tenure_labels = [
    "0–12",
    "12–24",
    "24–36",
    "36–48",
    "48–60",
    "60+"
]


df["tenure_group"] = pd.cut(
    df["tenure_months"],
    bins=tenure_bins,
    labels=tenure_labels,
    include_lowest=True
)


tenure_summary = (
    df.groupby(
        "tenure_group",
        observed=False
    )
    .agg(
        customers=("churn", "count"),
        churned=("churn", "sum")
    )
    .reset_index()
)

tenure_summary["churn_rate"] = (
    tenure_summary["churned"]
    /
    tenure_summary["customers"]
    * 100
)


fig, ax = plt.subplots()

ax.plot(
    tenure_summary["tenure_group"].astype(str),
    tenure_summary["churn_rate"],
    marker="o"
)

ax.set_xlabel(
    "Tenure (Months)"
)

ax.set_ylabel(
    "Churn Rate (%)"
)

ax.set_title(
    "Churn Rate Across Tenure Groups"
)

plt.xticks(rotation=30)

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


st.divider()


# ============================================================
# SATISFACTION ANALYSIS
# ============================================================

st.header("⭐ Churn by Satisfaction Score")


satisfaction_summary = (
    df.groupby("satisfaction_score")
    .agg(
        customers=("churn", "count"),
        churned=("churn", "sum")
    )
    .reset_index()
)

satisfaction_summary["churn_rate"] = (
    satisfaction_summary["churned"]
    /
    satisfaction_summary["customers"]
    * 100
)


fig, ax = plt.subplots()

ax.plot(
    satisfaction_summary["satisfaction_score"],
    satisfaction_summary["churn_rate"],
    marker="o"
)

ax.set_xlabel(
    "Satisfaction Score"
)

ax.set_ylabel(
    "Churn Rate (%)"
)

ax.set_title(
    "Churn Rate by Satisfaction Score"
)

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


st.divider()


# ============================================================
# SUPPORT TICKETS
# ============================================================

st.header("🎫 Support Tickets and Churn")


support_summary = (
    df.groupby("support_tickets")
    .agg(
        customers=("churn", "count"),
        churned=("churn", "sum")
    )
    .reset_index()
)

support_summary["churn_rate"] = (
    support_summary["churned"]
    /
    support_summary["customers"]
    * 100
)


fig, ax = plt.subplots()

ax.plot(
    support_summary["support_tickets"],
    support_summary["churn_rate"],
    marker="o"
)

ax.set_xlabel(
    "Support Tickets"
)

ax.set_ylabel(
    "Churn Rate (%)"
)

ax.set_title(
    "Churn Rate by Number of Support Tickets"
)

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


st.divider()


# ============================================================
# COMPETITOR USAGE
# ============================================================

st.header("🏆 Competitor Usage and Churn")


competitor_summary = (
    df.groupby("competitor_usage")
    .agg(
        customers=("churn", "count"),
        churned=("churn", "sum")
    )
    .reset_index()
)

competitor_summary["churn_rate"] = (
    competitor_summary["churned"]
    /
    competitor_summary["customers"]
    * 100
)

competitor_summary["competitor_usage"] = (
    competitor_summary["competitor_usage"]
    .map({
        0: "No",
        1: "Yes"
    })
)


comp_col1, comp_col2 = st.columns(2)


with comp_col1:

    fig, ax = plt.subplots()

    ax.bar(
        competitor_summary["competitor_usage"],
        competitor_summary["churn_rate"]
    )

    ax.set_xlabel(
        "Competitor Usage"
    )

    ax.set_ylabel(
        "Churn Rate (%)"
    )

    ax.set_title(
        "Churn Rate by Competitor Usage"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with comp_col2:

    st.dataframe(
        competitor_summary,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# KEY FINDINGS
# ============================================================

st.header("🔎 Key Findings")


st.write(
    """
    ### Customer churn patterns identified from the analysis
    """
)

finding_col1, finding_col2 = st.columns(2)


with finding_col1:

    st.info(
        """
        **Satisfaction**

        Satisfaction score shows a strong negative
        relationship with churn. Customers with lower
        satisfaction are more likely to churn.
        """
    )

    st.info(
        """
        **Support Tickets**

        Higher support-ticket activity is associated
        with increased churn risk.
        """
    )

    st.info(
        """
        **Payment Failures**

        Customers experiencing payment failures
        demonstrate higher churn risk.
        """
    )


with finding_col2:

    st.info(
        """
        **Competitor Usage**

        Competitor usage is positively associated
        with customer churn.
        """
    )

    st.info(
        """
        **Monthly Usage**

        Lower product usage is associated with
        higher churn rates.
        """
    )

    st.info(
        """
        **Customer Tenure**

        Customers with shorter tenure demonstrate
        higher churn rates.
        """
    )


st.divider()


# ============================================================
# CORRELATION SUMMARY
# ============================================================

st.header("📊 Correlation with Churn")


numeric_columns = [
    "age",
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


correlation = (
    df[numeric_columns]
    .corr()["churn"]
    .drop("churn")
    .sort_values(
        ascending=False
    )
)


correlation_df = (
    correlation
    .reset_index()
)

correlation_df.columns = [
    "Feature",
    "Correlation with Churn"
]

correlation_df[
    "Correlation with Churn"
] = correlation_df[
    "Correlation with Churn"
].round(3)


st.dataframe(
    correlation_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Retention Intelligence | "
    "Churn Analytics Dashboard"
)