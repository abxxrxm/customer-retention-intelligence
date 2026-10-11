import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODERN SAAS THEME & CSS VARIABLES
# ============================================================

st.markdown("""
<style>
    /* Hide Streamlit Default Sidebar Multi-Page Navigation */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Glassmorphism Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    .hero-title {
        font-size: 2rem !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Metadata Pills */
    .meta-chip {
        display: inline-block;
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
    }

    /* KPI Display Cards */
    .kpi-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .kpi-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.7;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
    }

    /* Custom Risk Status Badges */
    .risk-banner-high {
        background-color: rgba(239, 68, 68, 0.12);
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        color: #dc2626;
        font-weight: 700;
    }
    .risk-banner-medium {
        background-color: rgba(245, 158, 11, 0.12);
        border-left: 6px solid #f59e0b;
        border-radius: 8px;
        padding: 16px;
        color: #d97706;
        font-weight: 700;
    }
    .risk-banner-low {
        background-color: rgba(16, 185, 129, 0.12);
        border-left: 6px solid #10b981;
        border-radius: 8px;
        padding: 16px;
        color: #059669;
        font-weight: 700;
    }

    /* Drivers / Factors Box Styling */
    .driver-card-risk {
        background-color: rgba(239, 68, 68, 0.05);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 10px;
        padding: 16px;
        min-height: 140px;
    }
    .driver-card-protective {
        background-color: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 10px;
        padding: 16px;
        min-height: 140px;
    }

    /* Embed Wrapper */
    .powerbi-wrapper {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }

    /* Finding Item Box */
    .finding-card {
        background-color: var(--secondary-background-color);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT PATHS & MODEL LOADING
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

LOGISTIC_PATH = os.path.join(MODELS_DIR, "logistic_model.pkl")
LIGHTGBM_PATH = os.path.join(MODELS_DIR, "lightgbm_model.pkl")
CONFIG_PATH = os.path.join(MODELS_DIR, "ensemble_config.pkl")


@st.cache_resource
def load_models():
    logistic_model = joblib.load(LOGISTIC_PATH)
    lightgbm_model = joblib.load(LIGHTGBM_PATH)
    config = joblib.load(CONFIG_PATH)
    return logistic_model, lightgbm_model, config


try:
    logistic_model, lightgbm_model, config = load_models()
except Exception as e:
    st.error("Unable to load saved machine learning models.")
    st.exception(e)
    st.stop()


# MODEL CONFIGURATION
LOGISTIC_WEIGHT = config["logistic_weight"]
LIGHTGBM_WEIGHT = config["lightgbm_weight"]
THRESHOLD = config["threshold"]


# ============================================================
# HERO HEADER BANNER
# ============================================================

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">⚡ Customer Retention Intelligence Platform</div>
    <div style="font-size: 0.95rem; opacity: 0.85; margin-bottom: 12px;">
        Real-time predictive churn risk modeling & executive business analytics suite.
    </div>
    <div>
        <span class="meta-chip">🤖 Logistic + LightGBM Ensemble</span>
        <span class="meta-chip">⚙️ Weighting: 70/30</span>
        <span class="meta-chip">🟢 Status: Engine Active</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("### 📌 Navigation")
    
    selected_view = st.radio(
        "Select Module View",
        options=[
            "🎯 Individual Churn Predictor App",
            "📊 Executive Power BI Dashboard",
            "📊 Churn Analytics Dashboard"
        ],
        index=0
    )

    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Config")
    st.caption("MODEL WEIGHTING RATIO")
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        st.metric("Logistic", f"{LOGISTIC_WEIGHT:.0%}")
    with sb_col2:
        st.metric("LightGBM", f"{LIGHTGBM_WEIGHT:.0%}")
        
    st.markdown("---")
    st.caption("ACTIONABLE THRESHOLD")
    st.metric("Risk Cutoff", f"{THRESHOLD:.0%}")

    st.info(
        "💡 **Policy Rule:** Any account exceeding the risk threshold is flagged for immediate retention intervention."
    )


# ============================================================
# MAIN APPLICATION MODULES
# ============================================================

# ------------------------------------------------------------
# MODULE 1: INDIVIDUAL CHURN PREDICTOR APP
# ------------------------------------------------------------
if selected_view == "🎯 Individual Churn Predictor App":
    st.markdown("### 👤 Single Customer Churn Risk Calculator")
    st.caption("Select a preset profile for quick testing, or manually enter customer parameters below.")

    # Preset Test Profiles
    preset_cols = st.columns([1, 1, 1, 2])
    
    if "age" not in st.session_state:
        st.session_state.age = 35
        st.session_state.monthly_fee = 50.0
        st.session_state.tenure_months = 12.0
        st.session_state.monthly_usage_hours = 30.0
        st.session_state.support_tickets = 2
        st.session_state.payment_failures = 0
        st.session_state.discount = 5.0
        st.session_state.satisfaction_score = 3
        st.session_state.competitor_usage = 0

    with preset_cols[0]:
        if st.button("🔴 High-Risk Profile", use_container_width=True):
            st.session_state.age = 29
            st.session_state.monthly_fee = 110.0
            st.session_state.tenure_months = 4.0
            st.session_state.monthly_usage_hours = 8.0
            st.session_state.support_tickets = 6
            st.session_state.payment_failures = 3
            st.session_state.discount = 0.0
            st.session_state.satisfaction_score = 1
            st.session_state.competitor_usage = 1
            st.rerun()

    with preset_cols[1]:
        if st.button("🟢 Loyal Profile", use_container_width=True):
            st.session_state.age = 48
            st.session_state.monthly_fee = 45.0
            st.session_state.tenure_months = 36.0
            st.session_state.monthly_usage_hours = 65.0
            st.session_state.support_tickets = 0
            st.session_state.payment_failures = 0
            st.session_state.discount = 15.0
            st.session_state.satisfaction_score = 5
            st.session_state.competitor_usage = 0
            st.rerun()

    with preset_cols[2]:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.age = 35
            st.session_state.monthly_fee = 50.0
            st.session_state.tenure_months = 12.0
            st.session_state.monthly_usage_hours = 30.0
            st.session_state.support_tickets = 2
            st.session_state.payment_failures = 0
            st.session_state.discount = 5.0
            st.session_state.satisfaction_score = 3
            st.session_state.competitor_usage = 0
            st.rerun()

    # Input Form
    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics & Tier**")
            age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.age, step=1)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            subscription_type = st.selectbox("Subscription Tier", ["Basic", "Premium", "Standard"])

        with col2:
            st.markdown("**Financial & Engagement**")
            monthly_fee = st.number_input("Monthly Fee ($)", min_value=0.0, max_value=500.0, value=st.session_state.monthly_fee, step=0.01)
            tenure_months = st.number_input("Tenure (Months)", min_value=0.0, max_value=120.0, value=st.session_state.tenure_months, step=1.0)
            monthly_usage_hours = st.number_input("Usage Hours / Month", min_value=0.0, max_value=200.0, value=st.session_state.monthly_usage_hours, step=0.5)

        with col3:
            st.markdown("**Service Indicators**")
            support_tickets = st.number_input("Support Tickets Logged", min_value=0, max_value=50, value=st.session_state.support_tickets, step=1)
            payment_failures = st.number_input("Payment Failures", min_value=0, max_value=20, value=st.session_state.payment_failures, step=1)
            discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=st.session_state.discount, step=0.5)

        st.markdown("---")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            satisfaction_score = st.slider("Satisfaction Rating (1 = Dissatisfied, 5 = Highly Satisfied)", min_value=1, max_value=5, value=st.session_state.satisfaction_score, step=1)
        with col_s2:
            competitor_usage = st.selectbox("Known Competitor Usage?", options=[0, 1], index=st.session_state.competitor_usage, format_func=lambda x: "No" if x == 0 else "Yes")

        submit_button = st.form_submit_button(label="🚀 Calculate Churn Probability", type="primary", use_container_width=True)

    # Results Assessment
    if submit_button:
        customer_data = pd.DataFrame({
            "age": [age],
            "monthly_fee": [monthly_fee],
            "tenure_months": [tenure_months],
            "monthly_usage_hours": [monthly_usage_hours],
            "support_tickets": [support_tickets],
            "payment_failures": [payment_failures],
            "discount": [discount],
            "satisfaction_score": [satisfaction_score],
            "competitor_usage": [competitor_usage],
            "gender": [gender],
            "subscription_type": [subscription_type]
        })

        # Calculations
        logistic_probability = logistic_model.predict_proba(customer_data)[0, 1]
        lightgbm_probability = lightgbm_model.predict_proba(customer_data)[0, 1]
        churn_probability = (LOGISTIC_WEIGHT * logistic_probability) + (LIGHTGBM_WEIGHT * lightgbm_probability)
        churn_percentage = churn_probability * 100

        # Risk Banner
        if churn_probability >= 0.70:
            risk_level, risk_banner = "HIGH RISK ACCOUNT", "risk-banner-high"
        elif churn_probability >= THRESHOLD:
            risk_level, risk_banner = "MEDIUM RISK ACCOUNT", "risk-banner-medium"
        else:
            risk_level, risk_banner = "LOW RISK ACCOUNT", "risk-banner-low"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="{risk_banner}">⚠️ ASSESSMENT STATUS: {risk_level} ({churn_percentage:.1f}% Churn Probability)</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Custom KPI Scorecards
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Ensemble Churn Score</div>
                <div class="kpi-value">{churn_percentage:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Decision Cutoff</div>
                <div class="kpi-value">{THRESHOLD * 100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            delta_val = churn_percentage - (THRESHOLD * 100)
            delta_str = f"+{delta_val:.1f}% Above Cutoff" if delta_val >= 0 else f"{delta_val:.1f}% Below Cutoff"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Threshold Variance</div>
                <div class="kpi-value" style="font-size: 1.5rem; margin-top: 8px;">{delta_str}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(churn_probability))

        # Risk Factors
        st.markdown("#### 🔎 Driver Identification & Risk Factors")

        risk_factors = []
        protective_factors = []

        if satisfaction_score <= 2: risk_factors.append("Low customer satisfaction rating")
        elif satisfaction_score >= 4: protective_factors.append("High customer satisfaction rating")

        if support_tickets >= 4: risk_factors.append("High support ticket volume")
        elif support_tickets <= 1: protective_factors.append("Minimal support ticket history")

        if payment_failures >= 2: risk_factors.append("Multiple billing/payment failures")
        elif payment_failures == 0: protective_factors.append("Clean payment record")

        if competitor_usage == 1: risk_factors.append("Active competitor product usage detected")
        else: protective_factors.append("No competitor usage reported")

        if monthly_usage_hours < 20: risk_factors.append("Sub-optimal monthly usage hours")
        elif monthly_usage_hours >= 40: protective_factors.append("High product engagement hours")

        if tenure_months < 12: risk_factors.append("Early-stage account tenure (< 1 year)")
        elif tenure_months >= 24: protective_factors.append("Mature account tenure (> 2 years)")

        driver_col1, driver_col2 = st.columns(2)

        with driver_col1:
            st.markdown('<div class="driver-card-risk">', unsafe_allow_html=True)
            st.markdown("##### 🚨 Negative Drivers (Risk Factors)")
            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"• **{rf}**")
            else:
                st.caption("No prominent individual risk drivers detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        with driver_col2:
            st.markdown('<div class="driver-card-protective">', unsafe_allow_html=True)
            st.markdown("##### 🛡️ Positive Drivers (Protective Factors)")
            if protective_factors:
                for pf in protective_factors:
                    st.markdown(f"• **{pf}**")
            else:
                st.caption("No prominent protective drivers detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Strategy & Diagnostics
        st.markdown("<br>", unsafe_allow_html=True)
        
        diag_tab1, diag_tab2 = st.tabs(["💡 Recommended Retention Strategy", "🤖 Model Ensemble Diagnostics"])

        with diag_tab1:
            recommendations = []
            if satisfaction_score <= 2:
                recommendations.append("Assign customer success representative for immediate service recovery.")
            if support_tickets >= 4:
                recommendations.append("Escalate open technical tickets to engineering lead.")
            if payment_failures >= 2:
                recommendations.append("Contact billing contact to resolve payment method issues.")
            if competitor_usage == 1:
                recommendations.append("Deploy competitive feature matrix & targeted promotional incentive.")
            if monthly_usage_hours < 20:
                recommendations.append("Offer product training and feature discovery walkthrough.")
            if not recommendations:
                recommendations.append("Account is healthy. Continue standard automated nurture workflows.")

            for rec in recommendations:
                st.write(f"👉 {rec}")

        with diag_tab2:
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.metric("Logistic Regression Probability", f"{logistic_probability * 100:.1f}%")
            with d_col2:
                st.metric("LightGBM Model Probability", f"{lightgbm_probability * 100:.1f}%")
            with d_col3:
                st.metric("Weighted Ensemble Output", f"{churn_probability * 100:.1f}%")


# ------------------------------------------------------------
# MODULE 2: EXECUTIVE POWER BI DASHBOARD
# ------------------------------------------------------------
elif selected_view == "📊 Executive Power BI Dashboard":
    st.markdown("### 📊 Business Intelligence Command Center")
    st.caption("Interactive analytics covering cohort churn, usage distribution, and revenue risk across customer segments.")

    st.markdown('<div class="powerbi-wrapper">', unsafe_allow_html=True)
    components.html(
        """
        <iframe
            title="customer_retention_intelligence"
            width="100%"
            height="800"
            src="https://app.powerbi.com/reportEmbed?reportId=44a22155-21e6-44d1-b516-69872cd21477&autoAuth=true&ctid=420ec589-a866-4ad0-9a57-e6049e0d3bc0"
            frameborder="0"
            allowFullScreen="true">
        </iframe>
        """,
        height=810
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# MODULE 3: CHURN ANALYTICS DASHBOARD
# ------------------------------------------------------------
elif selected_view == "📊 Churn Analytics Dashboard":
    st.markdown("### 📊 Churn Analytics Dashboard")
    st.caption("Explore customer churn patterns and identify key characteristics associated with retention vs. churn.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Top KPI Metrics Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">5,000</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Churned Customers</div>
            <div class="kpi-value" style="color: #ef4444;">1,896</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Retained Customers</div>
            <div class="kpi-value" style="color: #10b981;">3,104</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Overall Churn Rate</div>
            <div class="kpi-value">37.9%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 1. CUSTOMER CHURN OVERVIEW
    st.markdown("### 📌 Customer Churn Overview")
    
    col_ov_chart, col_ov_table = st.columns([1.2, 1])
    
    with col_ov_chart:
        df_overview = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [3104, 1896]
        })
        fig_overview = px.bar(
            df_overview,
            x="Status",
            y="Customers",
            title="Customer Retention vs Churn",
            color="Status",
            color_discrete_map={"Retained": "#1f77b4", "Churned": "#1f77b4"},
            text_auto=True
        )
        fig_overview.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title="Number of Customers"
        )
        st.plotly_chart(fig_overview, use_container_width=True)

    with col_ov_table:
        df_ov_table = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [3104, 1896],
            "Percentage": [62.08, 37.92]
        })
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_ov_table, use_container_width=True, hide_index=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 2. CHURN BY SUBSCRIPTION TYPE
    st.markdown("### 💳 Churn by Subscription Type")
    
    col_sub_chart, col_sub_table = st.columns([1.2, 1])
    
    with col_sub_chart:
        df_sub = pd.DataFrame({
            "subscription_type": ["Basic", "Premium", "Standard"],
            "churn_rate": [36.07, 42.48, 37.63]
        })
        fig_sub = px.bar(
            df_sub,
            x="subscription_type",
            y="churn_rate",
            title="Churn Rate by Subscription Type",
            text_auto=".2f",
            labels={"subscription_type": "Subscription Type", "churn_rate": "Churn Rate (%)"}
        )
        fig_sub.update_traces(marker_color="#1f77b4")
        fig_sub.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sub, use_container_width=True)

    with col_sub_table:
        df_sub_table = pd.DataFrame({
            "subscription_type": ["Basic", "Premium", "Standard"],
            "customers": [2276, 1031, 1693],
            "churned": [821, 438, 637],
            "churn_rate": [36.07, 42.48, 37.63]
        })
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_sub_table, use_container_width=True, hide_index=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 3. COMPETITOR USAGE AND CHURN
    st.markdown("### 🏆 Competitor Usage and Churn")
    
    col_comp_chart, col_comp_table = st.columns([1.2, 1])
    
    with col_comp_chart:
        df_comp = pd.DataFrame({
            "competitor_usage": ["No", "Yes"],
            "churn_rate": [31.1577, 57.6351]
        })
        fig_comp = px.bar(
            df_comp,
            x="competitor_usage",
            y="churn_rate",
            title="Churn Rate by Competitor Usage",
            text_auto=".2f",
            labels={"competitor_usage": "Competitor Usage", "churn_rate": "Churn Rate (%)"}
        )
        fig_comp.update_traces(marker_color="#1f77b4")
        fig_comp.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_comp_table:
        df_comp_table = pd.DataFrame({
            "competitor_usage": ["No", "Yes"],
            "customers": [3723, 1277],
            "churned": [1160, 736],
            "churn_rate": [31.1577, 57.6351]
        })
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_comp_table, use_container_width=True, hide_index=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 4. SUPPORT TICKETS AND CHURN & SATISFACTION SCORE
    col_tix, col_sat = st.columns(2)

    with col_tix:
        st.markdown("### 🎫 Support Tickets and Churn")
        df_tix = pd.DataFrame({
            "support_tickets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "churn_rate": [15.2, 22.8, 36.1, 60.2, 75.5, 86.1, 94.8, 100.0, 100.0, 100.0]
        })
        fig_tix = px.line(
            df_tix,
            x="support_tickets",
            y="churn_rate",
            markers=True,
            title="Churn Rate by Number of Support Tickets",
            labels={"support_tickets": "Support Tickets", "churn_rate": "Churn Rate (%)"}
        )
        fig_tix.update_traces(line_color="#1f77b4", marker_size=8)
        fig_tix.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_tix, use_container_width=True)

    with col_sat:
        st.markdown("### ⭐ Churn by Satisfaction Score")
        df_sat = pd.DataFrame({
            "satisfaction_score": [1, 2, 3, 4, 5],
            "churn_rate": [93.8, 82.5, 49.6, 20.1, 6.2]
        })
        fig_sat = px.line(
            df_sat,
            x="satisfaction_score",
            y="churn_rate",
            markers=True,
            title="Churn Rate by Satisfaction Score",
            labels={"satisfaction_score": "Satisfaction Score", "churn_rate": "Churn Rate (%)"}
        )
        fig_sat.update_traces(line_color="#1f77b4", marker_size=8)
        fig_sat.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sat, use_container_width=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 5. KEY FINDINGS SECTION
    st.markdown("### 🔎 Key Findings")
    st.caption("Customer churn patterns identified from the analysis:")

    st.markdown("""
    <div class="finding-card">
        <strong>⭐ Satisfaction Score</strong><br>
        Strongest negative correlation (-0.71). Higher satisfaction dramatically decreases customer churn likelihood.
    </div>
    <div class="finding-card">
        <strong>🎫 Support Tickets & Payment Failures</strong><br>
        Strong positive correlations (+0.64 & +0.41). Service friction and billing issues directly drive churn risk.
    </div>
    <div class="finding-card">
        <strong>🏆 Competitor Usage</strong><br>
        Strong positive correlation (+0.52). Active competitor product usage significantly increases churn probability.
    </div>
    <div class="finding-card">
        <strong>⏱️ Usage & Tenure</strong><br>
        Moderate negative correlations (-0.38 & -0.45). Higher product engagement and longer account tenure act as retention shields.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌡️ Interactive Correlation Matrix")
    st.caption("Hover over cells to inspect exact correlation values across customer behavior metrics.")

    # Matrix Definition
    features = [
        "Churn", 
        "Support Tickets", 
        "Competitor Usage", 
        "Payment Failures", 
        "Monthly Fee", 
        "Age", 
        "Discount", 
        "Monthly Usage", 
        "Tenure", 
        "Satisfaction Score"
    ]
    
    corr_values = np.array([
        [ 1.00,  0.64,  0.52,  0.41,  0.12,  0.05, -0.08, -0.38, -0.45, -0.71],
        [ 0.64,  1.00,  0.31,  0.28,  0.08,  0.02, -0.04, -0.22, -0.29, -0.58],
        [ 0.52,  0.31,  1.00,  0.19,  0.06,  0.01, -0.03, -0.18, -0.21, -0.42],
        [ 0.41,  0.28,  0.19,  1.00,  0.04,  0.03, -0.01, -0.15, -0.19, -0.35],
        [ 0.12,  0.08,  0.06,  0.04,  1.00,  0.01,  0.05, -0.02,  0.01, -0.09],
        [ 0.05,  0.02,  0.01,  0.03,  0.01,  1.00,  0.02,  0.03,  0.08,  0.01],
        [-0.08, -0.04, -0.03, -0.01,  0.05,  0.02,  1.00,  0.06,  0.11,  0.07],
        [-0.38, -0.22, -0.18, -0.15, -0.02,  0.03,  0.06,  1.00,  0.34,  0.29],
        [-0.45, -0.29, -0.21, -0.19,  0.01,  0.08,  0.11,  0.34,  1.00,  0.38],
        [-0.71, -0.58, -0.42, -0.35, -0.09,  0.01,  0.07,  0.29,  0.38,  1.00]
    ])

    # Interactive Heatmap Plot
    fig_corr = px.imshow(
        corr_values,
        x=features,
        y=features,
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        text_auto=".2f",
        aspect="auto",
        labels=dict(x="Feature", y="Feature", color="Correlation")
    )

    fig_corr.update_traces(
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation Coefficient: <b>%{z:.2f}</b><extra></extra>"
    )

    fig_corr.update_layout(
        height=580,
        margin=dict(l=40, r=40, t=30, b=40),
        coloraxis_colorbar=dict(
            title="Correlation",
            ticks="outside",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0 (Strong Retention)", "-0.5", "0.0 (Neutral)", "0.5", "+1.0 (Strong Risk)"]
        )
    )

    st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption(
    "Customer Retention Intelligence | "
    "Churn Analytics Dashboard | "
    "Production Ensemble Engine (Logistic + LightGBM)"
)