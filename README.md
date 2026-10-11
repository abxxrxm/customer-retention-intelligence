# Customer Retention Intelligence

I built this project to explore a practical question: how can customer data help a retention team decide which accounts need attention, and why?

The project starts with data validation and exploratory analysis, moves into statistical testing and predictive modeling, and ends with a Streamlit app that turns a customer profile into a churn-risk score with suggested retention actions.

The dataset contains 5,000 synthetic customer records. It does not contain real customer or company data, so the findings and model results should be read as project demonstrations rather than production outcomes.

## What I built

I used Python to validate and analyze the data, test relationships between customer behavior and churn, and train a logistic-regression model. I also saved logistic regression and LightGBM pipelines and combined their probabilities in the Streamlit app using a 70/30 weighting.

The app has three main views:

- an individual churn-risk calculator
- a customer analytics dashboard
- an embedded Power BI report view

For each customer profile, the app calculates a churn-risk score, compares it with a 40% action threshold, and shows rule-based risk drivers, protective factors, and possible retention actions.

## Dataset snapshot

| Measure | Value |
|---|---:|
| Synthetic customer records | 5,000 |
| Input columns | 13 |
| Unique customer IDs | 5,000 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Churned customers | 1,896 |
| Retained customers | 3,104 |
| Overall churn rate | 37.92% |

The features include age, gender, subscription type, monthly fee, tenure, monthly usage, support tickets, payment failures, discount, satisfaction score, and competitor usage.

## What I found

A few patterns stood out in the synthetic data:

- Customers with reported competitor usage had a **57.64%** churn rate, compared with **31.16%** among customers without reported competitor usage.
- The relationship between competitor usage and churn was statistically significant: chi-square = **282.04**, p = **2.69e-63**, phi = **0.238**.
- Churn fell as monthly usage increased, from **50.58%** in the 0-10 hour group to **29.24%** in the 50-60 hour group.
- Premium customers had the highest observed churn rate at **42.48%**, compared with **37.63%** for Standard and **36.07%** for Basic.

These results show associations in synthetic data. They do not prove that any one factor causes churn.

## Model result

The logistic-regression model reaches **79.7% accuracy** (precision 0.767, recall 0.668, ROC-AUC 0.866; see the table below). The app combines LightGBM and logistic regression as an ensemble. TODO: add ensemble evaluation once it is measured on a held-out test set.

The executed logistic-regression notebook on its own reports the following performance on its held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.797 |
| Precision | 0.767 |
| Recall | 0.668 |
| F1 score | 0.714 |
| ROC-AUC | 0.866 |

The Streamlit app combines probabilities from the saved logistic-regression and LightGBM pipelines using weights of 0.70 and 0.30. It flags a profile when the combined score reaches the 0.40 action threshold.

The table above belongs to the logistic-regression notebook only. TODO: add a notebook or script evaluating the ensemble, so every number in this README can be traced to code in the repo.

## How the workflow fits together

1. **Validate the data** - check the schema, missing values, duplicate rows, customer IDs, and value ranges.
2. **Explore churn patterns** - compare churn across subscription, usage, tenure, satisfaction, support, payment, discount, and competitor groups.
3. **Test relationships** - use chi-square tests and Welch's t-tests to examine associations and group differences.
4. **Train and evaluate** - preprocess the features, train logistic regression, and evaluate it on a held-out test set.
5. **Turn the result into a tool** - combine saved model scores in Streamlit and add clear risk factors and suggested actions.

## Project structure

```text
customer-retention-intelligence/
├── app/
│   ├── app.py
│   └── pages/1_Analytics_Dashboard.py
├── data/customer_churn_dataset.csv
├── models/
│   ├── logistic_model.pkl
│   ├── lightgbm_model.pkl
│   └── ensemble_config.pkl
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_statistical_analysis.ipynb
│   └── 03_logistic_regression.ipynb
├── sql/
│   └── churn_analysis.sql
├── src/
│   ├── analytics.py
│   └── data_validation.py
├── README.md
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Limits

- The dataset is synthetic, so the patterns are demonstrations rather than production evidence.
- Model metrics come from one project split and do not establish real-world performance.
- The app's driver labels and retention suggestions are rule-based explanations, not causal recommendations.
- TODO: LightGBM training code and a separate ensemble evaluation are not yet in the repo.
- The Power BI iframe uses tenant authentication and may not be visible to public viewers. The Streamlit analytics dashboard remains available locally.

## License

MIT License. Copyright (c) 2026 Abhiram Suri.
