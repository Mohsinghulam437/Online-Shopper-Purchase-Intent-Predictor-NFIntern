"""
Online Shopper Purchase Intent Predictor - Streamlit App
Neuro Five Solutions Internship - Task 11 (Capstone Project)

Loads shopper_pipeline.joblib (XGBoost inside a full sklearn Pipeline,
picked as the best of 3 models trained on the UCI Online Shoppers
Purchasing Intention dataset) and serves it as a live conversion-intent
predictor - the kind of tool a Shopify store owner could use to sanity-check
whether a given browsing pattern looks like it's heading toward a sale.

Design note: the raw dataset has 17 columns, including 4 anonymized
categorical IDs (OperatingSystems, Browser, Region, TrafficType) that carry
little real business meaning and would confuse a non-technical user filling
out a form. Those are set to their most common (mode) values behind the
scenes rather than exposed as inputs - the form only asks for signals a
store owner would actually recognize and care about.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Purchase Intent Predictor", page_icon="🛒", layout="centered")

# Silent defaults for the low-signal anonymized categorical columns
# (most common value in the training data for each)
DEFAULTS = {"OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2}


@st.cache_resource
def load_pipeline():
    return joblib.load("shopper_pipeline.joblib")


pipeline = load_pipeline()

st.title("🛒 Online Shopper Purchase Intent Predictor")
st.write(
    "Predicts whether a browsing session is likely to end in a purchase, "
    "based on real on-site behavior signals. Trained on 12,330 real "
    "e-commerce sessions (XGBoost, ROC-AUC 0.92, F1 0.655 on the "
    "minority 'purchase' class)."
)

st.divider()
st.subheader("Session behavior")

col1, col2 = st.columns(2)

with col1:
    product_related = st.number_input("Product pages viewed", min_value=0, max_value=200, value=10, step=1)
    product_duration = st.number_input("Time on product pages (seconds)", min_value=0.0, max_value=10000.0, value=300.0, step=10.0)
    page_values = st.number_input(
        "Page Values (Google Analytics metric - avg. $ value of pages viewed before this event)",
        min_value=0.0, max_value=400.0, value=5.0, step=1.0,
        help="A GA metric estimating the average monetary value of a page a visitor viewed on the way to a goal/purchase. The single strongest predictor in this model.",
    )
    bounce_rate = st.slider("Bounce rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    exit_rate = st.slider("Exit rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

with col2:
    admin_pages = st.number_input("Administrative pages viewed", min_value=0, max_value=50, value=0, step=1)
    admin_duration = st.number_input("Time on administrative pages (seconds)", min_value=0.0, max_value=3000.0, value=0.0, step=10.0)
    info_pages = st.number_input("Informational pages viewed", min_value=0, max_value=50, value=0, step=1)
    info_duration = st.number_input("Time on informational pages (seconds)", min_value=0.0, max_value=3000.0, value=0.0, step=10.0)
    special_day = st.slider(
        "Proximity to a special day (0=far, 1=very close e.g. Valentine's/Mother's Day)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.1,
    )

st.divider()
st.subheader("Visitor context")

col3, col4, col5 = st.columns(3)
with col3:
    month = st.selectbox(
        "Month", options=["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8
    )
with col4:
    visitor_type = st.selectbox("Visitor type", options=["Returning_Visitor", "New_Visitor", "Other"])
with col5:
    weekend = st.selectbox("Weekend session?", options=["No", "Yes"]) == "Yes"

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    total_pages = admin_pages + info_pages + product_related
    total_duration = admin_duration + info_duration + product_duration
    avg_time_per_page = total_duration / total_pages if total_pages > 0 else 0.0

    input_df = pd.DataFrame([{
        "Administrative": admin_pages,
        "Administrative_Duration": admin_duration,
        "Informational": info_pages,
        "Informational_Duration": info_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "TotalPages": total_pages,
        "TotalDuration": total_duration,
        "AvgTimePerPage": avg_time_per_page,
        "Month": month,
        "VisitorType": visitor_type,
        "OperatingSystems": DEFAULTS["OperatingSystems"],
        "Browser": DEFAULTS["Browser"],
        "Region": DEFAULTS["Region"],
        "TrafficType": DEFAULTS["TrafficType"],
        "Weekend": weekend,
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("### ✅ Predicted: Likely to purchase")
    else:
        st.error("### ❌ Predicted: Unlikely to purchase")
    st.metric("Purchase probability", f"{probability:.1%}")
    st.progress(min(float(probability), 1.0))

    with st.expander("See full input sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Model: XGBoost inside a scikit-learn Pipeline (StandardScaler + OneHotEncoder), "
    "selected as the best of 3 models trained (Logistic Regression, Random Forest, XGBoost) "
    "on class-weighted training data to handle the dataset's ~85/15 imbalance. "
    "See README.md and CASE_STUDY.md for the full write-up."
)
