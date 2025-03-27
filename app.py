import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier


# Load the trained CatBoost model
model = CatBoostClassifier()
model.load_model("/Users/petromelnyk/Desktop/projects/bank_proj/catboost_credit_risk.cbm")

# Configure page
st.set_page_config(page_title="Credit Risk Assessment", layout="wide")
st.title("🏦 Bank Client Credit Risk Assessment")

st.markdown("---")

# Mappings (English ↔ Polish)
credit_history_map = {
    "Good History": "dobra historia",
    "No History": "brak historii"
}

overdue_payments_map = {
    "No Delays": "brak opóźnień",
    "1 Late Payment": "opóźnienia",
    "2 Late Payments": "2",
    "3 Late Payments": "3",
    "4+ Late Payments": "4"
}

employment_type_map = {
    "None": "brak",
    "Permanent": "stała",
    "Self-employed": "samozatrudnienie",
    "Fixed-term": "określona"
}

owns_property_map = {
    "Yes": "tak",
    "No": "nie"
}

education_map = {
    "Secondary": "średnie",
    "Higher": "wyższe",
    "Primary": "podstawowe"
}

city_map = {
    "Small": "małe",
    "Medium": "średnie",
    "Large": "duże"
}

marital_status_map = {
    "Married": "żonaty/zamężna",
    "Single": "kawaler/panna",
    "Divorced": "rozwiedziony/rozwiedziona"
}

# ---- LAYOUT ----
st.markdown("## 🧑 Personal Information")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    age = st.number_input("📅 Age", min_value=18, max_value=100, value=30)
    children = st.selectbox("👶 Number of Children", ["0", "1", "2", "3", "4+"])
    marital_status = st.selectbox(
        "💍 Marital Status", list(marital_status_map.keys()))

with col2:
    income = st.number_input("💰 Income", min_value=0.0,
                             value=3000.0, step=100.0)
    education = st.selectbox("🎓 Education Level", list(education_map.keys()))
    city = st.selectbox("🌆 City Size", list(city_map.keys()))

with col3:
    employment_type = st.selectbox(
        "👔 Employment Type", list(employment_type_map.keys()))
    years_in_job = st.number_input("⌛ Years in Job", min_value=0, value=2)
    owns_property = st.selectbox(
        "🏡 Owns Property", list(owns_property_map.keys()))

st.markdown("---")

st.markdown("## 💳 Financial & Credit Information")

col4, col5 = st.columns([1, 1])

with col4:
    active_loans = st.number_input("🏦 Active Loans", min_value=0, value=0)
    other_loans = st.number_input("📑 Other Loans", min_value=0, value=0)
    assets_value = st.number_input(
        "🏠 Assets Value", min_value=0.0, value=0.0, step=1000.0)

with col5:
    credit_history = st.selectbox(
        "📝 Credit History", list(credit_history_map.keys()))
    overdue_payments = st.selectbox(
        "⚠️ Overdue Payments", list(overdue_payments_map.keys()))

st.markdown("---")

# Convert selected English labels to Polish before passing to model
input_data = pd.DataFrame([{
    "age": age,
    "income": income,
    "children": children,
    "credit_history": credit_history_map[credit_history],
    "overdue_payments": overdue_payments_map[overdue_payments],
    "active_loans": active_loans,
    "years_in_job": years_in_job,
    "employment_type": employment_type_map[employment_type],
    "owns_property": owns_property_map[owns_property],
    "assets_value": assets_value,
    "other_loans": other_loans,
    "education": education_map[education],
    "city": city_map[city],
    "marital_status": marital_status_map[marital_status]
}])

# Prediction Button
if st.button("📊 Assess Credit Risk"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(
        input_data)[0][1]  # Probability of "Risk"

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk Client (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk Client (Probability: {1 - probability:.2%})")

    # Display input data in **English**
    st.markdown("### 📋 Input Data")
    display_data = input_data.copy()

    # Convert Polish feature values to English for display
    display_data["credit_history"] = display_data["credit_history"].map(
        {v: k for k, v in credit_history_map.items()})
    display_data["overdue_payments"] = display_data["overdue_payments"].map(
        {v: k for k, v in overdue_payments_map.items()})
    display_data["employment_type"] = display_data["employment_type"].map(
        {v: k for k, v in employment_type_map.items()})
    display_data["owns_property"] = display_data["owns_property"].map(
        {v: k for k, v in owns_property_map.items()})
    display_data["education"] = display_data["education"].map(
        {v: k for k, v in education_map.items()})
    display_data["city"] = display_data["city"].map(
        {v: k for k, v in city_map.items()})
    display_data["marital_status"] = display_data["marital_status"].map(
        {v: k for k, v in marital_status_map.items()})

    st.dataframe(display_data)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Convert feature names back to English for SHAP display
    shap_data = input_data.copy()

    shap_data.rename(columns={
        "credit_history": "Credit History",
        "overdue_payments": "Overdue Payments",
        "employment_type": "Employment Type",
        "owns_property": "Owns Property",
        "education": "Education",
        "city": "City Size",
        "marital_status": "Marital Status",
        "age": "Age",
        "income": "Income",
        "children": "Number of Children",
        "active_loans": "Active Loans",
        "years_in_job": "Years in Job",
        "assets_value": "Assets Value",
        "other_loans": "Other Loans"
    }, inplace=True)

    # Convert Polish feature values to English for SHAP visualization
    shap_data["Credit History"] = shap_data["Credit History"].map(
        {v: k for k, v in credit_history_map.items()})
    shap_data["Overdue Payments"] = shap_data["Overdue Payments"].map(
        {v: k for k, v in overdue_payments_map.items()})
    shap_data["Employment Type"] = shap_data["Employment Type"].map(
        {v: k for k, v in employment_type_map.items()})
    shap_data["Owns Property"] = shap_data["Owns Property"].map(
        {v: k for k, v in owns_property_map.items()})
    shap_data["Education"] = shap_data["Education"].map(
        {v: k for k, v in education_map.items()})
    shap_data["City Size"] = shap_data["City Size"].map(
        {v: k for k, v in city_map.items()})
    shap_data["Marital Status"] = shap_data["Marital Status"].map(
        {v: k for k, v in marital_status_map.items()})

    st.markdown(
        """
        <style>
        div[data-testid="stImage"] img {
            width: 100% !important;  /* Adjust width */
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display SHAP explanation with English labels
    st.markdown("### 🌊 SHAP Waterfall Plot")

    fig, ax = plt.subplots(dpi=100)  # Adjust DPI to resize

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=shap_data.iloc[0]  # Use translated data
        ),
        max_display=10,
        show=False
    )

    st.pyplot(fig)
