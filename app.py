import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load Saved Model
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# App Title
st.title("Customer Churn Prediction")

st.write(
    "Enter customer information to predict if they may leave the company."
)

st.divider()

# Customer Form
with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=50.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=600.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# Make Prediction
if submitted:

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Encode Customer Data
    customer_encoded = pd.get_dummies(
        customer_data,
        dtype=int
    )

    # Match Training Columns
    customer_encoded = customer_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(customer_encoded)[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    # Display Result
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Likely to Leave")
    else:
        st.success("✅ Likely to Stay")

    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}"
    )

    st.caption(
        "This is a model prediction, not a guarantee of customer behavior."
    )
