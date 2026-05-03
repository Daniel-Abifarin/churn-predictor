import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Load model ──────────────────────────────────────────────────────────────
model = joblib.load("model.pkl")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="centered")
st.title("📡 Telco Customer Churn Predictor")
st.markdown("Enter customer details to predict whether they will churn.")

# ── Input form ──────────────────────────────────────────────────────────────
st.subheader("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.subheader("🔧 Services")
col3, col4 = st.columns(2)

with col3:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col4:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# ── Predict button ───────────────────────────────────────────────────────────
if st.button("🔮 Predict Churn", use_container_width=True):

    # ── Encode exactly as notebook did ──────────────────────────────────────

    # Binary maps
    def binary(val):
        return 1 if val in ["Yes", "Male"] else 0

    def service_map(val):
        return 1 if val == "Yes" else 0

    # Original encoded features
    SeniorCitizen       = 1 if senior_citizen == "Yes" else 0
    Partner             = binary(partner)
    Dependents          = binary(dependents)
    MultipleLines       = 1 if multiple_lines == "Yes" else 0
    PaperlessBilling    = binary(paperless_billing)
    OnlineSecurity_v    = service_map(online_security)
    OnlineBackup_v      = service_map(online_backup)
    DeviceProtection_v  = service_map(device_protection)
    TechSupport_v       = service_map(tech_support)
    StreamingTV_v       = service_map(streaming_tv)
    StreamingMovies_v   = service_map(streaming_movies)
    gender_v            = 1 if gender == "Male" else 0
    PhoneService_v      = binary(phone_service)

    # One-hot encoded columns (drop_first=True)
    InternetService_Fiber = 1 if internet_service == "Fiber optic" else 0
    InternetService_No    = 1 if internet_service == "No" else 0

    Contract_OneYear  = 1 if contract == "One year" else 0
    Contract_TwoYear  = 1 if contract == "Two year" else 0

    Payment_CreditCard   = 1 if payment_method == "Credit card (automatic)" else 0
    Payment_Electronic   = 1 if payment_method == "Electronic check" else 0
    Payment_Mailed       = 1 if payment_method == "Mailed check" else 0

    # ── Engineered features (exactly as notebook) ────────────────────────────
    ChargesPerTenure       = monthly_charges / (tenure + 1)
    ChargesDifference      = total_charges - (monthly_charges * tenure)
    TotalStreamingServices = StreamingTV_v + StreamingMovies_v
    TotalValueServices     = OnlineSecurity_v + OnlineBackup_v + DeviceProtection_v + TechSupport_v
    TotalServices          = PhoneService_v + MultipleLines + TotalStreamingServices + TotalValueServices
    IsNewCustomer          = 1 if tenure < 6 else 0
    IsLoyalCustomer        = 1 if tenure > 24 else 0

    # ── Assemble feature vector in EXACT column order ────────────────────────
    # Order matches: df.drop(columns=['Churn']).columns after all processing
    input_data = pd.DataFrame([{
        'SeniorCitizen':                          SeniorCitizen,
        'Partner':                                Partner,
        'Dependents':                             Dependents,
        'tenure':                                 tenure,
        'MultipleLines':                          MultipleLines,
        'PaperlessBilling':                       PaperlessBilling,
        'MonthlyCharges':                         monthly_charges,
        'TotalCharges':                           total_charges,
        'OnlineSecurity':                         OnlineSecurity_v,
        'OnlineBackup':                           OnlineBackup_v,
        'DeviceProtection':                       DeviceProtection_v,
        'TechSupport':                            TechSupport_v,
        'StreamingTV':                            StreamingTV_v,
        'StreamingMovies':                        StreamingMovies_v,
        'gender':                                 gender_v,
        'PhoneService':                           PhoneService_v,
        'InternetService_Fiber optic':            InternetService_Fiber,
        'InternetService_No':                     InternetService_No,
        'Contract_One year':                      Contract_OneYear,
        'Contract_Two year':                      Contract_TwoYear,
        'PaymentMethod_Credit card (automatic)':  Payment_CreditCard,
        'PaymentMethod_Electronic check':         Payment_Electronic,
        'PaymentMethod_Mailed check':             Payment_Mailed,
        'ChargesPerTenure':                       ChargesPerTenure,
        'ChargesDifference':                      ChargesDifference,
        'TotalStreamingServices':                 TotalStreamingServices,
        'TotalValueServices':                     TotalValueServices,
        'TotalServices':                          TotalServices,
        'IsNewCustomer':                          IsNewCustomer,
        'IsLoyalCustomer':                        IsLoyalCustomer,
    }])

    # ── Predict ──────────────────────────────────────────────────────────────
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk** — {probability:.1%} probability of churning")
        st.markdown("**Recommendation:** Offer a contract upgrade or loyalty discount immediately.")
    else:
        st.success(f"✅ **Low Churn Risk** — {1 - probability:.1%} probability of staying")
        st.markdown("**Recommendation:** Customer is stable. Monitor monthly charges.")

    # Show breakdown
    with st.expander("See prediction details"):
        st.write(f"Churn probability: **{probability:.4f}**")
        st.write(f"Retention probability: **{1 - probability:.4f}**")
        st.write(f"Engineered features used:")
        st.write(f"- ChargesPerTenure: {ChargesPerTenure:.2f}")
        st.write(f"- TotalServices: {TotalServices}")
        st.write(f"- IsNewCustomer: {IsNewCustomer} | IsLoyalCustomer: {IsLoyalCustomer}")
