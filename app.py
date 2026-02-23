import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS for Black & White Theme
# -----------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #333;
    }
    
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        color: #000000;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.2);
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
    }
    
    .stNumberInput input, .stSelectbox select {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #666 !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1) !important;
    }
    
    label {
        color: #cccccc !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    .result-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        border: 1px solid #333;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .probability-display {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .risk-label {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .info-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    
    hr {
        border: none;
        border-top: 1px solid #333;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title Section
# -----------------------------
st.markdown('<h1 class="main-title">📞 Telecom Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered customer retention analysis</p>', unsafe_allow_html=True)

# -----------------------------
# Load Model + Features
# -----------------------------
try:
    model = joblib.load("churn_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    st.error("❌ Model loading failed!")
    st.write(e)
    st.stop()

# -----------------------------
# Inputs
# -----------------------------
st.markdown('<div class="section-header">📊 Customer Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Complaint Details**")
    total_complaints = st.number_input("Total Complaints", 0, 20, 1, help="Number of complaints filed")
    has_complaint = st.selectbox("Has Active Complaint?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    complaint_recency_days = st.number_input("Days Since Last Complaint", 0, 1000, 999)

with col2:
    st.markdown("**Usage Patterns**")
    data_used_gb = st.number_input("Data Used (GB)", 0.0, 100.0, 5.0, help="Monthly data usage")
    calls_made = st.number_input("Calls Made", 0, 500, 30, help="Number of calls per month")
    revenue_inr = st.number_input("Revenue (₹)", 0, 10000, 200, help="Monthly revenue generated")

with col3:
    st.markdown("**Account Details**")
    tenure = st.number_input("Tenure (Months)", 0, 100, 12, help="Customer tenure in months")
    monthly_charges = st.number_input("Monthly Charges (₹)", 0, 10000, 200)
    region = st.selectbox("Region", ["Chandigarh", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Mumbai", "Pune"])
    contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Churn Probability"):

    # Create base input
    input_data = {
        "total_complaints": total_complaints,
        "has_complaint": has_complaint,
        "complaint_recency_days": complaint_recency_days,
        "data_used_gb": data_used_gb,
        "calls_made": calls_made,
        "revenue_inr": revenue_inr,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
    }

    # Convert into DataFrame
    input_df = pd.DataFrame([input_data])

    # Manually add region dummy columns
    for r in ["Chandigarh", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Mumbai", "Pune"]:
        col = f"region_{r}"
        input_df[col] = 1 if region == r else 0

    # Manually add contract dummy columns
    for c in ["One Year", "Two Year"]:
        col = f"contract_type_{c}"
        input_df[col] = 1 if contract_type == c else 0

    # Add missing columns
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Correct column order
    input_df = input_df[feature_names]

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]

    # Display Results with Aesthetic Styling
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    # Determine risk level and styling
    if prob > 0.7:
        risk_emoji = "🔴"
        risk_label = "High Risk"
        risk_color = "#ff4444"
        risk_description = "Immediate intervention required"
    elif prob > 0.4:
        risk_emoji = "🟡"
        risk_label = "Medium Risk"
        risk_color = "#ffaa00"
        risk_description = "Monitor closely and engage"
    else:
        risk_emoji = "🟢"
        risk_label = "Low Risk"
        risk_color = "#44ff44"
        risk_description = "Customer likely to stay"
    
    # Display probability
    st.markdown(
        f'<div class="probability-display" style="color: {risk_color};">{prob:.1%}</div>',
        unsafe_allow_html=True
    )
    
    # Display risk label
    st.markdown(
        f'<div class="risk-label" style="color: {risk_color};">{risk_emoji} {risk_label}</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        f'<p style="text-align: center; color: #888; margin-top: 0.5rem; font-size: 1rem;">{risk_description}</p>',
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Insights
    st.markdown('<div class="section-header">💡 Customer Insights</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color: #ffffff; margin-bottom: 0.5rem;">Engagement</h4>
            <p style="color: #888; margin: 0;">Data: {data_used_gb} GB</p>
            <p style="color: #888; margin: 0;">Calls: {calls_made}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color: #ffffff; margin-bottom: 0.5rem;">Revenue</h4>
            <p style="color: #888; margin: 0;">Monthly: ₹{monthly_charges}</p>
            <p style="color: #888; margin: 0;">Total: ₹{revenue_inr}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color: #ffffff; margin-bottom: 0.5rem;">Loyalty</h4>
            <p style="color: #888; margin: 0;">Tenure: {tenure} months</p>
            <p style="color: #888; margin: 0;">Contract: {contract_type}</p>
        </div>
        """, unsafe_allow_html=True)
