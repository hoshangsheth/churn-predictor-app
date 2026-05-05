import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="◈",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080C10 !important;
    color: #E8EDF2 !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,200,150,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 90% 90%, rgba(0,140,255,0.08) 0%, transparent 60%),
        #080C10 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 780px !important; padding: 3rem 2rem 5rem !important; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 1px;
    height: 60px;
    background: linear-gradient(to bottom, transparent, #00C896);
}
.app-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00C896;
    border: 1px solid rgba(0,200,150,0.3);
    padding: 0.3rem 1rem;
    border-radius: 2px;
    margin-bottom: 1.2rem;
}
.app-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.2rem, 6vw, 3.6rem) !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em !important;
    color: #F0F4F8 !important;
    margin-bottom: 0.8rem !important;
}
.app-title span { color: #00C896; }
.app-subtitle {
    font-size: 0.8rem;
    color: rgba(200,215,225,0.5);
    letter-spacing: 0.05em;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #00C896;
    margin: 2.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0,200,150,0.2);
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 1.8rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease;
}
.card:hover { border-color: rgba(0,200,150,0.2); }

/* ── Streamlit Widgets ── */
label, .stSelectbox label, .stSlider label,
.stNumberInput label, [data-testid="stWidgetLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(200,215,225,0.6) !important;
    margin-bottom: 0.3rem !important;
}

/* Inputs */
input[type="number"],
.stTextInput input,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 3px !important;
    color: #E8EDF2 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s ease !important;
}
input[type="number"]:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #00C896 !important;
    box-shadow: 0 0 0 2px rgba(0,200,150,0.1) !important;
    outline: none !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 3px !important;
    color: #E8EDF2 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(0,200,150,0.4) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: #00C896 !important;
}
[data-testid="stSlider"] .stSlider > div {
    color: #E8EDF2 !important;
}

/* ── Predict Button ── */
div.stButton { margin-top: 2rem; }
div.stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid #00C896 !important;
    color: #00C896 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 1rem 2rem !important;
    border-radius: 3px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
div.stButton > button:hover {
    background: rgba(0,200,150,0.08) !important;
    box-shadow: 0 0 30px rgba(0,200,150,0.15) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result Box ── */
.result-box {
    margin-top: 2rem;
    padding: 2rem;
    border-radius: 4px;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease forwards;
}
.result-box.churn {
    background: rgba(255, 70, 70, 0.06);
    border: 1px solid rgba(255, 70, 70, 0.3);
}
.result-box.safe {
    background: rgba(0, 200, 150, 0.06);
    border: 1px solid rgba(0, 200, 150, 0.3);
}
.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
}
.result-box.churn::before { background: #FF4646; }
.result-box.safe::before  { background: #00C896; }

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.6;
}
.result-label.churn { color: #FF4646; }
.result-label.safe  { color: #00C896; }

.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}
.result-verdict.churn { color: #FF4646; }
.result-verdict.safe  { color: #00C896; }

.result-prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-prob-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(200,215,225,0.4);
    white-space: nowrap;
}
.result-prob-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
}
.result-prob-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}
.result-prob-bar-fill.churn { background: #FF4646; }
.result-prob-bar-fill.safe  { background: #00C896; }
.result-prob-value {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    white-space: nowrap;
}
.result-prob-value.churn { color: #FF4646; }
.result-prob-value.safe  { color: #00C896; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: rgba(200,215,225,0.25);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('artifacts/onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('artifacts/label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return ohe_geo, le_gender, scaler

ohe_geo, le_gender, scaler = load_artifacts()
model = load_model('artifacts/model.h5')

# ── Predict function ────────────────────────────────────────────────────────────
def predict_data(data, ohe_columns=None, label_columns=None):
    if isinstance(data, dict):
        data = [data]
    elif isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='records')

    results = []
    for d in data:
        df = pd.DataFrame([d])

        if ohe_columns:
            for col, encoder in ohe_columns.items():
                if col in df.columns:
                    encoded = encoder.transform([[df[col].values[0]]]).toarray()
                    feature_names = encoder.get_feature_names_out([col])
                    df[feature_names] = encoded
                    df = df.drop(col, axis=1)

        if label_columns:
            for col, encoder in label_columns.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col])

        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        probability = float(prediction[0][0])
        result = 'Likely to churn' if probability > 0.5 else 'Not likely to churn'
        results.append({'probability': round(probability, 4), 'result': result})

    return results


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-badge">◈ Neural Prediction Engine</div>
    <div class="app-title">Customer<br><span>Churn</span> Predictor</div>
    <div class="app-subtitle">ML-powered retention intelligence</div>
</div>
""", unsafe_allow_html=True)


# ── Section 01 — Profile ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 — Customer Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
with col2:
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
with col3:
    gender = st.selectbox('Gender', ['Male', 'Female'])

col4, col5 = st.columns(2)
with col4:
    age = st.slider('Age', min_value=18, max_value=92, value=40)
with col5:
    tenure = st.slider('Tenure (Years)', min_value=0, max_value=10, value=3)


# ── Section 02 — Financials ────────────────────────────────────────────────────
st.markdown('<div class="section-label">02 — Financial Details</div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    balance = st.number_input('Account Balance', min_value=0, value=60000, step=1000)
with col7:
    estimated_salary = st.number_input('Estimated Salary', min_value=0, value=50000, step=1000)


# ── Section 03 — Account ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">03 — Account Status</div>', unsafe_allow_html=True)

col8, col9, col10 = st.columns(3)
with col8:
    num_of_products = st.selectbox('No. of Products', [1, 2, 3, 4])
with col9:
    has_cr_card = st.selectbox('Has Credit Card', [1, 0], format_func=lambda x: 'Yes' if x else 'No')
with col10:
    is_active = st.selectbox('Active Member', [1, 0], format_func=lambda x: 'Yes' if x else 'No')


# ── Predict ────────────────────────────────────────────────────────────────────
if st.button('◈  Run Prediction'):
    input_data = {
        'CreditScore'     : credit_score,
        'Geography'       : geography,
        'Gender'          : gender,
        'Age'             : age,
        'Tenure'          : tenure,
        'Balance'         : balance,
        'NumOfProducts'   : num_of_products,
        'HasCrCard'       : has_cr_card,
        'IsActiveMember'  : is_active,
        'EstimatedSalary' : estimated_salary
    }

    results = predict_data(
        input_data,
        ohe_columns={'Geography': ohe_geo},
        label_columns={'Gender': le_gender}
    )

    prob      = results[0]['probability']
    is_churn  = prob > 0.5
    css_class = 'churn' if is_churn else 'safe'
    verdict   = '⚠ High Churn Risk' if is_churn else '✓ Retention Likely'
    pct       = f"{prob:.1%}"
    bar_width = f"{prob * 100:.1f}%"

    st.markdown(f"""
    <div class="result-box {css_class}">
        <div class="result-label {css_class}">Prediction Result</div>
        <div class="result-verdict {css_class}">{verdict}</div>
        <div class="result-prob-row">
            <div class="result-prob-label">Churn probability</div>
            <div class="result-prob-bar-bg">
                <div class="result-prob-bar-fill {css_class}" style="width:{bar_width}"></div>
            </div>
            <div class="result-prob-value {css_class}">{pct}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">◈ Churn Predictor · Powered by Neural Network</div>', unsafe_allow_html=True)