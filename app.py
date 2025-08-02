import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Embed your full CSS directly here
css = """
/* Professional banking background with soft blue and white tones */
.stApp {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 25%, #90caf9 50%, #64b5f6 75%, #42a5f5 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

/* Main container styling */
.main .block-container {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.5);
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}

/* Remove only truly empty elements */
.main .block-container > div:empty {
    display: none;
}

/* Header styling */
.main-header {
    font-size: 4rem;
    background: linear-gradient(45deg, #1976d2, #42a5f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
}

.sub-header {
    font-size: 1.5rem;
    color: #37474f;
    text-align: center;
    margin-bottom: 3rem;
    font-weight: 300;
}

/* Form container styling */
.form-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    border: 1px solid rgba(255, 255, 255, 0.5);
}

.form-section {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(25, 118, 210, 0.2);
}

.section-title {
    font-size: 1.3rem;
    color: #1565c0;
    font-weight: bold;
    margin-bottom: 1rem;
    text-align: center;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Prediction box styling */
.prediction-box {
    background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
    padding: 2.5rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 15px 35px rgba(25, 118, 210, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Metric cards styling */
.metric-card {
    background: rgba(255, 255, 255, 0.98);
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
}

/* Remove only empty metric cards */
.metric-card:empty {
    display: none;
}

/* Improve metric text visibility */
.metric-card label,
.metric-card div[data-testid="metric-container"] {
    color: #2c3e50;
    font-weight: 600;
}

/* Form elements styling */
.stSelectbox > div > div {
    background-color: rgba(255, 255, 255, 0.98);
    border-radius: 10px;
    border: 1px solid rgba(25, 118, 210, 0.3);
}

/* Number input styling - more specific selectors */
.stNumberInput > div > div > input,
.stNumberInput input,
input[type="number"],
div[data-baseweb="input"] input {
    background-color: rgba(255, 255, 255, 0.98) !important;
    border-radius: 10px;
    border: none !important;
    color: #2c3e50 !important;
    font-weight: 500;
    outline: none !important;
}

/* Target Streamlit's specific number input containers */
.stNumberInput > div > div,
div[data-baseweb="input"] {
    background-color: rgba(255, 255, 255, 0.98) !important;
    border-radius: 10px;
    border: none !important;
}

/* Style the +/- buttons (spinner buttons) */
.stNumberInput button,
.stNumberInput button[data-baseweb="button"],
div[data-baseweb="input"] button {
    background-color: rgba(25, 118, 210, 0.1) !important;
    border: none !important;
    color: #1976d2 !important;
    border-radius: 4px !important;
}

/* Hover effect for +/- buttons */
.stNumberInput button:hover,
.stNumberInput button[data-baseweb="button"]:hover,
div[data-baseweb="input"] button:hover {
    background-color: rgba(25, 118, 210, 0.2) !important;
    color: #1565c0 !important;
}

/* Remove all focus outlines */
.stNumberInput > div > div > input:focus,
.stNumberInput input:focus,
input[type="number"]:focus,
div[data-baseweb="input"] input:focus {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
}

/* Improve form labels visibility */
.stSelectbox label,
.stNumberInput label {
    color: #2c3e50;
    font-weight: 600;
    font-size: 1rem;
}

/* Dropdown text color - simplified selectors */
.stSelectbox div[data-baseweb="select"] {
    color: #2c3e50;
    font-weight: 500;
}

.stSelectbox div[data-baseweb="select"] > div {
    color: #2c3e50;
    background-color: rgba(255, 255, 255, 0.98);
    font-weight: 500;
}

/* Dropdown options */
.stSelectbox div[data-baseweb="option"] {
    color: #2c3e50;
    background-color: rgba(255, 255, 255, 0.98);
    font-weight: 500;
}

/* Selected option */
.stSelectbox div[data-baseweb="option"][aria-selected="true"] {
    background-color: rgba(25, 118, 210, 0.15);
    color: #1565c0;
    font-weight: 600;
}

/* Hover effect */
.stSelectbox div[data-baseweb="option"]:hover {
    background-color: rgba(25, 118, 210, 0.1);
}

/* Number input text color */
.stNumberInput > div > div > input::placeholder {
    color: #9e9e9e;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(45deg, #1976d2, #42a5f5);
    border-radius: 25px;
    border: none;
    color: white;
    font-weight: bold;
    padding: 1rem 3rem;
    font-size: 1.2rem;
    box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
    transition: all 0.3s ease;
    width: 100%;
    max-width: 400px;
    margin: 0 auto;
    display: block;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4);
    background: linear-gradient(45deg, #1565c0, #1976d2);
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    border: 1px solid rgba(25, 118, 210, 0.2);
}

/* Footer styling */
.footer {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 1rem;
    margin-top: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Additional professional touches */
.stMarkdown {
    color: #2c3e50;
    font-weight: 500;
}

/* Ensure consistent spacing */
.main .block-container > div {
    margin-bottom: 1rem;
}

/* Improve all text visibility */
.main .block-container p,
.main .block-container div,
.main .block-container span {
    color: #2c3e50;
}

/* Smooth transitions */
* {
    transition: all 0.2s ease;
}

/* Hide sidebar */
.sidebar .sidebar-content {
    display: none;
}

/* Remove only truly empty divs */
div:empty {
    display: none;
}

/* Ensure consistent background colors */
.main .block-container,
.form-container,
.form-section,
.metric-card {
    background: rgba(255, 255, 255, 0.98);
}
"""

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# === Main header ===
st.markdown('<h1 class="main-header">🏦 Bank Marketing Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether a customer will subscribe to a term deposit</p>', unsafe_allow_html=True)

# === Form container ===
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    # Personal Information Section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">👤 Personal Details</h3>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=95, value=35, help="Customer's age (18-95)")
    with col2:
        job_options = ["admin.", "unknown", "unemployed", "management", "housemaid",
                       "entrepreneur", "student", "blue-collar", "self-employed",
                       "retired", "technician", "services"]
        job = st.selectbox("Job", job_options, help="Customer's job category")
    with col3:
        marital_options = ["married", "divorced", "single"]
        marital = st.selectbox("Marital Status", marital_options, help="Customer's marital status")

    st.markdown('</div>', unsafe_allow_html=True)

    # Education and Loan Section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">🎓 Education & Loan</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        education_options = ["unknown", "primary", "secondary", "tertiary"]
        education = st.selectbox("Education Level", education_options, help="Customer's education level")
    with col2:
        loan = st.selectbox("Has Personal Loan?", ["no", "yes"], help="Whether customer has a personal loan")

    st.markdown('</div>', unsafe_allow_html=True)

    # Financial Information Section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">💰 Financial Details</h3>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        balance = st.number_input("Balance (€)", min_value=-10000, max_value=100000, value=1000,
                                 help="Average yearly balance in euros")
    with col2:
        default = st.selectbox("Has Credit in Default?", ["no", "yes"], help="Whether customer has credit in default")
    with col3:
        housing = st.selectbox("Has Housing Loan?", ["no", "yes"], help="Whether customer has housing loan")

    st.markdown('</div>', unsafe_allow_html=True)

    # Contact Information Section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📞 Contact Details</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        contact_options = ["unknown", "telephone", "cellular"]
        contact = st.selectbox("Contact Type", contact_options, help="Communication type used")
    with col2:
        month_options = ["jan", "feb", "mar", "apr", "may", "jun",
                         "jul", "aug", "sep", "oct", "nov", "dec"]
        month = st.selectbox("Month", month_options, help="Last contact month")

    st.markdown('</div>', unsafe_allow_html=True)

    # Campaign Information Section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📈 Campaign Details</h3>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        campaign = st.number_input("Campaign Contacts", min_value=1, max_value=50, value=2,
                                  help="Number of contacts performed during this campaign")
    with col2:
        pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=1000, value=-1,
                               help="Days since last contact from previous campaign (-1 if not contacted)")
    with col3:
        poutcome_options = ["unknown", "other", "failure", "success"]
        poutcome = st.selectbox("Previous Campaign Outcome", poutcome_options,
                               help="Outcome of the previous marketing campaign")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
if st.button("🔮 Predict Subscription", type="primary", use_container_width=False):
    try:
        # Prepare input dict
        input_data = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': 15,  # fixed for now
            'month': month,
            'campaign': campaign,
            'pdays': pdays,
            'previous': 0,  # fixed for now
            'poutcome': poutcome
        }

        df_input = pd.DataFrame([input_data])

        # Map and encode categories (exactly as your model expects)
        job_categories = ['admin.', 'unknown', 'unemployed', 'management', 'housemaid',
                          'entrepreneur', 'student', 'blue-collar', 'self-employed',
                          'retired', 'technician', 'services']
        marital_categories = ['married', 'divorced', 'single']
        education_categories = ['unknown', 'secondary', 'primary', 'tertiary']
        contact_categories = ['unknown', 'telephone', 'cellular']
        month_categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        poutcome_categories = ['unknown', 'other', 'failure', 'success']

        df_input['default'] = df_input['default'].map({'no': 0, 'yes': 1})
        df_input['housing'] = df_input['housing'].map({'no': 0, 'yes': 1})
        df_input['loan'] = df_input['loan'].map({'no': 0, 'yes': 1})

        # Manual one-hot encoding as your model requires
        ohe_features = []

        # Helper to encode categories dropping first
        def encode_cat(value, categories):
            arr = [0] * (len(categories) - 1)
            if value in categories[1:]:
                idx = categories[1:].index(value)
                arr[idx] = 1
            return arr

        ohe_features.extend(encode_cat(df_input['job'].iloc[0], job_categories))
        ohe_features.extend(encode_cat(df_input['marital'].iloc[0], marital_categories))
        ohe_features.extend(encode_cat(df_input['education'].iloc[0], education_categories))
        ohe_features.extend(encode_cat(df_input['contact'].iloc[0], contact_categories))
        ohe_features.extend(encode_cat(df_input['month'].iloc[0], month_categories))
        ohe_features.extend(encode_cat(df_input['poutcome'].iloc[0], poutcome_categories))

        numerical_features = [
            df_input['age'].iloc[0],
            df_input['default'].iloc[0],
            df_input['balance'].iloc[0],
            df_input['housing'].iloc[0],
            df_input['loan'].iloc[0],
            df_input['day'].iloc[0],
            df_input['campaign'].iloc[0],
            df_input['pdays'].iloc[0],
            df_input['previous'].iloc[0]
        ]

        # Combine features for scaler and model
        X = np.array(numerical_features + ohe_features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0][1]

        result = "YES" if prediction == 1 else "NO"

        st.markdown(f"""
        <div class="prediction-box">
            <h2>Prediction: <strong>{result}</strong></h2>
            <p>Probability of subscription: <strong>{prediction_proba*100:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

