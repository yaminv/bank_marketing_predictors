import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply CSS styling
load_css()

# Main header
st.markdown('<h1 class="main-header">🏦 Bank Marketing Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether a customer will subscribe to a term deposit</p>', unsafe_allow_html=True)

# Form container
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


# Prediction button at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
if st.button("🔮 Predict Subscription", type="primary", use_container_width=False):
    # Create input data
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
        'day': 15,  # fixed or improve later
        'month': month,
        'campaign': campaign,
        'pdays': pdays,
        'previous': 0,  # default for now
        'poutcome': poutcome
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    try:
        # Define categorical columns (same as training)
        job_categories = ['admin.', 'unknown', 'unemployed', 'management', 'housemaid',
                          'entrepreneur', 'student', 'blue-collar', 'self-employed',
                          'retired', 'technician', 'services']
        marital_categories = ['married', 'divorced', 'single']
        education_categories = ['unknown', 'secondary', 'primary', 'tertiary']
        contact_categories = ['unknown', 'telephone', 'cellular']
        month_categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        poutcome_categories = ['unknown', 'other', 'failure', 'success']

        # Map binary categorical columns
        df_input['default'] = df_input['default'].map({'no': 0, 'yes': 1})
        df_input['housing'] = df_input['housing'].map({'no': 0, 'yes': 1})
        df_input['loan'] = df_input['loan'].map({'no': 0, 'yes': 1})

        # Create OHE features manually (dropping first category for each)
        ohe_features = []

        # Job OHE (12 categories, drop first = 11 features)
        job_encoded = [0] * 11
        if df_input['job'].iloc[0] in job_categories[1:]:
            job_idx = job_categories[1:].index(df_input['job'].iloc[0])
            job_encoded[job_idx] = 1
        ohe_features.extend(job_encoded)

        # Marital OHE (3 categories, drop first = 2 features)
        marital_encoded = [0] * 2
        if df_input['marital'].iloc[0] in marital_categories[1:]:
            marital_idx = marital_categories[1:].index(df_input['marital'].iloc[0])
            marital_encoded[marital_idx] = 1
        ohe_features.extend(marital_encoded)

        # Education OHE (4 categories, drop first = 3 features)
        education_encoded = [0] * 3
        if df_input['education'].iloc[0] in education_categories[1:]:
            education_idx = education_categories[1:].index(df_input['education'].iloc[0])
            education_encoded[education_idx] = 1
        ohe_features.extend(education_encoded)

        # Contact OHE (3 categories, drop first = 2 features)
        contact_encoded = [0] * 2
        if df_input['contact'].iloc[0] in contact_categories[1:]:
            contact_idx = contact_categories[1:].index(df_input['contact'].iloc[0])
            contact_encoded[contact_idx] = 1
        ohe_features.extend(contact_encoded)

        # Month OHE (12 categories, drop first = 11 features)
        month_encoded = [0] * 11
        if df_input['month'].iloc[0] in month_categories[1:]:
            month_idx = month_categories[1:].index(df_input['month'].iloc[0])
            month_encoded[month_idx] = 1
        ohe_features.extend(month_encoded)

        # Poutcome OHE (4 categories, drop first = 3 features)
        poutcome_encoded = [0] * 3
        if df_input['poutcome'].iloc[0] in poutcome_categories[1:]:
            poutcome_idx = poutcome_categories[1:].index(df_input['poutcome'].iloc[0])
            poutcome_encoded[poutcome_idx] = 1
        ohe_features.extend(poutcome_encoded)

        # Combine numerical features with OHE features
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

        # Create final feature array (9 numerical + 44 OHE = 53 features)
        final_features = numerical_features + ohe_features

        # If less than 54 features, pad with zeros
        while len(final_features) < 54:
            final_features.append(0)

        final_features = final_features[:54]

        # Convert to numpy array and reshape
        X_pred = np.array(final_features).reshape(1, -1)

        # Apply scaling BEFORE prediction
        X_pred_scaled = scaler.transform(X_pred)

        # Predict with scaled input
        prediction = model.predict(X_pred_scaled)[0]
        probability = model.predict_proba(X_pred_scaled)[0]

        # Display prediction results
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

        if prediction == 1 or prediction == 'yes':
            st.markdown("## ✅ **PREDICTION: YES**")
            st.markdown("### The customer is likely to subscribe to a term deposit")
            confidence = probability[1] if len(probability) > 1 else probability[0]
            st.markdown(f"### Confidence: **{confidence:.1%}**")
        else:
            st.markdown("## ❌ **PREDICTION: NO**")
            st.markdown("### The customer is unlikely to subscribe to a term deposit")
            confidence = probability[0] if len(probability) > 1 else probability[0]
            st.markdown(f"### Confidence: **{confidence:.1%}**")

        st.markdown('</div>', unsafe_allow_html=True)

        # Show detailed probabilities
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("No Subscription", f"{probability[0]:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_prob2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Yes Subscription", f"{probability[1]:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Show input summary
        with st.expander("📋 Customer Information Summary"):
            st.write("**Personal Details:**")
            st.write(f"- Age: {age} years")
            st.write(f"- Job: {job}")
            st.write(f"- Marital Status: {marital}")
            st.write(f"- Education: {education}")

            st.write("**Financial Details:**")
            st.write(f"- Balance: €{balance:,}")
            st.write(f"- Credit Default: {default}")
            st.write(f"- Housing Loan: {housing}")
            st.write(f"- Personal Loan: {loan}")

            st.write("**Contact Details:**")
            st.write(f"- Contact Type: {contact}")
            st.write(f"- Contact Date: 15 {month} (default)")

            st.write("**Campaign Details:**")
            st.write(f"- Campaign Contacts: {campaign}")
            st.write(f"- Days Since Last Contact: {pdays}")
            st.write(f"- Previous Contacts: 0 (default)")
            st.write(f"- Previous Outcome: {poutcome}")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check that all input fields are filled correctly.")


