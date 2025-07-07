import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model

@st.cache_resource(show_spinner=False)
def load_all():
    model = load_model('model.h5')
    with open('label_encoder_cname.pkl', 'rb') as f:
        label_encoder_cname = pickle.load(f)
    with open('onehotEncoder_type.pkl', 'rb') as f:
        onehot_encoder_type = pickle.load(f)
    with open('scalar.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, label_encoder_cname, onehot_encoder_type, scaler

def fetch_company_rating(company_name: str) -> float:
    """
    Placeholder for fetching real-time company rating from an API.
    Replace this with your actual API call.
    """
    try:
        # Simulate an API call — replace with your real endpoint
        # response = requests.get(f"https://api.example.com/company_rating?name={company_name}")
        # response.raise_for_status()
        # data = response.json()
        # return float(data.get('rating', 5.0))
        
        # For demonstration, generate a dummy rating (e.g., 1 to 10)
        import random
        return random.uniform(3, 9)  # random rating between 3 and 9
    except Exception as e:
        st.warning(f"Failed to fetch live rating: {e}")
        return 5.0  # fallback rating

def classify_score(score: int) -> str:
    if score <= 3:
        return "Pioneering"
    elif score <= 7:
        return "Advanced"
    else:
        return "Fully Functional"

model, label_encoder_cname, onehot_encoder_type, scaler = load_all()

st.title("📊 Company Questionnaire - Predict Average Score with Real-Time Data")

with st.form("prediction_form"):
    st.markdown("### 🏢 Company Information")
    company_name = st.text_input("Company Name", value="Ultramatics1")
    type_val = st.selectbox("Type", options=onehot_encoder_type.categories_[0])

    st.markdown("### ✅ Answer Questions (Q1 to Q10)")
    q_cols = {}
    col1, col2 = st.columns(2)
    with col1:
        for i in range(1, 6):
            q_cols[f"Q{i}"] = st.slider(f"Q{i}", min_value=0, max_value=10, value=5, step=1)
    with col2:
        for i in range(6, 11):
            q_cols[f"Q{i}"] = st.slider(f"Q{i}", min_value=0, max_value=10, value=5, step=1)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    try:
        # Prepare input DataFrame with company name and question answers
        input_dict = {"Company Name": [company_name]}
        input_dict.update({k: [v] for k, v in q_cols.items()})
        input_df = pd.DataFrame(input_dict)

        # Add live company rating fetched from API
        live_rating = fetch_company_rating(company_name)
        input_df['Live Company Rating'] = live_rating

        # One-hot encode Type
        type_encoded = onehot_encoder_type.transform([[type_val]])
        if hasattr(type_encoded, "toarray"):
            type_encoded = type_encoded.toarray()
        type_encoded_df = pd.DataFrame(type_encoded, columns=onehot_encoder_type.get_feature_names_out(['Type']))

        input_df = input_df.reset_index(drop=True)
        type_encoded_df = type_encoded_df.reset_index(drop=True)

        input_df = pd.concat([input_df, type_encoded_df], axis=1)

        # Label encode Company Name
        input_df['Company Name'] = label_encoder_cname.transform(input_df['Company Name'])

        # Handle missing columns expected by scaler
        if hasattr(scaler, 'feature_names_in_'):
            expected_cols = scaler.feature_names_in_
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_cols]

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict average score
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction[0])
        predicted_average = predicted_class + 1  # map 0-9 to 1-10 scale

        category = classify_score(predicted_average)
        st.success(f"🎯 Predicted Average Score: **{predicted_average}**")
        st.info(f"Classification: **{category}**")

    except ValueError as ve:
        st.error(f"⚠️ Input Error: {ve}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
