import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# Load model and encoders/scaler once
@st.cache_data(allow_output_mutation=True)
def load_all():
    model = load_model('model.h5')
    with open('label_encoder_cname.pkl', 'rb') as f:
        label_encoder_cname = pickle.load(f)
    with open('onehotEncoder_type.pkl', 'rb') as f:
        onehot_encoder_type = pickle.load(f)
    with open('scalar.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, label_encoder_cname, onehot_encoder_type, scaler

model, label_encoder_cname, onehot_encoder_type, scaler = load_all()

# App Title
st.title("📊 Company Questionnaire - Predict Average Score")

# Form-based input
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
        # Prepare input DataFrame
        input_dict = {"Company Name": [company_name]}
        input_dict.update({k: [v] for k, v in q_cols.items()})
        input_df = pd.DataFrame(input_dict)

        # One-hot encode Type
        type_encoded = onehot_encoder_type.transform([[type_val]])
        type_encoded_df = pd.DataFrame(type_encoded, columns=onehot_encoder_type.get_feature_names_out(['Type']))
        input_df = pd.concat([input_df, type_encoded_df], axis=1)

        # Label encode Company Name
        input_df['Company Name'] = label_encoder_cname.transform(input_df['Company Name'])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction[0])
        predicted_average = predicted_class + 1  # Convert class index (0–9) to score (1–10)

        st.success(f"🎯 Predicted Average Score: **{predicted_average}**")

        # Visualize prediction probabilities
        st.markdown("### 📊 Prediction Confidence")
        probabilities = prediction[0]
        class_labels = [f"{i+1}" for i in range(len(probabilities))]
        prob_df = pd.DataFrame({
            'Predicted Score': class_labels,
            'Probability': probabilities
        })

        st.bar_chart(prob_df.set_index('Predicted Score'))

    except ValueError as ve:
        st.error(f"⚠️ Input Error: {ve}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
