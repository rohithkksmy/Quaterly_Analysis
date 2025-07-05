import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

model, label_encoder_cname, onehot_encoder_type, scaler = load_all()

st.title("📊 Company Questionnaire - Predict Average Score")

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

        st.write("Input df BEFORE encoding Type:")
        st.write(input_df)

        # One-hot encode Type
        type_encoded = onehot_encoder_type.transform([[type_val]])
        type_encoded_df = pd.DataFrame(type_encoded, columns=onehot_encoder_type.get_feature_names_out(['Type']))

        # Reset indices before concatenation
        input_df = input_df.reset_index(drop=True)
        type_encoded_df = type_encoded_df.reset_index(drop=True)

        # Concatenate input data with one-hot encoded Type columns
        input_df = pd.concat([input_df, type_encoded_df], axis=1)

        st.write("Input df AFTER concatenating one-hot encoded Type:")
        st.write(input_df)

        # Label encode Company Name
        input_df['Company Name'] = label_encoder_cname.transform(input_df['Company Name'])

        st.write("Input df AFTER label encoding Company Name:")
        st.write(input_df)

        # Check scaler expected columns order and reorder input_df accordingly
        if hasattr(scaler, 'feature_names_in_'):
            expected_cols = scaler.feature_names_in_
            st.write("Scaler expects columns (in this order):")
            st.write(expected_cols)
            # Reorder input_df columns to match scaler expected order
            input_df = input_df[expected_cols]
        else:
            st.warning("Scaler does not have feature_names_in_. Make sure input columns are in correct order.")

        st.write("Final input_df before scaling:")
        st.write(input_df)

        # Scale features
        input_scaled = scaler.transform(input_df)

        st.write("Scaled input shape:", input_scaled.shape)

        # Predict
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction[0])
        predicted_average = predicted_class + 1  # Classes 0-9 map to scores 1-10

        st.success(f"🎯 Predicted Average Score: **{predicted_average}**")

        # Visualization of prediction confidence
        st.markdown("### 📊 Prediction Confidence")
        probabilities = prediction[0]
        class_labels = [str(i + 1) for i in range(len(probabilities))]
        prob_df = pd.DataFrame({
            'Predicted Score': class_labels,
            'Probability': probabilities
        })

        st.bar_chart(prob_df.set_index('Predicted Score'))

    except ValueError as ve:
        st.error(f"⚠️ Input Error: {ve}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
