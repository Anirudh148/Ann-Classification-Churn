import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(page_title="Churn Predictor", layout="centered")

# 2. Cache the assets so they only load ONCE
@st.cache_resource
def load_all_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

# Load them
model, label_encoder_gender, onehot_encoder_geo, scaler = load_all_assets()

# 3. The User Interface
st.title('🏦 Bank Customer Churn Prediction')
st.markdown("Adjust the sliders and dropdowns, then click **Predict**.")

# Organization into columns
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input('Credit Score', 300, 850, 600)
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 40)
    tenure = st.slider('Tenure (Years)', 0, 10, 3)

with col2:
    balance = st.number_input('Account Balance', value=60000.0)
    num_of_products = st.slider('Number of Products', 1, 4, 2)
    has_cr_card = st.selectbox('Has Credit Card?', [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    is_active_member = st.selectbox('Is Active Member?', [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    estimated_salary = st.number_input('Estimated Salary', value=50000.0)

# 4. The Logic
if st.button('Predict Churn Probablity', use_container_width=True):
    # Prepare Input DataFrame
    input_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encoding for Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Combine and Scale
    final_input = pd.concat([input_df, geo_df], axis=1)
    final_input_scaled = scaler.transform(final_input)

    # Make Prediction
    prediction = model.predict(final_input_scaled)
    prob = prediction[0][0]

    # Display Results
    st.divider()
    if prob > 0.5:
        st.error(f"### Result: High Risk of Churn ({prob:.1%})")
        st.write("This customer is likely to leave the bank.")
    else:
        st.success(f"### Result: Low Risk of Churn ({prob:.1%})")
        st.write("This customer is likely to stay.")