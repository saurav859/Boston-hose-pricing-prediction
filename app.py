import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Load the trained model and scaler
with open('ridge_model.pkl', 'rb') as file:
    best_ridge_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the list of input features based on the training data columns
# (excluding 'MEDV' and 'price' as they are target or redundant)
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# 2. Set up the Streamlit page configuration
st.set_page_config(page_title="House Price Prediction App", layout="centered")
st.title("Boston House Price Prediction")
st.write("Enter the details of the house to predict its price.")

# 3. Create a Streamlit form for user input
with st.form("prediction_form"):
    st.header("House Features")

    # Input widgets for each feature
    col1, col2, col3 = st.columns(3)

    with col1:
        crim = st.number_input("CRIM (Per capita crime rate)", min_value=0.0, value=0.1, format="%.4f")
        zn = st.number_input("ZN (Proportion of residential land zoned for lots over 25,000 sq.ft.)", min_value=0.0, value=0.0, format="%.2f")
        indus = st.number_input("INDUS (Proportion of non-retail business acres per town)", min_value=0.0, value=7.07, format="%.2f")
        chas = st.selectbox("CHAS (Charles River dummy variable)", options=[0, 1], index=0)
        nox = st.number_input("NOX (Nitric oxides concentration (parts per 10 million))", min_value=0.0, value=0.538, format="%.3f")

    with col2:
        rm = st.number_input("RM (Average number of rooms per dwelling)", min_value=1.0, value=6.575, format="%.3f")
        age = st.number_input("AGE (Proportion of owner-occupied units built prior to 1940)", min_value=0.0, value=65.2, format="%.1f")
        dis = st.number_input("DIS (Weighted distances to five Boston employment centres)", min_value=0.0, value=4.09, format="%.4f")
        rad = st.number_input("RAD (Index of accessibility to radial highways)", min_value=1, value=1, format="%d")

    with col3:
        tax = st.number_input("TAX (Full-value property tax rate per $10,000)", min_value=100, value=296, format="%d")
        ptratio = st.number_input("PTRATIO (Pupil-teacher ratio by town)", min_value=1.0, value=15.3, format="%.1f")
        b = st.number_input("B (1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town)", min_value=0.0, value=396.90, format="%.2f")
        lstat = st.number_input("LSTAT (% lower status of the population)", min_value=0.0, value=4.98, format="%.2f")

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        # 4. Gather user inputs into a Pandas DataFrame
        input_data = pd.DataFrame([{
            'CRIM': crim,
            'ZN': zn,
            'INDUS': indus,
            'CHAS': float(chas), # Ensure CHAS is float if scaler expects it
            'NOX': nox,
            'RM': rm,
            'AGE': age,
            'DIS': dis,
            'RAD': float(rad), # Ensure RAD is float
            'TAX': float(tax), # Ensure TAX is float
            'PTRATIO': ptratio,
            'B': b,
            'LSTAT': lstat
        }])

        # Ensure the order of columns matches the training data
        input_data = input_data[feature_names]

        # 5. Preprocess the user input DataFrame
        scaled_input_data = scaler.transform(input_data)

        # 6. Make a prediction
        prediction = best_ridge_model.predict(scaled_input_data)[0]

        # 7. Display the predicted house price
        st.success(f"Predicted House Price: ${prediction:,.2f}")
        st.balloons()
