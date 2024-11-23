import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and label encoder
with open("weather_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder_state.pkl", "rb") as state_file:
    label_encoder_state = pickle.load(state_file)

# Sample data to show available states and years for user selection (Replace with actual data)
states = ["Kolkata", "Meghalaya", "Goa", "Mizoram"]
years = [2023, 2024]  # You can dynamically fetch years from your dataset

# Title
st.title("Rainfall Prediction App")

# User input for selecting state
state = st.selectbox("Select State", states)

# User input for selecting year
year = st.selectbox("Select Year", years)

# Predict rainfall
if st.button("Predict Rainfall"):
    try:
        # Encode the state input
        encoded_state = label_encoder_state.transform([state])[0]
        
        # Prepare the input data for prediction
        input_data = pd.DataFrame([[encoded_state, int(year)]], columns=["State", "Year"])
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        
        # Display the result
        st.success(f"The predicted average rainfall in {state} for {year} is {prediction:.2f} mm.")
    
    except Exception as e:
        st.error(f"Error: {e}")
