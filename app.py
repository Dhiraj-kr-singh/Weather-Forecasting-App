import streamlit as st
import pickle
import pandas as pd

# Load the trained model and label encoders
with open("weather_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder_year.pkl", "rb") as year_file:
    label_encoder_year = pickle.load(year_file)

with open("label_encoder_state.pkl", "rb") as state_file:
    label_encoder_state = pickle.load(state_file)

# Title
st.title("Rainfall Prediction App")

# Input: State
state = st.selectbox("Select State", ["Kolkata", "Meghalaya", "Goa", "Mizoram"])

# Input: Year (Text input for simplicity)
year = st.text_input("Enter Year (e.g., 2023)")

# Predict rainfall
if st.button("Predict Rainfall"):
    try:
        # Validate the year input
        if not year.isdigit() or len(year) != 4:
            st.error("Please enter a valid year (4 digits).")
        else:
            # Prepare the Month-Year format (e.g., '2023-Jan')
            month_year = f"{year}-Jan"  # You can modify this part based on user selection for a month

            # Encode inputs
            encoded_state = label_encoder_state.transform([state])[0]
            encoded_month_year = label_encoder_year.transform([month_year])[0]
            
            # Prepare input data for prediction
            data = pd.DataFrame([[encoded_month_year, encoded_state]], columns=["Month-Year", "State"])
            
            # Make prediction
            prediction = model.predict(data)[0]
            
            # Show result with an appropriate message
            st.success(f"The predicted rainfall in {state} for {year} is {prediction:.2f} mm.")
            
    except Exception as e:
        st.error(f"Error: {e}")
