import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Application Configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Load Model and Feature Columns
model_path = os.path.join('models', 'final_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

feature_columns_path = os.path.join('models', 'feature_columns.pkl')
with open(feature_columns_path, 'rb') as file:
    feature_columns = pickle.load(file)

# Main Title
st.title('🚗 Car Price Prediction Application')
st.markdown("---")

# Sidebar - Application Info
st.sidebar.header("About the Application")
st.sidebar.markdown("""
### Car Price Prediction System
This application uses advanced machine learning algorithms to predict car prices.

**Features:**
- Price predictions based on real market data
- Analysis of more than 50 vehicle attributes
- Instant price estimation
- Detailed vehicle feature analysis

**Data Source:**
- USA Cars Dataset
- 2,500+ real vehicle records
- Up-to-date market analysis
""")

# Main Panel - User Input
st.header('Enter Vehicle Features')

# 2-column Layout
col1, col2 = st.columns(2)

# First Column
with col1:
    brand = st.selectbox('Brand', ['ford', 'chevrolet', 'toyota', 'honda', 'bmw', 'nissan', 'dodge', 'mercedes-benz'])
    year = st.slider('Model Year', 2000, 2024, 2020)
    mileage = st.number_input('Mileage', min_value=0, max_value=300000, value=50000, step=1000)

# Second Column
with col2:
    color = st.selectbox('Color', ['white', 'black', 'silver', 'gray', 'blue', 'red'])
    title_status = st.selectbox('Vehicle Condition', ['clean vehicle', 'salvage insurance loss'])
    state = st.selectbox('State', ['california', 'florida', 'texas', 'new york', 'pennsylvania'])

# Prediction Button
if st.button('Predict Price', type='primary'):
    try:
        # Feature Engineering
        input_data = pd.DataFrame({
            'brand': [brand],
            'year': [year],
            'mileage': [mileage],
            'color': [color],
            'state': [state],
            'title_status': [title_status]
        })

        # Derived Features
        input_data['car_age'] = 2024 - input_data['year']
        input_data['avg_km_per_year'] = input_data['mileage'] / input_data['car_age']
        input_data['is_premium'] = input_data['brand'].isin(['bmw', 'mercedes-benz']).astype(int)
        input_data['is_popular_color'] = input_data['color'].isin(['white', 'black', 'silver', 'gray']).astype(int)
        input_data['clean_title_score'] = (input_data['title_status'] == 'clean vehicle').astype(int)

        # One-hot Encoding
        input_data = pd.get_dummies(input_data)

        # Ensure Input Matches Model's Expected Format
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Arrange Columns to Match Model
        input_data = input_data[feature_columns]

        # Prediction
        prediction = model.predict(input_data)[0]

        # Display Results
        st.success(f'Predicted Price: ${prediction:,.2f}')

        # Detailed Explanation
        st.markdown("---")
        st.markdown("### Factors Affecting the Price")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"- Car Age: {2024 - year} years")
            st.write(f"- Mileage: {mileage:,} km")
            st.write(f"- Premium Brand: {'Yes' if brand in ['bmw', 'mercedes-benz'] else 'No'}")

        with col2:
            st.write(f"- Condition: {title_status}")
            st.write(f"- Location: {state}")
            st.write(f"- Color: {color}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all fields are filled in correctly.")
