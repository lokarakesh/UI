import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import plotly.express as px
import joblib

# Load the dataset
file_path = "Car_Data_Zscore.csv"  # Update this path if needed
data = pd.read_csv(file_path)

# Features and Target
X = data.drop("Price(Lakhs)", axis=1)
y = data["Price(Lakhs)"]

# Define preprocessing
categorical_features = ["Make", "Model", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
numerical_features = ["Year", "Kilometers_Driven", "Mileage(KMPL)", "Engine(CC)", "Power(BHP)", "Seats"]

categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[("num", numerical_transformer, numerical_features),
                  ("cat", categorical_transformer, categorical_features)]
)

# Define the pipeline
model = Pipeline(steps=[("preprocessor", preprocessor), 
                        ("regressor", RandomForestRegressor(random_state=42))])

# Train the model
model.fit(X, y)

# Save the model for future use
joblib.dump(model, 'car_price_predictor.pkl')

# Set page configuration and theme
st.set_page_config(page_title="Car Price Prediction", layout="wide", initial_sidebar_state="expanded")

# Add custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f0f5;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

# Welcome message
st.write("Welcome to the **Used Car Price Prediction** App! Fill in the details below to get the predicted price of the used car.")

# Group inputs into sections using expander
with st.expander("Car Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Make", options=data["Make"].unique())
        filtered_models = data[data["Make"] == make]["Model"].unique()
        model_name = st.selectbox("Model", options=filtered_models)
        location = st.selectbox("Location", options=data["Location"].unique())
        year = st.number_input("Year", min_value=1980, max_value=2023, step=1)
    with col2:
        fuel_type = st.selectbox("Fuel Type", options=data["Fuel_Type"].unique())
        transmission = st.selectbox("Transmission", options=data["Transmission"].unique())
        owner_type = st.selectbox("Owner Type", options=data["Owner_Type"].unique())

with st.expander("Car Specifications", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
        mileage = st.slider("Mileage (KMPL)", 0.0, 40.0, 15.0, 0.1)
    with col4:
        engine = st.number_input("Engine (CC)", min_value=500.0, step=100.0)
        power = st.number_input("Power (BHP)", min_value=20.0, step=5.0)
        seats = st.number_input("Seats", min_value=2, max_value=10, step=1)

# Interactive chart
fig = px.scatter(data, x="Year", y="Price(Lakhs)", color="Fuel_Type", title="Price vs Year")
st.plotly_chart(fig)

# Using a form for better input handling
with st.form(key="car_form"):
    submit_button = st.form_submit_button(label="Predict Price")
    if submit_button:
        input_data = pd.DataFrame({
            "Make": [make], "Model": [model_name], "Location": [location], "Year": [year],
            "Fuel_Type": [fuel_type], "Kilometers_Driven": [kilometers_driven], "Transmission": [transmission],
            "Owner_Type": [owner_type], "Mileage(KMPL)": [mileage], "Engine(CC)": [engine],
            "Power(BHP)": [power], "Seats": [seats]
        })
        
        # Prediction
        with st.spinner('Calculating the price...'):
            prediction = model.predict(input_data)
            st.success(f"Predicted Price: ₹{prediction[0]:,.2f} Lakhs")
