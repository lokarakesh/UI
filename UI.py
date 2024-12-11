import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import joblib
import time

# Load and preprocess data efficiently
@st.cache_data
def load_data(file_path="Car_Data_Zscore.csv"):
    return pd.read_csv(file_path)

# Predefined lookup table for faster engine, power, and seats fetching
@st.cache_data
def create_lookup_table(data):
    return data.groupby(["Make", "Model", "Fuel_Type", "Transmission"])[
        ["Engine(CC)", "Power(BHP)", "Seats"]
    ].mean().reset_index()

# Car seat mapping based on the model
def get_seat_count(model_name):
    seat_mapping = {
        "Swift": 5,
        "Innova": 7,
        "Fortuner": 7,
        "Pajero": 7,
        "Creta": 5,
        "Breeza": 5,
        "Baleno": 5,
        "Verna": 5
    }
    return seat_mapping.get(model_name, 5)

# Enhanced UI with styling
st.set_page_config(page_title="Car Price Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI appearance
st.markdown("""
    <style>
        .stApp {
            background-color: #f1f5f8;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-size: 20px;
            border-radius: 8px;
            padding: 15px;
        }
        .stSelectbox, .stSlider, .stNumberInput, .stRadio {
            font-size: 16px;
            font-weight: bold;
            background-color: #ffffff;
            border-radius: 5px;
            padding: 8px;
        }
        .stSelectbox:hover, .stSlider:hover {
            background-color: #d3e5ff;
        }
        .stTextInput {
            font-size: 16px;
            font-weight: bold;
        }
        .stRadio {
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0069d9;
        }
        .container {
            margin-top: 50px;
        }
        .car-image {
            width: 100%;
            max-height: 300px;
            object-fit: cover;
        }
    </style>
    """, unsafe_allow_html=True)

# Load the dataset
data = load_data()
lookup_table = create_lookup_table(data)

# Features and Target
X = data.drop("Price(Lakhs)", axis=1)
y = data["Price(Lakhs)"]

# Define the preprocessing and model pipeline
categorical_features = ["Make", "Model", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
numerical_features = ["Year", "Kilometers_Driven", "Mileage(KMPL)", "Engine(CC)", "Power(BHP)", "Seats"]

categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[("num", numerical_transformer, numerical_features),
                  ("cat", categorical_transformer, categorical_features)]
)

model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("regressor", RandomForestRegressor(random_state=42))])

model.fit(X, y)

# Define a function to get engine, power, and seats for the selected car details
def get_engine_power_seats(make, model, fuel, transmission):
    match = lookup_table[
        (lookup_table["Make"] == make) &
        (lookup_table["Model"] == model) &
        (lookup_table["Fuel_Type"] == fuel) &
        (lookup_table["Transmission"] == transmission)
    ]
    if not match.empty:
        return match.iloc[0]["Engine(CC)"], match.iloc[0]["Power(BHP)"], match.iloc[0]["Seats"]
    else:
        return None, None, None

# Layout for input fields
st.title("🚗 Car Price Prediction")
st.write("#### Fill in the details below to get the estimated price of the used car.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Select Make", options=data["Make"].unique(), key="make", help="Select the make of the car.")
        filtered_models = data[data["Make"] == make]["Model"].unique()
        model_name = st.selectbox("Select Model", options=filtered_models, key="model", help="Select the car model.")
        fuel_type = st.radio("Fuel Type", options=data["Fuel_Type"].unique(), horizontal=True, key="fuel", help="Select fuel type.")
    with col2:
        transmission = st.radio("Transmission", options=data["Transmission"].unique(), horizontal=True, key="transmission", help="Select transmission type.")
        location = st.selectbox("Select Location", options=data["Location"].unique(), key="location", help="Select the car's location.")
        year = st.slider("Year", min_value=1980, max_value=2023, value=2015, step=1, key="year", help="Select the year of manufacture.")

# Auto-fill engine, power, and seats based on selections
engine_cc, power_bhp, seats = get_engine_power_seats(make, model_name, fuel_type, transmission)

# Set the seats dynamically based on model
seats = get_seat_count(model_name) if seats is None else seats

with st.container():
    col3, col4 = st.columns(2)
    with col3:
        kilometers_driven = st.number_input("Kilometers Driven (KM)", min_value=0, value=50000, step=100, key="kilometers", help="Enter the total kilometers driven.")
        mileage = st.slider("Mileage (KMPL)", min_value=0.0, max_value=40.0, value=20.0, step=0.1, key="mileage", help="Select mileage (KMPL).")
    with col4:
        st.text_input("Engine (CC)", value=f"{engine_cc:.0f}" if engine_cc else "N/A", disabled=True, key="engine", help="The engine capacity will be automatically filled.")
        st.text_input("Power (BHP)", value=f"{power_bhp:.0f}" if power_bhp else "N/A", disabled=True, key="power", help="The power will be automatically filled.")
        st.number_input("Seats", value=seats, min_value=2, max_value=10, step=1, key="seats", help="The number of seats will be auto-selected based on the model.")

# Display car image based on Make and Model selection
car_image_url = f"https://via.placeholder.com/800x300?text={model_name}+{make}+Car+Image"  # Placeholder for demo
st.image(car_image_url, caption=f"{make} {model_name}", use_column_width=True, class_="car-image")

# Interactive Scatter Plot (Price vs Year)
fig = px.scatter(data, x="Year", y="Price(Lakhs)", color="Fuel_Type", title="Price vs Year")
st.plotly_chart(fig)

# Submit button for price prediction
with st.form(key="car_form"):
    submit_button = st.form_submit_button(label="Predict Price")
    if submit_button:
        input_data = pd.DataFrame({
            "Make": [make],
            "Model": [model_name],
            "Location": [location],
            "Year": [year],
            "Fuel_Type": [fuel_type],
            "Kilometers_Driven": [kilometers_driven],
            "Transmission": [transmission],
            "Owner_Type": ["First Owner"],  # Assuming "First Owner" for demo purpose
            "Mileage(KMPL)": [mileage],
            "Engine(CC)": [engine_cc],
            "Power(BHP)": [power_bhp],
            "Seats": [seats]
        })

        with st.spinner('Calculating the price...'):
            time.sleep(1)  # Simulate a delay for the model prediction (Remove in production)
            prediction = model.predict(input_data)
            st.success(f"Predicted Price: ₹{prediction[0]:,.2f} Lakhs")
# Footer
st.markdown("---")
st.write("Made with ❤️ by [Your Name]")