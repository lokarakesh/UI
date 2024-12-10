import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import joblib

# Load and preprocess data efficiently
@st.cache_data
def load_data(file_path="Car_Data_Zscore.csv"):
    return pd.read_csv(file_path)

# Predefined lookup table for faster engine and power fetching
@st.cache_data
def create_lookup_table(data):
    return data.groupby(["Make", "Model", "Fuel_Type", "Transmission"])[
        ["Engine(CC)", "Power(BHP)"]
    ].mean().reset_index()

# Enhanced UI with styling
st.set_page_config(page_title="Car Price Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI appearance
st.markdown("""
    <style>
        .stApp {
            background-color: #e0f7fa;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 15px;
        }
        .stSelectbox, .stSlider {
            font-size: 16px;
            font-weight: bold;
            background-color: #ffffff;
        }
        .stTextInput {
            font-size: 16px;
            font-weight: bold;
        }
        .stRadio {
            font-size: 16px;
            font-weight: bold;
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

# Define a function to get engine and power for the selected car details
def get_engine_and_power(make, model, fuel, transmission):
    match = lookup_table[
        (lookup_table["Make"] == make) &
        (lookup_table["Model"] == model) &
        (lookup_table["Fuel_Type"] == fuel) &
        (lookup_table["Transmission"] == transmission)
    ]
    if not match.empty:
        return match.iloc[0]["Engine(CC)"], match.iloc[0]["Power(BHP)"]
    else:
        return None, None

# Layout for input fields
st.title("🚗 Car Price Prediction🚗 🚗  ")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Select Make", options=data["Make"].unique(), key="make")
        filtered_models = data[data["Make"] == make]["Model"].unique()
        model_name = st.selectbox("Select Model", options=filtered_models, key="model")
        fuel_type = st.radio("Fuel Type", options=data["Fuel_Type"].unique(), horizontal=True, key="fuel")
    with col2:
        transmission = st.radio("Transmission", options=data["Transmission"].unique(), horizontal=True, key="transmission")
        location = st.selectbox("Select Location", options=data["Location"].unique(), key="location")
        year = st.slider("Year", min_value=1980, max_value=2023, value=2015, step=1, key="year")

# Auto-fill engine and power based on selections
engine_cc, power_bhp = get_engine_and_power(make, model_name, fuel_type, transmission)

with st.container():
    col3, col4 = st.columns(2)
    with col3:
        kilometers_driven = st.number_input("Kilometers Driven (KM)", min_value=0, value=50000, step=100, key="kilometers")
        mileage = st.slider("Mileage (KMPL)", min_value=0.0, max_value=40.0, value=20.0, step=0.1, key="mileage")
    with col4:
        st.text_input("Engine (CC)", value=f"{engine_cc:.0f}" if engine_cc else "N/A", disabled=True, key="engine")
        st.text_input("Power (BHP)", value=f"{power_bhp:.0f}" if power_bhp else "N/A", disabled=True, key="power")
        seats = st.selectbox("Seats", options=range(2, 10), key="seats")

# Interactive Scatter Plot (Price vs Year)
with st.container():
    st.subheader("📊 Explore the Data")
    st.write("Interactive chart showcasing car price trends:")
    fig = px.scatter(data, x="Year", y="Price(Lakhs)", color="Fuel_Type", title="Price vs Year")
    st.plotly_chart(fig, use_container_width=True)

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
            prediction = model.predict(input_data)
            st.success(f"Predicted Price: ₹{prediction[0]:,.2f} Lakhs")

# Footer
st.markdown("---")
st.write("Made with ❤️ by Rakesh Loka")