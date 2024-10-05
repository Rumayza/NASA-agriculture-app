
import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import folium
from streamlit_folium import st_folium
from twilio.rest import Client
from data_preprocessing import fetch_data, preprocess_data
from tensorflow.keras.models import load_model
import plotly.express as px

# Constants for SMS Alerts
TWILIO_ACCOUNT_SID = "your_twilio_account_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "your_twilio_phone_number"
RECIPIENT_PHONE_NUMBER = "recipient_phone_number"

# Load pre-trained models
def load_crop_yield_model():
    with open('models/crop_yield_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_vegetation_health_model():
    model = load_model('models/vegetation_health_model.h5')
    return model

# Function to send SMS alerts
def send_sms_alert(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )

# Run the web app
def run_web_app():
    # Set page configuration
    st.set_page_config(page_title="Agricultural Decision-Making Tool", page_icon="🌾", layout="wide")

    # Add a header image
    st.image("https://spaceappschallenge.org/static/images/og-image.png", use_column_width=True, caption="NASA Space Apps Challenge 2024")
    st.title("🌱 Agricultural Decision-Making Tool")

    # Sidebar
    st.sidebar.image("https://spaceappschallenge.org/static/images/logo.svg", use_column_width=True)
    st.sidebar.title("Select Task")
    tasks = ["Crop Yield Prediction", "Vegetation Health Monitoring", "Irrigation Scheduling", "Drought Warning"]
    task = st.sidebar.selectbox("Choose a task", tasks)

    if task == "Crop Yield Prediction":
        st.subheader("🌾 Crop Yield Prediction")
        region = st.text_input("Enter region coordinates (latitude, longitude):")
        if st.button("Fetch Data & Predict Yield"):
            data = fetch_data()
            processed_data = preprocess_data(data)
            st.write("Fetched Data Sample:")
            st.write(processed_data.head())

            model = load_crop_yield_model()
            X = processed_data[['T2M', 'TS', 'PRECTOTCORR']]
            predictions = model.predict(X)
            st.write("Predicted Crop Yield:")
            st.write(predictions)

    elif task == "Vegetation Health Monitoring":
        st.subheader("🍃 Vegetation Health Monitoring")
        uploaded_file = st.file_uploader("Upload an NDVI image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = tf.image.decode_image(uploaded_file.read(), channels=3)
            image = tf.image.resize(image, [64, 64])
            image = tf.expand_dims(image, axis=0)

            model = load_vegetation_health_model()
            prediction = model.predict(image)
            st.write("Vegetation Health Status:")
            st.write("Healthy 🌿" if prediction[0][0] > 0.5 else "Unhealthy 🍂")

    elif task == "Irrigation Scheduling":
        st.subheader("💧 Irrigation Scheduling")
        data = fetch_data()
        processed_data = preprocess_data(data)

        st.write("Soil Moisture Levels Map:")
        m = folium.Map(location=[27.7172, 85.324], zoom_start=5)
        folium.Marker(location=[27.7172, 85.324], popup="Sample Farm Location", icon=folium.Icon(color='green')).add_to(m)
        st_folium(m, width=700, height=500)

        if st.button("Send Irrigation Alert"):
            send_sms_alert("Irrigation is needed for your farm based on current soil moisture levels.")

    elif task == "Drought Warning":
        st.subheader("🔥 Drought Warning System")
        data = fetch_data()
        processed_data = preprocess_data(data)

        st.write("Drought Condition Map:")
        drought_data = pd.DataFrame({
            'latitude': [27.7172],
            'longitude': [85.324],
            'Drought Severity': [3]
        })

        drought_map = px.scatter_mapbox(drought_data, lat="latitude", lon="longitude", color="Drought Severity",
                                        color_continuous_scale="thermal", size_max=15, zoom=5, mapbox_style="carto-positron")
        st.plotly_chart(drought_map)

        if st.button("Send Drought Warning Alert"):
            send_sms_alert("Drought warning: Your farm is at risk of drought. Take necessary precautions.")

if __name__ == "__main__":
    run_web_app()
