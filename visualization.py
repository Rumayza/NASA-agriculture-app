
import plotly.express as px
import streamlit as st

def plot_yield_trends(data):
    """
    Plot yield trends over time using Plotly.
    """
    fig = px.line(data, x='Date', y='Yield', title='Crop Yield Over Time')
    st.plotly_chart(fig)

def plot_soil_moisture_map(data):
    """
    Plot soil moisture levels on a map using Plotly.
    """
    fig = px.scatter_mapbox(data, lat='latitude', lon='longitude', color='Soil Moisture',
                            mapbox_style='carto-positron', title='Soil Moisture Levels')
    st.plotly_chart(fig)
