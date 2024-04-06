import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime, timedelta
from datetime import date
from matplotlib.patches import Patch
import time
from openai import OpenAI
import streamlit_shadcn_ui as ui
import requests
import plotly.graph_objects as go
import plotly.express as px

client = OpenAI()

from utils import *

st.set_page_config(page_title="AI MedReview: FFT")

# Assuming this HTML is for styling and doesn't need changes
html = """
<style>
.gradient-text {
    background: linear-gradient(45deg, #284d74, #d8ad45, #ae4f4d);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 2em;
    font-weight: bold;
}
</style>
<div class="gradient-text">AI MedReview: FFT</div>
"""

# Define a list of PCN names
pcn_names = ["Brompton Health PCN", "Oakwood PCN"]

# Initialize session state for PCN if not present
if "pcn" not in st.session_state:
    st.session_state.pcn = pcn_names[0]  # Default to first PCN


# Update PCN selection
def update_pcn():
    st.session_state.pcn = st.session_state.pcn_selector


# Load data
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("ai_medreview/data/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df


data = load_data()


# Define the get_surgeries_by_pcn function here
@st.cache_data(ttl=3600)
def get_surgeries_by_pcn(data, selected_pcn):
    filtered_data = data[data["pcn"] == selected_pcn]
    surgeries = filtered_data["surgery"].unique()
    return np.sort(surgeries)


# Sidebar content with PCN selection
st.sidebar.markdown(html, unsafe_allow_html=True)
st.sidebar.image("images/transparent2.png")

# Updating PCN selection and fetching corresponding surgeries
selected_pcn = st.sidebar.selectbox(
    "Select a PCN:", pcn_names, key="pcn_selector", on_change=update_pcn
)


# Define the get_surgeries_by_pcn function to ensure it returns sorted surgeries
@st.cache_data(ttl=3600)
def get_surgeries_by_pcn(data, selected_pcn):
    filtered_data = data[data["pcn"] == selected_pcn]
    surgeries = filtered_data["surgery"].unique()
    return sorted(surgeries)  # Return a sorted list


# Only get and display the surgery list if not on the PCN Dashboard or About pages
page = st.sidebar.radio(
    "Select a Page",
    [
        "PCN Dashboard",
        "Surgery Dashboard",
        "Feedback Classification",
        "Improvement Suggestions",
        "Feedback Timeline",
        "GPT4 Summary",
        "Word Cloud",
        "GPT-4 Summary",
        "Dataframe",
        "About",
    ],
)

if page not in ["PCN Dashboard", "About"]:
    surgery_list = get_surgeries_by_pcn(data, selected_pcn)
    if len(surgery_list) > 0:  # Ensuring there are surgeries to select
        selected_surgery = st.sidebar.selectbox("Select Surgery", surgery_list)

        surgery_data = data[
            (data["pcn"] == selected_pcn) & (data["surgery"] == selected_surgery)
        ]

        if not surgery_data.empty:
            start_date = surgery_data["time"].dt.date.min()
            end_date = surgery_data["time"].dt.date.max()

            # Ensure the dates are datetime.date objects; no need for to_pydatetime conversion
            selected_date_range = st.slider(
                f"**{selected_surgery}** Date Range:",
                min_value=start_date,
                max_value=end_date,
                value=(start_date, end_date),
                format="MM/DD/YYYY",
            )

            @st.cache_data(ttl=3600)
            def filter_data_by_date_range(data, date_range):
                return data[
                    (data["time"].dt.date >= date_range[0])
                    & (data["time"].dt.date <= date_range[1])
                ]

            filtered_data = filter_data_by_date_range(surgery_data, selected_date_range)
else:
    selected_surgery = None


# Content Start ========================================================================================== Content Start

# -- PCN Dashboard --------------------------------------------------------------------------------------- PCN Dashboard
if page == "PCN Dashboard":
    st.title(f"{selected_pcn}")


# -- Surgery Dashboard ------------------------------------------------------------------------------- Surgery Dashboard
elif page == "Surgery Dashboard":
    st.title(f"{selected_surgery}")


# -- Feedback Classification ------------------------------------------------------------------- Feedback Classification
elif page == "Feedback Classification":
    st.title("Feedback Classification")


# -- Improvement Suggestion --------------------------------------------------------------------- Improvement Suggestion
elif page == "Improvement Suggestion":
    st.title("Improvement Suggestion")


# -- Feedback Timeline ------------------------------------------------------------------------------- Feedback Timeline
elif page == "Feedback Timeline":
    st.title("Feedback Timeline")


# -- Word Cloud --------------------------------------------------------------------------------------------- Word Cloud
elif page == "Word Cloud":
    st.title("Word Cloud")


# -- GPT4 Summary ----------------------------------------------------------------------------------------- GPT4 Summary
elif page == "GPT-4 Summary":
    st.title("GPT-4 Summary")


# -- Dataframe ----------------------------------------------------------------------------------------------- Dataframe
elif page == "Dataframe":
    st.title("Dataframe")


# -- About ------------------------------------------------------------------------------------------------------- About
elif page == "About":
    st.title("About")
