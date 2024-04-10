import streamlit as st
import pandas as pd
from datetime import timedelta
import os

st.write(os.getcwd())

# Define a list of PCNs
pcn_names = ["Demo-PCN", "Brompton-Health-PCN"]

# PCN selection in sidebar
selected_pcn = st.sidebar.selectbox("Select a PCN:", pcn_names, key="pcn_selector")

# Function to load PCN specific data and get surgeries by PCN
@st.cache
def load_pcn_specific_data_and_surgeries(pcn_name):
    df = pd.read_csv("data/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    pcn_data = df[df["pcn"] == pcn_name]
    surgeries = sorted(pcn_data["surgery"].unique())
    return pcn_data, surgeries

pcn_data, surgery_list = load_pcn_specific_data_and_surgeries(selected_pcn)

# Page selection with radio buttons
page = st.sidebar.radio(
    "Select a Page",
    [
        "PCN Dashboard",
        "Surgery Dashboard",
        "Feedback Classification",
        "Improvement Suggestions",
        "Feedback Timeline",
        "Sentiment Analysis",
        "GPT-4 Summary",
        "Word Cloud",
        "Dataframe",
        "Reports",
        "About",
    ],
    key="page_selector"
)

# Conditionally display the surgery dropdown based on the selected page
if page != "PCN Dashboard" and len(surgery_list) > 0:
    selected_surgery = st.sidebar.selectbox("Select Surgery", surgery_list, key="surgery_selector")
else:
    selected_surgery = None