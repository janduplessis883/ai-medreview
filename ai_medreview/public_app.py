import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
import streamlit_shadcn_ui as ui
from matplotlib.patches import Patch
from openai import OpenAI
from wordcloud import WordCloud
from groq import Groq

st.set_page_config(page_title="AI-MedReview - FFT Patient Dashboard", layout="wide")
st.logo(
    "images/logo3.png",
    link="https://github.com/janduplessis883/ai-medreview",
    size="large",
)



tab1, tab2, tab3, tab4 = st.tabs(["Find Your Surgery", "Brompton Health PCN", "About AI-MedReview", "Contact Us"])

if tab1:
    st.header("Fined Your Surgery")
    st.sidebar.button("Find Your Surgery", key="find_surgery_button")
elif tab2:
    st.header("Brompton Health PCN")
    st.sidebar.button("Brompton Health PCN", key="brompton_health_button")
elif tab3:
    st.header("About AI-MedReview")

elif tab4:
    st.header("Contact Us")
