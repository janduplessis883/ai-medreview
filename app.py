#!/usr/bin/env python3
"""app.py — AI MedReview4 animated landing page.

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AI MedReview4",
    page_icon=":material/clinical_notes:",
    layout="centered",
)

st.title(":shimmer[AI MedReview4]")
st.caption("Friends & Family Test Reviews analysed with TypeSafe's Jev")

st.link_button(":material/clinical_notes: Launch v4", "https://ai-medreview4.streamlit.app", type="primary")
