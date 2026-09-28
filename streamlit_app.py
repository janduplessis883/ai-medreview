#!/usr/bin/env python3
"""streamlitapp4.py — FFT review analysis dashboard for data_v4.csv (JEV pipeline).

Run:  streamlit run streamlitapp4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve relative to this file so the app works both locally and on
# Streamlit Cloud (params.DATA_PATH is home-directory based and local-only).
DATA_FILE = str(Path(__file__).parent / "ai_medreview" / "data" / "data_v4.csv")

SENTIMENTS = ["Positive", "Neutral or Mixed", "Negative"]
SENTIMENT_COLORS = ["#2e9e5b", "#e8a13c", "#d64545"]

st.set_page_config(
    page_title="FFT review analysis v4",
    page_icon=":material/clinical_notes:",
    layout="wide",
)


# --- Data -------------------------------------------------------------------------------------------


@st.cache_data(ttl="5m", show_spinner="Loading data_v4.csv …")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    df["jev_secondary_topics"] = df["jev_secondary_topics"].fillna("")
    for flag in (
        "jev_clinical_safety_concern",
        "jev_dignity_or_inclusion_concern",
        "jev_leaving_risk",
    ):
        df[flag] = df[flag].fillna(False).astype(bool)
    # Alert = anything a human should look at soon.
    df["is_alert"] = (
        (df["jev_urgency"] >= 2)
        | df["jev_clinical_safety_concern"]
        | df["jev_dignity_or_inclusion_concern"]
        | df["jev_leaving_risk"]
    )
    return df.sort_values("time", ascending=False)


df_all = load_data()

# --- Sidebar filters ---------------------------------------------------------------------------------

with st.sidebar:
    st.header(":material/filter_alt: Filters")

    min_date = df_all["time"].min().date()
    max_date = df_all["time"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date))

    surgeries = sorted(df_all["surgery"].dropna().unique())
    picked_surgeries = st.multiselect("Surgery", surgeries, default=surgeries)

    picked_sentiments = st.pills("Sentiment", SENTIMENTS, default=SENTIMENTS, selection_mode="multi")

    topics = sorted(df_all["jev_primary_topic"].dropna().unique())
    picked_topics = st.multiselect("Primary topic", topics)

    staff_groups = sorted(df_all["jev_staff_group"].dropna().unique())
    picked_staff = st.multiselect("Staff group", staff_groups)

    alerts_only = st.toggle("Alerts only", value=False)

    st.caption(
        "Alert = urgency ≥ 2, or a clinical-safety, dignity/inclusion, "
        "or leaving-risk flag. Scores are 0-indexed (actionability/urgency 0–3, "
        "sentiment strength 0–4)."
    )

mask = df_all["surgery"].isin(picked_surgeries)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    mask &= (df_all["time"].dt.date >= start) & (df_all["time"].dt.date <= end)
if picked_sentiments:
    mask &= df_all["jev_sentiment"].isin(picked_sentiments)
if picked_topics:
    mask &= df_all["jev_primary_topic"].isin(picked_topics)
if picked_staff:
    mask &= df_all["jev_staff_group"].isin(picked_staff)
if alerts_only:
    mask &= df_all["is_alert"]

df = df_all[mask]

# --- Header & KPIs -----------------------------------------------------------------------------------

st.title(":material/clinical_notes: AI MedReview (Analysis with Jev)")
st.caption(f"{len(df):,} of {len(df_all):,} reviews · source: data_v4.csv")

n = max(len(df), 1)
monthly_counts = df.groupby("month").size().sort_index()

with st.container(horizontal=True):
    st.metric(
        "Reviews",
        f"{len(df):,}",
        border=True,
        chart_data=monthly_counts.tolist() or None,
        chart_type="bar",
    )
    st.metric("Avg rating", f"{df['rating_score'].mean():.2f} / 5" if len(df) else "—", border=True)
    st.metric(
        "Positive",
        f"{(df['jev_sentiment'] == 'Positive').sum() / n:.0%}",
        border=True,
    )
    st.metric(
        "Actionable (≥2)",
        f"{(df['jev_actionability'] >= 2).sum():,}",
        border=True,
    )
    st.metric(
        "Alerts",
        f"{df['is_alert'].sum():,}",
        f"{df['is_alert'].sum() / n:.0%} of reviews",
        delta_color="inverse",
        border=True,
    )

# --- Tabs --------------------------------------------------------------------------------------------

tab_overview, tab_topics, tab_alerts, tab_browser = st.tabs(
    [
        ":material/monitoring: Overview",
        ":material/category: Topics & staff",
        ":material/notification_important: Alerts",
        ":material/search: Review browser",
    ]
)

with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Reviews per month")
            monthly = (
                df.groupby(["month", "jev_sentiment"]).size().reset_index(name="reviews")
            )
            st.bar_chart(
                monthly,
                x="month",
                y="reviews",
                color="jev_sentiment",
                x_label="Month",
                y_label="Reviews",
            )

    with col2:
        with st.container(border=True):
            st.subheader("Sentiment")
            sent = df["jev_sentiment"].value_counts().reindex(SENTIMENTS).fillna(0)
            st.bar_chart(sent, horizontal=True, x_label="Reviews")

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border=True):
            st.subheader("Star rating (FFT)")
            rating = df["rating_score"].value_counts().sort_index()
            rating.index = [f"{int(r)} ★" for r in rating.index]
            st.bar_chart(rating, x_label="Rating", y_label="Reviews")

    with col4:
        with st.container(border=True):
            st.subheader("Reviews per surgery")
            per_surgery = df["surgery"].value_counts()
            st.bar_chart(per_surgery, horizontal=True, x_label="Reviews", color="#007185")

with tab_topics:
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Primary topics")
            topic_counts = df["jev_primary_topic"].value_counts()
            st.bar_chart(topic_counts, horizontal=True, x_label="Reviews")

    with col2:
        with st.container(border=True):
            st.subheader("Staff groups mentioned")
            staff_counts = df["jev_staff_group"].value_counts()
            st.bar_chart(staff_counts, horizontal=True, x_label="Reviews", color="#565959")

    with st.container(border=True):
        st.subheader("Topic × sentiment")
        crosstab = (
            df.groupby(["jev_primary_topic", "jev_sentiment"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=SENTIMENTS, fill_value=0)
        )
        crosstab["Total"] = crosstab.sum(axis=1)
        st.dataframe(
            crosstab.sort_values("Total", ascending=False),
            column_config={"Total": st.column_config.NumberColumn(format="%d")},
        )

    with st.container(border=True):
        st.subheader("Low-confidence classifications")
        st.caption(
            "Primary-topic confidence below 0.5 — secondary topics were recorded for these."
        )
        low_conf = df[df["jev_primary_topic_confidence"] < 0.5]
        st.dataframe(
            low_conf[
                [
                    "time",
                    "surgery",
                    "review",
                    "jev_primary_topic",
                    "jev_primary_topic_confidence",
                    "jev_secondary_topics",
                ]
            ],
            hide_index=True,
            column_config={
                "time": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY"),
                "review": st.column_config.TextColumn("Review", width="large"),
                "jev_primary_topic": "Primary topic",
                "jev_primary_topic_confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "jev_secondary_topics": "Secondary topics",
            },
        )

with tab_alerts:
    alerts = df[df["is_alert"]]

    with st.container(horizontal=True):
        st.metric("Urgency ≥ 2", f"{(alerts['jev_urgency'] >= 2).sum():,}", border=True)
        st.metric("Actionability ≥ 2", f"{(alerts['jev_actionability'] >= 2).sum():,}", border=True)
        st.metric(
            "Clinical safety",
            f"{alerts['jev_clinical_safety_concern'].sum():,}",
            border=True,
        )
        st.metric(
            "Dignity / inclusion",
            f"{alerts['jev_dignity_or_inclusion_concern'].sum():,}",
            border=True,
        )
        st.metric("Leaving risk", f"{alerts['jev_leaving_risk'].sum():,}", border=True)

    with st.container(border=True):
        st.subheader("Alerts per surgery")
        alert_surgery = alerts["surgery"].value_counts()
        if len(alert_surgery):
            st.bar_chart(alert_surgery, horizontal=True, x_label="Alerts")
        else:
            st.info("No alerts in the current filter selection.")

    with st.container(border=True):
        st.subheader("Alert reviews")
        st.dataframe(
            alerts[
                [
                    "time",
                    "surgery",
                    "review",
                    "jev_sentiment",
                    "jev_primary_topic",
                    "jev_urgency",
                    "jev_actionability",
                    "jev_clinical_safety_concern",
                    "jev_dignity_or_inclusion_concern",
                    "jev_leaving_risk",
                ]
            ],
            hide_index=True,
            column_config={
                "time": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY"),
                "review": st.column_config.TextColumn("Review", width="large"),
                "jev_sentiment": "Sentiment",
                "jev_primary_topic": "Topic",
                "jev_urgency": st.column_config.NumberColumn("Urgency", format="%d"),
                "jev_actionability": st.column_config.NumberColumn(
                    "Actionability", format="%d"
                ),
                "jev_clinical_safety_concern": st.column_config.CheckboxColumn("Safety"),
                "jev_dignity_or_inclusion_concern": st.column_config.CheckboxColumn(
                    "Dignity/inclusion"
                ),
                "jev_leaving_risk": st.column_config.CheckboxColumn("Leaving risk"),
            },
        )

with tab_browser:
    search = st.text_input(
        "Search reviews",
        placeholder="e.g. parking, blood test, rude …",
        label_visibility="collapsed",
        icon=":material/search:",
    )
    browser = df
    if search.strip():
        browser = df[df["review"].str.contains(search.strip(), case=False, na=False)]
    st.caption(f"{len(browser):,} matching reviews")

    event = st.dataframe(
        browser[
            [
                "time",
                "surgery",
                "rating_score",
                "review",
                "jev_sentiment",
                "jev_primary_topic",
                "jev_staff_group",
                "jev_actionability",
                "jev_urgency",
                "jev_dignity_or_inclusion_concern",
                "jev_clinical_safety_concern",
                "jev_leaving_risk",
            ]
        ],
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "time": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY"),
            "rating_score": st.column_config.NumberColumn("Rating", format="%d ★"),
            "review": st.column_config.TextColumn("Review", width="large"),
            "jev_sentiment": "Sentiment",
            "jev_primary_topic": "Topic",
            "jev_staff_group": "Staff group",
            "jev_actionability": st.column_config.NumberColumn("Action", format="%d"),
            "jev_urgency": st.column_config.NumberColumn("Urgency", format="%d"),
            "jev_dignity_or_inclusion_concern": st.column_config.CheckboxColumn(
                "Inclusion / dignity"
            ),
            "jev_clinical_safety_concern": st.column_config.CheckboxColumn("Safety"),
            "jev_leaving_risk": st.column_config.CheckboxColumn("Leaving risk"),
        },
    )

    if event.selection.rows:
        row = browser.iloc[event.selection.rows[0]]
        with st.container(border=True):
            st.subheader("Review detail")
            st.markdown(f"> {row['review']}")
            st.caption(
                f"{row['time']:%d %b %Y %H:%M} · {row['surgery']} · "
                f"rating {row['rating_score']:.0f}/5 · {row['review_len']} words"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Classification**")
                st.write(f"Sentiment: {row['jev_sentiment']} ({row['jev_sentiment_confidence']:.2f})")
                st.write(f"Topic: {row['jev_primary_topic']} ({row['jev_primary_topic_confidence']:.2f})")
                if row["jev_secondary_topics"]:
                    st.write(f"Secondary: {row['jev_secondary_topics']}")
                st.write(f"Staff: {row['jev_staff_group']} ({row['jev_staff_group_confidence']:.2f})")
            with col2:
                st.markdown("**Scores** (0-indexed)")
                st.write(f"Sentiment strength: {row['jev_sentiment_strength']:.0f} / 4")
                st.write(f"Actionability: {row['jev_actionability']:.0f} / 3")
                st.write(f"Urgency: {row['jev_urgency']:.0f} / 3")
            with col3:
                st.markdown("**Flags**")
                st.write(f"Clinical safety: {'⚠️' if row['jev_clinical_safety_concern'] else '—'} ({row['jev_clinical_safety_score']:.2f})")
                st.write(f"Dignity/inclusion: {'⚠️' if row['jev_dignity_or_inclusion_concern'] else '—'} ({row['jev_dignity_or_inclusion_score']:.2f})")
                st.write(f"Leaving risk: {'⚠️' if row['jev_leaving_risk'] else '—'} ({row['jev_leaving_risk_score']:.2f})")

            jev_fields = [column for column in browser.columns if column.startswith("jev_")]
            with st.expander("Complete Jev output (JSON)"):
                st.json(json.loads(row[jev_fields].to_json()))
