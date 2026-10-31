#!/usr/bin/env python3
"""streamlitapp4.py — FFT review analysis dashboard for data_v4.csv (JEV pipeline).

Run:  streamlit run streamlitapp4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_medreview.data_cleaning import deduplicate_reviews
from ai_medreview.presentation import (
    format_flag_score,
    format_review_scores,
    review_score_progress_options,
    review_people,
    reviews_for_display,
)

# Import the LLM client for external LLM integration
from ai_medreview.llm_client import ask_llm
from ai_medreview.monthly_report import (
    build_report_context,
    filter_report_reviews,
    generate_monthly_report_pdf,
    normalize_boolean_flag,
    report_selection_key,
    safe_report_filename,
)

# Resolve relative to this file so the app works both locally and on
# Streamlit Cloud (params.DATA_PATH is home-directory based and local-only).
DATA_FILE = str(Path(__file__).parent / "ai_medreview" / "data" / "data_v4.csv")

SENTIMENTS = ["Positive", "Neutral or Mixed", "Negative"]
SENTIMENT_COLORS = ["#2e9e5b", "#e8a13c", "#d64545"]

st.set_page_config(
    page_title="AI_MedReview",
    page_icon=":material/clinical_notes:",
    layout="wide",
)

# Custom CSS for chat avatar colors
st.markdown(
    """
<style>
/* User/Human avatar background color */
.stChatMessage[data-testid="chatAvatarIcon-user"] svg,
.stChatMessage[data-testid="chatAvatarIcon-user"] .stAvatar {
    background-color: #306f83 !important;
}

/* Alternative selector for user avatar */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stAvatar"] {
    background-color: #306f83 !important;
}

/* Target the user message avatar circle */
.stChatMessage[data-testid="stChatMessage"]:has([aria-label="user"]) [data-testid="stAvatar"],
.stChatMessage[aria-label="user"] [data-testid="stAvatar"],
div[data-testid="stChatMessage"] div[data-testid="stAvatar"][aria-label="user"] {
    background-color: #306f83 !important;
}

/* More specific targeting for user avatar */
.st-emotion-cache-1c7y2kd[aria-label="user"],
.st-emotion-cache-1c7y2kd[data-testid="stAvatar"][aria-label="user"] {
    background-color: #306f83 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- Data -------------------------------------------------------------------------------------------


@st.cache_data(ttl="5m", show_spinner="Loading data_v4.csv …")
def load_data() -> pd.DataFrame:
    df = deduplicate_reviews(pd.read_csv(DATA_FILE))
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    df["jev_secondary_topics"] = df["jev_secondary_topics"].fillna("")
    for flag in (
        "jev_clinical_safety_concern",
        "jev_dignity_or_inclusion_concern",
        "jev_leaving_risk",
    ):
        df[flag] = df[flag].map(normalize_boolean_flag)
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
    st.logo('images/symbol-white.png', size='large')

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
    show_review_names = st.toggle(
        "Show people mentioned", value=False, key="show_review_names",
        help="Show names from review_PER beneath reviews and include them in PDF reports.",
    )

    st.caption(
        "Alert = urgency ≥ 2, or a clinical-safety, dignity/inclusion, "
        "or leaving-risk flag. Scores are 0-indexed (actionability/urgency 0–3, "
        "sentiment strength 0–4)."
    )

    st.divider()
    st.subheader("PDF reports")
    report_provider = st.selectbox(
        "PDF AI provider",
        ["FreeLLMAPI", "Groq"],
        key="monthly_report_provider",
        help="Choose the AI service for the monthly PDF summary. Each uses its configured model.",
    )
    st.caption("If report generation fails, try the other provider and generate again.")

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
st.image('images/header_white.png', width=350)
st.caption(f"{len(df):,} of {len(df_all):,} reviews · source: data_v4.csv - FFT Analysis using TypeSafe SystemOne AI (model Jev)")
st.caption("")

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

tab_overview, tab_topics, tab_alerts, tab_browser, tab_chat, tab_report = st.tabs(
    [
        ":material/monitoring: Overview",
        ":material/category: Topics & staff",
        ":material/notification_important: Alerts",
        ":material/search: Review browser",
        ":material/chat: Chat with AI",
        ":material/picture_as_pdf: Monthly report",
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
            reviews_for_display(alerts, show_review_names)[
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
            row_height=100 if show_review_names else None,
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
        reviews_for_display(browser, show_review_names)[
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
        row_height=100 if show_review_names else None,
        selection_mode="single-row",
        column_config={
            "time": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY"),
            "rating_score": st.column_config.NumberColumn("Rating", format="%d ★"),
            "review": st.column_config.TextColumn("Review", width="large"),
            "jev_sentiment": "Sentiment",
            "jev_primary_topic": "Topic",
            "jev_staff_group": "Staff group",
            "jev_actionability": st.column_config.ProgressColumn(
                **review_score_progress_options("Actionability")
            ),
            "jev_urgency": st.column_config.ProgressColumn(
                **review_score_progress_options("Urgency")
            ),
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
            st.markdown(f"#### {row['review']}")
            if show_review_names:
                names = review_people(row.get("review_PER"))
                if names:
                    st.text(f"People mentioned: {', '.join(names)}")
            st.caption(
                f"{row['time']:%d %b %Y %H:%M} · {row['surgery']} · "
                f"rating {row['rating_score']:.0f}/5 · {row['review_len']} words"
            )
            topic = (
                f"Topic: {row['jev_primary_topic']} "
                f"({row['jev_primary_topic_confidence']:.2f})"
            )
            if row["jev_secondary_topics"]:
                topic += f" · Secondary: {row['jev_secondary_topics']}"

            review_summary = pd.DataFrame(
                {
                    "Classification": [
                        f"Sentiment: {row['jev_sentiment']} "
                        f"({row['jev_sentiment_confidence']:.2f})",
                        topic,
                        f"Staff: {row['jev_staff_group']} "
                        f"({row['jev_staff_group_confidence']:.2f})",
                    ],
                    "Scores (0-indexed)": format_review_scores(row),
                    "Flags": [
                        format_flag_score(
                            "Clinical safety",
                            row["jev_clinical_safety_concern"],
                            row["jev_clinical_safety_score"],
                        ),
                        format_flag_score(
                            "Dignity/inclusion",
                            row["jev_dignity_or_inclusion_concern"],
                            row["jev_dignity_or_inclusion_score"],
                        ),
                        format_flag_score(
                            "Leaving risk",
                            row["jev_leaving_risk"],
                            row["jev_leaving_risk_score"],
                        ),
                    ],
                }
            )
            st.dataframe(
                review_summary,
                hide_index=True,
                column_config={
                    column: st.column_config.TextColumn(width="large")
                    for column in review_summary.columns
                },
            )

            jev_fields = [column for column in browser.columns if column.startswith("jev_")]
            with st.expander("Complete Jev output (JSON)"):
                st.json(json.loads(row[jev_fields].to_json()))


# --- Monthly PDF report tab -------------------------------------------------------------------------

with tab_report:
    st.header(":material/picture_as_pdf: Monthly report")
    st.caption(
        "Create a downloadable report for one surgery and one calendar month. "
        "This report uses the complete dataset and is independent of the sidebar filters."
    )

    report_surgeries = sorted(df_all["surgery"].dropna().astype(str).unique())
    if not report_surgeries:
        st.warning("No surgeries are available for reporting.")
    else:
        report_surgery = st.selectbox(
            "Report surgery",
            report_surgeries,
            key="monthly_report_surgery",
        )
        surgery_times = pd.to_datetime(
            df_all.loc[df_all["surgery"] == report_surgery, "time"], errors="coerce"
        ).dropna()
        report_months = sorted(
            surgery_times.dt.to_period("M").unique(), reverse=True
        )

        if not report_months:
            st.warning("This surgery has no valid reporting months.")
        else:
            report_month = st.selectbox(
                "Reporting month",
                report_months,
                format_func=lambda period: period.strftime("%B %Y"),
                key="monthly_report_month",
            )
            active_report_key = (
                report_selection_key(report_surgery, report_month), report_provider,
                show_review_names,
            )

            if st.session_state.get("monthly_report_pdf_key") != active_report_key:
                st.session_state.pop("monthly_report_pdf_bytes", None)
                st.session_state.pop("monthly_report_pdf_filename", None)
                st.session_state.pop("monthly_report_pdf_key", None)

            try:
                report_reviews = filter_report_reviews(
                    df_all, report_surgery, report_month
                )
                report_preview = build_report_context(
                    report_reviews, report_surgery, report_month
                )
            except ValueError as error:
                st.warning(str(error))
                report_reviews = pd.DataFrame()
                report_preview = None

            if report_preview is not None:
                st.subheader("Report preview")
                st.caption(
                    f"{report_preview['total_reviews']:,} reviews for "
                    f"{report_surgery} in {report_preview['month_label']}"
                )
                with st.container(horizontal=True):
                    for rating in report_preview["ratings"]:
                        st.metric(
                            f"{rating['label']} · {rating['stars']} stars",
                            f"{rating['count']:,}",
                            f"{rating['percentage']:.1f}%",
                            border=True,
                        )

                if st.button(
                    "Generate report",
                    type="primary",
                    icon=":material/picture_as_pdf:",
                    key="generate_monthly_report",
                ):
                    st.session_state.pop("monthly_report_pdf_bytes", None)
                    st.session_state.pop("monthly_report_pdf_filename", None)
                    st.session_state.pop("monthly_report_pdf_key", None)
                    try:
                        with st.spinner(f"Analysing feedback with {report_provider} and creating your PDF report..."):
                            pdf_bytes = generate_monthly_report_pdf(
                                df_all, report_surgery, report_month,
                                use_groq=report_provider == "Groq",
                                show_names=show_review_names,
                            )
                        st.session_state["monthly_report_pdf_bytes"] = pdf_bytes
                        st.session_state["monthly_report_pdf_filename"] = (
                            safe_report_filename(report_surgery, report_month)
                        )
                        st.session_state["monthly_report_pdf_key"] = active_report_key
                        st.success("Your monthly report is ready to download.", icon=":material/check:")
                    except Exception as error:
                        st.error(
                            f"The report could not be created using {report_provider}. {error}",
                            icon=":material/cancel:",
                        )

                if (
                    st.session_state.get("monthly_report_pdf_key")
                    == active_report_key
                    and st.session_state.get("monthly_report_pdf_bytes")
                ):
                    st.download_button(
                        "Download PDF",
                        data=st.session_state["monthly_report_pdf_bytes"],
                        file_name=st.session_state["monthly_report_pdf_filename"],
                        mime="application/pdf",
                        icon=":material/download:",
                        type="primary",
                        key="download_monthly_report",
                    )


# --- Chat with AI tab --------------------------------------------------------------------------------

with tab_chat:
    st.header(":material/chat: Chat with AI about the reviews")
    st.caption(
        "Ask questions about the filtered reviews. The AI will consider the selected reviews as context. "
        "Use the slider to control how many reviews are included if the dataset is large."
    )

    # Check if there are any reviews to chat about
    if len(df) == 0:
        st.warning("No reviews match the current filters. Adjust the filters in the sidebar to chat about reviews.")
    else:
        # Slider to control how many reviews to include in context
        max_reviews = len(df)
        # Default to using all reviews if under 50, otherwise use 50
        default_reviews = min(max_reviews, 50)

        num_reviews = st.slider(
            "Number of reviews to include in context",
            min_value=1,
            max_value=max_reviews,
            value=default_reviews,
            step=1,
            help=f"Adjust how many of the {max_reviews} filtered reviews the AI should consider. "
                 "Lower values make responses faster but less comprehensive.",
        )

        # Select the most recent reviews (or all if under the limit)
        context_reviews = df.head(num_reviews)

        # Build context string from reviews
        context_parts = []
        for idx, row in context_reviews.iterrows():
            review_text = row["review"]
            surgery = row["surgery"]
            sentiment = row["jev_sentiment"]
            topic = row["jev_primary_topic"]
            rating = row["rating_score"]

            context_parts.append(
                f"Review from {surgery} (Rating: {rating}/5, Sentiment: {sentiment}, Topic: {topic}):\n"
                f"{review_text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the reviews..."):
            # Add user message to history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build system prompt with context
            system_prompt = (
                "You are a helpful AI assistant analyzing patient feedback from GP surgeries. "
                "You have been provided with a set of patient reviews as context. "
                "Use this context to answer questions about the reviews. "
                "If the answer cannot be determined from the context, say so. "
                "Be concise and focus on insights from the reviews.\n\n"
                f"Context ({num_reviews} reviews):\n{context}"
            )

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = ask_llm(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=0.7,
                        )
                        st.markdown(response)
                        # Add assistant response to history
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info(
                            "Make sure the LLM API is configured correctly in .streamlit/secrets.toml. "
                            "You may need to set LLM_API_KEY and LLM_BASE_URL."
                        )

        # Clear chat button
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()
