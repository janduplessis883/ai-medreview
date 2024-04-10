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
import os 


client = OpenAI()

from utils import *
from reports import *

st.set_page_config(page_title="AI MedReview v2")

# Styling with HTML
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
<div class="gradient-text">AI MedReview v2</div>
"""
st.sidebar.markdown(html, unsafe_allow_html=True)
st.sidebar.image("images/transparent2.png")

# Define a list of PCNs
pcn_names = ["Demo-PCN", "Brompton-Health-PCN"]

# PCN selection in sidebar
selected_pcn = st.sidebar.selectbox("Select a PCN:", pcn_names, key="pcn_selector")

# Function to load data
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_parquet("ai_medreview/data/data.parquet")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df

data = load_data()

# Function to load PCN specific data
@st.cache_data(ttl=3600)
def load_pcn_data(pcn_name):
    df = load_data()
    return df[df["pcn"] == pcn_name]

# Load PCN specific data
pcn_data = load_pcn_data(selected_pcn)

# Function to get surgeries by PCN
@st.cache_data(ttl=3600)
def get_surgeries_by_pcn(data, pcn):
    filtered_data = data[data["pcn"] == pcn]
    surgeries = sorted(filtered_data["surgery"].unique())
    return surgeries

# Page selection
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
)

# Surgery selection and data filtering
if page not in ["PCN Dashboard", "About"]:
    surgery_list = get_surgeries_by_pcn(pcn_data, selected_pcn)
    if len(surgery_list) > 0:
        selected_surgery = st.sidebar.selectbox("Select Surgery", surgery_list)
        surgery_data = pcn_data[pcn_data["surgery"] == selected_surgery]

        if not surgery_data.empty:
            start_date = surgery_data["time"].dt.date.min()
            end_date = surgery_data["time"].dt.date.max()

            if start_date == end_date:
                start_date -= timedelta(days=1)

            try:
                selected_date_range = st.slider(
                    f"{selected_pcn} - **{selected_surgery}**",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    format="MM/DD/YYYY",
                )
            except ValueError as e:
                st.error(f"Cannot display slider: {str(e)}")
            
            # Define the function to filter data based on selected date range
            @st.cache_data(ttl=3600)
            def filter_data_by_date_range(data, date_range):
                return data[
                    (data["time"].dt.date >= date_range[0]) & (data["time"].dt.date <= date_range[1])
                ]

            filtered_data = filter_data_by_date_range(surgery_data, selected_date_range)
else:
    selected_surgery = None
    filtered_data = None

# Content Start ========================================================================================== Content Start

# -- PCN Dashboard --------------------------------------------------------------------------------------- PCN Dashboard
if page == "PCN Dashboard":
    st.markdown(f"# ![dashboard](https://img.icons8.com/pastel-glyph/64/laptop-metrics--v1.png) {selected_pcn} ")
    st.markdown(
        """Accumulating and interpreting the **pooled patient feedback data** from member practices.
"""
    )
    st.write("")
    tab_selector = ui.tabs(
        options=[
            "PCN Rating",
            "PCN Responses",
            "Sentiment A.",
            "Topic A.",
            "Surgery Ratings",
            "Surgery Responses",
        ],
        default_value="PCN Rating",
        key="tab3",
    )

    if tab_selector == "PCN Responses":  # ---------------------------------------------------------- PCN Responses ----
        st.subheader("PCN Response Rate")
        st.markdown("**Dialy FFT Responses**")

        daily_count = pcn_data.resample("D", on="time").size()
        daily_count_df = daily_count.reset_index()
        daily_count_df.columns = ["Date", "Daily Count"]
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(
            data=daily_count_df,
            x="Date",
            y="Daily Count",
            color="#558387",
            linewidth=2,
        )

        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.xaxis.grid(False)

        # Customizing the x-axis labels for better readability
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax_title = ax.set_title(
            "Daily FFT Responses - Brompton Health PCN", loc="right"
        )  # loc parameter aligns the title
        ax_title.set_position(
            (1, 1)
        )  # Adjust these values to align your title as needed
        plt.xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("---")
        st.markdown("**Monthly FFT Responses**")
        with st.container(border=False):
            # Monthly Totals Plot

            monthly_count_filtered = pcn_data.resample("M", on="time").size()
            monthly_count_filtered_df = monthly_count_filtered.reset_index()
            monthly_count_filtered_df.columns = ["Month", "Monthly Count"]
            monthly_count_filtered_df["Month"] = monthly_count_filtered_df[
                "Month"
            ].dt.date

            # Create the figure and the bar plot
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(
                data=monthly_count_filtered_df,
                x="Month",
                y="Monthly Count",
                color="#aabd3b",
                edgecolor="black",
                linewidth=0.5,
            )

            # Set grid, spines and annotations as before
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
            ax.xaxis.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Annotate bars with the height (monthly count)
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 10),
                    textcoords="offset points",
                )

            # Set title to the right
            ax_title = ax.set_title(
                "Monthly FFT Responses - Brompton Health PCN", loc="right"
            )
            ax_title.set_position((1.02, 1))  # Adjust title position
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            # Remove xlabel as it's redundant with the dates
            plt.xlabel("")

            # Apply tight layout and display plot
            plt.tight_layout()
            st.pyplot(fig)

    elif tab_selector == "Surgery Ratings":  # --------------------------------------------------- Surgery Ratings -----
        st.subheader("Surgery Ratings")

        with st.container(border=False):

            # alldata_date_range = filter_data_by_date_range(data, selected_date_range)
            

            pivot_data = pcn_data.pivot_table(
                index="surgery", columns="rating", aggfunc="size", fill_value=0
            )
            total_responses_per_surgery = pivot_data.sum(axis=1)

            # Compute the percentage of each rating category for each surgery
            percentage_pivot_data = (
                pivot_data.div(total_responses_per_surgery, axis=0) * 100
            )
            # Define the desired column order based on the rating categories
            column_order = [
                "Extremely likely",
                "Likely",
                "Neither likely nor unlikely",
                "Unlikely",
                "Extremely unlikely",
                "Don't know",
            ]

            # Reorder the columns in the percentage pivot data
            ordered_percentage_pivot_data = percentage_pivot_data[column_order]

            # Create the heatmap with the ordered columns
            plt.figure(figsize=(12, 9))
            ordered_percentage_heatmap = sns.heatmap(
                ordered_percentage_pivot_data,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                linewidths=0.5,
            )
            plt.title("% Heatmap of Surgery Ratings", fontsize=16)
            plt.ylabel("")
            plt.xlabel("Rating (%)", fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Display the ordered percentage heatmap
            st.pyplot(plt)

    elif tab_selector == "Sentiment A.":   # --------------------------------------------------- Sentiment Analysis ----
        st.subheader("Sentiment Analysis")
        # Assuming 'data' is already defined and processed
        # Define labels and colors outside since they are the same for both plots
        labels = ["Negative", "Neutral", "Positive"]
        colors = ["#ae4f4d", "#eeeadb", "#7495a8"]  # Order adjusted to match labels
        explode = (0, 0, 0)  # No slice exploded

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # First pie chart - Cum Sentiment - Feedback
        sentiment_totals_feedback = pcn_data.groupby("sentiment_free_text")[
            "sentiment_score_free_text"
        ].sum()
        ax1.pie(
            sentiment_totals_feedback,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax1.axis(
            "equal"
        )  # Equal aspect ratio ensures that pie is drawn as a circle.
        centre_circle = plt.Circle((0, 0), 0.50, fc="white")
        ax1.add_artist(centre_circle)
        ax1.set_title("Cum Sentiment - Feedback")
 
        # Second pie chart - Cum Sentiment - Improvement Suggestions
        sentiment_totals_improvement = pcn_data.groupby("sentiment_do_better")[
            "sentiment_score_do_better"
        ].sum()
        ax2.pie(
            sentiment_totals_improvement,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax2.axis("equal")
        centre_circle = plt.Circle((0, 0), 0.50, fc="white")
        ax2.add_artist(centre_circle)
        ax2.set_title("Cum Sentiment - Improvement Sugg.")

        # Display the subplot
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("Average Monthly **Sentiment Analysis Score** - Feedback")
        pcn_data["time"] = pd.to_datetime(pcn_data["time"])
        pcn_data.set_index("time", inplace=True)

        # Assuming filtered_data is your DataFrame and 'sentiment_score' is the column with the scores
        # Also assuming that 'time' column has been converted to datetime and set as the index

        # Calculate the standard deviation for each month and sentiment
        monthly_sentiment_std = (
            pcn_data.groupby("sentiment_free_text")
            .resample("M")["sentiment_score_free_text"]
            .std()
            .unstack(level=0)
        )

        # Fill NaN values
        monthly_sentiment_std.fillna(0, inplace=True)

        # Calculate the mean sentiment scores for each month and sentiment, if not already done
        monthly_sentiment_means_adjusted = (
            pcn_data.groupby("sentiment_free_text")
            .resample("M")["sentiment_score_free_text"]
            .mean()
            .unstack(level=0)
        )

        # Fill NaN values for the means
        monthly_sentiment_means_adjusted.fillna(0, inplace=True)

        # Define colors for each sentiment
        colors = {
            "negative": "#ae4f4d",
            "neutral": "#edeadc",
            "positive": "#7b94a6",
        }

        # Creating the plot for monthly sentiment scores
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot each sentiment mean with standard deviation as shaded area
        for sentiment in monthly_sentiment_means_adjusted.columns:
            # Plot the mean sentiment scores
            ax.plot(
                monthly_sentiment_means_adjusted.index,
                monthly_sentiment_means_adjusted[sentiment],
                label=sentiment.capitalize(),
                marker="o",
                color=colors[sentiment],
                linewidth=2,
            )

            # Add the standard deviation with a shaded area
            ax.fill_between(
                monthly_sentiment_means_adjusted.index,
                monthly_sentiment_means_adjusted[sentiment]
                - monthly_sentiment_std[sentiment],
                monthly_sentiment_means_adjusted[sentiment]
                + monthly_sentiment_std[sentiment],
                color=colors[sentiment],
                alpha=0.2,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        plt.title(
            "Monthly Sentiment Score Averages with Standard Deviation - Feedback",
            fontsize=16,
        )
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Average Sentiment Score", fontsize=12)
        plt.legend(title="Sentiment")
        plt.tight_layout()

        # Show the plot (or use st.pyplot(plt) if you are using Streamlit)
        st.pyplot(plt)
        st.markdown("---")
        st.markdown(
            "Average Monthly **Sentiment Analysis Score** - Improvement Suggestions"
        )
        monthly_sentiment_std = (
            pcn_data.groupby("sentiment_do_better")
            .resample("M")["sentiment_score_do_better"]
            .std()
            .unstack(level=0)
        )

        # Fill NaN values
        monthly_sentiment_std.fillna(0, inplace=True)

        # Calculate the mean sentiment scores for each month and sentiment, if not already done
        monthly_sentiment_means_adjusted = (
            pcn_data.groupby("sentiment_do_better")
            .resample("M")["sentiment_score_do_better"]
            .mean()
            .unstack(level=0)
        )

        # Fill NaN values for the means
        monthly_sentiment_means_adjusted.fillna(0, inplace=True)

        # Define colors for each sentiment
        colors = {
            "negative": "#ae4f4d",
            "neutral": "#edeadc",
            "positive": "#7b94a6",
        }

        # Creating the plot for monthly sentiment scores
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot each sentiment mean with standard deviation as shaded area
        for sentiment in monthly_sentiment_means_adjusted.columns:
            # Plot the mean sentiment scores
            ax.plot(
                monthly_sentiment_means_adjusted.index,
                monthly_sentiment_means_adjusted[sentiment],
                label=sentiment.capitalize(),
                marker="o",
                color=colors[sentiment],
                linewidth=2,
            )

            # Add the standard deviation with a shaded area
            ax.fill_between(
                monthly_sentiment_means_adjusted.index,
                monthly_sentiment_means_adjusted[sentiment]
                - monthly_sentiment_std[sentiment],
                monthly_sentiment_means_adjusted[sentiment]
                + monthly_sentiment_std[sentiment],
                color=colors[sentiment],
                alpha=0.2,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        plt.title(
            "Monthly Sentiment Score Averages with Standard Deviation - Improvement Sugg.",
            fontsize=16,
        )
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Average Sentiment Score", fontsize=12)
        plt.legend(title="Sentiment")
        plt.tight_layout()

        # Show the plot (or use st.pyplot(plt) if you are using Streamlit)
        st.pyplot(plt)

    elif tab_selector == "Surgery Responses":  # ----------------------------------------------- Surgery Responses------
        st.subheader("Surgery Responses")
        with st.container(border=False):
      
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(y="surgery", data=pcn_data, color="#59646b")
            for p in ax.patches:
                width = p.get_width()
                try:
                    y = p.get_y() + p.get_height() / 2
                    ax.text(
                        width + 1,
                        y,
                        f"{int(width)}",
                        va="center",
                        fontsize=10,
                    )
                except ValueError:
                    pass
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
            ax.yaxis.grid(False)
            plt.xlabel("Count")
            plt.ylabel("")
            plt.title("Total FFT Responses by Surgery", loc="right")
            plt.tight_layout()
            st.pyplot(plt)

        st.markdown("---")
     
        data_sorted = pcn_data.sort_values("time")

        # Group by 'surgery' and 'time', then calculate the cumulative count
        data_sorted["cumulative_count"] = (
            data_sorted.groupby("surgery").cumcount() + 1
        )

        # Pivot the table to have surgeries as columns and their cumulative counts as values
        data_pivot = data_sorted.pivot_table(
            index="time",
            columns="surgery",
            values="cumulative_count",
            aggfunc="first",
        )

        # Forward fill the NaN values to maintain the cumulative nature
        data_pivot_filled = data_pivot.fillna(method="ffill").fillna(0)

        # Plotting
        fig = go.Figure()

        for column in data_pivot_filled.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_pivot_filled.index,
                    y=data_pivot_filled[column],
                    name=column,
                    mode="lines",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title="Cumulative FFT Responses Over Time for Each Surgery",
            xaxis=dict(
                title="Time",
                gridcolor="#888888",
                gridwidth=0.5,
                showgrid=True,
                showline=True,  # Make the bottom spine visible
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            yaxis=dict(
                title="Cumulative FFT Responses",
                gridcolor="#888888",
                gridwidth=0.5,
                showgrid=True,
                showline=True,  # Make the left spine visible
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            plot_bgcolor="white",
            legend=dict(
                title="Surgery", x=1.05, y=1, xanchor="left", yanchor="top"
            ),
            width=750,  # Set the width to 750 pixels
            height=750,  # Set the height to 850 pixels
        )

        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig)

    elif (
        tab_selector == "PCN Rating"
    ):  # ------------------------------------------------------------------------------------- PCN Rating -------------
        st.subheader("PCN Rating")
        st.markdown("**Average Monthly Rating**")
        
        try:
            with st.container(border=False):
                # Convert 'time' to datetime and extract the date
                pcn_data["date"] = pd.to_datetime(pcn_data["time"]).dt.date

                # Group by the new 'date' column and calculate the mean 'rating_score' for each day
                daily_mean_rating = (
                    pcn_data.groupby("date")["rating_score"].mean().reset_index()
                )
                # Ensure the 'date' column is in datetime format for resampling
                daily_mean_rating["date"] = pd.to_datetime(daily_mean_rating["date"])

                # Set the 'date' column as the index
                daily_mean_rating.set_index("date", inplace=True)

                # Resample the data by week and calculate the mean 'rating_score' for each week
                weekly_mean_rating = (
                    daily_mean_rating["rating_score"].resample("M").mean().reset_index()
                )

                # Create a seaborn line plot for weekly mean rating scores
                fig, ax = plt.subplots(figsize=(12, 5))
                weekly_lineplot = sns.lineplot(
                    x="date",
                    y="rating_score",
                    data=weekly_mean_rating,
                    color="#d2b570",
                    linewidth=4,
                )

                for index, row in weekly_mean_rating.iterrows():
                    ax.annotate(
                        f'{row["rating_score"]:.2f}',
                        (row["date"], row["rating_score"]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=12,  # Adjust this value as needed
                    )

                plt.xlabel("Month")
                plt.ylabel("Mean Rating Score")
                plt.xticks(rotation=45)
                ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
                ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                # Set title to the right
                ax_title = ax.set_title(
                    "Mean Monthly Rating Score - Brompton Health PCN", loc="right"
                )
                plt.tight_layout()
                st.pyplot(plt)
        except KeyError as e:
            st.warning(f"Error plotting: {e}")
            
        st.markdown("---")
        st.markdown("**Rating Count**")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(data=pcn_data, x="rating_score", color="#616884")

        total = len(pcn_data)  # Total number of observations

        # Annotate each bar
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)  # Calculate percentage
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='bottom')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        plt.title("Rating Score Count")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.tight_layout()

        # Display the figure in Streamlit
        st.pyplot(fig)

    elif (
        tab_selector == "Topic A."
    ):  # ------------------------------------------------------------------------------------------ Topic Analysis ----
        st.subheader("Topic Analysis")
        toggle = ui.switch(
            default_checked=False, label="Time Series", key="switch_dash_pcn"
        )
        if toggle:

            radio_options = [
                {"label": "All", "value": "all", "id": "r7"},
                {"label": "Negative", "value": "neg", "id": "r8"},
                {"label": "Neutral + Positive", "value": "pos", "id": "r9"},
            ]
            radio_value = ui.radio_group(
                options=radio_options, default_value="all", key="radio3"
            )

            if radio_value == "pos":
                pcn_data = pcn_data[
                    (
                        (pcn_data["sentiment_free_text"] == "neutral")
                        | (pcn_data["sentiment_free_text"] == "positive")
                    )
                ]
            elif radio_value == "neg":

                pcn_data = pcn_data[(pcn_data["sentiment_free_text"] == "negative")]
            else:
                pcn_pcn_data = pcn_data[(pcn_data['pcn'] == selected_pcn)]

            pcn_data["time"] = pd.to_datetime(pcn_data["time"])
            # Setting the 'time' column as the index
            pcn_data.set_index("time", inplace=True)

            # Grouping by month and 'feedback_labels' and then counting the occurrences
            # Converting the time index to a period index for monthly resampling
            pcn_data.index = pcn_data.index.to_period("M")
  
            monthly_feedback_counts = (
                pcn_data.groupby([pcn_data.index, "feedback_labels"])
                .size()
                .unstack(fill_value=0)
            )

            # Converting the period index back to a timestamp for compatibility with Plotly
            monthly_feedback_counts.index = (
                monthly_feedback_counts.index.to_timestamp()
            )

            # Plotting the data using Plotly Express
            fig1 = px.line(
                monthly_feedback_counts,
                x=monthly_feedback_counts.index,
                y=monthly_feedback_counts.columns,
                title="Time Series of Feedback Labels (Monthly Aggregation)",
                labels={
                    "x": "Month",
                    "value": "Count of Feedback Labels",
                    "variable": "Feedback Labels",
                },
            )

            # Updating the layout
            fig1.update_layout(
                width=900,
                legend=dict(
                    title="Feedback Labels",
                    x=1.05,
                    y=1,
                    xanchor="left",
                    yanchor="top",
                ),
                xaxis=dict(
                    gridcolor="lightgray",
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                ),
                yaxis=dict(
                    gridcolor="lightgray",
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                ),
                plot_bgcolor="white",
            )

            # Displaying the plot in Streamlit
            st.plotly_chart(fig1)

            st.markdown("---")

            # Grouping by month and 'improvement_labels' and then counting the occurrences
   
            monthly_improvement_counts = (
                pcn_data.groupby([pcn_data.index, "improvement_labels"])
                .size()
                .unstack(fill_value=0)
            )

            # Converting the period index back to a timestamp for compatibility with Plotly
            monthly_improvement_counts.index = (
                monthly_improvement_counts.index.to_timestamp()
            )

            # Plotting the data for 'improvement_labels' using Plotly Express
            fig2 = px.line(
                monthly_improvement_counts,
                x=monthly_improvement_counts.index,
                y=monthly_improvement_counts.columns,
                title="Time Series of Improvement Labels (Monthly Aggregation)",
                labels={
                    "x": "Month",
                    "value": "Count of Improvement Labels",
                    "variable": "Improvement Labels",
                },
            )

            # Updating the layout
            fig2.update_layout(
                width=900,
                legend=dict(
                    title="Improvement Labels",
                    x=1.05,
                    y=1,
                    xanchor="left",
                    yanchor="top",
                ),
                xaxis=dict(
                    gridcolor="lightgray",
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                ),
                yaxis=dict(
                    gridcolor="lightgray",
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                ),
                plot_bgcolor="white",
            )

            # Displaying the plot in Streamlit
            st.plotly_chart(fig2)
        else:
            palette = {
                "positive": "#2e5f77",
                "negative": "#d7662a",
                "neutral": "#d7d8d7",
            }
            hue_order = ["negative", "neutral", "positive"]

            # Create a cross-tabulation of feedback labels and sentiment categories

            crosstab = pd.crosstab(
                pcn_data["feedback_labels"], pcn_data["sentiment_free_text"]
            )
            crosstab = crosstab.reindex(columns=hue_order)

            # Sort the feedback labels by total counts in descending order
            crosstab_sorted = crosstab.sort_values(by=hue_order, ascending=False)

            # Create a horizontal stacked bar chart using Plotly

            fig = go.Figure(
                data=[
                    go.Bar(
                        y=crosstab_sorted.index,
                        x=crosstab_sorted[sentiment],
                        name=sentiment,
                        orientation="h",
                        marker=dict(color=palette[sentiment]),
                    )
                    for sentiment in hue_order
                ],
                layout=go.Layout(
                    title="Feedback Classification",
                    xaxis=dict(
                        title="Counts",
                        gridcolor="#888888",
                        gridwidth=0.5,
                        showgrid=True,
                    ),
                    yaxis=dict(title="Feedback Labels", showgrid=False),
                    barmode="stack",
                    plot_bgcolor="white",
                    showlegend=True,
                    legend=dict(x=1.0, y=1.0),
                    width=750,  # Set the width to 1200 pixels (12 inches)
                    height=550,  # Set the height to 800 pixels (8 inches)
                ),
            )

            # Streamlit function to display Plotly figures
            st.plotly_chart(fig)

            st.markdown("---")

            palette = {
                "positive": "#90bfca",
                "negative": "#f3aa49",
                "neutral": "#ece7e3",
            }
            hue_order = ["negative", "neutral", "positive"]

            # Create a cross-tabulation of feedback labels and sentiment categories
 
            crosstab = pd.crosstab(
                pcn_data["improvement_labels"], pcn_data["sentiment_do_better"]
            )
            crosstab = crosstab.reindex(columns=hue_order)

            # Sort the feedback labels by total counts in descending order
            crosstab_sorted = crosstab.sort_values(by=hue_order, ascending=False)

            # Create a horizontal stacked bar chart using Plotly
            fig = go.Figure(
                data=[
                    go.Bar(
                        y=crosstab_sorted.index,
                        x=crosstab_sorted[sentiment],
                        name=sentiment,
                        orientation="h",
                        marker=dict(color=palette[sentiment]),
                    )
                    for sentiment in hue_order
                ],
                layout=go.Layout(
                    title="Improvement Suggestion Classification",
                    xaxis=dict(
                        title="Counts",
                        gridcolor="#888888",
                        gridwidth=0.5,
                        showgrid=True,
                    ),
                    yaxis=dict(
                        title="Improvement Suggestion Labels", showgrid=False
                    ),
                    barmode="stack",
                    plot_bgcolor="white",
                    showlegend=True,
                    legend=dict(x=1.0, y=1.0),
                    width=750,  # Set the width to 1200 pixels (12 inches)
                    height=550,  # Set the height to 800 pixels (8 inches)
                ),
            )

            # Streamlit function to display Plotly figures
            st.plotly_chart(fig)




# -- Surgery Dashboard ------------------------------------------------------------------------------- Surgery Dashboard
elif page == "Surgery Dashboard":
    st.markdown(f"# ![dashboard](https://img.icons8.com/pastel-glyph/64/laptop-metrics--v1.png) {selected_surgery}")
    st.write("")
    surgery_tab_selector = ui.tabs(
        options=["Surgery Rating", "Surgery Responses", "Feedback Length", "Total Word Count", "Missing Data"],
        default_value="Surgery Rating",
        key="tab4",
    )

    if surgery_tab_selector == "Surgery Rating":

        try:
            # Resample to get monthly average rating
            monthly_avg = filtered_data.resample("M", on="time")["rating_score"].mean()

            # Reset index to make 'time' a column again
            monthly_avg_df = monthly_avg.reset_index()

            # Create a line plot
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(
                x="time",
                y="rating_score",
                data=monthly_avg_df,
                ax=ax,
                linewidth=4,
                color="#e5c17e",
            )

            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
            ax.xaxis.grid(False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            # Customize the plot - remove the top, right, and left spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Rotate x-axis labels
            # plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Annotate the line graph
            for index, row in monthly_avg_df.iterrows():
                ax.annotate(
                    f'{row["rating_score"]:.2f}',
                    (row["time"], row["rating_score"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=12,  # Adjust this value as needed
                )

            # Add labels and title
            plt.xlabel("")
            plt.ylabel("Average Rating")
            plt.tight_layout()
            ax_title = ax.set_title(
                "Mean Monthly Rating", loc="right"
            )  # loc parameter aligns the title
            ax_title.set_position(
                (1, 1)
            )  # Adjust these values to align your title as needed
            # Display the plot in Streamlit
            st.pyplot(fig)

        except:
            st.info("No rating available for this date range.")

        st.write("---")
        st.markdown(
            f"**Total Responses** for selected time period: **{filtered_data.shape[0]}**"
        )
        order = [
            "Extremely likely",
            "Likely",
            "Neither likely nor unlikely",
            "Unlikely",
            "Extremely unlikely",
            "Don't know",
        ]

        palette = {
            "Extremely likely": "#112f45",
            "Likely": "#4d9cb9",
            "Neither likely nor unlikely": "#9bc8e3",
            "Unlikely": "#f4ba41",
            "Extremely unlikely": "#ec8b33",
            "Don't know": "#ae4f4d",
        }

        # Set the figure size (width, height) in inches

        plt.figure(figsize=(12, 5))

        # Create the countplot
        sns.countplot(data=filtered_data, y="rating", order=order, palette=palette)
        ax = plt.gca()

        # Remove y-axis labels
        ax.set_yticklabels([])

        # Create a custom legend
        from matplotlib.patches import Patch

        legend_patches = [
            Patch(color=color, label=label) for label, color in palette.items()
        ]
        plt.legend(
            handles=legend_patches,
            title="Rating Categories",
            bbox_to_anchor=(1.05, 1),
            loc="best",
        )

        # Iterate through the rectangles (bars) of the plot for width annotations
        for p in ax.patches:
            width = p.get_width()
            offset = width * 0.02
            try:
                y = p.get_y() + p.get_height() / 2
                ax.text(
                    width + offset,
                    y,
                    f"{int(width)} / {round((int(width)/filtered_data.shape[0])*100, 1)}%",
                    va="center",
                    fontsize=10,
                )
            except ValueError:
                pass

        # Adjust plot appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(False)
        plt.xlabel("Count")
        plt.ylabel("Rating")
        plt.tight_layout()
        st.pyplot(plt)

    elif surgery_tab_selector == "Surgery Responses":
        cols = st.columns(2)
        with cols[0]:
            ui.metric_card(
                title="Total Responses",
                content=f"{filtered_data.shape[0]}",
                description=f"Since {start_date}.",
                key="total",
            )
        with cols[1]:
            pass

        # Plotting the line plot
        # Add more content to col2 as needed
        daily_count = filtered_data.resample("D", on="time").size()
        daily_count_df = daily_count.reset_index()
        daily_count_df.columns = ["Date", "Daily Count"]
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(
            data=daily_count_df, x="Date", y="Daily Count", color="#558387", linewidth=2
        )

        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.xaxis.grid(False)

        # Customizing the x-axis labels for better readability
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax_title = ax.set_title(
            "Daily FFT Responses", loc="right"
        )  # loc parameter aligns the title
        ax_title.set_position(
            (1, 1)
        )  # Adjust these values to align your title as needed
        plt.xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("---")
        # Monthly Totals Plot
        monthly_count_filtered = filtered_data.resample("M", on="time").size()
        monthly_count_filtered_df = monthly_count_filtered.reset_index()
        monthly_count_filtered_df.columns = ["Month", "Monthly Count"]
        monthly_count_filtered_df["Month"] = monthly_count_filtered_df["Month"].dt.date

        # Create the figure and the bar plot
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=monthly_count_filtered_df,
            x="Month",
            y="Monthly Count",
            color="#aabd3b",
            edgecolor="black",
            linewidth=0.5,
        )

        # Set grid, spines and annotations as before
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Annotate bars with the height (monthly count)
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )

        # Set title to the right
        ax_title = ax.set_title("Monthly FFT Responses", loc="right")
        ax_title.set_position((1.02, 1))  # Adjust title position
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # Remove xlabel as it's redundant with the dates
        plt.xlabel("")

        # Apply tight layout and display plot
        plt.tight_layout()
        st.pyplot(fig)

    elif surgery_tab_selector == "Feedback Length":
        fig, ax = plt.subplots(
            1, 2, figsize=(12, 6)
        )  # figsize can be adjusted as needed

        # Plot the first histogram on the first subplot
        sns.histplot(filtered_data["free_text_len"], ax=ax[0], color="#708695", bins=25)
        ax[0].yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax[0].xaxis.grid(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["left"].set_visible(False)
        ax[0].set_title(
            "Distribution of Free Text Feedback Word Count"
        )  # Optional title for the first plot

        # Plot the second histogram on the second subplot
        sns.histplot(filtered_data["do_better_len"], ax=ax[1], color="#985e5b", bins=25)
        ax[1].yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax[1].xaxis.grid(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["left"].set_visible(False)
        ax[1].set_title(
            "Distribution of Imporvement Suggestion Word Count"
        )  # Optional title for the second plot

        # Show the plots next to each other
        plt.tight_layout()
        st.pyplot(plt)
        
    elif surgery_tab_selector == "Total Word Count":
        filtered_data["prompt"] = (
            filtered_data["free_text"].fillna("")
            + " "
            + filtered_data["do_better"].fillna("")
        )
    
        # Step 2: Drop NaN values (now unnecessary as we handled NaNs during concatenation)
        filtered_data.dropna(subset=["prompt"], inplace=True)

        # Step 3: Join all rows to form one large corpus of words
        text = " ".join(filtered_data["prompt"])
        words = text.split()
        word_count = len(words)
        
        cols = st.columns(2)
        with cols[0]:
            ui.metric_card(
                title="Total Word Count",
                content=f"{word_count}",
                description=f"",
                key="totalwords",
            )
        with cols[1]:
            pass
        
    elif surgery_tab_selector == "Missing Data":   
        plt.figure(figsize=(12, 5))
        sns.heatmap(filtered_data.isnull(), cbar=False, cmap='Blues', yticklabels=False)
        plt.title('Heatmap of Missing Values in Data')
        st.pyplot(plt)
# -- Feedback Classification ------------------------------------------------------------------- Feedback Classification
elif page == "Feedback Classification":
    st.markdown("# ![Feedback](https://img.icons8.com/ios/50/thumbs-up-down.png) Feedback Classification")
    st.markdown("Responses to **FFT Q1**: Please tell us why you feel this way?")

    toggle = ui.switch(default_checked=False, label="Time Series", key="switch_dash")
    if toggle:

        radio_options = [
            {"label": "All", "value": "all", "id": "r1"},
            {"label": "Negative", "value": "neg", "id": "r2"},
            {"label": "Neutral + Positive", "value": "pos", "id": "r3"},
        ]
        radio_value = ui.radio_group(
            options=radio_options, default_value="all", key="radio1"
        )

        if radio_value == "pos":
            filtered_data = filtered_data[
                (
                    (filtered_data["sentiment_free_text"] == "neutral")
                    | (filtered_data["sentiment_free_text"] == "positive")
                )
            ]
        elif radio_value == "neg":
            filtered_data = filtered_data[
                (filtered_data["sentiment_free_text"] == "negative")
            ]
        else:
            filtered_data = filtered_data

        filtered_data["time"] = pd.to_datetime(filtered_data["time"])

        # Setting the 'time' column as the index
        filtered_data.set_index("time", inplace=True)

        # Grouping by month and 'feedback_labels' and then counting the occurrences
        # Converting the time index to a period index for monthly resampling
        filtered_data.index = filtered_data.index.to_period("M")
        monthly_feedback_counts = (
            filtered_data.groupby([filtered_data.index, "feedback_labels"])
            .size()
            .unstack(fill_value=0)
        )

        # Converting the period index back to a timestamp for compatibility with Seaborn
        monthly_feedback_counts.index = monthly_feedback_counts.index.to_timestamp()

        fig = px.line(
            monthly_feedback_counts,
            x=monthly_feedback_counts.index,
            y=monthly_feedback_counts.columns,
            title="Time Series of Feedback Topics (Monthly Aggregation)",
            labels={
                "x": "Month",
                "value": "Count of Feedback Labels",
                "variable": "Feedback Labels",
            },
        )

        # Updating the layout with increased width
        fig.update_layout(
            width=900,  # Adjust the width value as needed
            legend=dict(
                title="Feedback Labels", x=1.05, y=1, xanchor="left", yanchor="top"
            ),
            xaxis=dict(
                gridcolor="lightgray", showline=True, linewidth=1, linecolor="black"
            ),
            yaxis=dict(
                gridcolor="lightgray", showline=True, linewidth=1, linecolor="black"
            ),
            plot_bgcolor="white",
        )

        # Displaying the plot in Streamlit
        st.plotly_chart(fig)

        st.markdown("---")
        # View Patient Feedback
        st.subheader("View Patient Feedback")
        class_list = list(filtered_data["feedback_labels"].unique())
        cleaned_class_list = [x for x in class_list if not pd.isna(x)]
        selected_ratings = st.multiselect(
            "Select Feedback Categories:",
            cleaned_class_list,
            help="Feedback in orange have a negative sentiment.",
        )

        # Filter the data based on the selected classifications
        filtered_classes = filtered_data[
            filtered_data["feedback_labels"].isin(selected_ratings)
        ]

        if not selected_ratings:
            ui.badges(
                badge_list=[("Please select at least one classification.", "outline")],
                class_name="flex gap-2",
                key="badges10",
            )
        else:
            for rating in selected_ratings:
                specific_class = filtered_classes[
                    filtered_classes["feedback_labels"] == rating
                ]
                st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
                for index, row in specific_class.iterrows():
                    text = row["free_text"]
                    sentiment = row["sentiment_free_text"]
                    if sentiment == "negative":
                        text_color = "orange"
                    elif sentiment == "neutral":
                        text_color = "gray"
                    else:
                        text_color = "black"

                    if str(text).lower() != "nan":
                        st.markdown(f"- :{text_color}[{str(text)}] ")

    else:
        palette = {"positive": "#2e5f77", "negative": "#d7662a", "neutral": "#d7d8d7"}
        hue_order = ["negative", "neutral", "positive"]

        # Create a cross-tabulation of feedback labels and sentiment categories
        crosstab = pd.crosstab(
            filtered_data["feedback_labels"], filtered_data["sentiment_free_text"]
        )
        crosstab = crosstab.reindex(columns=hue_order)

        # Sort the feedback labels by total counts in descending order
        crosstab_sorted = crosstab.sort_values(by=hue_order, ascending=False)

        # Create a horizontal stacked bar chart using Plotly
        fig = go.Figure(
            data=[
                go.Bar(
                    y=crosstab_sorted.index,
                    x=crosstab_sorted[sentiment],
                    name=sentiment,
                    orientation="h",
                    marker=dict(color=palette[sentiment]),
                )
                for sentiment in hue_order
            ],
            layout=go.Layout(
                title="Feedback Classification",
                xaxis=dict(
                    title="Counts", gridcolor="#888888", gridwidth=0.5, showgrid=True
                ),
                yaxis=dict(title="Feedback Labels", showgrid=False),
                barmode="stack",
                plot_bgcolor="white",
                showlegend=True,
                legend=dict(x=1.0, y=1.0),
                width=750,  # Set the width to 1200 pixels (12 inches)
                height=550,  # Set the height to 800 pixels (8 inches)
            ),
        )

        # Streamlit function to display Plotly figures
        st.plotly_chart(fig)

        st.markdown("---")

        # View Patient Feedback
        st.subheader("View Patient Feedback")
        class_list = list(filtered_data["feedback_labels"].unique())
        cleaned_class_list = [x for x in class_list if not pd.isna(x)]
        selected_ratings = st.multiselect(
            "Select Feedback Categories:",
            cleaned_class_list,
            help="Feedback in orange have a negative sentiment.",
        )

        # Filter the data based on the selected classifications
        filtered_classes = filtered_data[
            filtered_data["feedback_labels"].isin(selected_ratings)
        ]

        if not selected_ratings:
            ui.badges(
                badge_list=[("Please select at least one classification.", "outline")],
                class_name="flex gap-2",
                key="badges10",
            )
        else:
            for rating in selected_ratings:
                specific_class = filtered_classes[
                    filtered_classes["feedback_labels"] == rating
                ]
                st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
                for index, row in specific_class.iterrows():
                    text = row["free_text"]
                    sentiment = row["sentiment_free_text"]
                    if sentiment == "negative":
                        text_color = "orange"
                    elif sentiment == "neutral":
                        text_color = "gray"
                    else:
                        text_color = "black"

                    if str(text).lower() != "nan":
                        st.markdown(f"- :{text_color}[{str(text)}] ")

# -- Improvement Suggestions ------------------------------------------------------------------- Improvement Suggestions
elif page == "Improvement Suggestions":
    st.markdown("# ![Improvement](https://img.icons8.com/ios/50/improvement.png) Improvement Suggestions")
    st.markdown(
        "Responses to **FFT Q2**: Is there anything that would have made your experience better?"
    )

    toggle = ui.switch(default_checked=False, label="Time Series", key="switch_dash")
    if toggle:

        radio_options = [
            {"label": "All", "value": "all", "id": "r4"},
            {"label": "Negative", "value": "neg", "id": "r5"},
            {"label": "Neutral + Positive", "value": "pos", "id": "r6"},
        ]
        radio_value = ui.radio_group(
            options=radio_options, default_value="all", key="radio2"
        )

        if radio_value == "pos":
            filtered_data = filtered_data[
                (
                    (filtered_data["sentiment_do_better"] == "neutral")
                    | (filtered_data["sentiment_do_better"] == "positive")
                )
            ]
        elif radio_value == "neg":
            filtered_data = filtered_data[
                (filtered_data["sentiment_do_better"] == "negative")
            ]
        else:
            filtered_data = filtered_data

        filtered_data["time"] = pd.to_datetime(filtered_data["time"])

        # Setting the 'time' column as the index
        filtered_data.set_index("time", inplace=True)

        # Grouping by month and 'feedback_labels' and then counting the occurrences
        # Converting the time index to a period index for monthly resampling
        filtered_data.index = filtered_data.index.to_period("M")
        monthly_feedback_counts = (
            filtered_data.groupby([filtered_data.index, "improvement_labels"])
            .size()
            .unstack(fill_value=0)
        )

        # Converting the period index back to a timestamp for compatibility with Seaborn
        monthly_feedback_counts.index = monthly_feedback_counts.index.to_timestamp()

        fig = px.line(
            monthly_feedback_counts,
            x=monthly_feedback_counts.index,
            y=monthly_feedback_counts.columns,
            title="Time Series of Improvement Topics (Monthly Aggregation)",
            labels={
                "x": "Month",
                "value": "Count of Improvement Labels",
                "variable": "Improvement Labels",
            },
        )

        # Updating the layout
        fig.update_layout(
            width=900,
            legend=dict(
                title="Improvement Labels", x=1.05, y=1, xanchor="left", yanchor="top"
            ),
            xaxis=dict(
                gridcolor="lightgray", showline=True, linewidth=1, linecolor="black"
            ),
            yaxis=dict(
                gridcolor="lightgray", showline=True, linewidth=1, linecolor="black"
            ),
            plot_bgcolor="white",
        )

        # Displaying the plot in Streamlit
        st.plotly_chart(fig)

        st.markdown("---")
        improvement_data = filtered_data[
            (filtered_data["improvement_labels"] != "No Improvement Suggestion")
        ]
        # Calculate value counts
        label_counts = improvement_data["improvement_labels"].value_counts(
            ascending=False
        )  # Use ascending=True to match the order in your image

        # Convert the Series to a DataFrame
        label_counts_df = label_counts.reset_index()
        label_counts_df.columns = ["Improvement Labels", "Counts"]

        st.subheader("View Patient Improvement Suggestions")
        improvement_list = [label for label in label_counts_df["Improvement Labels"]]

        selected_ratings = st.multiselect(
            "Select Categories:",
            improvement_list,
            help="Improvement Suggestions in orange have a negative sentiment.",
        )

        # Filter the data based on the selected classifications
        filtered_classes = improvement_data[
            improvement_data["improvement_labels"].isin(selected_ratings)
        ]

        if not selected_ratings:
            ui.badges(
                badge_list=[("Please select at least one classification.", "outline")],
                class_name="flex gap-2",
                key="badges10",
            )
        else:
            for rating in selected_ratings:
                specific_class = filtered_classes[
                    filtered_classes["improvement_labels"] == rating
                ]
                st.subheader(
                    f"{str(rating).capitalize()} ({str(specific_class.shape[0])})"
                )
                for index, row in specific_class.iterrows():
                    text = row["do_better"]
                    text = text.replace("[PERSON]", "PERSON")
                    sentiment = row["sentiment_do_better"]
                    if sentiment == "positive" or sentiment == "neutral":
                        text_color = "black"
                    else:
                        text_color = "orange"

                    if str(text).lower() != "nan":
                        st.markdown(f"- :{text_color}[{str(text)}] ")

    else:
        improvement_data = filtered_data[
            (filtered_data["improvement_labels"] != "No Improvement Suggestion")
        ]
        # Calculate value counts
        label_counts = improvement_data["improvement_labels"].value_counts(
            ascending=False
        )  # Use ascending=True to match the order in your image

        # Convert the Series to a DataFrame
        label_counts_df = label_counts.reset_index()
        label_counts_df.columns = ["Improvement Labels", "Counts"]

        # Define the palette conditionally based on the category names
        palette = {"positive": "#90bfca", "negative": "#f3aa49", "neutral": "#ece7e3"}
        hue_order = ["negative", "neutral", "positive"]

        # Create a cross-tabulation of feedback labels and sentiment categories
        crosstab = pd.crosstab(
            filtered_data["improvement_labels"], filtered_data["sentiment_do_better"]
        )
        crosstab = crosstab.reindex(columns=hue_order)

        # Sort the feedback labels by total counts in descending order
        crosstab_sorted = crosstab.sort_values(by=hue_order, ascending=False)

        # Create a horizontal stacked bar chart using Plotly
        fig = go.Figure(
            data=[
                go.Bar(
                    y=crosstab_sorted.index,
                    x=crosstab_sorted[sentiment],
                    name=sentiment,
                    orientation="h",
                    marker=dict(color=palette[sentiment]),
                )
                for sentiment in hue_order
            ],
            layout=go.Layout(
                title="Improvement Suggestion Classification",
                xaxis=dict(
                    title="Counts", gridcolor="#888888", gridwidth=0.5, showgrid=True
                ),
                yaxis=dict(title="Improvement Suggestion Labels", showgrid=False),
                barmode="stack",
                plot_bgcolor="white",
                showlegend=True,
                legend=dict(x=1.0, y=1.0),
                width=750,  # Set the width to 1200 pixels (12 inches)
                height=550,  # Set the height to 800 pixels (8 inches)
            ),
        )

        # Streamlit function to display Plotly figures
        st.plotly_chart(fig)

        st.markdown("---")

        st.subheader("View Patient Improvement Suggestions")
        improvement_list = [label for label in label_counts_df["Improvement Labels"]]

        selected_ratings = st.multiselect(
            "Select Categories:",
            improvement_list,
            help="Improvement Suggestions in orange have a negative sentiment.",
        )

        # Filter the data based on the selected classifications
        filtered_classes = improvement_data[
            improvement_data["improvement_labels"].isin(selected_ratings)
        ]

        if not selected_ratings:
            ui.badges(
                badge_list=[("Please select at least one classification.", "outline")],
                class_name="flex gap-2",
                key="badges10",
            )
        else:
            for rating in selected_ratings:
                specific_class = filtered_classes[
                    filtered_classes["improvement_labels"] == rating
                ]
                st.subheader(
                    f"{str(rating).capitalize()} ({str(specific_class.shape[0])})"
                )
                for index, row in specific_class.iterrows():
                    text = row["do_better"]
                    text = text.replace("[PERSON]", "PERSON")
                    sentiment = row["sentiment_do_better"]
                    if sentiment == "positive" or sentiment == "neutral":
                        text_color = "black"
                    else:
                        text_color = "orange"

                    if str(text).lower() != "nan":
                        st.markdown(f"- :{text_color}[{str(text)}] ")

# -- Feedback Timeline ------------------------------------------------------------------------------- Feedback Timeline
elif page == "Feedback Timeline":
    st.markdown("# ![Timeline](https://img.icons8.com/dotty/80/timeline.png) Feedback Timeline")
    
    daily_count = filtered_data.resample("D", on="time").size()
    daily_count_df = daily_count.reset_index()
    daily_count_df.columns = ["Date", "Daily Count"]
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(
        data=daily_count_df, x="Date", y="Daily Count", color="#558387", linewidth=2
    )

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.xaxis.grid(False)

    # Customizing the x-axis labels for better readability
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax_title = ax.set_title(
        "Daily FFT Responses", loc="right"
    )  # loc parameter aligns the title
    ax_title.set_position((1, 1))  # Adjust these values to align your title as needed
    plt.xlabel("")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")
    st.markdown(f"Showing **{filtered_data.shape[0]}** FFT Responses")

    with st.container(height=500, border=True):
        for _, row in filtered_data.iterrows():
            free_text = row["free_text"]
            do_better = row["do_better"]
            feedback_labels = row['feedback_labels']
            imp_labels = row['improvement_labels']
            time = row["time"]
            rating = row["rating"]

            with st.chat_message("user"):
                st.markdown(f"**{rating}** `{time}`")
                if str(free_text) not in ["nan"]:
                    st.markdown("🗣️ " + str(free_text))
                    st.markdown(f"`{feedback_labels}`")
                    if str(do_better) not in ["nan"]:
                        st.markdown("💡 " + str(do_better))
                        st.markdown(f"`{imp_labels}`")
    

# -- Sentiment Analysis ----------------------------------------------------------------------------- Sentiment Analysis
elif page == "Sentiment Analysis":
    st.markdown("# ![Sentiment Analysis](https://img.icons8.com/ios/50/like--v1.png) Sentiment Analysis")

    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )

    # React to the toggle's state
    if toggle:
        st.markdown(
            """1. **Scatter Plot (Top Plot)**:
This plot compares patient feedback sentiment scores with feedback rating scores. On the x-axis, we have the rating score, which likely corresponds to a numerical score given by the patient in their feedback, and on the y-axis, we have the sentiment score, which is derived from sentiment analysis of the textual feedback provided by the patient. Each point represents a piece of feedback, categorized as 'positive', 'neutral', or 'negative' sentiment, depicted by different markers. The scatter plot shows a clear positive correlation between the sentiment score and the feedback rating score, especially visible with the concentration of 'positive' sentiment scores at the higher end of the rating score scale, suggesting that more positive text feedback corresponds to higher numerical ratings.
2. **Histogram with a Density Curve (Bottom Left - NEGATIVE Sentiment)**:
This histogram displays the distribution of sentiment scores specifically for negative sentiment feedback. The x-axis represents the sentiment score (presumably on a scale from 0 to 1), and the y-axis represents the count of feedback instances within each score range. The bars show the frequency of feedback at different levels of negative sentiment, and the curve overlaid on the histogram provides a smooth estimate of the distribution. The distribution suggests that most negative feedback has a sentiment score around 0.7 to 0.8.
3. **Histogram with a Density Curve (Bottom Right - POSITIVE Sentiment)**:
Similar to the negative sentiment histogram, this one represents the distribution of sentiment scores for positive sentiment feedback. Here, we see a right-skewed distribution with a significant concentration of feedback in the higher sentiment score range, particularly close to 1.0. This indicates that the positive feedback is often associated with high sentiment scores, which is consistent with the expected outcome of sentiment analysis.
4. **View Patient Feedback (Multi-Select Input)**:
Select Patient feedback to review, this page only displays feedback that on Sentiment Analysis scored **NEGATIVE > Selected Value (using slider)**, indicating negative feedback despite rating given by the patient. It is very important to review feedback with a high NEG sentiment analysis. In this section both feedback and Improvement Suggestions are displayed to review them in context, together with the automated category assigned by our machine learning model."""
        )
    try:
        # Assuming 'data' is already defined and processed
        # Define labels and colors outside since they are the same for both plots
        labels = ["Negative", "Neutral", "Positive"]
        colors = ["#ae4f4d", "#eeeadb", "#7495a8"]  # Order adjusted to match labels
        explode = (0, 0, 0)  # No slice exploded

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # First pie chart - Cum Sentiment - Feedback
        sentiment_totals_feedback = filtered_data.groupby("sentiment_free_text")[
            "sentiment_score_free_text"
        ].sum()
        ax1.pie(
            sentiment_totals_feedback,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        centre_circle = plt.Circle((0, 0), 0.50, fc="white")
        ax1.add_artist(centre_circle)
        ax1.set_title("Cum Sentiment - Feedback")

        # Second pie chart - Cum Sentiment - Improvement Suggestions
        sentiment_totals_improvement = filtered_data.groupby("sentiment_do_better")[
            "sentiment_score_do_better"
        ].sum()
        ax2.pie(
            sentiment_totals_improvement,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax2.axis("equal")
        centre_circle = plt.Circle((0, 0), 0.50, fc="white")
        ax2.add_artist(centre_circle)
        ax2.set_title("Cum Sentiment - Improvement Sugg")

        # Display the subplot
        st.pyplot(fig)

    except ValueError:
        ui.badges(
            badge_list=[("Not able to display Cumm Sentiment Totals.", "outline")],
            class_name="flex gap-2",
            key="badges1_waring_warning",
        )

    st.markdown("---")

    filtered_data["time"] = pd.to_datetime(filtered_data["time"])
    filtered_data.set_index("time", inplace=True)

    # Assuming filtered_data is your DataFrame and 'sentiment_score' is the column with the scores
    # Also assuming that 'time' column has been converted to datetime and set as the index

    # Calculate the standard deviation for each month and sentiment
    monthly_sentiment_std = (
        filtered_data.groupby("sentiment_free_text")
        .resample("M")["sentiment_score_free_text"]
        .std()
        .unstack(level=0)
    )

    # Fill NaN values
    monthly_sentiment_std.fillna(0, inplace=True)

    # Calculate the mean sentiment scores for each month and sentiment, if not already done
    monthly_sentiment_means_adjusted = (
        filtered_data.groupby("sentiment_free_text")
        .resample("M")["sentiment_score_free_text"]
        .mean()
        .unstack(level=0)
    )

    # Fill NaN values for the means
    monthly_sentiment_means_adjusted.fillna(0, inplace=True)

    # Define colors for each sentiment
    colors = {"negative": "#ae4f4d", "neutral": "#edeadc", "positive": "#7b94a6"}

    # Creating the plot for monthly sentiment scores
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot each sentiment mean with standard deviation as shaded area
    for sentiment in monthly_sentiment_means_adjusted.columns:
        # Plot the mean sentiment scores
        ax.plot(
            monthly_sentiment_means_adjusted.index,
            monthly_sentiment_means_adjusted[sentiment],
            label=sentiment.capitalize(),
            marker="o",
            color=colors[sentiment],
            linewidth=2,
        )

        # Add the standard deviation with a shaded area
        ax.fill_between(
            monthly_sentiment_means_adjusted.index,
            monthly_sentiment_means_adjusted[sentiment]
            - monthly_sentiment_std[sentiment],
            monthly_sentiment_means_adjusted[sentiment]
            + monthly_sentiment_std[sentiment],
            color=colors[sentiment],
            alpha=0.2,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    plt.title(
        "Monthly Sentiment Score Averages with Standard Deviation - Feedback",
        fontsize=16,
    )
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Sentiment Score", fontsize=12)
    plt.legend(title="Sentiment")
    plt.tight_layout()

    # Show the plot (or use st.pyplot(plt) if you are using Streamlit)
    st.pyplot(plt)
    st.markdown("---")

    monthly_sentiment_std = (
        filtered_data.groupby("sentiment_do_better")
        .resample("M")["sentiment_score_do_better"]
        .std()
        .unstack(level=0)
    )

    # Fill NaN values
    monthly_sentiment_std.fillna(0, inplace=True)

    # Calculate the mean sentiment scores for each month and sentiment, if not already done
    monthly_sentiment_means_adjusted = (
        filtered_data.groupby("sentiment_do_better")
        .resample("M")["sentiment_score_do_better"]
        .mean()
        .unstack(level=0)
    )

    # Fill NaN values for the means
    monthly_sentiment_means_adjusted.fillna(0, inplace=True)

    # Define colors for each sentiment
    colors = {"negative": "#ae4f4d", "neutral": "#edeadc", "positive": "#7b94a6"}

    # Creating the plot for monthly sentiment scores
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot each sentiment mean with standard deviation as shaded area
    for sentiment in monthly_sentiment_means_adjusted.columns:
        # Plot the mean sentiment scores
        ax.plot(
            monthly_sentiment_means_adjusted.index,
            monthly_sentiment_means_adjusted[sentiment],
            label=sentiment.capitalize(),
            marker="o",
            color=colors[sentiment],
            linewidth=2,
        )

        # Add the standard deviation with a shaded area
        ax.fill_between(
            monthly_sentiment_means_adjusted.index,
            monthly_sentiment_means_adjusted[sentiment]
            - monthly_sentiment_std[sentiment],
            monthly_sentiment_means_adjusted[sentiment]
            + monthly_sentiment_std[sentiment],
            color=colors[sentiment],
            alpha=0.2,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    plt.title(
        "Monthly Sentiment Score Averages with Standard Deviation - Improvement Sugg.",
        fontsize=16,
    )
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Sentiment Score", fontsize=12)
    plt.legend(title="Sentiment")
    plt.tight_layout()

    # Show the plot (or use st.pyplot(plt) if you are using Streamlit)
    st.pyplot(plt)

    st.markdown("---")

    st.markdown(f"### FFT Feedback with a `NEGATIVE` Sentiment Score.")

    toggle = ui.switch(
        default_checked=False, label="Show last 30 days only.", key="switch_dash_neg"
    )

    # React to the toggle's state
    if toggle:
        # Convert 'time' column to datetime
        negative = filtered_data.reset_index("time")
        negative["time"] = pd.to_datetime(negative["time"])
        neg = negative[
            (negative["sentiment_free_text"] == "negative")
            | (negative["sentiment_do_better"] == "negative")
        ]
        neg["time"] = pd.to_datetime(neg["time"])

        # Calculate the date 30 days ago from today
        latest_date = datetime.now().date()
        thirty_days_ago = latest_date - timedelta(days=30)

        # Filter the DataFrame based on the 'time' column
        neg = neg[neg["time"].dt.date > thirty_days_ago]
        neg1 = neg[neg["sentiment_free_text"] == "negative"]
        neg2 = neg[neg["sentiment_do_better"] == "negative"]

    else:
        negative = filtered_data.reset_index("time")
        negative["time"] = pd.to_datetime(negative["time"])
        neg = negative[
            (negative["sentiment_free_text"] == "negative")
            | (negative["sentiment_do_better"] == "negative")
        ]
        neg1 = neg[neg["sentiment_free_text"] == "negative"]
        neg2 = neg[neg["sentiment_do_better"] == "negative"]

    sentiment_tab_selector = ui.tabs(
        options=["Feedback Responses", "Improvement Suggestions"],
        default_value="Feedback Responses",
        key="tab45",
    )
    if sentiment_tab_selector == "Feedback Responses":

        if neg.shape[0] > 0:
            fig, ax = plt.subplots(figsize=(12, 2.5))
            sns.histplot(neg1["sentiment_score_free_text"], color="#ae4f4d", kde=True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            plt.xlabel("Sentiment Score")
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
            plt.tight_layout()
            st.pyplot(plt)
            st.write("")
            st.markdown(f"Showing **{neg1.shape[0]}** Feedback Responses.")
            with st.container(height=500, border=True):
                for _, row in neg1.iterrows():
                    free_text = row["free_text"]
                    cat = row["feedback_labels"]
                    time_ = row["time"]
                    rating = row["rating"]
                    score = row["sentiment_score_free_text"]
                    sentiment = row["sentiment_free_text"]

                    with st.chat_message("user"):
                        st.markdown(f"**{rating}** `{time_}`")

                        if str(free_text) not in ["nan"]:
                            st.markdown("🗣️ " + str(free_text))
                        st.markdown(f"`{sentiment} {score}` `{cat}`")

    elif sentiment_tab_selector == "Improvement Suggestions":
        if neg.shape[0] > 0:
            fig, ax = plt.subplots(figsize=(12, 2.5))
            sns.histplot(neg2["sentiment_score_do_better"], color="#d7662a", kde=True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            plt.xlabel("Sentiment Score")
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
            plt.tight_layout()
            st.pyplot(plt)
            st.write("")
            st.markdown(f"Showing **{neg2.shape[0]}** Improvement Suggestions.")
            with st.container(height=500, border=True):
                for _, row in neg2.iterrows():
                    do_better = row["do_better"]
                    cat = row["improvement_labels"]
                    time_ = row["time"]
                    rating = row["rating"]
                    score = row["sentiment_score_do_better"]
                    sentiment = row["sentiment_do_better"]

                    with st.chat_message("user"):
                        st.markdown(f"**{rating}** `{time_}`")

                        if str(do_better) not in ["nan"]:
                            st.markdown("💡 " + str(do_better))
                        st.markdown(f"`{sentiment} {score}` `{cat}`")
                        

# -- Word Cloud --------------------------------------------------------------------------------------------- Word Cloud
elif page == "Word Cloud":
    st.markdown("# ![Word CLoud](https://img.icons8.com/ios/50/cloud-refresh--v1.png) Word Cloud")
    try:
        toggle = ui.switch(
            default_checked=False, label="Explain this page.", key="switch_dash"
        )
        if toggle:
            st.markdown(
                """1. The **Feedback Word Cloud**:
    From response to FFT Q1: Please tell us why you feel this way? 
    A **word cloud** is a visual representation of text data where the size of each word indicates its frequency or importance. In a word cloud, commonly occurring words are usually displayed in larger fonts or bolder colors, while less frequent words appear smaller. This makes it easy to perceive the most prominent terms within a large body of text at a glance.
    In the context of patient feedback, a word cloud can be especially useful to quickly identify the key themes or subjects that are most talked about by patients. For example, if many patients mention terms like "waiting times" or "friendly staff," these words will stand out in the word cloud, indicating areas that are notably good or need improvement.  
2. The **Improvement Suggestions Word Cloud** is a creative and intuitive representation of the feedback collected from patients through the Friends and Family Test (FFT). When patients are asked, "Is there anything that would have made your experience better?" their responses provide invaluable insights into how healthcare services can be enhanced."""
            )
        st.subheader("Feedback Word Cloud")
        text = " ".join(filtered_data["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Blues").generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    except:
        ui.badges(
            badge_list=[("No Feedback available for this date range.", "outline")],
            class_name="flex gap-2",
            key="badges10",
        )
    try:
        st.subheader("Improvement Suggestions Word Cloud")

        text2 = " ".join(filtered_data["do_better"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Reds").generate(text2)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    except:
        ui.badges(
            badge_list=[
                ("No improvement suggestions available for this date range.", "outline")
            ],
            class_name="flex gap-2",
            key="badges11",
        )


# -- GPT4 Summary ----------------------------------------------------------------------------------------- GPT4 Summary
elif page == "GPT-4 Summary":
    st.markdown("# ![GPT-4](https://img.icons8.com/ios/50/chatgpt.png) GPT-4 Summary")

    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """**What This Page Offers:**

**Automated Summaries**: Leveraging OpenAI's cutting-edge ChatGPT-4, we transform the Friends & Family Test feedback and improvement suggestions into concise, actionable insights.  
**Time-Specific Insights**: Select the period that matters to you. Whether it's a week, a month, or a custom range, our tool distills feedback relevant to your chosen timeframe.  
**Efficient Meeting Preparations**: Prepare for meetings with ease. Our summaries provide a clear overview of patient feedback, enabling you to log actions and decisions swiftly and accurately.  

**How It Works**:

1. **Select Your Time Period**: Choose the dates that you want to analyze.  
2. **AI-Powered Summarization**: ChatGPT-4 reads through the feedback and suggestions, understanding the nuances and key points.  
3. **Receive Your Summary**: Get a well-structured, comprehensive summary that highlights the core sentiments and suggestions from your patients."""
        )
    st.markdown("**Follow the steps below to summarise free-text with GPT4.**")

    def call_chatgpt_api(text):
        # Example OpenAI Python library request
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in summarizing Friends & Family Test feedback for GP surgeries.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the folloing text, making sure to highlight any trend in feedback and improvement suggestions: \n{text}",
                },
            ],
        )

        output = completion.choices[0].message.content
        return output

    def send_webhook(user_name, selected_surgery, word_count):
        """
        Send a webhook POST request with the provided data.

        Parameters:
        user_name (str): The user's name.
        selected_surgery (str): The selected surgery.
        word_count (int): The word count.
        """
        webhook_url = "https://eodj7kiwqrtobj4.m.pipedream.net"
        data = {
            "user_name": user_name,
            "surgery": selected_surgery,
            "word_count": word_count,
        }
        response = requests.post(webhook_url, json=data)
        return response

    filtered_data["prompt"] = (
        filtered_data["free_text"].fillna("")
        + " "
        + filtered_data["do_better"].fillna("")
    )

    # Step 2: Drop NaN values (now unnecessary as we handled NaNs during concatenation)
    filtered_data.dropna(subset=["prompt"], inplace=True)

    # Step 3: Join all rows to form one large corpus of words
    text = " ".join(filtered_data["prompt"])
    words = text.split()
    word_count = len(words)
    text = " ".join(words)

    # Display the text container
    with st.container(height=200, border=True):
        st.write(text)

    # Display and handle the word count badge
    if 0 < word_count <= 6400:
        ui.badges(
            badge_list=[
                (f"Word count: {word_count}", "outline"),
                ("✔️ Summarise with GPT-4 API", "secondary"),
            ],
            class_name="flex gap-2",
            key="badges10",
        )

        # Get user's name input
        name_input_value = ui.input(
            default_value="",
            type="text",
            placeholder="Enter your name, to continue...",
            key="gpt_name_input",
        )

        if name_input_value:
            st.markdown(f"You entered: **{name_input_value}**")
            st.session_state["user_name"] = (
                name_input_value  # Save the user name to the session state
            )

            # Handle the 'Submit' button click
            if ui.button("Submit", key="clk_btn"):
                st.session_state["submitted"] = (
                    True  # Mark as submitted in the session state
                )

                # Send the webhook only once upon submission
                if not st.session_state.get("webhook_sent", False):
                    send_webhook(name_input_value, selected_surgery, word_count)
                    st.session_state["webhook_sent"] = (
                        True  # Avoid sending the webhook again on rerun
                    )
                    st.write("Webhook sent successfully!")

        # Conditionally show the 'Summarize with GPT-4' button based on the submission state
        if st.session_state.get("submitted", False):
            if ui.button(
                text="Summarize with GPT-4",
                key="styled_btn_tailwind",
                className="bg-orange-500 text-white",
            ):
                # Once the 'Summarize with GPT-4' is clicked, fetch and display the summary
                summary = call_chatgpt_api(text)
                with st.container(border=True):
                    st.subheader("Friends & Family Test Feedback Summary")
                    st.markdown(f"**{selected_surgery}**")
                    st.markdown(
                        f"Date range: {selected_date_range[0]} - {selected_date_range[1]}"
                    )
                    st.markdown("---")
                    st.write(summary)

        # Display the logo and entered name (if any)
        st.image("images/openailogo.png")

    elif word_count == 0:
        ui.badges(
            badge_list=[
                (f"Word Count: {word_count}", "destructive"),
                ("⤬ Nothing to summarise.", "secondary"),
            ],
            class_name="flex gap-2",
            key="badges11",
        )

    else:
        ui.badges(
            badge_list=[
                (f"Word Count: {word_count}", "destructive"),
                (
                    "⤬ The input text surpasses the maximum limit allowed for GPT-4 API.",
                    "secondary",
                ),
            ],
            class_name="flex gap-2",
            key="badges11",
        )
        ui.badges(
            badge_list=[
                (f"Option 1:", "default"),
                ("⤬ Adjust the date range to reduce input text size.", "outline"),
            ],
            class_name="flex gap-2",
            key="badges12",
        )
        ui.badges(
            badge_list=[
                (f"Option 2:", "default"),
                (
                    "⤬ Download feedback as .txt file - upload to ChatGPT & prompt to summarise.",
                    "outline",
                ),
            ],
            class_name="flex gap-2",
            key="badges13",
        )
        st.download_button(
            "Download feedback as .txt",
            data=text,
            file_name=f"FFT_Feedback-{selected_surgery}-{selected_date_range[0]} to {selected_date_range[1]}.txt",
            help="Upload this Plain Text file to ChtGPT and prompt to summarize.",
        )

# -- Dataframe ----------------------------------------------------------------------------------------------- Dataframe
elif page == "Dataframe":
    st.markdown("# ![Dataframe](https://img.icons8.com/ios/50/new-spreadsheet.png) Dataframe")

    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """**Dataframe**:
A dataFrame as a big, organized table full of raw data. It's like a virtual spreadsheet with many rows and columns, where every row represents a single record, and each column stands for a particular variable. If your DataFrame contains all the raw data, it means that it hasn't been processed or filtered - it's the data in its original form as collected.

Each column in a DataFrame has a name, which you can use to locate data more easily. Columns can contain all sorts of data types, including numbers, strings, and dates, and each one typically holds the same kind of data throughout. For instance, one column might hold ages while another lists names, and yet another records dates of visits.

Rows are labeled with an Index, which you can think of as the address of the data. This makes finding specific records simple and fast."""
        )
    st.write("The data below is filtered based on the date range selected above.")

    # Display the filtered DataFrame
    st.dataframe(filtered_data)
    
   
    
# -- Reports --------------------------------------------------------------------------------------------------- Reports
elif page == "Reports":
    st.markdown("# ![Reports](https://img.icons8.com/ios/50/graph-report.png) Reports")
    st.write("")
    ui.badges(badge_list=[("NEW", "destructive"), ("coming soon...", "outline")], class_name="flex gap-2", key="badges_soon")
    
    
    # S Tips Plot --------------------------------------------------------------------------------------- Tips Plot ----
    
    # Path for the generated report
    report_path = "tips_report.pdf"

    # Only proceed with month and year selection if a specific surgery is selected  -------------Month and Year Selector
    if page not in ["PCN Dashboard", "About"] and selected_surgery:
        surgery_data = pcn_data[pcn_data["surgery"] == selected_surgery]

        if not surgery_data.empty:
            # Find the earliest and latest dates in the data
            min_date = surgery_data['time'].min().to_pydatetime()
            max_date = surgery_data['time'].max().to_pydatetime()

            # Generate a list of years and months based on the data range
            years = list(range(min_date.year, max_date.year + 1))
            months = [datetime.strftime(datetime(2000, i, 1), "%B") for i in range(1, 13)]

            # Create two columns for selectors
            col1, col2 = st.columns(2)

            # Create year and month selectors inside the columns
            with col1:
                selected_year = st.selectbox("Select the Year", options=years, index=years.index(max_date.year))
            
            # Adjust month options based on selected year
            if selected_year == min_date.year:
                months = months[min_date.month - 1:]
            if selected_year == max_date.year:
                months = months[:max_date.month]

            with col2:
                selected_month = st.selectbox("Select the Month", options=months, index=0)

            # Convert selected month to number
            selected_month_number = datetime.strptime(selected_month, "%B").month

            # Filter data based on the selected month and year
            filtered_data = surgery_data[
                (surgery_data['time'].dt.year == selected_year) & 
                (surgery_data['time'].dt.month == selected_month_number)
            ]
        
     
    

    # Your existing setup code...

    if st.button('Generate CQRS Report'):
        # Call the function with the parameters from Streamlit widgets
        simple_pdf(filtered_data, selected_month, selected_year, selected_surgery, selected_pcn)
    
        # Inform the user of success
        ui.badges(badge_list=[("Report Generated Successfully!", "default")], class_name="flex gap-2", key="badges_success")

        # Provide a download link for the generated PDF
        with open("report.pdf", "rb") as file:
            st.download_button(label="Download CQRS Report", data=file, file_name="report.pdf", mime="application/pdf")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# -- About ------------------------------------------------------------------------------------------------------- About
elif page == "About":
    st.markdown("# ![About](https://img.icons8.com/ios/50/about.png) About")

    st.markdown(
        """### Patient Feedback Analysis in Healthcare
Welcome to **AI MedReview**, your powerful new dashboard for elevating healthcare providers' understanding and utilization of patient feedback. Our cutting-edge solution focuses on the essential Friends and Family Test (FFT), empowering you to extract deeper insights from this invaluable data source.
At the core of AI MedReview lies a transformative approach that goes beyond mere quantitative metrics. By leveraging advanced natural language processing and machine learning techniques, we unlock the nuanced sentiments behind patient responses. Our dashboard assigns detailed scores to each piece of feedback, painting a more comprehensive picture of patient satisfaction levels.

Through **sentiment analysis** powered by Hugging Face's `cardiffnlp/twitter-roberta-base-sentiment-latest model`, we precisely determine the emotional tone of patient comments, be it positive, negative, or neutral. This level of granular understanding enables you to celebrate areas of excellence and swiftly identify opportunities for improvement.

But we don't stop there. To protect patient privacy, we employ robust named **entity recognition (NER)** capabilities, utilizing the Hugging Face `dbmdz/bert-large-cased-finetuned-conll03-english model`. This ensures any personally identifiable information (PII) is seamlessly anonymized, safeguarding the confidentiality of your valuable data.

Furthermore, our innovative **zero-shot classification** approach, powered by the Facebook `BART-large-mnli` architecture, allows us to categorize patient feedback with remarkable accuracy – even without specialized healthcare training data. By carefully curating our classification labels, we achieved a striking 0.91 accuracy, demonstrating the remarkable versatility of this model.

This comprehensive suite of advanced analytics empowers healthcare providers like yourself to move beyond mere data presentation and unlock a clearer, more actionable understanding of patient experiences. Armed with these insights, you can drive continuous improvements, elevate service quality, and enhance patient outcomes.

Explore the AI MedReview dashboard today and experience the transformative power of data-driven decision-making in healthcare. Join us on our mission to revolutionize the way patient feedback is leveraged to deliver exceptional care.

Visit ![GitHub](https://img.icons8.com/material-outlined/24/github.png) [AI MedReview on GitHub](https://github.com/janduplessis883/ai-medreview), where collaboration and contributions are warmly welcomed."""
    )

    form_url = "https://tally.so/r/w2ed0e"  # Replace this URL with the URL of your form
    iframe_code = f'<iframe src="{form_url}" width="100%" height="400"></iframe>'
    st.markdown(iframe_code, unsafe_allow_html=True)

    debug_toggle = ui.switch(default_checked=False, label="Debug", key="debug")
    if debug_toggle:
        st.dataframe(data.tail(50))
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # Use 'col1' to display content in the first column
    with col1:
        st.image(
            "images/about.png",
            width=200,
        )

    # Use 'col2' to display content in the second column
    with col2:
        st.image(
            "images/hf-logo-with-title.png",
            width=200,
        )
    with col3:
        st.image(
            "images/openai.png",
            width=200,
        )