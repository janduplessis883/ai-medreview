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

st.set_page_config(page_title="AI MedReview v2")

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
<div class="gradient-text">AI MedReview v2</div>
"""

# ----------------------------------------------------------------------------------------------- Define a list of PCN'S
pcn_names = [ "Demo-PCN", "Brompton-Health-PCN"]

# Initialize session state for PCN if not present
if "pcn" not in st.session_state:
    st.session_state.pcn = pcn_names[0]  # Default to first PCN


# Update PCN selection
def update_pcn():
    st.session_state.pcn = st.session_state.pcn_selector


@st.cache_data(ttl=3600)  # -------------------------------------------------------------------------------  Load 'data'
def load_data():
    df = pd.read_csv("ai_medreview/data/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df

data = load_data()

@st.cache_data(ttl=3600) # ----------------------------------------------------------------------------  Load 'pcn_data'
def load_pcn_data(pcn_name):
    df = pd.read_csv("ai_medreview/data/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    # Filter based on the selected PCN
    pcn_specific_df = df[df["pcn"] == pcn_name]
    return pcn_specific_df

# Assume 'selected_pcn' is determined from user selection as shown in your provided code
pcn_data = load_pcn_data(st.session_state.pcn)


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


# Only get and display the surgery list if not on the PCN Dashboard or About pages ---------------------- define SIDEBAR
page = st.sidebar.radio(
    "Select a Page",
    [
        "PCN Dashboard",
        "Surgery Dashboard",
        "Feedback Classification",
        "Improvement Suggestions",
        "Feedback Timeline",
        "GPT-4 Summary",
        "Word Cloud",
        "Dataframe",
        "Reports",
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

    if not surgery_data.empty:   # ----------------------------------------------------------------------  Define Slider
        start_date = surgery_data["time"].dt.date.min()
        end_date = surgery_data["time"].dt.date.max()

        # Check if the start date is the same as the end date
        if start_date == end_date:
            start_date -= timedelta(days=1)  # Subtract one day from start_date
            
        try:
            # Ensure the dates are datetime.date objects; no need for to_pydatetime conversion
            selected_date_range = st.slider(
                f"{st.session_state.pcn} - **{selected_surgery}**",
                min_value=start_date,
                max_value=end_date,
                value=(start_date, end_date),
                format="MM/DD/YYYY",
            )
        except ValueError as e:  # Replace ValueError with RangeError or the appropriate error if different
            st.error(f"Cannot display slider: {str(e)}")
                
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
    st.markdown(f"# ![dashboard](https://img.icons8.com/pastel-glyph/64/laptop-metrics--v1.png) {selected_pcn} ")
    st.markdown(
        """Aggregating and analyzing the **collective patient feedback data** received by member practices.  
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


# -- Feedback Classification ------------------------------------------------------------------- Feedback Classification
elif page == "Feedback Classification":
    st.markdown("# ![Feedback](https://img.icons8.com/ios/50/thumbs-up-down.png) Feedback Classification")


# -- Improvement Suggestions ------------------------------------------------------------------- Improvement Suggestions
elif page == "Improvement Suggestions":
    st.markdown("# ![Improvement](https://img.icons8.com/ios/50/improvement.png) Improvement Suggestions")


# -- Feedback Timeline ------------------------------------------------------------------------------- Feedback Timeline
elif page == "Feedback Timeline":
    st.markdown("# ![Timeline](https://img.icons8.com/dotty/80/timeline.png) Feedback Timeline")


# -- Word Cloud --------------------------------------------------------------------------------------------- Word Cloud
elif page == "Word Cloud":
    st.markdown("# ![Word CLoud](https://img.icons8.com/ios/50/cloud-refresh--v1.png) Word Cloud")


# -- GPT4 Summary ----------------------------------------------------------------------------------------- GPT4 Summary
elif page == "GPT-4 Summary":
    st.markdown("# ![GPT-4](https://img.icons8.com/ios/50/chatgpt.png) GPT-4 Summary")


# -- Dataframe ----------------------------------------------------------------------------------------------- Dataframe
elif page == "Dataframe":
    st.markdown("# ![Dataframe](https://img.icons8.com/ios/50/new-spreadsheet.png) Dataframe")


# -- Reports --------------------------------------------------------------------------------------------------- Reports
elif page == "Reports":
    st.markdown("# ![Reports](https://img.icons8.com/ios/50/graph-report.png) Reports")
    
    
# -- About ------------------------------------------------------------------------------------------------------- About
elif page == "About":
    st.markdown("# ![About](https://img.icons8.com/ios/50/about.png) About")
