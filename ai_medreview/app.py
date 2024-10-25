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


# Initialize OpenAI API
client = OpenAI()

# Loading my OWN FUNCTIONS
from utils import *
from reports import *

# Initialize the Groq client
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# Simple function to get a response from Groq
@st.cache_resource
def ask_groq(prompt: str, model: str = "llama-3.1-8b-instant"):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content


st.set_page_config(page_title="AI MedReview v2")
st.logo('images/logo2.png', link='https://github.com/janduplessis883/ai-medreview')
# Function to check passcode
def check_passcode():
    passcode = st.secrets["passcode"]["pin"]
    with st.form("passcode_form", border=False):
        entered_passcode = st.text_input("Enter passcode:", type="password")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if entered_passcode == passcode:
                st.session_state["authenticated"] = True
            else:
                st.write()
                ui.badges(
                    badge_list=[("Incorrect Passcode, please try again.", "default")],
                    class_name="flex gap-2",
                    key="error1",
                )
                st.toast(body="**Incorrect Password?** [Get Passcode](mailto:jan.duplessis@nhs.net?subject=AI-MedReview-Passcode)", icon=":material/lock_person:")

# Check if the user is authenticated
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    c1, c2, c3 = st.columns([1,5,1])

    with c2:
        # st.image("images/private.png")
        st.image("images/private.png")
        st.markdown(
            f"# ![Protected](https://img.icons8.com/pastel-glyph/64/fingerprint.png) AI MedReview v2.2.1"
        )
        check_passcode()
        with st.container(height=80, border=False):
            st.write()

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.write()
        with c2:
            st.html("""<img alt="Static Badge" src="https://img.shields.io/badge/GitHub-janduplessis883-%23f5c244?link=https%3A%2F%2Fgithub.com%2Fjanduplessis883%2Fai-medreview">""")
            st.html("""<a href="https://groq.com" target="_blank" rel="noopener noreferrer"><BR>
  <img
    src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg"
    alt="Powered by Groq for fast inference." width=130
  />
</a>
""")
        with c3:
            st.write()

else:
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
    pcn_names = ["Brompton-Health-PCN", "Demo-PCN"]

    # PCN selection in sidebar
    selected_pcn = st.sidebar.selectbox("Select a Primary Care Network", pcn_names, key="pcn_selector")


    # Function to load data
    @st.cache_data(ttl=3600)
    def load_data():
        df = pd.read_csv("ai_medreview/data/data.csv")
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values(by='time')
        return df

    data = load_data()

    def data_version(df):
        last_time = df.iloc[-1]['time']
        return last_time

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


    # Define the function to filter data based on selected date range
    @st.cache_data(ttl=3600)
    def filter_data_by_date_range(data, date_range):
        return data[
            (data["time"].dt.date >= date_range[0])
            & (data["time"].dt.date <= date_range[1])
        ]


    # Page selection
    page = st.sidebar.radio(
        "Select a Page",
        [
            "**:blue-background[PCN Dashboard]**",
            "Surgery Dashboards",
            "Feedback Classification",
            "Improvement Suggestions",
            "Feedback Timeline",
            "Emotion Detection",
            "Sentiment Analysis",
            "NER People",
            "GPT-4 Summary",
            "Word Cloud",
            "Dataframe",
            "Reports",
            "Campaigns",
            "**About**",
        ],
    )

    # Surgery selection and data filtering
    if page not in ["**:blue-background[PCN Dashboard]**", "**About**"]:
        surgery_list = get_surgeries_by_pcn(pcn_data, selected_pcn)
        if len(surgery_list) > 0:
            selected_surgery = st.sidebar.selectbox("Select a Surgery", surgery_list)
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

                # Create data filtered by date via slider
                filtered_data = filter_data_by_date_range(surgery_data, selected_date_range)

    else:
        selected_surgery = None
        filtered_data = None


    # Content Start ========================================================================================== Content Start

    # -- PCN Dashboard --------------------------------------------------------------------------------------- PCN Dashboard
    if page == "**:blue-background[PCN Dashboard]**":
        st.toast(f"Listen to the **AI-MedReview podcast** for insightful discussions—find it on the **About** page!", icon=":material/mic:")
        ui.badges(badge_list=[("Data version:", "outline"), (f"{data_version(data)}", "secondary"), (f"{pcn_data.shape[0]}", "secondary")], class_name="flex gap-2", key="badges_pcn_dashboard_alert")

        st.markdown(
            f"# :material/health_metrics: {selected_pcn} "
        )


        st.markdown(
            """Accumulating and interpreting the **pooled patient feedback data** from member practices.
    """
        )

        st.write("")
        tab_selector = ui.tabs(
            options=[
                "PCN Rating",
                "PCN Responses",
                "Sentim.-Emotion",
                "Topic A.",
                "Surg. Ratings",
                "Surg. Responses",
            ],
            default_value="PCN Rating",
            key="tab3",
        )

        if (
            tab_selector == "PCN Responses"
        ):  # ------------------------------------------------------------------------------------------- PCN Responses ----
            st.subheader("PCN Responses")

            cols = st.columns(2)
            with cols[0]:
                ui.metric_card(
                    title="Total Responses",
                    content=f"{pcn_data.shape[0]}",
                    description=f"Since 17 July 2023.",
                    key="total_pcn_responses",
                )
            with cols[1]:
                pass

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

                monthly_count_filtered = pcn_data.resample("MS", on="time").size()
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

        elif (
            tab_selector == "Surg. Ratings"
        ):  # ---------------------------------------------------------------------------------------- Surgery Ratings -----
            st.subheader("Surgeries Ratings")

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
                    "Very good",
                    "Good",
                    "Neither good nor poor",
                    "Poor",
                    "Very poor",
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
                plt.title("% Heatmap of Surgeries Ratings", fontsize=16)
                plt.ylabel("")
                plt.xlabel("Rating (%)", fontsize=12)
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()

                # Display the ordered percentage heatmap
                st.pyplot(plt)

                st.markdown("---")

                brompton_health_pcn_data = load_pcn_data(selected_pcn)
                # Set the time as index
                brompton_health_pcn_data.set_index("time", inplace=True)

                # Resample and calculate the average monthly rating_score for each surgery
                monthly_rating = (
                    brompton_health_pcn_data.groupby("surgery")
                    .resample("MS")["rating_score"]
                    .mean()
                    .unstack(level=0)
                )

                # Plotting
                # Set up the figure and axis with your specified size
                plt.figure(figsize=(12, 8))
                ax = plt.gca()  # Get the current axis

                # Loop through each column in your DataFrame to plot the data
                for column in monthly_rating.columns:
                    ax.plot(
                        monthly_rating.index,
                        monthly_rating[column],
                        marker="o",
                        label=column,
                    )

                # Set the title and labels for the axes
                ax.set_title(
                    f"Average Monthly Rating Score for Each Surgery in {selected_pcn}"
                )
                ax.set_xlabel("Month")
                ax.set_ylabel("Average Rating Score")

                # Adjust the legend to be outside the plot area
                ax.legend(title="Surgery", bbox_to_anchor=(1.05, 1), loc="upper left")

                # Customize the grid lines
                ax.xaxis.grid(
                    True, linestyle="--", linewidth=0.5, color="#888888"
                )  # Customize x-axis grid lines
                ax.yaxis.grid(
                    True, linestyle="--", linewidth=0.5, color="#888888"
                )  # Customize y-axis grid lines

                # Hide the top and right borders
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Use tight layout to fit everything nicely
                plt.tight_layout()

                # Display the plot in Streamlit
                st.pyplot(plt)

        elif (
            tab_selector == "Sentim.-Emotion"
        ):  # -------------------------------------------------------------------------------------- Sentiment Analysis ----
            st.subheader("Emotion Detection")
            # Assuming 'data' is already defined and processed
            # Define labels and colors outside since they are the same for both plots

            data['time'] = pd.to_datetime(data['time'])

            # Combine the data from 'emotion_free_text' and 'emotion_do_better' into a single dataframe
            pooled_data = pd.concat([data[['time', 'emotion_free_text']].rename(columns={'emotion_free_text': 'emotion'}),
                                    data[['time', 'emotion_do_better']].rename(columns={'emotion_do_better': 'emotion'})])

            # Extract the year and month from the 'time' column
            pooled_data['year_month'] = pooled_data['time'].dt.to_period('M').astype(str)  # Convert Period to string

            # Group by 'year_month' and 'emotion' to get the monthly counts
            pooled_monthly_emotion_counts = pooled_data.groupby(['year_month', 'emotion']).size().unstack(fill_value=0)

            # Normalize the counts by dividing each emotion's count by the total counts for that month
            pooled_normalized_monthly_emotion_counts = pooled_monthly_emotion_counts.div(pooled_monthly_emotion_counts.sum(axis=1), axis=0)

            # Convert the normalized monthly counts dataframe to long format for Plotly
            pooled_normalized_long = pooled_normalized_monthly_emotion_counts.reset_index().melt(id_vars='year_month', var_name='emotion', value_name='proportion')

            # Create a Plotly line plot
            fig = px.line(
                pooled_normalized_long,
                x='year_month',
                y='proportion',
                color='emotion',
                markers=True,
                title='Normalized Monthly Counts of Pooled Emotions Over Time',
                labels={'year_month': 'Time (Year-Month)', 'proportion': 'Proportion'}
            )

            # Update layout for better appearance
            fig.update_layout(
                legend_title_text='Emotion',
                xaxis_title='Time (Year-Month)',
                yaxis_title='Proportion',
                autosize=False,
                width=750,
                height=550,
                margin=dict(l=40, r=40, b=40, t=40)
            )

            st.plotly_chart(fig)
            st.markdown("---")
        # Convert the 'time' column to datetime format
            data['time'] = pd.to_datetime(data['time'])

            # Combine the data from 'emotion_free_text' and 'emotion_do_better' into a single dataframe with a source column
            pooled_data_with_source = pd.concat([
                data[['time', 'emotion_free_text']].rename(columns={'emotion_free_text': 'emotion'}).assign(source='emotion_free_text'),
                data[['time', 'emotion_do_better']].rename(columns={'emotion_do_better': 'emotion'}).assign(source='emotion_do_better')
            ])

            # Filter out rows with NaN values in 'emotion'
            pooled_data_with_source = pooled_data_with_source.dropna(subset=['emotion'])

            # Pivot the data to get counts of each emotion per source
            pivot_table = pooled_data_with_source.pivot_table(index='emotion', columns='source', aggfunc='size', fill_value=0)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Define the colors for the sources
            colors = ["#4088a9", "#f1c045"]

            # Plot each source as a stacked bar
            bottom = None
            for i, source in enumerate(pivot_table.columns):
                counts = pivot_table[source]
                ax.barh(pivot_table.index, counts, left=bottom, label=source, color=colors[i])
                if bottom is None:
                    bottom = counts
                else:
                    bottom += counts

            # Customize the plot
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.yaxis.grid(False)
            ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
            plt.xlabel('Counts')
            plt.ylabel('Emotions')
            plt.title('Distribution of Emotions in Feedback and Improvement Suggestions')
            plt.legend(title='Source', loc='upper right')
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("Sentiment Analysis")
            st.markdown("Average Monthly **Sentiment Analysis Score** - Feedback")
            pcn_data["time"] = pd.to_datetime(pcn_data["time"])
            pcn_data.set_index("time", inplace=True)

            # Assuming filtered_data is your DataFrame and 'sentiment_score' is the column with the scores
            # Also assuming that 'time' column has been converted to datetime and set as the index

            # Calculate the standard deviation for each month and sentiment
            monthly_sentiment_std = (
                pcn_data.groupby("sentiment_free_text")
                .resample("MS")["sentiment_score_free_text"]
                .std()
                .unstack(level=0)
            )

            # Fill NaN values
            monthly_sentiment_std.fillna(0, inplace=True)

            # Calculate the mean sentiment scores for each month and sentiment, if not already done
            monthly_sentiment_means_adjusted = (
                pcn_data.groupby("sentiment_free_text")
                .resample("MS")["sentiment_score_free_text"]
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
                .resample("MS")["sentiment_score_do_better"]
                .std()
                .unstack(level=0)
            )

            # Fill NaN values
            monthly_sentiment_std.fillna(0, inplace=True)

            # Calculate the mean sentiment scores for each month and sentiment, if not already done
            monthly_sentiment_means_adjusted = (
                pcn_data.groupby("sentiment_do_better")
                .resample("MS")["sentiment_score_do_better"]
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

        elif (
            tab_selector == "Surg. Responses"
        ):  # -------------------------------------------------------------------------------------- Surgery Responses------
            st.subheader("Surgeries Responses")
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
            data_sorted["cumulative_count"] = data_sorted.groupby("surgery").cumcount() + 1

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
                legend=dict(title="Surgery", x=1.05, y=1, xanchor="left", yanchor="top"),
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
                        daily_mean_rating["rating_score"]
                        .resample("MS")
                        .mean()
                        .reset_index()
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
            except ValueError as e:
                st.warning(f"Error plotting: {e}")

            st.markdown("---")
            st.markdown("**Rating Count**")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.countplot(data=pcn_data, x="rating_score", color="#616884")

            total = len(pcn_data)  # Total number of observations

            # Annotate each bar
            for p in ax.patches:
                percentage = "{:.1f}%".format(
                    100 * p.get_height() / total
                )  # Calculate percentage
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.annotate(percentage, (x, y), ha="center", va="bottom")

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
            # st.markdown("![sub-menu](https://img.icons8.com/material-sharp/24/menu--v1.png) Sub-menu")
            st.subheader("Topic Analysis")
            tab_selector = ui.tabs(
                options=[
                    "Bar Chart - Totals",
                    "TS Line Chart",
                    "TS Heatmap (Normalized)",
                    "By Surgery Heatmap (Normalized)",
                ],
                default_value="Bar Chart - Totals",
                key="tab_topic_pcn",
            )


            # implement PCN Slider to load data by date for comparison -------------------------------PCN TOPIC A Slider----
            start_date = pcn_data["time"].dt.date.min()
            end_date = pcn_data["time"].dt.date.max()

            if start_date == end_date:
                start_date -= timedelta(days=1)

            try:
                pcn_date_range = st.slider(
                    f"Topic Analysis Date Range",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    format="MM/DD/YYYY",
                    help="Use to adjust the time frame for Topic Analaysis"
                )
            except ValueError as e:
                st.error(f"Cannot display slider: {str(e)}")

            # Create data filtered by date via slider
            filtered_pcn_data = filter_data_by_date_range(pcn_data, pcn_date_range)

            if tab_selector == "TS Line Chart":

                radio_options = [
                    {"label": "All", "value": "all", "id": "r7"},
                    {"label": "Negative", "value": "neg", "id": "r8"},
                    {"label": "Neutral + Positive", "value": "pos", "id": "r9"},
                ]
                radio_value = ui.radio_group(
                    options=radio_options, default_value="all", key="radio3"
                )

                if radio_value == "pos":
                    filtered_pcn_data = filtered_pcn_data[
                        (
                            (filtered_pcn_data["sentiment_free_text"] == "neutral")
                            | (filtered_pcn_data["sentiment_free_text"] == "positive")
                        )
                    ]
                elif radio_value == "neg":
                    filtered_pcn_data = filtered_pcn_data[
                        (filtered_pcn_data["sentiment_free_text"] == "negative")
                    ]

                filtered_pcn_data["time"] = pd.to_datetime(filtered_pcn_data["time"])
                # Setting the 'time' column as the index
                filtered_pcn_data.set_index("time", inplace=True)

                # Grouping by month and 'feedback_labels' and then counting the occurrences
                # Converting the time index to a period index for monthly resampling
                filtered_pcn_data.index = filtered_pcn_data.index.to_period("M")

                monthly_feedback_counts = (
                    filtered_pcn_data.groupby([filtered_pcn_data.index, "feedback_labels"])
                    .size()
                    .unstack(fill_value=0)
                )

                # Converting the period index back to a timestamp for compatibility with Plotly
                monthly_feedback_counts.index = monthly_feedback_counts.index.to_timestamp()

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

                monthly_improvement_counts = (
                    filtered_pcn_data.groupby(
                        [filtered_pcn_data.index, "improvement_labels"]
                    )
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
            elif (
                tab_selector == "Bar Chart - Totals"
            ):  # ------------------------------------------------------------------------------TAB_SELECTOR TIME SERIES----
                palette = {
                    "positive": "#2e5f77",
                    "negative": "#d7662a",
                    "neutral": "#d7d8d7",
                }
                hue_order = ["negative", "neutral", "positive"]

                # Create a cross-tabulation of feedback labels and sentiment categories

                crosstab = pd.crosstab(
                    filtered_pcn_data["feedback_labels"],
                    filtered_pcn_data["sentiment_free_text"],
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
                    filtered_pcn_data["improvement_labels"],
                    filtered_pcn_data["sentiment_do_better"],
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

            elif (
                tab_selector == "TS Heatmap (Normalized)"
            ):  # -------------------------------------------------------------------------NORMALIZED HEATMAP OF TOPICS-----

                radio_options = [
                    {"label": "All", "value": "all", "id": "r10"},
                    {"label": "Negative", "value": "neg", "id": "r11"},
                    {"label": "Neutral + Positive", "value": "pos", "id": "r12"},
                ]
                radio_value = ui.radio_group(
                    options=radio_options, default_value="all", key="radio6"
                )

                if radio_value == "pos":
                    filtered_pcn_data = filtered_pcn_data[
                        (
                            (filtered_pcn_data["sentiment_free_text"] == "neutral")
                            | (filtered_pcn_data["sentiment_free_text"] == "positive")
                        )
                    ]
                    title_string = "Heatmap of Normalized Feedback (NEUT + POS)"
                elif radio_value == "neg":
                    filtered_pcn_data = filtered_pcn_data[
                        (filtered_pcn_data["sentiment_free_text"] == "negative")
                    ]

                    title_string = "Heatmap of Normalized Feedback (NEGATIVE)"
                else:

                    filtered_pcn_data = filtered_pcn_data[
                        (
                            (filtered_pcn_data["sentiment_free_text"] == "negative")
                            | (filtered_pcn_data["sentiment_free_text"] == "positive")
                            | (filtered_pcn_data["sentiment_free_text"] == "neutral")
                        )
                    ]
                    title_string = "Heatmap of Normalized Feedback (ALL)"

                filtered_pcn_data["time"] = pd.to_datetime(filtered_pcn_data["time"])
                # Setting the 'time' column as the index
                filtered_pcn_data = filtered_pcn_data[["time", "feedback_labels"]].copy()
                filtered_pcn_data.set_index("time", inplace=True)
                filtered_pcn_data.index = filtered_pcn_data.index.to_period("M")

                monthly_feedback_counts = (
                    filtered_pcn_data.groupby([filtered_pcn_data.index, "feedback_labels"])
                    .size()
                    .unstack(fill_value=0)
                )
                monthly_feedback_counts["TOTAL"] = monthly_feedback_counts.sum(axis=1)
                normalized_df = monthly_feedback_counts.loc[
                    :, "Accessibility and Convenience":"Waiting Time"
                ].div(monthly_feedback_counts["TOTAL"], axis=0)
                st.markdown(
                    "#### Monthly Distribution of Feedback Labels - Heatmap: [Normalized]"
                )
                # Setting the plot size
                plt.figure(figsize=(15, 10))

                # Creating a heatmap
                sns.heatmap(
                    normalized_df.T, annot=True, cmap="Oranges", fmt=".2f", linewidths=0.5
                )

                # Adding titles and labels
                plt.title(title_string, fontsize=20)
                plt.xlabel("Month", fontsize=15)
                plt.ylabel("Feedback Categories", fontsize=15)

                # Displaying the plot
                st.pyplot(plt)

                st.markdown("---")
                pcn_data2 = load_pcn_data(selected_pcn)
                filtered_pcn_data2 = filter_data_by_date_range(pcn_data2, pcn_date_range)

                if radio_value == "pos":
                    filtered_pcn_data2 = filtered_pcn_data2[
                        (
                            (filtered_pcn_data2["sentiment_do_better"] == "neutral")
                            | (filtered_pcn_data2["sentiment_do_better"] == "positive")
                        )
                    ]
                    title_string = "Heatmap of Normalized Improvement Sugg. (NEUT + POS)"
                elif radio_value == "neg":
                    filtered_pcn_data2 = filtered_pcn_data2[
                        (filtered_pcn_data2["sentiment_do_better"] == "negative")
                    ]

                    title_string = "Heatmap of Normalized Improvement Sugg. (NEGATIVE)"
                else:

                    filtered_pcn_data2 = filtered_pcn_data2[
                        (
                            (filtered_pcn_data2["sentiment_do_better"] == "negative")
                            | (filtered_pcn_data2["sentiment_do_better"] == "positive")
                            | (filtered_pcn_data2["sentiment_do_better"] == "neutral")
                        )
                    ]
                    title_string = "Heatmap of Normalized Improvement Sugg. (ALL)"

                filtered_pcn_data2["time"] = pd.to_datetime(filtered_pcn_data2["time"])
                # Setting the 'time' column as the index
                filtered_pcn_data2 = filtered_pcn_data2[["time", "improvement_labels"]]
                filtered_pcn_data2.set_index("time", inplace=True)
                filtered_pcn_data2.index = filtered_pcn_data2.index.to_period("M")

                monthly_imp_counts = (
                    filtered_pcn_data2.groupby(
                        [filtered_pcn_data2.index, "improvement_labels"]
                    )
                    .size()
                    .unstack(fill_value=0)
                )
                monthly_imp_counts["TOTAL"] = monthly_imp_counts.sum(axis=1)
                normalized_df2 = monthly_imp_counts.loc[
                    :, "Accessibility and Convenience":"Waiting Time"
                ].div(monthly_imp_counts["TOTAL"], axis=0)
                st.markdown(
                    "#### Monthly Distribution of Improvement Labels - Heatmap: [Normalized]"
                )
                # Setting the plot size
                plt.figure(figsize=(15, 10))

                # Creating a heatmap
                sns.heatmap(
                    normalized_df2.T, annot=True, cmap="Greys", fmt=".2f", linewidths=0.5
                )

                # Adding titles and labels
                plt.title(title_string, fontsize=20)
                plt.xlabel("Month", fontsize=15)
                plt.ylabel("Improvement Sugg. Categories", fontsize=15)

                # Displaying the plot
                st.pyplot(plt)

            elif tab_selector == "By Surgery Heatmap (Normalized)":

                radio_options = [
                    {"label": "All", "value": "all", "id": "r10"},
                    {"label": "Negative", "value": "neg", "id": "r11"},
                    {"label": "Neutral + Positive", "value": "pos", "id": "r12"},
                ]
                radio_value = ui.radio_group(
                    options=radio_options, default_value="all", key="radio6"
                )

                if radio_value == "pos":
                    filtered_pcn_data = filtered_pcn_data[
                        (
                            (filtered_pcn_data["sentiment_free_text"] == "neutral")
                            | (filtered_pcn_data["sentiment_free_text"] == "positive")
                        )
                    ]
                    feedback_title = "Normalized Heatmap of Feedback Labels against Brompton-Health-PCN Surgeries (NEUT + POS)"
                    imp_title = "Normalized Heatmap of Improvement Labels against Brompton-Health-PCN Surgeries (NEUT + POS)"
                elif radio_value == "neg":
                    filtered_pcn_data = filtered_pcn_data[
                        (filtered_pcn_data["sentiment_free_text"] == "negative")
                    ]

                    feedback_title = "Normalized Heatmap of Feedback Labels against Brompton-Health-PCN Surgeries (NEG)"
                    imp_title = "Normalized Heatmap of Improvement Labels against Brompton-Health-PCN Surgeries (NEG)"
                else:

                    filtered_pcn_data = filtered_pcn_data[
                        (
                            (filtered_pcn_data["sentiment_free_text"] == "negative")
                            | (filtered_pcn_data["sentiment_free_text"] == "positive")
                            | (filtered_pcn_data["sentiment_free_text"] == "neutral")
                        )
                    ]
                    feedback_title = "Normalized Heatmap of Feedback Labels against Brompton-Health-PCN Surgeries (ALL)"
                    imp_title = "Normalized Heatmap of Improvement Labels against Brompton-Health-PCN Surgeries (ALL)"

                st.markdown(
                    "#### Feedback labels aggregated by Surgery - Heatmap: [Normalized]"
                )
                filtered_pcn_data.dropna(subset="free_text", inplace=True)
                heatmap_data = pd.crosstab(
                    filtered_pcn_data["surgery"], filtered_pcn_data["feedback_labels"]
                )
                normalized_heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

                # Create a new heatmap using the normalized data
                plt.figure(figsize=(14, 10))
                ax = sns.heatmap(
                    normalized_heatmap_data.T, annot=True, cmap="Reds", fmt=".2f"
                )
                plt.title(feedback_title)
                plt.xlabel("Surgeries")
                plt.ylabel("Feedback Labels")
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                # Show the normalized heatmap
                st.pyplot(plt)

                st.markdown("---")

                st.markdown(
                    "#### Improvmeent labels aggregated by Surgery - Heatmap: [Normalized]"
                )
                filtered_pcn_data.dropna(subset="do_better", inplace=True)
                heatmap_data = pd.crosstab(
                    filtered_pcn_data["surgery"], filtered_pcn_data["improvement_labels"]
                )
                normalized_heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

                # Create a new heatmap using the normalized data
                plt.figure(figsize=(14, 10))
                ax = sns.heatmap(
                    normalized_heatmap_data.T, annot=True, cmap="Grays", fmt=".2f"
                )
                plt.title(imp_title)
                plt.xlabel("Surgeries")
                plt.ylabel("Feedback Labels")
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                # Show the normalized heatmap
                st.pyplot(plt)


    # -- Surgery Dashboard --------------------------------------------------------------------------- Surgery Dashboard ---
    elif page == "Surgery Dashboards":
        st.markdown(
            f"# :material/stethoscope: {selected_surgery}"
        )
        st.write("")
        surgery_tab_selector = ui.tabs(
            options=[
                "Surgery Rating",
                "Surgery Responses",
                "Feedback Length",
                "Total Word Count",
                "Missing Data",
            ],
            default_value="Surgery Rating",
            key="tab4",
        )

        if surgery_tab_selector == "Surgery Rating":

            try:
                # Resample to get monthly average rating
                monthly_avg = filtered_data.resample("MS", on="time")["rating_score"].mean()

                # Reset index to make 'time' a column again
                monthly_avg_df = monthly_avg.reset_index()

                # Create a line plot
                fig, ax = plt.subplots(figsize=(12, 4))
                sns.lineplot(
                    x="time",
                    y="rating_score",
                    data=monthly_avg_df,
                    ax=ax,
                    marker="o",
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
                    "Very good",
                    "Good",
                    "Neither good nor poor",
                    "Poor",
                    "Very poor",
                    "Don't know",
            ]

            palette = {
                "Very good": "#112f45",
                "Good": "#4d9cb9",
                "Neither good nor poor": "#9bc8e3",
                "Poor": "#f4ba41",
                "Very poor": "#ec8b33",
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
                bbox_to_anchor=(1, 0),  # Place legend at the bottom right
                loc="lower right",
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
                        fontsize=14,
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

            st.divider()
            st.code(filtered_data['rating'].value_counts())

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
            monthly_count_filtered = filtered_data.resample("MS", on="time").size()
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

        elif (
            surgery_tab_selector == "Feedback Length"
        ):  # ------------------------------------------------Feedback Length -
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

        elif (
            surgery_tab_selector == "Total Word Count"
        ):  # ----------------------------------------------Total Word Count-
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

        elif (
            surgery_tab_selector == "Missing Data"
        ):  # -----------------------------------------------------Missing Data -
            plt.figure(figsize=(12, 5))
            sns.heatmap(filtered_data.isnull(), cbar=False, cmap="Blues", yticklabels=False)
            plt.title("Missing Data")
            st.pyplot(plt)


    # -- Feedback Classification ------------------------------------------------------------------- Feedback Classification
    elif page == "Feedback Classification":
        st.markdown(
            "# :material/recommend: Feedback Classification"
        )
        st.markdown("Responses to **FFT Q1**: Please tell us why you feel this way?")

        tab_selector = ui.tabs(
            options=[
                "Bar Chart - Totals",
                "Time Series - Line Chart",
                "Heatmap (Normalized)",
                "Pareto Analysis",
            ],
            default_value="Bar Chart - Totals",
            key="tab_topic_surgery",
        )
        if tab_selector == "Time Series - Line Chart":
            radio_options = [
                {"label": "All", "value": "all", "id": "r1"},
                {"label": "Negative", "value": "neg", "id": "r2"},
                {"label": "Neutral + Positive", "value": "pos", "id": "r3"},
            ]
            radio_value = ui.radio_group(
                options=radio_options, default_value="all", key="radio18"
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

        elif tab_selector == "Bar Chart - Totals":
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

        elif tab_selector == "Heatmap (Normalized)":

            radio_options = [
                {"label": "All", "value": "all", "id": "r10"},
                {"label": "Negative", "value": "neg", "id": "r11"},
                {"label": "Neutral + Positive", "value": "pos", "id": "r12"},
            ]
            radio_value = ui.radio_group(
                options=radio_options, default_value="all", key="radio6"
            )

            if radio_value == "pos":
                filtered_data = filtered_data[
                    (
                        (filtered_data["sentiment_free_text"] == "neutral")
                        | (filtered_data["sentiment_free_text"] == "positive")
                    )
                ]
                title_string = "Heatmap of Normalized Feedback (NEUT + POS)"
            elif radio_value == "neg":
                filtered_data = filtered_data[
                    (filtered_data["sentiment_free_text"] == "negative")
                ]

                title_string = "Heatmap of Normalized Feedback (NEGATIVE)"
            else:

                filtered_data = filtered_data[
                    (
                        (filtered_data["sentiment_free_text"] == "negative")
                        | (filtered_data["sentiment_free_text"] == "positive")
                        | (filtered_data["sentiment_free_text"] == "neutral")
                    )
                ]
                title_string = "Heatmap of Normalized Feedback (ALL)"

            filtered_data["time"] = pd.to_datetime(filtered_data["time"])
            # Setting the 'time' column as the index
            filtered_data2 = filtered_data[["time", "feedback_labels"]].copy()
            filtered_data2.set_index("time", inplace=True)
            filtered_data2.index = filtered_data2.index.to_period("M")

            monthly_feedback_counts = (
                filtered_data2.groupby([filtered_data2.index, "feedback_labels"])
                .size()
                .unstack(fill_value=0)
            )
            monthly_feedback_counts["TOTAL"] = monthly_feedback_counts.sum(axis=1)
            normalized_df = monthly_feedback_counts.loc[
                :, monthly_feedback_counts.columns[0] : monthly_feedback_counts.columns[-2]
            ].div(monthly_feedback_counts["TOTAL"], axis=0)
            st.markdown("#### Feedback Classification")
            # Setting the plot size
            plt.figure(figsize=(15, 10))

            # Creating a heatmap
            sns.heatmap(
                normalized_df.T, annot=True, cmap="Oranges", fmt=".2f", linewidths=0.5
            )

            # Adding titles and labels
            plt.title(title_string, fontsize=20)
            plt.xlabel("Month", fontsize=15)
            plt.ylabel("Feedback Categories", fontsize=15)

            # Displaying the plot
            st.pyplot(plt)

        elif tab_selector == "Pareto Analysis":
            st.html("<div class='status' style='background-color: #aa485b; color: white; padding-top: 2px; padding-bottom: 2px; padding-left: 7px; padding-right: 7px; border-radius: 6px; font-family: Arial, sans-serif; font-size: 12px; display: inline-block; text-align: center; box-shadow: 0px 3px 4px rgba(0, 0, 0, 0.2);'><b>NEW Pareto Analysis</b></div>")
            neg = filtered_data[filtered_data['sentiment_free_text'] == 'negative']
            negative_df = neg['feedback_labels'].value_counts().reset_index()
            negative_df.columns = ['Feedback Label', 'Count']
            negative_df['Cumulative Count'] = negative_df['Count'].cumsum()
            negative_df['Cumulative Percentage'] = 100 * negative_df['Cumulative Count'] / negative_df['Count'].sum()

            # Create the figure for plotting
            fig, ax1 = plt.subplots(figsize=(12, 8))

            # Bar plot for the count of feedback labels
            sns.barplot(x='Feedback Label', y='Count', data=negative_df, color='#4aa2e5', ax=ax1)
            plt.xticks(rotation=45, ha='right')

            # Create a secondary axis for the cumulative percentage
            ax2 = ax1.twinx()
            sns.lineplot(x='Feedback Label', y='Cumulative Percentage', data=negative_df, sort=False, color='#ae4f4d', marker='D', ax=ax2)

            # Set labels for the axes
            ax1.set_ylabel('Count')
            ax2.set_ylabel('Cumulative Percentage (%)')

            # Add a dashed line at 80% on the cumulative percentage axis
            ax2.axhline(80, color='green', linestyle='dashed')

            # Grid and spines adjustments for ax1 (the bar plot axis)
            ax1.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
            ax1.xaxis.grid(False)
            # Removing spines from ax1
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["left"].set_visible(False)

            # Removing spines from ax2 (the line plot axis)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)

            # Title and labels
            ax1.set_xlabel('Feedback Category')
            plt.title('Negative Feedback Pareto Chart')

            # Adjust layout for better appearance
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig)

            with st.expander("How to Interpret the **Pareto Plot**", icon=":material/downloading:"):

                st.write("""A **Pareto Analysis plot** is a useful tool for identifying the most significant factors that contribute to an overall problem or outcome. It is based on the **Pareto Principle**, often called the “80/20 rule,” which suggests that roughly 80% of the effects come from 20% of the causes. In the context of patient reviews, this plot highlights key areas that may need improvement in order to address the majority of negative feedback.

### How to Interpret the Pareto Plot:

1.	**Bar Chart** (Blue Bars) – Negative Feedback Count:
    - Each bar represents a feedback category (e.g., “Waiting Time”, “Appointment Availability”, etc.).
    - The height of the bar shows how frequently this particular issue was mentioned negatively. The taller the bar, the more common the complaint is.
    - Categories are arranged in descending order of significance, with the most frequent complaints on the left.
2.	**Line Chart** (Red Line) – Cumulative Percentage:
    - The red line tracks the cumulative percentage of negative feedback as you move from left to right.
    - It indicates how addressing the most frequent complaints (starting from the left) can significantly reduce the total number of negative reviews.
    - The 80% threshold (green dashed line) is particularly important. Issues to the left of this line often represent the vital few—the critical areas that cause the most dissatisfaction.
3.	**Focus on the Leftmost Bars**:
    - The first few bars usually cover the bulk of complaints. For example, in this chart, addressing Waiting Time and Appointment Availability could potentially resolve a large proportion of negative feedback.
    - The goal of a Pareto Analysis is to help prioritize improvements by focusing on these high-impact areas first, which is more efficient than spreading resources thinly across all categories.
4.	**Cumulative Insights**:
    - By fixing issues that make up 80% of the negative feedback (those left of the green dashed line), you can maximize patient satisfaction with minimal effort.
    - The remaining feedback categories, while still important, may have less impact overall on improving the perception of the service.

### Final Notes:

This type of analysis can be customized per GP surgery based on patient reviews. The feedback categories will differ depending on each surgery’s specific circumstances. However, the general rule remains the same: focus on the most frequent and impactful issues first to efficiently improve overall patient experience.
""")

            with st.expander("**Evidence Based Intervention**", icon=":material/search_insights:"):
                st.write("""## **1. Reception Staff Interaction**

**Key Issues Identified:**

- Rude and unhelpful behavior from reception staff.
- Lack of communication skills and professionalism.
- Patients feeling disrespected and not valued.
- Inappropriate questioning that invades patient privacy.

**Evidence-Based Interventions:**

1. **Customer Service and Communication Training:**
    - **Action:** Implement regular training programs focused on customer service excellence, effective communication, and handling difficult situations.
    - **Evidence:** Studies show that training staff in communication skills improves patient satisfaction and staff-patient interactions (Hobgood et al., 2010).
2. **Establish a Clear Code of Conduct:**
    - **Action:** Develop and enforce a code of conduct outlining expected behaviors, professionalism, and standards for patient interactions.
    - **Evidence:** Clear guidelines help staff understand expectations, leading to improved behavior (Bismark et al., 2013).
3. **Regular Performance Reviews and Feedback:**
    - **Action:** Conduct periodic evaluations of reception staff, providing constructive feedback and setting improvement goals.
    - **Evidence:** Ongoing feedback mechanisms enhance employee performance and patient satisfaction (Miller et al., 2015).
4. **Implement Mystery Shopper Programs:**
    - **Action:** Use anonymous patient representatives to assess the quality of service and identify areas for improvement.
    - **Evidence:** Mystery shopper programs effectively evaluate service quality and highlight training needs (Mahmoud et al., 2017).
5. **Enhance Patient Privacy Protocols:**
    - **Action:** Train staff on appropriate ways to gather necessary information without infringing on patient privacy.
    - **Evidence:** Respecting patient confidentiality builds trust and improves the patient experience (Ozair et al., 2015).
6. **Improve Staff Well-being and Support:**
    - **Action:** Address workload issues, provide adequate breaks, and offer support programs to reduce staff burnout.
    - **Evidence:** Supporting staff well-being leads to better patient interactions and reduces turnover (Shanafelt et al., 2015).

---

## **2. Appointment Availability**

**Key Issues Identified:**

- Difficulty securing timely appointments, especially face-to-face.
- Long waiting times and lack of appointment slots.
- Inefficient appointment booking processes.
- Insufficient consultation time with doctors.

**Evidence-Based Interventions:**

1. **Optimize Appointment Scheduling Systems:**
    - **Action:** Analyze and streamline scheduling processes to reduce bottlenecks and improve efficiency.
    - **Evidence:** Efficient scheduling systems reduce wait times and enhance patient satisfaction (Murray & Berwick, 2003).
2. **Implement Advanced Access Scheduling:**
    - **Action:** Offer same-day appointments and minimize delays by adjusting scheduling practices.
    - **Evidence:** Advanced access reduces wait times and no-show rates (Third Next Available Appointment metric; Murray & Tantau, 2000).
3. **Increase Appointment Capacity:**
    - **Action:** Extend clinic hours, hire additional healthcare providers, or utilize nurse practitioners and physician assistants.
    - **Evidence:** Increasing provider availability improves access to care (Hooker & Muchow, 2015).
4. **Provide Multiple Booking Options:**
    - **Action:** Enable online, phone, and in-person booking, allowing patients to choose their preferred method.
    - **Evidence:** Multi-channel booking systems increase patient engagement and convenience (Paré et al., 2014).
5. **Allow Direct In-Person Appointments:**
    - **Action:** Give patients the option to book face-to-face appointments without mandatory preliminary telephone consultations.
    - **Evidence:** Patient-centered scheduling respects patient preferences and improves satisfaction (LaGanga & Lawrence, 2007).
6. **Enhance Communication About Availability:**
    - **Action:** Transparently communicate appointment availability and expected wait times.
    - **Evidence:** Clear communication reduces frustration and manages patient expectations (Franco et al., 2014).
7. **Adjust Appointment Durations:**
    - **Action:** Offer flexible appointment lengths based on patient needs, allowing more time for complex cases.
    - **Evidence:** Tailoring appointment durations improves care quality and patient satisfaction (Campbell et al., 2013).

---

## **3. Prescriptions and Medication Management**

**Key Issues Identified:**

- Inadequate supply of medications for chronic conditions.
- Difficulty communicating with pharmacists or getting medication advice.
- Lack of proactive communication regarding test results.
- Concerns about incorrect or unsafe prescribing practices.

**Evidence-Based Interventions:**

1. **Implement Medication Management Programs:**
    - **Action:** Conduct regular medication reviews to ensure patients receive appropriate quantities and understand their treatment plans.
    - **Evidence:** Medication management improves adherence and health outcomes (Jin et al., 2008).
2. **Enhance Access to Pharmacists:**
    - **Action:** Integrate pharmacists into the care team and make them accessible to patients for consultations.
    - **Evidence:** Pharmacist involvement in patient care enhances medication safety and satisfaction (Chisholm-Burns et al., 2010).
3. **Establish Proactive Follow-Up Systems:**
    - **Action:** Create protocols for timely communication of test results and follow-up actions required.
    - **Evidence:** Systematic follow-up reduces missed results and improves patient outcomes (Callen et al., 2012).
4. **Implement Electronic Prescribing (e-Prescribing):**
    - **Action:** Use electronic systems to manage prescriptions, reducing errors and improving efficiency.
    - **Evidence:** e-Prescribing enhances accuracy and safety in medication management (Kaushal et al., 2010).
5. **Provide Staff Training on Safe Prescribing:**
    - **Action:** Educate healthcare providers on up-to-date prescribing guidelines and medication safety.
    - **Evidence:** Ongoing education reduces prescribing errors and adverse drug events (Phillips et al., 2001).
6. **Develop Clear Communication Channels:**
    - **Action:** Establish direct lines of communication for patients to discuss medication concerns promptly.
    - **Evidence:** Effective communication is key to safe medication practices (Manias et al., 2015).

---

## **4. Patient Respect**

**Key Issues Identified:**

- Patients feeling disrespected, humiliated, or undervalued.
- Rude and unprofessional behavior from attendants.
- Preference for in-person interactions due to poor online services.
- Need for better customer service manners among reception staff.

**Evidence-Based Interventions:**

1. **Train Staff in Empathy and Compassion:**
    - **Action:** Provide training focused on developing empathy, active listening, and compassionate care.
    - **Evidence:** Empathy training enhances patient-provider relationships and satisfaction (Stepien & Baernstein, 2006).
2. **Promote a Patient-Centered Care Culture:**
    - **Action:** Foster an organizational culture that prioritizes patient needs and respects individual preferences.
    - **Evidence:** Patient-centered care leads to better health outcomes and satisfaction (Barry & Edgman-Levitan, 2012).
3. **Improve Online Service Platforms:**
    - **Action:** Upgrade online systems to be user-friendly, reliable, and responsive to patient needs.
    - **Evidence:** Effective online services improve patient engagement and access to care (Goldzweig et al., 2013).
4. **Implement Professionalism Standards:**
    - **Action:** Set and enforce standards for professional behavior across all staff interactions with patients.
    - **Evidence:** Professionalism is linked to higher patient trust and satisfaction (Chisholm & Askham, 2006).
5. **Collect and Act on Patient Feedback:**
    - **Action:** Regularly solicit patient feedback on their experiences and use it to drive improvements.
    - **Evidence:** Patient feedback mechanisms are effective in enhancing service quality (Doyle et al., 2013).

---

## **5. Test Results**

**Key Issues Identified:**

- Delays in scheduling appointments and conducting tests.
- Misdiagnoses leading to severe health consequences.
- Inaction or dismissal of significant test results.

**Evidence-Based Interventions:**

1. **Strengthen Diagnostic Protocols:**
    - **Action:** Implement evidence-based guidelines to improve diagnostic accuracy and timeliness.
    - **Evidence:** Adherence to clinical guidelines reduces diagnostic errors (Graber et al., 2012).
2. **Implement Test Result Management Systems:**
    - **Action:** Use electronic systems to track test orders, results, and ensure timely patient notification.
    - **Evidence:** Automated result management improves follow-up and patient safety (Callen et al., 2011).
3. **Adopt Safety Netting Strategies:**
    - **Action:** Educate clinicians on safety netting, including advising patients on what to do if symptoms persist or worsen.
    - **Evidence:** Safety netting reduces the risk of missed diagnoses and improves outcomes (Almond et al., 2009).
4. **Provide Continuing Education for Clinicians:**
    - **Action:** Offer regular training on recognizing atypical presentations and when to consider alternative diagnoses.
    - **Evidence:** Continuous professional development enhances diagnostic skills (Eva et al., 2010).
5. **Enhance Communication and Shared Decision-Making:**
    - **Action:** Encourage open dialogue between patients and providers about symptoms, concerns, and treatment options.
    - **Evidence:** Shared decision-making improves patient satisfaction and adherence (Elwyn et al., 2012).

---

**References:**

- Almond, S., Mant, D., & Thompson, M. (2009). Diagnostic safety-netting. British Journal of General Practice, 59(568), 872–874.
- Barry, M. J., & Edgman-Levitan, S. (2012). Shared decision making—The pinnacle of patient-centered care. New England Journal of Medicine, 366(9), 780–781.
- Bismark, M. M., Spittal, M. J., Gurrin, L. C., Ward, M., & Studdert, D. M. (2013). Identification of doctors at risk of recurrent complaints: a national study of healthcare complaints in Australia. BMJ Quality & Safety, 22(7), 532–540.
- Callen, J. L., Westbrook, J. I., Georgiou, A., & Li, J. (2011). Failure to follow-up test results for ambulatory patients: a systematic review. Journal of General Internal Medicine, 27(10), 1334–1348.
- Callen, J., et al. (2012). Emergency physicians’ views of direct notification of laboratory and radiology results to patients using the Internet: A multisite survey. Journal of Medical Internet Research, 14(1), e60.
- Campbell, J. L., et al. (2013). Range and causes of GP stress. British Journal of General Practice, 63(614), e722–e729.
- Chisholm, A., & Askham, J. (2006). What do you think of your doctor? A review of questionnaires for gathering patients' feedback on their doctor. Picker Institute Europe.
- Chisholm-Burns, M. A., et al. (2010). US pharmacists' effect as team members on patient care: systematic review and meta-analyses. Medical Care, 48(10), 923–933.
- Davis, K. (2010). Effective communication strategies in medicine. The Permanente Journal, 14(3), 68.
- Doyle, C., Lennox, L., & Bell, D. (2013). A systematic review of evidence on the links between patient experience and clinical safety and effectiveness. BMJ Open, 3(1), e001570.
- Elwyn, G., et al. (2012). Shared decision making: a model for clinical practice. Journal of General Internal Medicine, 27(10), 1361–1367.
- Eva, K. W., et al. (2010). Factors influencing responsiveness to feedback: on the interplay between fear, confidence, and reasoning processes. Advances in Health Sciences Education, 17(1), 15–26.
- Franco, L. M., et al. (2014). Effectiveness of collaborative improvement in reproductive health: a quasi-experimental study in Ghana. Health Policy and Planning, 29(5), 641–653.
- Goldzweig, C. L., et al. (2013). Patient internet portals: evidence on health outcomes, satisfaction, efficiency, and attitudes: a systematic review. Annals of Internal Medicine, 159(10), 677–687.
- Graber, M. L., et al. (2012). Improving diagnosis by improving education: a policy brief on education in healthcare professions. Diagnosis, 1(1), 51–56.
- Hobgood, C., et al. (2010). Communication training in emergency medicine: a systematic review. Academic Emergency Medicine, 17(6), 637–650.
- Hooker, R. S., & Muchow, A. N. (2015). Modifying state laws for nurse practitioners and physician assistants can reduce cost of medical services. Nursing Economics, 33(2), 88–94.
- Jin, J., et al. (2008). Factors affecting therapeutic compliance: A review from the patient’s perspective. Therapeutics and Clinical Risk Management, 4(1), 269–286.
- Kaushal, R., et al. (2010). Electronic prescribing improves medication safety in community-based office practices. Journal of General Internal Medicine, 25(6), 530–536.
- LaGanga, L. R., & Lawrence, S. R. (2007). Clinic overbooking to improve patient access and increase provider productivity. Decision Sciences, 38(2), 251–276.
- Mahmoud, M. A., et al. (2017). Evaluating patients’ satisfaction using mystery shopping technique in community pharmacy settings in Kuwait. Pharmacy Practice, 15(1), 881.
- Manias, E., et al. (2015). Interventions to reduce medication errors in adult intensive care: a systematic review. British Journal of Clinical Pharmacology, 80(5), 1262–1278.
- Miller, A., et al. (2015). Positive deviance: A different approach to achieving patient safety. Healthcare Quarterly, 18(1), 8–14.
- Mueller, S. K., et al. (2012). Hospital-based medication reconciliation practices: a systematic review. Archives of Internal Medicine, 172(14), 1057–1069.
- Murray, M., & Berwick, D. M. (2003). Advanced access: reducing waiting and delays in primary care. JAMA, 289(8), 1035–1040.
- Murray, M., & Tantau, C. (2000). Same-day appointments: exploding
                         """)


        st.toast("**Pareto Analysis Chart** Select from submenu.", icon=":material/linear_scale:")

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
                selected_text = ''
                for index, row in specific_class.iterrows():
                    text = row["free_text"]
                    sentiment = row["sentiment_free_text"]
                    if sentiment == "neutral":
                        text_color = "gray"
                    elif sentiment == 'negative':
                        text_color = "orange"
                    elif sentiment == 'positive':
                        text_color = "blue"

                    if str(text).lower() != "nan":
                        st.markdown(f"- :{text_color}[{str(text)}] ")

                        selected_text = selected_text + ' ' + text

                _ = st.popover(f"Summarize **{rating}** Feedback", icon=":material/robot_2:")
                with _:
                    _ = st.button("Summarize with Groq LLM", key=f"{rating}", icon=":material/robot_2:")
                    if _ == True:
                        summary = ask_groq(f"Summarize the following feedback all classified as {rating}, highlighting Positive and Negative trends, feedback text to summarizie: {selected_text}")
                        st.markdown(f"### {rating}:\n {summary}")

    # -- Improvement Suggestions ------------------------------------------------------------------- Improvement Suggestions
    elif page == "Improvement Suggestions":
        st.markdown(
            "# :material/home_improvement_and_tools: Improvement Suggestions"
        )
        st.markdown(
            "Responses to **FFT Q2**: Is there anything that would have made your experience better?"
        )

        tab_selector = ui.tabs(
            options=[
                "Bar Chart - Totals",
                "Time Series - Line Chart",
                "Heatmap (Normalized)",
            ],
            default_value="Bar Chart - Totals",
            key="tab_topic_surgery2",
        )
        if tab_selector == "Time Series - Line Chart":

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
            try:
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
                        title="Improvement Labels",
                        x=1.05,
                        y=1,
                        xanchor="left",
                        yanchor="top",
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
            except:
                ui.badges(
                    badge_list=[
                        (
                            "No improvement suggestions available for this date range.",
                            "outline",
                        )
                    ],
                    class_name="flex gap-2",
                    key="badges114",
                )

        elif tab_selector == "Bar Chart - Totals":
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

        elif tab_selector == "Heatmap (Normalized)":

            radio_options = [
                {"label": "All", "value": "all", "id": "r4"},
                {"label": "Negative", "value": "neg", "id": "r5"},
                {"label": "Neutral + Positive", "value": "pos", "id": "r6"},
            ]
            radio_value = ui.radio_group(
                options=radio_options, default_value="all", key="radio2"
            )

            filtered_data2 = filtered_data.copy()

            if radio_value == "pos":
                filtered_data2 = filtered_data2[
                    (
                        (filtered_data2["sentiment_do_better"] == "neutral")
                        | (filtered_data2["sentiment_do_better"] == "positive")
                    )
                ]
                title_string = "Heatmap of Normalized Improvement Sugg. (NEUT + POS)"
            elif radio_value == "neg":
                filtered_data2 = filtered_data2[
                    (filtered_data2["sentiment_do_better"] == "negative")
                ]

                title_string = "Heatmap of Normalized Improvement Sugg. (NEGATIVE)"
            else:

                filtered_data2 = filtered_data2[
                    (
                        (filtered_data2["sentiment_do_better"] == "negative")
                        | (filtered_data2["sentiment_do_better"] == "positive")
                        | (filtered_data2["sentiment_do_better"] == "neutral")
                    )
                ]
                title_string = "Heatmap of Normalized Improvement Sugg. (ALL)"

            filtered_data2["time"] = pd.to_datetime(filtered_data2["time"])
            # Setting the 'time' column as the index
            filtered_data2 = filtered_data2[["time", "improvement_labels"]]
            filtered_data2.set_index("time", inplace=True)
            filtered_data2.index = filtered_data2.index.to_period("M")

            monthly_imp_counts = (
                filtered_data2.groupby([filtered_data2.index, "improvement_labels"])
                .size()
                .unstack(fill_value=0)
            )
            monthly_imp_counts["TOTAL"] = monthly_imp_counts.sum(axis=1)
            normalized_df2 = monthly_imp_counts.loc[
                :, monthly_imp_counts.columns[0] : monthly_imp_counts.columns[-2]
            ].div(monthly_imp_counts["TOTAL"], axis=0)
            st.markdown("#### Improvement Suggestions")
            # Setting the plot size
            plt.figure(figsize=(15, 10))

            # Creating a heatmap
            sns.heatmap(
                normalized_df2.T, annot=True, cmap="Grays", fmt=".2f", linewidths=0.5
            )

            # Adding titles and labels
            plt.title(title_string, fontsize=20)
            plt.xlabel("Month", fontsize=15)
            plt.ylabel("Improvement Sugg. Categories", fontsize=15)

            # Displaying the plot
            st.pyplot(plt)

        st.markdown("---")
        st.toast("Use the **Date Slider** to define the review period.", icon=":material/linear_scale:")

        st.subheader("View Patient Improvement Suggestions")
        improvement_data = filtered_data[
            (filtered_data["improvement_labels"] != "No Improvement Suggestion")
        ]
        label_counts = improvement_data["improvement_labels"].value_counts(
            ascending=False
        )  # Use ascending=True to match the order in your image

        # Convert the Series to a DataFrame
        label_counts_df = label_counts.reset_index()
        label_counts_df.columns = ["Improvement Labels", "Counts"]
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
                st.subheader(f"{str(rating).capitalize()} ({str(specific_class.shape[0])})")
                selected_text = ''
                for index, row in specific_class.iterrows():
                    text = row["do_better"]
                    text = text.replace("[PERSON]", "PERSON")
                    sentiment = row["sentiment_do_better"]
                    if sentiment == "neutral":
                        text_color = "gray"
                    elif sentiment == 'negative':
                        text_color = "orange"
                    elif sentiment == 'positive':
                        text_color = "blue"

                    if str(text).lower() != "nan":
                        st.markdown(f"- :{text_color}[{str(text)}] ")

                        selected_text = selected_text + ' ' + text

                _ = st.popover(f"Summarize **{rating}** Improvement Suggestions", icon=":material/robot_2:")
                with _:
                    _ = st.button("Summarize with Groq LLM", key=f"{rating}", icon=":material/robot_2:")
                    if _ == True:
                        summary = ask_groq(f"Summarize the following feedback all classified as {rating}, highlighting Positive and Negative trends, feedback text to summarizie: {selected_text}")
                        st.markdown(f"### {rating}:\n {summary}")


    # -- Emotion Detection ------------------------------------------------------------------------------- Emotion Detection
    elif page == "Emotion Detection":
        st.markdown(
            "# :material/add_reaction: Emotion Detection"
        )
        st.markdown(
            "Identifying and classifying the **emotions** expressed in FFT feedback."
        )

        # Convert the 'time' column to datetime format
        filtered_data['time'] = pd.to_datetime(filtered_data['time'])

        # Combine the filtered_data from 'emotion_free_text' and 'emotion_do_better' into a single dataframe
        pooled_data = pd.concat([filtered_data[['time', 'emotion_free_text']].rename(columns={'emotion_free_text': 'emotion'}),
                                filtered_data[['time', 'emotion_do_better']].rename(columns={'emotion_do_better': 'emotion'})])

        # Extract the year and month from the 'time' column
        pooled_data['year_month'] = pooled_data['time'].dt.to_period('M').astype(str)  # Convert Period to string

        # Group by 'year_month' and 'emotion' to get the monthly counts
        pooled_monthly_emotion_counts = pooled_data.groupby(['year_month', 'emotion']).size().unstack(fill_value=0)

        # Normalize the counts by dividing each emotion's count by the total counts for that month
        pooled_normalized_monthly_emotion_counts = pooled_monthly_emotion_counts.div(pooled_monthly_emotion_counts.sum(axis=1), axis=0)

        # Convert the normalized monthly counts dataframe to long format for Plotly
        pooled_normalized_long = pooled_normalized_monthly_emotion_counts.reset_index().melt(id_vars='year_month', var_name='emotion', value_name='proportion')

        # Create a Plotly line plot
        fig = px.line(
            pooled_normalized_long,
            x='year_month',
            y='proportion',
            color='emotion',
            markers=True,
            title='Normalized Monthly Counts of Pooled Emotions Over Time',
            labels={'year_month': 'Time (Year-Month)', 'proportion': 'Proportion'}
        )

        # Update layout for better appearance
        fig.update_layout(
            legend_title_text='Emotion',
            xaxis_title='Time (Year-Month)',
            yaxis_title='Proportion',
            autosize=False,
            width=750,
            height=550,
            margin=dict(l=40, r=40, b=40, t=40)
        )

        st.plotly_chart(fig)
        st.markdown("---")
        # Convert the 'time' column to datetime format
        # Assuming filtered_data is your DataFrame
    # Convert the 'time' column to datetime
        filtered_data['time'] = pd.to_datetime(filtered_data['time'])

        # Combine the filtered_data from 'emotion_free_text' and 'emotion_do_better' into a single dataframe with a source column
        pooled_data_with_source = pd.concat([
            filtered_data[['time', 'emotion_free_text']].rename(columns={'emotion_free_text': 'emotion'}).assign(source='emotion_free_text'),
            filtered_data[['time', 'emotion_do_better']].rename(columns={'emotion_do_better': 'emotion'}).assign(source='emotion_do_better')
        ])

        # Filter out rows with NaN values in 'emotion'
        pooled_data_with_source = pooled_data_with_source.dropna(subset=['emotion'])

        # Pivot the data to get counts of each emotion per source
        pivot_table = pooled_data_with_source.pivot_table(index='emotion', columns='source', aggfunc='size', fill_value=0)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define the colors for the sources
        colors = ["#985e5b", "#ca8f56"]

        # Plot each source as a stacked bar
        bottom = None
        for i, source in enumerate(pivot_table.columns):
            counts = pivot_table[source]
            ax.barh(pivot_table.index, counts, left=bottom, label=source, color=colors[i])
            if bottom is None:
                bottom = counts
            else:
                bottom += counts

        # Customize the plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        plt.xlabel('Counts')
        plt.ylabel('Emotions')
        plt.title('Distribution of Emotions in Feedback and Improvement Suggestions')
        plt.legend(title='Source', loc='upper right')
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)






    # -- Feedback Timeline ------------------------------------------------------------------------------- Feedback Timeline
    elif page == "Feedback Timeline":
        st.markdown(
            "# :material/view_timeline: Feedback Timeline"
        )

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
        st.markdown(f"Showing 150/**{filtered_data.shape[0]}** FFT Responses.")

        sample = filtered_data.tail(150)

        with st.container(height=500, border=True):
            with st.spinner(text="Loading Feedback..."):
                time.sleep(0.5)
                icounter = 1
                for _, row in sample.iterrows():
                    free_text = row["free_text"]
                    do_better = row["do_better"]
                    feedback_labels = row["feedback_labels"]
                    imp_labels = row["improvement_labels"]
                    time_ = row["time"]
                    rating = row["rating"]
                    fb_emotion = row['emotion_free_text']
                    ip_emotion = row['emotion_do_better']

                    with st.container(border=True):
                        ui.badges(badge_list=[(f"{icounter}", "destructive"), (f"{rating}", "default"), (f"{time_}", "secondary")], key=f"badge_ratingss_{icounter}")

                        if str(free_text) not in ["nan"]:
                            st.markdown(f"🤔 " + str(free_text))
                            ui.badges(badge_list=[(f"{fb_emotion}", "outline"), (f"{feedback_labels}", "outline")], key=f"badges_feedback_{icounter}")

                            if str(do_better) not in ["nan"]:
                                st.markdown("🛠️ " + str(do_better))
                                ui.badges(badge_list=[(f"{ip_emotion}", "outline"), (f"{imp_labels}", "outline")], key=f"badges_improve_{icounter}")

                    icounter += 1


    # -- Sentiment Analysis ----------------------------------------------------------------------------- Sentiment Analysis
    elif page == "Sentiment Analysis":
        st.markdown(
            "# :material/sentiment_neutral: Sentiment Analysis"
        )

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
            .resample("MS")["sentiment_score_free_text"]
            .std()
            .unstack(level=0)
        )

        # Fill NaN values
        monthly_sentiment_std.fillna(0, inplace=True)

        # Calculate the mean sentiment scores for each month and sentiment, if not already done
        monthly_sentiment_means_adjusted = (
            filtered_data.groupby("sentiment_free_text")
            .resample("MS")["sentiment_score_free_text"]
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
            .resample("MS")["sentiment_score_do_better"]
            .std()
            .unstack(level=0)
        )

        # Fill NaN values
        monthly_sentiment_std.fillna(0, inplace=True)

        # Calculate the mean sentiment scores for each month and sentiment, if not already done
        monthly_sentiment_means_adjusted = (
            filtered_data.groupby("sentiment_do_better")
            .resample("MS")["sentiment_score_do_better"]
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

        st.markdown(f"### FFT Feedback with a :red-background[NEGATIVE] Sentiment Score.")

        toggle = ui.switch(
            default_checked=True, label="Show last 30 days only.", key="switch_dash_neg"
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
                    icounter = 1
                    for _, row in neg1.iterrows():
                        free_text = row["free_text"]
                        cat = row["feedback_labels"]
                        time_ = row["time"]
                        rating = row["rating"]
                        score = row["sentiment_score_free_text"]
                        sentiment = row["sentiment_free_text"]
                        emotion = row["emotion_free_text"]

                        with st.container(border=True):
                        # st.markdown(f"**{rating}** `{time_}`")
                            ui.badges(badge_list=[(f"{icounter}", "destructive"), (f"{rating}", "default"), (f"{time_}", "secondary")], class_name=f"badges_improve_head_{icounter}")
                            if str(free_text) not in ["nan"]:
                                st.markdown("🤔 " + str(free_text))
                                # st.markdown(f"`{sentiment} {score}` `{cat}`")
                                ui.badges(badge_list=[(f"{emotion}", "outline"), (f"{cat}", "outline"), (f"{sentiment}", "secondary"), (f"{score}", "secondary")], class_name=f"badges_improve_{icounter}")

                        icounter += 1

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
                with st.container(height=550, border=True):
                    icounter = 1
                    for _, row in neg2.iterrows():
                        do_better = row["do_better"]
                        cat = row["improvement_labels"]
                        time_ = row["time"]
                        rating = row["rating"]
                        score = row["sentiment_score_do_better"]
                        sentiment = row["sentiment_do_better"]
                        emotion = row["emotion_do_better"]

                        # with st.chat_message("user"):
                        #     st.markdown(f"**{rating}** `{time_}`")

                        #     if str(do_better) not in ["nan"]:
                        #         st.markdown("💡 " + str(do_better))
                        #     # st.markdown(f"`{sentiment} {score}` `{cat}`")
                        #     ui.badges(badge_list=[(f"{cat}", "outline"), (f"{sentiment}", "secondary"), (f"{score}", "outline")], class_name=f"badges_improve_{icounter}")

                        with st.container(border=True):
                        # st.markdown(f"**{rating}** `{time_}`")
                            ui.badges(badge_list=[(f"{icounter}", "destructive"), (f"{rating}", "default"), (f"{time_}", "secondary")], class_name=f"badges_improve_head_{icounter}")
                            if str(do_better) not in ["nan"]:
                                st.markdown("🛠️ " + str(do_better))
                                # st.markdown(f"`{sentiment} {score}` `{cat}`")
                                ui.badges(badge_list=[(f"{emotion}", "outline"), (f"{cat}", "outline"), (f"{sentiment}", "secondary"), (f"{score}", "secondary")], class_name=f"badges_improve_{icounter}")

                        icounter += 1
    # -- Word Cloud --------------------------------------------------------------------------------------------- Word Cloud
    elif page == "Word Cloud":
        st.markdown(
            "# :material/cloud: Word Cloud"
        )
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
        st.markdown("# :material/robot_2: GPT-4o Summary")

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
                model="gpt-4o-mini",
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
            webhook_url = "https://hook.eu1.make.com/37kqw0pre1scl6tslnguib5cwbpft4rn"
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
                        ui.badges(
                        badge_list=[
                    (f"⚡️Webhook sent successfully!", "outline"),
                ],
                class_name="flex gap-2",
                key="badges50",
            )

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
        st.markdown(
            "# :material/table: Dataframe"
        )

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
        st.write("")
        with st.container(height=600):
            st.help(pd.DataFrame)

    # -- Reports --------------------------------------------------------------------------------------------------- Reports
    elif page == "Reports":
        st.markdown("# :material/sim_card_download: Reports")
        st.toast("**Reports Updated** now included insights generated with Groq LLM.", icon=":material/robot_2:")
        st.write("")
        ui.badges(
            badge_list=[("NEW", "destructive"), ("beta v1.0.3", "outline")],
            class_name="flex gap-2",
            key="badges_soon",
        )

        # Only proceed with month and year selection if a specific surgery is selected  -------------Month and Year Selector
        if page not in ["**:blue-background[PCN Dashboard]**", "**About**"] and selected_surgery:
            surgery_data = pcn_data[pcn_data["surgery"] == selected_surgery]

            if not surgery_data.empty:
                # Find the earliest and latest dates in the data
                min_date = surgery_data["time"].min().to_pydatetime()
                max_date = surgery_data["time"].max().to_pydatetime()

                # Generate a list of years and months based on the data range
                years = list(range(min_date.year, max_date.year + 1))
                months = [
                    datetime.strftime(datetime(2023, i, 1), "%B") for i in range(1, 13)
                ]

                # Create two columns for selectors
                col1, col2 = st.columns(2)

                # Create year and month selectors inside the columns
                with col1:
                    selected_year = st.selectbox(
                        "Select the **Year**", options=years, index=years.index(max_date.year)
                    )

                # Adjust month options based on selected year
                if selected_year == min_date.year:
                    months = months[min_date.month - 1 :]
                if selected_year == max_date.year:
                    months = months[: max_date.month]

                with col2:
                    selected_month = st.selectbox(
                        "Select the **Month**", options=months, index=0
                    )

                # Convert selected month to number
                selected_month_number = datetime.strptime(selected_month, "%B").month

                # Filter data based on the selected month and year
                filtered_data = surgery_data[
                    (surgery_data["time"].dt.year == selected_year)
                    & (surgery_data["time"].dt.month == selected_month_number)
                ]
                pcn_filtered_data = pcn_data[
                    (pcn_data["time"].dt.year == selected_year)
                    & (pcn_data["time"].dt.month == selected_month_number)
                ]


        # Your existing setup code...

        if st.button("Generate AI MedReview Report"):
            try:
                with st.spinner("Generating Report..."):
                # Call the function with the parameters from Streamlit widgets
                    simple_pdf(
                        filtered_data, pcn_filtered_data, selected_month, selected_year, selected_surgery, selected_pcn, 'rating'
                    )

                # Inform the user of success

                ui.badges(
                    badge_list=[(f"Report generated successfully! - {selected_month} {selected_year}", "default")],
                    class_name="flex gap-2",
                    key="badges_success",
                )

                # Provide a download link for the generated PDF
                with open("reports/report.pdf", "rb") as file:
                    st.download_button(
                        label="Download AI MedReview Report",
                        data=file,
                        file_name="reports/report.pdf",
                        mime="application/pdf",
                    )
            except AttributeError as e:
                st.warning(f"Cannot generate report {e}")



    # -- About ------------------------------------------------------------------------------------------------------- About
    elif page == "**About**":
        st.markdown("# :material/info: About")



        st.markdown("### Patient Feedback Analysis in Healthcare")
        st.markdown("Welcome to **AI MedReview**, your powerful new dashboard for elevating healthcare providers' understanding and utilization of patient feedback. Our solution focuses on the essential Friends and Family Test (FFT), empowering you to extract deeper insights from this invaluable data source. At the core of AI MedReview lies a transformative approach that goes beyond mere quantitative metrics. By leveraging natural language processing and machine learning techniques, we unlock the nuanced sentiments behind patient responses. Our dashboard assigns detailed scores to each piece of feedback, painting a more comprehensive picture of patient satisfaction levels.")
        with st.popover("Listern to **Podcast** -  Sept 2024", icon=":material/mic:"):
            st.image('images/podcast2.png')
            st.audio('images/ai-medreview_podcast.wav')

        st.markdown("Through **sentiment analysis** powered by Hugging Face's `cardiffnlp/twitter-roberta-base-sentiment-latest model`, we precisely determine the emotional tone of patient comments, be it positive, negative, or neutral. This level of granular understanding enables you to celebrate areas of excellence and swiftly identify opportunities for improvement.")
        st.markdown("But we don't stop there. To protect patient privacy, we employ robust named **entity recognition (NER)** capabilities, utilizing the Hugging Face `dbmdz/bert-large-cased-finetuned-conll03-english model`. This ensures any personally identifiable information (PII) is seamlessly anonymized, safeguarding the confidentiality of your valuable data.")
        st.markdown("Furthermore, our innovative **zero-shot classification** approach, powered by the Facebook `BART-large-mnli` architecture, allows us to categorize patient feedback with remarkable accuracy – even without specialized healthcare training data. By carefully curating our classification labels, we achieved a striking 0.91 accuracy, demonstrating the remarkable versatility of this model.")
        st.markdown("This comprehensive suite of advanced analytics empowers healthcare providers like yourself to move beyond mere data presentation and unlock a clearer, more actionable understanding of patient experiences. Armed with these insights, you can drive continuous improvements, elevate service quality, and enhance patient outcomes.")
        st.markdown("Explore the AI MedReview dashboard today and experience the transformative power of data-driven decision-making in healthcare.")

        st.markdown("## Change Log")
        with st.container(height=400, border=True):
            st.markdown("### version 2.2.1")
            st.markdown("**GROQ LLM Summaries added for All monthly Feedback.** Integrated GROQ LLM with the `llama-3.1-8b-instant` model for generating summaries in all monthly reports. The AI-powered summaries now provide insights into key trends and suggest areas for improvement.")
            st.markdown("""**PARETO ANALYSIS Plots**  In the Feedback Classification section of app you will see a new tab **Pareto Analysis**. A Pareto plot is created here from feedback classification information. There is an overview of how to interoperate the plot and some suggestions for evidence backed interventions.

**Improvments made to**
- Campaigns
- NER People
- Reports
- About""")
            st.markdown("### version 2.2.0")
            st.markdown("**GROQ LLM Summaries / Insights** introduced across the app. You can now generate insights from feedback in the following pages: NER People, Campaigns, Feedback Classification and Improvmeent Suggestions.")

        st.markdown("![GitHub](https://img.icons8.com/material-outlined/24/github.png) [AI MedReview on GitHub](https://github.com/janduplessis883/ai-medreview), where collaboration and contributions are warmly welcomed.")
        st.markdown("![Privicy](https://img.icons8.com/material/24/privacy--v1.png) [Privacy Notice & DPIA](https://janduplessis.notion.site/AI-MedReview-Privacy-Notice-52e518a957d04446a5aa5397018ea92d?pvs=4)")

        st.write("")
        st.write("")
        st.write("")

        debug_toggle = ui.switch(default_checked=False, label="Debug", key="debug")
        if debug_toggle:
            st.dataframe(data.tail(50))
        st.write("")
        st.write("")
        with st.expander(label="Leave Feedback"):
            form_url = (
                "https://tally.so/r/w2ed0e"  # Replace this URL with the URL of your form
            )
            iframe_code = f'<iframe src="{form_url}" width="100%" height="400"></iframe>'
            st.markdown(iframe_code, unsafe_allow_html=True)

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
        st.write("")


    # -- About ------------------------------------------------------------------------------------------------------- NER People
    elif page == "NER People":
        st.markdown("# :material/psychology_alt: NER - People")
        st.markdown("**Named Entity Recognistion** - People")
        st.toast("**NER People** De-anonymized reviews. **New** Summarize with Groq.", icon=":material/lock:")
        # Set the correct PIN

        # Free Test
        free_text_per = filtered_data.dropna(subset='free_text_PER')
        do_better_per = filtered_data.dropna(subset='do_better_PER')

        free_text_list = free_text_per['free_text_PER'].unique().tolist()
        selected_names = st.multiselect("Select People", options=free_text_list)

        filtered_free_text_per = free_text_per[free_text_per['free_text_PER'].isin(selected_names)]
        filtered_do_better_per = do_better_per[do_better_per['do_better_PER'].isin(selected_names)]

        if len(selected_names) > 0:

            with st.container(height=550, border=True):

                st.html("<div class='status' style='background-color: #f2a947; color: white; padding-top: 2px; padding-bottom: 2px; padding-left: 7px; padding-right: 7px; border-radius: 6px; font-family: Arial, sans-serif; font-size: 18px; display: inline-block; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);'><b>Free-Text Feedback</b> - NER People</div>")
                for index, row in filtered_free_text_per.iterrows():

                    with st.container(border=True):
                        date = row['time']
                        free_text = row['free_text']
                        names = row['free_text_PER']
                        sentiment = row['sentiment_free_text']
                        st.write(free_text)
                        ui.badges(badge_list=[(f"{names}", "default"), (f"{date}", "outline"), (f"{sentiment}", "secondary")], key=f"free_text_name_{index}")

                st.divider()
                st.html("<div class='status' style='background-color: #f2a947; color: white; padding-top: 2px; padding-bottom: 2px; padding-left: 7px; padding-right: 7px; border-radius: 6px; font-family: Arial, sans-serif; font-size: 18px; display: inline-block; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);'><b>Improvement Suggestions</b> - NER People</div>")
                for index, row in filtered_do_better_per.iterrows():

                    with st.container(border=True):
                        date = row['time']
                        do_better = row['do_better']
                        names2 = row['do_better_PER']
                        sentiment2 = row['sentiment_do_better']
                        st.write(do_better)
                        ui.badges(badge_list=[(f"{names2}", "default"),(f"{date}", "outline"), (f"{sentiment2}", "secondary")], key=f"do_better_name_{index}")
                st.divider()
            # Create a download button for selected entries outside of the container
            selected_entries = "\n\n".join([f"Free Text Feedback:\n{row['free_text']}\nNames: {row['free_text_PER']}\nDate: {row['time']}\nSentiment: {row['sentiment_free_text']}" for _, row in filtered_free_text_per.iterrows()])
            selected_entries += "\n\n" + "\n\n".join([f"Improvement Suggestion:\n{row['do_better']}\nNames: {row['do_better_PER']}\nDate: {row['time']}\nSentiment: {row['sentiment_do_better']}" for _, row in filtered_do_better_per.iterrows()])

            st.download_button(label="Download Selected Entries", data=selected_entries, file_name=f"Selected_NER_People_{selected_surgery}.txt", mime="text/plain", icon=":material/download:")

            with st.popover("Summarize Personal feedback", icon=":material/robot_2:"):
                llm_button = st.button("Summarize with LLM", icon=":material/robot_2:")
                if llm_button == True:
                    summary = ask_groq(f"Summarize the following personal feedback, PERSON is a NER placeholder for a person's name, individuals are listed below the feedback in the order they appear in the feedback text, summarize the following feedback highlighting trends and list the name of the person the feedback relates to (all the people mentioned are members of staff): {selected_entries}")
                    st.markdown(summary)

        else:
            ui.badges(
                    badge_list=[
                        ("There’s nothing to display at the moment. Select names from the dropdown list.", "outline")
                    ],
                    class_name="flex gap-2",
                    key="badges113empty",
                )


        # -- About ------------------------------------------------------------------------------------------------------- Campaings
    elif page == "Campaigns":
        st.markdown("# :material/campaign: Campaigns")

        campaign_df = filtered_data[['campaing_id', 'campaign_rating', 'campaign_freetext']]
        campaign_df = campaign_df.dropna(subset=['campaing_id'])
        campaign_df = campaign_df[campaign_df['campaign_rating'] != 0]
        campaig_df_freetext = campaign_df.dropna(subset=['campaign_freetext'])
        campaig_df_freetext = campaig_df_freetext.sort_values(by='campaign_rating', ascending=False)
        campaign_rating_mean = str(round(campaign_df['campaign_rating'].mean() * 20, 1))

        pcn_campaign_df = pcn_data[['campaing_id', 'campaign_rating']]
        pcn_campaign_df = pcn_campaign_df.dropna(subset=['campaing_id'])
        pcn_campaign_df = pcn_campaign_df[pcn_campaign_df['campaign_rating'] != 0]
        pcn_campaign_rating_mean = str(round(pcn_campaign_df['campaign_rating'].mean() * 20, 1))

        # Prepare list of campaigns
        campaign_list = list(campaign_df['campaing_id'].unique())
        # remove NaN from the list
        campaign_list = list(filter(lambda x: not (isinstance(x, float) and np.isnan(x)), campaign_list))

        with st.container(height=300, border=False):
            st.html("""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f8f8f8;
                padding: 20px;
            }
            .timeline {
                max-width: 600px;
                margin: 0 auto;
            }
            .day {
                color: #e11d48;
                font-weight: bold;
                margin-bottom: 20px;
            }
            .event {
                border-left: 3px solid #eab308;
                padding-left: 20px;
                margin-bottom: 20px;
                position: relative;
            }
            .event:before {
                content: '';
                width: 10px;
                height: 10px;
                background-color: #eab308;
                border-radius: 50%;
                position: absolute;
                left: -7px;
                top: 5px;
            }
            .time {
                font-size: 0.9em;
                color: #9ca3af;
            }
            .id {
                font-size: 0.9em;
                color: #364153;
            }
            .time_placeholder {
                font-size: 0.9em;
                color: #d1d5db;;
            }
            .title {
                font-size: 1.1em;
                color: #334155;
                font-weight: bold;
            }
            .placeholder {
            font-size: 1.1em;
            color: #d1d5db;
            font-weight: bold;
        }
        </style>
        <title>Timeline Example</title>
    </head>
    <body>

    <div class="timeline">
        <div class="day">1 September 2024</div>
        <div class="event">
        <div class="title">Influenza Campaign 24/25</div>
        <div class="id">ID: flu_campaign_24</div>
        <div class="time">Influenza - GP Surgeries</div>
        <div class="time">RSV - GP Surgeries</div>
        <div class="time">Covid-19 & Housbound Influenza - GP Federation</div>
        <div class="time_placeholder">.</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 December 2024</div>
        <div class="event">
        <div class="title">NHS App Survey 24</div>
        <div class="id">ID: nhs_app_24</div>
        <div class="time">Category of poor preformance during GP Patient Survey 23/24</div>
        <div class="time">Appointment booking & Repeat Prescribing</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 February 2025</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 April 2025</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 June 2025</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 August 2025</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 September 2025</div>
        <div class="event">
        <div class="placeholder">Influenza Campaign 25/26</div>
        <div class="time_placeholder">Influenza - GP Surgeries</div>
        <div class="time_placeholder">RSV - GP Surgeries</div>
        <div class="time_placeholder">Covid-19 & Housbound Influenza - GP Federation</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 Decemmber 2025</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 February 2026</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 April 2026</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 June 2026</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 August 2026</div>
        <div class="event">
        <div class="placeholder">Future Campaign</div>
        <div class="time_placeholder">.</div>
        </div>

        <div class="day">1 September 2026</div>
        <div class="event">
        <div class="placeholder">Influenza Campaign 26/27</div>
        <div class="time_placeholder">Influenza - GP Surgeries</div>
        <div class="time_placeholder">RSV - GP Surgeries</div>
        <div class="time_placeholder">Covid-19 & Housbound Influenza - GP Federation</div>
        <div class="time_placeholder">.</div>
        </div>

    </body>
    </html>
                    """)
        st.container(height=8, border=False)
        if len(campaign_list) != 0:

            selected_campaign = st.selectbox("Select a **Campaign**", options=campaign_list, index=0)
            st.markdown(f"# {selected_campaign}")


            st.metric(f"Campaign Satisfaction: {selected_surgery}", value=campaign_rating_mean + "%", delta=f"{pcn_campaign_rating_mean}% - {selected_pcn}", delta_color="off")

            # Plot seaborn histogram of campaign ratings
            try:
                fig, ax = plt.subplots(figsize=(12, 3))
                sns.histplot(campaign_df['campaign_rating'], kde=True, bins=5, ax=ax, color="#f2a947")
                ax.set_title('Distribution of Campaign Ratings')
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
                plt.tight_layout()
                ax.set_xlabel('Campaign Rating')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            except:
                st.warning("Could not display histogram.")

            try:

                text2 = " ".join(campaign_df["campaign_freetext"].dropna())
                wordcloud = WordCloud(background_color="white", colormap="Oranges").generate(text2)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wordcloud, interpolation="bilinear")
                ax2.axis("off")
                st.pyplot(fig2)

                st.divider()
                st.subheader("Campaign Free Text")
                with st.container(height=400, border=True):
                    for _, row in campaig_df_freetext.iterrows():
                        rating = row['campaign_rating']
                        text = row['campaign_freetext']
                        st.write(int(rating), text)
            except:
                ui.badges(
                    badge_list=[
                        ("Cannot display word cloud, insufficient data.", "outline")
                    ],
                    class_name="flex gap-2",
                    key="badges113",
                )
            # Aggregate all campaign free text for LLM summarization
            aggregated_freetext = " ".join(campaign_df['campaign_freetext'].dropna())

            with st.popover(f"Summarize **{selected_campaign}** feedback", icon=":material/robot_2:"):
                llm_button = st.button("Summarize with LLM", icon=":material/robot_2:")
                if llm_button == True:
                    summary = ask_groq(f"Summarize the following campaign feedback for {selected_campaign}, highlighting Positive and Negative trends: {aggregated_freetext}")
                    st.markdown(summary)

        else:
            ui.badges(
                badge_list=[
                    ("No campaign data captured for this practice.", "outline")
                ],
                class_name="flex gap-2",
                key="badges1134",
            )
