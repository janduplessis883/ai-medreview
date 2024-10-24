import re
import unicodedata

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from fpdf import FPDF
from wordcloud import WordCloud
import streamlit as st
from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

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


def generate_sns_countplot(df, column, filename="reports/rating.png"):
    # Create a Seaborn count plot
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
    sns.countplot(data=df, y="rating", order=order, palette=palette)
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
                f"{int(width)} / {round((int(width)/df.shape[0])*100, 1)}%",
                va="center",
                fontsize=12,
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
    ax.set_title('Feedback Ratings', fontweight='bold', fontsize=12)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()

def recommendation_plot(recomended, not_recomended, pcn_recomended, pcn_not_recomended, filename):
    # Data
    categories = ['Recommended', 'Not Recommended']
    your_data = [recomended, not_recomended]
    pcn_avg_data = [pcn_recomended, pcn_not_recomended]

    # Bar width
    bar_width = 0.35

    # Bar positions
    r1 = np.arange(len(categories))
    r2 = [x + bar_width for x in r1]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 5))

    # Create bars
    bars1 = ax.bar(r1, your_data, color='#a3b638', width=bar_width, edgecolor='grey', label='Your Data')
    bars2 = ax.bar(r2, pcn_avg_data, color='#e78531', width=bar_width, edgecolor='grey', label='PCN Average')

    # Add text annotations
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontsize=12)

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontsize=12)

    # Add labels, title and custom x-axis tick labels
    ax.set_xlabel('Categories', fontsize=10)
    ax.set_ylabel('Percentage', fontsize=10)
    ax.set_title('Recommendation vs PCN Average', fontweight='bold', fontsize=12)
    ax.set_xticks([r + bar_width / 2 for r in range(len(categories))])
    ax.set_xticklabels(categories)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.xaxis.grid(False)
    plt.tight_layout()

    # Add legend
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_daily_count(df, filename='reports/daily_count.png'):
    daily_count = df.resample("D", on="time").size()
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

    plt.savefig(filename)
    plt.close()


def strip_emojis(text):
    # Remove emojis using regex
    text_without_emojis = re.sub(r'[^\x00-\x7F]+', '', text)

    # Convert to 'latin1' encoding, replacing unencodable characters with a replacement marker
    encoded_text = unicodedata.normalize('NFKD', text_without_emojis).encode('ascii', errors='ignore')

    return encoded_text.decode('ascii')  # Decode back to string

def col_to_list(df, colname):
    df.dropna(subset=colname, inplace=True)
    return df[colname].to_list()

def send_webhook(url, surgery, month, year):
    payload = {
        'surgery': surgery,
        'month': month,
        'year': year
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, json=payload, headers=headers)
    return response

def create_wordcloud(df, col_name, filename='reports/wordcloud1.png', colors='Blues'):
        text = " ".join(df[col_name].dropna())
        wordcloud = WordCloud(background_color="white", colormap=colors).generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        plt.savefig(filename)
        plt.close()



def simple_pdf(df, pcn_df, selected_month, selected_year, selected_surgery, selected_pcn, plot_column):
    # Generate the Seaborn count plot
    plot_filename = "reports/rating.png"
    generate_sns_countplot(df, plot_column, plot_filename)
    plot_daily_count(df)
    total_feedback_count = df.shape[0]
    rating_value_counts = df['rating'].value_counts()
    try:
        create_wordcloud(df, 'free_text', filename='reports/wordcloud1.png', colors='Blues')
        display_wc1 = True
    except ValueError as e:
        display_wc1 = False
    try:
        create_wordcloud(df, 'do_better', filename='reports/wordcloud2.png', colors='Reds')
        display_wc2 = True
    except ValueError as e:
        display_wc2 = False

    send_webhook('https://hook.eu1.make.com/nqpv7r14si8vu0qbv3eqw1r6jutrge6r', selected_surgery, selected_month, selected_year)
    # Pivit DF to cature rating categories
    categories = [
        "Very good",
        "Good",
        "Neither good nor poor",
        "Poor",
        "Very poor",
        "Don't know",
    ]



    rating_value_counts = df['rating'].value_counts().reindex(categories, fill_value=0)
    vg_count = rating_value_counts['Very good']
    g_count = rating_value_counts['Good']
    nn_count = rating_value_counts['Neither good nor poor']
    p_count = rating_value_counts['Poor']
    vp_count = rating_value_counts['Very poor']
    dk_count = rating_value_counts["Don't know"]
    recomended = round(((vg_count + g_count) / (vg_count + g_count + nn_count + p_count + vp_count + dk_count)) * 100, 1)
    not_recomended = round(((p_count + vp_count) / (vg_count + g_count + nn_count + p_count + vp_count + dk_count)) * 100, 1)


    pcn_rating_value_counts = pcn_df['rating'].value_counts().reindex(categories, fill_value=0)
    pcn_vg_count = pcn_rating_value_counts['Very good']
    pcn_g_count = pcn_rating_value_counts['Good']
    pcn_nn_count = pcn_rating_value_counts['Neither good nor poor']
    pcn_p_count = pcn_rating_value_counts['Poor']
    pcn_vp_count = pcn_rating_value_counts['Very poor']
    pcn_dk_count = pcn_rating_value_counts["Don't know"]
    pcn_recomended = round(((pcn_vg_count + pcn_g_count) / (pcn_vg_count + pcn_g_count + pcn_nn_count + pcn_p_count + pcn_vp_count + pcn_dk_count)) * 100, 1)
    pcn_not_recomended = round(((pcn_p_count + pcn_vp_count) / (pcn_vg_count + pcn_g_count + pcn_nn_count + pcn_p_count + pcn_vp_count + pcn_dk_count)) * 100, 1)

    recommendation_plot(recomended, not_recomended, pcn_recomended, pcn_not_recomended, "reports/recommendation.png")
    # Create the PDF
    pdf = FPDF()
    # Set document metadata
    pdf.set_title(f"AI MedReview: FFT Monthly Report - {selected_surgery} {selected_month} {selected_year}")
    pdf.set_author("Jan du Plessis")
    pdf.set_subject("Monthly Medical Review Report")
    pdf.set_keywords("AIMedReview, Medical, Report, Monthly, FFT")
    pdf.set_creator("AI MedReview System")

    pdf.add_page()

    # Header "AI MedReview" with Arial in bold
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(39, 69, 98)
    pdf.cell(0, 5, "AI MedReview: FFT Analysis", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 10)
    info_string2 = f"{selected_pcn.replace('-', ' ')}"
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, info_string2, 0, 1)

    # Additional info in Arial, not bold
    pdf.set_font("Arial", "B", 20)
    info_string = f"{selected_surgery.replace('-', ' ')}"
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 10, info_string, 0, 1)

    pdf.set_font("Arial", "B", 16)
    info_string = f"{selected_month} {selected_year}"
    pdf.cell(0, 10, info_string, 0, 1)

    # Insert a horizontal line after the header
    pdf.set_line_width(0.2)
    pdf.line(10, 45, 200, 45)  # (x1, y1, x2, y2)

    pdf.ln(7)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(197, 58, 50)
    pdf.cell(0, 10, "SECTION 1: Recommendation % and Rating Counts", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 10, f"The total feedback received during {selected_month} {selected_year} was {total_feedback_count}.", 0, 1)

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, f"Recommended - {recomended}%  (PCN Average - {pcn_recomended}%)", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, f"Not Recommended - {not_recomended}%  (PCN Average - {pcn_not_recomended}%)", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.image("reports/recommendation.png", x=10, y=85, w=180)  # Adjust x, y, w as necessary
    pdf.image('images/nhs_scoring.png', x=50, y=165, w=100)

    pdf.image(plot_filename, x=10, y=195, w=180)  # Adjust x, y, w as necessary


    pdf.add_page()


    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(197, 58, 50)
    pdf.cell(0, 10, "SECTION 2: Response Rate", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.image("reports/daily_count.png", x=10, y=25, w=180)  # Adjust x, y, w as necessary


    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(197, 58, 50)
    pdf.cell(0, 10, "SECTION 3: Feedback - Responses", 0, 1)  # '0' for cell width, '1' for the new line


    pdf.set_font("Arial", "", 8)
    pdf.set_text_color(35, 37, 41)

    text_list = col_to_list(df, 'free_text')
    all_feedback = ''
    for index, text in enumerate(text_list):
        pdf.multi_cell(0, 4, f"{index}: {strip_emojis(text)}")
        all_feedback = all_feedback + f"{index} - {text} "

    pdf.ln(8)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(39, 69, 98)
    pdf.cell(0, 10, "Feedback Insights by Groq LLM", 0, 1)

    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(39, 69, 98)
    pdf.multi_cell(0, 4, ask_groq(f"Summarize this GP Surgery feedback, identifying positive and negative trends: {all_feedback}, your output should be plain text only, don't use markdown in your output.").replace("*", "").replace("#", ""))


    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(197, 58, 50)
    pdf.cell(0, 10, "SECTION 4: Improvement Suggestions - Responses", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 8)
    pdf.set_text_color(35, 37, 41)

    text_list2 = col_to_list(df, 'do_better')
    all_improvement = ''
    for index, text in enumerate(text_list2):
        pdf.multi_cell(0, 4, f"{index}: {strip_emojis(text)}")
        all_improvement = all_improvement + f"{index} - {text} "

    pdf.ln(8)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(39, 69, 98)
    pdf.cell(0, 10, "Improvement Suggestions Insights by Groq LLM", 0, 1)

    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(39, 69, 98)
    pdf.multi_cell(0, 4, ask_groq(f"Summarize this GP Surgery improvement suggestions, identifying trends: {all_improvement}, your output should be plain text only, don't use markdown in your output.").replace("*", "").replace("#", ""))

    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(197, 58, 50)
    pdf.cell(0, 10, "SECTION 5: Word Clouds", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, f"Feedback Free-Text (Blue)", 0, 1)

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, f"Improvement Suggestions (Red)", 0, 1)

    if display_wc1:
        pdf.image("reports/wordcloud1.png", x=10, y=20, w=180)
    if display_wc2:
        pdf.image("reports/wordcloud2.png", x=10, y=120, w=180)

    # Output the PDF
    pdf.output("reports/report.pdf", "F")
