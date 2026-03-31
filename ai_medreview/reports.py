import re
import unicodedata
from datetime import datetime

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
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_resource
def ask_groq(prompt: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
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
    ax.set_title("Feedback Ratings", fontweight="bold", fontsize=12)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()


def recommendation_plot(
    recomended, not_recomended, pcn_recomended, pcn_not_recomended, filename
):
    # Data
    categories = ["Recommended", "Not Recommended"]
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
    bars1 = ax.bar(
        r1,
        your_data,
        color="#a3b638",
        width=bar_width,
        edgecolor="grey",
        label="Your Data",
    )
    bars2 = ax.bar(
        r2,
        pcn_avg_data,
        color="#e78531",
        width=bar_width,
        edgecolor="grey",
        label="PCN Average",
    )

    # Add text annotations
    for bar in bars1:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    for bar in bars2:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Add labels, title and custom x-axis tick labels
    ax.set_xlabel("Categories", fontsize=10)
    ax.set_ylabel("Percentage", fontsize=10)
    ax.set_title("Recommendation vs PCN Average", fontweight="bold", fontsize=12)
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


def plot_daily_count(df, filename="reports/daily_count.png"):
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
    ax_title.set_position((1, 1))  # Adjust these values to align your title as needed
    plt.xlabel("")
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()


def strip_emojis(text):
    # Remove emojis using regex
    text_without_emojis = re.sub(r"[^\x00-\x7F]+", "", text)

    # Convert to 'latin1' encoding, replacing unencodable characters with a replacement marker
    encoded_text = unicodedata.normalize("NFKD", text_without_emojis).encode(
        "ascii", errors="ignore"
    )

    return encoded_text.decode("ascii")  # Decode back to string


def pdf_safe_text(text):
    if text is None:
        return ""

    normalized_text = str(text).translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2026": "...",
                "\xa0": " ",
            }
        )
    )
    return strip_emojis(normalized_text)


def col_to_list(df, colname):
    return df.dropna(subset=[colname])[colname].to_list()


def safe_percentage(numerator, denominator):
    if not denominator:
        return 0.0
    return round((numerator / denominator) * 100, 1)


class AIReportPDF(FPDF):
    NAVY = (21, 46, 74)
    BLUE = (79, 124, 172)
    GOLD = (202, 168, 89)
    ORANGE = (224, 129, 55)
    RED = (197, 58, 50)
    INK = (35, 37, 41)
    SLATE = (100, 116, 139)
    LIGHT = (244, 247, 250)
    BORDER = (221, 226, 232)
    SOFT_BORDER = (232, 236, 241)

    def __init__(self):
        super().__init__()
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=16)
        self.report_title = "AI MedReview Report"
        self.generated_at = datetime.now().strftime("%d %b %Y")
        self._suppress_header = False

    @property
    def content_width(self):
        return self.w - self.l_margin - self.r_margin

    def header(self):
        if self._suppress_header:
            return

        self.set_fill_color(*self.NAVY)
        self.rect(0, 0, self.w, 16, "F")
        self.set_xy(self.l_margin, 5)
        self.set_font("Arial", "B", 10)
        self.set_text_color(255, 255, 255)
        self.cell(0, 4, pdf_safe_text(self.report_title), 0, 1, "L")
        self.set_y(24)

    def footer(self):
        if self._suppress_header:
            return

        self.set_y(-12)
        self.set_draw_color(*self.BORDER)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(-10)
        self.set_font("Arial", "", 8)
        self.set_text_color(*self.SLATE)
        self.cell(0, 4, pdf_safe_text(f"Generated {self.generated_at}"), 0, 0, "L")
        self.cell(0, 4, pdf_safe_text(f"Page {self.page_no()}/{{nb}}"), 0, 0, "R")

    def add_cover_page(self, surgery, pcn, month, year, metrics):
        self._suppress_header = True
        self.add_page()

        self.set_fill_color(*self.NAVY)
        self.rect(0, 0, self.w, 65, "F")
        self.set_fill_color(*self.ORANGE)
        self.rect(0, 65, self.w, 4, "F")

        self.set_xy(self.l_margin, 16)
        self.set_font("Arial", "B", 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 6, pdf_safe_text("AI MedReview"), 0, 1)

        self.set_x(self.l_margin)
        self.set_font("Arial", "B", 24)
        self.multi_cell(0, 10, pdf_safe_text(surgery.replace("-", " ")))

        self.set_x(self.l_margin)
        self.set_font("Arial", "", 14)
        self.cell(0, 8, pdf_safe_text(f"FFT Monthly Report - {month} {year}"), 0, 1)

        self.ln(14)
        self.draw_info_box(
            "Report Overview",
            [
                f"Primary Care Network: {pcn.replace('-', ' ')}",
                f"Reporting period: {month} {year}",
                f"Generated on: {self.generated_at}",
            ],
        )
        self.ln(6)
        self.draw_metric_cards(metrics)
        self._suppress_header = False

    def add_section_heading(self, title, subtitle=None):
        self.set_font("Arial", "B", 16)
        self.set_text_color(*self.NAVY)
        self.cell(0, 8, pdf_safe_text(title), 0, 1)
        self.set_draw_color(*self.ORANGE)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), self.l_margin + 34, self.get_y())
        self.ln(4)
        if subtitle:
            self.set_font("Arial", "", 10)
            self.set_text_color(*self.SLATE)
            self.multi_cell(0, 5, pdf_safe_text(subtitle))
            self.ln(2)

    def draw_info_box(self, title, lines):
        self.set_font("Arial", "B", 11)
        self.set_text_color(*self.NAVY)
        self.set_fill_color(*self.LIGHT)
        self.set_draw_color(*self.SOFT_BORDER)
        self.set_line_width(0.15)
        self.cell(0, 8, pdf_safe_text(title), 1, 1, "L", True)
        self.set_font("Arial", "", 10)
        self.set_text_color(*self.INK)
        for line in lines:
            self.cell(0, 6, pdf_safe_text(line), "LR", 1, "L", True)
        self.cell(0, 2, "", "LRB", 1, "L", True)

    def draw_metric_cards(self, metrics):
        gap = 4
        card_width = (self.content_width - gap * (len(metrics) - 1)) / len(metrics)
        card_height = 24
        start_x = self.l_margin
        start_y = self.get_y()

        for index, metric in enumerate(metrics):
            x = start_x + index * (card_width + gap)
            y = start_y
            fill = metric.get("fill", self.LIGHT)
            text_color = metric.get("text_color", self.NAVY)

            self.set_fill_color(*fill)
            self.set_draw_color(*self.SOFT_BORDER)
            self.set_line_width(0.15)
            self.rect(x, y, card_width, card_height, "DF")
            self.set_xy(x + 3, y + 3)
            self.set_font("Arial", "B", 9)
            self.set_text_color(*text_color)
            self.cell(card_width - 6, 4, pdf_safe_text(metric["label"]), 0, 1)
            self.set_xy(x + 3, y + 10)
            self.set_font("Arial", "B", 16)
            self.cell(card_width - 6, 6, pdf_safe_text(metric["value"]), 0, 1)
            self.set_xy(x + 3, y + 18)
            self.set_font("Arial", "", 8)
            self.cell(card_width - 6, 3, pdf_safe_text(metric["caption"]), 0, 1)

        self.set_xy(self.l_margin, start_y + card_height + 6)

    def add_chart(self, title, image_path, *, subtitle=None, image_height=78):
        self.set_font("Arial", "B", 12)
        self.set_text_color(*self.INK)
        self.cell(0, 6, pdf_safe_text(title), 0, 1)
        if subtitle:
            self.set_font("Arial", "", 9)
            self.set_text_color(*self.SLATE)
            self.multi_cell(0, 4.5, pdf_safe_text(subtitle))
        self.ln(2)
        image_width = self.content_width
        self.image(image_path, x=self.l_margin, y=self.get_y(), w=image_width, h=image_height)
        self.ln(image_height + 6)

    def add_insight_box(self, title, body):
        self.set_font("Arial", "B", 11)
        self.set_text_color(*self.BLUE)
        self.set_fill_color(236, 243, 249)
        self.set_draw_color(224, 232, 240)
        self.set_line_width(0.15)
        self.cell(0, 8, pdf_safe_text(title), 1, 1, "L", True)
        self.set_font("Arial", "", 10)
        self.set_text_color(*self.INK)
        self.multi_cell(0, 5, pdf_safe_text(body), 1, "L", True)
        self.ln(4)

    def add_response_entries(self, title, responses):
        self.set_font("Arial", "B", 11)
        self.set_text_color(*self.INK)
        self.cell(0, 6, pdf_safe_text(title), 0, 1)
        self.set_font("Arial", "", 9)
        self.set_text_color(*self.INK)

        for index, text in enumerate(responses, start=1):
            self.set_fill_color(*self.LIGHT)
            self.set_draw_color(*self.SOFT_BORDER)
            self.set_line_width(0.15)
            self.cell(10, 7, str(index), 1, 0, "C", True)
            self.multi_cell(self.content_width - 10, 7, pdf_safe_text(text), 1, "L", False)



def send_webhook(url, surgery, month, year):
    payload = {"surgery": surgery, "month": month, "year": year}

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    return response


def create_wordcloud(df, col_name, filename="reports/wordcloud1.png", colors="Blues"):
    text = " ".join(df[col_name].dropna())
    wordcloud = WordCloud(background_color="white", colormap=colors).generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.savefig(filename)
    plt.close()


def simple_pdf(
    df,
    pcn_df,
    selected_month,
    selected_year,
    selected_surgery,
    selected_pcn,
    plot_column,
):
    # Generate the Seaborn count plot
    plot_filename = "reports/rating.png"
    generate_sns_countplot(df, plot_column, plot_filename)
    plot_daily_count(df)
    total_feedback_count = df.shape[0]
    rating_value_counts = df["rating"].value_counts()
    try:
        create_wordcloud(
            df, "free_text", filename="reports/wordcloud1.png", colors="Blues"
        )
        display_wc1 = True
    except ValueError as e:
        display_wc1 = False
    try:
        create_wordcloud(
            df, "do_better", filename="reports/wordcloud2.png", colors="Reds"
        )
        display_wc2 = True
    except ValueError as e:
        display_wc2 = False

    send_webhook(
        "https://hook.eu1.make.com/nqpv7r14si8vu0qbv3eqw1r6jutrge6r",
        selected_surgery,
        selected_month,
        selected_year,
    )
    # Pivit DF to cature rating categories
    categories = [
        "Very good",
        "Good",
        "Neither good nor poor",
        "Poor",
        "Very poor",
        "Don't know",
    ]

    rating_value_counts = df["rating"].value_counts().reindex(categories, fill_value=0)
    vg_count = rating_value_counts["Very good"]
    g_count = rating_value_counts["Good"]
    nn_count = rating_value_counts["Neither good nor poor"]
    p_count = rating_value_counts["Poor"]
    vp_count = rating_value_counts["Very poor"]
    dk_count = rating_value_counts["Don't know"]
    total_rated_count = vg_count + g_count + nn_count + p_count + vp_count + dk_count
    recomended = safe_percentage(vg_count + g_count, total_rated_count)
    not_recomended = safe_percentage(p_count + vp_count, total_rated_count)

    pcn_rating_value_counts = (
        pcn_df["rating"].value_counts().reindex(categories, fill_value=0)
    )
    pcn_vg_count = pcn_rating_value_counts["Very good"]
    pcn_g_count = pcn_rating_value_counts["Good"]
    pcn_nn_count = pcn_rating_value_counts["Neither good nor poor"]
    pcn_p_count = pcn_rating_value_counts["Poor"]
    pcn_vp_count = pcn_rating_value_counts["Very poor"]
    pcn_dk_count = pcn_rating_value_counts["Don't know"]
    pcn_total_rated_count = (
        pcn_vg_count
        + pcn_g_count
        + pcn_nn_count
        + pcn_p_count
        + pcn_vp_count
        + pcn_dk_count
    )
    pcn_recomended = safe_percentage(pcn_vg_count + pcn_g_count, pcn_total_rated_count)
    pcn_not_recomended = safe_percentage(pcn_p_count + pcn_vp_count, pcn_total_rated_count)

    recommendation_plot(
        recomended,
        not_recomended,
        pcn_recomended,
        pcn_not_recomended,
        "reports/recommendation.png",
    )
    text_list = col_to_list(df, "free_text")
    all_feedback = ""
    for index, text in enumerate(text_list):
        all_feedback = all_feedback + f"{index} - {text} "

    text_list2 = col_to_list(df, "do_better")
    all_improvement = ""
    for index, text in enumerate(text_list2):
        all_improvement = all_improvement + f"{index} - {text} "

    feedback_summary = (
        ask_groq(
            f"Summarize this GP Surgery feedback, identifying positive and negative trends: {all_feedback}, your output should be plain text only, don't use markdown in your output."
        )
        .replace("*", "")
        .replace("#", "")
        if all_feedback.strip()
        else "No free-text feedback was available for this reporting period."
    )
    improvement_summary = (
        ask_groq(
            f"Summarize this GP Surgery improvement suggestions, identifying trends: {all_improvement}, your output should be plain text only, don't use markdown in your output."
        )
        .replace("*", "")
        .replace("#", "")
        if all_improvement.strip()
        else "No improvement suggestions were available for this reporting period."
    )

    total_recommendation_count = total_rated_count
    response_period = f"{selected_month} {selected_year}"
    monthly_metrics = [
        {
            "label": "Total Responses",
            "value": str(total_feedback_count),
            "caption": "FFT responses this month",
            "fill": (239, 244, 248),
        },
        {
            "label": "Recommended",
            "value": f"{recomended}%",
            "caption": f"PCN avg {pcn_recomended}%",
            "fill": (235, 244, 233),
        },
        {
            "label": "Not Recommended",
            "value": f"{not_recomended}%",
            "caption": f"PCN avg {pcn_not_recomended}%",
            "fill": (252, 239, 232),
        },
    ]

    # Create the PDF
    pdf = AIReportPDF()
    pdf.report_title = f"{selected_surgery.replace('-', ' ')} - {response_period}"
    pdf.set_title(
        pdf_safe_text(
            f"AI MedReview: FFT Monthly Report - {selected_surgery} {selected_month} {selected_year}"
        )
    )
    pdf.set_author(pdf_safe_text("Jan du Plessis"))
    pdf.set_subject(pdf_safe_text("Monthly Medical Review Report"))
    pdf.set_keywords(pdf_safe_text("AIMedReview, Medical, Report, Monthly, FFT"))
    pdf.set_creator(pdf_safe_text("AI MedReview System"))

    pdf.add_cover_page(
        selected_surgery,
        selected_pcn,
        selected_month,
        selected_year,
        monthly_metrics,
    )

    pdf.add_page()
    pdf.add_section_heading(
        "1. Performance Overview",
        f"Summary of patient feedback volumes and recommendation performance for {response_period}.",
    )
    pdf.draw_metric_cards(monthly_metrics)
    pdf.draw_info_box(
        "At a glance",
        [
            f"The surgery received {total_feedback_count} FFT responses during {response_period}.",
            f"Recommended responses accounted for {recomended}% of submissions compared with a PCN average of {pcn_recomended}%.",
            f"Not recommended responses accounted for {not_recomended}% of submissions from {total_recommendation_count} rated responses.",
        ],
    )
    pdf.ln(5)
    pdf.add_chart(
        "Recommendation comparison",
        "reports/recommendation.png",
        subtitle="Surgery performance against the selected PCN average for the same reporting period.",
        image_height=62,
    )
    pdf.add_chart(
        "Rating distribution",
        plot_filename,
        subtitle="Distribution of FFT rating categories across all responses in the selected month.",
        image_height=66,
    )

    pdf.add_page()
    pdf.add_section_heading(
        "2. Response Pattern",
        "Daily submission trend to help spot surges, gaps, and review activity across the month.",
    )
    pdf.draw_info_box(
        "Interpretation",
        [
            "Use the time series below to identify higher-volume days and quieter periods.",
            "Sharp spikes can reflect service events, campaigns, or operational changes worth exploring alongside the comments.",
        ],
    )
    pdf.ln(5)
    pdf.add_chart(
        "Daily FFT responses",
        "reports/daily_count.png",
        subtitle="Review volume plotted by day across the selected reporting window.",
        image_height=90,
    )

    pdf.add_page()
    pdf.add_section_heading(
        "3. Feedback Themes",
        "Patient free-text comments with a companion AI summary to surface the main experience patterns.",
    )
    pdf.add_insight_box("Groq insight summary", feedback_summary)
    pdf.add_response_entries(
        f"Feedback responses ({len(text_list)})",
        [f"{index}: {text}" for index, text in enumerate(text_list)],
    )

    pdf.add_page()
    pdf.add_section_heading(
        "4. Improvement Suggestions",
        "Suggestions from patients on what could be improved, supported by an AI-generated theme summary.",
    )
    pdf.add_insight_box("Groq insight summary", improvement_summary)
    pdf.add_response_entries(
        f"Improvement suggestions ({len(text_list2)})",
        [f"{index}: {text}" for index, text in enumerate(text_list2)],
    )

    pdf.add_page()
    pdf.add_section_heading(
        "5. Language Snapshot",
        "Word clouds provide a quick visual summary of the language patients used in feedback and improvement comments.",
    )
    if display_wc1:
        pdf.add_chart(
            "Feedback free-text",
            "reports/wordcloud1.png",
            subtitle="Most frequently used terms in patient feedback comments.",
            image_height=82,
        )
    if display_wc2:
        pdf.add_chart(
            "Improvement suggestions",
            "reports/wordcloud2.png",
            subtitle="Most frequently used terms in patient suggestions for improvement.",
            image_height=82,
        )
    if not display_wc1 and not display_wc2:
        pdf.draw_info_box(
            "Word clouds unavailable",
            ["There was not enough free-text content in this reporting period to generate word clouds."],
        )

    # Output the PDF
    pdf.output("reports/report.pdf", "F")
