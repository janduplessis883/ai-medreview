import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
import seaborn as sns
import numpy as np

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


def simple_pdf(df, pcn_df, selected_month, selected_year, selected_surgery, selected_pcn, plot_column):
    # Generate the Seaborn count plot
    plot_filename = "reports/rating.png"
    generate_sns_countplot(df, plot_column, plot_filename)
    total_feedback_count = df.shape[0]
    rating_value_counts = df['rating'].value_counts()

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
    pdf.add_page()

    # Header "AI MedReview" with Arial in bold
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(39, 69, 98)
    pdf.cell(0, 10, "AI MedReview", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 10)
    info_string2 = f"{selected_pcn.replace('-', ' ')}"
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, info_string2, 0, 1)

    # Additional info in Arial, not bold
    pdf.set_font("Arial", "B", 16)
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

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, f"The total feedback received during {selected_month} {selected_year} was {total_feedback_count}.", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 14)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 10, f"Recommended - {recomended}%  (PCN Average - {pcn_recomended}%)", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 14)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, f"Not Recommended - {not_recomended}%  (PCN Average - {pcn_not_recomended}%)", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.image("reports/recommendation.png", x=10, y=85, w=180)  # Adjust x, y, w as necessary
    pdf.image(plot_filename, x=10, y=160, w=180)  # Adjust x, y, w as necessary
    pdf.image('images/nhs_scoring.png', x=10, y=235, w=180)


    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(197, 58, 50)
    pdf.cell(0, 10, "SECTION 2: Response Rate", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(197, 58, 50)
    pdf.cell(0, 10, "SECTION 3: Emontion Detection & Sentiment Analysis", 0, 1)  # '0' for cell width, '1' for the new line

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, f"Emotion Detection identify and interpret human emotions from text. By analyzing the language used in patient feedback.", 0, 1)

    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(197, 58, 50)
    pdf.cell(0, 10, "SECTION 4: Topic Analysis", 0, 1)  # '0' for cell width, '1' for the new line

    # Header "AI MedReview" with Arial in bold
    pdf.set_font("Arial", "", 6)
    pdf.set_text_color(131, 131, 130)
    pdf.cell(0, 10, "AI MedReview: FFT Analysis - GitHub: janduplessis883 jan.duplessis@nhs.net", 0, 1)  # '0' for cell width, '1' for the new line
    # Output the PDF
    pdf.output("reports/report.pdf", "F")
