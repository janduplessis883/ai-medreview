import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def generate_cqrs_report(data, month, year, surgery, pcn, filename="cqrs_report.pdf"):
    # Define A4 dimensions and margins
    a4_width_inch = 8.27  # A4 width
    margin_inch = 2  # Margin width
    plot_width = a4_width_inch - 2 * margin_inch  # Width of the plot
    plot_height = 5  # Explicitly setting plot height to 6 inches

    # Initialize the figure with the specified width and height
    plt.figure(figsize=(plot_width, plot_height))

    # Adding title and total responses text at the top, left-aligned
    title_text = f"AI MedReview CQRS Report {month} {year}"
    response_text = f"Total Responses: {data.shape[0]}"

    plt.figtext(0.01, 0.99, title_text, fontsize=14, fontweight='bold', ha='left', va='top')
    plt.figtext(0.01, 0.95, response_text, fontsize=11, ha='left', va='top')

    # Adjust the plot position
    plt.subplots_adjust(top=0.85, bottom=0.2, left=0.1, right=0.95)

    # Create the plot
    order = ["Extremely likely", "Likely", "Neither likely nor unlikely", "Unlikely", "Extremely unlikely", "Don't know"]
    palette = {"Extremely likely": "#112f45", "Likely": "#4d9cb9", "Neither likely nor unlikely": "#9bc8e3", "Unlikely": "#f4ba41", "Extremely unlikely": "#ec8b33", "Don't know": "#ae4f4d"}
    sns.countplot(data=data, y="rating", order=order, palette=palette)
    ax = plt.gca()

    # Adjusting labels, legends, and annotations
    ax.set_yticklabels([])
    legend_patches = [Patch(color=color, label=label) for label, color in palette.items()]
    
    # Position the legend at the bottom right corner
    plt.legend(handles=legend_patches, title="Rating Categories", loc="lower right", fontsize=9)

    for p in ax.patches:
        width = p.get_width()
        offset = width * 0.02
        y = p.get_y() + p.get_height() / 2
        ax.text(width + offset, y, f"{int(width)} / {round((int(width)/data.shape[0])*100, 1)}%", va="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)

    plt.xlabel("Count", fontsize=10)
    plt.ylabel("")

    # Saving the figure into a PDF file
    with PdfPages(filename) as pdf:
        pdf.savefig(plt.gcf(), bbox_inches='tight')
        plt.close()