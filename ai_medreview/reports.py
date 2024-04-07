# reports.py
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch  # Import Patch here

def generate_cqrs_report(data, month, year, surgery, pcn, filename="cqrs_report.pdf"):
    # Create a seaborn plot
    order = ["Extremely likely", "Likely", "Neither likely nor unlikely", 
             "Unlikely", "Extremely unlikely", "Don't know"]
    palette = {"Extremely likely": "#112f45", "Likely": "#4d9cb9", 
               "Neither likely nor unlikely": "#9bc8e3", "Unlikely": "#f4ba41", 
               "Extremely unlikely": "#ec8b33", "Don't know": "#ae4f4d"}

    plt.figure(figsize=(12, 5))
    sns.countplot(data=data, y="rating", order=order, palette=palette)
    ax = plt.gca()
    ax.set_yticklabels([])
    legend_patches = [Patch(color=color, label=label) for label, color in palette.items()]
    plt.legend(handles=legend_patches, title="Rating Categories", bbox_to_anchor=(1.05, 1), loc="upper left")

    for p in ax.patches:
        width = p.get_width()
        offset = width * 0.02
        y = p.get_y() + p.get_height() / 2
        ax.text(width + offset, y, f"{int(width)} / {round((int(width)/data.shape[0])*100, 1)}%", va="center", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    plt.xlabel("Count")
    plt.ylabel("")

    # Save the plot to a PDF file using ReportLab
    with PdfPages(filename) as pdf:
        pdf.savefig(plt.gcf())
        plt.close()