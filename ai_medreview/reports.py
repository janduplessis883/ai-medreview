import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd

def generate_plot(df, filename="plot.png"):
    count = df["rating"].value_counts()
    count.plot(kind='bar')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(filename)
    plt.close()

def simple_pdf(df, selected_month, selected_year, selected_surgery, selected_pcn):
    # Generate the plot
    plot_filename = "plot.png"
    generate_plot(df, plot_filename)

    # Create the PDF
    pdf = FPDF()
    pdf.add_page()

    count = df["rating"].value_counts()

    # Header "AI MedReview" with Arial in bold
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(39, 69, 98)
    pdf.cell(0, 10, "AI MedReview", 0, 1)  # '0' for cell width, '1' for the new line

    # Additional info in Arial, not bold
    pdf.set_font("Arial", "", 18)
    info_string = f"{selected_pcn} {selected_surgery}"
    pdf.set_text_color(35, 37, 41)
    pdf.cell(0, 5, info_string, 0, 1)

    pdf.set_font("Arial", "B", 16)
    info_string = f"{selected_month} {selected_year}"
    pdf.cell(0, 10, info_string, 0, 1)

    # Insert a horizontal line after the header
    pdf.set_line_width(0.2)
    pdf.line(10, 35, 200, 35)  # (x1, y1, x2, y2)

    pdf.ln(7)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(39, 69, 98)
    pdf.cell(0, 10, "SECTION 1: Feedback Count", 0, 1)  # '0' for cell width, '1' for the new line

    # Insert the plot
    pdf.image(plot_filename, x=10, y=60, w=180)  # Adjust x, y, w as necessary

    # Output the PDF
    pdf.output("report.pdf", "F")

# Sample usage
data = {
    'rating': [1, 2, 3, 3, 2, 1, 4, 5, 5, 4, 3, 2, 1, 3, 4, 5]
}
df = pd.DataFrame(data)
simple_pdf(df, "July", 2024, "Sample Surgery", "Sample PCN")
