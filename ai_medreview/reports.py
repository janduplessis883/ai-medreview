from fpdf import FPDF


def simple_pdf(df, selected_month, selected_year, selected_surgery, selected_pcn):
    pdf = FPDF()
    pdf.add_page()

    count = df["rating"].value_counts()

    # Header "AI MedReview" with Inter in bold
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI MedReview", 0, 1)  # '0' for cell width, '1' for the new line

    # Additional info in Inter, not bold
    pdf.set_font("Arial", "", 16)
    info_string = f"{selected_pcn} {selected_surgery}: {selected_month} {selected_year}"
    pdf.cell(0, 10, info_string, 0, 1)

    # Output the PDF
    pdf.output("report.pdf", "F")
