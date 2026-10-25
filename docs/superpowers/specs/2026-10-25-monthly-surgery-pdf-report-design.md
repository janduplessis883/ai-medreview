# Monthly surgery PDF report design

## Intent

Add a report area to the existing Streamlit dashboard so a user can select one surgery and one calendar month, generate a polished PDF containing only that surgery's reviews for that month, and download it immediately. The report is intended to make classified patient feedback easy for surgery teams to understand and act on without requiring an AI service at report-generation time.

## Goals

- Scope every report to exactly one surgery and one calendar month.
- Show the full FFT rating distribution using the existing five-point mapping: Very good = 5, Good = 4, Neither good nor poor = 3, Poor = 2, Very poor = 1.
- Summarize actionability, urgency, dignity/inclusion concerns, leaving risk, and primary-topic categorization using deterministic counts, percentages, and plain-language observations.
- Present every review in a readable, user-friendly list.
- Produce a modern, restrained HTML/CSS design converted to PDF and downloadable from Streamlit.
- Keep report generation local and deterministic: no LLM calls, webhooks, or other network requests.

## Non-goals

- Comparing a surgery with another surgery, PCN, or benchmark.
- Editing, annotating, or emailing reports from the app.
- Reclassifying reviews or changing the Jev pipeline.
- Replacing the existing dashboard filters or existing `ai_medreview/reports.py` workflow.
- Adding a zero-star category.

## Anti-goals

- A report that can silently inherit multi-surgery dashboard filters.
- Dense raw-data tables, raw JSON, confidence scores, or internal identifiers in the PDF.
- AI-written conclusions that may vary between runs.
- A PDF that depends on remote fonts, images, scripts, or browser automation.

## Chosen approach and alternatives

The chosen approach is an HTML/CSS template rendered to PDF with WeasyPrint. It gives the report print-specific layout, strong typography and pagination, while returning PDF bytes directly to Streamlit without a browser process.

Alternatives considered:

1. A dedicated ReportLab document generator would have fewer rendering dependencies and precise drawing control, but the user preferred the greater styling flexibility of HTML/CSS.
2. Retrofitting `ai_medreview/reports.py` would reuse some FPDF layout code, but that module contains older data assumptions, filesystem output, external webhooks, and AI summaries that conflict with the requested deterministic, surgery-specific flow.
3. Browser-based HTML printing would closely match browser CSS, but Chromium/Playwright is not installed and would add a large runtime dependency. WeasyPrint is the narrower deployment choice.

## User experience

Add a sixth top-level tab named `Monthly report` with a description explaining that it creates a surgery-specific monthly PDF. This area has its own controls and does not use the sidebar's surgery, date, sentiment, topic, staff, or alert filters.

The controls are:

- Surgery: a single-select control populated from the complete deduplicated dataset.
- Reporting month: a single-select control populated only with months that have reviews for the selected surgery, newest first.
- Generate report: an explicit primary action so PDF conversion does not run on every widget rerun.

Before generation, the tab shows a compact preview with the surgery, month, number of reviews, and rating counts. If no valid data exists, it explains why and disables generation. After successful generation, it shows a download button whose filename contains a filesystem-safe surgery slug and `YYYY-MM`. Generation errors stay in the report area and do not break the rest of the dashboard.

## PDF structure

The document uses A4 pages, print-safe margins, a system sans-serif font stack, a restrained navy/teal/slate palette, subtle borders, and generous whitespace. It contains no remote assets.

1. **Report header** - AI MedReview label, surgery display name, reporting month, generated date, and total review count.
2. **Rating overview** - five clearly labelled cards or horizontal bars for Very good (5 stars), Good (4), Neither good nor poor (3), Poor (2), and Very poor (1), including counts and percentages. Missing ratings appear as zero.
3. **Deterministic classification summary** - concise prose and small distributions for:
   - actionability scores 0-3, emphasizing the count and percentage scored 2 or 3;
   - urgency scores 0-3, emphasizing the count and percentage scored 2 or 3;
   - dignity/inclusion concern count and percentage;
   - leaving-risk count and percentage;
   - primary-topic counts and percentages, ordered by frequency, with a plain-language statement naming the leading themes.
4. **Review list** - every review, ordered newest first. Each entry shows date, FFT label plus numeric stars, the review text, primary topic, actionability level, and urgency level. Dignity/inclusion and leaving-risk badges appear only when flagged. Clinical-safety data is not added as a headline section because it was not requested, though the existing app continues to show it elsewhere.
5. **Footer** - surgery name, reporting month, and page number.

Plain-language statements are built from fixed templates and computed values. They state what the classifications show without implying causation or clinical judgement. When there are no reviews in a subgroup, the report says so directly. Percentages use the report's total valid review count as denominator and show both count and percentage to avoid overstating small samples.

## Data and scoping

Report data is derived from `df_all`, after the app's existing deduplication and type normalization. A dedicated function filters by exact surgery equality and a calendar-month period. It validates that the result contains one surgery and one month before rendering. The sidebar-filtered `df` is never passed to report generation.

The generator uses these existing columns: `time`, `surgery`, `review`, `rating`, `rating_score`, `jev_primary_topic`, `jev_actionability`, `jev_urgency`, `jev_dignity_or_inclusion_concern`, and `jev_leaving_risk`. Missing optional classifications receive neutral display defaults such as `Not classified`; malformed dates and rows outside the chosen scope are excluded. Review text is HTML-escaped by template autoescaping.

## Code organization

- `ai_medreview/monthly_report.py` will contain scope validation, aggregate calculations, deterministic narrative construction, HTML rendering, PDF conversion, and safe filename creation. It will have no Streamlit dependency.
- `ai_medreview/templates/monthly_report.html` will contain semantic HTML and self-contained print CSS.
- `streamlit_app.py` will contain only the new tab controls, preview, generation action, session-state storage for generated bytes, and download button.
- `requirements.txt` will add pinned-compatible direct dependencies for `Jinja2` and `WeasyPrint`.
- `tests/test_monthly_report.py` will test surgery/month isolation, FFT mapping and zero counts, narrative thresholds, HTML escaping, stable review ordering, safe filenames, and PDF byte generation.

The legacy `ai_medreview/reports.py` remains untouched.

## State and reruns

Generated PDF bytes and their identifying selection `(surgery, month)` are stored in session state. Changing either selector invalidates the prior download so a user cannot download a stale report under a new selection. PDF work runs only after the Generate report button is pressed.

## Verification

- Unit tests cover aggregation, filtering, prose, escaping, ordering, and filename behavior.
- A PDF smoke test verifies a non-empty document with a PDF header.
- A representative report is generated from repository data, rendered to PNG, and visually inspected for clipping, overlaps, page breaks, card consistency, footer placement, and long-review behavior.
- Extracted PDF text is checked for the selected surgery and month and checked to ensure a different surgery's review text is absent.
- The Streamlit app is syntax-checked and started locally if the user approves running it.

## Builder decisions

The builder may choose exact spacing, font sizes, colors within the restrained visual direction, CSS class names, helper names, and wording of deterministic sentences so long as the calculations and meaning above remain unchanged. The builder may also choose compatible minimum-version constraints after checking the local Python version and deployment requirements.
