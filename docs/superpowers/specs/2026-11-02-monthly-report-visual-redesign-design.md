# Monthly Report Visual Redesign

## Intent

Update only the newer HTML/WeasyPrint monthly surgery report so it closely matches the supplied editorial PDF references while retaining AI-MedReview's existing clinical content and data behavior.

## Goals

- Use a warm off-white page background, near-black ink, vivid orange accents, muted olive-grey secondary text, and restrained green status accents.
- Introduce editorial page hierarchy: oversized headings, generous spacing, small uppercase labels, compact metadata, and a consistent footer/page-number treatment.
- Rework summaries, AI insights, and review entries into orange outlined callouts, offset-shadow panels, metric cards, and clean ruled lists inspired by the references.
- Add the supplied AI-MedReview logo as a monochrome brand mark where it remains legible and unobtrusive.
- Keep all existing report context, review text, AI summary content, names opt-in behavior, and pagination-safe long-text handling intact.

## Non-goals

- Do not change the legacy FPDF report.
- Do not change the report data model, AI prompts, Streamlit controls, filenames, or privacy behavior.
- Do not copy the reference document's content or branding; only its visual language is being adapted.

## Approach

Keep WeasyPrint and the Jinja template. Replace the current navy/teal dashboard styling with a reusable editorial CSS system and make small structural template changes only where needed for the cover, callouts, metrics, AI summary, and review list. Use an available open sans-serif with an Avenir-like feel and a monospace face for evidence/code-style blocks, with safe fallbacks for PDF generation.

## Verification

Render a representative monthly report, inspect the resulting PNG pages for clipping, overlap, contrast, and page-break defects, and run the existing monthly-report test suite. Confirm the output remains a readable, valid PDF with the expected report text and page structure.
