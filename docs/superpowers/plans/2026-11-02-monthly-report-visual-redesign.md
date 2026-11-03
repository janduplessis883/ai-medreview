# Monthly Report Visual Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the newer HTML/WeasyPrint monthly surgery report so its rendered PDF closely matches the supplied editorial reference style while preserving existing report content and behavior.

**Architecture:** Keep `generate_monthly_report_pdf()` and the report context unchanged. Rebuild the Jinja template's CSS and only make small markup additions needed for branding and the new editorial layout. Validate with existing PDF text tests plus a rendered visual sample.

**Tech Stack:** Jinja2, HTML/CSS, WeasyPrint, pytest, pypdf, Poppler rendering.

**Spec:** `docs/superpowers/specs/2026-11-02-monthly-report-visual-redesign-design.md`

## Global Constraints

- Update only the newer HTML/WeasyPrint monthly surgery report.
- Preserve report data, AI summary content, names opt-in behavior, filenames, and privacy behavior.
- Use ASCII hyphens in generated PDF text where new copy is introduced.
- Keep the final PDF readable, valid, and safe for long clinical text.

## Review Focus

- Long review text and AI summaries must wrap without clipping or overlap; cover with the existing long-review PDF test and rendered inspection.
- Reports with no classified sentiment/staff/topics must retain their fallback text; cover with existing fallback tests.
- The names opt-in switch must remain unchanged; cover with existing names tests.
- The logo/font assets must degrade to safe fallbacks when unavailable; cover with a template-render assertion and PDF generation smoke test.
- Page breaks and footer numbering must remain coherent across multi-page reports; cover with rendered sample inspection and page-count assertions.

### Task 1: Add brand assets and editorial design tokens

**Files:**
- Create: `ai_medreview/templates/assets/ai-medreview-logo.png`
- Modify: `ai_medreview/templates/monthly_report.html:6-308`
- Test: `tests/test_monthly_report.py`

**Interfaces:**
- Consumes: existing `render_report_html(context)` and WeasyPrint base URL behavior.
- Produces: self-contained template styling with warm background, orange/black palette, safe font fallbacks, logo reference, and footer system.

- [ ] **Step 1: Add a focused template test** asserting the rendered HTML includes the AI-MedReview logo reference and the new design-token values.
- [ ] **Step 2: Run the focused test and confirm it fails** because the logo and tokens do not yet exist.
- [ ] **Step 3: Add the supplied logo asset** under the template directory and replace the current navy/teal CSS tokens with the reference-inspired palette, typography, page margins, footer, and base spacing.
- [ ] **Step 4: Run the focused template test** and confirm it passes.

### Task 2: Recompose report sections into the reference visual language

**Files:**
- Modify: `ai_medreview/templates/monthly_report.html`
- Test: `tests/test_monthly_report.py`

**Interfaces:**
- Consumes: unchanged context keys: `surgery`, `month_label`, `ratings`, `topics`, `sentiment`, `staff_groups`, `ai_summary`, and `reviews`.
- Produces: cover/hero, metrics, summaries, AI insights, and review-list markup using editorial headings, orange outlined callouts with offset shadows, ruled lists, chips, and restrained status colours.

- [ ] **Step 1: Add/adjust template assertions** for the cover logo, orange prompt-style callouts, AI summary sections, review references, and footer label while keeping existing report text assertions intact.
- [ ] **Step 2: Run the focused PDF tests and confirm the new structural assertions fail.**
- [ ] **Step 3: Update the template markup and CSS** to recompose each report section without changing any Jinja data logic or conditional behavior.
- [ ] **Step 4: Run the full monthly-report test module** and confirm all tests pass.

### Task 3: Render and visually verify a representative PDF

**Files:**
- Create/replace: `output/pdf/monthly-surgery-report-sample.pdf`
- Temporary: `tmp/pdfs/monthly-surgery-report-sample-page-*.png`

**Interfaces:**
- Consumes: the updated `generate_monthly_report_pdf()` output and the existing sample-report fixture path.
- Produces: a verified sample PDF with no clipping, overlap, broken glyphs, or unreadable contrast.

- [ ] **Step 1: Run the artifact-operation marker** for one PDF edit output.
- [ ] **Step 2: Generate the sample monthly report PDF** using the repository's existing sample/report test data path.
- [ ] **Step 3: Render the PDF pages to PNG** with Poppler and inspect the cover, summary, AI summary, and review-list pages.
- [ ] **Step 4: Fix any visual defects** in the template and rerun the relevant tests and rendering checks.
- [ ] **Step 5: Confirm the final PDF opens with pypdf**, contains expected report text, and has the expected multi-page structure.

### Task 4: Final verification

**Files:**
- Modify: none unless verification exposes a defect.

- [ ] **Step 1: Run the full monthly-report test suite.**
- [ ] **Step 2: Run `git diff --check` and inspect the final diff** to confirm only the new monthly-report presentation and approved design note/plan changed.
- [ ] **Step 3: Report the updated template, verification commands, and final PDF artifact.**
