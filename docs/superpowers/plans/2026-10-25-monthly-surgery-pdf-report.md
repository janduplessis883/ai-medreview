# Monthly Surgery PDF Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit tab that generates and downloads a deterministic, user-friendly monthly PDF containing reviews from exactly one surgery.

**Architecture:** A Streamlit-independent report module filters and aggregates normalized review data, renders a self-contained Jinja2 HTML template, and converts it to PDF bytes with WeasyPrint. The existing Streamlit entry point owns only selectors, preview metrics, session state, generation, and download.

**Tech Stack:** Python, pandas, Jinja2, HTML/CSS paged media, WeasyPrint, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-10-25-monthly-surgery-pdf-report-design.md`

## Global Constraints

- Use the existing FFT mapping: Very good = 5, Good = 4, Neither good nor poor = 3, Poor = 2, Very poor = 1; do not add zero stars.
- A report contains exactly one surgery and one calendar month, sourced from normalized `df_all`, never the sidebar-filtered `df`.
- Commentary is deterministic and based only on classified counts and themes; no LLM calls, webhooks, or network resources.
- Include all reviews newest first and escape all review text before it enters HTML.
- Return PDF bytes in memory for download; do not write shared report files.
- Keep `ai_medreview/reports.py` unchanged.

## Review Focus

- A changed selector must never leave an old report downloadable under the new surgery/month selection; Task 3 tests state invalidation through a pure selection-key helper and verifies the UI wiring.
- A month containing reviews from several surgeries must include only exact matches for the chosen surgery; Task 1 tests a deliberately mixed fixture.
- Missing, NaN, or malformed optional classifications must render as neutral values without crashing; Task 1 tests this fallback behavior.
- Patient text containing HTML or very long unbroken content must be escaped and wrap without corrupting the PDF; Task 2 tests escaping and generates a long-content PDF.
- Months with no matching valid reviews must not produce a misleading empty report; Task 1 tests rejection and Task 3 disables generation.

---

### Task 1: Report data model and deterministic analysis

**Files:**
- Create: `ai_medreview/monthly_report.py`
- Create: `tests/test_monthly_report.py`

**Interfaces:**
- Consumes: normalized `pandas.DataFrame` using the columns named in the spec, a surgery string, and a month accepted by `pandas.Period`.
- Produces: `filter_report_reviews(df: pd.DataFrame, surgery: str, month: str | pd.Period) -> pd.DataFrame`, `build_report_context(reviews: pd.DataFrame, surgery: str, month: str | pd.Period, generated_on: date | None = None) -> dict[str, object]`, `safe_report_filename(surgery: str, month: str | pd.Period) -> str`, and `report_selection_key(surgery: str, month: str | pd.Period) -> tuple[str, str]`.

- [ ] **Step 1: Write failing tests for exact surgery/month scoping**

Add `test_filter_report_reviews_isolates_exact_surgery_and_month`, asserting a mixed-surgery, mixed-month fixture returns only the chosen surgery/month in newest-first order, and `test_filter_report_reviews_rejects_empty_scope`, asserting `ValueError` for no valid matches.

- [ ] **Step 2: Run the scoping tests and verify they fail**

Run: `rtk pytest tests/test_monthly_report.py -k filter_report_reviews -v`
Expected: FAIL because `ai_medreview.monthly_report` does not exist.

- [ ] **Step 3: Implement `filter_report_reviews` and `report_selection_key`**

Normalize the requested month with `pd.Period(..., freq="M")`, coerce `time`, filter exact surgery equality plus calendar period, remove invalid dates, validate one surgery/period, sort newest first, and return a copy.

- [ ] **Step 4: Run the scoping tests and verify they pass**

Run: `rtk pytest tests/test_monthly_report.py -k filter_report_reviews -v`
Expected: PASS.

- [ ] **Step 5: Write failing aggregation and narrative tests**

Add tests asserting all five ordered rating entries exist with explicit zero counts, counts and percentages are correct, score 2/3 totals are emphasized for actionability and urgency, dignity/inclusion and leaving-risk flags are counted, primary topics are frequency ordered, neutral defaults handle NaN values, and generated prose contains fixed evidence statements rather than causal claims.

- [ ] **Step 6: Implement `build_report_context`**

Return a template-ready dictionary containing report metadata, five rating summaries, classification distributions and deterministic observations, topic summaries, and review entries. Map score labels to `None`, `Low`, `Moderate`, and `High` display language while retaining numeric scores.

- [ ] **Step 7: Implement and test `safe_report_filename`**

Return `<lowercase-ascii-surgery-slug>-reviews-YYYY-MM.pdf`; collapse non-alphanumeric runs to hyphens and provide `surgery` when the display name has no ASCII alphanumerics.

- [ ] **Step 8: Run Task 1 tests**

Run: `rtk pytest tests/test_monthly_report.py -k 'filter or context or filename or selection' -v`
Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
rtk git add ai_medreview/monthly_report.py tests/test_monthly_report.py
rtk git commit -m "feat: add monthly review report analysis"
```

### Task 2: HTML template and PDF renderer

**Files:**
- Modify: `ai_medreview/monthly_report.py`
- Create: `ai_medreview/templates/monthly_report.html`
- Modify: `tests/test_monthly_report.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: Task 1 `build_report_context` output.
- Produces: `render_report_html(context: dict[str, object]) -> str` and `generate_monthly_report_pdf(reviews: pd.DataFrame, surgery: str, month: str | pd.Period, generated_on: date | None = None) -> bytes`.

- [ ] **Step 1: Write failing HTML rendering tests**

Assert the rendered HTML contains the surgery, month, all five FFT labels, deterministic classification headings, every review in newest-first order, and escaped output for review text containing `<script>` and `&`. Assert the template contains no `http://` or `https://` resources.

- [ ] **Step 2: Run HTML tests and verify they fail**

Run: `rtk pytest tests/test_monthly_report.py -k html -v`
Expected: FAIL because the renderer and template do not exist.

- [ ] **Step 3: Create the Jinja2 template and implement `render_report_html`**

Use a package-relative template loader with autoescaping. Add A4 `@page` rules, running footer/page numbers, self-contained system fonts and CSS, rating bars, classification cards, topic rows, and break-safe review cards with overflow wrapping.

- [ ] **Step 4: Run HTML tests and verify they pass**

Run: `rtk pytest tests/test_monthly_report.py -k html -v`
Expected: PASS.

- [ ] **Step 5: Add Jinja2 and WeasyPrint to `requirements.txt`, then install project requirements**

Run: `rtk pip install -r requirements.txt`
Expected: installation succeeds and `python -c 'import weasyprint'` exits 0.

- [ ] **Step 6: Mark the first PDF artifact authoring operation**

Run exactly once immediately before the first PDF generation command:
`rtk proxy node container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format pdf`
Expected: command succeeds.

- [ ] **Step 7: Write the failing PDF smoke and long-content tests**

Assert `generate_monthly_report_pdf` returns bytes beginning `%PDF`, exceeds 5 KB, contains no other surgery's unique review text after extraction, and succeeds with a 2,000-character unbroken review.

- [ ] **Step 8: Implement `generate_monthly_report_pdf`**

Compose Task 1 scoping/context functions with `render_report_html`, then call `weasyprint.HTML(string=html, base_url=<template directory>).write_pdf()` and return bytes.

- [ ] **Step 9: Run Task 2 tests**

Run: `rtk pytest tests/test_monthly_report.py -v`
Expected: PASS.

- [ ] **Step 10: Commit Task 2**

```bash
rtk git add ai_medreview/monthly_report.py ai_medreview/templates/monthly_report.html tests/test_monthly_report.py requirements.txt
rtk git commit -m "feat: render monthly review reports as PDF"
```

### Task 3: Streamlit report tab and end-to-end verification

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_monthly_report.py`
- Create: `output/pdf/` representative report only for visual verification; do not commit it.

**Interfaces:**
- Consumes: Task 1 `filter_report_reviews`, `build_report_context`, `safe_report_filename`, `report_selection_key`; Task 2 `generate_monthly_report_pdf`.
- Produces: a sixth `Monthly report` tab with independent surgery/month controls, preview, explicit generation action, selection-bound session state, and PDF download.

- [ ] **Step 1: Write a failing source-level integration test**

Assert `streamlit_app.py` imports the report interfaces, defines the `Monthly report` tab, creates surgery/month widgets from `df_all`, guards generation with a button, binds cached PDF bytes to `report_selection_key`, and renders `st.download_button` with MIME type `application/pdf`.

- [ ] **Step 2: Run the integration test and verify it fails**

Run: `rtk pytest tests/test_monthly_report.py -k streamlit -v`
Expected: FAIL because the tab is absent.

- [ ] **Step 3: Add the sixth tab and report UI to `streamlit_app.py`**

Populate surgery choices from all normalized reviews and available month choices from the selected surgery. Show preview metrics before generation. Disable generation for an empty scope. Invalidate session PDF state whenever `report_selection_key` differs, generate inside a status/spinner boundary, surface conversion errors locally, and offer download only when stored bytes match the active key.

- [ ] **Step 4: Run the integration and full test suites**

Run: `rtk pytest tests/test_monthly_report.py -v`
Expected: PASS.

Run: `rtk pytest tests/ -v`
Expected: PASS.

- [ ] **Step 5: Perform static verification**

Run: `rtk python -m compileall streamlit_app.py ai_medreview/monthly_report.py`
Expected: both files compile without errors.

- [ ] **Step 6: Generate a representative PDF artifact**

Use a surgery/month from `data_v4.csv`, save the single verification artifact under `output/pdf/monthly-surgery-report-sample.pdf`, and assert extracted text includes that surgery and month but excludes a known review unique to another surgery.

- [ ] **Step 7: Render and visually inspect every PDF page**

Run `rtk proxy pdftoppm -png output/pdf/monthly-surgery-report-sample.pdf tmp/pdfs/monthly-surgery-report`; inspect all resulting PNGs for clipping, overlap, page breaks, long-text wrapping, and consistent headers/footers. Fix and repeat until no defects remain.

- [ ] **Step 8: Check whether Streamlit is already running**

Run: `rtk proxy sh -c "lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep -i python | grep ':85' || true"`
Expected: report any existing Streamlit process; do not start a new server without user approval.

- [ ] **Step 9: Commit Task 3**

```bash
rtk git add streamlit_app.py tests/test_monthly_report.py
rtk git commit -m "feat: add monthly report download tab"
```

- [ ] **Step 10: Run final verification and inspect the diff**

Run: `rtk pytest tests/ -v`
Expected: PASS.

Run: `rtk git diff --check`
Expected: no whitespace errors.

Run: `rtk git status --short`
Expected: only the uncommitted verification PDF/PNGs remain, or a clean tree after those temporary artifacts are removed.
