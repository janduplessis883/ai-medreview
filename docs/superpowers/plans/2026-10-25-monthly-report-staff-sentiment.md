# Monthly report staff and sentiment page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a polished additional PDF page showing surgery/month-scoped staff-group coverage and overall sentiment.

**Architecture:** Extend `build_report_context` with deterministic staff and sentiment summaries, then render those summaries in a new page in the existing Jinja/WeasyPrint template. Keep all computation in the report module and all presentation in the template.

**Tech Stack:** Python, pandas, Jinja2, WeasyPrint, pytest, pypdf.

**Spec:** `docs/superpowers/specs/2026-10-25-monthly-report-staff-sentiment-design.md`

## Global Constraints

- Use only reviews for the selected surgery and calendar month.
- Sentiment categories are `Positive`, `Neutral or Mixed`, and `Negative`.
- Average sentiment strength uses valid integer scores 0–4 and is rounded to one decimal.
- Staff groups sort by descending review count, then alphabetically for ties.
- Blank staff groups display as `No specific staff mentioned`.
- Add no new Python or system dependencies.

## Review Focus

- Missing sentiment/staff columns: context construction must return safe empty summaries instead of failing — covered by a neutral-default context test in Task 1.
- Invalid sentiment labels: counts must not silently enter a known category — covered by a classified-count test in Task 1.
- Non-integer or out-of-range strength values: average must ignore them — covered by the strength test in Task 1.
- Staff-count ties and blank values: deterministic alphabetical order and fallback label — covered by the staff ranking test in Task 1.
- Empty classified values: the new page must show a readable not-classified state — covered by the template test in Task 2.

### Task 1: Add deterministic staff and sentiment context

**Files:**
- Modify: `ai_medreview/monthly_report.py`
- Test: `tests/test_monthly_report.py`

**Interfaces:**
- Consumes: the existing surgery/month-scoped `scoped` DataFrame in `build_report_context`.
- Produces: `context["sentiment"]` with `distribution`, `dominant_label`, `dominant_count`, `classified_count`, `average_strength`; `context["staff_groups"]` with ranked group dictionaries containing `label`, `count`, `percentage`, and sentiment counts.

- [ ] **Step 1: Write failing tests**

Add tests asserting known sentiment counts and percentages, dominant sentiment tie-breaking, average strength ignoring invalid values, staff ranking by count then label, and neutral defaults when fields are absent.

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `python -m pytest tests/test_monthly_report.py -k 'sentiment or staff' -v`

Expected: FAIL because the new context keys do not exist.

- [ ] **Step 3: Implement context aggregation**

Add small private helpers for sentiment normalization/counts and staff-label normalization. Build the summaries from `scoped` only, preserving existing report behavior and avoiding changes to the review-list ordering.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `python -m pytest tests/test_monthly_report.py -k 'sentiment or staff' -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_medreview/monthly_report.py tests/test_monthly_report.py
git commit -m "feat: add monthly report staff sentiment context"
```

### Task 2: Render the additional report page

**Files:**
- Modify: `ai_medreview/templates/monthly_report.html`
- Test: `tests/test_monthly_report.py`

**Interfaces:**
- Consumes: `sentiment` and `staff_groups` from Task 1.
- Produces: a page headed `People and sentiment` with a dominant-sentiment callout, three sentiment cards, average-strength treatment, and ranked staff cards.

- [ ] **Step 1: Write failing template/PDF tests**

Assert the HTML contains the new page, all three sentiment labels, dominant summary, staff labels/counts, and a not-classified fallback. Assert the generated PDF contains the new heading and has an additional page.

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `python -m pytest tests/test_monthly_report.py -k 'people or sentiment or staff' -v`

Expected: FAIL because the template has no new page.

- [ ] **Step 3: Implement the page**

Add a page break after score distributions. Use compact cards and a ranked staff list with sentiment mix text; use CSS colors consistent with the existing positive/neutral/negative palette and preserve print-safe wrapping.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `python -m pytest tests/test_monthly_report.py -k 'people or sentiment or staff' -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_medreview/templates/monthly_report.html tests/test_monthly_report.py
git commit -m "feat: add staff and sentiment report page"
```

### Task 3: Full verification and visual PDF review

**Files:**
- Modify: none
- Test: `tests/test_monthly_report.py`

**Interfaces:**
- Consumes: the completed report context and template from Tasks 1–2.
- Produces: verified PDF output with no clipping, overlap, or missing page content.

- [ ] **Step 1: Run the complete suite**

Run: `python -m pytest tests/`

Expected: all tests pass.

- [ ] **Step 2: Generate and inspect a representative PDF**

Render a report containing multiple staff groups, all sentiment categories, ties, and missing values. Inspect every page as a PNG and confirm the new page is readable and the existing pages are unchanged.

- [ ] **Step 3: Check the working tree and commit verification changes if any**

Run: `git status --short --branch`

Expected: only intentional implementation commits remain.
