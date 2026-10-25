# Monthly Report Rating Cards and Pills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five PDF rating cards visually uniform with a wider middle card and replace each review's inline classifications with accessible, colour-coded pills.

**Architecture:** Keep Python report calculations and Streamlit integration unchanged. Update only the Jinja2 HTML/print CSS template and its real rendering tests, then verify a representative PDF visually.

**Tech Stack:** Jinja2, HTML/CSS paged media, WeasyPrint, pypdf, pytest

**Spec:** `docs/superpowers/specs/2026-10-25-monthly-report-rating-cards-and-pills-design.md`

## Global Constraints

- Preserve the existing five-to-one FFT mapping, counts, percentages, and progress bars.
- Keep `Neither good nor poor` on one line by widening its column; all rating cards remain equal height.
- Every review displays Category, Actionability, Urgency, Dignity/Inclusion, and Leaving Risk pills.
- Actionability/Urgency colours are 0 white-outline, 1 grey, 2 yellow, 3 red, and unclassified white with a dashed grey outline.
- Risk pills are white-outline when inactive and red when active.
- Labels must communicate state without relying on colour.
- Do not change report calculations, Streamlit controls, surgery/month scoping, or download behavior.

## Review Focus

- Long category labels must wrap inside a pill without widening or clipping the review card; the rendering test uses a long literal category.
- A review with unclassified scores must show `Not classified` pills rather than an incorrect severity colour; the rendering test covers missing scores.
- Risk pills must remain present when inactive so review cards have a consistent information set; the rendering test covers both states.
- The wider middle card must not push the five-card row beyond A4 printable width; visual verification checks the rendered page edges.
- Pill colours must remain readable in print and understandable without colour; tests assert text labels while visual verification checks contrast.

---

### Task 1: Rating geometry and review classification pills

**Files:**
- Modify: `ai_medreview/templates/monthly_report.html`
- Modify: `tests/test_monthly_report.py`

**Interfaces:**
- Consumes: the unchanged `build_report_context()` dictionary, including `ratings` and each review's `topic`, score dictionaries, and boolean flags.
- Produces: self-contained HTML/PDF presentation with fixed rating-column proportions and five classification pills per review.

- [ ] **Step 1: Write failing HTML rendering tests**

Add a test that renders reviews covering scores 0, 1, 2, 3, and missing; asserts explicit `<col>` classes for four equal side columns plus one wider middle column; asserts the middle rating label does not use wrapping markup; and asserts every review contains Category, Actionability, Urgency, Dignity/Inclusion, and Leaving Risk pill text with the expected semantic severity class.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `rtk proxy env XDG_CACHE_HOME=/private/tmp/ai-medreview-font-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_monthly_report.py -k 'rating_card_geometry or review_classification_pills' -v`

Expected: FAIL because the wider column markup and pill rows are absent.

- [ ] **Step 3: Implement rating-card column geometry**

Add a five-column `<colgroup>` with four equal side widths and a wider middle width. Add equal-height card layout and a no-wrap middle label while keeping all content within the A4 table width.

- [ ] **Step 4: Implement review pill markup and print CSS**

Replace `.review-meta` and conditional risk-only badges with a wrapping `.pill-row`. Render all five pills per review and map existing score/flag values to the approved classes using Jinja conditionals. Use pale fills, dark text, visible borders, and overflow wrapping.

- [ ] **Step 5: Run focused and full tests**

Run: `rtk proxy env XDG_CACHE_HOME=/private/tmp/ai-medreview-font-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_monthly_report.py -v`

Expected: all report tests pass.

Run: `rtk proxy env XDG_CACHE_HOME=/private/tmp/ai-medreview-font-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v`

Expected: all project tests pass.

- [ ] **Step 6: Generate and visually inspect a representative PDF**

Generate the October 2026 Earls Court Medical Centre sample under `output/pdf/`, render all pages to `tmp/pdfs/`, and inspect every PNG for one-line middle label, equal card heights, A4 containment, all pill states, long-label wrapping, clipping, overlap, and contrast. Fix under RED-GREEN tests if needed.

- [ ] **Step 7: Commit**

```bash
rtk git add ai_medreview/templates/monthly_report.html tests/test_monthly_report.py docs/superpowers/specs/2026-10-25-monthly-report-rating-cards-and-pills-design.md docs/superpowers/plans/2026-10-25-monthly-report-rating-cards-and-pills.md
rtk git commit -m "feat: refine monthly report cards and pills"
```

- [ ] **Step 8: Final verification**

Run: `rtk git diff --check`

Expected: no whitespace errors.

Run: `rtk git status --short`

Expected: clean tree or only generated visual-verification artifacts remain.
