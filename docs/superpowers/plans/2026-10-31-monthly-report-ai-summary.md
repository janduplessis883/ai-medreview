# Monthly report AI summary implementation plan

> Execute inline with superpowers:executing-plans; the user authorised implementation.

**Goal:** Add evidence-linked monthly insights and practical responses to the PDF.
**Architecture:** A focused summary module calls the existing external LLM client and validates structured JSON. PDF generation attaches validated output to deterministic report context; Streamlit clears previous downloads before each attempt.
**Tech stack:** Python, requests, Jinja2, WeasyPrint, Streamlit, pytest.
**Spec:** docs/superpowers/specs/2026-10-31-monthly-report-ai-summary-design.md

## Constraints and review focus

Current surgery/month only; no provider change or automatic fallback. No network in tests. Require valid evidence references, escaped output, a maximum of three entries per section and 350 words overall. Sparse or missing feedback, malformed responses, stale downloads, service failures and oversized input must fail or render honestly.

## Task 1: Summary generation and validation

- [x] Add failing tests for scoped evidence, no usable text, malformed JSON, unknown references, excessive items/words, empty findings and service failure.
- [x] Implement ai_medreview/monthly_summary.py with generate_ai_summary(context) using ai_medreview.llm_client.call_llm and the approved prompt.
- [x] Supply all scoped review text with references and verified aggregate statistics. Validate JSON before returning template data. No silent truncation.
- [x] Run focused tests with mocked service responses.

## Task 2: PDF integration and retry flow

- [x] Add failing PDF tests for section order, evidence references and escaped model content; add an AppTest for failure after successful generation followed by recovery.
- [x] Add stable references to report entries and call summary generation only from generate_monthly_report_pdf, never preview.
- [x] Render an AI summary page between People and sentiment and Review list, with explicit AI/suggestion labelling.
- [x] Clear prior download state before report generation and show a safe actionable retry message on failure.
- [x] Update existing PDF tests to mock the external service; run report tests and full pytest suite, then review the diff and render a sample PDF for visual inspection.

## Execution record

Plan follows the approved design. Implement in the current workspace; do not publish or push. Summary evidence is paraphrased, as proposed in the approved design.

Validation: full suite 61 passed; git diff --check clean. Rendered and visually inspected the AI summary page with synthetic feedback: headings, evidence references and footer fit cleanly. Service calls were mocked; live endpoint behaviour was not tested.

Reliability follow-up: reproduced rejection of valid JSON code fences and duplicate valid citations; added safe normalisation, specific validation codes and a single evidence-preserving correction attempt. New regression tests failed before the fix and passed afterwards. Full suite now 70 passed; live synthetic service probe failed to connect, so actual production rejection frequency is unverified. No patient feedback or response text is logged.
