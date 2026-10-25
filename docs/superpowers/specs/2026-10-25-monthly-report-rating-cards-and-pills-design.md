# Monthly report rating cards and pills design

## Intent

Refine the existing monthly PDF report so the five rating cards have a uniform visual rhythm and every review presents its classifications as clear, consistently placed pills.

## Goals

- Give the `Neither good nor poor` rating card enough width to keep its label on one line.
- Keep all five rating cards exactly the same height and retain the existing five-to-one FFT mapping, counts, percentages, and bars.
- Replace the review card's inline topic/actionability/urgency metadata with a readable pill row.
- Show the category, Actionability, Urgency, Dignity/Inclusion, and Leaving Risk pills on every review, including inactive risk states.
- Use colour consistently without relying on colour alone: every pill includes its criterion name and current level or state.

## Chosen approach

Keep the report data and Python interfaces unchanged. Implement the refinement in `ai_medreview/templates/monthly_report.html` using semantic pill markup, conditional Jinja classes, a print-safe CSS palette, and a `<colgroup>` that allocates extra width to the middle rating card.

The rejected alternative was calculating CSS class names in Python. That would mix visual presentation into deterministic report analysis without adding capability.

## Rating-card layout

The table uses fixed layout with five explicit column widths. The middle `Neither good nor poor` column receives extra width; the other four remain equal. The label is kept on one line, and all cards use the same fixed/minimum content height with count, star text, and progress bar aligned consistently.

## Review pills

Each review shows one wrapping pill row beneath the review text:

- `Category: <primary topic>` - neutral blue.
- `Actionability: <level>` - severity colour from its 0-3 score.
- `Urgency: <level>` - severity colour from its 0-3 score.
- `Dignity / inclusion: No concern` - white, thin grey outline; `Concern` - red.
- `Leaving risk: Not indicated` - white, thin grey outline; `Indicated` - red.

Actionability and Urgency share this scale:

- 0 / None - white background, thin grey outline, grey text.
- 1 / Low - grey background, darker grey text.
- 2 / Moderate - pale yellow background, darker amber text.
- 3 / High - pale red background, dark red text.
- Not classified - white background, dashed grey outline, grey text.

The palette uses pale fills and dark text for print legibility. Labels make the state understandable in greyscale and accessible without colour perception.

## Scope

Only the PDF template and its rendering tests change. The Streamlit controls, report calculations, surgery/month isolation, output filename, download flow, and dashboard are unchanged.

## Verification

- Rendering tests assert the wider middle-column markup and all pill labels/classes.
- Tests cover all Actionability/Urgency states, inactive and active risk states, and unclassified values.
- A representative repository-data PDF is regenerated, rendered to PNG, and every page is visually checked for equal rating-card heights, a single-line middle label, pill wrapping, clipping, overlap, and readable colours.
