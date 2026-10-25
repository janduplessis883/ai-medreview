# Monthly report staff and sentiment page

## Intent

Add one engaging page to the existing surgery-specific monthly PDF report. The
page helps a reader understand who was mentioned in the month's reviews and the
overall emotional tone of those reviews without opening the Streamlit dashboard.

## Goals

- Use only reviews for the selected surgery and calendar month.
- Show Positive, Neutral or Mixed, and Negative sentiment counts and percentages.
- Show the average sentiment strength for the month when classified values exist.
- Show staff groups ranked by review count, with a compact sentiment mix for each.
- Keep the existing modern, restrained HTML/CSS PDF visual language.
- Add no new Python or system dependencies.

## Non-goals

- No new dashboard controls or filters.
- No new machine-learning classification.
- No replacement of the existing sentiment or staff classifications.

## Approach options

1. HTML/CSS page in the existing Jinja/WeasyPrint template (selected): keeps one
   layout system, avoids dependencies, and supports polished cards and metrics.
2. Matplotlib image charts: expressive but adds a dependency and embeds raster
   visuals into an otherwise vector-oriented report.
3. ReportLab page: would create a second PDF layout system and duplicate styling.

## Proposed page

The page follows the classification summary and score distributions. It opens
with a large sentiment headline based on the dominant category, a three-part
sentiment count/percentage strip, and an average-strength callout on the 0–4
scale. Below that, ranked staff cards show the group name, review count and a
small positive/neutral/negative text mix. Missing or unclassified values are
excluded from the corresponding metric; if no values are classified, the page
shows a clear neutral “not classified” state rather than failing.

## Data rules

- Sentiment categories use the existing labels and order: Positive, Neutral or
  Mixed, Negative.
- Dominant sentiment uses highest count, with the existing category order as a
  deterministic tie-break.
- Average sentiment strength is the numeric mean of valid integer scores 0–4,
  rounded to one decimal.
- Staff groups are sorted by descending review count, then alphabetically for
  ties. Blank staff groups are displayed as “No specific staff mentioned”.
- Staff sentiment mix counts only classified sentiment labels and remains
  surgery/month scoped.

## Verification

Add context tests for sentiment and staff aggregation, template tests for the
new page, PDF text/page checks, and a rendered visual inspection of every page.
