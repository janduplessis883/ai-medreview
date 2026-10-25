from __future__ import annotations

import pandas as pd
import pytest

from datetime import date
from io import BytesIO

from pypdf import PdfReader

from ai_medreview.monthly_report import (
    build_report_context,
    filter_report_reviews,
    generate_monthly_report_pdf,
    render_report_html,
    report_selection_key,
    safe_report_filename,
)


def mixed_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": "2026-09-05 09:00:00",
                "surgery": "Oak Medical Centre",
                "review": "September review",
            },
            {
                "time": "2026-10-03 10:00:00",
                "surgery": "River Surgery",
                "review": "Other surgery review",
            },
            {
                "time": "2026-10-08 08:30:00",
                "surgery": "Oak Medical Centre",
                "review": "Earlier October review",
            },
            {
                "time": "not-a-date",
                "surgery": "Oak Medical Centre",
                "review": "Invalid date review",
            },
            {
                "time": "2026-10-20 14:15:00",
                "surgery": "Oak Medical Centre",
                "review": "Latest October review",
            },
        ]
    )


def test_filter_report_reviews_isolates_exact_surgery_and_month() -> None:
    result = filter_report_reviews(
        mixed_reviews(), "Oak Medical Centre", "2026-10"
    )

    assert result["surgery"].tolist() == [
        "Oak Medical Centre",
        "Oak Medical Centre",
    ]
    assert result["review"].tolist() == [
        "Latest October review",
        "Earlier October review",
    ]
    assert result["time"].dt.to_period("M").astype(str).tolist() == [
        "2026-10",
        "2026-10",
    ]


def test_filter_report_reviews_rejects_empty_scope() -> None:
    with pytest.raises(ValueError, match="No reviews"):
        filter_report_reviews(mixed_reviews(), "Oak Medical Centre", "2026-11")


def classified_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": "2026-10-20 14:15:00",
                "surgery": "Oak Medical Centre",
                "review": "Very helpful reception team.",
                "rating": "Very good",
                "rating_score": 5,
                "jev_primary_topic": "Overall service",
                "jev_actionability": 0,
                "jev_urgency": 0,
                "jev_dignity_or_inclusion_concern": False,
                "jev_leaving_risk": False,
            },
            {
                "time": "2026-10-14 09:30:00",
                "surgery": "Oak Medical Centre",
                "review": "The appointment process could be clearer.",
                "rating": "Good",
                "rating_score": 4,
                "jev_primary_topic": "Access",
                "jev_actionability": 2,
                "jev_urgency": 1,
                "jev_dignity_or_inclusion_concern": False,
                "jev_leaving_risk": False,
            },
            {
                "time": "2026-10-08 08:30:00",
                "surgery": "Oak Medical Centre",
                "review": "I felt dismissed and may change practice.",
                "rating": "Poor",
                "rating_score": 2,
                "jev_primary_topic": "Overall service",
                "jev_actionability": 3,
                "jev_urgency": 2,
                "jev_dignity_or_inclusion_concern": True,
                "jev_leaving_risk": True,
            },
            {
                "time": "2026-10-02 11:00:00",
                "surgery": "Oak Medical Centre",
                "review": "Thank you.",
                "rating": "Very good",
                "rating_score": 5,
                "jev_primary_topic": pd.NA,
                "jev_actionability": pd.NA,
                "jev_urgency": pd.NA,
                "jev_dignity_or_inclusion_concern": pd.NA,
                "jev_leaving_risk": pd.NA,
            },
        ]
    )


def test_build_report_context_calculates_ratings_and_classifications() -> None:
    reviews = filter_report_reviews(
        classified_reviews(), "Oak Medical Centre", "2026-10"
    )

    context = build_report_context(
        reviews,
        "Oak Medical Centre",
        "2026-10",
        generated_on=date(2026, 10, 25),
    )

    assert context["total_reviews"] == 4
    assert context["month_label"] == "October 2026"
    assert context["generated_on"] == "25 October 2026"
    assert [rating["label"] for rating in context["ratings"]] == [
        "Very good",
        "Good",
        "Neither good nor poor",
        "Poor",
        "Very poor",
    ]
    assert [rating["stars"] for rating in context["ratings"]] == [5, 4, 3, 2, 1]
    assert [rating["count"] for rating in context["ratings"]] == [2, 1, 0, 1, 0]
    assert [rating["percentage"] for rating in context["ratings"]] == [
        50.0,
        25.0,
        0.0,
        25.0,
        0.0,
    ]
    assert context["actionability"]["elevated_count"] == 2
    assert context["actionability"]["elevated_percentage"] == 50.0
    assert context["urgency"]["elevated_count"] == 1
    assert context["urgency"]["elevated_percentage"] == 25.0
    assert context["dignity_inclusion"]["count"] == 1
    assert context["leaving_risk"]["count"] == 1
    assert [(topic["label"], topic["count"]) for topic in context["topics"]] == [
        ("Overall service", 2),
        ("Access", 1),
        ("Not classified", 1),
    ]
    assert "2 of 4 reviews (50.0%)" in context["actionability"]["summary"]
    assert "Overall service" in context["theme_summary"]


def test_build_report_context_uses_neutral_defaults_for_missing_values() -> None:
    reviews = filter_report_reviews(
        classified_reviews(), "Oak Medical Centre", "2026-10"
    )

    context = build_report_context(reviews, "Oak Medical Centre", "2026-10")
    missing = next(item for item in context["reviews"] if item["text"] == "Thank you.")

    assert missing["topic"] == "Not classified"
    assert missing["actionability"] == {"score": None, "label": "Not classified"}
    assert missing["urgency"] == {"score": None, "label": "Not classified"}
    assert missing["dignity_inclusion"] is False
    assert missing["leaving_risk"] is False


def test_safe_report_filename_and_selection_key_are_stable() -> None:
    assert safe_report_filename("St John's & West", "2026-10") == (
        "st-john-s-west-reviews-2026-10.pdf"
    )
    assert safe_report_filename("診療所", pd.Period("2026-10", freq="M")) == (
        "surgery-reviews-2026-10.pdf"
    )
    assert report_selection_key("Oak Medical Centre", "2026-10") == (
        "Oak Medical Centre",
        "2026-10",
    )


def test_render_report_html_contains_report_sections_and_ordered_reviews() -> None:
    source = classified_reviews()
    source.loc[source["review"] == "Thank you.", "review"] = (
        "Use <script>alert('x')</script> & confirm"
    )
    reviews = filter_report_reviews(source, "Oak Medical Centre", "2026-10")
    context = build_report_context(
        reviews,
        "Oak Medical Centre",
        "2026-10",
        generated_on=date(2026, 10, 25),
    )

    html = render_report_html(context)

    assert "Oak Medical Centre" in html
    assert "October 2026" in html
    for label in (
        "Very good",
        "Good",
        "Neither good nor poor",
        "Poor",
        "Very poor",
        "Actionability",
        "Urgency",
        "Dignity and inclusion",
        "Leaving risk",
        "Review list",
    ):
        assert label in html
    assert html.index("Very helpful reception team.") < html.index(
        "The appointment process could be clearer."
    )
    assert "&lt;script&gt;alert" in html
    assert "<script>alert" not in html
    assert "&amp; confirm" in html
    assert "http://" not in html
    assert "https://" not in html


def test_generate_monthly_report_pdf_contains_only_selected_surgery() -> None:
    other_surgery = classified_reviews().iloc[[0]].copy()
    other_surgery["surgery"] = "River Surgery"
    other_surgery["review"] = "RIVER-SURGERY-UNIQUE-REVIEW"
    source = pd.concat([classified_reviews(), other_surgery], ignore_index=True)

    pdf_bytes = generate_monthly_report_pdf(
        source,
        "Oak Medical Centre",
        "2026-10",
        generated_on=date(2026, 10, 25),
    )
    extracted = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(pdf_bytes)).pages
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5_000
    assert "Oak Medical Centre" in extracted
    assert "Very helpful reception team." in extracted
    assert "RIVER-SURGERY-UNIQUE-REVIEW" not in extracted


def test_generate_monthly_report_pdf_handles_long_unbroken_review() -> None:
    source = classified_reviews().iloc[[0]].copy()
    source["review"] = "A" * 2_000

    pdf_bytes = generate_monthly_report_pdf(
        source, "Oak Medical Centre", "2026-10"
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert len(PdfReader(BytesIO(pdf_bytes)).pages) >= 2
