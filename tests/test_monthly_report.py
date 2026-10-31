from __future__ import annotations

import pandas as pd
import pytest
import json
from unittest.mock import Mock

from datetime import date
from io import BytesIO
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from pypdf import PdfReader
from streamlit.testing.v1 import AppTest
from weasyprint import HTML

from ai_medreview.monthly_report import (
    build_report_context,
    filter_report_reviews,
    generate_monthly_report_pdf,
    render_report_html,
    report_selection_key,
    safe_report_filename,
    TEMPLATE_DIRECTORY,
)


@pytest.fixture(autouse=True)
def summary_service(monkeypatch):
    """Keep PDF and app integration tests off the real patient-data service."""
    from ai_medreview import llm_client

    service = Mock(return_value=json.dumps({
        "evidence_note": "AI interpretation of this month's written feedback.",
        "strengths": [{"text": "Patients valued helpful support.", "references": ["R1"]}],
        "concerns": [],
        "responses": [{"text": "Consider preserving helpful support.", "references": ["R1"]}],
    }))
    monkeypatch.setattr(llm_client, "call_llm", service)
    return service


def test_pdf_contains_evidence_linked_ai_summary_in_requested_order(summary_service):
    pdf = generate_monthly_report_pdf(classified_reviews(), "Oak Medical Centre", "2026-10")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages)
    assert text.index("People and sentiment") < text.index("AI summary") < text.index("Review list")
    assert "What patients value" in text
    assert "Where patients experience difficulties" in text
    assert "Suggested practical responses" in text
    assert "Consider preserving helpful support." in text
    assert text.count("R1") >= 3
    assert "R2" in text
    summary_service.assert_called_once()


def test_ai_summary_html_is_escaped():
    context = build_report_context(classified_reviews(), "Oak Medical Centre", "2026-10")
    context["ai_summary"] = {
        "evidence_note": "<script>private()</script>",
        "strengths": [{"text": "<img src='https://example.com'>", "references": ["R1"]}],
        "concerns": [],
        "responses": [{"text": "Consider checking instructions & wording.", "references": ["R1"]}],
    }
    html = render_report_html(context)
    assert "&lt;script&gt;private()&lt;/script&gt;" in html
    assert "&lt;img" in html
    assert "<img src=" not in html
    assert "instructions &amp; wording" in html


def test_failed_retry_removes_previous_download_and_can_recover(summary_service):
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py").run(timeout=30)
    assert not app.exception
    summary_service.assert_not_called()
    app.button(key="generate_monthly_report").click().run(timeout=60)
    assert app.download_button
    summary_service.side_effect = ValueError("private provider details")
    app.button(key="generate_monthly_report").click().run(timeout=60)
    assert not app.exception
    assert not app.download_button
    assert any("Generate report" in error.value for error in app.error)
    assert all("private provider details" not in error.value for error in app.error)
    summary_service.side_effect = None
    app.button(key="generate_monthly_report").click().run(timeout=60)
    assert not app.exception
    assert app.download_button


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


def staff_sentiment_reviews() -> pd.DataFrame:
    source = classified_reviews().iloc[[0]].copy()
    rows = []
    values = [
        ("Positive", 4, "GP"),
        ("Positive", 2, "GP"),
        ("Positive", 9, "GP"),
        ("Neutral or Mixed", 3, "Receptionist"),
        ("Neutral or Mixed", "bad", "Receptionist"),
        ("Negative", 0, ""),
        ("Unexpected label", 1, ""),
    ]
    for index, (sentiment, strength, staff) in enumerate(values):
        row = source.copy()
        row["time"] = f"2026-10-{20 - index:02d} 12:00:00"
        row["review"] = f"Staff sentiment review {index}"
        row["jev_sentiment"] = sentiment
        row["jev_sentiment_strength"] = strength
        row["jev_staff_group"] = staff
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def test_report_context_summarizes_sentiment_and_staff_for_the_month() -> None:
    context = build_report_context(
        staff_sentiment_reviews(), "Oak Medical Centre", "2026-10"
    )

    sentiment = context["sentiment"]
    assert sentiment["distribution"] == [
        {"label": "Positive", "count": 3, "percentage": 42.9},
        {"label": "Neutral or Mixed", "count": 2, "percentage": 28.6},
        {"label": "Negative", "count": 1, "percentage": 14.3},
    ]
    assert sentiment["dominant_label"] == "Positive"
    assert sentiment["dominant_count"] == 3
    assert sentiment["classified_count"] == 6
    assert sentiment["average_strength"] == 2.0

    assert context["staff_groups"] == [
        {
            "label": "GP",
            "count": 3,
            "percentage": 42.9,
            "positive_count": 3,
            "neutral_count": 0,
            "negative_count": 0,
        },
        {
            "label": "No specific staff mentioned",
            "count": 2,
            "percentage": 28.6,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 1,
        },
        {
            "label": "Receptionist",
            "count": 2,
            "percentage": 28.6,
            "positive_count": 0,
            "neutral_count": 2,
            "negative_count": 0,
        },
    ]


def test_report_context_staff_sentiment_defaults_when_fields_are_missing() -> None:
    source = classified_reviews().drop(
        columns=["jev_sentiment", "jev_sentiment_strength", "jev_staff_group"],
        errors="ignore",
    )

    context = build_report_context(source, "Oak Medical Centre", "2026-10")

    assert context["sentiment"] == {
        "distribution": [
            {"label": "Positive", "count": 0, "percentage": 0.0},
            {"label": "Neutral or Mixed", "count": 0, "percentage": 0.0},
            {"label": "Negative", "count": 0, "percentage": 0.0},
        ],
        "dominant_label": "Not classified",
        "dominant_count": 0,
        "classified_count": 0,
        "average_strength": None,
    }
    assert context["staff_groups"] == []


def test_report_context_ignores_boolean_sentiment_strength_values() -> None:
    source = staff_sentiment_reviews().iloc[[0, 1]].copy()
    source["jev_sentiment_strength"] = [True, False]

    context = build_report_context(source, "Oak Medical Centre", "2026-10")

    assert context["sentiment"]["average_strength"] is None


def test_people_and_sentiment_page_renders_summary_and_staff_cards() -> None:
    context = build_report_context(
        staff_sentiment_reviews(), "Oak Medical Centre", "2026-10"
    )

    html = render_report_html(context)

    for expected in (
        "People and sentiment",
        "Positive",
        "Neutral or Mixed",
        "Negative",
        "Overall sentiment",
        "Average sentiment strength",
        "Staff groups mentioned",
        "GP",
        "Receptionist",
        "No specific staff mentioned",
    ):
        assert expected in html


def test_people_and_sentiment_page_has_not_classified_fallback_in_pdf() -> None:
    source = classified_reviews().drop(
        columns=["jev_sentiment", "jev_sentiment_strength", "jev_staff_group"],
        errors="ignore",
    )
    pdf_bytes = generate_monthly_report_pdf(source, "Oak Medical Centre", "2026-10")
    reader = PdfReader(BytesIO(pdf_bytes))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert len(reader.pages) >= 4
    assert "People and sentiment" in extracted
    assert "Not classified" in extracted


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


def test_malformed_classifications_use_neutral_defaults() -> None:
    source = classified_reviews().iloc[:2].copy()
    source["jev_actionability"] = [2.9, float("inf")]
    source["jev_urgency"] = [True, "urgent"]
    source["jev_dignity_or_inclusion_concern"] = ["False", "not-a-bool"]
    source["jev_leaving_risk"] = ["0", "unknown"]
    source["jev_primary_topic"] = ["   ", pd.NA]

    context = build_report_context(source, "Oak Medical Centre", "2026-10")

    assert context["actionability"]["elevated_count"] == 0
    assert context["urgency"]["elevated_count"] == 0
    assert context["dignity_inclusion"]["count"] == 0
    assert context["leaving_risk"]["count"] == 0
    assert context["topics"] == [
        {"label": "Not classified", "count": 2, "percentage": 100.0}
    ]
    for review in context["reviews"]:
        assert review["topic"] == "Not classified"
        assert review["actionability"] == {
            "score": None,
            "label": "Not classified",
        }
        assert review["urgency"] == {"score": None, "label": "Not classified"}
        assert review["dignity_inclusion"] is False
        assert review["leaving_risk"] is False


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


def test_review_entries_group_by_category_frequency_then_keep_newest_first() -> None:
    context = build_report_context(classified_reviews(), "Oak Medical Centre", "2026-10")

    assert [review["topic"] for review in context["reviews"]] == [
        "Overall service",
        "Overall service",
        "Access",
        "Not classified",
    ]
    assert [review["text"] for review in context["reviews"][:2]] == [
        "Very helpful reception team.",
        "I felt dismissed and may change practice.",
    ]


def pill_reviews() -> pd.DataFrame:
    base = classified_reviews().iloc[[0]].copy()
    rows = []
    for index, score in enumerate([0, 1, 2, 3, pd.NA]):
        row = base.copy()
        row["time"] = f"2026-10-{20 - index:02d} 12:00:00"
        row["review"] = f"Pill state {index}"
        row["jev_actionability"] = score
        row["jev_urgency"] = score
        row["jev_primary_topic"] = (
            "A very long category label that needs to wrap safely inside its pill"
            if index == 0
            else "Access"
        )
        row["jev_dignity_or_inclusion_concern"] = index == 3
        row["jev_leaving_risk"] = index == 3
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def test_rating_card_geometry_widens_middle_card_and_keeps_label_on_one_line() -> None:
    context = build_report_context(
        pill_reviews(), "Oak Medical Centre", "2026-10"
    )

    html = render_report_html(context)

    assert html.count('<col class="rating-col rating-col-side">') == 4
    assert html.count('<col class="rating-col rating-col-middle">') == 1
    assert "rating-label rating-label-single-line" in html
    assert "white-space: nowrap" in html
    assert "height: 34mm" in html


def test_review_classification_pills_show_every_criterion_and_severity() -> None:
    context = build_report_context(
        pill_reviews(), "Oak Medical Centre", "2026-10"
    )

    html = render_report_html(context)

    assert html.count('class="pill ') == 25
    for expected in (
        "A very long category label that needs to wrap safely inside its pill",
        "Actionability: None",
        "Actionability: Low",
        "Actionability: Moderate",
        "Actionability: High",
        "Actionability: Not classified",
        "Urgency: None",
        "Urgency: Low",
        "Urgency: Moderate",
        "Urgency: High",
        "Urgency: Not classified",
        "Dignity / inclusion: No concern",
        "Dignity / inclusion: Concern",
        "Leaving risk: Not indicated",
        "Leaving risk: Indicated",
    ):
        assert expected in html
    cards = re.findall(
        r'<article class="review-card">(.*?)</article>', html, flags=re.DOTALL
    )
    cards_by_state = {
        state: next(card for card in cards if f"Pill state {state}" in card)
        for state in range(5)
    }

    def pill_class(card: str, text: str) -> str:
        match = re.search(
            rf'<span class="pill ([^"]+)">{re.escape(text)}</span>', card
        )
        assert match is not None
        return match.group(1)

    expected_score_classes = {
        0: ("None", "severity-0"),
        1: ("Low", "severity-1"),
        2: ("Moderate", "severity-2"),
        3: ("High", "severity-3"),
        4: ("Not classified", "severity-unclassified"),
    }
    for state, (label, expected_class) in expected_score_classes.items():
        card = cards_by_state[state]
        assert pill_class(card, f"Actionability: {label}") == expected_class
        assert pill_class(card, f"Urgency: {label}") == expected_class
        category_text = (
            "A very long category label that needs to wrap safely inside its pill"
            if state == 0
            else "Access"
        )
        assert pill_class(card, category_text) == "pill-category"

    assert "Category:" not in html

    assert pill_class(cards_by_state[0], "Dignity / inclusion: No concern") == (
        "risk-inactive"
    )
    assert pill_class(cards_by_state[0], "Leaving risk: Not indicated") == (
        "risk-inactive"
    )
    assert pill_class(cards_by_state[3], "Dignity / inclusion: Concern") == (
        "risk-active"
    )
    assert pill_class(cards_by_state[3], "Leaving risk: Indicated") == "risk-active"


def test_long_category_pill_wraps_within_review_card_bounds() -> None:
    source = classified_reviews().iloc[[0]].copy()
    source["jev_primary_topic"] = "A" * 220
    context = build_report_context(source, "Oak Medical Centre", "2026-10")
    html = render_report_html(context)

    document = HTML(string=html, base_url=str(TEMPLATE_DIRECTORY)).render()
    review_box = None
    category_pill = None
    for page in document.pages:
        stack = [page._page_box]
        while stack:
            box = stack.pop()
            element = getattr(box, "element", None)
            classes = element.get("class", "") if element is not None else ""
            if classes == "review-card" and type(box).__name__ == "BlockBox":
                review_box = box
            if classes == "pill pill-category" and type(box).__name__ == (
                "InlineBlockBox"
            ):
                category_pill = box
            stack.extend(reversed(list(box.all_children())))

    assert review_box is not None
    assert category_pill is not None
    assert category_pill.position_x >= review_box.position_x
    assert category_pill.position_x + category_pill.width <= (
        review_box.position_x + review_box.width
    )
    assert category_pill.height > 20


def test_inactive_pill_text_matches_its_outline_colour() -> None:
    source = pill_reviews().iloc[[0]]
    context = build_report_context(source, "Oak Medical Centre", "2026-10")
    html = render_report_html(context)

    document = HTML(string=html, base_url=str(TEMPLATE_DIRECTORY)).render()
    inactive_pills = []
    for page in document.pages:
        stack = [page._page_box]
        while stack:
            box = stack.pop()
            element = getattr(box, "element", None)
            classes = element.get("class", "") if element is not None else ""
            if type(box).__name__ == "InlineBlockBox" and (
                "severity-0" in classes or "risk-inactive" in classes
            ):
                inactive_pills.append(box)
            stack.extend(reversed(list(box.all_children())))

    assert len(inactive_pills) == 4
    for pill in inactive_pills:
        assert pill.style["color"] == pill.style["border_top_color"]


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


def test_pdf_starts_score_distributions_on_a_fresh_page() -> None:
    pdf_bytes = generate_monthly_report_pdf(
        classified_reviews(), "Oak Medical Centre", "2026-10"
    )
    pages = PdfReader(BytesIO(pdf_bytes)).pages
    first_page = pages[0].extract_text() or ""
    second_page = pages[1].extract_text() or ""

    assert "Actionability distribution" not in first_page
    assert "Score distributions" in second_page
    assert "0 · None" in second_page
    assert "3 · High" in second_page


def test_unrated_review_uses_a_single_user_friendly_label() -> None:
    source = classified_reviews().iloc[[0]].copy()
    source["rating"] = pd.NA
    source["rating_score"] = pd.NA
    context = build_report_context(source, "Oak Medical Centre", "2026-10")

    html = render_report_html(context)

    assert "Not rated · Not rated" not in html
    assert ">Not rated<" in html


def test_streamlit_app_generates_download_from_monthly_report_tab() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py").run(
        timeout=30
    )

    assert not app.exception
    assert any(tab.label.endswith("Monthly report") for tab in app.tabs)
    assert "Report surgery" in [selectbox.label for selectbox in app.selectbox]
    assert "Reporting month" in [selectbox.label for selectbox in app.selectbox]

    generate = next(button for button in app.button if button.label == "Generate report")
    generate.click().run(timeout=60)

    assert not app.exception
    assert "Download PDF" in [button.label for button in app.download_button]

    month_selector = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Reporting month"
    )
    assert len(month_selector.options) > 1
    month_selector.set_value(month_selector.options[1]).run(timeout=30)
    assert not app.download_button

    next(button for button in app.button if button.label == "Generate report").click().run(
        timeout=60
    )
    assert "Download PDF" in [button.label for button in app.download_button]

    surgery_selector = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Report surgery"
    )
    surgery_selector.set_value(surgery_selector.options[1]).run(timeout=30)
    assert not app.download_button


def test_streamlit_cloud_declares_weasyprint_system_dependencies() -> None:
    package_lines = (Path(__file__).parents[1] / "packages.txt").read_text().splitlines()
    packages = {
        line.strip()
        for line in package_lines
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert all(line.strip() and not line.lstrip().startswith("#") for line in package_lines)
    assert {
        "libcairo2",
        "libffi-dev",
        "libglib2.0-0t64",
        "libgdk-pixbuf-2.0-0",
        "libharfbuzz-subset0",
        "libpango-1.0-0",
        "libpangoft2-1.0-0",
        "libpangocairo-1.0-0",
        "shared-mime-info",
    } <= packages


def test_built_wheel_includes_monthly_report_template(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    dist_dir = tmp_path / "dist"
    egg_dir = tmp_path / "egg"
    dist_dir.mkdir()
    egg_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "egg_info",
            "--egg-base",
            str(egg_dir),
            "build",
            "--build-base",
            str(tmp_path / "build"),
            "bdist_wheel",
            "--bdist-dir",
            str(tmp_path / "bdist"),
            "--dist-dir",
            str(dist_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_path = next(dist_dir.glob("*.whl"))

    with zipfile.ZipFile(wheel_path) as wheel:
        assert "ai_medreview/templates/monthly_report.html" in wheel.namelist()
