import pandas as pd
import pytest

from ai_medreview import presentation
from ai_medreview.presentation import format_flag_score


def test_format_flag_score_shows_plain_score_when_inactive():
    assert format_flag_score("Clinical safety", False, 0.02) == "Clinical safety: 0.02"


def test_format_flag_score_adds_warning_indicator_when_active():
    assert format_flag_score("Clinical safety", True, 0.82) == "⚠️ Clinical safety: 0.82"


def test_format_score_includes_its_confidence():
    formatter = getattr(presentation, "format_score_with_confidence", None)

    assert formatter is not None, "score formatter is missing"
    assert (
        formatter("Sentiment strength", 3, 4, 0.87)
        == "Sentiment strength: 3 / 4 (confidence 0.87)"
    )


def test_review_scores_use_their_matching_confidence_fields():
    formatter = getattr(presentation, "format_review_scores", None)
    row = {
        "jev_sentiment_strength": 3,
        "jev_sentiment_strength_confidence": 0.91,
        "jev_actionability": 2,
        "jev_actionability_confidence": 0.82,
        "jev_urgency": 1,
        "jev_urgency_confidence": 0.73,
    }

    assert formatter is not None, "review score formatter is missing"
    assert formatter(row) == [
        "Sentiment strength: 3 / 4 (confidence 0.91)",
        "Actionability: 2 / 3 (confidence 0.82)",
        "Urgency: 1 / 3 (confidence 0.73)",
    ]


def test_review_score_progress_bar_uses_zero_to_three_scale():
    options_builder = getattr(presentation, "review_score_progress_options", None)

    assert options_builder is not None, "progress-bar options builder is missing"
    assert options_builder("Actionability") == {
        "label": "Actionability",
        "min_value": 0,
        "max_value": 3,
        "format": "%d",
    }
@pytest.mark.parametrize("value, expected", [
    ("['Alice', 'Bob', 'Alice']", ["Alice", "Bob"]),
    (["Alice", " Alice ", ""], ["Alice"]),
    (None, []), (float("nan"), []), ("[]", []),
    ("malformed [", []), ("{'Alice': 1}", []),
])
def test_review_people_handles_csv_lists_and_missing_values(value, expected):
    from ai_medreview.presentation import review_people
    assert review_people(value) == expected


def test_review_display_adds_names_only_when_enabled_without_changing_source():
    from ai_medreview.presentation import reviews_for_display
    source = pd.DataFrame({"review": ["Helpful PERSON", "No names"], "review_PER": ["['Alice']", None]})
    assert reviews_for_display(source, False)["review"].tolist() == source["review"].tolist()
    assert reviews_for_display(source, True)["review"].tolist() == [
        "Helpful PERSON\n\nPeople mentioned: Alice", "No names",
    ]
    assert source.iloc[0]["review"] == "Helpful PERSON"
