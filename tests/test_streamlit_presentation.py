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
