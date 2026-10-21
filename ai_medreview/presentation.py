"""Formatting helpers for the Streamlit dashboard."""

from collections.abc import Mapping


def format_flag_score(label: str, flagged: bool, score: float) -> str:
    warning = "⚠️ " if flagged else ""
    return f"{warning}{label}: {score:.2f}"


def format_score_with_confidence(
    label: str,
    score: float,
    maximum: int,
    confidence: float,
) -> str:
    return f"{label}: {score:.0f} / {maximum} (confidence {confidence:.2f})"


def format_review_scores(row: Mapping[str, float]) -> list[str]:
    return [
        format_score_with_confidence(
            "Sentiment strength",
            row["jev_sentiment_strength"],
            4,
            row["jev_sentiment_strength_confidence"],
        ),
        format_score_with_confidence(
            "Actionability",
            row["jev_actionability"],
            3,
            row["jev_actionability_confidence"],
        ),
        format_score_with_confidence(
            "Urgency",
            row["jev_urgency"],
            3,
            row["jev_urgency_confidence"],
        ),
    ]
