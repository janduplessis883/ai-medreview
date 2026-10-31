"""Formatting helpers for the Streamlit dashboard."""

from collections.abc import Mapping
import ast

import pandas as pd


def review_people(value: object) -> list[str]:
    """Read the CSV's literal name list without executing its contents."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(name.strip() for name in value if isinstance(name, str) and name.strip()))


def reviews_for_display(reviews: pd.DataFrame, show_names: bool) -> pd.DataFrame:
    """Append identified people below review text in dashboard tables."""
    display = reviews.copy()
    if show_names and "review_PER" in display:
        display["review"] = [
            f"{text}\n\nPeople mentioned: {', '.join(names)}" if names else text
            for text, names in zip(display["review"], display["review_PER"].map(review_people))
        ]
    return display


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


def review_score_progress_options(label: str) -> dict[str, object]:
    return {
        "label": label,
        "min_value": 0,
        "max_value": 3,
        "format": "%d",
    }
