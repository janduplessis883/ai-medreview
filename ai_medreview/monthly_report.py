"""Monthly, surgery-scoped patient review reports."""

from __future__ import annotations

from datetime import date
import re
import unicodedata

import pandas as pd


RATINGS = (
    ("Very good", 5),
    ("Good", 4),
    ("Neither good nor poor", 3),
    ("Poor", 2),
    ("Very poor", 1),
)
SCORE_LABELS = {0: "None", 1: "Low", 2: "Moderate", 3: "High"}


def _month_period(month: str | pd.Period) -> pd.Period:
    return pd.Period(month, freq="M")


def report_selection_key(
    surgery: str, month: str | pd.Period
) -> tuple[str, str]:
    """Return a stable key for binding generated output to active selectors."""

    return surgery, str(_month_period(month))


def filter_report_reviews(
    df: pd.DataFrame, surgery: str, month: str | pd.Period
) -> pd.DataFrame:
    """Return reviews for exactly one surgery and calendar month."""

    requested_month = _month_period(month)
    times = pd.to_datetime(df["time"], errors="coerce")
    mask = (df["surgery"] == surgery) & (times.dt.to_period("M") == requested_month)
    reviews = df.loc[mask].copy()
    reviews["time"] = times.loc[mask]
    reviews = reviews.dropna(subset=["time"]).sort_values("time", ascending=False)

    if reviews.empty:
        raise ValueError(
            f"No reviews found for {surgery} in {requested_month.strftime('%B %Y')}."
        )
    if reviews["surgery"].nunique(dropna=False) != 1:
        raise ValueError("Report data must contain exactly one surgery.")
    if reviews["time"].dt.to_period("M").nunique() != 1:
        raise ValueError("Report data must contain exactly one calendar month.")

    return reviews


def _percentage(count: int, total: int) -> float:
    return round((count / total * 100) if total else 0.0, 1)


def _score_value(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    return score if score in SCORE_LABELS else None


def _score_info(value: object) -> dict[str, int | str | None]:
    score = _score_value(value)
    return {
        "score": score,
        "label": SCORE_LABELS[score] if score is not None else "Not classified",
    }


def _is_true(value: object) -> bool:
    return False if pd.isna(value) else bool(value)


def _score_summary(
    reviews: pd.DataFrame, column: str, display_name: str
) -> dict[str, object]:
    values = reviews[column] if column in reviews else pd.Series(index=reviews.index)
    scores = values.map(_score_value)
    distribution = []
    for score, label in SCORE_LABELS.items():
        count = int((scores == score).sum())
        distribution.append(
            {
                "score": score,
                "label": label,
                "count": count,
                "percentage": _percentage(count, len(reviews)),
            }
        )
    elevated_count = int(scores.isin([2, 3]).sum())
    elevated_percentage = _percentage(elevated_count, len(reviews))
    return {
        "distribution": distribution,
        "elevated_count": elevated_count,
        "elevated_percentage": elevated_percentage,
        "summary": (
            f"{elevated_count} of {len(reviews)} reviews "
            f"({elevated_percentage:.1f}%) had moderate or high {display_name.lower()}."
        ),
    }


def _flag_summary(
    reviews: pd.DataFrame, column: str, description: str
) -> dict[str, object]:
    values = reviews[column] if column in reviews else pd.Series(False, index=reviews.index)
    count = int(values.map(_is_true).sum())
    percentage = _percentage(count, len(reviews))
    return {
        "count": count,
        "percentage": percentage,
        "summary": (
            f"{count} of {len(reviews)} reviews ({percentage:.1f}%) "
            f"were classified with {description}."
        ),
    }


def build_report_context(
    reviews: pd.DataFrame,
    surgery: str,
    month: str | pd.Period,
    generated_on: date | None = None,
) -> dict[str, object]:
    """Build deterministic, template-ready monthly report data."""

    scoped = filter_report_reviews(reviews, surgery, month)
    period = _month_period(month)
    total = len(scoped)

    rating_scores = pd.to_numeric(scoped.get("rating_score"), errors="coerce")
    ratings = []
    for label, stars in RATINGS:
        count = int((rating_scores == stars).sum())
        ratings.append(
            {
                "label": label,
                "stars": stars,
                "count": count,
                "percentage": _percentage(count, total),
            }
        )

    topics = scoped.get(
        "jev_primary_topic", pd.Series(index=scoped.index, dtype="object")
    ).fillna("Not classified")
    topics = topics.replace("", "Not classified")
    topic_counts = topics.value_counts(sort=True)
    topic_summaries = [
        {
            "label": str(label),
            "count": int(count),
            "percentage": _percentage(int(count), total),
        }
        for label, count in topic_counts.items()
    ]
    leading_topics = ", ".join(topic["label"] for topic in topic_summaries[:3])
    theme_summary = (
        f"The most frequent classified themes were {leading_topics}."
        if leading_topics
        else "No primary topics were classified for this reporting period."
    )

    rating_labels = {stars: label for label, stars in RATINGS}
    review_entries = []
    for index, row in scoped.iterrows():
        score_value = pd.to_numeric(row.get("rating_score"), errors="coerce")
        stars = int(score_value) if pd.notna(score_value) and score_value in rating_labels else None
        topic = row.get("jev_primary_topic")
        topic = "Not classified" if pd.isna(topic) or not str(topic).strip() else str(topic)
        review_entries.append(
            {
                "date": row["time"].strftime("%d %B %Y"),
                "text": "" if pd.isna(row.get("review")) else str(row.get("review")),
                "rating_label": rating_labels.get(stars, "Not rated"),
                "stars": stars,
                "topic": topic,
                "actionability": _score_info(row.get("jev_actionability")),
                "urgency": _score_info(row.get("jev_urgency")),
                "dignity_inclusion": _is_true(
                    row.get("jev_dignity_or_inclusion_concern")
                ),
                "leaving_risk": _is_true(row.get("jev_leaving_risk")),
            }
        )

    return {
        "surgery": surgery,
        "month_key": str(period),
        "month_label": period.strftime("%B %Y"),
        "generated_on": (generated_on or date.today()).strftime("%d %B %Y"),
        "total_reviews": total,
        "ratings": ratings,
        "actionability": _score_summary(scoped, "jev_actionability", "Actionability"),
        "urgency": _score_summary(scoped, "jev_urgency", "Urgency"),
        "dignity_inclusion": _flag_summary(
            scoped,
            "jev_dignity_or_inclusion_concern",
            "a dignity or inclusion concern",
        ),
        "leaving_risk": _flag_summary(
            scoped, "jev_leaving_risk", "a risk of leaving the practice"
        ),
        "topics": topic_summaries,
        "theme_summary": theme_summary,
        "reviews": review_entries,
    }


def safe_report_filename(surgery: str, month: str | pd.Period) -> str:
    """Return a portable filename for a generated monthly report."""

    ascii_name = (
        unicodedata.normalize("NFKD", surgery).encode("ascii", "ignore").decode()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return f"{slug or 'surgery'}-reviews-{_month_period(month)}.pdf"
