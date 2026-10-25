"""Monthly, surgery-scoped patient review reports."""

from __future__ import annotations

from datetime import date
import math
from numbers import Real
from pathlib import Path
import re
import unicodedata

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd


RATINGS = (
    ("Very good", 5),
    ("Good", 4),
    ("Neither good nor poor", 3),
    ("Poor", 2),
    ("Very poor", 1),
)
SCORE_LABELS = {0: "None", 1: "Low", 2: "Moderate", 3: "High"}
TEMPLATE_DIRECTORY = Path(__file__).parent / "templates"


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
    if pd.api.types.is_bool(value) or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    score = int(numeric)
    return score if score in SCORE_LABELS else None


def _score_info(value: object) -> dict[str, int | str | None]:
    score = _score_value(value)
    return {
        "score": score,
        "label": SCORE_LABELS[score] if score is not None else "Not classified",
    }


def normalize_boolean_flag(value: object) -> bool:
    """Strictly normalize supported boolean representations."""

    if pd.api.types.is_bool(value):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return False
    if isinstance(value, Real) and not pd.isna(value):
        return float(value) == 1.0
    return False


def _topic_label(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Not classified"
    return value.strip()


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
    count = int(values.map(normalize_boolean_flag).sum())
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
    ).map(_topic_label)
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
        topic = _topic_label(row.get("jev_primary_topic"))
        review_entries.append(
            {
                "date": row["time"].strftime("%d %B %Y"),
                "text": "" if pd.isna(row.get("review")) else str(row.get("review")),
                "rating_label": rating_labels.get(stars, "Not rated"),
                "stars": stars,
                "topic": topic,
                "actionability": _score_info(row.get("jev_actionability")),
                "urgency": _score_info(row.get("jev_urgency")),
                "dignity_inclusion": normalize_boolean_flag(
                    row.get("jev_dignity_or_inclusion_concern")
                ),
                "leaving_risk": normalize_boolean_flag(row.get("jev_leaving_risk")),
            }
        )
    review_entries.sort(key=lambda review: str(review["topic"]).casefold())

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


def render_report_html(context: dict[str, object]) -> str:
    """Render a self-contained monthly report HTML document."""

    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIRECTORY),
        autoescape=select_autoescape(enabled_extensions=("html", "xml"), default=True),
    )
    return environment.get_template("monthly_report.html").render(**context)


def generate_monthly_report_pdf(
    reviews: pd.DataFrame,
    surgery: str,
    month: str | pd.Period,
    generated_on: date | None = None,
) -> bytes:
    """Generate an in-memory PDF for a surgery's monthly reviews."""

    from weasyprint import HTML

    scoped = filter_report_reviews(reviews, surgery, month)
    context = build_report_context(scoped, surgery, month, generated_on=generated_on)
    html = render_report_html(context)
    return HTML(string=html, base_url=str(TEMPLATE_DIRECTORY)).write_pdf()
