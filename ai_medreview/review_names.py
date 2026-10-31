"""Shared parsing and display of people identified in reviews."""

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


