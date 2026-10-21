"""Data-cleaning helpers shared by dashboard entry points."""

from __future__ import annotations

import pandas as pd


REVIEW_IDENTITY_COLUMNS = ["review"]


def deduplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last CSV occurrence of each review text."""
    return df.drop_duplicates(
        subset=REVIEW_IDENTITY_COLUMNS,
        keep="last",
    ).copy()
