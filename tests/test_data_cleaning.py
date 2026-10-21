import importlib
import importlib.util

import pandas as pd


def test_duplicate_review_text_keeps_only_the_latest_csv_occurrence():
    spec = importlib.util.find_spec("ai_medreview.data_cleaning")
    assert spec is not None, "data-cleaning module is missing"
    deduplicate_reviews = importlib.import_module(
        "ai_medreview.data_cleaning"
    ).deduplicate_reviews
    reviews = pd.DataFrame(
        [
            {
                "review": "Helpful consultation",
                "surgery": "Practice A",
                "time": "2026-10-01 09:00:00",
                "rating_score": 5,
                "classification": "older",
            },
            {
                "review": "Helpful consultation",
                "surgery": "Practice A",
                "time": "2026-10-01 09:00:00",
                "rating_score": 5,
                "classification": "latest",
            },
            {
                "review": "Helpful consultation",
                "surgery": "Practice A",
                "time": "2026-10-02 09:00:00",
                "rating_score": 5,
                "classification": "separate submission",
            },
        ]
    )

    cleaned = deduplicate_reviews(reviews)

    assert cleaned["classification"].tolist() == ["separate submission"]
