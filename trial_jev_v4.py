#!/usr/bin/env python3
"""Live trial: run the data_v4 JEV analysis on 5 real 2026 reviews.

Uses the actual build_jev_questions / parse_jev_response from ai_medreview.data_v4
so what you see here is exactly what the pipeline will produce.
"""

from __future__ import annotations

import asyncio
import json
import os

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from ai_medreview.data_v4 import (
    JEV_COLUMNS,
    JEV_MAX_RETRIES,
    JEV_MODEL,
    JEV_TIMEOUT,
    build_jev_questions,
    parse_jev_response,
)

DATA = "ai_medreview/data/data_v2.csv"
N_REVIEWS = 5


def pick_trial_reviews() -> list[str]:
    df = pd.read_csv(DATA, usecols=["time", "free_text", "do_better"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df[df["time"] >= "2026-01-01"].copy()
    df["review"] = (
        df["free_text"].fillna("").astype(str).str.strip()
        + " "
        + df["do_better"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["wc"] = df["review"].str.split().str.len()
    df = df[df["wc"] >= 8]
    # Deterministic but varied sample: spread across the period and lengths
    df = df.sort_values("time")
    idx = [int(i * (len(df) - 1) / (N_REVIEWS - 1)) for i in range(N_REVIEWS)]
    return df.iloc[idx]["review"].tolist()


async def main() -> None:
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY not set")

    reviews = pick_trial_reviews()
    questions = build_jev_questions()
    retry = RetryPolicy(max_retries=JEV_MAX_RETRIES, timeout=JEV_TIMEOUT)

    rows: list[dict] = []

    async with AsyncTypeSafeClient(
        api_key=api_key, model=JEV_MODEL, timeout=JEV_TIMEOUT, retry=retry
    ) as client:
        for i, review in enumerate(reviews, 1):
            response = await client.system_one(
                state={"review": review}, questions=questions, timeout=JEV_TIMEOUT
            )
            result = parse_jev_response(response)
            rows.append({"review": review, **result})
            print("=" * 100)
            print(f"REVIEW {i}: {review}")
            print("-" * 100)
            for col in JEV_COLUMNS:
                value = result[col]
                if col == "jev_topic_probabilities":
                    probs = json.loads(value)
                    top5 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    print(f"  {col} (top 5):")
                    for topic, prob in top5:
                        print(f"      {prob:.3f}  {topic}")
                else:
                    print(f"  {col}: {value}")
            print()

    out_path = "ai_medreview/data/trial_jev_v4_output.csv"
    pd.DataFrame(rows, columns=["review", *JEV_COLUMNS]).to_csv(
        out_path, encoding="utf-8", index=False
    )
    print(f"💾 Trial output saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
