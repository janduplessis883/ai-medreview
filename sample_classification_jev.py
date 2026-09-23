#!/usr/bin/env python3
"""Classify sample Friends and Family Test reviews with TypeSafe Jev."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import tempfile
from pathlib import Path

from tqdm import tqdm
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

from classification_jev_v3 import (
    DEFAULT_MODEL,
    PRIMARY_TOPIC_CRITERIA,
    PRIMARY_TOPIC_INSTRUCTIONS,
    parse_choice_answer,
)


DEFAULT_INPUT_PATH = Path(__file__).parent / "ai_medreview/data/sample_jev.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "ai_medreview/data/sample_jev_classified.csv"


async def classify_review(
    client: AsyncTypeSafeClient,
    text: str,
    *,
    timeout: float,
) -> tuple[str, float, list[str], dict[str, float]]:
    primary_instructions = dict(PRIMARY_TOPIC_INSTRUCTIONS)
    primary_instructions["selection_rule"] = [
        *PRIMARY_TOPIC_INSTRUCTIONS["selection_rule"],
        "If several topics are mentioned, still choose exactly one primary topic: the issue receiving the greatest emphasis, impact, or detail.",
        "Never choose a multiple-topics or no-clear-primary option.",
    ]
    primary_criteria = {
        category: definition
        for category, definition in PRIMARY_TOPIC_CRITERIA.items()
        if category != "Multiple Topics / No Clear Primary Topic"
    }
    questions = {
        "primary_topic": Choice(instructions=primary_instructions, criteria=primary_criteria),
    }
    secondary_categories = [
        category
        for category in PRIMARY_TOPIC_CRITERIA
        if category not in {"Multiple Topics / No Clear Primary Topic", "Other / Unclassifiable / No Actionable Feedback"}
    ]
    for index, category in enumerate(secondary_categories):
        questions[f"secondary_{index}"] = {
            "type": "noul",
            "instructions": f"Is {category!r} an independently mentioned and actionable topic in this review?",
            "criteria": {
                "true": f"The review makes a distinct comment about {category}.",
                "false": f"{category} is absent or only incidental background detail.",
            },
        }

    response = await client.system_one(
        state=f"Friends and Family Test review: {text}",
        questions=questions,
        timeout=timeout,
    )
    primary, score = parse_choice_answer(response, "primary_topic")
    secondary_scores = {
        category: response.nouls[f"secondary_{index}"].noul
        for index, category in enumerate(secondary_categories)
    }
    return primary, score, [], secondary_scores


async def classify_file(
    input_path: Path,
    output_path: Path,
    *,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    concurrency: int,
    secondary_threshold: float,
) -> int:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if not 0 <= secondary_threshold <= 1:
        raise ValueError("secondary_threshold must be between 0 and 1")

    with input_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "free_text" not in reader.fieldnames:
            raise ValueError("CSV must contain a free_text column")
        fieldnames = list(reader.fieldnames)
        for column in ("jev_category", "jev_score", "jev_secondary_topics", "jev_secondary_scores", "jev_topic_count"):
            if column not in fieldnames:
                fieldnames.append(column)
        rows = list(reader)

    jobs = [
        (index, str(row["free_text"]).strip())
        for index, row in enumerate(rows)
        if str(row.get("free_text", "")).strip()
    ]
    results: list[tuple[str, float, list[str], dict[str, float]] | None] = [None] * len(jobs)
    semaphore = asyncio.Semaphore(concurrency)

    retry = RetryPolicy(max_retries=max_retries, timeout=timeout)
    async with AsyncTypeSafeClient(
        api_key=api_key,
        model=model,
        timeout=timeout,
        retry=retry,
    ) as client:
        async def classify_one(
            job_index: int, row_index: int, text: str
        ) -> tuple[int, tuple[str, float, list[str], dict[str, float]]]:
            async with semaphore:
                return job_index, await classify_review(client, text, timeout=timeout)

        tasks = [
            asyncio.create_task(classify_one(job_index, row_index, text))
            for job_index, (row_index, text) in enumerate(jobs)
        ]
        with tqdm(total=len(tasks), desc="Classifying sample Jev reviews", unit="review") as progress:
            for task in asyncio.as_completed(tasks):
                job_index, result = await task
                results[job_index] = result
                progress.update(1)

    for (row_index, _), result in zip(jobs, results):
        if result is None:
            raise RuntimeError("A Jev classification task completed without a result")
        category, score, _, secondary_scores = result
        secondary_topics = [
            topic for topic, probability in secondary_scores.items()
            if topic != category and probability >= secondary_threshold
        ]
        rows[row_index]["jev_category"] = category
        rows[row_index]["jev_score"] = str(score)
        rows[row_index]["jev_secondary_topics"] = " | ".join(secondary_topics)
        rows[row_index]["jev_secondary_scores"] = json.dumps(
            {topic: round(probability, 6) for topic, probability in secondary_scores.items() if topic in secondary_topics},
            ensure_ascii=False,
            sort_keys=True,
        )
        rows[row_index]["jev_topic_count"] = str(1 + len(secondary_topics))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=output_path.parent, delete=False
    ) as temporary_file:
        writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(output_path)
    return len(jobs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--secondary-threshold",
        type=float,
        default=0.6,
        help="Minimum Noul probability for including a secondary topic (default: 0.6).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("TYPESAFE_API_KEY must be set")

    asyncio.run(
        classify_file(
            args.input,
            args.output,
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            concurrency=args.concurrency,
            secondary_threshold=args.secondary_threshold,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
