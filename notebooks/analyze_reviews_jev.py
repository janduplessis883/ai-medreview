#!/usr/bin/env python3
"""Analyze patient feedback with TypeSafe Jev."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy, Score

from classification_jev_v3 import PRIMARY_TOPIC_CRITERIA, parse_choice_answer


DEFAULT_MODEL = "jev-latest"
MULTIPLE_TOPIC = "Multiple Topics / No Clear Primary Topic"
NO_ACTIONABLE = "Other / Unclassifiable / No Actionable Feedback"


def build_questions() -> tuple[dict[str, Any], list[str]]:
    topic_criteria = {
        name: definition
        for name, definition in PRIMARY_TOPIC_CRITERIA.items()
        if name != MULTIPLE_TOPIC
    }
    questions: dict[str, Any] = {
        "sentiment": Choice(
            instructions={
                "question": "What is the overall sentiment of this patient feedback?",
                "rules": [
                    "Classify sentiment, not the topic.",
                    "Use Neutral or Mixed when the feedback is factual, ambiguous, or balanced.",
                ],
            },
            criteria={
                "Positive": "The overall feedback is favourable, appreciative, or endorsing.",
                "Negative": "The overall feedback is dissatisfied, critical, or reports a problem.",
                "Neutral or Mixed": "The feedback is factual, ambiguous, or contains balanced positive and negative content.",
            },
        ),
        "primary_topic": Choice(
            instructions={
                "question": "What is the primary topic of this patient feedback?",
                "rules": [
                    "Choose exactly one topic.",
                    "If several topics are present, choose the issue receiving the greatest emphasis, impact, or detail.",
                    "Never choose a multiple-topics category.",
                ],
            },
            criteria=topic_criteria,
        ),
        "sentiment_strength": Score(
            instructions="How strongly does the review express its overall sentiment?",
            criteria=[
                "Neutral or purely factual; little evaluative language.",
                "Mildly positive or mildly negative.",
                "Clearly positive or clearly negative.",
                "Strongly positive or strongly negative.",
                "Extremely emphatic praise or criticism, including serious dissatisfaction.",
            ],
        ),
        "actionability": Score(
            instructions="How actionable is this feedback for practice improvement?",
            criteria=[
                "No actionable issue or only general praise.",
                "Some indication of a possible improvement.",
                "A clear service issue or useful improvement suggestion.",
                "A specific issue that can be addressed by the practice.",
            ],
        ),
        "urgency": Score(
            instructions="How urgently should this feedback be reviewed by a human?",
            criteria=[
                "No urgency; routine feedback.",
                "Worth monitoring but not time-sensitive.",
                "Should be reviewed soon.",
                "Requires prompt attention because it suggests serious harm, safety, discrimination, or unresolved risk.",
            ],
        ),
    }

    secondary_topics = [
        name for name in topic_criteria if name != NO_ACTIONABLE
    ]
    for index, topic in enumerate(secondary_topics):
        questions[f"secondary_{index}"] = Noul(
            instructions=f"Is {topic!r} an independently mentioned and actionable topic in this review?",
            criteria={
                "true": f"The review makes a distinct comment about {topic}.",
                "false": f"{topic} is absent or only incidental background detail.",
            },
        )
    return questions, secondary_topics


def answer_value(answer: Any, attribute: str) -> Any:
    return getattr(answer, attribute)


async def analyze_one(
    client: AsyncTypeSafeClient,
    text: str,
    *,
    questions: dict[str, Any],
    secondary_topics: list[str],
    timeout: float,
    secondary_threshold: float,
) -> dict[str, Any]:
    response = await client.system_one(
        state={"review": text},
        questions=questions,
        timeout=timeout,
    )
    sentiment_answer = response.choices["sentiment"]
    primary_answer = response.choices["primary_topic"]
    primary_topic, primary_confidence = parse_choice_answer(response, "primary_topic")
    secondary_scores = {
        topic: answer_value(response.nouls[f"secondary_{index}"], "noul")
        for index, topic in enumerate(secondary_topics)
    }
    selected_secondary = [
        topic
        for topic, probability in secondary_scores.items()
        if topic != primary_topic and probability >= secondary_threshold
    ]
    return {
        "sentiment": answer_value(sentiment_answer, "choice"),
        "sentiment_confidence": answer_value(sentiment_answer, "confidence"),
        "primary_topic": primary_topic,
        "primary_topic_confidence": primary_confidence,
        "secondary_topics": selected_secondary,
        "secondary_topic_scores": secondary_scores,
        "sentiment_strength": answer_value(response.scores["sentiment_strength"], "score"),
        "sentiment_strength_confidence": answer_value(response.scores["sentiment_strength"], "confidence"),
        "actionability": answer_value(response.scores["actionability"], "score"),
        "actionability_confidence": answer_value(response.scores["actionability"], "confidence"),
        "urgency": answer_value(response.scores["urgency"], "score"),
        "urgency_confidence": answer_value(response.scores["urgency"], "confidence"),
    }


async def analyze_dataframe(
    reviews: pd.DataFrame,
    *,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    concurrency: int,
    secondary_threshold: float,
) -> pd.DataFrame:
    if "free_text" not in reviews.columns:
        raise ValueError("DataFrame must contain a free_text column")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if not 0 <= secondary_threshold <= 1:
        raise ValueError("secondary_threshold must be between 0 and 1")
    rows = reviews.copy().fillna("").to_dict(orient="records")

    questions, secondary_topics = build_questions()
    jobs = [(index, str(row["free_text"]).strip()) for index, row in enumerate(rows) if str(row.get("free_text", "")).strip()]
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    semaphore = asyncio.Semaphore(concurrency)
    retry = RetryPolicy(max_retries=max_retries, timeout=timeout)

    async with AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout, retry=retry) as client:
        async def run_one(job_index: int, text: str) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                return job_index, await analyze_one(
                    client,
                    text,
                    questions=questions,
                    secondary_topics=secondary_topics,
                    timeout=timeout,
                    secondary_threshold=secondary_threshold,
                )

        tasks = [asyncio.create_task(run_one(index, text)) for index, (_, text) in enumerate(jobs)]
        with tqdm(total=len(tasks), desc="Analyzing Jev reviews", unit="review") as progress:
            for task in asyncio.as_completed(tasks):
                index, result = await task
                results[index] = result
                progress.update(1)

    output_fields = [
        *reviews.columns.tolist(),
        "jev_sentiment",
        "jev_sentiment_confidence",
        "jev_primary_topic",
        "jev_primary_topic_confidence",
        "jev_secondary_topics",
        "jev_secondary_topic_scores",
        "jev_sentiment_strength",
        "jev_sentiment_strength_confidence",
        "jev_actionability",
        "jev_actionability_confidence",
        "jev_urgency",
        "jev_urgency_confidence",
    ]
    output_fields = list(dict.fromkeys(output_fields))
    for (row_index, _), result in zip(jobs, results):
        if result is None:
            raise RuntimeError("A Jev analysis task completed without a result")
        rows[row_index].update(
            {
                "jev_sentiment": result["sentiment"],
                "jev_sentiment_confidence": str(result["sentiment_confidence"]),
                "jev_primary_topic": result["primary_topic"],
                "jev_primary_topic_confidence": str(result["primary_topic_confidence"]),
                "jev_secondary_topics": " | ".join(result["secondary_topics"]),
                "jev_secondary_topic_scores": json.dumps(result["secondary_topic_scores"], sort_keys=True),
                "jev_sentiment_strength": str(result["sentiment_strength"]),
                "jev_sentiment_strength_confidence": str(result["sentiment_strength_confidence"]),
                "jev_actionability": str(result["actionability"]),
                "jev_actionability_confidence": str(result["actionability_confidence"]),
                "jev_urgency": str(result["urgency"]),
                "jev_urgency_confidence": str(result["urgency_confidence"]),
            }
        )

    return pd.DataFrame(rows, columns=output_fields)


async def analyze_file(
    input_path: Path,
    output_path: Path,
    **kwargs: Any,
) -> None:
    reviews = pd.read_csv(input_path)
    analyzed = await analyze_dataframe(reviews, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyzed.to_csv(output_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV containing a free_text column")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--secondary-threshold", type=float, default=0.6)
    args = parser.parse_args()
    if args.concurrency < 1 or not 0 <= args.secondary_threshold <= 1:
        parser.error("concurrency must be positive and secondary-threshold must be between 0 and 1")

    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("TYPESAFE_API_KEY must be set")
    asyncio.run(
        analyze_file(
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
