#!/usr/bin/env python3
"""Classify Friends and Family Test free text with Jev via Vercel AI Gateway."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Callable, Iterable

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r"urllib3 .*", category=Warning)
import requests
from tqdm import tqdm


AI_GATEWAY_EVALUATE_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
DEFAULT_MODEL = "typesafe-ai/jev"
DEFAULT_INPUT_PATH = Path(__file__).parent / "ai_medreview/data/data_jev.csv"

FINAL_CATEGORIES = [
    "Appointment Booking and Online Systems",
    "Appointment Availability and Waiting Times",
    "Difficulty Getting Through on Phone",
    "Reception Staff Rude or Unhelpful",
    "Reception Staff Friendly and Helpful",
    "Prescriptions and Repeat Medication Issues",
    "Blood Tests and Results Delays",
    "Waiting Time in Surgery / Waiting Room",
    "Excellent Clinical Care and Thorough Explanation",
    "Rushed Consultation or Not Listened To",
    "Staff Kindness, Empathy and Compassion",
    "Staff Professionalism and Knowledge",
    "Vaccinations and Immunisations",
    "Telehealth / Phone Consultations",
    "Treatment Quality and Effectiveness",
    "Follow-up and Continuity of Care",
    "Overall Excellent Service and Practice",
    "Irrelevant / Unclassifiable / Noise",
]


def is_missing_text(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def parse_choice_answer(response: dict, answer_id: str) -> tuple[str, float]:
    answers = response.get("answers")
    if not isinstance(answers, dict) or not isinstance(answers.get(answer_id), dict):
        raise RuntimeError(f"Malformed Jev response: missing answer '{answer_id}'")

    answer = answers[answer_id]
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in FINAL_CATEGORIES:
        raise RuntimeError(f"Malformed Jev response: invalid category {choice!r}")

    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)):
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, dict):
            confidence = probabilities.get(choice)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    return choice, max(0.0, min(1.0, float(confidence)))


def classify_with_jev(
    text: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "state": f"Friends and Family Test review: {text}",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Choose the single best category for this review.",
                "criteria": {category: f"The review fits: {category}" for category in FINAL_CATEGORIES},
            }
        },
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(AI_GATEWAY_EVALUATE_URL, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return parse_choice_answer(response.json(), "category")
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            time.sleep(min(2**attempt, 8))

    raise RuntimeError("Unreachable")


def classify_csv(
    input_path: Path,
    output_path: Path,
    *,
    classifier: Callable[[str], tuple[str, float]],
    limit: int | None = None,
    dry_run: bool = False,
    checkpoint_every: int = 100,
) -> tuple[int, int]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")

    with input_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "free_text" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'free_text' column")
        fieldnames = list(reader.fieldnames)
        for column in ("freetext_jev", "freetext_jev_score"):
            if column not in fieldnames:
                fieldnames.append(column)
        rows = list(reader)

    extra_value_count = max((len(row.get(None, [])) for row in rows), default=0)
    for index in range(extra_value_count):
        fieldname = f"extra_csv_value_{index + 1}"
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)
    for row in rows:
        for index, value in enumerate(row.pop(None, []), start=1):
            row[f"extra_csv_value_{index}"] = value

    def write_checkpoint() -> None:
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

    classified = 0
    missing = 0
    updated_since_checkpoint = 0
    for row in tqdm(rows, desc="Classifying reviews", unit="review"):
        text = row.get("free_text")
        if is_missing_text(text):
            row["freetext_jev"] = "nan"
            row["freetext_jev_score"] = "nan"
            missing += 1
            updated_since_checkpoint += 1
        elif not is_missing_text(row.get("freetext_jev")):
            continue
        elif limit is not None and classified >= limit:
            continue
        else:
            category, confidence = classifier(str(text).strip())
            row["freetext_jev"] = category
            row["freetext_jev_score"] = str(confidence)
            classified += 1
            updated_since_checkpoint += 1

        if not dry_run and updated_since_checkpoint >= checkpoint_every:
            write_checkpoint()
            updated_since_checkpoint = 0

    if dry_run:
        return classified, missing

    write_checkpoint()
    return classified, missing


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (defaults to overwriting input).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="AI Gateway model id.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum non-empty reviews to classify.")
    parser.add_argument("--dry-run", action="store_true", help="Classify rows without writing the CSV.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for transient request failures.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    api_key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if not api_key:
        parser.error("AI_GATEWAY_API_KEY must be set")

    output_path = args.output or args.input
    classifier = lambda text: classify_with_jev(
        text, api_key=api_key, model=args.model, timeout=args.timeout, max_retries=args.max_retries
    )
    classified, missing = classify_csv(
        args.input, output_path, classifier=classifier, limit=args.limit, dry_run=args.dry_run
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
