"""Persistent sampling and metrics for manual Jev accuracy reviews."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"free_text", "freetext_jev"}


def _has_value(value: object) -> bool:
    return not pd.isna(value) and str(value).strip().lower() not in {"", "nan", "none", "null"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary_file:
        json.dump(payload, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def balanced_sample_indices(
    data: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> list[int]:
    """Return a reproducible, as-even-as-possible sample by Jev category."""
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {columns}")

    by_category: dict[str, list[int]] = {}
    for index, row in data.iterrows():
        if _has_value(row["free_text"]) and _has_value(row["freetext_jev"]):
            by_category.setdefault(str(row["freetext_jev"]), []).append(int(index))

    available = sum(len(indices) for indices in by_category.values())
    if available < sample_size:
        raise ValueError(f"Only {available} rows have both review text and a Jev prediction.")

    randomizer = random.Random(seed)
    for indices in by_category.values():
        randomizer.shuffle(indices)

    selected: list[int] = []
    categories = sorted(by_category)
    while len(selected) < sample_size:
        made_selection = False
        for category in categories:
            if by_category[category] and len(selected) < sample_size:
                selected.append(by_category[category].pop())
                made_selection = True
        if not made_selection:
            break
    randomizer.shuffle(selected)
    return selected


def load_or_create_session(
    data: pd.DataFrame,
    session_path: Path,
    *,
    sample_size: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Load a saved review session or create a reproducible random sample."""
    if session_path.exists():
        with session_path.open(encoding="utf-8") as session_file:
            return json.load(session_file)

    session = {
        "sample_indices": balanced_sample_indices(data, sample_size=sample_size, seed=seed),
        "decisions": {},
        "sample_size": sample_size,
        "seed": seed,
    }
    _write_json(session_path, session)
    return session


def record_decision(
    session: dict[str, Any],
    session_path: Path,
    *,
    row_index: int,
    is_correct: bool,
) -> None:
    """Record one answer and persist it immediately for interruption-safe resume."""
    session["decisions"][str(row_index)] = is_correct
    _write_json(session_path, session)


def build_report(session: dict[str, Any]) -> dict[str, int | float | None]:
    """Calculate binary classification metrics from manual Y/N decisions."""
    decisions = session["decisions"]
    reviewed = len(decisions)
    correct = sum(bool(answer) for answer in decisions.values())
    incorrect = reviewed - correct
    total = len(session["sample_indices"])
    accuracy = correct / reviewed if reviewed else None

    if accuracy is None:
        confidence_low = None
        confidence_high = None
    else:
        z = 1.96
        denominator = 1 + z**2 / reviewed
        centre = (accuracy + z**2 / (2 * reviewed)) / denominator
        margin = z * math.sqrt((accuracy * (1 - accuracy) + z**2 / (4 * reviewed)) / reviewed) / denominator
        confidence_low = max(0.0, centre - margin)
        confidence_high = min(1.0, centre + margin)

    return {
        "sample_size": total,
        "reviewed": reviewed,
        "pending": total - reviewed,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "error_rate": None if accuracy is None else 1 - accuracy,
        "accuracy_ci_95_low": confidence_low,
        "accuracy_ci_95_high": confidence_high,
    }


def category_accuracy(data: pd.DataFrame, session: dict[str, Any]) -> list[dict[str, int | float | str]]:
    """Summarise manual correctness rates for each Jev-predicted category."""
    results: dict[str, list[bool]] = {}
    for row_index_as_text, is_correct in session["decisions"].items():
        predicted = str(data.loc[int(row_index_as_text), "freetext_jev"])
        results.setdefault(predicted, []).append(bool(is_correct))

    return [
        {
            "category": category,
            "reviewed": len(decisions),
            "correct": sum(decisions),
            "accuracy": sum(decisions) / len(decisions),
        }
        for category, decisions in sorted(results.items())
    ]
