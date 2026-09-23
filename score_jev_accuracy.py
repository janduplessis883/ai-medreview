#!/usr/bin/env python3
"""Manually score a balanced Jev sample with an interruption-safe Rich interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jev_accuracy_review import build_report, category_accuracy, load_or_create_session, record_decision


DEFAULT_INPUT_PATH = Path(__file__).parent / "ai_medreview/data/data_jev.csv"


def display_value(value: object) -> str:
    return "—" if pd.isna(value) or not str(value).strip() else str(value).strip()


def print_report(console: Console, data: pd.DataFrame, session: dict) -> dict:
    report = build_report(session)
    console.print()
    console.rule("Jev accuracy results")
    console.print(f"Reviewed: [bold]{report['reviewed']}[/] / {report['sample_size']}")
    console.print(f"Correct: [green]{report['correct']}[/]  Incorrect: [red]{report['incorrect']}[/]")
    if report["accuracy"] is not None:
        console.print(f"Accuracy: [bold]{report['accuracy']:.1%}[/]  Error rate: {report['error_rate']:.1%}")
        console.print(
            "95% confidence interval: "
            f"{report['accuracy_ci_95_low']:.1%}–{report['accuracy_ci_95_high']:.1%}"
        )

    by_category = category_accuracy(data, session)
    if by_category:
        table = Table(title="Manual accuracy by Jev prediction")
        table.add_column("Jev prediction")
        table.add_column("Reviewed", justify="right")
        table.add_column("Correct", justify="right")
        table.add_column("Accuracy", justify="right")
        for row in by_category:
            table.add_row(row["category"], str(row["reviewed"]), str(row["correct"]), f"{row['accuracy']:.1%}")
        console.print(table)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--session", type=Path, default=None, help="Progress JSON path.")
    parser.add_argument("--results", type=Path, default=None, help="Completed results JSON path.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    data = pd.read_csv(args.input, low_memory=False)
    session_path = args.session or args.input.with_name("jev_accuracy_review_progress.json")
    results_path = args.results or args.input.with_name("jev_accuracy_review_results.json")
    session = load_or_create_session(data, session_path, sample_size=args.sample_size, seed=args.seed)
    console = Console()

    console.print(
        Panel(
            "Answer [bold]Y[/] if Jev is correct, [bold]N[/] if it is incorrect, or [bold]Q[/] to stop.\n"
            "Your answer is saved immediately after each review. Examples are balanced by Jev category.",
            title="Jev accuracy review",
        )
    )
    for position, row_index in enumerate(session["sample_indices"], start=1):
        if str(row_index) in session["decisions"]:
            continue
        row = data.loc[row_index]
        console.rule(f"Review {position}/{len(session['sample_indices'])}")
        console.print(Panel(display_value(row["free_text"]), title="Patient feedback", border_style="cyan"))
        console.print(f"[green]🟢 Reference feedback label:[/] {display_value(row.get('feedback_labels'))}")
        console.print(f"[magenta]⭕ Jev prediction:[/] {display_value(row['freetext_jev'])}")

        while True:
            answer = console.input("[bold]Is the Jev prediction correct? [Y/N/Q]: [/]").strip().lower()
            if answer in {"y", "n"}:
                record_decision(session, session_path, row_index=row_index, is_correct=answer == "y")
                break
            if answer == "q":
                print_report(console, data, session)
                return 0
            console.print("[yellow]Please enter Y, N, or Q.[/]")

    report = print_report(console, data, session)
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(report, results_file, indent=2)
        results_file.write("\n")
    console.print(f"[green]Saved final results to {results_path}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
