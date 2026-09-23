#!/usr/bin/env python3
"""Classify v3 Friends and Family Test reviews with Jev primary-topic criteria."""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Awaitable, Callable, Iterable

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r"urllib3 .*", category=Warning)
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy
from tqdm import tqdm


DEFAULT_MODEL = "jev-latest"
DEFAULT_INPUT_PATH = Path(__file__).parent / "ai_medreview/data/data_v3.csv"
CHECKPOINT_EVERY = 100

PRIMARY_TOPIC_INSTRUCTIONS = {
    "question": "What is the primary topic of this patient feedback?",
    "selection_rule": [
        "Choose exactly one topic.",
        "Classify the main subject, not whether it is positive or negative.",
        "A specific topic always beats Overall Service and Practice.",
        "If two independent topics are equally important and neither is primary, choose Multiple Topics / No Clear Primary Topic.",
        "Do not infer a topic that is not stated.",
    ],
    "boundary_rules": [
        "Waiting to obtain an appointment = Appointment Availability and Lead Times; waiting after arriving at the practice = Waiting Time at the Surgery.",
        "Difficulty calling the practice = Telephone Access; quality of a telephone or video appointment = Remote Consultation.",
        "Obtaining, renewing, or correcting a prescription = Prescriptions and Medication Management; medication benefit, side effects, or safety = Treatment Effectiveness and Safety.",
        "A named receptionist or front-desk interaction = Reception Service.",
    ],
}

PRIMARY_TOPIC_CRITERIA = {
    "Digital Booking and Online Systems": {
        "what": "Using online booking, the NHS App, patient portals, forms, digital check-in, or another digital practice system.",
        "not_for": "Not being able to get an appointment at all; use Appointment Availability and Lead Times.",
        "examples": ["The online form never works.", "Booking through the app was straightforward."],
    },
    "Appointment Availability and Lead Times": {
        "what": "Whether appointments are available, how long it takes to obtain one, or how far in advance appointments are offered.",
        "not_for": "Waiting after arriving at the surgery; use Waiting Time at the Surgery.",
        "examples": ["I waited three weeks for a GP appointment.", "There were plenty of same-day appointments."],
    },
    "Telephone Access": {
        "what": "Being able or unable to contact the practice by telephone, including busy lines, long hold times, call queues, or calls not being answered.",
        "not_for": "The quality of a telephone consultation; use Remote Consultation.",
        "examples": ["I could not get through on the phone all morning.", "The call queue was answered quickly."],
    },
    "Waiting Time at the Surgery": {
        "what": "Delays after the patient arrives for an in-person appointment, including waiting-room delays and clinicians running late.",
        "not_for": "The delay before an appointment could be booked; use Appointment Availability and Lead Times.",
        "examples": ["I waited 45 minutes after checking in.", "The doctor saw me exactly on time."],
    },
    "Reception Service": {
        "what": "Service, helpfulness, attitude, or competence of receptionists, front-desk staff, or check-in staff.",
        "not_for": "General staff kindness when reception is not mentioned; use Staff Kindness and Empathy.",
        "examples": ["The receptionist was dismissive.", "The front-desk team were extremely helpful."],
    },
    "Remote Consultation": {
        "what": "The suitability, quality, clarity, or technical delivery of a phone, video, or other remote clinical consultation.",
        "not_for": "Difficulty phoning the practice; use Telephone Access.",
        "examples": ["The telephone consultation felt rushed.", "The video appointment was very convenient."],
    },
    "Prescriptions and Medication Management": {
        "what": "Requesting, renewing, collecting, correcting, or administering prescriptions and repeat medication.",
        "not_for": "Whether a treatment worked or caused harm; use Treatment Effectiveness and Safety.",
        "examples": ["My repeat prescription was delayed.", "The pharmacy had the wrong dosage on my prescription."],
    },
    "Tests, Investigations and Results": {
        "what": "Blood tests, scans, samples, investigations, test booking, missing results, or communication of results.",
        "not_for": "A specialist referral or hospital appointment; use Referral, Specialist and Hospital Coordination.",
        "examples": ["I never received my blood-test result.", "The nurse explained my scan result clearly."],
    },
    "Referral, Specialist and Hospital Coordination": {
        "what": "Referrals to hospitals or specialists, shared care, referral status, or coordination with external services.",
        "not_for": "Routine test results handled directly by the practice; use Tests, Investigations and Results.",
        "examples": ["My referral was never sent.", "The practice chased the hospital appointment for me."],
    },
    "Follow-up and Continuity of Care": {
        "what": "Ongoing review, being contacted again, seeing the same clinician, care-plan continuity, or an unresolved issue not being followed up.",
        "not_for": "The quality of one isolated consultation; use the relevant consultation or treatment topic.",
        "examples": ["Nobody followed up after my appointment.", "It helped to see the same GP each time."],
    },
    "Consultation Communication and Listening": {
        "what": "Whether the clinician listened, explained, gave enough time, involved the patient, or communicated clearly during a consultation.",
        "not_for": "The medical correctness of diagnosis or treatment; use Clinical Assessment and Diagnosis or Treatment Effectiveness and Safety.",
        "examples": ["I felt rushed and not listened to.", "The doctor explained everything in a way I understood."],
    },
    "Clinical Assessment and Diagnosis": {
        "what": "The thoroughness, appropriateness, or correctness of examination, assessment, investigation, or diagnosis.",
        "not_for": "Communication style alone; use Consultation Communication and Listening.",
        "examples": ["The GP carried out a very thorough examination.", "My symptoms were dismissed without proper assessment."],
    },
    "Treatment Effectiveness and Safety": {
        "what": "Whether treatment, advice, medication, or a clinical plan helped, harmed, was appropriate, or addressed the condition.",
        "not_for": "Problems obtaining a prescription; use Prescriptions and Medication Management.",
        "examples": ["The treatment finally resolved the problem.", "The medication caused side effects and nobody reviewed it."],
    },
    "Staff Kindness and Empathy": {
        "what": "Compassion, reassurance, warmth, emotional support, or staff showing care for the patient.",
        "not_for": "Technical competence or expertise; use Staff Professionalism and Knowledge.",
        "examples": ["The nurse was so kind when I was anxious.", "Nobody showed any empathy."],
    },
    "Staff Professionalism and Knowledge": {
        "what": "Staff competence, knowledge, reliability, professional conduct, or confidence in their role.",
        "not_for": "Whether a clinical diagnosis or treatment was correct; use Clinical Assessment and Diagnosis or Treatment Effectiveness and Safety.",
        "examples": ["The clinician was knowledgeable and professional.", "The staff seemed poorly trained."],
    },
    "Respect, Dignity, Privacy and Inclusion": {
        "what": "Being treated with respect, dignity, fairness, confidentiality, cultural sensitivity, accessibility for identity needs, or freedom from discrimination.",
        "not_for": "A named receptionist interaction; use Reception Service.",
        "examples": ["I felt judged because of my disability.", "My privacy was protected throughout."],
    },
    "Vaccinations and Immunisations": {
        "what": "Vaccines, immunisation appointments, vaccine eligibility, advice, administration, or vaccine records.",
        "not_for": "General appointment booking with no vaccine-specific issue.",
        "examples": ["Getting my flu jab was quick and easy.", "I could not book a vaccination appointment."],
    },
    "Mental Health Support": {
        "what": "Access to, quality of, or support for mental-health concerns, counselling, wellbeing, anxiety, depression, or crisis support.",
        "not_for": "General empathy with no mental-health context; use Staff Kindness and Empathy.",
        "examples": ["I could not get support for my anxiety.", "The GP took my mental health seriously."],
    },
    "Patient Information and Education": {
        "what": "Written or verbal information that helps a patient understand a condition, self-care, next steps, risks, or how to use services.",
        "not_for": "Explanation during a specific consultation; use Consultation Communication and Listening.",
        "examples": ["I was given useful information about managing my condition.", "Nobody told me what to do next."],
    },
    "Facilities, Cleanliness and Environment": {
        "what": "Cleanliness, comfort, seating, toilets, signage, temperature, noise, or the physical condition of the practice.",
        "not_for": "Physical access barriers; use Physical Accessibility.",
        "examples": ["The waiting room was clean and comfortable.", "The toilets were dirty."],
    },
    "Physical Accessibility": {
        "what": "Access needs relating to mobility, disability, step-free entry, parking, hearing, vision, interpretation, or reasonable adjustments.",
        "not_for": "General convenience or a website issue; use the relevant digital or facilities topic.",
        "examples": ["There was no wheelchair access.", "The hearing loop made the visit much easier."],
    },
    "Website and Practice Information": {
        "what": "The practice website, published opening times, contact details, online information, or finding information about the practice.",
        "not_for": "A transactional booking-system failure; use Digital Booking and Online Systems.",
        "examples": ["The website has out-of-date opening hours.", "The website clearly explained how to register."],
    },
    "Feedback and Complaints Handling": {
        "what": "Making a complaint, receiving a response, acknowledgment, investigation, apology, or resolution of feedback.",
        "not_for": "The original service problem when the complaint process is not mentioned.",
        "examples": ["My complaint was ignored.", "The practice dealt with my feedback promptly."],
    },
    "Overall Service and Practice": {
        "what": "A broad overall judgment about the practice with no more specific topic stated.",
        "not_for": "Any review that names a concrete issue such as appointments, prescriptions, reception, tests, or staff behaviour.",
        "examples": ["An excellent practice all round.", "The service is generally poor."],
    },
    "Multiple Topics / No Clear Primary Topic": {
        "what": "Two or more distinct topics are discussed with similar importance and no topic is clearly the main subject.",
        "not_for": "A review with one clear primary issue plus background detail.",
        "examples": ["I could not get through by phone, then waited an hour, and my prescription was wrong."],
    },
    "Other / Unclassifiable / No Actionable Feedback": {
        "what": "Names only, blank-like content, unrelated text, survey answers without feedback, gibberish, or feedback with no identifiable service topic.",
        "not_for": "A short review that still identifies a real topic.",
        "examples": ["Dr Smith", "N/A"],
    },
}

CLASSIFICATION_COLUMNS = {
    "free_text": ("jev_freetext_cat", "jev_freetext_score"),
    "do_better": ("jev_dobetter_cat", "jev_dobetter_score"),
}


def is_missing_text(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def parse_choice_answer(response: object, answer_id: str) -> tuple[str, float]:
    answers = response.answers if hasattr(response, "answers") else response.get("answers")
    answer = answers.get(answer_id) if isinstance(answers, dict) else None
    if answer is None:
        raise RuntimeError(f"Malformed Jev response: missing answer '{answer_id}'")

    choice = answer.choice if hasattr(answer, "choice") else answer.get("choice")
    if not isinstance(choice, str) or choice not in PRIMARY_TOPIC_CRITERIA:
        raise RuntimeError(f"Malformed Jev response: invalid category {choice!r}")

    confidence = answer.confidence if hasattr(answer, "confidence") else answer.get("confidence")
    if not isinstance(confidence, (int, float)):
        probabilities = answer.probabilities if hasattr(answer, "probabilities") else answer.get("probabilities")
        if isinstance(probabilities, dict):
            confidence = probabilities.get(choice)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    return choice, max(0.0, min(1.0, float(confidence)))


async def classify_with_jev_async(
    text: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> tuple[str, float]:
    retry = RetryPolicy(max_retries=max_retries, timeout=timeout)
    async with AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout, retry=retry) as client:
        return await classify_with_client_async(client, text, timeout=timeout)


async def classify_with_client_async(
    client: AsyncTypeSafeClient,
    text: str,
    *,
    timeout: float = 30.0,
) -> tuple[str, float]:
    question = Choice(instructions=PRIMARY_TOPIC_INSTRUCTIONS, criteria=PRIMARY_TOPIC_CRITERIA)
    response = await client.system_one(
        state=f"Friends and Family Test review: {text}",
        questions={"primary_topic": question},
        timeout=timeout,
    )
    return parse_choice_answer(response, "primary_topic")


def classify_with_jev(
    text: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> tuple[str, float]:
    return asyncio.run(
        classify_with_jev_async(
            text, api_key=api_key, model=model, timeout=timeout, max_retries=max_retries
        )
    )


def classify_csv(
    input_path: Path,
    output_path: Path,
    *,
    classifier: Callable[[str], tuple[str, float]],
    limit: int | None = None,
    dry_run: bool = False,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> tuple[int, int]:
    """Classify both review fields, returning new classifications and missing source values."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")

    with input_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = set(CLASSIFICATION_COLUMNS)
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            missing_columns = ", ".join(sorted(required_columns - set(reader.fieldnames or [])))
            raise ValueError(f"CSV must contain {missing_columns}")
        fieldnames = list(reader.fieldnames)
        for category_column, score_column in CLASSIFICATION_COLUMNS.values():
            for column in (category_column, score_column):
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
    classified_since_checkpoint = 0
    for row in tqdm(rows, desc="Classifying Jev v3 reviews", unit="review"):
        for source_column, (category_column, score_column) in CLASSIFICATION_COLUMNS.items():
            text = row.get(source_column)
            if is_missing_text(text):
                missing += 1
                if row.get(category_column) != "nan" or row.get(score_column) != "nan":
                    row[category_column] = "nan"
                    row[score_column] = "nan"
            elif not is_missing_text(row.get(category_column)) and not is_missing_text(row.get(score_column)):
                continue
            elif limit is not None and classified >= limit:
                continue
            else:
                category, confidence = classifier(str(text).strip())
                row[category_column] = category
                row[score_column] = str(confidence)
                classified += 1
                classified_since_checkpoint += 1

            if not dry_run and classified_since_checkpoint >= checkpoint_every:
                write_checkpoint()
                classified_since_checkpoint = 0

    if dry_run:
        return classified, missing

    write_checkpoint()
    return classified, missing


async def classify_csv_async(
    input_path: Path,
    output_path: Path,
    *,
    classifier: Callable[[str], Awaitable[tuple[str, float]]],
    limit: int | None = None,
    dry_run: bool = False,
    checkpoint_every: int = CHECKPOINT_EVERY,
    concurrency: int = 8,
) -> tuple[int, int]:
    """Classify reviews concurrently while preserving CSV order and checkpoints."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    with input_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = set(CLASSIFICATION_COLUMNS)
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            missing_columns = ", ".join(sorted(required_columns - set(reader.fieldnames or [])))
            raise ValueError(f"CSV must contain {missing_columns}")
        fieldnames = list(reader.fieldnames)
        for category_column, score_column in CLASSIFICATION_COLUMNS.values():
            for column in (category_column, score_column):
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

    jobs: list[tuple[dict[str, str], str, str, str]] = []
    missing = 0
    for row in rows:
        for source_column, (category_column, score_column) in CLASSIFICATION_COLUMNS.items():
            text = row.get(source_column)
            if is_missing_text(text):
                missing += 1
                row[category_column] = "nan"
                row[score_column] = "nan"
            elif is_missing_text(row.get(category_column)) or is_missing_text(row.get(score_column)):
                if limit is None or len(jobs) < limit:
                    jobs.append((row, category_column, score_column, str(text).strip()))

    semaphore = asyncio.Semaphore(concurrency)

    async def classify_one(text: str) -> tuple[str, float]:
        async with semaphore:
            return await classifier(text)

    results: list[tuple[str, float] | None] = [None] * len(jobs)

    async def classify_indexed(index: int, text: str) -> tuple[int, tuple[str, float]]:
        return index, await classify_one(text)

    with tqdm(total=len(jobs), desc="Classifying Jev v3 reviews", unit="review") as progress:
        tasks = [asyncio.create_task(classify_indexed(index, text)) for index, (_, _, _, text) in enumerate(jobs)]
        for completed_task in asyncio.as_completed(tasks):
            index, result = await completed_task
            results[index] = result
            progress.update(1)

    classified_since_checkpoint = 0
    for (row, category_column, score_column, _), result in zip(jobs, results):
        if result is None:
            raise RuntimeError("A Jev classification task completed without a result")
        category, confidence = result
        row[category_column] = category
        row[score_column] = str(confidence)
        classified_since_checkpoint += 1
        if not dry_run and classified_since_checkpoint >= checkpoint_every:
            write_checkpoint()
            classified_since_checkpoint = 0

    if not dry_run:
        write_checkpoint()
    return len(results), missing


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (defaults to overwriting input).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="TypeSafe model name.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum non-empty values to classify.")
    parser.add_argument("--dry-run", action="store_true", help="Classify reviews without writing the CSV.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for transient request failures.")
    parser.add_argument("--concurrency", type=int, default=8, help="Maximum concurrent TypeSafe requests.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("TYPESAFE_API_KEY must be set")

    output_path = args.output or args.input
    async def run() -> None:
        retry = RetryPolicy(max_retries=args.max_retries, timeout=args.timeout)
        async with AsyncTypeSafeClient(
            api_key=api_key, model=args.model, timeout=args.timeout, retry=retry
        ) as client:
            async def classifier(text: str) -> tuple[str, float]:
                return await classify_with_client_async(client, text, timeout=args.timeout)

            await classify_csv_async(
                args.input,
                output_path,
                classifier=classifier,
                limit=args.limit,
                dry_run=args.dry_run,
                concurrency=args.concurrency,
            )

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
