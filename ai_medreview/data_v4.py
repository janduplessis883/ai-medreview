#!/usr/bin/env python3
"""data_v4.py — JEV-based Friends & Family Test review pipeline.

Differences from data_v2.py:
- Only processes reviews from 2026-01-01 onward (explicit date filter).
- `free_text` and `do_better` are combined into a single `review` column.
- Reviews with fewer than 8 words (after combining) are dropped entirely.
- Zero-shot classification (DeBERTa), HF sentiment, emotion classification and
  question answering are removed. Topic classification AND sentiment are done
  by Jev (typesafe.ai) in a single call per review, including actionability,
  urgency and safety/inclusion scores with confidences.
- Incremental processing uses the original Google Sheet row number and a
  separate cursor, including rows intentionally skipped for missing/short text.
- Output is written to data_v4.csv. data_v2.csv and the Streamlit app are
  untouched.

Run manually:  python -m ai_medreview.data_v4
Requires:      TYPESAFE_API_KEY environment variable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from colorama import init
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (
    normalize_whitespace,
    remove_eol_characters,
    remove_punct,
    replace_phone_numbers,
)
from nlpretext.social.preprocess import remove_hashtag, remove_mentions
from tqdm import tqdm
from transformers import pipeline

from ai_medreview.sheethelper import SheetHelper

tqdm.pandas()

from ai_medreview.automation.git_merge import do_git_merge
from ai_medreview.params import DATA_PATH
from ai_medreview.utils import time_it

os.environ["TOKENIZERS_PARALLELISM"] = "false"
init(autoreset=True)
warnings.filterwarnings("ignore")

from loguru import logger

logger.add("/tmp/ai_medreview_v4_debug.log", rotation="5000 KB")

# --- Configuration ---------------------------------------------------------------------------------

DATE_FLOOR = "2026-01-01"  # only process reviews submitted on/after this date
MIN_WORDS = 8  # combined reviews shorter than this are dropped
OUTPUT_CSV = f"{DATA_PATH}/data_v4.csv"
CHECKPOINT_CSV = f"{DATA_PATH}/data_v4_checkpoint.csv"
JEV_MODEL = "jev-latest"
JEV_TIMEOUT = 30.0
JEV_MAX_RETRIES = 2
JEV_CONCURRENCY = 8
JEV_CHECKPOINT_EVERY = 200  # write a checkpoint after this many JEV analyses

# If the primary-topic confidence is below this, secondary topics are recorded.
SECONDARY_TOPIC_CONFIDENCE_THRESHOLD = 0.5
# A secondary topic must reach this probability to be listed.
SECONDARY_TOPIC_MIN_PROB = 0.15
MAX_SECONDARY_TOPICS = 3

# --- Jev question definitions ----------------------------------------------------------------------

SENTIMENT_OPTIONS = {
    "Positive": "The overall feedback is favourable, appreciative, or endorsing.",
    "Negative": "The overall feedback is dissatisfied, critical, or reports a problem.",
    "Neutral or Mixed": "The feedback is factual, ambiguous, or contains balanced positive and negative content.",
}

STAFF_GROUP_OPTIONS = {
    "GP": "A doctor, GP, or physician.",
    "Nurse": "A practice nurse, nurse practitioner, or nursing team member.",
    "HCA": "A healthcare assistant or phlebotomist working in a support role.",
    "Receptionist": "Reception or front-desk staff.",
    "Admin Staff": "Back-office administrative, secretarial, practice-management staff or Practce Manager",
    "Clinical Pharmacist": "A practice-based or clinical pharmacist.",
    "No Specific Staff Group / Multiple": "The feedback does not focus on one staff group, or mentions several equally.",
}

PRIMARY_TOPIC_INSTRUCTIONS = {
    "question": "What is the primary topic of this patient feedback?",
    "selection_rule": [
        "Choose exactly one topic.",
        "Classify the main subject, not whether it is positive or negative.",
        "A specific topic always beats Overall Service and Practice.",
        "If several topics are present, choose the issue receiving the greatest emphasis, impact, or detail.",
        "Do not infer a topic that is not stated.",
    ],
    "boundary_rules": [
        "Waiting to obtain an appointment = Appointment Availability and Lead Times; waiting after arriving at the practice = Waiting Time at the Surgery.",
        "Difficulty calling the practice = Telephone Access; quality of a telephone or video appointment = Remote Consultation.",
        "Obtaining, renewing, or correcting a prescription = Prescriptions and Medication Management; medication benefit, side effects, or safety = Treatment Effectiveness and Safety.",
        "A named receptionist or front-desk interaction = Reception Service.",
        "Parking, transport, or travel to the practice = Car Parking and Transport, not Facilities.",
        "The experience of having blood taken = Phlebotomy; ordering tests, missing or delayed results = Tests, Investigations and Results.",
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
        "not_for": "A specialist referral or hospital appointment; use Referral, Specialist and Hospital Coordination. The experience of the blood draw itself; use Phlebotomy.",
        "examples": ["I never received my blood-test result.", "The nurse explained my scan result clearly."],
    },
    "Phlebotomy": {
        "what": "The experience of having blood taken: the phlebotomist's skill and manner, pain, bruising, children's blood tests, or the blood-test appointment itself.",
        "not_for": "Why a test was ordered, or missing or delayed results; use Tests, Investigations and Results.",
        "examples": ["The phlebotomist got my vein first time and was so gentle.", "My arm was badly bruised after the blood test."],
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
        "not_for": "Physical access barriers; use Physical Accessibility. Parking or transport; use Car Parking and Transport.",
        "examples": ["The waiting room was clean and comfortable.", "The toilets were dirty."],
    },
    "Physical Accessibility": {
        "what": "Access needs relating to mobility, disability, step-free entry, hearing, vision, interpretation, or reasonable adjustments.",
        "not_for": "General convenience or a website issue; use the relevant digital or facilities topic.",
        "examples": ["There was no wheelchair access.", "The hearing loop made the visit much easier."],
    },
    "Car Parking and Transport": {
        "what": "Car parking availability or cost, transport links, or travel to and from the practice.",
        "not_for": "The physical condition of the building itself; use Facilities, Cleanliness and Environment.",
        "examples": ["There is never anywhere to park.", "The bus stop right outside is handy."],
    },
    "Carer and Family Involvement": {
        "what": "How the practice involves, informs, or supports carers, family members, or friends of the patient.",
        "not_for": "The patient's own clinical care with no carer or family dimension.",
        "examples": ["They kept me updated about my mother's care.", "I was not allowed to accompany my disabled son."],
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
    "Other / Unclassifiable / No Actionable Feedback": {
        "what": "Names only, blank-like content, unrelated text, survey answers without feedback, gibberish, or feedback with no identifiable service topic.",
        "not_for": "A short review that still identifies a real topic.",
        "examples": ["Dr Smith", "N/A"],
    },
}


def build_jev_questions() -> dict[str, Any]:
    """Build the typed questions sent with each review to Jev."""
    from typesafe_sdk import Choice, Noul, Score

    return {
        "sentiment": Choice(
            instructions={
                "question": "What is the overall sentiment of this patient feedback?",
                "rules": [
                    "Classify sentiment, not the topic.",
                    "Use Neutral or Mixed when the feedback is factual, ambiguous, or balanced.",
                ],
            },
            criteria=SENTIMENT_OPTIONS,
        ),
        "primary_topic": Choice(
            instructions=PRIMARY_TOPIC_INSTRUCTIONS,
            criteria=PRIMARY_TOPIC_CRITERIA,
        ),
        "staff_group": Choice(
            instructions={
                "question": "Which staff group is this feedback mainly about?",
                "rules": [
                    "Choose the staff group receiving the greatest emphasis, praise, or criticism.",
                    "Base the choice only on roles explicitly mentioned or clearly implied.",
                    "Use No Specific Staff Group / Multiple when no single staff group is the focus.",
                ],
            },
            criteria=STAFF_GROUP_OPTIONS,
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
        "clinical_safety_concern": Noul(
            instructions=(
                "Does the review mention a clinical safety concern: harm, misdiagnosis, "
                "medication or treatment error, safeguarding risk, or symptoms being "
                "dangerously ignored?"
            ),
            criteria={
                "true": "The review explicitly raises a clinical safety or harm concern.",
                "false": "The review does not raise a clinical safety or harm concern.",
            },
        ),
        "dignity_or_inclusion_concern": Noul(
            instructions=(
                "Does the review mention a concern about dignity, respect, discrimination, "
                "privacy, confidentiality, or inclusion (including disability, language, "
                "or cultural needs not being met)?"
            ),
            criteria={
                "true": "The review explicitly raises a dignity, discrimination, privacy, or inclusion concern.",
                "false": "The review does not raise such a concern.",
            },
        ),
        "leaving_risk": Noul(
            instructions=(
                "Does the reviewer indicate they may stop using the practice, change practice, "
                "delay or avoid seeking care, or that they have lost trust in the practice?"
            ),
            criteria={
                "true": "The review explicitly states or clearly implies giving up on the practice or avoiding future care.",
                "false": "The review does not indicate leaving, avoiding care, or lost trust.",
            },
        ),
    }


# --- Google Sheet loading (retained from data_v2) ---------------------------------------------------


@time_it
def load_google_sheet() -> pd.DataFrame:
    sh = SheetHelper(
        sheet_url="https://docs.google.com/spreadsheets/d/1c-811fFJYT9ulCneTZ7Z8b4CK4feEDRheR0Zea5--d0/edit#gid=0",
        sheet_id=0,
    )
    columns = [
        "submission_id",
        "respondent-id",
        "time",
        "rating",
        "free_text",
        "do_better",
        "pcn",
        "surgery",
        "campaing_id",
        "logic",
        "campaign_rating",
        "campaign_freetext",
    ]
    values = sh.sheet_instance.get_all_values()
    rows = [
        (row[: len(columns)] + [""] * max(0, len(columns) - len(row)))
        for row in values[1:]
    ]
    data = pd.DataFrame(rows, columns=columns)
    # Google Sheets row 1 contains headers; data rows therefore start at row 2.
    data["source_sheet_row"] = np.arange(2, len(data) + 2)

    data["time"] = pd.to_datetime(
        data["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    # Preserve source row order and blank rows so sheet row numbers stay exact.
    data.reset_index(drop=True, inplace=True)
    return data


@time_it
def filter_date_floor(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only reviews submitted on/after DATE_FLOOR."""
    filtered = data[data["time"] >= pd.Timestamp(DATE_FLOOR)].copy()
    logger.info(
        f"📅 Date filter >= {DATE_FLOOR}: {len(filtered)} of {len(data)} rows kept."
    )
    return filtered


# --- Review identity & combining --------------------------------------------------------------------


def make_review_uid(row: pd.Series) -> str:
    """Stable fingerprint for dedup; survives sheet reordering and missing submission_ids."""
    raw = "|".join(
        [
            str(row.get("time", "")),
            str(row.get("surgery", "")),
            str(row.get("free_text", "")),
            str(row.get("do_better", "")),
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def combine_review(row: pd.Series) -> str:
    """Join free_text and do_better into a single review, whichever exist."""
    parts = []
    for column in ("free_text", "do_better"):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)


@time_it
def add_rating_score(data: pd.DataFrame) -> pd.DataFrame:
    rating_map = {
        "Very good": 5,
        "Good": 4,
        "Neither good nor poor": 3,
        "Poor": 2,
        "Very poor": 1,
    }
    data["rating_score"] = data["rating"].map(rating_map)
    return data


# --- Anonymisation (NER, retained from data_v2 — single pass capturing names) -----------------------

_ner_pipeline = None


def get_ner_pipeline():
    """Lazy-load the NER model so importing this module stays fast."""
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def anonymize_review(text: str) -> tuple[str, list[str] | None]:
    """Replace person names with [*PERSON*]; returns (anonymized_text, names_found)."""
    if not text or not isinstance(text, str):
        return text, None

    anonymized_text = text
    person_names: list[str] = []

    try:
        entities = get_ner_pipeline()(text)
        for entity in entities:
            if entity["entity_group"] == "PER":
                person_names.append(entity["word"])
                anonymized_text = anonymized_text.replace(entity["word"], "[*PERSON*]")
    except ValueError as e:
        print(f"Error processing text: {text}")
        raise e

    return anonymized_text, (person_names if person_names else None)


@time_it
def anonymize_reviews(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("🫥 Anonymize person names with Transformer NER - review")
    results = data["review"].progress_apply(anonymize_review)
    data["review"] = results.apply(lambda r: r[0])
    data["review_PER"] = results.apply(lambda r: r[1])
    return data


# --- Text preprocessing (retained from data_v2 — keeps phone-number masking) ------------------------


def text_preprocessing(text: str) -> str:
    preprocessor = Preprocessor()
    preprocessor.pipe(remove_mentions)
    preprocessor.pipe(remove_hashtag)
    preprocessor.pipe(remove_eol_characters)
    preprocessor.pipe(remove_punct)
    preprocessor.pipe(normalize_whitespace)
    preprocessor.pipe(
        replace_phone_numbers,
        args={"country_to_detect": ["GB", "FR"], "replace_with": "[*PHONE*]"},
    )
    return preprocessor.run(text)


@time_it
def preprocess_reviews(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("📗 Text Preprocessing with *NLPretext")
    data["review"] = data["review"].apply(
        lambda x: text_preprocessing(str(x)) if not pd.isna(x) else np.nan
    )
    return data


@time_it
def drop_short_reviews(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose combined review is shorter than MIN_WORDS words."""
    data["review_len"] = data["review"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )
    kept = data[data["review_len"] >= MIN_WORDS].copy()
    dropped = len(data) - len(kept)
    logger.info(
        f"🧽 Dropped {dropped} rows with combined review < {MIN_WORDS} words; "
        f"{len(kept)} rows remain."
    )
    return kept


# --- Jev analysis ------------------------------------------------------------------------------------

JEV_COLUMNS = [
    "jev_sentiment",
    "jev_sentiment_confidence",
    "jev_primary_topic",
    "jev_primary_topic_confidence",
    "jev_secondary_topics",
    "jev_topic_probabilities",
    "jev_staff_group",
    "jev_staff_group_confidence",
    "jev_sentiment_strength",
    "jev_sentiment_strength_confidence",
    "jev_actionability",
    "jev_actionability_confidence",
    "jev_urgency",
    "jev_urgency_confidence",
    "jev_clinical_safety_concern",
    "jev_clinical_safety_score",
    "jev_dignity_or_inclusion_concern",
    "jev_dignity_or_inclusion_score",
    "jev_leaving_risk",
    "jev_leaving_risk_score",
]


def parse_jev_response(response: Any) -> dict[str, Any]:
    """Convert a SystemOneResponse into flat output-column values."""
    sentiment_answer = response.choices["sentiment"]
    topic_answer = response.choices["primary_topic"]
    staff_answer = response.choices["staff_group"]
    strength_answer = response.scores["sentiment_strength"]
    actionability_answer = response.scores["actionability"]
    urgency_answer = response.scores["urgency"]
    clinical_safety_answer = response.nouls["clinical_safety_concern"]
    dignity_inclusion_answer = response.nouls["dignity_or_inclusion_concern"]
    leaving_risk_answer = response.nouls["leaving_risk"]

    primary_topic = topic_answer.choice
    primary_confidence = float(topic_answer.confidence or 0.0)

    probabilities = topic_answer.probabilities or {}
    probabilities = {str(k): float(v) for k, v in probabilities.items()}

    # Secondary topics: only when primary confidence is low.
    secondary_topics: list[str] = []
    if primary_confidence < SECONDARY_TOPIC_CONFIDENCE_THRESHOLD and probabilities:
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        secondary_topics = [
            topic
            for topic, prob in ranked
            if topic != primary_topic and prob >= SECONDARY_TOPIC_MIN_PROB
        ][:MAX_SECONDARY_TOPICS]

    clinical_safety_probability = float(clinical_safety_answer.noul)
    dignity_inclusion_probability = float(dignity_inclusion_answer.noul)
    leaving_risk_probability = float(leaving_risk_answer.noul)

    return {
        "jev_sentiment": sentiment_answer.choice,
        "jev_sentiment_confidence": float(sentiment_answer.confidence or 0.0),
        "jev_primary_topic": primary_topic,
        "jev_primary_topic_confidence": primary_confidence,
        "jev_secondary_topics": " | ".join(secondary_topics),
        "jev_topic_probabilities": json.dumps(probabilities, sort_keys=True),
        "jev_staff_group": staff_answer.choice,
        "jev_staff_group_confidence": float(staff_answer.confidence or 0.0),
        "jev_sentiment_strength": int(strength_answer.score),
        "jev_sentiment_strength_confidence": float(strength_answer.confidence or 0.0),
        "jev_actionability": int(actionability_answer.score),
        "jev_actionability_confidence": float(actionability_answer.confidence or 0.0),
        "jev_urgency": int(urgency_answer.score),
        "jev_urgency_confidence": float(urgency_answer.confidence or 0.0),
        "jev_clinical_safety_concern": bool(clinical_safety_probability >= 0.5),
        "jev_clinical_safety_score": clinical_safety_probability,
        "jev_dignity_or_inclusion_concern": bool(dignity_inclusion_probability >= 0.5),
        "jev_dignity_or_inclusion_score": dignity_inclusion_probability,
        "jev_leaving_risk": bool(leaving_risk_probability >= 0.5),
        "jev_leaving_risk_score": leaving_risk_probability,
    }


async def jev_analyze_reviews(
    data: pd.DataFrame,
    *,
    api_key: str,
    checkpoint_path: str = CHECKPOINT_CSV,
) -> pd.DataFrame:
    """Run Jev over the `review` column with concurrency and checkpointing.

    Writes progress to checkpoint_path so an interrupted run can be resumed by
    simply re-running the script.
    """
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    data = data.copy()
    for column in JEV_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    # Resume from a previous checkpoint if one exists.
    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_csv(checkpoint_path)
        if "review_uid" in checkpoint.columns:
            done = (
                checkpoint.dropna(subset=["jev_primary_topic"])
                .drop_duplicates(subset=["review_uid"], keep="last")
                .set_index("review_uid")
            )
            for column in JEV_COLUMNS:
                data[column] = data.apply(
                    lambda row, c=column: done[c].get(row["review_uid"], row[c])
                    if row["review_uid"] in done.index
                    else row[c],
                    axis=1,
                )
            logger.info(f"♻️ Resumed {len(done)} JEV results from checkpoint.")

    pending = data[data["jev_primary_topic"].isna()]
    if pending.empty:
        logger.info("✅ JEV analysis already complete for all rows.")
        return data

    questions = build_jev_questions()
    retry = RetryPolicy(max_retries=JEV_MAX_RETRIES, timeout=JEV_TIMEOUT)
    semaphore = asyncio.Semaphore(JEV_CONCURRENCY)

    async with AsyncTypeSafeClient(
        api_key=api_key, model=JEV_MODEL, timeout=JEV_TIMEOUT, retry=retry
    ) as client:

        async def analyze_one(text: str) -> dict[str, Any]:
            async with semaphore:
                response = await client.system_one(
                    state={"review": text},
                    questions=questions,
                    timeout=JEV_TIMEOUT,
                )
                return parse_jev_response(response)

        async def analyze_indexed(idx: Any, text: str) -> tuple[Any, dict[str, Any]]:
            return idx, await analyze_one(text)

        pending_items = list(
            zip(pending.index, pending["review"].astype(str))
        )
        results: dict[Any, dict[str, Any]] = {}
        completed = 0

        tasks = [
            asyncio.ensure_future(analyze_indexed(idx, text))
            for idx, text in pending_items
        ]
        with tqdm(
            total=len(tasks), desc="JEV analyzing reviews", unit="review"
        ) as progress:
            for future in asyncio.as_completed(tasks):
                idx, result = await future
                results[idx] = result
                completed += 1
                progress.update(1)

                if completed % JEV_CHECKPOINT_EVERY == 0:
                    for row_idx, res in results.items():
                        for column, value in res.items():
                            data.at[row_idx, column] = value
                    data.to_csv(checkpoint_path, encoding="utf-8", index=False)
                    logger.info(f"💾 JEV checkpoint written ({completed} done).")

    for row_idx, res in results.items():
        for column, value in res.items():
            data.at[row_idx, column] = value

    return data


# --- Local data & final save ------------------------------------------------------------------------


@time_it
def load_local_data(output_path: str = OUTPUT_CSV) -> pd.DataFrame:
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        df["time"] = pd.to_datetime(df["time"], dayfirst=False)
        return df
    return pd.DataFrame()


@time_it
def concat_save_final_df(
    processed_df: pd.DataFrame,
    new_df: pd.DataFrame,
    output_path: str = OUTPUT_CSV,
    checkpoint_path: str = CHECKPOINT_CSV,
) -> None:
    logger.info(f"💾 Concat Dataframes to {output_path}")
    combined_data = pd.concat([processed_df, new_df], ignore_index=True)
    if "source_sheet_row" in combined_data.columns:
        has_source_row = combined_data["source_sheet_row"].notna()
        legacy_rows = combined_data[~has_source_row]
        indexed_rows = combined_data[has_source_row].drop_duplicates(
            subset=["source_sheet_row"], keep="first"
        )
        combined_data = pd.concat([legacy_rows, indexed_rows], ignore_index=True)
    combined_data.sort_values(by="time", inplace=True, ascending=True)
    combined_data.to_csv(output_path, encoding="utf-8", index=False)
    print(f"💾 Output saved to: {output_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def load_progress(path: str) -> int | None:
    """Return the last fully handled Google Sheet row, if a cursor exists."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as progress_file:
        return int(json.load(progress_file)["last_completed_sheet_row"])


def save_progress(path: str, last_completed_sheet_row: int) -> None:
    """Atomically persist the Google Sheet row cursor."""
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as progress_file:
        json.dump(
            {"last_completed_sheet_row": int(last_completed_sheet_row)},
            progress_file,
        )
        progress_file.write("\n")
    os.replace(temporary_path, path)


# --- Main --------------------------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "source_sheet_row",
    "submission_id",
    "respondent-id",
    "time",
    "rating",
    "rating_score",
    "surgery",
    "pcn",
    "review",
    "review_len",
    "review_PER",
    "review_uid",
    *JEV_COLUMNS,
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only N randomly sampled reviews (end-to-end test).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (defaults to data_v4.csv, or data_v4_sample.csv with --sample).",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip the automatic git commit/push at the end.",
    )
    args = parser.parse_args()

    # Safe defaults for sample runs: separate output file, no git push.
    output_path = args.output or (
        f"{DATA_PATH}/data_v4_sample.csv" if args.sample else OUTPUT_CSV
    )
    checkpoint_path = output_path.replace(".csv", "_checkpoint.csv")
    push_to_git = not args.no_git and not args.sample

    logger.info("▶️ AI Medreview FFT - MAKE DATA v4 (JEV) - Started")
    if args.sample:
        logger.info(
            f"🧪 SAMPLE MODE: {args.sample} reviews -> {output_path} (no git push)"
        )

    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TYPESAFE_API_KEY environment variable is not set. "
            "Export it before running this script."
        )

    # Load the sheet with original row numbers, including blank rows.
    raw_data = load_google_sheet()
    logger.info("Google Sheet data loaded")

    processed_data = load_local_data(output_path)
    progress_path = output_path.replace(".csv", "_progress.json")
    last_completed_row = None if args.sample else load_progress(progress_path)

    if not args.sample and last_completed_row is None:
        if not processed_data.empty:
            # The legacy output has no source row mapping. Treat the current
            # sheet as the migration baseline, so historical rows are not
            # re-analyzed; subsequent appended rows are tracked exactly.
            last_completed_row = (
                int(raw_data["source_sheet_row"].max()) if len(raw_data) else 1
            )
            logger.warning(
                "No row cursor found for the legacy output; initializing at "
                f"the current last sheet row ({last_completed_row})."
            )
        else:
            last_completed_row = 1  # header row; first Google Sheet data row is 2

    if args.sample:
        source_rows = raw_data.copy()
    else:
        processed_sheet_rows = set()
        if "source_sheet_row" in processed_data.columns:
            processed_sheet_rows = set(
                pd.to_numeric(processed_data["source_sheet_row"], errors="coerce")
                .dropna()
                .astype(int)
            )
        source_rows = raw_data[
            (raw_data["source_sheet_row"] > last_completed_row)
            & ~raw_data["source_sheet_row"].isin(processed_sheet_rows)
        ].copy()

    data = filter_date_floor(source_rows)
    data["review_uid"] = data.apply(make_review_uid, axis=1)
    logger.info(
        f"🆕 New sheet rows to process: {len(source_rows)}; "
        f"eligible by date: {data.shape[0]}"
    )

    if args.sample:
        # Oversample so the 8-word filter still leaves ~N reviews.
        data = data.sample(
            n=min(args.sample * 5, len(data)), random_state=42
        ).copy()

    batch_last_row = (
        int(source_rows["source_sheet_row"].max()) if len(source_rows) else None
    )
    progress_saved = False

    if data.shape[0] != 0:
        # Combine free_text + do_better into a single review
        data["review"] = data.apply(combine_review, axis=1)
        data = data[data["review"].str.strip() != ""].copy()

        if not data.empty:
            data = add_rating_score(data)
            data = anonymize_reviews(data)
            data = preprocess_reviews(data)
            data = drop_short_reviews(data)

        if args.sample:
            data = data.head(args.sample).copy()

        if data.shape[0] != 0:
            # JEV: sentiment + topic + actionability/urgency/safety in one call
            data = asyncio.run(
                jev_analyze_reviews(
                    data, api_key=api_key, checkpoint_path=checkpoint_path
                )
            )
            logger.info("Data pre-processing completed")

            data = data[[c for c in OUTPUT_COLUMNS if c in data.columns]]
            concat_save_final_df(
                processed_data, data, output_path, checkpoint_path
            )
            if not args.sample and batch_last_row is not None:
                save_progress(progress_path, batch_last_row)
                progress_saved = True

            if push_to_git:
                do_git_merge()  # Push everything to GitHub (master)
                logger.info("👍 Pushed to GitHub - Master Branch")
            logger.info("🎉 Successful Run completed")
        else:
            logger.info("No analyzable reviews in the new sheet rows; rows are skipped.")
    else:
        logger.info("No eligible new sheet rows to analyze.")

    if not args.sample and not progress_saved:
        # Persist only after analysis output has been saved, or after all rows
        # in the batch were deliberately skipped (date, empty, or short text).
        cursor_to_save = batch_last_row or last_completed_row
        save_progress(progress_path, cursor_to_save)
        logger.info(f"Saved sheet row cursor at {cursor_to_save}.")
