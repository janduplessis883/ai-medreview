# Turning Friends and Family Test comments into decisions with TypeSafe Jev

Friends and Family Test data gives GP practices a steady stream of patient comments. Star ratings make the first chart easy. The harder questions sit inside the free text:

- Is the patient talking about appointment access, reception, prescriptions, or clinical care?
- Does the comment contain a safety or dignity concern?
- Which reviews need a person to read them soon?

Those questions need language understanding, but they do not need a chatbot. They need a small set of bounded judgments that application code can store, aggregate, filter, and audit.

That is the role Jev plays in AI MedReview. Jev is TypeSafe's System One model. You send it a piece of state and one or more typed questions. It returns choices, scores, and probabilities instead of generated prose. The application keeps control of the workflow.

This article walks through the implementation in [`ai_medreview/data_v4.py`](../ai_medreview/data_v4.py) and the dashboard in [`streamlit_app.py`](../streamlit_app.py). The examples use the same design as the production pipeline, with the larger taxonomies shortened where needed.

## The shape of the pipeline

AI MedReview separates inference from presentation:

1. Read new FFT rows from a Google Sheet.
2. Combine the `free_text` and `do_better` fields.
3. Remove empty and short comments, anonymise the text, and prepare a review identifier.
4. Send each review to Jev with nine independent questions.
5. Flatten the typed answers into columns and save them in `data_v4.csv`.
6. Let Streamlit filter, chart, and route the stored results.

The Streamlit app does not call Jev during a page view. That decision keeps the dashboard responsive and makes every chart reproducible from the saved CSV. It also lets the data job checkpoint long batches and resume after an interruption.

## Why Jev fits FFT analysis

TypeSafe exposes three question primitives:

| Primitive | Use in AI MedReview | Returned value |
| --- | --- | --- |
| `Choice` | Sentiment, primary topic, staff group | Selected option, option probabilities, confidence |
| `Score` | Sentiment strength, actionability, urgency | Position on an ordered rubric, level probabilities, confidence |
| `Noul` | Clinical safety, dignity or inclusion, leaving risk | Probability that the answer is yes |

The distinction matters. A topic is one option from a closed taxonomy, so it is a `Choice`. Urgency sits on an ordered scale, so it is a `Score`. A clinical safety concern may be present or absent, so it is a `Noul`.

Jev can answer all nine questions against the same review in one call. Each answer remains independent, so the predicted sentiment does not become hidden context for the urgency judgment.

## Install the SDK and create a client

The Python SDK requires Python 3.10 or later. Install it and keep the API key in an environment variable:

```bash
pip install typesafe-sdk
export TYPESAFE_API_KEY="your-api-key"
```

The data pipeline uses the asynchronous client, an explicit model alias, a 30-second timeout, and a retry policy:

```python
import os

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

api_key = os.environ["TYPESAFE_API_KEY"]
retry = RetryPolicy(max_retries=2, timeout=30.0)

async with AsyncTypeSafeClient(
    api_key=api_key,
    model="jev-latest",
    timeout=30.0,
    retry=retry,
) as client:
    # Call client.system_one(...) here.
    ...
```

The model alias lets the project receive current Jev improvements. Pin a model version when you need stable behaviour around calibrated thresholds, then re-evaluate those thresholds before changing versions.

## Define narrow questions

The full implementation builds nine questions in `build_jev_questions()`. A smaller version shows the pattern:

```python
from typesafe_sdk import Choice, Noul, Score

questions = {
    "sentiment": Choice(
        instructions={
            "question": "What is the overall sentiment of this patient feedback?",
            "rules": [
                "Classify sentiment, not the topic.",
                "Use Neutral or Mixed when the feedback is factual, ambiguous, or balanced.",
            ],
        },
        criteria={
            "Positive": "Favourable, appreciative, or endorsing feedback.",
            "Negative": "Dissatisfied or critical feedback, or a reported problem.",
            "Neutral or Mixed": "Factual, ambiguous, or balanced feedback.",
        },
    ),
    "primary_topic": Choice(
        instructions={
            "question": "What is the primary topic of this patient feedback?",
            "selection_rule": [
                "Choose exactly one topic.",
                "Classify the main subject, not its sentiment.",
                "Prefer a specific topic over Overall Service and Practice.",
            ],
        },
        criteria={
            "Appointment Availability and Lead Times": {
                "what": "Getting an appointment or the delay before one is available.",
                "not_for": "Waiting after arrival at the surgery.",
            },
            "Waiting Time at the Surgery": {
                "what": "A delay after the patient arrives for an appointment.",
                "not_for": "The delay before an appointment can be booked.",
            },
            "Other / Unclassifiable / No Actionable Feedback": {
                "what": "No identifiable service topic or unrelated content."
            },
        },
    ),
    "urgency": Score(
        instructions="How urgently should a human review this feedback?",
        criteria=[
            "No urgency; routine feedback.",
            "Worth monitoring but not time-sensitive.",
            "Should be reviewed soon.",
            "Prompt attention due to possible harm, discrimination, or unresolved risk.",
        ],
    ),
    "clinical_safety_concern": Noul(
        instructions=(
            "Does the review mention harm, misdiagnosis, a medication or treatment "
            "error, a safeguarding risk, or symptoms being dangerously ignored?"
        ),
        criteria={
            "true": "The review raises a clinical safety or harm concern.",
            "false": "The review does not raise a clinical safety or harm concern.",
        },
    ),
}
```

The production topic `Choice` contains 28 options. Each option says what belongs there, what does not, and often gives examples. Boundary rules handle easy-to-confuse pairs. A wait for an appointment belongs under appointment availability; a wait after check-in belongs under waiting time at the surgery. A prescription renewal problem differs from a medication safety problem.

Those contrasts do more work than a long generic prompt. They give Jev a closed decision surface that matches the reports the practice wants to produce.

## Call Jev once per review

Pass the review as structured state and send the complete question map:

```python
review = (
    "I waited three weeks for an appointment. When I was seen, "
    "the GP listened carefully and explained the treatment."
)

response = await client.system_one(
    state={"review": review},
    questions=questions,
    timeout=30.0,
)
```

The field name in `state` gives the text a stable identity. It also leaves room to add relevant context later, such as the survey question or service channel. Keep identifiers and unrelated record data out of the state unless a judgment needs them.

## Read the typed response

The SDK groups answers by primitive. AI MedReview flattens them into ordinary values that pandas and Streamlit can use:

```python
sentiment = response.choices["sentiment"]
topic = response.choices["primary_topic"]
urgency = response.scores["urgency"]
safety = response.nouls["clinical_safety_concern"]

row = {
    "jev_sentiment": sentiment.choice,
    "jev_sentiment_confidence": float(sentiment.confidence or 0.0),
    "jev_primary_topic": topic.choice,
    "jev_primary_topic_confidence": float(topic.confidence or 0.0),
    "jev_topic_probabilities": topic.probabilities,
    "jev_urgency": urgency.score,
    "jev_urgency_confidence": float(urgency.confidence or 0.0),
    "jev_clinical_safety_score": float(safety.noul),
    "jev_clinical_safety_concern": safety.noul >= 0.5,
}
```

`Choice` and `Score` confidence describe how concentrated the returned probability distribution is. A low topic confidence means that several topic options competed closely. It does not mean that the chosen label has a known error rate for your dataset.

A `Noul` returns the probability of yes and has no separate confidence field. The `0.5` safety threshold above is an application policy. A healthcare workflow should set that threshold from labelled FFT examples and the cost of missing a concern, not from the convenience of a round number.

TypeSafe scores can fall between rubric levels. `data_v4.py` converts each score to `int` before writing the CSV, which gives the current dashboard discrete, zero-indexed bands. Keep the raw floating-point score if you want to preserve the model's position between levels. You can derive display bands as a separate column.

## Use uncertainty instead of hiding it

The pipeline stores the full topic probability distribution. When primary-topic confidence falls below `0.5`, it records up to three secondary topics whose probability reaches `0.15`:

```python
primary_topic = topic.choice
primary_confidence = float(topic.confidence or 0.0)
probabilities = {
    str(name): float(probability)
    for name, probability in (topic.probabilities or {}).items()
}

secondary_topics = []
if primary_confidence < 0.5:
    ranked = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    secondary_topics = [
        name
        for name, probability in ranked
        if name != primary_topic and probability >= 0.15
    ][:3]
```

Streamlit exposes those rows in a low-confidence table. A reviewer can see the selected topic, its confidence, and the plausible alternatives. That turns uncertainty into a queue for taxonomy improvement or human review.

The thresholds in this repository are working policies, not universal TypeSafe defaults. Evaluate them on comments labelled by people who understand local primary-care operations.

## Process a batch without losing progress

FFT data arrives in batches, so the pipeline limits concurrent requests with an `asyncio.Semaphore`:

```python
import asyncio

semaphore = asyncio.Semaphore(8)

async def analyse_one(text: str) -> dict:
    async with semaphore:
        response = await client.system_one(
            state={"review": text},
            questions=questions,
            timeout=30.0,
        )
        return parse_jev_response(response)

tasks = [
    asyncio.create_task(analyse_one(text))
    for text in pending_reviews
]

results = []
for task in asyncio.as_completed(tasks):
    results.append(await task)
```

`data_v4.py` retains each DataFrame index so completed answers return to the correct row. It writes a checkpoint every 200 analyses, then uses a stable `review_uid` to restore completed work on the next run. A separate Google Sheet row cursor prevents skipped or historical rows from reappearing as new work.

This operational code matters as much as the model call. A classifier that cannot resume safely will eventually produce duplicate calls, partial exports, or confusing gaps.

## Turn Jev outputs into a review queue

The Streamlit app loads the saved columns and defines an alert with plain pandas logic:

```python
df["is_alert"] = (
    (df["jev_urgency"] >= 2)
    | df["jev_clinical_safety_concern"]
    | df["jev_dignity_or_inclusion_concern"]
    | df["jev_leaving_risk"]
)
```

The rule combines one graded judgment with three specific risk flags. Staff can filter to alerts, inspect the original comment, and see the topic, staff group, urgency, actionability, and underlying probabilities.

Keep this alert logic in code. Practice policy can change without re-running Jev, and the rule remains visible to reviewers. A practice might lower the safety threshold, add a confidence condition, or send urgency level 3 to a separate queue.

The rest of the dashboard uses the same saved fields for:

- sentiment and review-volume trends;
- topic and staff-group breakdowns;
- topic-by-sentiment tables;
- low-confidence classifications;
- review search and row-level JSON inspection.

Jev supplies semantic features. pandas and Streamlit handle counting, filtering, dates, thresholds, and charts.

## Production checks for patient feedback

FFT analysis affects how staff prioritise patient voices. Treat the model output as decision support.

Start with a labelled evaluation set that includes praise, mixed comments, short comments, multi-topic reviews, safety concerns, access barriers, and local terminology. Measure each question separately. Topic accuracy does not establish safety-flag recall.

Review false negatives for the risk Nouls and low-confidence topic choices. Record the model version with every evaluation. Re-run the evaluation when you change criteria, thresholds, preprocessing, or the model version.

Protect patient data before inference. The AI MedReview pipeline anonymises reviews before the Jev stage. Keep API keys on the server, restrict access to raw comments, and agree retention and governance with the organisation responsible for the data.

Most important, keep a person in the path for consequential action. A high urgency score can put a comment at the top of a queue. It should not make a clinical or disciplinary decision by itself.

## A useful division of labour

The implementation works because each component has a small job:

- Jev judges meaning against explicit options and rubrics.
- Python applies thresholds, concurrency, retries, checkpoints, and alert policy.
- pandas stores reusable features and computes aggregates.
- Streamlit gives staff a review queue and an audit surface.

You can start with three questions: topic, urgency, and clinical safety. Add another question only when the answer will change a report, filter, or workflow. That constraint keeps the analysis legible and gives each new model judgment a reason to exist.

## References

- [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python)
- [TypeSafe quick start](https://docs.typesafe.ai/introduction/quickstart)
- [Choice](https://docs.typesafe.ai/primitives/choice)
- [Score](https://docs.typesafe.ai/primitives/score)
- [Noul](https://docs.typesafe.ai/primitives/noul)
- [Confidence](https://docs.typesafe.ai/confidence)
- [AI MedReview Jev pipeline](../ai_medreview/data_v4.py)
- [AI MedReview Streamlit dashboard](../streamlit_app.py)
