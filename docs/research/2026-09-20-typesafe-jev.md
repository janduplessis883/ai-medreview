# TypeSafe Jev research

Research date: 2026-09-20. Sources below are TypeSafe's first-party documentation.

## What Jev is

Jev is TypeSafe's flagship and first “System One” model. It takes text state plus typed questions and returns decisions, probability distributions, and (for Choice and Score) confidence rather than generated prose. The supported primitives are Choice (one supplied option), Score (an ordered rubric), and Noul (a 0–1 answer to a proposition).

Sources: [Introduction](https://docs.typesafe.ai/introduction), [System One](https://docs.typesafe.ai/concepts/system-one), [Primitives](https://docs.typesafe.ai/primitives).

## Training and customization

TypeSafe describes Jev as a pretrained language model post-trained with RLCD: “reinforcement learning for calibrated decisions.” The stated objective is decisions with probability estimates whose accuracy corresponds to their reported probability across groups of predictions, rather than text generations preferred by people. This is presented as a third post-training path alongside RLHF and RLVR.

The published documentation does **not** disclose Jev's base model, architecture, parameter count, training corpus, RLCD reward design, training compute, or independent benchmark methodology. It says Jev is not fine-tuned or LoRA-adapted using customer data; the same weights serve every account. Users adapt it through state, instructions, criteria, atomic questions, and deterministic composition in code.

Sources: [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer), [Models — Customizing Jev](https://docs.typesafe.ai/models#customizing-jev).

## How it differs from a conventional LLM

| Jev / System One | Conventional generative LLM |
| --- | --- |
| Optimized for narrow, typed decisions and calibrated probabilities | Optimized to generate natural-language continuations and, commonly, preference-aligned replies |
| Returns constrained schema-like answers, probabilities, and confidence | Returns text that an application often has to parse or validate |
| Application code owns workflow, side effects, and composition | Agentic workflows may let the LLM select the next step or generate plans/tool calls |
| Designed for independent questions in parallel over one state | Multi-step prompting often uses serial turns and carries generated context forward |
| Does not generate explanations, code, or free-form text | Strong fit for explanations, writing, coding, and open-ended interaction |

Sources: [Introduction](https://docs.typesafe.ai/introduction), [System One](https://docs.typesafe.ai/concepts/system-one#how-it-differs-from-an-llm), [How to build with TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one).

## What is distinctive

1. **Calibrated uncertainty.** TypeSafe's stated goal is for a probability of 0.8 to correspond to roughly 80% correctness across a group of similar predictions. This is not a guarantee for an individual prediction.
2. **Output contract.** The model returns values restricted to the supplied question type/options, with full distributions for Choice and Score, avoiding free-text parsing.
3. **Parallel, independent questions.** Multiple questions evaluate against one state independently; TypeSafe says this avoids one result becoming hidden context for another and adds little latency.
4. **Code-first composition.** Jev is intended as a semantic-decision component inside conventional software, with deterministic rules, risk thresholds, and escalation kept in code.

Sources: [AI primer — RLCD](https://docs.typesafe.ai/introduction/machine-learning-primer#rlcd-and-calibrated-decisions), [Confidence](https://docs.typesafe.ai/confidence), [How to build with TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one#what-makes-system-one-composable).

## Advantages

- Useful when software needs a bounded semantic judgment: classification, detection, scoring, routing, ranking, retrieval, verification, and structured field extraction.
- Confidence/probability outputs enable a practical three-way policy: act automatically, review/gather more context, or escalate to a human or a reasoning model.
- Narrow questions can be parallelized and then combined deterministically, which suits latency-sensitive, large-scale workflow automation.
- The current model reference lists text/JSON input, 64k tokens per request, and a 32k state-plus-longest-question budget. TypeSafe's workflow guide says most queries complete in about 100 ms; treat this as a vendor performance claim, not an independent measurement.
- Customer requests/responses are not used to train Jev, according to TypeSafe's model reference.

Sources: [Example use cases](https://docs.typesafe.ai/concepts/use-case-map), [Confidence](https://docs.typesafe.ai/confidence#three-paths-for-using-confidence-in-your-code), [Models](https://docs.typesafe.ai/models), [How to build with TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one#what-makes-system-one-composable).

## Limitations and disadvantages

- **Not generative:** it is not trained to write prose, explanations, code, or reason through multi-step open-ended tasks. Use a generative/reasoning model for those.
- **Narrow-question requirement:** broad or ambiguous tasks must be decomposed; this moves system-design work into the application and may not suit exploratory tasks.
- **Text only:** no native image, audio, or video input.
- **English is strongest:** other languages are handled but are not documented as equally accurate; validate on target data.
- **Numeric and temporal weakness:** TypeSafe documents poor reliability for arithmetic, counting, numeric comparison, and date/time comparisons. Keep these operations in code.
- **Context sensitivity:** irrelevant material in a large state reduces accuracy (“context rot”); retrieve/filter first.
- **Adversarial sensitivity:** state is not treated as hostile by default, so prompt-injection-like or misleading input can shift an answer.
- **No guaranteed logical consistency across separately phrased questions:** do not assume negations or different primitive formulations obey arithmetic identities.
- **Operational dependency:** Jev is a hosted service, subject to token/request limits and version changes if an alias such as `jev-latest` is used. Pin `jev-1.13.0` if confidence thresholds were calibrated for that version.

Sources: [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [System One](https://docs.typesafe.ai/concepts/system-one), [Models](https://docs.typesafe.ai/models).

## Typical applications

- Support-ticket intent, queue routing, urgency, frustration, and escalation.
- Review/feedback categorization, such as assigning one known category to medical-practice feedback.
- Risk, fraud, spam, jailbreak, sensitive-data, and policy-violation detection.
- Verification of citations, AI outputs, tool calls, extraction results, and reasoning traces.
- Search, retrieval, ranking, semantic filtering, and feature extraction for downstream classical ML.
- Domain workflows in insurance claims, financial crime, legal/compliance, moderation, recruiting, e-commerce, advertising, and gaming.

Sources: [Example use cases](https://docs.typesafe.ai/concepts/use-case-map), [System One workflow example](https://docs.typesafe.ai/concepts/system-one#fast-judgments-inside-a-larger-workflow).

## Implication for the review-classification workflow

Your `Choice` setup is aligned with Jev's intended use: one bounded semantic decision from a predefined set of labels. The highest-leverage improvements are clear, contrastive category criteria; an explicit “noise/other” option; minimal relevant text in state; and empirical calibration against your own manually reviewed sample. Route low-confidence rows for review rather than treating the confidence as proof of correctness.

Sources: [Choice](https://docs.typesafe.ai/primitives/choice), [How to build with TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one#route-on-uncertainty), [Confidence](https://docs.typesafe.ai/confidence#thresholds-scale-with-risk).
