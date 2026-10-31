# Monthly report AI summary

## Intent and scope

Help practice managers decide what to do after reading the current month's patient feedback. Add an AI summary directly after People and sentiment and before Review list in the surgery-specific monthly PDF. Use only the selected surgery and calendar month; make no historical comparisons.

## Output

Keep the summary within 350 words, in plain British English, under three headings:

- What patients value: up to three specific strengths and why patients value them.
- Where patients experience difficulties: up to three concerns and their reported impact, distinguishing recurring themes from isolated experiences.
- Suggested practical responses: up to three practical suggestions, each explaining what to do and which finding supports it. Include strengths worth preserving and opportunities to improve. Decisions remain with the practice.

Proposed default: paraphrase findings and attach stable review references, also shown in the review list, so readers can inspect the evidence. Do not quote or name patients or individual staff in the summary. Include relevant mixed feedback; do not assume sentiment labels capture every aspect of a review.

## Approaches

Reuse the existing external LLM client and its configured freeLLMAPI endpoint, credentials and model, as requested. Generate the summary when Generate report is clicked. This fits the existing workflow and keeps each summary tied to its report. Persisting summaries would provide reuse but needs storage and invalidation; a separate provider would introduce unnecessary configuration. Neither is needed for this change.

## Prompt and evidence

Use the starting prompt agreed in the conversation, with the output headings above. Supply the scoped reviews with stable references and deterministic report statistics. Treat review content as untrusted evidence, never instructions. Require evidence references, no invented causes or clinical conclusions, and no unsupported counts or trend claims. Describe allegations as reported experiences. Flag potentially serious concerns for human review without presenting them as verified. State when evidence is sparse rather than padding the output. Reviews without usable text contribute only to the supplied statistics.

## Integration and failure behaviour

Keep report previews deterministic and free of LLM requests. Call the existing ai_medreview.llm_client infrastructure during PDF generation, using its configured model and external endpoint. Validate the response structure and review references before rendering; render content as escaped text in the existing HTML template.

On API failure, empty or invalid output, stop report generation and show a clear message inviting the user to click Generate report again. Do not offer a partial PDF. Clear any prior downloadable report before a new generation attempt so a failed retry cannot expose a stale report. If the month's reviews have no usable text, stop with an explanation that an evidence-based summary cannot be generated.

Reliability refinement: accept a complete JSON object wrapped in a Markdown code fence and deduplicate repeated valid citations. On validation failure, make at most one automatic correction request to the same model, retaining the original evidence and giving the specific rejection reason. Apply all evidence, structure and length checks again; stop if correction fails. Log only a fixed rejection code and attempt number, never model output or source feedback. Aim for 250-300 words to leave room below the 350-word limit. Transport/configuration failures still stop immediately.

Do not silently sample or truncate reviews: if input cannot be processed, fail clearly rather than present partial coverage as a complete monthly summary. No automatic provider fallback or new credentials.

## Validation

Verify surgery/month isolation, summary section order and PDF rendering, review references, escaped model content, no LLM call during preview, API and response failures, and removal of a stale download after a failed retry. Use mocked LLM responses for automated checks; do not send patient reviews to the service during tests.

## Builder choices

The builder can choose a structured response format and internal interfaces compatible with the existing client, and adapt the section typography to the current report. Proposed defaults requiring review are paraphrased evidence with review references and stopping when no review text is available.
