from __future__ import annotations

import json
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from ai_medreview import llm_client
from ai_medreview.monthly_report import build_report_context


def report_context():
    return build_report_context(pd.DataFrame([
        {"time": "2026-10-20", "surgery": "Oak", "review": "Kind reception team.", "rating_score": 5},
        {"time": "2026-10-19", "surgery": "Oak", "review": "Unclear appointment booking.", "rating_score": 2},
        {"time": "2026-09-19", "surgery": "Oak", "review": "EXCLUDE PREVIOUS MONTH", "rating_score": 1},
        {"time": "2026-10-19", "surgery": "Other", "review": "EXCLUDE OTHER SURGERY", "rating_score": 1},
    ]), "Oak", "2026-10")


def valid_summary():
    return {
        "evidence_note": "Based on two written reviews; individual experiences may differ.",
        "strengths": [{"text": "A patient valued the kind reception team.", "references": ["R1"]}],
        "concerns": [{"text": "One patient reported unclear booking.", "references": ["R2"]}],
        "responses": [{"text": "Consider clarifying booking instructions while preserving courteous reception support.", "references": ["R1", "R2"]}],
    }


def test_summary_sends_only_scoped_evidence_to_existing_client(monkeypatch):
    from ai_medreview.monthly_summary import generate_ai_summary

    client = Mock(return_value=json.dumps(valid_summary()))
    monkeypatch.setattr(llm_client, "call_llm", client)
    context = report_context()
    result = generate_ai_summary(context)
    assert result == valid_summary()
    request = client.call_args.kwargs
    payload = json.loads(request["messages"][1]["content"])
    assert payload["surgery"] == "Oak"
    assert payload["month"] == "October 2026"
    assert len(payload["reviews"]) == 2
    assert "EXCLUDE" not in request["messages"][1]["content"]
    assert payload["reviews"][0]["reference"] == "R1"
    assert payload["statistics"]["total_reviews"] == 2
    assert request.get("use_groq", False) is False
    assert request.get("model") is None


@pytest.mark.parametrize("response", ["", "not JSON", "{}", "[]", "null"])
def test_summary_rejects_invalid_response(monkeypatch, response):
    from ai_medreview.monthly_summary import generate_ai_summary

    monkeypatch.setattr(llm_client, "call_llm", Mock(return_value=response))
    with pytest.raises(ValueError, match="AI summary"):
        generate_ai_summary(report_context())


@pytest.mark.parametrize("case", ["unknown_reference", "missing_reference", "too_many", "too_long", "blank_text", "empty_findings", "empty_responses", "missing_note"])
def test_summary_rejects_unsupported_or_excessive_content(monkeypatch, case):
    from ai_medreview.monthly_summary import generate_ai_summary

    response = valid_summary()
    if case == "unknown_reference":
        response["strengths"][0]["references"] = ["R999"]
    elif case == "missing_reference":
        response["responses"][0]["references"] = []
    elif case == "too_many":
        response["responses"] *= 4
    elif case == "too_long":
        response["strengths"][0]["text"] = "word " * 351
    elif case == "blank_text":
        response["concerns"][0]["text"] = " "
    elif case == "empty_findings":
        response["strengths"] = response["concerns"] = []
    elif case == "empty_responses":
        response["responses"] = []
    else:
        response.pop("evidence_note")
    monkeypatch.setattr(llm_client, "call_llm", Mock(return_value=json.dumps(response)))
    with pytest.raises(ValueError, match="AI summary"):
        generate_ai_summary(report_context())


def test_summary_requires_written_feedback_without_calling_service(monkeypatch):
    from ai_medreview.monthly_summary import generate_ai_summary

    context = report_context()
    for review in context["reviews"]:
        review["text"] = "  "
    client = Mock()
    monkeypatch.setattr(llm_client, "call_llm", client)
    with pytest.raises(ValueError, match="written feedback"):
        generate_ai_summary(context)
    client.assert_not_called()


def test_summary_service_failure_stops_generation(monkeypatch):
    from ai_medreview.monthly_summary import generate_ai_summary

    monkeypatch.setattr(llm_client, "call_llm", Mock(side_effect=requests.Timeout("private provider details")))
    with pytest.raises(ValueError, match="AI summary.*Generate report") as error:
        generate_ai_summary(report_context())
    assert "private provider details" not in str(error.value)


def test_preview_does_not_call_llm(monkeypatch):
    client = Mock(side_effect=AssertionError("Preview must not call the service"))
    monkeypatch.setattr(llm_client, "call_llm", client)
    report_context()
    client.assert_not_called()


@pytest.mark.parametrize("fence", ["```json\n{}\n```", "```\n{}\n```", "```JSON\n{}\n```"])
def test_summary_accepts_json_code_fences(monkeypatch, fence):
    from ai_medreview.monthly_summary import generate_ai_summary

    client = Mock(return_value=fence.replace("{}", json.dumps(valid_summary())))
    monkeypatch.setattr(llm_client, "call_llm", client)
    assert generate_ai_summary(report_context()) == valid_summary()
    client.assert_called_once()


def test_repeated_evidence_references_are_safely_deduplicated(monkeypatch):
    from ai_medreview.monthly_summary import generate_ai_summary

    response = valid_summary()
    response["strengths"][0]["references"] = ["R1", "R1"]
    monkeypatch.setattr(llm_client, "call_llm", Mock(return_value=json.dumps(response)))
    assert generate_ai_summary(report_context()) == valid_summary()


@pytest.mark.parametrize("defect", ["length", "reference", "json"])
def test_invalid_summary_is_corrected_once_using_original_evidence(monkeypatch, defect):
    from ai_medreview.monthly_summary import generate_ai_summary

    response = valid_summary()
    if defect == "length":
        response["strengths"][0]["text"] = "word " * 351
    elif defect == "reference":
        response["strengths"][0]["references"] = ["R999"]
    bad_response = "not JSON" if defect == "json" else json.dumps(response)
    client = Mock(side_effect=[bad_response, json.dumps(valid_summary())])
    monkeypatch.setattr(llm_client, "call_llm", client)
    assert generate_ai_summary(report_context()) == valid_summary()
    assert client.call_count == 2
    repair = client.call_args.kwargs["messages"]
    original = json.loads(repair[1]["content"])
    assert len(original["reviews"]) == 2
    assert original["reviews"][1]["reference"] == "R2"
    assert repair[-2] == {"role": "assistant", "content": bad_response}
    assert repair[-1]["role"] == "user"
    assert "correct" in repair[-1]["content"].lower()


def test_invalid_summary_retries_are_bounded_and_diagnostics_are_private(monkeypatch, caplog):
    from ai_medreview.monthly_summary import generate_ai_summary

    response = valid_summary()
    response["strengths"][0]["text"] = "PRIVATE REVIEW TEXT"
    response["strengths"][0]["references"] = ["R999"]
    client = Mock(return_value=json.dumps(response))
    monkeypatch.setattr(llm_client, "call_llm", client)
    with pytest.raises(ValueError, match="review references.*Generate report"):
        generate_ai_summary(report_context())
    assert client.call_count == 2
    assert "unknown_reference" in caplog.text
    assert "PRIVATE REVIEW TEXT" not in caplog.text
    assert "Kind reception team" not in caplog.text


def test_invalid_response_cannot_skip_evidence_checks_after_retry(monkeypatch):
    from ai_medreview.monthly_summary import generate_ai_summary

    response = valid_summary()
    response["responses"][0]["references"] = ["R999"]
    monkeypatch.setattr(llm_client, "call_llm", Mock(return_value=json.dumps(response)))
    with pytest.raises(ValueError, match="AI summary"):
        generate_ai_summary(report_context())
