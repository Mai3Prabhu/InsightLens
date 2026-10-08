"""Tests for the Gemini provider wrapper (no network; fake SDK client injected)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.genai import errors

from app.analysis import load_interview_questions
from app.analyzer import ExpertAnalyzer
from app.comparison import build_demo_analyses, build_demo_comparison
from app.llm_client import GeminiLLMClient, LLMClientError, _sanitize_gemini_error
from app.parser import parse_all_transcripts
from app.prompts import QA_SYSTEM_INSTRUCTIONS
from app.qa import MAX_QUESTION_CHARS, QAError, QuestionAnswering
from app.retrieval import build_retriever
from app.schemas import CrossExpertComparison, QAAnswerPayload

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class _FakeResponse:
    def __init__(self, text: str | None, parsed: object = None) -> None:
        self.text = text
        self.parsed = parsed


class _FakeModels:
    def __init__(self, response: object = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.exc is not None:
            raise self.exc
        return self.response


class _FakeGenAIClient:
    def __init__(self, response: object = None, exc: Exception | None = None) -> None:
        self.models = _FakeModels(response, exc)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm_client.time.sleep", lambda _s: None)


@pytest.fixture(scope="module")
def transcripts():
    return parse_all_transcripts(DATA_DIR)


def _client(response=None, exc=None) -> tuple[GeminiLLMClient, _FakeGenAIClient]:
    fake = _FakeGenAIClient(response, exc)
    return GeminiLLMClient(api_key="unused", model="gemini-2.5-flash", client=fake), fake


# ---------------------------------------------------------------------------
# Structured output parsing
# ---------------------------------------------------------------------------


def test_analysis_json_is_parsed_and_validated_end_to_end(transcripts) -> None:
    fr = transcripts[0]
    expected = build_demo_analyses(transcripts)[0]
    client, fake = _client(_FakeResponse(expected.model_dump_json()))

    analysis = ExpertAnalyzer(client).analyze(fr, load_interview_questions())

    assert [a.question_id for a in analysis.answers] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    call = fake.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    # Transcript segments are sent; system rules go in system_instruction.
    assert "segment_id=FR_002 |" in call["contents"]
    assert "Do not invent or create segment IDs" in call["config"].system_instruction
    assert call["config"].response_mime_type == "application/json"


def test_sdk_parsed_object_is_used_when_present() -> None:
    payload = build_demo_comparison()
    client, _ = _client(_FakeResponse(text=None, parsed=payload))
    result = client.compare_expert_analyses([])
    assert isinstance(result, CrossExpertComparison)
    assert result == payload


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_llm_response_raises_clean_error(text) -> None:
    client, _ = _client(_FakeResponse(text))
    with pytest.raises(LLMClientError, match="empty response"):
        client.answer_question("What are the barriers?", [])


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        json.dumps({"evidence": []}),  # missing answer
        json.dumps({"answer": "x", "evidence": [{"segment_id": "FR_001"}]}),  # bad evidence
    ],
)
def test_malformed_structured_response_raises_clean_error(text) -> None:
    client, _ = _client(_FakeResponse(text))
    with pytest.raises(LLMClientError, match="does not match QAAnswerPayload"):
        client.answer_question("What are the barriers?", [])


def test_text_property_that_raises_is_treated_as_empty() -> None:
    class _BlockedResponse:
        parsed = None

        @property
        def text(self):
            raise ValueError("no candidates")

    client, _ = _client(_BlockedResponse())
    with pytest.raises(LLMClientError, match="empty response"):
        client.answer_question("What are the barriers?", [])


# ---------------------------------------------------------------------------
# API failures are sanitized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (errors.ClientError(401, {"error": {"message": "bad key"}}), "authentication failed"),
        (errors.ClientError(403, {"error": {"message": "denied"}}), "authentication failed"),
        (errors.ClientError(404, {"error": {"message": "nope"}}), "GEMINI_MODEL"),
        (errors.ClientError(429, {"error": {"message": "quota"}}), "rate limit"),
        (errors.ServerError(503, {"error": {"message": "overloaded"}}), "temporarily unavailable"),
        (TimeoutError("read timed out"), "timed out"),
        (ConnectionError("refused"), "connect"),
    ],
)
def test_api_failures_are_mapped_to_safe_messages(exc, expected) -> None:
    client, _ = _client(exc=exc)
    with pytest.raises(LLMClientError, match=expected) as info:
        client.answer_question("What are the barriers?", [])
    # Original exception chain is suppressed so details cannot leak via __cause__.
    assert info.value.__cause__ is None


def test_transient_error_is_retried_then_succeeds() -> None:
    class _FlakyModels(_FakeModels):
        def generate_content(self, *, model, contents, config):
            self.calls.append({"model": model})
            if len(self.calls) == 1:
                raise errors.ServerError(503, {"error": {"message": "high demand"}})
            return _FakeResponse(json.dumps({"answer": "ok", "evidence": []}))

    fake = _FakeGenAIClient()
    fake.models = _FlakyModels()
    client = GeminiLLMClient(api_key="unused", client=fake)
    assert client.answer_question("q", []).answer == "ok"
    assert len(fake.models.calls) == 2


def test_non_transient_error_is_not_retried() -> None:
    client, fake = _client(exc=errors.ClientError(401, {"error": {"message": "bad"}}))
    with pytest.raises(LLMClientError):
        client.answer_question("q", [])
    assert len(fake.models.calls) == 1


def test_sanitized_client_error_never_echoes_server_message() -> None:
    secret = "AIzaSy-LEAKED-IN-SERVER-MESSAGE"
    exc = errors.ClientError(400, {"error": {"message": f"key {secret} invalid"}})
    sanitized = _sanitize_gemini_error(exc)
    assert secret not in str(sanitized)


# ---------------------------------------------------------------------------
# Q&A input guards and prompt grounding rules
# ---------------------------------------------------------------------------


def test_extremely_long_question_is_rejected_before_llm(transcripts) -> None:
    client, fake = _client(_FakeResponse(json.dumps({"answer": "x", "evidence": []})))
    qa = QuestionAnswering(build_retriever(transcripts), client, transcripts=transcripts)
    with pytest.raises(QAError, match="too long"):
        qa.answer_question("training " * MAX_QUESTION_CHARS)
    assert fake.models.calls == []


@pytest.mark.parametrize(
    "question",
    ["What is Apple's stock price?", "What happened in the industry in 2026?", "?!?..."],
)
def test_no_overlap_questions_make_no_llm_call(transcripts, question) -> None:
    client, fake = _client(_FakeResponse(json.dumps({"answer": "x", "evidence": []})))
    qa = QuestionAnswering(build_retriever(transcripts), client, transcripts=transcripts)
    response = qa.answer_question(question)
    assert response.evidence == []
    assert "do not contain enough information" in response.answer
    assert fake.models.calls == []


def test_qa_llm_receives_only_retrieved_segments(transcripts) -> None:
    payload = QAAnswerPayload(answer="Training matters.", evidence=[])
    client, fake = _client(_FakeResponse(payload.model_dump_json()))
    retriever = build_retriever(transcripts)
    qa = QuestionAnswering(retriever, client, transcripts=transcripts)
    qa.answer_question("What role does training play?")

    sent = fake.models.calls[0]["contents"]
    retrieved = {s.segment_id for s in retriever.search("What role does training play?", top_k=qa.top_k)}
    all_ids = {s.segment_id for t in transcripts for s in t.segments}
    for sid in all_ids:
        assert (f"segment_id={sid} |" in sent) == (sid in retrieved)


def test_qa_prompt_guards_against_false_premises_and_rankings() -> None:
    rules = QA_SYSTEM_INSTRUCTIONS.lower()
    assert "do not accept the premise" in rules
    assert "do not rank" in rules
    assert "outside knowledge" in rules


def test_schema_is_sent_as_json_schema_so_strict_models_are_accepted() -> None:
    """CrossExpertComparison forbids extra fields; Gemini's legacy response_schema
    rejects additionalProperties, so the JSON-schema field must be used."""
    client, fake = _client(_FakeResponse(build_demo_comparison().model_dump_json()))
    client.compare_expert_analyses([])
    config = fake.models.calls[0]["config"]
    assert config.response_schema is None
    assert config.response_json_schema["additionalProperties"] is False
