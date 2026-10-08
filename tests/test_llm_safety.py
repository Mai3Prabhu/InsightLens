"""Tests for LLM failure modes, error sanitization, and live API safety (Loop 7).

Verifies:
1. Controlled application error handling across API timeouts, connection drops,
   refusals, and malformed structured outputs (no hallucinated fallback answers).
2. Secret safety: API keys are never printed, leaked in logs or error messages,
   and .env is strictly ignored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis import load_interview_questions
from app.analyzer import AnalyzerError, ExpertAnalyzer
from app.comparison import ComparisonError, CrossExpertComparer, build_demo_analyses
from app.llm_client import (
    DEFAULT_GEMINI_MODEL,
    GeminiLLMClient,
    LLMClient,
    LLMClientError,
    _sanitize_gemini_error,
)
from app.parser import parse_all_transcripts, parse_transcript
from app.qa import QAError, QuestionAnswering
from app.retrieval import build_retriever

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRANCE = DATA_DIR / "Transcript_1_France.txt"


class MockFailureClient(LLMClient):
    """Mock client configured to fail in specific ways."""

    def __init__(self, exc_to_raise: Exception | None = None, return_val: object = None) -> None:
        self.exc = exc_to_raise
        self.val = return_val

    def analyze_expert_transcript(self, transcript, questions):
        if self.exc:
            raise self.exc
        return self.val

    def compare_expert_analyses(self, analyses):
        if self.exc:
            raise self.exc
        return self.val

    def answer_question(self, question, evidence):
        if self.exc:
            raise self.exc
        return self.val


@pytest.fixture(scope="module")
def transcripts():
    return parse_all_transcripts(DATA_DIR)


@pytest.fixture(scope="module")
def fr_transcript():
    return parse_transcript(FRANCE)


# ---------------------------------------------------------------------------
# Section 10: LLM Failure Simulations (No Fabricated Fallbacks)
# ---------------------------------------------------------------------------


def test_analyzer_handles_api_exception(fr_transcript) -> None:
    """Network or API exceptions must raise LLMClientError, not fabricate answers."""
    client = MockFailureClient(exc_to_raise=LLMClientError("Connection refused by Gemini API."))
    analyzer = ExpertAnalyzer(client)
    with pytest.raises(LLMClientError, match="Connection refused"):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_analyzer_handles_timeout_failure(fr_transcript) -> None:
    """Timeouts must raise LLMClientError with clean messaging."""
    client = MockFailureClient(exc_to_raise=LLMClientError("Gemini request timed out."))
    analyzer = ExpertAnalyzer(client)
    with pytest.raises(LLMClientError, match="timed out"):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_analyzer_handles_malformed_structured_response(fr_transcript) -> None:
    """Payloads that cannot be validated as ExpertAnalysis raise AnalyzerError."""
    client = MockFailureClient(return_val={"invalid": "structure", "no_answers": True})
    analyzer = ExpertAnalyzer(client)
    with pytest.raises(AnalyzerError, match="not a valid ExpertAnalysis"):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_analyzer_handles_empty_response(fr_transcript) -> None:
    """Empty or None response from LLM raises AnalyzerError without crash."""
    client = MockFailureClient(return_val=None)
    analyzer = ExpertAnalyzer(client)
    with pytest.raises(AnalyzerError):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_comparer_handles_api_exception(transcripts) -> None:
    """Comparison service propagates LLMClientError cleanly."""
    analyses = build_demo_analyses(transcripts)
    client = MockFailureClient(exc_to_raise=LLMClientError("Gemini rate limit exceeded."))
    comparer = CrossExpertComparer(client)
    with pytest.raises(LLMClientError, match="rate limit"):
        comparer.compare(analyses, transcripts)


def test_comparer_handles_malformed_comparison_payload(transcripts) -> None:
    """Malformed comparison payload raises ComparisonError."""
    analyses = build_demo_analyses(transcripts)
    client = MockFailureClient(return_val={"corrupted": "payload"})
    comparer = CrossExpertComparer(client)
    with pytest.raises(ComparisonError, match="not a valid CrossExpertComparison"):
        comparer.compare(analyses, transcripts)


def test_qa_handles_api_exception(transcripts) -> None:
    """Q&A service propagates LLMClientError cleanly."""
    retriever = build_retriever(transcripts)
    client = MockFailureClient(exc_to_raise=LLMClientError("Authentication failure."))
    qa = QuestionAnswering(retriever, client, transcripts=transcripts)
    with pytest.raises(LLMClientError, match="Authentication failure"):
        qa.answer_question("What are the barriers to adoption?")


def test_qa_handles_malformed_qa_payload(transcripts) -> None:
    """Malformed Q&A response raises QAError."""
    retriever = build_retriever(transcripts)
    client = MockFailureClient(return_val="Just a string instead of QAAnswerPayload")
    qa = QuestionAnswering(retriever, client, transcripts=transcripts)
    with pytest.raises(QAError, match="not a valid QAAnswerPayload"):
        qa.answer_question("What are the barriers to adoption?")


# ---------------------------------------------------------------------------
# Section 11: Live API Safety, Secrets, and Error Sanitization
# ---------------------------------------------------------------------------


def test_gitignore_ignores_dotenv_files() -> None:
    """.gitignore must explicitly ignore .env and .env.local."""
    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore file must exist"
    content = gitignore_path.read_text(encoding="utf-8")
    ignored = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
    assert ".env" in ignored
    assert ".env.local" in ignored


def test_env_example_contains_only_placeholders() -> None:
    """.env.example must document the variables without carrying a real key."""
    example_file = ROOT / ".env.example"
    assert example_file.exists()
    values = {}
    for line in example_file.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    assert "GEMINI_API_KEY" in values
    assert values["GEMINI_API_KEY"] == "", ".env.example must not contain a real API key"
    assert values.get("GEMINI_MODEL") == DEFAULT_GEMINI_MODEL


def test_gemini_model_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model name defaults to gemini-2.5-flash and can be configured via env or init."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    stub = object()
    client_default = GeminiLLMClient(api_key="dummy", model=None, client=stub)
    assert client_default.model == DEFAULT_GEMINI_MODEL

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    client_env = GeminiLLMClient(api_key="dummy", client=stub)
    assert client_env.model == "gemini-2.5-pro"

    client_custom = GeminiLLMClient(api_key="dummy", model="gemini-custom", client=stub)
    assert client_custom.model == "gemini-custom"


def test_error_sanitizer_masks_secrets() -> None:
    """Error sanitizer maps SDK exceptions to safe messages without leaking keys."""
    secret_key = "AIzaSy-SUPERSECRETKEY1234567890"

    # Simulate an arbitrary exception containing a secret key in its message
    raw_exc = Exception(f"Failed request with header x-goog-api-key: {secret_key}")
    sanitized = _sanitize_gemini_error(raw_exc)

    assert isinstance(sanitized, LLMClientError)
    # The secret key must NEVER appear in the sanitized message
    assert secret_key not in str(sanitized)


def test_gemini_client_string_representation_does_not_leak_key() -> None:
    """String representations of GeminiLLMClient must not print raw API keys."""
    secret = "AIzaSy-TESTKEY-DO-NOT-PRINT"
    stub = object()
    client = GeminiLLMClient(api_key=secret, client=stub)
    assert secret not in repr(client)
    assert secret not in str(client)
