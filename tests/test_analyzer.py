"""Tests for Loop 3 LLM-powered expert analysis (mocked LLM only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis import load_interview_questions
from app.analyzer import AnalyzerError, ExpertAnalyzer, resolve_evidence
from app.llm_client import DEFAULT_GEMINI_MODEL, GeminiLLMClient, LLMClient, LLMClientError
from app.parser import parse_transcript
from app.schemas import (
    EvidenceReference,
    ExpertAnalysis,
    ExpertAnswer,
    InterviewQuestion,
    ResolvedEvidence,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRANCE = DATA_DIR / "Transcript_1_France.txt"


class FakeLLMClient(LLMClient):
    """Deterministic stand-in for GeminiLLMClient (no network calls)."""

    def __init__(self, response: ExpertAnalysis | Exception) -> None:
        self.response = response
        self.calls: list[tuple] = []

    def analyze_expert_transcript(
        self,
        transcript,
        questions,
    ) -> ExpertAnalysis:
        self.calls.append((transcript, list(questions)))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response.model_copy(deep=True)

    def compare_expert_analyses(self, analyses):
        raise NotImplementedError("Loop 3 tests do not use comparison.")

    def answer_question(self, question, evidence):
        raise NotImplementedError("Loop 3 tests do not use Q&A.")


def _questions() -> list[InterviewQuestion]:
    return load_interview_questions()


def _base_analysis_for(transcript) -> ExpertAnalysis:
    """Build a plausible full ExpertAnalysis using real FR segment IDs."""
    return ExpertAnalysis(
        transcript_id=transcript.transcript_id,
        expert_name=transcript.expert_name,
        role=transcript.role,
        market=transcript.market,
        answers=[
            ExpertAnswer(
                question_id="Q1",
                answer=(
                    "Adoption is growing but still concentrated in larger "
                    "academic hospitals and private centres."
                ),
                evidence=[
                    EvidenceReference(
                        segment_id="FR_002",
                        reason="Expert describes current adoption pattern.",
                    )
                ],
            ),
            ExpertAnswer(
                question_id="Q2",
                answer="Capital budget approval is the biggest barrier.",
                evidence=[
                    EvidenceReference(
                        segment_id="FR_004",
                        reason="Expert names capital budget approval as the main issue.",
                    )
                ],
            ),
            ExpertAnswer(
                question_id="Q3",
                answer="ROI and economics are very important to finance teams.",
                evidence=[
                    EvidenceReference(
                        segment_id="FR_006",
                        reason="Expert stresses utilisation, volume, and payback.",
                    )
                ],
            ),
            ExpertAnswer(
                question_id="Q4",
                answer="Training matters, especially in the first year.",
                evidence=[
                    EvidenceReference(
                        segment_id="FR_008",
                        reason="Expert discusses surgeon training and utilisation.",
                    )
                ],
            ),
            ExpertAnswer(
                question_id="Q5",
                answer=(
                    "Adoption should increase steadily, perhaps 15–20% more "
                    "procedures annually in stronger centres."
                ),
                evidence=[
                    EvidenceReference(
                        segment_id="FR_010",
                        reason="Expert gives a three-to-five-year outlook.",
                    )
                ],
            ),
            ExpertAnswer(
                question_id="Q6",
                answer="Purchase decisions typically take six to twelve months.",
                evidence=[
                    EvidenceReference(
                        segment_id="FR_014",
                        reason="Expert states the usual purchase timeline.",
                    )
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Successful analysis
# ---------------------------------------------------------------------------


def test_successful_analysis_of_transcript() -> None:
    transcript = parse_transcript(FRANCE)
    expected = _base_analysis_for(transcript)
    analyzer = ExpertAnalyzer(FakeLLMClient(expected))

    result = analyzer.analyze(transcript, _questions())

    assert isinstance(result, ExpertAnalysis)
    assert result.transcript_id == "FR"
    assert result.expert_name == "Dr. Jean Martin"
    assert result.answers[0].answer.startswith("Adoption is growing")


def test_exactly_six_answers_returned() -> None:
    transcript = parse_transcript(FRANCE)
    result = ExpertAnalyzer(FakeLLMClient(_base_analysis_for(transcript))).analyze(
        transcript, _questions()
    )
    assert len(result.answers) == 6


def test_every_answer_has_valid_question_id() -> None:
    transcript = parse_transcript(FRANCE)
    result = ExpertAnalyzer(FakeLLMClient(_base_analysis_for(transcript))).analyze(
        transcript, _questions()
    )
    assert [a.question_id for a in result.answers] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
    ]


def test_valid_evidence_segment_ids_accepted() -> None:
    transcript = parse_transcript(FRANCE)
    result = ExpertAnalyzer(FakeLLMClient(_base_analysis_for(transcript))).analyze(
        transcript, _questions()
    )
    all_ids = {s.segment_id for s in transcript.segments}
    for answer in result.answers:
        for ref in answer.evidence:
            assert ref.segment_id in all_ids


def test_invalid_evidence_segment_ids_rejected() -> None:
    transcript = parse_transcript(FRANCE)
    bad = _base_analysis_for(transcript)
    bad.answers[0].evidence = [
        EvidenceReference(segment_id="FR_999", reason="Invented segment.")
    ]
    analyzer = ExpertAnalyzer(FakeLLMClient(bad))

    with pytest.raises(AnalyzerError, match="FR_999"):
        analyzer.analyze(transcript, _questions())


def test_missing_question_ids_detected() -> None:
    transcript = parse_transcript(FRANCE)
    incomplete = _base_analysis_for(transcript)
    incomplete.answers = incomplete.answers[:5]  # drop Q6
    analyzer = ExpertAnalyzer(FakeLLMClient(incomplete))

    with pytest.raises(AnalyzerError, match="missing question IDs"):
        analyzer.analyze(transcript, _questions())


def test_duplicate_question_ids_rejected() -> None:
    transcript = parse_transcript(FRANCE)
    duplicated = _base_analysis_for(transcript)
    duplicated.answers[5] = ExpertAnswer(
        question_id="Q1",
        answer="Duplicate Q1 payload.",
        evidence=[],
    )
    analyzer = ExpertAnalyzer(FakeLLMClient(duplicated))

    with pytest.raises(AnalyzerError, match="duplicate question IDs"):
        analyzer.analyze(transcript, _questions())


def test_empty_evidence_list_allowed_when_no_support() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = _base_analysis_for(transcript)
    analysis.answers[4] = ExpertAnswer(
        question_id="Q5",
        answer=(
            "The transcript does not provide enough information to answer "
            "this question."
        ),
        evidence=[],
    )
    result = ExpertAnalyzer(FakeLLMClient(analysis)).analyze(transcript, _questions())
    assert result.answers[4].evidence == []
    assert "does not provide enough information" in result.answers[4].answer


def test_evidence_resolution_returns_original_source_fields() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = ExpertAnalyzer(FakeLLMClient(_base_analysis_for(transcript))).analyze(
        transcript, _questions()
    )
    resolved = resolve_evidence(analysis, transcript)

    assert resolved
    assert all(isinstance(item, ResolvedEvidence) for item in resolved)

    # Q6 evidence should match FR_014 source segment exactly.
    q6_items = [r for r in resolved if r.question_id == "Q6"]
    assert len(q6_items) == 1
    source = transcript.segments[-1]
    assert q6_items[0].segment_id == source.segment_id == "FR_014"
    assert q6_items[0].timestamp == source.timestamp
    assert q6_items[0].speaker == source.speaker
    assert q6_items[0].text == source.text
    assert q6_items[0].text == (
        "Six to twelve months is realistic once the hospital becomes serious. "
        "It can be longer if the capital committee pushes the purchase into "
        "the next budget cycle."
    )


def test_analyzer_does_not_modify_original_transcript() -> None:
    transcript = parse_transcript(FRANCE)
    original = transcript.model_dump()
    ExpertAnalyzer(FakeLLMClient(_base_analysis_for(transcript))).analyze(
        transcript, _questions()
    )
    assert transcript.model_dump() == original


def test_metadata_mismatch_rejected() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = _base_analysis_for(transcript)
    analysis.market = "Germany"
    with pytest.raises(AnalyzerError, match="metadata does not match"):
        ExpertAnalyzer(FakeLLMClient(analysis)).analyze(transcript, _questions())


def test_llm_client_error_propagates() -> None:
    transcript = parse_transcript(FRANCE)
    client = FakeLLMClient(LLMClientError("GEMINI_API_KEY is not set."))
    with pytest.raises(LLMClientError, match="GEMINI_API_KEY"):
        ExpertAnalyzer(client).analyze(transcript, _questions())


def test_gemini_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMClientError, match="GEMINI_API_KEY"):
        GeminiLLMClient(api_key=None)


def test_gemini_client_rejects_whitespace_api_key() -> None:
    with pytest.raises(LLMClientError, match="GEMINI_API_KEY"):
        GeminiLLMClient(api_key="   ")


def test_gemini_client_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    # Inject a stub client so we do not need a real key/network.
    stub = object()
    client = GeminiLLMClient(api_key="test-key-not-used", client=stub)
    assert client.model == DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"


def test_resolve_evidence_rejects_unknown_segment() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = _base_analysis_for(transcript)
    analysis.answers[0].evidence = [
        EvidenceReference(segment_id="FR_999", reason="Bad ref.")
    ]
    # Bypass analyzer validation to unit-test resolve_evidence directly.
    with pytest.raises(AnalyzerError, match="FR_999"):
        resolve_evidence(analysis, transcript)
