"""Adversarial evidence and traceability tests for Loop 7 audit.

Tests that invalid evidence IDs, wrong transcript pairings, empty/duplicate IDs,
and missing or malformed answers are strictly rejected, and that original
transcript text is the sole authoritative source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.analysis import load_interview_questions
from app.analyzer import AnalyzerError, ExpertAnalyzer, resolve_evidence
from app.comparison import (
    ComparisonError,
    CrossExpertComparer,
    build_demo_analyses,
    build_demo_comparison,
    resolve_comparison_evidence,
)
from app.llm_client import LLMClient
from app.parser import parse_all_transcripts, parse_transcript
from app.qa import (
    QAError,
    QuestionAnswering,
    resolve_qa_evidence,
)
from app.retrieval import build_retriever
from app.schemas import (
    ComparisonEvidence,
    CrossExpertComparison,
    EvidenceReference,
    ExpertAnalysis,
    ExpertAnswer,
    QAAnswerPayload,
    QAEvidence,
    QAResponse,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRANCE = DATA_DIR / "Transcript_1_France.txt"
GERMANY = DATA_DIR / "Transcript_2_Germany.txt"
UK = DATA_DIR / "Transcript_3_UK.txt"


class FakeAdversarialLLMClient(LLMClient):
    """Configurable mock LLM for injecting adversarial payloads."""

    def __init__(self, response: object) -> None:
        self.response = response

    def analyze_expert_transcript(self, transcript, questions):
        if isinstance(self.response, Exception):
            raise self.response
        if isinstance(self.response, ExpertAnalysis):
            return self.response.model_copy(deep=True)
        return self.response

    def compare_expert_analyses(self, analyses):
        if isinstance(self.response, Exception):
            raise self.response
        if isinstance(self.response, CrossExpertComparison):
            return self.response.model_copy(deep=True)
        return self.response

    def answer_question(self, question, evidence):
        if isinstance(self.response, Exception):
            raise self.response
        if isinstance(self.response, QAAnswerPayload):
            return self.response.model_copy(deep=True)
        return self.response


@pytest.fixture(scope="module")
def all_transcripts():
    return parse_all_transcripts(DATA_DIR)


@pytest.fixture(scope="module")
def fr_transcript():
    return parse_transcript(FRANCE)


@pytest.fixture(scope="module")
def de_transcript():
    return parse_transcript(GERMANY)


@pytest.fixture(scope="module")
def uk_transcript():
    return parse_transcript(UK)


@pytest.fixture(scope="module")
def demo_analyses(all_transcripts):
    return build_demo_analyses(all_transcripts)


# ---------------------------------------------------------------------------
# Section 2: Complete Traceability Audit & Source-Text Authoritativeness
# ---------------------------------------------------------------------------


def test_expert_analysis_traceability_exact_match(fr_transcript) -> None:
    """Expert Analysis evidence resolves strictly to original TranscriptSegment text."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    fr_analysis = analyses[0]
    resolved = resolve_evidence(fr_analysis, fr_transcript)

    by_id = {s.segment_id: s for s in fr_transcript.segments}
    assert resolved
    for item in resolved:
        original = by_id[item.segment_id]
        assert item.timestamp == original.timestamp
        assert item.speaker == original.speaker
        assert item.text == original.text
        # Strict exact string equality
        assert item.text is original.text or item.text == original.text


def test_comparison_traceability_exact_match(all_transcripts, demo_analyses) -> None:
    """Comparison evidence resolves strictly to original TranscriptSegment text."""
    comparison = build_demo_comparison()
    resolved = resolve_comparison_evidence(comparison, all_transcripts)

    indexed = {
        (t.transcript_id, s.segment_id): s
        for t in all_transcripts
        for s in t.segments
    }
    assert resolved
    for item in resolved:
        original = indexed[(item.transcript_id, item.segment_id)]
        assert item.timestamp == original.timestamp
        assert item.speaker == original.speaker
        assert item.text == original.text


def test_qa_traceability_exact_match(all_transcripts) -> None:
    """Q&A evidence resolves strictly to original TranscriptSegment text."""
    retriever = build_retriever(all_transcripts)
    query = "What did experts say about procurement timelines?"
    retrieved = retriever.search(query, top_k=6)
    assert retrieved

    # Craft Q&A response citing retrieved segments
    payload = QAAnswerPayload(
        answer="Timelines vary from 6 to 18 months depending on market.",
        evidence=[
            QAEvidence(
                transcript_id=retrieved[0].transcript_id,
                segment_id=retrieved[0].segment_id,
                reason="Procurement timeline evidence.",
            ),
            QAEvidence(
                transcript_id=retrieved[1].transcript_id,
                segment_id=retrieved[1].segment_id,
                reason="Comparison timeline evidence.",
            ),
        ],
    )
    response = QAResponse(
        question=query,
        answer=payload.answer,
        evidence=payload.evidence,
    )
    resolved = resolve_qa_evidence(
        response, retrieved=retrieved, transcripts=all_transcripts
    )
    indexed = {
        (t.transcript_id, s.segment_id): s
        for t in all_transcripts
        for s in t.segments
    }
    for item in resolved:
        original = indexed[(item.transcript_id, item.segment_id)]
        assert item.timestamp == original.timestamp
        assert item.speaker == original.speaker
        assert item.text == original.text


def test_pipeline_never_trusts_hallucinated_or_modified_llm_quote(fr_transcript) -> None:
    """If an LLM or caller attempts to attach a fabricated quote, resolution ignores it."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    fr_analysis = analyses[0]

    # EvidenceReference schema does not store quotes
    ref = fr_analysis.answers[0].evidence[0]
    assert not hasattr(ref, "quote")

    # When resolved, source text is taken from fr_transcript.segments, NOT any hypothetical quote
    resolved = resolve_evidence(fr_analysis, fr_transcript)
    source_segment = next(s for s in fr_transcript.segments if s.segment_id == ref.segment_id)
    assert resolved[0].text == source_segment.text
    assert resolved[0].speaker == source_segment.speaker
    assert resolved[0].timestamp == source_segment.timestamp


# ---------------------------------------------------------------------------
# Section 3: Invalid Evidence Tests (Non-existent, Wrong Market, Empty, Duplicates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["FR_999", "DE_999", "UK_999", "INVALID_001"])
def test_expert_analysis_rejects_nonexistent_segment_id(fr_transcript, bad_id) -> None:
    """Expert analysis with nonexistent segment IDs (FR_999, etc.) must be rejected."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    analysis = analyses[0].model_copy(deep=True)
    analysis.answers[0].evidence = [EvidenceReference(segment_id=bad_id, reason="Bad ID")]

    analyzer = ExpertAnalyzer(FakeAdversarialLLMClient(analysis))
    with pytest.raises(AnalyzerError, match=bad_id):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_expert_analysis_rejects_evidence_belonging_to_another_expert(fr_transcript) -> None:
    """France analysis citing German segment DE_001 must be rejected."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    analysis = analyses[0].model_copy(deep=True)
    analysis.answers[0].evidence = [
        EvidenceReference(segment_id="DE_001", reason="German segment in France analysis.")
    ]

    analyzer = ExpertAnalyzer(FakeAdversarialLLMClient(analysis))
    with pytest.raises(AnalyzerError, match="DE_001"):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_expert_analysis_rejects_duplicate_evidence_ids(fr_transcript) -> None:
    """Duplicate evidence IDs within an answer must be rejected with controlled error."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    analysis = analyses[0].model_copy(deep=True)
    analysis.answers[0].evidence = [
        EvidenceReference(segment_id="FR_002", reason="First citation."),
        EvidenceReference(segment_id="FR_002", reason="Duplicate citation."),
    ]

    analyzer = ExpertAnalyzer(FakeAdversarialLLMClient(analysis))
    with pytest.raises(AnalyzerError, match="Duplicate evidence segment_id 'FR_002'"):
        analyzer.analyze(fr_transcript, load_interview_questions())


@pytest.mark.parametrize("empty_val", ["", "   ", "\t\n"])
def test_evidence_reference_rejects_empty_segment_id(empty_val) -> None:
    """EvidenceReference model rejects empty or whitespace-only segment_id."""
    with pytest.raises(ValidationError):
        EvidenceReference(segment_id=empty_val, reason="Some reason")


def test_comparison_rejects_nonexistent_segment_id(all_transcripts, demo_analyses) -> None:
    """Comparison citing FR_999 or DE_999 must be rejected."""
    bad_comparison = build_demo_comparison()
    bad_comparison.themes[0].evidence.append(
        ComparisonEvidence(
            transcript_id="DE",
            segment_id="DE_999",
            reason="Nonexistent German segment.",
        )
    )
    comparer = CrossExpertComparer(FakeAdversarialLLMClient(bad_comparison))
    with pytest.raises(ComparisonError, match="DE_999"):
        comparer.compare(demo_analyses, all_transcripts)


def test_comparison_rejects_valid_segment_id_paired_with_wrong_transcript_id(
    all_transcripts, demo_analyses
) -> None:
    """Comparison pairing transcript_id='FR' with segment_id='DE_001' must be rejected."""
    bad_comparison = build_demo_comparison()
    bad_comparison.themes[0].evidence[0] = ComparisonEvidence(
        transcript_id="FR",
        segment_id="DE_001",
        reason="Mismatched market and segment.",
    )
    comparer = CrossExpertComparer(FakeAdversarialLLMClient(bad_comparison))
    with pytest.raises(ComparisonError, match="DE_001"):
        comparer.compare(demo_analyses, all_transcripts)


def test_comparison_rejects_duplicate_evidence_in_theme(all_transcripts, demo_analyses) -> None:
    """Duplicate evidence references inside a comparison theme must be rejected."""
    bad_comparison = build_demo_comparison()
    bad_comparison.themes[0].evidence = [
        ComparisonEvidence(transcript_id="FR", segment_id="FR_002", reason="First citation"),
        ComparisonEvidence(transcript_id="FR", segment_id="FR_002", reason="Duplicate citation"),
    ]
    comparer = CrossExpertComparer(FakeAdversarialLLMClient(bad_comparison))
    with pytest.raises(ComparisonError, match="Duplicate evidence reference FR/FR_002"):
        comparer.compare(demo_analyses, all_transcripts)


def test_comparison_rejects_duplicate_evidence_in_difference(all_transcripts, demo_analyses) -> None:
    """Duplicate evidence references inside a comparison difference must be rejected."""
    bad_comparison = build_demo_comparison()
    bad_comparison.differences[0].evidence = [
        ComparisonEvidence(transcript_id="DE", segment_id="DE_014", reason="First citation"),
        ComparisonEvidence(transcript_id="DE", segment_id="DE_014", reason="Duplicate citation"),
    ]
    comparer = CrossExpertComparer(FakeAdversarialLLMClient(bad_comparison))
    with pytest.raises(ComparisonError, match="Duplicate evidence reference DE/DE_014"):
        comparer.compare(demo_analyses, all_transcripts)


def test_qa_rejects_nonexistent_segment_id(all_transcripts) -> None:
    """Q&A citing UK_999 must be rejected."""
    retriever = build_retriever(all_transcripts)
    query = "How long does procurement take?"
    retrieved = retriever.search(query, top_k=4)

    bad_payload = QAAnswerPayload(
        answer="Fabricated segment.",
        evidence=[
            QAEvidence(transcript_id="UK", segment_id="UK_999", reason="Invalid ID")
        ],
    )
    qa = QuestionAnswering(retriever, FakeAdversarialLLMClient(bad_payload), transcripts=all_transcripts)
    with pytest.raises(QAError, match="UK_999"):
        qa.answer_question(query)


def test_qa_rejects_valid_segment_with_wrong_transcript_id(all_transcripts) -> None:
    """Q&A pairing transcript_id='FR' with 'UK_002' must be rejected."""
    retriever = build_retriever(all_transcripts)
    query = "What are the barriers to adoption?"
    retrieved = retriever.search(query, top_k=6)

    bad_payload = QAAnswerPayload(
        answer="Mismatched transcript ID.",
        evidence=[
            QAEvidence(transcript_id="FR", segment_id="UK_002", reason="Cross-market pairing.")
        ],
    )
    qa = QuestionAnswering(retriever, FakeAdversarialLLMClient(bad_payload), transcripts=all_transcripts)
    with pytest.raises(QAError, match="FR/UK_002"):
        qa.answer_question(query)


def test_qa_rejects_duplicate_evidence_ids(all_transcripts) -> None:
    """Q&A payload with duplicate evidence citations must be rejected."""
    retriever = build_retriever(all_transcripts)
    query = "What are the barriers to adoption?"
    retrieved = retriever.search(query, top_k=6)
    assert retrieved

    first = retrieved[0]
    bad_payload = QAAnswerPayload(
        answer="Duplicate citations in answer.",
        evidence=[
            QAEvidence(transcript_id=first.transcript_id, segment_id=first.segment_id, reason="Citation 1"),
            QAEvidence(transcript_id=first.transcript_id, segment_id=first.segment_id, reason="Citation 2"),
        ],
    )
    qa = QuestionAnswering(retriever, FakeAdversarialLLMClient(bad_payload), transcripts=all_transcripts)
    with pytest.raises(QAError, match="Duplicate QA evidence reference"):
        qa.answer_question(query)


def test_qa_rejects_evidence_outside_retrieved_context(all_transcripts) -> None:
    """Q&A citing a valid segment that was NOT in the retrieved set must be rejected."""
    retriever = build_retriever(all_transcripts)
    query = "How important is training?"
    retrieved = retriever.search(query, top_k=3)
    retrieved_keys = {(r.transcript_id, r.segment_id) for r in retrieved}

    # Find a valid segment clearly absent from the retrieved set
    all_keys = [
        (t.transcript_id, s.segment_id)
        for t in all_transcripts
        for s in t.segments
    ]
    unretrieved = next(k for k in all_keys if k not in retrieved_keys)

    bad_payload = QAAnswerPayload(
        answer="Citing unretrieved segment.",
        evidence=[
            QAEvidence(
                transcript_id=unretrieved[0],
                segment_id=unretrieved[1],
                reason="Valid in transcript, but unretrieved.",
            )
        ],
    )
    qa = QuestionAnswering(retriever, FakeAdversarialLLMClient(bad_payload), transcripts=all_transcripts)
    with pytest.raises(QAError, match="not in the retrieved evidence"):
        qa.answer_question(query)


# ---------------------------------------------------------------------------
# Section 4: Missing / Malformed Answer Tests
# ---------------------------------------------------------------------------


def test_expert_analysis_rejects_missing_q1(fr_transcript) -> None:
    """Missing Q1 must raise AnalyzerError."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    incomplete = analyses[0].model_copy(deep=True)
    incomplete.answers = [a for a in incomplete.answers if a.question_id != "Q1"]

    analyzer = ExpertAnalyzer(FakeAdversarialLLMClient(incomplete))
    with pytest.raises(AnalyzerError, match="missing question IDs.*Q1"):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_expert_analysis_rejects_duplicate_q2(fr_transcript) -> None:
    """Duplicate Q2 must raise AnalyzerError."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    duplicated = analyses[0].model_copy(deep=True)
    # Replace Q6 with another Q2
    duplicated.answers[5] = ExpertAnswer(
        question_id="Q2",
        answer="Duplicate Q2 answer.",
        evidence=[],
    )

    analyzer = ExpertAnalyzer(FakeAdversarialLLMClient(duplicated))
    with pytest.raises(AnalyzerError, match="duplicate question IDs.*Q2"):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_expert_analysis_rejects_unexpected_q7(fr_transcript) -> None:
    """Unexpected Q7 payload must be rejected."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    raw_dict = analyses[0].model_dump()
    raw_dict["answers"].append(
        {"question_id": "Q7", "answer": "Unexpected question", "evidence": []}
    )

    analyzer = ExpertAnalyzer(FakeAdversarialLLMClient(raw_dict))
    with pytest.raises(AnalyzerError):
        analyzer.analyze(fr_transcript, load_interview_questions())


def test_expert_analysis_reorders_out_of_order_questions(fr_transcript) -> None:
    """Questions returned out of order (e.g. Q6, Q5... Q1) must be re-ordered to Q1–Q6."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    shuffled = analyses[0].model_copy(deep=True)
    shuffled.answers = list(reversed(shuffled.answers))

    assert [a.question_id for a in shuffled.answers] == ["Q6", "Q5", "Q4", "Q3", "Q2", "Q1"]

    analyzer = ExpertAnalyzer(FakeAdversarialLLMClient(shuffled))
    result = analyzer.analyze(fr_transcript, load_interview_questions())

    assert [a.question_id for a in result.answers] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


@pytest.mark.parametrize("empty_answer", ["", "   ", "\t\r\n"])
def test_expert_analysis_rejects_empty_answer_text(fr_transcript, empty_answer) -> None:
    """Empty or whitespace-only answer text must be rejected with AnalyzerError."""
    analyses = build_demo_analyses([fr_transcript, parse_transcript(GERMANY), parse_transcript(UK)])
    incomplete = analyses[0].model_copy(deep=True)
    incomplete.answers[2].answer = empty_answer

    analyzer = ExpertAnalyzer(FakeAdversarialLLMClient(incomplete))
    with pytest.raises(AnalyzerError, match="Answer for question Q3 cannot be empty"):
        analyzer.analyze(fr_transcript, load_interview_questions())
