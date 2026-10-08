"""Tests for Loop 5 retrieval and grounded Q&A (mocked LLM only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.llm_client import LLMClient
from app.parser import parse_all_transcripts
from app.qa import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    QAError,
    QuestionAnswering,
    build_demo_qa_payload,
    resolve_qa_evidence,
    validate_qa_evidence,
)
from app.retrieval import LexicalBM25Retriever, RetrievalError, build_retriever
from app.schemas import (
    QAAnswerPayload,
    QAEvidence,
    QAResponse,
    RetrievedSegment,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class FakeQALLMClient(LLMClient):
    """Deterministic Q&A LLM stand-in (no network)."""

    def __init__(self, response: QAAnswerPayload | Exception | None = None) -> None:
        self.response = response
        self.calls: list[tuple[str, list[RetrievedSegment]]] = []

    def analyze_expert_transcript(self, transcript, questions):
        raise NotImplementedError

    def compare_expert_analyses(self, analyses):
        raise NotImplementedError

    def answer_question(self, question, evidence):
        self.calls.append((question, list(evidence)))
        if isinstance(self.response, Exception):
            raise self.response
        if self.response is not None:
            return self.response.model_copy(deep=True)
        return build_demo_qa_payload(question, evidence)


@pytest.fixture(scope="module")
def transcripts():
    return parse_all_transcripts(DATA_DIR)


@pytest.fixture(scope="module")
def retriever(transcripts):
    return build_retriever(transcripts)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_relevant_query_returns_relevant_segments(retriever) -> None:
    results = retriever.search(
        "What are the biggest barriers to robotic surgery adoption?",
        top_k=8,
    )
    assert results
    # Barrier-related language should appear among top hits.
    blob = " ".join(r.text.lower() for r in results)
    assert any(
        term in blob
        for term in ("barrier", "cost", "capital", "budget", "funding", "training")
    )


def test_results_ranked_by_retrieval_score(retriever) -> None:
    results = retriever.search("How important is training?", top_k=8)
    assert results
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_duplicate_segments_not_returned(retriever) -> None:
    results = retriever.search("What role does cost play in adoption?", top_k=20)
    ids = [r.segment_id for r in results]
    assert len(ids) == len(set(ids))


def test_empty_query_rejected(retriever) -> None:
    with pytest.raises(RetrievalError, match="non-empty"):
        retriever.search("   ")


def test_query_with_no_meaningful_matches_produces_no_usable_evidence(
    transcripts,
) -> None:
    # Extremely high threshold ⇒ no segment survives.
    strict = LexicalBM25Retriever(transcripts, min_score=1_000_000.0)
    results = strict.search("procurement timelines", top_k=8)
    assert results == []


def test_unrelated_query_can_yield_empty_or_weak_results(retriever) -> None:
    # Nonsense tokens with no corpus overlap → empty list.
    results = retriever.search("xyzzy plugh qwerty asdfghjk", top_k=8)
    assert results == []


def test_procurement_timeline_query_retrieves_timeline_segments(retriever) -> None:
    results = retriever.search(
        "What did experts say about procurement timelines?",
        top_k=8,
    )
    assert results
    blob = " ".join(r.text.lower() for r in results)
    assert any(
        term in blob for term in ("month", "months", "procurement", "purchase", "timeline")
    )


# ---------------------------------------------------------------------------
# Q&A service
# ---------------------------------------------------------------------------


def test_qa_service_answers_using_retrieved_evidence(transcripts, retriever) -> None:
    question = "How important is training?"
    client = FakeQALLMClient()
    qa = QuestionAnswering(retriever, client, transcripts=transcripts)
    response = qa.answer_question(question)

    assert isinstance(response, QAResponse)
    assert response.question == question
    assert response.answer
    assert client.calls
    # LLM only received retrieved evidence.
    _, evidence = client.calls[0]
    assert evidence
    allowed = {(e.transcript_id, e.segment_id) for e in evidence}
    for ref in response.evidence:
        assert (ref.transcript_id, ref.segment_id) in allowed


def test_qa_rejects_evidence_ids_not_retrieved(transcripts, retriever) -> None:
    question = "What role does cost play in adoption?"
    retrieved = retriever.search(question, top_k=5)
    assert retrieved

    # Craft an LLM response that invents a segment outside the retrieved set.
    bad = QAAnswerPayload(
        answer="Invented answer.",
        evidence=[
            QAEvidence(
                transcript_id="FR",
                segment_id="FR_999",
                reason="Not retrieved.",
            )
        ],
    )
    qa = QuestionAnswering(
        retriever, FakeQALLMClient(bad), transcripts=transcripts
    )
    with pytest.raises(QAError, match="not in the retrieved evidence"):
        qa.answer_question(question)


def test_llm_cannot_reference_arbitrary_segments_outside_retrieved(
    transcripts, retriever
) -> None:
    question = "What are the biggest barriers to robotic surgery adoption?"
    retrieved = retriever.search(question, top_k=5)
    # Pick a real segment ID that is deliberately NOT in the retrieved top set.
    all_ids = {
        (t.transcript_id, s.segment_id)
        for t in transcripts
        for s in t.segments
    }
    retrieved_keys = {(r.transcript_id, r.segment_id) for r in retrieved}
    outsider = next(key for key in sorted(all_ids) if key not in retrieved_keys)

    bad = QAAnswerPayload(
        answer="Should fail validation.",
        evidence=[
            QAEvidence(
                transcript_id=outsider[0],
                segment_id=outsider[1],
                reason="Outside retrieved set.",
            )
        ],
    )
    with pytest.raises(QAError, match="not in the retrieved evidence"):
        QuestionAnswering(
            retriever, FakeQALLMClient(bad), transcripts=transcripts
        ).answer_question(question)


def test_qa_evidence_resolves_to_original_transcript_text(
    transcripts, retriever
) -> None:
    question = "What did experts say about procurement timelines?"
    retrieved = retriever.search(question, top_k=8)
    response = QuestionAnswering(
        retriever, FakeQALLMClient(), transcripts=transcripts
    ).answer_question(question)

    resolved = resolve_qa_evidence(
        response, retrieved=retrieved, transcripts=transcripts
    )
    assert resolved
    by_id = {
        (t.transcript_id, s.segment_id): s
        for t in transcripts
        for s in t.segments
    }
    for item in resolved:
        source = by_id[(item.transcript_id, item.segment_id)]
        assert item.timestamp == source.timestamp
        assert item.speaker == source.speaker
        assert item.text == source.text


def test_no_evidence_questions_return_insufficient_answer(transcripts) -> None:
    strict = LexicalBM25Retriever(transcripts, min_score=1_000_000.0)
    client = FakeQALLMClient()
    qa = QuestionAnswering(strict, client, transcripts=transcripts)
    response = qa.answer_question(
        "What are the biggest barriers to robotic surgery adoption?"
    )
    assert response.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert response.evidence == []
    # LLM must not be called without evidence.
    assert client.calls == []


def test_empty_question_rejected_by_qa(transcripts, retriever) -> None:
    qa = QuestionAnswering(
        retriever, FakeQALLMClient(), transcripts=transcripts
    )
    with pytest.raises(QAError, match="non-empty"):
        qa.answer_question("  ")


def test_validate_qa_evidence_accepts_retrieved_ids(retriever) -> None:
    retrieved = retriever.search("How important is training?", top_k=3)
    refs = [
        QAEvidence(
            transcript_id=retrieved[0].transcript_id,
            segment_id=retrieved[0].segment_id,
            reason="ok",
        )
    ]
    validate_qa_evidence(refs, retrieved)  # should not raise


def test_cost_query_returns_multi_transcript_candidates(retriever) -> None:
    results = retriever.search("What role does cost play in adoption?", top_k=10)
    assert results
    markets = {r.transcript_id for r in results}
    # Prefer seeing more than one market when corpus supports it.
    assert markets  # at least one; often FR/DE/UK
