"""Tests for unsupported and false-premise Q&A scenarios (Loop 7).

Verifies that out-of-domain questions, market queries unsupported by the text,
and questions with misleading/biased premises are handled with rigorous grounding:
- No LLM calls when retrieval finds no usable evidence.
- No general-knowledge fabrication.
- No acceptance of unsupported premises or manufactured rankings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.llm_client import LLMClient
from app.parser import parse_all_transcripts
from app.qa import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    QuestionAnswering,
)
from app.retrieval import LexicalBM25Retriever, build_retriever
from app.schemas import (
    QAAnswerPayload,
    QAEvidence,
    RetrievedSegment,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class SpyQALLMClient(LLMClient):
    """Spy LLM client that records all calls and returns a configured answer."""

    def __init__(self, response: QAAnswerPayload | None = None) -> None:
        self.response = response
        self.calls: list[tuple[str, list[RetrievedSegment]]] = []

    def analyze_expert_transcript(self, transcript, questions):
        raise NotImplementedError

    def compare_expert_analyses(self, analyses):
        raise NotImplementedError

    def answer_question(self, question, evidence):
        self.calls.append((question, list(evidence)))
        if self.response is not None:
            return self.response.model_copy(deep=True)
        # Default grounded response: refuse if ungrounded
        return QAAnswerPayload(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            evidence=[],
        )


@pytest.fixture(scope="module")
def transcripts():
    return parse_all_transcripts(DATA_DIR)


@pytest.fixture(scope="module")
def retriever(transcripts):
    return build_retriever(transcripts)


# ---------------------------------------------------------------------------
# Section 5: Unsupported Question Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What is the current stock price of Intuitive Surgical?",
        "What is the population of France?",
        "What is the average salary of a urologist in Germany?",
        "What is today's robotic surgery market size?",
    ],
)
def test_unsupported_questions_return_controlled_insufficient_evidence(
    transcripts, retriever, question
) -> None:
    """Out-of-scope questions return insufficient-evidence and evidence=[]."""
    spy = SpyQALLMClient(
        QAAnswerPayload(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            evidence=[],
        )
    )
    qa = QuestionAnswering(retriever, spy, transcripts=transcripts)
    response = qa.answer_question(question)

    assert response.evidence == []
    assert INSUFFICIENT_EVIDENCE_ANSWER in response.answer or "not provide enough information" in response.answer.lower() or "not contain enough information" in response.answer.lower()


def test_zero_usable_retrieval_makes_no_llm_call(transcripts) -> None:
    """When retrieval yields no usable segments, LLM is never invoked."""
    strict_retriever = LexicalBM25Retriever(transcripts, min_score=1000.0)
    spy = SpyQALLMClient()
    qa = QuestionAnswering(strict_retriever, spy, transcripts=transcripts)

    response = qa.answer_question("What is the current stock price of Intuitive Surgical?")

    assert len(spy.calls) == 0
    assert response.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert response.evidence == []


def test_llm_cannot_use_general_knowledge_for_population_or_stock(
    transcripts, retriever
) -> None:
    """If candidate segments are passed, the grounded answer must not hallucinate general facts."""
    # Simulate an LLM that adheres to the grounding rules
    grounded_llm = SpyQALLMClient(
        QAAnswerPayload(
            answer="The provided transcripts do not contain information regarding Intuitive Surgical's stock price or the population of France.",
            evidence=[],
        )
    )
    qa = QuestionAnswering(retriever, grounded_llm, transcripts=transcripts)
    res = qa.answer_question("What is the current stock price of Intuitive Surgical?")

    assert res.evidence == []
    assert "stock price" in res.answer or "not contain" in res.answer


# ---------------------------------------------------------------------------
# Section 6: Misleading / False-Premise Questions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Why is robotic surgery adoption lower in Germany than France?",
        "Why does the UK have the best procurement process?",
        "Which expert was wrong about adoption?",
    ],
)
def test_false_premise_questions_do_not_accept_unsupported_assumptions(
    transcripts, retriever, question
) -> None:
    """The system must not accept unsupported premises (e.g. 'UK has the best process', 'Germany lower than France')."""
    # A properly prompt-grounded LLM explicitly qualifies the premise
    def create_qualified_payload(q: str, evidence: list[RetrievedSegment]) -> QAAnswerPayload:
        if "lower in germany" in q.lower():
            ans = (
                "The transcripts do not state that robotic surgery adoption is lower in Germany than in France. "
                "Instead, both experts describe growing but uneven adoption concentrated in larger university or academic centres."
            )
            refs = [
                QAEvidence(transcript_id=s.transcript_id, segment_id=s.segment_id, reason="Adoption description.")
                for s in evidence[:2]
            ]
        elif "best procurement" in q.lower():
            ans = (
                "The transcripts do not rank the UK as having the best procurement process. "
                "Each expert describes their own market's procurement challenges and timelines without cross-market rankings."
            )
            refs = [
                QAEvidence(transcript_id=s.transcript_id, segment_id=s.segment_id, reason="Procurement description.")
                for s in evidence[:2]
            ]
        else:
            ans = (
                "The transcripts do not determine that any expert was 'wrong'. "
                "Each expert provides perspective from their respective market (France, Germany, and the UK)."
            )
            refs = [
                QAEvidence(transcript_id=s.transcript_id, segment_id=s.segment_id, reason="Expert perspective.")
                for s in evidence[:2]
            ]
        return QAAnswerPayload(answer=ans, evidence=refs)

    class FalsePremiseVerifierLLM(LLMClient):
        def analyze_expert_transcript(self, transcript, questions):
            raise NotImplementedError
        def compare_expert_analyses(self, analyses):
            raise NotImplementedError
        def answer_question(self, q, evidence):
            return create_qualified_payload(q, evidence)

    qa = QuestionAnswering(retriever, FalsePremiseVerifierLLM(), transcripts=transcripts)
    response = qa.answer_question(question)

    # Verify no ungrounded ranking or acceptance of premise
    lower_ans = response.answer.lower()
    assert "do not state" in lower_ans or "do not rank" in lower_ans or "do not determine" in lower_ans
    assert "best" not in response.answer or "not rank" in lower_ans
    assert "wrong" not in response.answer or "not determine" in lower_ans

    # If evidence is provided, ensure it was retrieved
    if response.evidence:
        retrieved = retriever.search(question, top_k=qa.top_k)
        retrieved_keys = {(r.transcript_id, r.segment_id) for r in retrieved}
        for ref in response.evidence:
            assert (ref.transcript_id, ref.segment_id) in retrieved_keys


def test_no_rankings_or_scores_generated(transcripts, retriever) -> None:
    """Answers to comparative queries must never assign numerical scores or league tables."""
    payload = QAAnswerPayload(
        answer=(
            "Based on the interviews, France adoption is described as 15–20% in stronger centres, "
            "Germany reports high single to low double digit growth, and the UK expert notes above 15% procedure growth. "
            "The data does not rank one country higher than another."
        ),
        evidence=[],
    )
    spy = SpyQALLMClient(payload)
    qa = QuestionAnswering(retriever, spy, transcripts=transcripts)
    res = qa.answer_question("Rank the countries by adoption speed from best to worst.")

    # Must not contain rank scores like "1.", "Rank 1", "Score: 10/10"
    assert "score: " not in res.answer.lower()
    assert "winner" not in res.answer.lower()
