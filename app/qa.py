"""User-driven cross-transcript Q&A for InsightLens (Loop 5).

Retrieve first, reason second: the LLM only sees retrieved transcript segments.
Deterministic code validates evidence IDs and resolves original source text.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.llm_client import (
    LLMClient,
    LLMClientError,
    _load_dotenv_if_available,
    create_llm_client_from_env,
)
from app.retrieval import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    RetrievalError,
    Retriever,
    build_retriever,
)
from app.schemas import (
    QAAnswerPayload,
    QAEvidence,
    QAResponse,
    ResolvedQAEvidence,
    RetrievedSegment,
    Transcript,
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "The provided transcripts do not contain enough information to answer "
    "this question."
)

# Guards the LLM prompt against pasted documents or accidental huge inputs.
MAX_QUESTION_CHARS = 1000


class QAError(ValueError):
    """Raised when Q&A inputs or LLM evidence fail validation."""


class QuestionAnswering:
    """Grounded Q&A service over a Retriever + LLMClient."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float | None = None,
        transcripts: Sequence[Transcript] | Mapping[str, Transcript] | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client
        self.top_k = top_k
        # Prefer explicit override; otherwise use retriever.min_score when present.
        if min_score is not None:
            self.min_score = float(min_score)
        else:
            self.min_score = float(getattr(retriever, "min_score", DEFAULT_MIN_SCORE))
        self._transcripts = _index_transcripts(transcripts) if transcripts else {}

    def answer_question(self, question: str) -> QAResponse:
        """Answer a user question using retrieve → reason → validate."""
        cleaned = _validate_question(question)

        try:
            retrieved = self._retriever.search(cleaned, top_k=self.top_k)
        except RetrievalError as exc:
            raise QAError(str(exc)) from exc

        usable = [seg for seg in retrieved if seg.score > self.min_score]
        if not usable:
            return QAResponse(
                question=cleaned,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                evidence=[],
            )

        try:
            payload = self._llm.answer_question(cleaned, usable)
        except LLMClientError:
            raise
        except Exception as exc:
            raise QAError(
                f"LLM Q&A failed unexpectedly: {type(exc).__name__}"
            ) from exc

        if not isinstance(payload, QAAnswerPayload):
            try:
                payload = QAAnswerPayload.model_validate(payload)
            except Exception as exc:
                raise QAError(
                    "LLM returned a payload that is not a valid QAAnswerPayload."
                ) from exc

        validate_qa_evidence(payload.evidence, usable)

        return QAResponse(
            question=cleaned,
            answer=payload.answer,
            evidence=list(payload.evidence),
        )


def validate_qa_evidence(
    evidence: Sequence[QAEvidence],
    allowed: Sequence[RetrievedSegment],
) -> None:
    """Ensure every QAEvidence ID was present in the retrieved evidence set."""
    allowed_keys = {(s.transcript_id, s.segment_id) for s in allowed}
    allowed_segment_ids = {s.segment_id for s in allowed}

    seen_keys: set[tuple[str, str]] = set()
    for ref in evidence:
        key = (ref.transcript_id, ref.segment_id)
        if key in seen_keys:
            raise QAError(
                f"Duplicate QA evidence reference: {ref.transcript_id}/{ref.segment_id}."
            )
        seen_keys.add(key)
        if key not in allowed_keys:
            raise QAError(
                f"QA evidence references segment {ref.transcript_id}/{ref.segment_id} "
                f"that was not in the retrieved evidence. "
                f"Allowed segment_ids: {sorted(allowed_segment_ids)}"
            )


def resolve_qa_evidence(
    response: QAResponse,
    *,
    retrieved: Sequence[RetrievedSegment] | None = None,
    transcripts: Sequence[Transcript] | Mapping[str, Transcript] | None = None,
) -> list[ResolvedQAEvidence]:
    """Resolve QA evidence IDs to original timestamp/speaker/text.

    Prefer *retrieved* segments when provided (exact retrieval payload).
    Otherwise look up segments from *transcripts*.
    """
    segment_lookup: dict[tuple[str, str], RetrievedSegment | object] = {}
    market_by_tid: dict[str, str] = {}

    if retrieved:
        for seg in retrieved:
            segment_lookup[(seg.transcript_id, seg.segment_id)] = seg

    if transcripts is not None:
        tx_map = _index_transcripts(transcripts)
        for tid, tx in tx_map.items():
            market_by_tid[tid] = tx.market
            for seg in tx.segments:
                key = (tid, seg.segment_id)
                if key not in segment_lookup:
                    segment_lookup[key] = seg

    resolved: list[ResolvedQAEvidence] = []
    for ref in response.evidence:
        key = (ref.transcript_id, ref.segment_id)
        source = segment_lookup.get(key)
        if source is None:
            raise QAError(
                f"Cannot resolve QA evidence: {ref.transcript_id}/{ref.segment_id} "
                f"not found in retrieved segments or transcripts."
            )
        resolved.append(
            ResolvedQAEvidence(
                transcript_id=ref.transcript_id,
                segment_id=getattr(source, "segment_id"),
                timestamp=getattr(source, "timestamp"),
                speaker=getattr(source, "speaker"),
                text=getattr(source, "text"),
                reason=ref.reason,
                market=market_by_tid.get(ref.transcript_id),
            )
        )
    return resolved


def _validate_question(question: str) -> str:
    if question is None or not str(question).strip():
        raise QAError("Question must be a non-empty string.")
    cleaned = str(question).strip()
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise QAError(
            f"Question is too long ({len(cleaned)} characters). "
            f"Please keep it under {MAX_QUESTION_CHARS} characters."
        )
    return cleaned


def _index_transcripts(
    transcripts: Sequence[Transcript] | Mapping[str, Transcript],
) -> dict[str, Transcript]:
    if isinstance(transcripts, Mapping):
        return dict(transcripts)
    indexed: dict[str, Transcript] = {}
    for tx in transcripts:
        if tx.transcript_id in indexed:
            raise QAError(f"Duplicate transcript_id: {tx.transcript_id!r}")
        indexed[tx.transcript_id] = tx
    return indexed


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def build_demo_qa_payload(question: str, evidence: Sequence[RetrievedSegment]) -> QAAnswerPayload:
    """Deterministic grounded payload for offline demos/tests."""
    if not evidence:
        return QAAnswerPayload(answer=INSUFFICIENT_EVIDENCE_ANSWER, evidence=[])

    # Prefer segments that look like expert answers (not interviewer prompts).
    expertish = [s for s in evidence if s.speaker.lower() != "interviewer"]
    chosen = (expertish or list(evidence))[:3]
    refs = [
        QAEvidence(
            transcript_id=s.transcript_id,
            segment_id=s.segment_id,
            reason="Retrieved segment relevant to the user question.",
        )
        for s in chosen
    ]
    summary_bits = []
    for s in chosen:
        snippet = s.text if len(s.text) <= 120 else s.text[:117] + "..."
        summary_bits.append(f"{s.transcript_id}: {snippet}")
    answer = (
        "Based on the retrieved expert interviews: "
        + " | ".join(summary_bits)
    )
    return QAAnswerPayload(answer=answer, evidence=refs)


def _print_qa(
    response: QAResponse,
    resolved: list[ResolvedQAEvidence],
) -> None:
    print("QUESTION")
    print(response.question)
    print()
    print("ANSWER")
    print(response.answer)
    print()
    print("SOURCES")
    if not resolved:
        print("  (no sources)")
        return
    for src in resolved:
        market = src.market or src.transcript_id
        print(f"  market: {market}")
        print(f"  [{src.segment_id}] {src.timestamp} | {src.speaker}")
        print(f"  reason: {src.reason}")
        print(f"  text: {src.text}")
        print()


def _demo() -> None:
    """CLI demo: retrieve + answer a sample question (mocked LLM by default)."""
    from app.parser import parse_all_transcripts

    root = Path(__file__).resolve().parent.parent
    transcripts = parse_all_transcripts(root / "data")
    retriever = build_retriever(transcripts)
    sample_question = (
        "What are the biggest barriers to robotic surgery adoption?"
    )

    _load_dotenv_if_available()
    use_live = bool(os.getenv("GEMINI_API_KEY")) and os.getenv(
        "INSIGHTLENS_LIVE_QA", ""
    ).lower() in {"1", "true", "yes"}

    if use_live:
        print("Running LIVE LLM Q&A (INSIGHTLENS_LIVE_QA enabled)...")
        try:
            client: LLMClient = create_llm_client_from_env()
        except LLMClientError as exc:
            print(f"Live Q&A unavailable: {exc}")
            print("Falling back to mocked LLM.")
            use_live = False
            client = _FakeQAClient()
    else:
        print(
            "Using mocked LLM Q&A "
            "(set GEMINI_API_KEY and INSIGHTLENS_LIVE_QA=1 for live mode)."
        )
        client = _FakeQAClient()

    qa = QuestionAnswering(retriever, client, transcripts=transcripts)
    # Capture retrieved set for resolution by re-searching with same settings.
    retrieved = retriever.search(sample_question, top_k=qa.top_k)
    response = qa.answer_question(sample_question)
    resolved = resolve_qa_evidence(
        response, retrieved=retrieved, transcripts=transcripts
    )
    _print_qa(response, resolved)


class _FakeQAClient(LLMClient):
    def analyze_expert_transcript(self, transcript, questions):
        raise NotImplementedError

    def compare_expert_analyses(self, analyses):
        raise NotImplementedError

    def answer_question(self, question, evidence):
        return build_demo_qa_payload(question, evidence)


if __name__ == "__main__":
    _demo()
