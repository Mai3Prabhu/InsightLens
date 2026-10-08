"""Presentation-layer data helpers for InsightLens Streamlit UI.

No Streamlit imports here — keeps orchestration testable and separate from widgets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.analysis import load_interview_questions
from app.analyzer import ExpertAnalyzer, resolve_evidence
from app.comparison import (
    CrossExpertComparer,
    build_demo_analyses,
    build_demo_comparison,
    resolve_comparison_evidence,
)
from app.llm_client import (
    DEFAULT_GEMINI_MODEL,
    LLMClient,
    LLMClientError,
    create_llm_client_from_env,
)
from app.parser import parse_all_transcripts
from app.qa import (
    QuestionAnswering,
    build_demo_qa_payload,
    resolve_qa_evidence,
)
from app.retrieval import build_retriever
from app.schemas import (
    CrossExpertComparison,
    ExpertAnalysis,
    InterviewQuestion,
    QAResponse,
    ResolvedComparisonEvidence,
    ResolvedEvidence,
    ResolvedQAEvidence,
    Transcript,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

EXPERT_CHOICES: tuple[tuple[str, str], ...] = (
    ("FR", "France — Dr. Jean Martin"),
    ("DE", "Germany — Anna Keller"),
    ("UK", "UK — Dr. Emily Carter"),
)


@dataclass(frozen=True)
class AppMode:
    """Whether the UI is using live Gemini or deterministic demo responses."""

    live: bool
    label: str


class DemoLLMClient(LLMClient):
    """Deterministic mock LLM used by the UI when no API key / demo mode."""

    def __init__(self, transcripts: list[Transcript]) -> None:
        analyses = build_demo_analyses(transcripts)
        self._by_id = {a.transcript_id: a for a in analyses}

    def analyze_expert_transcript(self, transcript, questions):
        analysis = self._by_id.get(transcript.transcript_id)
        if analysis is None:
            raise LLMClientError(
                f"No demo analysis available for {transcript.transcript_id!r}."
            )
        return analysis.model_copy(deep=True)

    def compare_expert_analyses(self, analyses):
        return build_demo_comparison()

    def answer_question(self, question, evidence):
        return build_demo_qa_payload(question, evidence)


def project_root() -> Path:
    return ROOT


def api_key_configured() -> bool:
    """True when GEMINI_API_KEY is available (env or .env via dotenv)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def configured_model() -> str:
    """Model name used in live mode (never exposes the API key)."""
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL


def resolve_app_mode(prefer_live: bool) -> AppMode:
    """Choose demo vs live mode. Live requires an API key."""
    if prefer_live and api_key_configured():
        return AppMode(live=True, label=f"Live mode — Gemini ({configured_model()})")
    if prefer_live and not api_key_configured():
        return AppMode(
            live=False,
            label="Demo mode — API key missing; using deterministic mock responses",
        )
    return AppMode(
        live=False,
        label="Demo mode — using deterministic mock AI responses",
    )


def load_transcripts() -> list[Transcript]:
    return parse_all_transcripts(DATA_DIR)


def load_questions() -> list[InterviewQuestion]:
    return load_interview_questions()


def transcripts_by_id(transcripts: list[Transcript]) -> dict[str, Transcript]:
    return {t.transcript_id: t for t in transcripts}


def build_llm_client(mode: AppMode, transcripts: list[Transcript]) -> LLMClient:
    if mode.live:
        return create_llm_client_from_env()
    return DemoLLMClient(transcripts)


def compute_expert_analyses(
    transcripts: list[Transcript],
    questions: list[InterviewQuestion],
    client: LLMClient,
) -> dict[str, ExpertAnalysis]:
    """Run ExpertAnalyzer for each transcript (mock or live)."""
    analyzer = ExpertAnalyzer(client)
    results: dict[str, ExpertAnalysis] = {}
    for transcript in transcripts:
        results[transcript.transcript_id] = analyzer.analyze(transcript, questions)
    return results


def compute_comparison(
    analyses: dict[str, ExpertAnalysis],
    transcripts: list[Transcript],
    client: LLMClient,
) -> CrossExpertComparison:
    ordered = [analyses[tid] for tid in ("FR", "DE", "UK") if tid in analyses]
    return CrossExpertComparer(client).compare(ordered, transcripts)


def resolve_analysis_evidence(
    analysis: ExpertAnalysis,
    transcript: Transcript,
) -> dict[str, list[ResolvedEvidence]]:
    resolved = resolve_evidence(analysis, transcript)
    by_q: dict[str, list[ResolvedEvidence]] = {}
    for item in resolved:
        by_q.setdefault(item.question_id, []).append(item)
    return by_q


def resolve_comparison_sources(
    comparison: CrossExpertComparison,
    transcripts: list[Transcript],
) -> list[ResolvedComparisonEvidence]:
    return resolve_comparison_evidence(comparison, transcripts)


def run_qa(
    question: str,
    transcripts: list[Transcript],
    client: LLMClient,
) -> tuple[QAResponse, list[ResolvedQAEvidence]]:
    retriever = build_retriever(transcripts)
    service = QuestionAnswering(retriever, client, transcripts=transcripts)
    response = service.answer_question(question)
    retrieved = retriever.search(question, top_k=service.top_k)
    sources = resolve_qa_evidence(
        response, retrieved=retrieved, transcripts=transcripts
    )
    return response, sources


def expert_label(transcript_id: str) -> str:
    for tid, label in EXPERT_CHOICES:
        if tid == transcript_id:
            return label
    return transcript_id
