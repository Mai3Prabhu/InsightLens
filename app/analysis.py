"""Deterministic analysis helpers for InsightLens (no LLM in Loop 2).

Defines the structured output contract the future LLM will fill:
interview questions, empty ExpertAnalysis templates, and evidence validation
against source transcript segment IDs.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.schemas import (
    VALID_QUESTION_IDS,
    EvidenceReference,
    ExpertAnalysis,
    ExpertAnswer,
    InterviewQuestion,
    Transcript,
)

# Numbered guide lines: "1. How would you describe ..."
_QUESTION_LINE_RE = re.compile(r"^(\d+)\.\s+(.+)$")

_DEFAULT_GUIDE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "Guide.txt"
)


class AnalysisError(ValueError):
    """Raised when interview-guide loading or analysis validation fails."""


def load_interview_questions(
    path: str | Path | None = None,
) -> list[InterviewQuestion]:
    """Load the six interview questions from the interview guide (data/Guide.txt).

    Question wording is preserved exactly as written in the file.
    IDs are assigned deterministically as Q1 … Q6 in file order.

    Args:
        path: Optional path to the guide file. Defaults to
            ``data/Guide.txt`` at the project root.

    Returns:
        Exactly six :class:`~app.schemas.InterviewQuestion` objects.

    Raises:
        AnalysisError: If the file is missing or does not yield six questions.
    """
    guide_path = Path(path) if path is not None else _DEFAULT_GUIDE_PATH

    if not guide_path.exists():
        raise AnalysisError(f"Interview guide file does not exist: {guide_path}")
    if not guide_path.is_file():
        raise AnalysisError(f"Path is not a file: {guide_path}")

    try:
        raw = guide_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnalysisError(f"Unable to read interview guide: {guide_path}") from exc

    questions: list[InterviewQuestion] = []
    for line in raw.splitlines():
        stripped = line.strip()
        match = _QUESTION_LINE_RE.match(stripped)
        if not match:
            continue

        number = int(match.group(1))
        text = match.group(2).strip()
        if not text:
            raise AnalysisError(
                f"Empty question text for item {number} in {guide_path}"
            )

        question_id = f"Q{number}"
        if question_id not in VALID_QUESTION_IDS:
            raise AnalysisError(
                f"Unexpected question number {number} in {guide_path}; "
                f"expected 1–6"
            )

        questions.append(
            InterviewQuestion(question_id=question_id, question=text)
        )

    if len(questions) != 6:
        raise AnalysisError(
            f"Expected exactly 6 interview questions in {guide_path}, "
            f"found {len(questions)}"
        )

    expected_ids = [f"Q{i}" for i in range(1, 7)]
    actual_ids = [q.question_id for q in questions]
    if actual_ids != expected_ids:
        raise AnalysisError(
            f"Question IDs must be {expected_ids}, got {actual_ids}"
        )

    return questions


def build_analysis_template(
    transcript: Transcript,
    questions: list[InterviewQuestion] | None = None,
) -> ExpertAnalysis:
    """Build an empty ExpertAnalysis scaffold for a parsed transcript.

    Copies transcript metadata and creates one ExpertAnswer per interview
    question with empty ``answer`` text and an empty ``evidence`` list.
    Does not call an LLM.

    Args:
        transcript: Parsed :class:`~app.schemas.Transcript`.
        questions: Optional pre-loaded questions; loads from the guide if omitted.

    Returns:
        An :class:`~app.schemas.ExpertAnalysis` ready for future LLM filling.
    """
    if questions is None:
        questions = load_interview_questions()

    if len(questions) != 6:
        raise AnalysisError(
            f"build_analysis_template expects 6 questions, got {len(questions)}"
        )

    answers = [
        ExpertAnswer(question_id=q.question_id, answer="", evidence=[])
        for q in questions
    ]

    return ExpertAnalysis(
        transcript_id=transcript.transcript_id,
        expert_name=transcript.expert_name,
        role=transcript.role,
        market=transcript.market,
        answers=answers,
    )


def validate_evidence_references(
    analysis: ExpertAnalysis,
    transcript: Transcript,
) -> None:
    """Ensure every evidence segment_id exists on the given transcript.

    Raises:
        AnalysisError: If transcript IDs disagree, or any evidence
            ``segment_id`` is not present on ``transcript``.
    """
    if analysis.transcript_id != transcript.transcript_id:
        raise AnalysisError(
            f"Analysis transcript_id {analysis.transcript_id!r} does not "
            f"match transcript {transcript.transcript_id!r}"
        )

    valid_ids = {segment.segment_id for segment in transcript.segments}
    if not valid_ids:
        raise AnalysisError(
            f"Transcript {transcript.transcript_id!r} has no segments to "
            f"validate evidence against"
        )

    for answer in analysis.answers:
        seen_sids: set[str] = set()
        for ref in answer.evidence:
            if ref.segment_id in seen_sids:
                raise AnalysisError(
                    f"Duplicate evidence segment_id {ref.segment_id!r} for "
                    f"{answer.question_id} in transcript {transcript.transcript_id!r}."
                )
            seen_sids.add(ref.segment_id)
            if ref.segment_id not in valid_ids:
                raise AnalysisError(
                    f"Evidence segment_id {ref.segment_id!r} for "
                    f"{answer.question_id} does not exist in transcript "
                    f"{transcript.transcript_id!r}. "
                    f"Valid IDs include e.g. {sorted(valid_ids)[:3]}…"
                )


def attach_evidence(
    analysis: ExpertAnalysis,
    transcript: Transcript,
    question_id: str,
    evidence: list[EvidenceReference],
) -> ExpertAnalysis:
    """Return a copy of *analysis* with evidence set for *question_id*.

    Validates that every ``segment_id`` exists on *transcript* before applying.
    Intended for tests and future pipeline steps; does not call an LLM.
    """
    if question_id not in VALID_QUESTION_IDS:
        raise AnalysisError(
            f"question_id must be one of {sorted(VALID_QUESTION_IDS)}, "
            f"got {question_id!r}"
        )

    updated = analysis.model_copy(deep=True)
    found = False
    for answer in updated.answers:
        if answer.question_id == question_id:
            answer.evidence = list(evidence)
            found = True
            break

    if not found:
        raise AnalysisError(
            f"No ExpertAnswer with question_id {question_id!r} in analysis"
        )

    validate_evidence_references(updated, transcript)
    return updated


# ---------------------------------------------------------------------------
# CLI / quick demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Demonstrate guide loading + analysis template creation."""
    from app.parser import parse_transcript

    root = Path(__file__).resolve().parent.parent
    questions = load_interview_questions()
    transcript = parse_transcript(root / "data" / "Transcript_1_France.txt")
    analysis = build_analysis_template(transcript, questions=questions)

    print("Question IDs:", [q.question_id for q in questions])
    print("First question:", questions[0].question)
    print("Transcript metadata:")
    print(f"  transcript_id={analysis.transcript_id}")
    print(f"  expert_name={analysis.expert_name}")
    print(f"  role={analysis.role}")
    print(f"  market={analysis.market}")
    print(f"  answer slots={len(analysis.answers)}")


if __name__ == "__main__":
    _demo()
