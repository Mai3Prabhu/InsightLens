"""LLM-powered expert transcript analyzer for InsightLens (Loop 3).

Deterministic code remains responsible for validation and source-of-truth
evidence resolution. The LLM only reasons over the transcript and selects
segment IDs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.analysis import load_interview_questions, validate_evidence_references
from app.llm_client import LLMClient, LLMClientError, create_llm_client_from_env
from app.schemas import (
    VALID_QUESTION_IDS,
    ExpertAnalysis,
    InterviewQuestion,
    ResolvedEvidence,
    Transcript,
)


class AnalyzerError(ValueError):
    """Raised when LLM analysis output fails application validation."""


class ExpertAnalyzer:
    """Orchestrates LLM analysis + deterministic validation for one transcript."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def analyze(
        self,
        transcript: Transcript,
        questions: Sequence[InterviewQuestion] | None = None,
    ) -> ExpertAnalysis:
        """Analyze a transcript against the interview guide.

        Does not modify *transcript*. Returns a validated ExpertAnalysis.
        """
        # Snapshot identity so we can assert the caller's object was untouched.
        transcript_snapshot = transcript.model_dump()

        if questions is None:
            questions = load_interview_questions()
        questions = list(questions)

        if len(questions) != 6:
            raise AnalyzerError(
                f"Expected exactly 6 interview questions, got {len(questions)}"
            )

        try:
            raw_analysis = self._llm.analyze_expert_transcript(transcript, questions)
        except LLMClientError:
            raise
        except Exception as exc:
            raise AnalyzerError(
                f"LLM analysis failed unexpectedly: {type(exc).__name__}"
            ) from exc

        if not isinstance(raw_analysis, ExpertAnalysis):
            try:
                raw_analysis = ExpertAnalysis.model_validate(raw_analysis)
            except Exception as exc:
                raise AnalyzerError(
                    "LLM returned a payload that is not a valid ExpertAnalysis."
                ) from exc

        validated = self._validate_analysis(raw_analysis, transcript, questions)

        if transcript.model_dump() != transcript_snapshot:
            raise AnalyzerError(
                "Internal error: analyzer mutated the original Transcript."
            )

        return validated

    def _validate_analysis(
        self,
        analysis: ExpertAnalysis,
        transcript: Transcript,
        questions: Sequence[InterviewQuestion],
    ) -> ExpertAnalysis:
        self._validate_metadata(analysis, transcript)
        ordered_answers = self._validate_question_coverage(analysis, questions)

        # Rebuild with canonical metadata from the transcript (source of truth)
        # after verifying the LLM echoed it correctly.
        normalized = ExpertAnalysis(
            transcript_id=transcript.transcript_id,
            expert_name=transcript.expert_name,
            role=transcript.role,
            market=transcript.market,
            answers=ordered_answers,
        )

        try:
            validate_evidence_references(normalized, transcript)
        except Exception as exc:
            raise AnalyzerError(str(exc)) from exc

        return normalized

    @staticmethod
    def _validate_metadata(
        analysis: ExpertAnalysis, transcript: Transcript
    ) -> None:
        checks = (
            ("transcript_id", analysis.transcript_id, transcript.transcript_id),
            ("expert_name", analysis.expert_name, transcript.expert_name),
            ("role", analysis.role, transcript.role),
            ("market", analysis.market, transcript.market),
        )
        mismatches = [
            f"{field}: LLM={got!r}, transcript={expected!r}"
            for field, got, expected in checks
            if got != expected
        ]
        if mismatches:
            raise AnalyzerError(
                "LLM analysis metadata does not match the transcript: "
                + "; ".join(mismatches)
            )

    @staticmethod
    def _validate_question_coverage(
        analysis: ExpertAnalysis,
        questions: Sequence[InterviewQuestion],
    ) -> list:
        expected_ids = [q.question_id for q in questions]
        actual_ids = [a.question_id for a in analysis.answers]

        if not actual_ids:
            raise AnalyzerError("LLM analysis contains no answers.")

        duplicates = sorted({qid for qid in actual_ids if actual_ids.count(qid) > 1})
        if duplicates:
            raise AnalyzerError(
                f"LLM analysis contains duplicate question IDs: {duplicates}"
            )

        missing = [qid for qid in expected_ids if qid not in actual_ids]
        if missing:
            raise AnalyzerError(
                f"LLM analysis is missing question IDs: {missing}"
            )

        unexpected = [qid for qid in actual_ids if qid not in expected_ids]
        if unexpected:
            raise AnalyzerError(
                f"LLM analysis contains unexpected question IDs: {unexpected}"
            )

        if len(actual_ids) != len(expected_ids):
            raise AnalyzerError(
                f"LLM analysis must contain exactly {len(expected_ids)} answers, "
                f"got {len(actual_ids)}"
            )

        for answer in analysis.answers:
            if answer.question_id not in VALID_QUESTION_IDS:
                raise AnalyzerError(
                    f"Invalid question_id in LLM analysis: {answer.question_id!r}"
                )
            if not answer.answer or not answer.answer.strip():
                raise AnalyzerError(
                    f"Answer for question {answer.question_id} cannot be empty."
                )

        # Return answers in interview-guide order.
        by_id = {a.question_id: a for a in analysis.answers}
        return [by_id[qid] for qid in expected_ids]


def resolve_evidence(
    analysis: ExpertAnalysis,
    transcript: Transcript,
) -> list[ResolvedEvidence]:
    """Resolve EvidenceReference IDs into original transcript segments.

    Timestamp, speaker, and text always come from TranscriptSegment.
    Raises AnalyzerError if any segment_id is missing.
    """
    if analysis.transcript_id != transcript.transcript_id:
        raise AnalyzerError(
            f"Cannot resolve evidence: analysis transcript_id "
            f"{analysis.transcript_id!r} != transcript {transcript.transcript_id!r}"
        )

    by_id = {segment.segment_id: segment for segment in transcript.segments}
    resolved: list[ResolvedEvidence] = []

    for answer in analysis.answers:
        for ref in answer.evidence:
            segment = by_id.get(ref.segment_id)
            if segment is None:
                raise AnalyzerError(
                    f"Cannot resolve evidence: segment_id {ref.segment_id!r} "
                    f"for {answer.question_id} not found in transcript "
                    f"{transcript.transcript_id!r}"
                )
            resolved.append(
                ResolvedEvidence(
                    question_id=answer.question_id,
                    segment_id=segment.segment_id,
                    timestamp=segment.timestamp,
                    speaker=segment.speaker,
                    text=segment.text,
                    reason=ref.reason,
                )
            )

    return resolved


def analyze_transcript_file(
    path: str | Path,
    llm_client: LLMClient | None = None,
) -> tuple[Transcript, list[InterviewQuestion], ExpertAnalysis]:
    """Convenience helper: parse file, load questions, run analysis."""
    from app.parser import parse_transcript

    transcript = parse_transcript(path)
    questions = load_interview_questions()
    client = llm_client or create_llm_client_from_env()
    analysis = ExpertAnalyzer(client).analyze(transcript, questions)
    return transcript, questions, analysis


# ---------------------------------------------------------------------------
# CLI / real API demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Run a real LLM analysis for the France transcript when a key is set."""
    from app.parser import parse_transcript

    root = Path(__file__).resolve().parent.parent
    transcript_path = root / "data" / "Transcript_1_France.txt"

    questions = load_interview_questions()
    transcript = parse_transcript(transcript_path)

    try:
        client = create_llm_client_from_env()
    except LLMClientError as exc:
        print(f"Cannot run live demo: {exc}")
        print("Set GEMINI_API_KEY (and optionally GEMINI_MODEL) then retry.")
        raise SystemExit(1) from exc

    print(f"Model: {getattr(client, 'model', '(unknown)')}")
    print(f"Expert: {transcript.expert_name} | Market: {transcript.market}")
    print("=" * 72)

    analyzer = ExpertAnalyzer(client)
    analysis = analyzer.analyze(transcript, questions)
    resolved = resolve_evidence(analysis, transcript)
    resolved_by_q: dict[str, list[ResolvedEvidence]] = {}
    for item in resolved:
        resolved_by_q.setdefault(item.question_id, []).append(item)

    question_text = {q.question_id: q.question for q in questions}

    for answer in analysis.answers:
        print(f"\n{answer.question_id}: {question_text[answer.question_id]}")
        print("-" * 72)
        print("ANSWER")
        print(answer.answer)
        print()
        print("SOURCE EVIDENCE")
        sources = resolved_by_q.get(answer.question_id, [])
        if not sources:
            print("  (no evidence segments)")
            continue
        for src in sources:
            print(f"  [{src.segment_id}] {src.timestamp} | {src.speaker}")
            print(f"  reason: {src.reason}")
            print(f"  text: {src.text}")
            print()


if __name__ == "__main__":
    _demo()
