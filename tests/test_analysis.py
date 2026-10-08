"""Tests for InsightLens interview-guide loading and analysis templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis import (
    AnalysisError,
    attach_evidence,
    build_analysis_template,
    load_interview_questions,
    validate_evidence_references,
)
from app.parser import parse_transcript
from app.schemas import (
    EvidenceReference,
    ExpertAnalysis,
    ExpertAnswer,
    InterviewQuestion,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GUIDE = DATA_DIR / "Guide.txt"
FRANCE = DATA_DIR / "Transcript_1_France.txt"

# Exact wording from data/Guide.txt (do not rewrite).
EXPECTED_QUESTIONS = [
    ("Q1", "How would you describe current adoption of robotic surgery in your market?"),
    ("Q2", "What are the main barriers to adoption?"),
    ("Q3", "How important are hospital budgets and ROI in purchasing decisions?"),
    ("Q4", "How important are surgeon training and clinical outcomes?"),
    ("Q5", "What adoption trend do you expect over the next 3–5 years?"),
    ("Q6", "What is the typical hospital decision-making timeline for purchasing a new robotic system?"),
]


# ---------------------------------------------------------------------------
# Interview guide loading
# ---------------------------------------------------------------------------


def test_loads_exactly_six_questions() -> None:
    questions = load_interview_questions(GUIDE)
    assert len(questions) == 6
    assert all(isinstance(q, InterviewQuestion) for q in questions)


def test_question_ids_are_q1_through_q6() -> None:
    questions = load_interview_questions(GUIDE)
    assert [q.question_id for q in questions] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
    ]


def test_question_text_matches_interview_guide() -> None:
    questions = load_interview_questions(GUIDE)
    actual = [(q.question_id, q.question) for q in questions]
    assert actual == EXPECTED_QUESTIONS

    # Spot-check the en-dash in Q5 is preserved from the source file.
    assert "3–5" in questions[4].question
    assert "3-5" not in questions[4].question


def test_default_path_loads_guide() -> None:
    questions = load_interview_questions()
    assert len(questions) == 6
    assert questions[0].question_id == "Q1"


def test_missing_guide_raises() -> None:
    with pytest.raises(AnalysisError, match="does not exist"):
        load_interview_questions(DATA_DIR / "missing_guide.txt")


# ---------------------------------------------------------------------------
# Analysis template
# ---------------------------------------------------------------------------


def test_build_analysis_template_creates_expert_analysis() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = build_analysis_template(transcript)
    assert isinstance(analysis, ExpertAnalysis)
    assert analysis.transcript_id == "FR"
    assert analysis.expert_name == "Dr. Jean Martin"
    assert analysis.role == "Head of Urology"
    assert analysis.market == "France"


def test_template_contains_all_six_questions() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = build_analysis_template(transcript)
    assert len(analysis.answers) == 6
    assert [a.question_id for a in analysis.answers] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
    ]


def test_template_answers_initially_empty() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = build_analysis_template(transcript)
    for answer in analysis.answers:
        assert answer.answer == ""
        assert isinstance(answer, ExpertAnswer)


def test_template_evidence_initially_empty() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = build_analysis_template(transcript)
    for answer in analysis.answers:
        assert answer.evidence == []


# ---------------------------------------------------------------------------
# Evidence validation
# ---------------------------------------------------------------------------


def test_valid_evidence_segment_ids_pass_validation() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = build_analysis_template(transcript)

    analysis.answers[5].evidence = [
        EvidenceReference(
            segment_id="FR_014",
            reason="Directly addresses the hospital purchasing timeline.",
        ),
        EvidenceReference(
            segment_id="FR_013",
            reason="Interviewer question prompting the timeline response.",
        ),
    ]

    # Should not raise
    validate_evidence_references(analysis, transcript)


def test_invalid_evidence_segment_ids_raise() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = build_analysis_template(transcript)
    analysis.answers[0].evidence = [
        EvidenceReference(
            segment_id="FR_999",
            reason="This segment does not exist.",
        )
    ]

    with pytest.raises(AnalysisError, match="FR_999"):
        validate_evidence_references(analysis, transcript)


def test_attach_evidence_validates_before_returning() -> None:
    transcript = parse_transcript(FRANCE)
    analysis = build_analysis_template(transcript)

    updated = attach_evidence(
        analysis,
        transcript,
        "Q6",
        [
            EvidenceReference(
                segment_id="FR_014",
                reason="Directly addresses the hospital purchasing timeline.",
            )
        ],
    )
    assert updated.answers[5].evidence[0].segment_id == "FR_014"
    # Original template unchanged (deep copy)
    assert analysis.answers[5].evidence == []

    with pytest.raises(AnalysisError, match="does not exist"):
        attach_evidence(
            analysis,
            transcript,
            "Q1",
            [
                EvidenceReference(
                    segment_id="DE_001",
                    reason="Wrong transcript prefix.",
                )
            ],
        )


def test_expert_answer_rejects_invalid_question_id() -> None:
    with pytest.raises(Exception):
        ExpertAnswer(question_id="Q99", answer="", evidence=[])


def test_evidence_does_not_store_quotes() -> None:
    """EvidenceReference only carries segment_id + reason (no quote field)."""
    ref = EvidenceReference(segment_id="FR_004", reason="Addresses budgets.")
    assert set(ref.model_dump().keys()) == {"segment_id", "reason"}
