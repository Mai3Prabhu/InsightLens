"""Pydantic models for InsightLens transcript and analysis data."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Deterministic interview-guide question IDs (Loop 2 contract).
VALID_QUESTION_IDS: frozenset[str] = frozenset(
    {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
)


class TranscriptSegment(BaseModel):
    """A single timed utterance within a transcript."""

    segment_id: str = Field(
        ...,
        description="Deterministic identifier, e.g. FR_001",
        min_length=1,
    )
    timestamp: str = Field(
        ...,
        description="Original timestamp string from the source file, e.g. '00:18'",
        min_length=1,
    )
    speaker: str = Field(
        ...,
        description="Speaker name or label exactly as written in the source",
        min_length=1,
    )
    text: str = Field(
        ...,
        description="Exact utterance text from the source transcript",
        min_length=1,
    )

    @field_validator("segment_id", "timestamp", "speaker", "text")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class Transcript(BaseModel):
    """A fully parsed expert-call transcript."""

    transcript_id: str = Field(
        ...,
        description="Stable short ID derived from market, e.g. FR, DE, UK",
        min_length=1,
    )
    expert_name: str = Field(
        ...,
        description="Expert full name extracted from the header",
        min_length=1,
    )
    role: str = Field(
        ...,
        description="Expert role extracted from the header",
        min_length=1,
    )
    market: str = Field(
        ...,
        description="Market / geography extracted from the header",
        min_length=1,
    )
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Chronologically ordered transcript segments",
    )

    @field_validator("transcript_id", "expert_name", "role", "market")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


# ---------------------------------------------------------------------------
# Interview-guide and expert-analysis contracts (Loop 2)
# ---------------------------------------------------------------------------


class InterviewQuestion(BaseModel):
    """One canonical question from the interview guide."""

    question_id: str = Field(
        ...,
        description="Deterministic ID such as Q1",
        min_length=1,
    )
    question: str = Field(
        ...,
        description="Exact question wording from the interview guide (data/Guide.txt)",
        min_length=1,
    )

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned not in VALID_QUESTION_IDS:
            raise ValueError(
                f"question_id must be one of {sorted(VALID_QUESTION_IDS)}, "
                f"got {cleaned!r}"
            )
        return cleaned

    @field_validator("question")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty or whitespace-only")
        return cleaned


class EvidenceReference(BaseModel):
    """Pointer to a source transcript segment (no generated quotes)."""

    segment_id: str = Field(
        ...,
        description="Existing TranscriptSegment.segment_id, e.g. FR_004",
        min_length=1,
    )
    reason: str = Field(
        ...,
        description="Why this segment supports the answer",
        min_length=1,
    )

    @field_validator("segment_id", "reason")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class ExpertAnswer(BaseModel):
    """Structured answer to one interview question for a single expert."""

    question_id: str = Field(
        ...,
        description="Must match an InterviewQuestion.question_id (Q1–Q6)",
        min_length=1,
    )
    answer: str = Field(
        default="",
        description="Synthesized answer text (empty until LLM fills it)",
    )
    evidence: list[EvidenceReference] = Field(
        default_factory=list,
        description="References to transcript segment IDs supporting the answer",
    )

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned not in VALID_QUESTION_IDS:
            raise ValueError(
                f"question_id must be one of {sorted(VALID_QUESTION_IDS)}, "
                f"got {cleaned!r}"
            )
        return cleaned


class ExpertAnalysis(BaseModel):
    """Full structured analysis of one expert transcript against the guide."""

    transcript_id: str = Field(..., min_length=1)
    expert_name: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    market: str = Field(..., min_length=1)
    answers: list[ExpertAnswer] = Field(default_factory=list)

    @field_validator("transcript_id", "expert_name", "role", "market")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class ResolvedEvidence(BaseModel):
    """Source-of-truth evidence resolved from a TranscriptSegment."""

    question_id: str = Field(..., min_length=1)
    segment_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    speaker: str = Field(..., min_length=1)
    text: str = Field(
        ...,
        description="Exact original transcript text from TranscriptSegment",
        min_length=1,
    )
    reason: str = Field(
        ...,
        description="Reason from EvidenceReference explaining relevance",
        min_length=1,
    )


# ---------------------------------------------------------------------------
# Cross-expert comparison contracts (Loop 4)
# ---------------------------------------------------------------------------


class ComparisonEvidence(BaseModel):
    """Evidence pointer for a theme or difference (no generated quotes)."""

    transcript_id: str = Field(
        ...,
        description="Transcript ID among the compared analyses, e.g. FR",
        min_length=1,
    )
    segment_id: str = Field(
        ...,
        description="Existing TranscriptSegment.segment_id, e.g. FR_004",
        min_length=1,
    )
    reason: str = Field(
        ...,
        description="Why this segment supports the theme or difference",
        min_length=1,
    )

    @field_validator("transcript_id", "segment_id", "reason")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class ComparisonTheme(BaseModel):
    """A theme supported across one or more expert analyses."""

    theme_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    evidence: list[ComparisonEvidence] = Field(default_factory=list)

    @field_validator("theme_id", "title", "summary")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class ComparisonDifference(BaseModel):
    """A meaningful divergence between expert analyses."""

    difference_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    evidence: list[ComparisonEvidence] = Field(default_factory=list)

    @field_validator("difference_id", "title", "summary")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class CrossExpertComparison(BaseModel):
    """Structured cross-expert comparison of ExpertAnalysis objects."""

    model_config = {"extra": "forbid"}

    themes: list[ComparisonTheme] = Field(default_factory=list)
    differences: list[ComparisonDifference] = Field(default_factory=list)


class ResolvedComparisonEvidence(BaseModel):
    """Source-of-truth comparison evidence resolved from TranscriptSegment."""

    item_kind: str = Field(
        ...,
        description="'theme' or 'difference'",
        min_length=1,
    )
    item_id: str = Field(..., min_length=1)
    item_title: str = Field(..., min_length=1)
    transcript_id: str = Field(..., min_length=1)
    segment_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    speaker: str = Field(..., min_length=1)
    text: str = Field(
        ...,
        description="Exact original transcript text from TranscriptSegment",
        min_length=1,
    )
    reason: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Cross-transcript Q&A contracts (Loop 5)
# ---------------------------------------------------------------------------


class RetrievedSegment(BaseModel):
    """A transcript segment returned by the retrieval layer."""

    transcript_id: str = Field(..., min_length=1)
    segment_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    speaker: str = Field(..., min_length=1)
    text: str = Field(
        ...,
        description="Exact original transcript text (never rewritten)",
        min_length=1,
    )
    score: float = Field(..., description="Retrieval relevance score")
    question_context: str | None = Field(
        default=None,
        description="Exact text of the interviewer question this segment answers",
    )

    @field_validator("transcript_id", "segment_id", "timestamp", "speaker", "text")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class QAEvidence(BaseModel):
    """Evidence pointer for a Q&A answer (no generated quotes)."""

    transcript_id: str = Field(..., min_length=1)
    segment_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)

    @field_validator("transcript_id", "segment_id", "reason")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class QAResponse(BaseModel):
    """Grounded answer to a user question over expert transcripts."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    evidence: list[QAEvidence] = Field(default_factory=list)

    @field_validator("question", "answer")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty or whitespace-only")
        return cleaned


class QAAnswerPayload(BaseModel):
    """LLM structured payload for Q&A (question is supplied by the application)."""

    answer: str = Field(..., min_length=1)
    evidence: list[QAEvidence] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer must not be empty or whitespace-only")
        return cleaned


class ResolvedQAEvidence(BaseModel):
    """Source-of-truth Q&A evidence resolved from original transcript text."""

    transcript_id: str = Field(..., min_length=1)
    segment_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    speaker: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    market: str | None = Field(
        default=None,
        description="Optional market label from the source Transcript",
    )
