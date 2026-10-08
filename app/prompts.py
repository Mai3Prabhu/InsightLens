"""Prompt templates for InsightLens expert analysis, comparison, and Q&A."""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas import (
    ExpertAnalysis,
    InterviewQuestion,
    RetrievedSegment,
    Transcript,
    TranscriptSegment,
)

SYSTEM_INSTRUCTIONS = """\
You are analyzing an expert interview transcript for InsightLens.

Rules you must follow:
- Use ONLY information explicitly present in the provided transcript.
- Do not use outside knowledge.
- Do not infer facts that are not supported by the transcript.
- If a question cannot be answered from the transcript, set the answer to \
exactly: "The transcript does not provide enough information to answer this question." \
and set evidence to an empty list.
- For every answer that is supported by the transcript, provide the IDs of the \
transcript segments that support the answer.
- Do not generate quotes.
- Do not invent or create segment IDs.
- Only use segment IDs that appear in the provided transcript.
- Answers should be concise but informative.
- Do not duplicate the full transcript text inside your response.
- Evidence items must contain only segment_id and reason (never a quote field).

Transcript structure:
- Metadata fields: transcript_id, expert_name, role, market.
- Segments: each has segment_id, timestamp, speaker, and text.
- Segment IDs look like FR_001, DE_002, UK_003 and are the only valid evidence keys.
- Copy segment_id values exactly as shown (e.g. "FR_004"); never append the \
timestamp, speaker, or transcript_id to a segment_id.

You must answer ALL interview questions provided (typically Q1 through Q6).
Return structured output matching the ExpertAnalysis schema:
- transcript_id, expert_name, role, market (copy exactly from the provided metadata)
- answers: one ExpertAnswer per question with question_id, answer, and evidence[]
"""

COMPARISON_SYSTEM_INSTRUCTIONS = """\
You are comparing structured analyses of expert interviews for InsightLens.

Rules you must follow:
- Use ONLY the supplied expert analyses.
- Do not use outside knowledge.
- Do not re-analyze raw transcripts; reason from the structured answers and \
their evidence segment IDs.
- Identify themes that are supported by multiple experts when the supplied \
answers show genuine agreement.
- Identify meaningful differences where experts describe the same topic \
differently (perspective, experience, market conditions, or reported values).
- Do not manufacture disagreements.
- Do not treat differences in wording alone as disagreements.
- Distinguish genuine agreement, genuine divergence, and differences in emphasis.
- Do not rank countries or experts.
- Do not determine which expert is "correct".
- Do not assign scores. The comparison is descriptive only.
- For every theme and difference, provide evidence references using existing \
transcript_id + segment_id pairs from the supplied analyses.
- Do not generate quotes.
- Do not invent segment IDs or transcript IDs.
- If there is insufficient evidence for a theme or difference, do not include it.
- Empty themes and/or differences lists are allowed when evidence is insufficient.
- Do not duplicate transcript text in your response.
- Evidence items must contain only transcript_id, segment_id, and reason.

Return structured output matching CrossExpertComparison:
- themes: list of ComparisonTheme (theme_id, title, summary, evidence[])
- differences: list of ComparisonDifference (difference_id, title, summary, evidence[])
"""


def format_segments_for_prompt(segments: Sequence[TranscriptSegment]) -> str:
    """Format transcript segments for inclusion in the user prompt.

    Currently formats the full segment list. Later this helper can receive a
    retrieved subset instead of the entire transcript.
    """
    lines: list[str] = []
    for segment in segments:
        lines.append(
            f"segment_id={segment.segment_id} | timestamp={segment.timestamp} | "
            f"speaker={segment.speaker}\n    {segment.text}"
        )
    return "\n".join(lines)


def format_questions_for_prompt(questions: Sequence[InterviewQuestion]) -> str:
    """Format interview questions for the user prompt."""
    lines = [f"{q.question_id}: {q.question}" for q in questions]
    return "\n".join(lines)


def select_segments_for_analysis(transcript: Transcript) -> list[TranscriptSegment]:
    """Choose which segments to send to the LLM.

    Loop 3 sends the complete transcript. This function is the extension point
    for future retrieval of relevant segments only (no vector DB yet).
    """
    return list(transcript.segments)


def build_analysis_user_prompt(
    transcript: Transcript,
    questions: Sequence[InterviewQuestion],
    segments: Sequence[TranscriptSegment] | None = None,
) -> str:
    """Build the user message containing metadata, segments, and questions."""
    if segments is None:
        segments = select_segments_for_analysis(transcript)

    return (
        "Analyze the following expert interview transcript against the "
        "interview questions.\n\n"
        "=== TRANSCRIPT METADATA ===\n"
        f"transcript_id: {transcript.transcript_id}\n"
        f"expert_name: {transcript.expert_name}\n"
        f"role: {transcript.role}\n"
        f"market: {transcript.market}\n\n"
        "=== TRANSCRIPT SEGMENTS ===\n"
        f"{format_segments_for_prompt(segments)}\n\n"
        "=== INTERVIEW QUESTIONS ===\n"
        f"{format_questions_for_prompt(questions)}\n\n"
        "Return one answer for each question. Use only the segment IDs shown above "
        "as evidence. Do not invent segment IDs or quotes."
    )


def build_analysis_messages(
    transcript: Transcript,
    questions: Sequence[InterviewQuestion],
    segments: Sequence[TranscriptSegment] | None = None,
) -> list[dict[str, str]]:
    """Build system/user messages for structured expert analysis."""
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": build_analysis_user_prompt(transcript, questions, segments),
        },
    ]


def format_expert_analysis_for_prompt(analysis: ExpertAnalysis) -> str:
    """Serialize one ExpertAnalysis for the comparison prompt (no raw transcript)."""
    lines = [
        f"transcript_id: {analysis.transcript_id}",
        f"expert_name: {analysis.expert_name}",
        f"role: {analysis.role}",
        f"market: {analysis.market}",
        "answers:",
    ]
    for answer in analysis.answers:
        lines.append(f"  {answer.question_id}: {answer.answer}")
        if not answer.evidence:
            lines.append("    evidence: (none)")
            continue
        for ref in answer.evidence:
            lines.append(
                f"    - segment_id={ref.segment_id}; reason={ref.reason}"
            )
    return "\n".join(lines)


def build_comparison_user_prompt(analyses: Sequence[ExpertAnalysis]) -> str:
    """Build the user message from structured ExpertAnalysis objects only."""
    blocks = [
        "Compare the following structured expert analyses.\n",
        "Identify common themes and meaningful differences.\n",
        "Use only transcript_id and segment_id values that appear below.\n",
    ]
    for analysis in analyses:
        blocks.append(f"=== EXPERT ANALYSIS ({analysis.transcript_id}) ===")
        blocks.append(format_expert_analysis_for_prompt(analysis))
        blocks.append("")
    blocks.append(
        "Return CrossExpertComparison with themes and differences. "
        "Do not invent segment IDs, transcript IDs, or quotes. "
        "Do not rank experts."
    )
    return "\n".join(blocks)


def build_comparison_messages(
    analyses: Sequence[ExpertAnalysis],
) -> list[dict[str, str]]:
    """Build system/user messages for cross-expert comparison."""
    return [
        {"role": "system", "content": COMPARISON_SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": build_comparison_user_prompt(analyses)},
    ]


QA_SYSTEM_INSTRUCTIONS = """\
You are answering a user question about expert interview transcripts for InsightLens.

Rules you must follow:
- Answer the user's question using only the supplied transcript evidence.
- If the evidence does not contain enough information, state that the transcripts \
do not provide enough information to answer this question, and return an empty \
evidence list.
- Do not use outside knowledge.
- Do not invent facts.
- Do not create transcript segment IDs.
- Only reference segment IDs provided in the evidence.
- Do not generate quotes in the evidence objects.
- Evidence items must contain only transcript_id, segment_id, and reason.
- Copy transcript_id (e.g. "DE") and segment_id (e.g. "DE_004") exactly as \
shown, as two separate fields. Never combine them or append a timestamp.
- Prefer concise, informative answers grounded in the supplied segments.
- The evidence was selected by keyword search, so some segments may be only \
loosely related. Ignore segments that do not actually address the question.
- Questions about facts the interviews do not discuss (e.g. stock prices, \
population, salaries, market size, events not described by the experts) must be \
answered with the insufficient-information statement, even if a segment \
mentions a related word.
- Do not accept the premise of a question uncritically. If the question assumes \
something the evidence does not establish (e.g. that one market is "worse", \
"best", or has "the highest" adoption, or that an expert is "wrong"), say \
clearly that the transcripts do not establish that premise, then summarise \
what the experts actually said.
- Do not rank countries, markets, or experts, and do not judge which expert \
is correct. Reported figures may be stated side by side as each expert gave \
them, attributed to that expert.
- Attribute views to the relevant expert or market (France, Germany, UK).

Return structured output matching QAAnswerPayload:
- answer: string
- evidence: list of {transcript_id, segment_id, reason}
"""


def format_retrieved_evidence_for_prompt(segments: Sequence[RetrievedSegment]) -> str:
    """Format retrieved segments for the Q&A prompt (exact source text)."""
    lines: list[str] = []
    for seg in segments:
        lines.append(
            f"transcript_id={seg.transcript_id} | segment_id={seg.segment_id} | "
            f"timestamp={seg.timestamp} | speaker={seg.speaker}"
        )
        if seg.question_context:
            lines.append(f"    (answering interviewer question: {seg.question_context})")
        lines.append(f"    {seg.text}")
    return "\n".join(lines)


def build_qa_user_prompt(
    question: str,
    segments: Sequence[RetrievedSegment],
) -> str:
    """Build the user message for grounded Q&A over retrieved evidence only."""
    return (
        "Answer the following question using ONLY the transcript evidence below.\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        "=== RETRIEVED TRANSCRIPT EVIDENCE ===\n"
        f"{format_retrieved_evidence_for_prompt(segments)}\n\n"
        "Return QAAnswerPayload. Use only transcript_id/segment_id values shown above. "
        "Do not invent IDs or use outside knowledge."
    )


def build_qa_messages(
    question: str,
    segments: Sequence[RetrievedSegment],
) -> list[dict[str, str]]:
    """Build system/user messages for grounded transcript Q&A."""
    return [
        {"role": "system", "content": QA_SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": build_qa_user_prompt(question, segments)},
    ]
