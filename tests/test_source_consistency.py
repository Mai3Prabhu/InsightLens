"""Source consistency audit tests for Loop 7.

Validates that no transformation, truncation, paraphrasing, or normalization
has altered the source evidence displayed to users across:
1. Raw transcript files on disk
2. Parsed TranscriptSegment objects
3. ResolvedEvidence (Expert Analysis)
4. ResolvedComparisonEvidence (Comparison)
5. ResolvedQAEvidence (Q&A)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.analyzer import resolve_evidence
from app.comparison import (
    build_demo_analyses,
    build_demo_comparison,
    resolve_comparison_evidence,
)
from app.parser import parse_all_transcripts, parse_transcript
from app.qa import resolve_qa_evidence
from app.retrieval import build_retriever
from app.schemas import (
    QAAnswerPayload,
    QAEvidence,
    QAResponse,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRANSCRIPT_FILES = {
    "FR": DATA_DIR / "Transcript_1_France.txt",
    "DE": DATA_DIR / "Transcript_2_Germany.txt",
    "UK": DATA_DIR / "Transcript_3_UK.txt",
}


def _extract_raw_turns(file_path: Path) -> list[tuple[str, str, str]]:
    """Independent simple parser to extract (timestamp, speaker, text) directly from raw text."""
    raw = file_path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    turns: list[tuple[str, str, str]] = []
    i = 0
    ts_re = re.compile(r"^(\d{1,2}:\d{2})$")
    speaker_re = re.compile(r"^([^:]+):\s*(.*)$")

    while i < len(lines):
        line = lines[i].strip()
        ts_match = ts_re.match(line)
        if ts_match:
            ts = ts_match.group(1)
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                sp_match = speaker_re.match(lines[i].strip())
                if sp_match:
                    turns.append((ts, sp_match.group(1).strip(), sp_match.group(2).strip()))
        i += 1
    return turns


@pytest.fixture(scope="module")
def all_transcripts():
    return parse_all_transcripts(DATA_DIR)


# ---------------------------------------------------------------------------
# Section 13: Exact Character-for-Character Source Consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("market_id", ["FR", "DE", "UK"])
def test_parsed_segments_match_raw_file_identically(market_id) -> None:
    """Each parsed TranscriptSegment must match raw file text character-for-character."""
    file_path = TRANSCRIPT_FILES[market_id]
    raw_turns = _extract_raw_turns(file_path)
    parsed = parse_transcript(file_path)

    assert len(parsed.segments) == len(raw_turns) == 14

    for idx, (raw_ts, raw_speaker, raw_text) in enumerate(raw_turns):
        seg = parsed.segments[idx]
        assert seg.timestamp == raw_ts
        assert seg.speaker == raw_speaker
        assert seg.text == raw_text
        # Verify length and exact bytes
        assert len(seg.text) == len(raw_text)
        assert seg.text.encode("utf-8") == raw_text.encode("utf-8")


def test_resolved_expert_analysis_evidence_matches_source_text(all_transcripts) -> None:
    """Resolved evidence in Expert Analysis must match the source segment without alteration."""
    analyses = build_demo_analyses(all_transcripts)
    for analysis in analyses:
        transcript = next(t for t in all_transcripts if t.transcript_id == analysis.transcript_id)
        resolved = resolve_evidence(analysis, transcript)
        seg_map = {s.segment_id: s for s in transcript.segments}

        for item in resolved:
            source = seg_map[item.segment_id]
            assert item.timestamp == source.timestamp
            assert item.speaker == source.speaker
            assert item.text == source.text
            assert item.text == source.text.strip()


def test_resolved_comparison_evidence_matches_source_text(all_transcripts) -> None:
    """Resolved comparison evidence matches the original segment verbatim."""
    comparison = build_demo_comparison()
    resolved = resolve_comparison_evidence(comparison, all_transcripts)

    indexed = {
        (t.transcript_id, s.segment_id): s
        for t in all_transcripts
        for s in t.segments
    }

    assert resolved
    for item in resolved:
        source = indexed[(item.transcript_id, item.segment_id)]
        assert item.timestamp == source.timestamp
        assert item.speaker == source.speaker
        assert item.text == source.text


def test_resolved_qa_evidence_matches_source_text(all_transcripts) -> None:
    """Resolved Q&A evidence matches the original segment verbatim."""
    retriever = build_retriever(all_transcripts)
    query = "What did experts say about training?"
    retrieved = retriever.search(query, top_k=5)
    assert retrieved

    payload = QAAnswerPayload(
        answer="Training is critical for utilization.",
        evidence=[
            QAEvidence(
                transcript_id=retrieved[0].transcript_id,
                segment_id=retrieved[0].segment_id,
                reason="Training relevance.",
            )
        ],
    )
    response = QAResponse(question=query, answer=payload.answer, evidence=payload.evidence)
    resolved = resolve_qa_evidence(response, retrieved=retrieved, transcripts=all_transcripts)

    assert len(resolved) == 1
    assert resolved[0].text == retrieved[0].text
    assert resolved[0].timestamp == retrieved[0].timestamp
    assert resolved[0].speaker == retrieved[0].speaker


def test_no_normalization_mangles_dashes_or_numbers(all_transcripts) -> None:
    """Specific punctuation like numbers, percentages, and en-dashes must not be corrupted."""
    fr = next(t for t in all_transcripts if t.transcript_id == "FR")
    # FR_012 contains '15 to 20 percent'
    fr_012 = next(s for s in fr.segments if s.segment_id == "FR_012")
    assert "15 to 20 percent" in fr_012.text

    # FR_014 contains 'Six to twelve months'
    fr_014 = next(s for s in fr.segments if s.segment_id == "FR_014")
    assert "Six to twelve months" in fr_014.text

    # DE_014 contains 'Nine to eighteen months'
    de = next(t for t in all_transcripts if t.transcript_id == "DE")
    de_014 = next(s for s in de.segments if s.segment_id == "DE_014")
    assert "Nine to eighteen months" in de_014.text
