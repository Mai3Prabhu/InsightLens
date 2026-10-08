"""Tests for the InsightLens transcript parser.

Uses the three case-study transcripts under data/ as fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parser import TranscriptParseError, parse_all_transcripts, parse_transcript
from app.schemas import Transcript, TranscriptSegment

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

FRANCE = DATA_DIR / "Transcript_1_France.txt"
GERMANY = DATA_DIR / "Transcript_2_Germany.txt"
UK = DATA_DIR / "Transcript_3_UK.txt"

ALL_TRANSCRIPTS = [FRANCE, GERMANY, UK]

# Each case-study file has 14 timed turns (7 interviewer + 7 expert).
EXPECTED_SEGMENT_COUNTS = {
    "FR": 14,
    "DE": 14,
    "UK": 14,
}


# ---------------------------------------------------------------------------
# Successful parsing of all three transcripts
# ---------------------------------------------------------------------------


def test_all_three_transcripts_parse_successfully() -> None:
    for path in ALL_TRANSCRIPTS:
        result = parse_transcript(path)
        assert isinstance(result, Transcript)
        assert result.segments


def test_parse_all_transcripts() -> None:
    results = parse_all_transcripts(DATA_DIR)
    assert len(results) == 3
    assert [t.transcript_id for t in results] == ["FR", "DE", "UK"]


# ---------------------------------------------------------------------------
# Expert metadata
# ---------------------------------------------------------------------------


def test_france_metadata() -> None:
    t = parse_transcript(FRANCE)
    assert t.transcript_id == "FR"
    assert t.expert_name == "Dr. Jean Martin"
    assert t.role == "Head of Urology"
    assert t.market == "France"


def test_germany_metadata() -> None:
    t = parse_transcript(GERMANY)
    assert t.transcript_id == "DE"
    assert t.expert_name == "Anna Keller"
    assert t.role == "Former Hospital Procurement Director"
    assert t.market == "Germany"


def test_uk_metadata() -> None:
    t = parse_transcript(UK)
    assert t.transcript_id == "UK"
    assert t.expert_name == "Dr. Emily Carter"
    assert t.role == "Consultant Urologist"
    assert t.market == "United Kingdom"


# ---------------------------------------------------------------------------
# Segment counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,transcript_id,expected_count",
    [
        (FRANCE, "FR", 14),
        (GERMANY, "DE", 14),
        (UK, "UK", 14),
    ],
)
def test_expected_segment_counts(
    path: Path, transcript_id: str, expected_count: int
) -> None:
    t = parse_transcript(path)
    assert t.transcript_id == transcript_id
    assert len(t.segments) == expected_count
    assert len(t.segments) == EXPECTED_SEGMENT_COUNTS[transcript_id]


# ---------------------------------------------------------------------------
# Timestamps preserved as strings
# ---------------------------------------------------------------------------


def test_timestamps_preserved_as_strings() -> None:
    t = parse_transcript(FRANCE)
    assert t.segments[0].timestamp == "00:00"
    assert t.segments[1].timestamp == "00:18"
    assert t.segments[-1].timestamp == "06:08"
    for seg in t.segments:
        assert isinstance(seg.timestamp, str)


def test_germany_and_uk_timestamps_preserved() -> None:
    de = parse_transcript(GERMANY)
    uk = parse_transcript(UK)
    assert de.segments[0].timestamp == "00:00"
    assert de.segments[-1].timestamp == "06:05"
    assert uk.segments[0].timestamp == "00:00"
    assert uk.segments[-1].timestamp == "06:04"


# ---------------------------------------------------------------------------
# Speakers preserved
# ---------------------------------------------------------------------------


def test_speaker_names_preserved() -> None:
    t = parse_transcript(FRANCE)
    assert t.segments[0].speaker == "Interviewer"
    assert t.segments[1].speaker == "Dr. Martin"

    de = parse_transcript(GERMANY)
    assert de.segments[0].speaker == "Interviewer"
    assert de.segments[1].speaker == "Anna Keller"

    uk = parse_transcript(UK)
    assert uk.segments[0].speaker == "Interviewer"
    assert uk.segments[1].speaker == "Dr. Carter"


def test_handles_both_interviewer_and_expert_speech() -> None:
    for path in ALL_TRANSCRIPTS:
        t = parse_transcript(path)
        speakers = {s.speaker for s in t.segments}
        assert "Interviewer" in speakers
        assert len(speakers) == 2  # interviewer + one expert label


# ---------------------------------------------------------------------------
# Exact text preservation
# ---------------------------------------------------------------------------


def test_transcript_text_preserved_exactly() -> None:
    t = parse_transcript(FRANCE)
    assert t.segments[0].text == (
        "Thanks for joining. To begin, how would you describe "
        "robotic surgery adoption in France today?"
    )
    assert t.segments[1].text == (
        "Adoption is growing, but it is still concentrated in larger "
        "academic hospitals and private centres with stronger capital budgets. "
        "Smaller regional hospitals are much slower."
    )
    # Spot-check a later expert turn that includes numbers
    last_expert = t.segments[-1]
    assert last_expert.speaker == "Dr. Martin"
    assert last_expert.text == (
        "Six to twelve months is realistic once the hospital becomes serious. "
        "It can be longer if the capital committee pushes the purchase into "
        "the next budget cycle."
    )


def test_uk_text_preserved_exactly() -> None:
    t = parse_transcript(UK)
    assert t.segments[-1].text == (
        "The key point is that adoption is not just about buying the machine. "
        "Hospitals need enough trained people and enough procedure volume to "
        "make the programme sustainable."
    )


# ---------------------------------------------------------------------------
# Deterministic segment IDs
# ---------------------------------------------------------------------------


def test_segment_ids_deterministic_and_traceable() -> None:
    t1 = parse_transcript(FRANCE)
    t2 = parse_transcript(FRANCE)
    assert [s.segment_id for s in t1.segments] == [
        s.segment_id for s in t2.segments
    ]
    assert t1.segments[0].segment_id == "FR_001"
    assert t1.segments[1].segment_id == "FR_002"
    assert t1.segments[-1].segment_id == "FR_014"

    de = parse_transcript(GERMANY)
    assert de.segments[0].segment_id == "DE_001"
    assert de.segments[1].segment_id == "DE_002"

    uk = parse_transcript(UK)
    assert uk.segments[0].segment_id == "UK_001"
    assert uk.segments[1].segment_id == "UK_002"


# ---------------------------------------------------------------------------
# Chronological order
# ---------------------------------------------------------------------------


def _timestamp_to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def test_segments_remain_in_chronological_order() -> None:
    for path in ALL_TRANSCRIPTS:
        t = parse_transcript(path)
        times = [_timestamp_to_seconds(s.timestamp) for s in t.segments]
        assert times == sorted(times)
        # Segment IDs also increase with order
        ids = [s.segment_id for s in t.segments]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_file_raises() -> None:
    with pytest.raises(TranscriptParseError, match="does not exist"):
        parse_transcript(DATA_DIR / "does_not_exist.txt")


def test_missing_metadata_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text(
        "Expert 1 – Someone\nRole: Something\n\n00:00\nInterviewer: Hi\n",
        encoding="utf-8",
    )
    with pytest.raises(TranscriptParseError, match="Market"):
        parse_transcript(bad)


def test_bad_timestamp_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad_ts.txt"
    bad.write_text(
        "Expert 1 – Someone\n"
        "Role: Something\n"
        "Market: France\n"
        "\n"
        "not-a-timestamp\n"
        "Interviewer: Hi\n",
        encoding="utf-8",
    )
    with pytest.raises(TranscriptParseError, match="timestamp"):
        parse_transcript(bad)


def test_missing_speaker_text_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad_segment.txt"
    bad.write_text(
        "Expert 1 – Someone\n"
        "Role: Something\n"
        "Market: France\n"
        "\n"
        "00:00\n"
        "Interviewer:\n",
        encoding="utf-8",
    )
    with pytest.raises(TranscriptParseError, match="Empty text"):
        parse_transcript(bad)


def test_segment_model_rejects_blank_fields() -> None:
    with pytest.raises(Exception):
        TranscriptSegment(
            segment_id="FR_001",
            timestamp="00:00",
            speaker="Interviewer",
            text="   ",
        )
