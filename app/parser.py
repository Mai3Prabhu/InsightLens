"""Deterministic parser for InsightLens expert-call transcripts.

Converts raw .txt transcript files into validated Pydantic Transcript objects
without any LLM involvement. Source text is preserved exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.schemas import Transcript, TranscriptSegment

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# "Expert 1 – Dr. Jean Martin" or "Expert 2 - Anna Keller"
_EXPERT_HEADER_RE = re.compile(
    r"^Expert\s+\d+\s*[–—-]\s*(.+)$",
    re.IGNORECASE,
)

_ROLE_RE = re.compile(r"^Role:\s*(.+)$", re.IGNORECASE)
_MARKET_RE = re.compile(r"^Market:\s*(.+)$", re.IGNORECASE)

# Timestamps like "00:00", "06:08" (also tolerant of H:MM:SS if present)
_TIMESTAMP_RE = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)$")

# "Speaker Name: utterance text"
_SPEAKER_LINE_RE = re.compile(r"^([^:]+):\s*(.*)$")

# Market name -> stable transcript / segment ID prefix
_MARKET_ID_MAP: dict[str, str] = {
    "france": "FR",
    "germany": "DE",
    "united kingdom": "UK",
    "uk": "UK",
}


class TranscriptParseError(ValueError):
    """Raised when a transcript file cannot be parsed cleanly."""


def parse_transcript(path: str | Path) -> Transcript:
    """Parse a single expert-call transcript file into a Transcript.

    Args:
        path: Path to a raw ``.txt`` transcript file.

    Returns:
        A validated :class:`~app.schemas.Transcript` with deterministic
        segment IDs and exact source text.

    Raises:
        TranscriptParseError: If the file is missing, metadata is incomplete,
            timestamps are malformed, or segments lack speaker/text.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise TranscriptParseError(f"Transcript file does not exist: {file_path}")
    if not file_path.is_file():
        raise TranscriptParseError(f"Path is not a file: {file_path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TranscriptParseError(f"Unable to read transcript file: {file_path}") from exc

    lines = raw.splitlines()
    expert_name, role, market, body_start = _parse_header(lines, file_path)
    transcript_id = _market_to_id(market, file_path)
    segments = _parse_segments(lines[body_start:], transcript_id, file_path)

    if not segments:
        raise TranscriptParseError(
            f"No transcript segments found in {file_path}"
        )

    return Transcript(
        transcript_id=transcript_id,
        expert_name=expert_name,
        role=role,
        market=market,
        segments=segments,
    )


def parse_all_transcripts(directory: str | Path) -> list[Transcript]:
    """Parse every ``Transcript_*.txt`` file in *directory* (sorted by name)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise TranscriptParseError(f"Directory does not exist: {dir_path}")

    files = sorted(dir_path.glob("Transcript_*.txt"))
    if not files:
        raise TranscriptParseError(f"No Transcript_*.txt files found in {dir_path}")

    return [parse_transcript(f) for f in files]


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def _parse_header(
    lines: list[str], file_path: Path
) -> tuple[str, str, str, int]:
    """Extract expert_name, role, market and the index where body starts."""
    expert_name: str | None = None
    role: str | None = None
    market: str | None = None
    body_start = 0

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if expert_name is None:
            match = _EXPERT_HEADER_RE.match(line)
            if match:
                expert_name = match.group(1).strip()
                body_start = idx + 1
                continue
            raise TranscriptParseError(
                f"Expected expert header at start of {file_path}, got: {line!r}"
            )

        if role is None:
            match = _ROLE_RE.match(line)
            if match:
                role = match.group(1).strip()
                body_start = idx + 1
                continue
            raise TranscriptParseError(
                f"Expected 'Role:' line after expert header in {file_path}, got: {line!r}"
            )

        if market is None:
            match = _MARKET_RE.match(line)
            if match:
                market = match.group(1).strip()
                body_start = idx + 1
                continue
            raise TranscriptParseError(
                f"Expected 'Market:' line after role in {file_path}, got: {line!r}"
            )

        # Header complete – remaining content is the dialogue body
        body_start = idx
        break
    else:
        # Exhausted file while still collecting header fields
        body_start = len(lines)

    missing = [
        name
        for name, value in (
            ("expert_name", expert_name),
            ("role", role),
            ("market", market),
        )
        if not value
    ]
    if missing:
        raise TranscriptParseError(
            f"Expert metadata missing in {file_path}: {', '.join(missing)}"
        )

    assert expert_name is not None and role is not None and market is not None
    return expert_name, role, market, body_start


def _market_to_id(market: str, file_path: Path) -> str:
    key = market.strip().lower()
    transcript_id = _MARKET_ID_MAP.get(key)
    if transcript_id is None:
        raise TranscriptParseError(
            f"Unknown market {market!r} in {file_path}; "
            f"expected one of: {sorted(set(_MARKET_ID_MAP.values()))}"
        )
    return transcript_id


# ---------------------------------------------------------------------------
# Segment parsing
# ---------------------------------------------------------------------------


def _parse_segments(
    lines: list[str], transcript_id: str, file_path: Path
) -> list[TranscriptSegment]:
    """Parse timestamp + speaker/text blocks into TranscriptSegment list."""
    segments: list[TranscriptSegment] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        ts_match = _TIMESTAMP_RE.match(line)
        if not ts_match:
            raise TranscriptParseError(
                f"Expected timestamp in {file_path} at body line {i + 1}, got: {line!r}"
            )

        timestamp = ts_match.group(1)  # preserve exact matched string
        i += 1

        # Skip blank lines between timestamp and speaker line
        while i < n and not lines[i].strip():
            i += 1

        if i >= n:
            raise TranscriptParseError(
                f"Timestamp {timestamp!r} in {file_path} has no following speaker/text"
            )

        speaker_line = lines[i].strip()
        speaker_match = _SPEAKER_LINE_RE.match(speaker_line)
        if not speaker_match:
            raise TranscriptParseError(
                f"Cannot parse speaker/text after timestamp {timestamp!r} "
                f"in {file_path}: {speaker_line!r}"
            )

        speaker = speaker_match.group(1).strip()
        text = speaker_match.group(2).strip()

        if not speaker:
            raise TranscriptParseError(
                f"Empty speaker after timestamp {timestamp!r} in {file_path}"
            )
        if not text:
            raise TranscriptParseError(
                f"Empty text after timestamp {timestamp!r} "
                f"(speaker={speaker!r}) in {file_path}"
            )

        segment_number = len(segments) + 1
        segment_id = f"{transcript_id}_{segment_number:03d}"

        segments.append(
            TranscriptSegment(
                segment_id=segment_id,
                timestamp=timestamp,
                speaker=speaker,
                text=text,
            )
        )
        i += 1

    return segments


# ---------------------------------------------------------------------------
# CLI / quick demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Small executable example showing how the parser is used."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "Transcript_1_France.txt"

    transcript = parse_transcript(path)
    print(transcript.expert_name)
    print(transcript.market)
    print(transcript.segments[0])
    print(f"Total segments: {len(transcript.segments)}")


if __name__ == "__main__":
    _demo()
