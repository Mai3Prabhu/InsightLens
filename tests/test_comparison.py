"""Tests for Loop 4 cross-expert comparison (mocked LLM only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.comparison import (
    CASE_STUDY_TRANSCRIPT_IDS,
    ComparisonError,
    CrossExpertComparer,
    build_demo_analyses,
    build_demo_comparison,
    resolve_comparison_evidence,
    validate_analyses_for_comparison,
    validate_comparison_evidence,
)
from app.llm_client import LLMClient
from app.parser import parse_all_transcripts
from app.schemas import (
    ComparisonDifference,
    ComparisonEvidence,
    ComparisonTheme,
    CrossExpertComparison,
    ExpertAnalysis,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class FakeComparisonLLMClient(LLMClient):
    """Deterministic comparison LLM stand-in (no network calls)."""

    def __init__(self, response: CrossExpertComparison | Exception) -> None:
        self.response = response
        self.calls: list[list[ExpertAnalysis]] = []

    def analyze_expert_transcript(self, transcript, questions):
        raise NotImplementedError

    def compare_expert_analyses(self, analyses):
        self.calls.append(list(analyses))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response.model_copy(deep=True)

    def answer_question(self, question, evidence):
        raise NotImplementedError("Loop 4 tests do not use Q&A.")


@pytest.fixture(scope="module")
def transcripts():
    return parse_all_transcripts(DATA_DIR)


@pytest.fixture(scope="module")
def analyses(transcripts):
    return build_demo_analyses(transcripts)


# ---------------------------------------------------------------------------
# Successful comparison
# ---------------------------------------------------------------------------


def test_successful_comparison_of_fr_de_uk(transcripts, analyses) -> None:
    comparison = CrossExpertComparer(
        FakeComparisonLLMClient(build_demo_comparison())
    ).compare(analyses, transcripts)

    assert isinstance(comparison, CrossExpertComparison)
    assert len(comparison.themes) >= 1
    assert len(comparison.differences) >= 1
    ids = {a.transcript_id for a in analyses}
    assert ids == CASE_STUDY_TRANSCRIPT_IDS


@pytest.mark.parametrize("missing_id", ["FR", "DE", "UK"])
def test_missing_expert_analysis_rejected(analyses, transcripts, missing_id) -> None:
    incomplete = [a for a in analyses if a.transcript_id != "FR" and a.transcript_id != missing_id]
    if missing_id in {"DE", "UK"}:
        incomplete = [a for a in analyses if a.transcript_id != missing_id]
    with pytest.raises(ComparisonError, match="Missing expert analyses"):
        CrossExpertComparer(
            FakeComparisonLLMClient(build_demo_comparison())
        ).compare(incomplete, transcripts)


def test_comparison_represents_difference_in_emphasis_without_contradiction() -> None:
    """Verifies France saying X and Germany saying X strongly is represented as emphasis, not contradiction."""
    comparison = build_demo_comparison()
    # In build_demo_comparison, D2 captures difference in emphasis (economics vs clinical strategy)
    d2 = next(d for d in comparison.differences if d.difference_id == "D2")
    assert "weight" in d2.title.lower() or "emphasis" in d2.summary.lower() or "economics" in d2.summary.lower()
    # Must not contain ungrounded contradiction or claim one expert is 'wrong'
    assert "wrong" not in d2.summary.lower()
    assert "contradict" not in d2.summary.lower()


def test_comparison_contains_no_scores_or_rankings() -> None:
    """Themes and differences must be descriptive only (no scores, ranks, or league tables)."""
    comparison = build_demo_comparison()
    for theme in comparison.themes:
        assert "rank" not in theme.title.lower()
        assert "score" not in theme.title.lower()
    for diff in comparison.differences:
        assert "rank" not in diff.title.lower()
        assert "score" not in diff.title.lower()


def test_duplicate_transcript_ids_rejected(analyses) -> None:
    duplicated = list(analyses) + [analyses[0].model_copy(deep=True)]
    with pytest.raises(ComparisonError, match="Duplicate transcript_id"):
        validate_analyses_for_comparison(duplicated)


def test_unknown_transcript_ids_in_evidence_rejected(transcripts, analyses) -> None:
    bad = build_demo_comparison()
    bad.themes[0].evidence.append(
        ComparisonEvidence(
            transcript_id="US",
            segment_id="US_001",
            reason="Invented market.",
        )
    )
    with pytest.raises(ComparisonError, match="Unknown transcript_id"):
        CrossExpertComparer(FakeComparisonLLMClient(bad)).compare(
            analyses, transcripts
        )


def test_unknown_segment_ids_rejected(transcripts, analyses) -> None:
    bad = build_demo_comparison()
    bad.differences[0].evidence[0] = ComparisonEvidence(
        transcript_id="FR",
        segment_id="FR_999",
        reason="Invented segment.",
    )
    with pytest.raises(ComparisonError, match="Unknown segment_id"):
        CrossExpertComparer(FakeComparisonLLMClient(bad)).compare(
            analyses, transcripts
        )


def test_valid_comparison_evidence_accepted(transcripts) -> None:
    comparison = build_demo_comparison()
    # Should not raise
    validate_comparison_evidence(comparison, transcripts)


def test_theme_evidence_resolves_to_original_segments(transcripts, analyses) -> None:
    comparison = CrossExpertComparer(
        FakeComparisonLLMClient(build_demo_comparison())
    ).compare(analyses, transcripts)

    resolved = resolve_comparison_evidence(comparison, transcripts)
    theme_items = [r for r in resolved if r.item_kind == "theme"]
    assert theme_items

    fr_map = {s.segment_id: s for s in next(t for t in transcripts if t.transcript_id == "FR").segments}
    sample = next(r for r in theme_items if r.segment_id == "FR_002")
    source = fr_map["FR_002"]
    assert sample.timestamp == source.timestamp
    assert sample.speaker == source.speaker
    assert sample.text == source.text


def test_difference_evidence_resolves_to_original_segments(
    transcripts, analyses
) -> None:
    comparison = CrossExpertComparer(
        FakeComparisonLLMClient(build_demo_comparison())
    ).compare(analyses, transcripts)

    resolved = resolve_comparison_evidence(comparison, transcripts)
    diff_items = [r for r in resolved if r.item_kind == "difference"]
    assert diff_items

    de_map = {
        s.segment_id: s
        for s in next(t for t in transcripts if t.transcript_id == "DE").segments
    }
    sample = next(r for r in diff_items if r.segment_id == "DE_014")
    source = de_map["DE_014"]
    assert sample.timestamp == source.timestamp
    assert sample.speaker == source.speaker
    assert sample.text == source.text


def test_empty_themes_and_differences_allowed(transcripts, analyses) -> None:
    empty = CrossExpertComparison(themes=[], differences=[])
    result = CrossExpertComparer(FakeComparisonLLMClient(empty)).compare(
        analyses, transcripts
    )
    assert result.themes == []
    assert result.differences == []


def test_expected_ids_override_allows_arbitrary_set(transcripts, analyses) -> None:
    """Validation is isolated: expected_ids=None accepts any unique set."""
    subset = [a for a in analyses if a.transcript_id in {"FR", "DE"}]
    empty = CrossExpertComparison(themes=[], differences=[])
    result = CrossExpertComparer(
        FakeComparisonLLMClient(empty),
        expected_ids=None,
    ).compare(subset, [t for t in transcripts if t.transcript_id in {"FR", "DE"}])
    assert result.themes == []


def test_comparison_evidence_schema_has_no_quote_field() -> None:
    ref = ComparisonEvidence(
        transcript_id="FR",
        segment_id="FR_002",
        reason="Supports theme.",
    )
    assert set(ref.model_dump().keys()) == {
        "transcript_id",
        "segment_id",
        "reason",
    }


def test_theme_and_difference_models_construct() -> None:
    theme = ComparisonTheme(
        theme_id="T1",
        title="Example",
        summary="Summary",
        evidence=[],
    )
    diff = ComparisonDifference(
        difference_id="D1",
        title="Example",
        summary="Summary",
        evidence=[],
    )
    assert theme.theme_id == "T1"
    assert diff.difference_id == "D1"
