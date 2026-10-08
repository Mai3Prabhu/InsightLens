"""Retrieval quality, edge cases, and threshold audit tests for Loop 7.

Validates:
1. BM25 semantic usefulness and ranking on core interview themes.
2. Graceful edge-case handling (empty, long, punctuation, stop words, repeated words).
3. Evaluates min_score threshold behavior to prevent irrelevant leakage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parser import parse_all_transcripts
from app.retrieval import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    LexicalBM25Retriever,
    RetrievalError,
    build_retriever,
    tokenize,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@pytest.fixture(scope="module")
def transcripts():
    return parse_all_transcripts(DATA_DIR)


@pytest.fixture(scope="module")
def retriever(transcripts):
    return build_retriever(transcripts)


# ---------------------------------------------------------------------------
# Section 7: Retrieval Quality on Realistic Queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_terms,expected_min_count",
    [
        (
            "What are the barriers to adoption?",
            {"barrier", "barriers", "cost", "capital", "budget", "funding", "training"},
            3,
        ),
        (
            "What did experts say about training?",
            {"train", "training", "trained", "surgeons", "surgeon", "utilisation"},
            3,
        ),
        (
            "How long does procurement take?",
            {"procurement", "purchase", "timeline", "months", "month", "cycle"},
            3,
        ),
        (
            "What role does cost play?",
            {"cost", "costs", "capital", "roi", "economics", "budget", "payback"},
            3,
        ),
        (
            "Funding",
            {"funding", "capital", "budget"},
            1,
        ),
        (
            "training",
            {"training", "trained", "surgeon"},
            3,
        ),
        (
            "hospital procurement",
            {"hospital", "hospitals", "procurement", "purchase", "purchasing"},
            3,
        ),
    ],
)
def test_retrieval_quality_on_realistic_queries(
    retriever, query, expected_terms, expected_min_count
) -> None:
    """Core realistic queries retrieve semantically relevant segments with descending scores."""
    results = retriever.search(query, top_k=DEFAULT_TOP_K)
    assert len(results) >= expected_min_count

    # Scores must be strictly in descending order
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

    # All returned scores must be positive
    assert all(s > 0.0 for s in scores)

    # Segments must be meaningfully related to expected semantic terms
    combined_text = " ".join(r.text.lower() for r in results)
    found_terms = [term for term in expected_terms if term in combined_text]
    assert len(found_terms) >= 1, f"None of {expected_terms} found in retrieval for {query!r}"


def test_cross_market_representation_for_adoption_barriers(retriever) -> None:
    """Broad queries like 'barriers to adoption' return candidates from multiple markets."""
    results = retriever.search("What are the barriers to adoption?", top_k=6)
    markets = {r.transcript_id for r in results}
    # Should represent at least 2 distinct markets in top 6
    assert len(markets) >= 2


# ---------------------------------------------------------------------------
# Section 8: Retrieval Edge Cases
# ---------------------------------------------------------------------------


def test_empty_and_whitespace_queries_fail_gracefully(retriever) -> None:
    """Empty and whitespace strings raise a clean RetrievalError, never unhandled exceptions."""
    with pytest.raises(RetrievalError, match="non-empty"):
        retriever.search("")
    with pytest.raises(RetrievalError, match="non-empty"):
        retriever.search("   \t\n  ")


def test_extremely_long_query_handles_gracefully(retriever) -> None:
    """A 1,000-word repeated query executes safely without memory error or crash."""
    long_query = "robotic surgery adoption barriers hospital budget " * 200
    results = retriever.search(long_query, top_k=5)
    assert isinstance(results, list)
    assert len(results) <= 5
    if results:
        assert results[0].score > 0.0


def test_punctuation_heavy_query_returns_empty_safely(retriever) -> None:
    """Queries composed only of punctuation characters return empty list without crashing."""
    punct_queries = ["???!!!...", "---====***", "$$$###@@@%%%^^^&&&", "()[]{}<>:;,./"]
    for q in punct_queries:
        results = retriever.search(q)
        assert results == []


def test_query_with_uncommon_or_out_of_vocabulary_words(retriever) -> None:
    """Words completely absent from the vocabulary return empty results safely."""
    results = retriever.search("pneumonoultramicroscopicsilicovolcanoconiosis")
    assert results == []


def test_query_with_no_corpus_overlap(retriever) -> None:
    """Queries about unrelated domains (e.g. astrophysics) return empty results."""
    results = retriever.search("astrophysics gravitational waves exoplanet interstellar nebula")
    assert results == []


def test_query_with_only_stop_words_returns_empty_safely(retriever) -> None:
    """Queries containing only common English stop words produce no matches, preventing false positives."""
    stop_queries = [
        "what is the",
        "and or but",
        "in of on at",
        "why would you be so",
    ]
    for q in stop_queries:
        results = retriever.search(q)
        assert results == [], f"Expected empty result for stop-word query {q!r}, got {len(results)} matches"


def test_repeated_words_query_handles_gracefully(retriever) -> None:
    """Repeated keywords (e.g. 'training training training') do not distort ranking or crash."""
    results = retriever.search("training training training training", top_k=5)
    assert results
    # Training terms should be present
    assert any("train" in r.text.lower() for r in results)
    # Scores must be valid non-negative numbers
    assert all(r.score > 0.0 for r in results)
    # No duplicate segment IDs
    ids = [r.segment_id for r in results]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Section 9: No-Evidence Threshold Audit
# ---------------------------------------------------------------------------


def test_min_score_threshold_filters_weak_results(transcripts) -> None:
    """Verifies min_score properly filters results below the threshold."""
    # When min_score=5.0, only high-scoring segments survive
    filtered_retriever = LexicalBM25Retriever(transcripts, min_score=5.0)
    results = filtered_retriever.search("What are the barriers to adoption?", top_k=10)
    for r in results:
        assert r.score > 5.0


def test_default_min_score_preserves_valid_short_queries(retriever) -> None:
    """The default min_score must not discard legitimate low-frequency keywords like 'Funding'."""
    results = retriever.search("Funding", top_k=5)
    assert results, "Single-word query 'Funding' should return results under default min_score"
    assert any("funding" in r.text.lower() for r in results)


def test_retriever_min_score_is_configurable(transcripts) -> None:
    """Retriever min_score parameter is fully configurable."""
    r_default = LexicalBM25Retriever(transcripts)
    assert r_default.min_score == DEFAULT_MIN_SCORE

    r_custom = LexicalBM25Retriever(transcripts, min_score=3.5)
    assert r_custom.min_score == 3.5

    r_factory = build_retriever(transcripts, min_score=1.5)
    assert r_factory.min_score == 1.5


def test_tokenize_preserves_content_tokens_and_filters_stop_words() -> None:
    """tokenize() correctly preserves content tokens while stripping default stop words."""
    raw = "What is the typical hospital procurement timeline in 2026?"
    tokens = tokenize(raw)
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "in" not in tokens
    assert "hospital" in tokens
    assert "procurement" in tokens
    assert "timeline" in tokens
    assert "2026" in tokens


# ---------------------------------------------------------------------------
# Regression: expert answers (not interviewer prompts) must be retrieved
# ---------------------------------------------------------------------------


def test_interviewer_turns_are_never_returned_as_evidence(retriever) -> None:
    for query in ["What are the main barriers?", "adoption", "training", "procurement timeline"]:
        for seg in retriever.search(query, top_k=DEFAULT_TOP_K):
            assert seg.speaker.lower() != "interviewer"


def test_barriers_query_retrieves_each_experts_barrier_answer(retriever) -> None:
    """Experts answer 'barriers' without the word (e.g. 'biggest issue'); Q/A pairing finds them."""
    ids = {s.segment_id for s in retriever.search("What are the main barriers to adoption?")}
    assert {"FR_004", "DE_004", "UK_004"} <= ids


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What role does training play?", {"FR_008", "DE_008", "UK_004"}),
        ("How long does a purchase decision take?", {"FR_014", "DE_014", "UK_012"}),
        ("What funding challenges were mentioned?", {"UK_004"}),
    ],
)
def test_realistic_queries_retrieve_expected_expert_answers(retriever, query, expected) -> None:
    ids = {s.segment_id for s in retriever.search(query)}
    assert expected <= ids


def test_question_context_is_exact_preceding_interviewer_text(transcripts, retriever) -> None:
    by_id = {s.segment_id: (t, i) for t in transcripts for i, s in enumerate(t.segments)}
    for seg in retriever.search("What are the main barriers to adoption?"):
        transcript, idx = by_id[seg.segment_id]
        original = transcript.segments[idx]
        previous = transcript.segments[idx - 1]
        assert seg.text == original.text
        assert seg.question_context == previous.text


def test_plural_normalisation_matches_singular_forms() -> None:
    assert tokenize("barriers hospitals surgeons") == ["barrier", "hospital", "surgeon"]
    assert tokenize("process analysis") == ["process", "analysis"]
