"""Lightweight tests for Streamlit UI data helpers (no Streamlit runtime)."""

from __future__ import annotations

import pytest

from ui.data import (
    DemoLLMClient,
    api_key_configured,
    compute_comparison,
    compute_expert_analyses,
    expert_label,
    load_questions,
    load_transcripts,
    resolve_app_mode,
    run_qa,
    transcripts_by_id,
)


def test_load_transcripts_and_questions() -> None:
    transcripts = load_transcripts()
    questions = load_questions()
    assert len(transcripts) == 3
    assert [t.transcript_id for t in transcripts] == ["FR", "DE", "UK"]
    assert [q.question_id for q in questions] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
    ]


def test_demo_pipeline_produces_analyses_and_comparison() -> None:
    transcripts = load_transcripts()
    questions = load_questions()
    client = DemoLLMClient(transcripts)
    analyses = compute_expert_analyses(transcripts, questions, client)
    assert set(analyses) == {"FR", "DE", "UK"}
    comparison = compute_comparison(analyses, transcripts, client)
    assert comparison.themes or comparison.differences


def test_demo_qa_returns_grounded_response() -> None:
    transcripts = load_transcripts()
    client = DemoLLMClient(transcripts)
    response, sources = run_qa(
        "How important is training?",
        transcripts,
        client,
    )
    assert response.answer
    # Sources (if any) must match evidence IDs.
    evidence_ids = {(e.transcript_id, e.segment_id) for e in response.evidence}
    for src in sources:
        assert (src.transcript_id, src.segment_id) in evidence_ids


def test_resolve_app_mode_demo_by_default() -> None:
    mode = resolve_app_mode(prefer_live=False)
    assert mode.live is False
    assert "Demo mode" in mode.label


def test_expert_label_and_index() -> None:
    assert "France" in expert_label("FR")
    assert "Germany" in expert_label("DE")
    assert transcripts_by_id(load_transcripts())["UK"].market == "United Kingdom"


def test_api_key_configured_is_bool() -> None:
    assert isinstance(api_key_configured(), bool)


def test_resolve_app_mode_live_fallback_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """When prefer_live=True but GEMINI_API_KEY is unset, resolves to demo mode safely."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Patch api_key_configured to False
    monkeypatch.setattr("ui.data.api_key_configured", lambda: False)
    mode = resolve_app_mode(prefer_live=True)
    assert mode.live is False
    assert "Demo mode — API key missing" in mode.label


def test_resolve_app_mode_live_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """When prefer_live=True and key is present, resolves to live mode."""
    monkeypatch.setattr("ui.data.api_key_configured", lambda: True)
    mode = resolve_app_mode(prefer_live=True)
    assert mode.live is True
    assert "Live mode" in mode.label
    assert "Gemini" in mode.label


def test_resolve_analysis_evidence_by_question_id() -> None:
    """resolve_analysis_evidence groups resolved segments under question IDs."""
    from ui.data import resolve_analysis_evidence
    transcripts = load_transcripts()
    questions = load_questions()
    client = DemoLLMClient(transcripts)
    analyses = compute_expert_analyses(transcripts, questions, client)

    fr_resolved = resolve_analysis_evidence(analyses["FR"], transcripts[0])
    assert isinstance(fr_resolved, dict)
    assert "Q1" in fr_resolved
    assert all(r.segment_id.startswith("FR_") for r in fr_resolved["Q1"])


def test_demo_client_copies_objects_without_mutating_cache() -> None:
    """DemoLLMClient returns deep copies so caller modifications do not mutate cached state."""
    transcripts = load_transcripts()
    client = DemoLLMClient(transcripts)
    first = client.analyze_expert_transcript(transcripts[0], load_questions())
    first.answers[0].answer = "MODIFIED BY CALLER"

    second = client.analyze_expert_transcript(transcripts[0], load_questions())
    assert second.answers[0].answer != "MODIFIED BY CALLER"
