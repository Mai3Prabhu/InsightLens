"""Reusable Streamlit presentation helpers (no business logic)."""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from app.schemas import (
    ResolvedComparisonEvidence,
    ResolvedEvidence,
    ResolvedQAEvidence,
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 1100px; }
        .insight-subtitle {
            color: #4a5568;
            font-size: 1.05rem;
            margin-top: -0.6rem;
            margin-bottom: 0.4rem;
        }
        .insight-desc {
            color: #5f6b7a;
            margin-bottom: 1.2rem;
        }
        .source-card {
            border-left: 3px solid #2c5282;
            background: #f7fafc;
            padding: 0.75rem 0.9rem;
            margin: 0.45rem 0 0.7rem 0;
            border-radius: 0 6px 6px 0;
        }
        .source-meta {
            font-size: 0.85rem;
            color: #2d3748;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .source-reason {
            font-size: 0.8rem;
            color: #718096;
            margin-bottom: 0.35rem;
        }
        .source-text {
            font-size: 0.95rem;
            color: #1a202c;
            line-height: 1.45;
        }
        .meta-chip {
            display: inline-block;
            background: #edf2f7;
            color: #2d3748;
            padding: 0.15rem 0.55rem;
            border-radius: 4px;
            margin-right: 0.35rem;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mode_banner(label: str, live: bool) -> None:
    if live:
        st.success(label)
    else:
        st.info(label)


def render_expert_meta(name: str, role: str, market: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<span class="meta-chip">Expert</span> **{html.escape(name)}**',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<span class="meta-chip">Role</span> **{html.escape(role)}**',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<span class="meta-chip">Market</span> **{html.escape(market)}**',
        unsafe_allow_html=True,
    )


def render_source_block(
    *,
    market: str,
    timestamp: str,
    speaker: str,
    text: str,
    segment_id: str,
    reason: str | None = None,
) -> None:
    reason_html = (
        f'<div class="source-reason">Why cited: {html.escape(reason)}</div>'
        if reason
        else ""
    )
    st.markdown(
        f"""
        <div class="source-card">
          <div class="source-meta">
            SOURCE · {html.escape(market)} · {html.escape(timestamp)} · {html.escape(speaker)}
            <span style="color:#718096; font-weight:500;"> · {html.escape(segment_id)}</span>
          </div>
          {reason_html}
          <div class="source-text">“{html.escape(text)}”</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resolved_evidence(items: Sequence[ResolvedEvidence], market: str) -> None:
    if not items:
        st.caption("No supporting evidence segments for this answer.")
        return
    for item in items:
        render_source_block(
            market=market,
            timestamp=item.timestamp,
            speaker=item.speaker,
            text=item.text,
            segment_id=item.segment_id,
            reason=item.reason,
        )


def render_comparison_sources(
    items: Sequence[ResolvedComparisonEvidence],
) -> None:
    if not items:
        st.caption("No supporting evidence.")
        return
    market_names = {"FR": "France", "DE": "Germany", "UK": "United Kingdom"}
    for item in items:
        market = market_names.get(item.transcript_id, item.transcript_id)
        render_source_block(
            market=market,
            timestamp=item.timestamp,
            speaker=item.speaker,
            text=item.text,
            segment_id=item.segment_id,
            reason=item.reason,
        )


def render_qa_sources(items: Sequence[ResolvedQAEvidence]) -> None:
    if not items:
        st.caption("No supporting sources.")
        return
    for item in items:
        market = item.market or item.transcript_id
        render_source_block(
            market=market,
            timestamp=item.timestamp,
            speaker=item.speaker,
            text=item.text,
            segment_id=item.segment_id,
            reason=item.reason,
        )
