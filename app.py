"""InsightLens Streamlit application (presentation layer only).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from app.analyzer import AnalyzerError
from app.comparison import ComparisonError
from app.llm_client import LLMClientError
from app.qa import QAError
from app.retrieval import RetrievalError
from ui.components import (
    inject_styles,
    render_comparison_sources,
    render_expert_meta,
    render_mode_banner,
    render_qa_sources,
    render_resolved_evidence,
)
from ui.data import (
    EXPERT_CHOICES,
    api_key_configured,
    build_llm_client,
    compute_comparison,
    compute_expert_analyses,
    load_questions,
    load_transcripts,
    resolve_analysis_evidence,
    resolve_app_mode,
    resolve_comparison_sources,
    run_qa,
    transcripts_by_id,
)

logger = logging.getLogger(__name__)

PAGES = (
    "Expert Analysis",
    "Compare Experts",
    "Ask InsightLens",
)

EXAMPLE_QUESTIONS = [
    "What are the main barriers to adoption?",
    "What role does training play?",
    "What did the experts say about hospital procurement?",
    "What are the main cost concerns?",
    "What differences exist between markets?",
]

# Errors raised by our own validation layers carry safe, readable messages.
_KNOWN_ERRORS = (LLMClientError, AnalyzerError, ComparisonError, QAError, RetrievalError)


def _user_error_message(exc: Exception) -> str:
    """Return a clean message for the UI (no stack traces or internals)."""
    if isinstance(exc, _KNOWN_ERRORS):
        return str(exc)
    return (
        "Something went wrong while processing this request "
        f"({type(exc).__name__}). Please retry."
    )


def _init_session() -> None:
    defaults = {
        "page": PAGES[0],
        "prefer_live": False,
        "analyses": None,
        "comparison": None,
        "comparison_sources": None,
        "qa_response": None,
        "qa_sources": None,
        "qa_error": None,
        "pipeline_error": None,
        "loaded_mode_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def _cached_transcripts():
    return load_transcripts()


@st.cache_resource(show_spinner=False)
def _cached_questions():
    return load_questions()


def _ensure_pipeline() -> None:
    """Load analyses + comparison once per mode (session-scoped)."""
    mode = resolve_app_mode(st.session_state.prefer_live)
    mode_key = "live" if mode.live else "demo"

    if (
        st.session_state.analyses is not None
        and st.session_state.comparison is not None
        and st.session_state.loaded_mode_key == mode_key
    ):
        return

    transcripts = _cached_transcripts()
    questions = _cached_questions()
    st.session_state.pipeline_error = None

    try:
        client = build_llm_client(mode, transcripts)
        spinner_label = (
            "Analyzing expert interviews with the live model..."
            if mode.live
            else "Preparing demo expert analyses..."
        )
        with st.spinner(spinner_label):
            analyses = compute_expert_analyses(transcripts, questions, client)
        with st.spinner(
            "Comparing expert perspectives..."
            if mode.live
            else "Preparing demo comparison..."
        ):
            comparison = compute_comparison(analyses, transcripts, client)
            sources = resolve_comparison_sources(comparison, transcripts)

        st.session_state.analyses = analyses
        st.session_state.comparison = comparison
        st.session_state.comparison_sources = sources
        st.session_state.loaded_mode_key = mode_key
        # Clear prior Q&A when switching mode so evidence stays coherent.
        st.session_state.qa_response = None
        st.session_state.qa_sources = None
        st.session_state.qa_error = None
    except Exception as exc:  # noqa: BLE001 - surface a clean message instead of a trace
        logger.warning("Pipeline failed: %s", type(exc).__name__)
        message = _user_error_message(exc)
        if mode.live:
            message += " You can switch off live mode in the sidebar to use demo mode."
        st.session_state.pipeline_error = message
        st.session_state.analyses = None
        st.session_state.comparison = None
        st.session_state.comparison_sources = None
        st.session_state.loaded_mode_key = None


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Navigation")
        # Bound to session state via key: passing a computed index instead
        # changes the widget identity and makes navigation need two clicks.
        st.radio("Section", PAGES, key="page", label_visibility="collapsed")

        st.divider()
        st.markdown("### Analysis mode")
        has_key = api_key_configured()
        if has_key:
            st.toggle(
                "Use live Gemini analysis",
                key="prefer_live",
                help="When off, InsightLens uses deterministic demo responses.",
            )
        else:
            st.session_state.prefer_live = False
            st.caption(
                "No `GEMINI_API_KEY` detected. Running in demo mode. "
                "Add a key to `.env` to enable live analysis."
            )

        mode = resolve_app_mode(st.session_state.prefer_live)
        render_mode_banner(mode.label, mode.live)

        st.divider()
        st.caption("Evidence is always resolved from original transcript text.")


def _page_expert_analysis() -> None:
    st.header("Expert Analysis")
    st.write(
        "Review structured answers for each expert against the interview guide. "
        "Evidence text is taken from the original transcript segments."
    )

    if st.session_state.pipeline_error:
        st.error(st.session_state.pipeline_error)
        return
    if not st.session_state.analyses:
        st.warning("Expert analyses are not available yet.")
        return

    transcripts = transcripts_by_id(_cached_transcripts())
    questions = {q.question_id: q.question for q in _cached_questions()}
    labels = [label for _, label in EXPERT_CHOICES]
    ids = [tid for tid, _ in EXPERT_CHOICES]
    selected_label = st.selectbox("Select expert", labels, index=0)
    transcript_id = ids[labels.index(selected_label)]

    analysis = st.session_state.analyses[transcript_id]
    transcript = transcripts[transcript_id]
    resolved = resolve_analysis_evidence(analysis, transcript)

    render_expert_meta(analysis.expert_name, analysis.role, analysis.market)
    st.divider()

    for answer in analysis.answers:
        q_text = questions.get(answer.question_id, answer.question_id)
        with st.expander(f"{answer.question_id}: {q_text}", expanded=False):
            st.markdown("**QUESTION**")
            st.write(q_text)
            st.markdown("**ANSWER**")
            st.write(answer.answer)
            st.markdown("**EVIDENCE**")
            render_resolved_evidence(
                resolved.get(answer.question_id, []),
                market=analysis.market,
            )


def _page_compare_experts() -> None:
    st.header("Compare Experts")
    st.write(
        "Descriptive themes and differences across France, Germany, and the UK. "
        "Markets are not ranked or scored."
    )

    if st.session_state.pipeline_error:
        st.error(st.session_state.pipeline_error)
        return

    comparison = st.session_state.comparison
    all_sources = st.session_state.comparison_sources or []
    if comparison is None:
        st.warning("Comparison results are not available yet.")
        return

    by_item: dict[tuple[str, str], list] = {}
    for item in all_sources:
        by_item.setdefault((item.item_kind, item.item_id), []).append(item)

    st.subheader("Common themes")
    if not comparison.themes:
        st.caption("No common themes identified from the available evidence.")
    for theme in comparison.themes:
        with st.container(border=True):
            st.markdown(f"#### {theme.title}")
            st.write(theme.summary)
            st.markdown("**Supporting evidence**")
            render_comparison_sources(by_item.get(("theme", theme.theme_id), []))

    st.subheader("Areas of difference")
    if not comparison.differences:
        st.caption("No meaningful differences identified from the available evidence.")
    for diff in comparison.differences:
        with st.container(border=True):
            st.markdown(f"#### {diff.title}")
            st.write(diff.summary)
            st.markdown("**Supporting evidence**")
            render_comparison_sources(
                by_item.get(("difference", diff.difference_id), [])
            )


def _page_ask() -> None:
    st.header("Ask InsightLens")
    st.write(
        "Ask a free-form question across all three expert interviews. "
        "Answers are grounded in retrieved transcript segments only."
    )

    st.markdown("**Example questions**")
    for example in EXAMPLE_QUESTIONS:
        st.caption(f"• {example}")

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Ask a question about the expert interviews",
            height=100,
            placeholder="e.g. What are the biggest barriers to adoption?",
            key="qa_question_input",
        )
        submitted = st.form_submit_button("Ask InsightLens", type="primary")

    if submitted:
        st.session_state.qa_error = None
        st.session_state.qa_response = None
        st.session_state.qa_sources = None
        mode = resolve_app_mode(st.session_state.prefer_live)
        transcripts = _cached_transcripts()
        try:
            client = build_llm_client(mode, transcripts)
            with st.spinner("Searching transcripts and generating a grounded answer..."):
                response, sources = run_qa(question, transcripts, client)
            st.session_state.qa_response = response
            st.session_state.qa_sources = sources
        except Exception as exc:  # noqa: BLE001 - surface a clean message instead of a trace
            logger.warning("Q&A failed: %s", type(exc).__name__)
            st.session_state.qa_error = _user_error_message(exc)

    if st.session_state.qa_error:
        st.error(st.session_state.qa_error)

    response = st.session_state.qa_response
    if response is not None:
        st.divider()
        st.markdown("**QUESTION**")
        st.write(response.question)
        st.markdown("**ANSWER**")
        if response.evidence:
            st.write(response.answer)
        else:
            st.info(response.answer)
        st.markdown("**SOURCES**")
        if response.evidence:
            render_qa_sources(st.session_state.qa_sources or [])
        else:
            st.caption(
                "No transcript segments support an answer to this question. "
                "InsightLens does not answer from general knowledge."
            )


def main() -> None:
    st.set_page_config(
        page_title="InsightLens",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session()
    inject_styles()

    st.title("InsightLens")
    st.markdown(
        '<p class="insight-subtitle">AI-Powered Expert Interview Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="insight-desc">'
        "Analyze expert interviews, compare perspectives, and ask "
        "evidence-grounded questions across transcripts."
        "</p>",
        unsafe_allow_html=True,
    )

    _render_sidebar()
    _ensure_pipeline()

    page = st.session_state.page
    if page == "Expert Analysis":
        _page_expert_analysis()
    elif page == "Compare Experts":
        _page_compare_experts()
    else:
        _page_ask()


if __name__ == "__main__":
    main()
