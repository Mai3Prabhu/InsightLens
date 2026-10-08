"""Streamlit app smoke tests via AppTest (demo mode; no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    return at


def test_expert_analysis_page_shows_six_questions_for_each_expert(app: AppTest) -> None:
    for label in app.selectbox[0].options:
        app.selectbox[0].select(label).run()
        assert not app.exception
        assert [e.label.split(":")[0] for e in app.expander] == [
            "Q1", "Q2", "Q3", "Q4", "Q5", "Q6",
        ]
        assert not app.error


def test_navigation_switches_pages_on_every_click(app: AppTest) -> None:
    """Regression: a computed radio index made every second click a no-op."""
    for page in ["Compare Experts", "Ask InsightLens", "Expert Analysis", "Ask InsightLens"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        assert app.header[0].value == page


def test_ask_page_handles_supported_unsupported_and_invalid_input(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Ask InsightLens").run()

    app.text_area[0].input("What role does training play?")
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert "source-card" in " ".join(m.value for m in app.markdown if "SOURCE" in m.value)

    app.text_area[0].input("What is Apple's stock price?")
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert any("do not contain enough information" in i.value for i in app.main.info)

    app.text_area[0].input("   ")
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Question must be a non-empty string."
