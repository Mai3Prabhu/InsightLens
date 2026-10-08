"""Gemini client wrapper for InsightLens structured analysis, comparison, and Q&A."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from app.prompts import (
    build_analysis_messages,
    build_comparison_messages,
    build_qa_messages,
    select_segments_for_analysis,
)
from app.schemas import (
    CrossExpertComparison,
    ExpertAnalysis,
    InterviewQuestion,
    QAAnswerPayload,
    RetrievedSegment,
    Transcript,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_MS = 120_000
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


class LLMClientError(RuntimeError):
    """Raised when the LLM client cannot complete a request safely."""


class LLMClient(ABC):
    """Abstract LLM interface used by analyzers/comparers/QA (easy to mock)."""

    @abstractmethod
    def analyze_expert_transcript(
        self,
        transcript: Transcript,
        questions: Sequence[InterviewQuestion],
    ) -> ExpertAnalysis:
        """Return a structured ExpertAnalysis for one transcript."""

    @abstractmethod
    def compare_expert_analyses(
        self,
        analyses: Sequence[ExpertAnalysis],
    ) -> CrossExpertComparison:
        """Return a structured CrossExpertComparison across experts."""

    @abstractmethod
    def answer_question(
        self,
        question: str,
        evidence: Sequence[RetrievedSegment],
    ) -> QAAnswerPayload:
        """Return a grounded answer over the supplied retrieved evidence only."""


class GeminiLLMClient(LLMClient):
    """Thin Google GenAI SDK wrapper for structured InsightLens outputs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        resolved_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")

        if client is not None:
            self._client = client
            return

        if not resolved_key or not resolved_key.strip():
            raise LLMClientError(
                "GEMINI_API_KEY is not set. Add it to your environment or .env file."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMClientError(
                "The google-genai package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self._client = genai.Client(
            api_key=resolved_key.strip(),
            http_options=types.HttpOptions(timeout=DEFAULT_TIMEOUT_MS),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def analyze_expert_transcript(
        self,
        transcript: Transcript,
        questions: Sequence[InterviewQuestion],
    ) -> ExpertAnalysis:
        segments = select_segments_for_analysis(transcript)
        messages = build_analysis_messages(transcript, questions, segments)
        return self._generate_structured(messages, ExpertAnalysis)

    def compare_expert_analyses(
        self,
        analyses: Sequence[ExpertAnalysis],
    ) -> CrossExpertComparison:
        messages = build_comparison_messages(analyses)
        return self._generate_structured(messages, CrossExpertComparison)

    def answer_question(
        self,
        question: str,
        evidence: Sequence[RetrievedSegment],
    ) -> QAAnswerPayload:
        messages = build_qa_messages(question, evidence)
        return self._generate_structured(messages, QAAnswerPayload)

    def _generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
    ):
        system_instruction, contents = _split_messages(messages)

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_instruction or None,
                response_mime_type="application/json",
                response_json_schema=response_model.model_json_schema(),
                temperature=DEFAULT_TEMPERATURE,
            )
        except ImportError:
            # Injected test clients do not need the real SDK types.
            config = {
                "system_instruction": system_instruction or None,
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema(),
                "temperature": DEFAULT_TEMPERATURE,
            }

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as exc:
                if attempt < MAX_ATTEMPTS and _is_transient(exc):
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise _sanitize_gemini_error(exc, model=self.model) from None

        return _parse_response(response, response_model)


def _is_transient(exc: Exception) -> bool:
    """True for overload / rate-limit responses worth a short retry."""
    code = getattr(exc, "code", None)
    return isinstance(code, int) and (code == 429 or code >= 500)


def _split_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Map provider-neutral role/content messages to Gemini system + contents."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_parts = [m["content"] for m in messages if m.get("role") != "system"]
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def _parse_response(response: Any, response_model: type[BaseModel]):
    """Validate the model's JSON output against the expected Pydantic schema."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, response_model):
        return parsed

    try:
        text = getattr(response, "text", None)
    except Exception:  # SDK raises when the candidate has no text parts
        text = None

    if not text or not str(text).strip():
        raise LLMClientError(
            f"Model returned an empty response for {response_model.__name__} "
            "(possibly blocked or truncated). Please retry."
        )

    try:
        return response_model.model_validate_json(text)
    except ValidationError as exc:
        raise LLMClientError(
            f"Model returned a structured payload that does not match "
            f"{response_model.__name__}."
        ) from exc


def create_llm_client_from_env() -> GeminiLLMClient:
    """Construct a GeminiLLMClient from environment configuration."""
    _load_dotenv_if_available()
    return GeminiLLMClient()


def _load_dotenv_if_available() -> None:
    """Optionally load a local .env file without requiring python-dotenv."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _sanitize_gemini_error(exc: Exception, model: str | None = None) -> LLMClientError:
    """Map Gemini/SDK failures to safe user-facing errors.

    The original exception message is never forwarded, so request details or
    credentials cannot leak into the UI or logs.
    """
    name = type(exc).__name__

    try:
        from google.genai import errors
    except ImportError:
        return LLMClientError(f"LLM request failed ({name}).")

    if isinstance(exc, errors.ClientError):
        code = getattr(exc, "code", None)
        if code in (401, 403):
            return LLMClientError(
                "Gemini authentication failed. Check that GEMINI_API_KEY is valid."
            )
        if code == 400 and "api key" in str(exc).lower():
            return LLMClientError(
                "Gemini rejected the API key. Check that GEMINI_API_KEY is valid."
            )
        if code == 404:
            return LLMClientError(
                f"Gemini model {model or ''!r} is not available for this API key. "
                "Set GEMINI_MODEL in .env to a model your key can access "
                "(e.g. gemini-3.5-flash)."
            )
        if code == 429:
            return LLMClientError("Gemini rate limit or quota exceeded. Retry later.")
        return LLMClientError(f"Gemini rejected the request (HTTP {code}).")
    if isinstance(exc, errors.ServerError):
        return LLMClientError("Gemini service is temporarily unavailable. Retry later.")
    if isinstance(exc, errors.APIError):
        return LLMClientError(f"Gemini API error ({name}).")

    lowered = name.lower()
    if "timeout" in lowered:
        return LLMClientError("Gemini request timed out.")
    if "connect" in lowered or "network" in lowered:
        return LLMClientError("Failed to connect to the Gemini API.")

    return LLMClientError(f"LLM request failed ({name}).")
