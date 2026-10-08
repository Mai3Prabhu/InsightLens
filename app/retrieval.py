"""Lightweight local retrieval over InsightLens transcript segments.

Uses deterministic lexical BM25 ranking. Designed behind a Retriever protocol
so a future vector/hybrid implementation can replace this without changing Q&A.

Only expert turns are retrievable: interviewer turns are questions, not
evidence. Each expert turn is scored together with the interviewer question
it answers (Q/A pairing), because experts rarely repeat the question's
wording ("What are the main barriers?" -> "The biggest issue is ...").
The returned text is always the expert's exact original segment text.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

from app.schemas import RetrievedSegment, Transcript

# Default: keep any segment with a positive BM25 score.
DEFAULT_MIN_SCORE = 0.0
DEFAULT_TOP_K = 10

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

INTERVIEWER_SPEAKER = "interviewer"

DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "aren", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can",
        "could", "did", "do", "does", "doing", "down", "during", "each",
        "few", "for", "from", "further", "had", "has", "have", "having",
        "he", "her", "here", "hers", "herself", "him", "himself", "his",
        "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
        "of", "off", "on", "once", "only", "or", "other", "our", "ours",
        "ourselves", "out", "over", "own", "same", "she", "should", "so",
        "some", "such", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was",
        "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "with", "would", "you", "your", "yours", "yourself",
        "yourselves",
    }
)


class RetrievalError(ValueError):
    """Raised when retrieval inputs are invalid."""


class Retriever(ABC):
    """Abstract retrieval interface (local BM25 today; vector later)."""

    @abstractmethod
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedSegment]:
        """Return the highest-scoring segments for *query* (no duplicates)."""


def tokenize(text: str, *, filter_stop_words: bool = True) -> list[str]:
    """Lowercase alphanumeric tokenization with light plural normalisation."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if filter_stop_words:
        tokens = [t for t in tokens if t not in DEFAULT_STOP_WORDS]
    return [_normalize_plural(t) for t in tokens]


def _normalize_plural(token: str) -> str:
    """Map simple English plurals to singular (barriers -> barrier).

    Deliberately conservative: applied identically to queries and documents,
    so an imperfect stem still matches itself.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def is_interviewer(speaker: str) -> bool:
    return speaker.strip().lower() == INTERVIEWER_SPEAKER


class LexicalBM25Retriever(Retriever):
    """In-memory BM25 Okapi retriever over TranscriptSegment objects."""

    def __init__(
        self,
        transcripts: Sequence[Transcript],
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not transcripts:
            raise RetrievalError("At least one Transcript is required to build a retriever.")

        self.min_score = float(min_score)
        self.k1 = float(k1)
        self.b = float(b)

        self._docs: list[RetrievedSegment] = []
        index_texts: list[str] = []
        seen: set[tuple[str, str]] = set()

        for transcript in transcripts:
            previous = None
            for segment in transcript.segments:
                key = (transcript.transcript_id, segment.segment_id)
                if key in seen or is_interviewer(segment.speaker):
                    previous = segment
                    continue
                seen.add(key)
                context = (
                    previous.text
                    if previous is not None and is_interviewer(previous.speaker)
                    else None
                )
                self._docs.append(
                    RetrievedSegment(
                        transcript_id=transcript.transcript_id,
                        segment_id=segment.segment_id,
                        timestamp=segment.timestamp,
                        speaker=segment.speaker,
                        text=segment.text,
                        score=0.0,
                        question_context=context,
                    )
                )
                index_texts.append(f"{context}\n{segment.text}" if context else segment.text)
                previous = segment

        if not self._docs:
            raise RetrievalError("No expert transcript segments available to index.")

        self._tokenized: list[list[str]] = [tokenize(text) for text in index_texts]
        self._doc_len = [len(tokens) or 1 for tokens in self._tokenized]
        self._avgdl = sum(self._doc_len) / len(self._doc_len)
        self._df: Counter[str] = Counter()
        for tokens in self._tokenized:
            for term in set(tokens):
                self._df[term] += 1
        self._n_docs = len(self._tokenized)
        self._idf = {
            term: math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    @property
    def size(self) -> int:
        return len(self._docs)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedSegment]:
        if query is None or not str(query).strip():
            raise RetrievalError("Query must be a non-empty string.")
        if top_k < 1:
            raise RetrievalError("top_k must be >= 1.")

        query_tokens = tokenize(query)
        if not query_tokens:
            # Query has no alphanumeric tokens → no meaningful lexical matches.
            return []

        scores = self._score_all(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: list[RetrievedSegment] = []
        seen_ids: set[str] = set()

        for idx in ranked_indices:
            score = scores[idx]
            if score <= self.min_score:
                break
            doc = self._docs[idx]
            if doc.segment_id in seen_ids:
                continue
            seen_ids.add(doc.segment_id)
            results.append(doc.model_copy(update={"score": float(score)}))
            if len(results) >= top_k:
                break

        return results

    def _score_all(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self._n_docs
        query_tf = Counter(query_tokens)
        for term, qtf in query_tf.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tokens in enumerate(self._tokenized):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self._doc_len[i] / self._avgdl
                )
                scores[i] += idf * (tf * (self.k1 + 1) / denom) * qtf
        return scores


def build_retriever(
    transcripts: Sequence[Transcript],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
) -> LexicalBM25Retriever:
    """Factory for the default local lexical retriever."""
    return LexicalBM25Retriever(transcripts, min_score=min_score)
