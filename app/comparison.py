"""Cross-expert comparison service for InsightLens (Loop 4).

Primary input is structured ExpertAnalysis objects (not raw transcripts).
The LLM synthesizes themes/differences; deterministic code validates evidence
IDs and resolves original source text from Transcript objects.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from app.llm_client import (
    LLMClient,
    LLMClientError,
    _load_dotenv_if_available,
    create_llm_client_from_env,
)
from app.schemas import (
    ComparisonDifference,
    ComparisonEvidence,
    ComparisonTheme,
    CrossExpertComparison,
    ExpertAnalysis,
    ResolvedComparisonEvidence,
    Transcript,
    TranscriptSegment,
)

# Case-study expectation. Pass expected_ids=None to allow arbitrary expert sets.
CASE_STUDY_TRANSCRIPT_IDS: frozenset[str] = frozenset({"FR", "DE", "UK"})


class ComparisonError(ValueError):
    """Raised when comparison inputs or LLM evidence fail validation."""


class CrossExpertComparer:
    """Orchestrates LLM comparison + deterministic evidence validation."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        expected_ids: frozenset[str] | None = CASE_STUDY_TRANSCRIPT_IDS,
    ) -> None:
        self._llm = llm_client
        self._expected_ids = expected_ids

    def compare(
        self,
        analyses: Sequence[ExpertAnalysis],
        transcripts: Sequence[Transcript] | Mapping[str, Transcript],
    ) -> CrossExpertComparison:
        """Compare expert analyses and validate evidence against transcripts.

        Args:
            analyses: One ExpertAnalysis per expert/transcript.
            transcripts: Matching Transcript objects used for segment lookup.
        """
        ordered = validate_analyses_for_comparison(
            analyses, expected_ids=self._expected_ids
        )
        transcript_map = _index_transcripts(transcripts)

        # Ensure every analysis has a matching transcript for evidence checks.
        missing_tx = [tid for tid in ordered if tid not in transcript_map]
        if missing_tx:
            raise ComparisonError(
                f"Missing Transcript objects for transcript_ids: {missing_tx}"
            )

        try:
            raw = self._llm.compare_expert_analyses(
                [ordered[tid] for tid in _ordered_ids(ordered, self._expected_ids)]
            )
        except LLMClientError:
            raise
        except Exception as exc:
            raise ComparisonError(
                f"LLM comparison failed unexpectedly: {type(exc).__name__}"
            ) from exc

        if not isinstance(raw, CrossExpertComparison):
            if isinstance(raw, dict) and "themes" not in raw and "differences" not in raw:
                raise ComparisonError(
                    "LLM returned a payload that is not a valid CrossExpertComparison."
                )
            try:
                raw = CrossExpertComparison.model_validate(raw)
            except Exception as exc:
                raise ComparisonError(
                    "LLM returned a payload that is not a valid CrossExpertComparison."
                ) from exc

        validate_comparison_evidence(raw, transcript_map)
        return raw


def validate_analyses_for_comparison(
    analyses: Sequence[ExpertAnalysis],
    *,
    expected_ids: frozenset[str] | None = CASE_STUDY_TRANSCRIPT_IDS,
) -> dict[str, ExpertAnalysis]:
    """Validate the set of ExpertAnalysis objects before comparison.

    Raises ComparisonError on duplicates or (when expected_ids is set) missing
    / unexpected transcript IDs.
    """
    if not analyses:
        raise ComparisonError("At least one ExpertAnalysis is required for comparison.")

    by_id: dict[str, ExpertAnalysis] = {}
    for analysis in analyses:
        tid = analysis.transcript_id
        if tid in by_id:
            raise ComparisonError(f"Duplicate transcript_id in analyses: {tid!r}")
        by_id[tid] = analysis

    if expected_ids is not None:
        present = set(by_id)
        missing = sorted(expected_ids - present)
        unexpected = sorted(present - expected_ids)
        if missing:
            raise ComparisonError(
                f"Missing expert analyses for transcript_ids: {missing}. "
                f"Expected: {sorted(expected_ids)}"
            )
        if unexpected:
            raise ComparisonError(
                f"Unexpected transcript_ids in analyses: {unexpected}. "
                f"Expected: {sorted(expected_ids)}"
            )

    return by_id


def validate_comparison_evidence(
    comparison: CrossExpertComparison,
    transcripts: Sequence[Transcript] | Mapping[str, Transcript],
) -> None:
    """Ensure every ComparisonEvidence points at a real transcript segment."""
    transcript_map = _index_transcripts(transcripts)
    valid_segments: dict[str, set[str]] = {
        tid: {s.segment_id for s in tx.segments} for tid, tx in transcript_map.items()
    }

    def _check(ref: ComparisonEvidence, context: str) -> None:
        if ref.transcript_id not in transcript_map:
            raise ComparisonError(
                f"Unknown transcript_id {ref.transcript_id!r} in {context}. "
                f"Known: {sorted(transcript_map)}"
            )
        allowed = valid_segments[ref.transcript_id]
        if ref.segment_id not in allowed:
            raise ComparisonError(
                f"Unknown segment_id {ref.segment_id!r} for transcript "
                f"{ref.transcript_id!r} in {context}."
            )

    for theme in comparison.themes:
        seen_theme_refs: set[tuple[str, str]] = set()
        for ref in theme.evidence:
            key = (ref.transcript_id, ref.segment_id)
            if key in seen_theme_refs:
                raise ComparisonError(
                    f"Duplicate evidence reference {ref.transcript_id}/{ref.segment_id} "
                    f"in theme {theme.theme_id!r}."
                )
            seen_theme_refs.add(key)
            _check(ref, f"theme {theme.theme_id!r}")

    for diff in comparison.differences:
        seen_diff_refs: set[tuple[str, str]] = set()
        for ref in diff.evidence:
            key = (ref.transcript_id, ref.segment_id)
            if key in seen_diff_refs:
                raise ComparisonError(
                    f"Duplicate evidence reference {ref.transcript_id}/{ref.segment_id} "
                    f"in difference {diff.difference_id!r}."
                )
            seen_diff_refs.add(key)
            _check(ref, f"difference {diff.difference_id!r}")


def resolve_comparison_evidence(
    comparison: CrossExpertComparison,
    transcripts: Sequence[Transcript] | Mapping[str, Transcript],
) -> list[ResolvedComparisonEvidence]:
    """Resolve comparison evidence IDs to original transcript source fields."""
    transcript_map = _index_transcripts(transcripts)
    validate_comparison_evidence(comparison, transcript_map)

    segment_index: dict[str, dict[str, TranscriptSegment]] = {
        tid: {s.segment_id: s for s in tx.segments}
        for tid, tx in transcript_map.items()
    }

    resolved: list[ResolvedComparisonEvidence] = []

    def _resolve(
        item_kind: str,
        item_id: str,
        item_title: str,
        refs: Iterable[ComparisonEvidence],
    ) -> None:
        for ref in refs:
            segment = segment_index[ref.transcript_id][ref.segment_id]
            resolved.append(
                ResolvedComparisonEvidence(
                    item_kind=item_kind,
                    item_id=item_id,
                    item_title=item_title,
                    transcript_id=ref.transcript_id,
                    segment_id=segment.segment_id,
                    timestamp=segment.timestamp,
                    speaker=segment.speaker,
                    text=segment.text,
                    reason=ref.reason,
                )
            )

    for theme in comparison.themes:
        _resolve("theme", theme.theme_id, theme.title, theme.evidence)
    for diff in comparison.differences:
        _resolve("difference", diff.difference_id, diff.title, diff.evidence)

    return resolved


def _index_transcripts(
    transcripts: Sequence[Transcript] | Mapping[str, Transcript],
) -> dict[str, Transcript]:
    if isinstance(transcripts, Mapping):
        return dict(transcripts)

    indexed: dict[str, Transcript] = {}
    for tx in transcripts:
        if tx.transcript_id in indexed:
            raise ComparisonError(
                f"Duplicate transcript_id in transcripts: {tx.transcript_id!r}"
            )
        indexed[tx.transcript_id] = tx
    return indexed


def _ordered_ids(
    by_id: Mapping[str, ExpertAnalysis],
    expected_ids: frozenset[str] | None,
) -> list[str]:
    if expected_ids is not None:
        # Stable case-study order: FR, DE, UK
        preferred = ["FR", "DE", "UK"]
        ordered = [tid for tid in preferred if tid in by_id]
        extras = sorted(set(by_id) - set(ordered))
        return ordered + extras
    return sorted(by_id)


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def build_demo_analyses(
    transcripts: Sequence[Transcript],
) -> list[ExpertAnalysis]:
    """Build deterministic mock ExpertAnalysis objects for offline demos/tests.

    Grounded in real segment IDs from the three case-study transcripts.
    """
    from app.schemas import EvidenceReference, ExpertAnswer

    by_id = {t.transcript_id: t for t in transcripts}
    required = ("FR", "DE", "UK")
    missing = [tid for tid in required if tid not in by_id]
    if missing:
        raise ComparisonError(f"Demo requires transcripts {required}; missing {missing}")

    fr, de, uk = by_id["FR"], by_id["DE"], by_id["UK"]

    def _answers(
        pairs: list[tuple[str, str, list[tuple[str, str]]]],
    ) -> list[ExpertAnswer]:
        out: list[ExpertAnswer] = []
        for qid, answer, evidence in pairs:
            out.append(
                ExpertAnswer(
                    question_id=qid,
                    answer=answer,
                    evidence=[
                        EvidenceReference(segment_id=sid, reason=reason)
                        for sid, reason in evidence
                    ],
                )
            )
        return out

    return [
        ExpertAnalysis(
            transcript_id=fr.transcript_id,
            expert_name=fr.expert_name,
            role=fr.role,
            market=fr.market,
            answers=_answers(
                [
                    (
                        "Q1",
                        "Adoption is growing but concentrated in larger centres.",
                        [("FR_002", "Describes uneven adoption in France.")],
                    ),
                    (
                        "Q2",
                        "Capital budget approval is the main barrier.",
                        [("FR_004", "Names capital budget approval as the biggest issue.")],
                    ),
                    (
                        "Q3",
                        "ROI and economics are very important.",
                        [("FR_006", "Finance wants utilisation and payback.")],
                    ),
                    (
                        "Q4",
                        "Training matters for utilisation.",
                        [("FR_008", "Discusses training and utilisation.")],
                    ),
                    (
                        "Q5",
                        "Expect steady growth, ~15–20% in stronger centres.",
                        [("FR_010", "Gives 3–5 year outlook with growth range.")],
                    ),
                    (
                        "Q6",
                        "Purchase decisions take six to twelve months.",
                        [("FR_014", "States typical purchase timeline.")],
                    ),
                ]
            ),
        ),
        ExpertAnalysis(
            transcript_id=de.transcript_id,
            expert_name=de.expert_name,
            role=de.role,
            market=de.market,
            answers=_answers(
                [
                    (
                        "Q1",
                        "Adoption is growing but uneven across hospital types.",
                        [("DE_002", "Describes uneven German adoption.")],
                    ),
                    (
                        "Q2",
                        "Cost and utilisation proof are the main barriers.",
                        [("DE_004", "Names cost and utilisation barriers.")],
                    ),
                    (
                        "Q3",
                        "Procurement focuses on TCO and economic case.",
                        [("DE_006", "Details procurement economic focus.")],
                    ),
                    (
                        "Q4",
                        "Surgeon training is operationally critical.",
                        [("DE_008", "Links training to utilisation.")],
                    ),
                    (
                        "Q5",
                        "Growth likely high single to low double digits.",
                        [("DE_010", "Gives cautious growth outlook.")],
                    ),
                    (
                        "Q6",
                        "Purchase process often takes nine to eighteen months.",
                        [("DE_014", "States longer procurement timeline.")],
                    ),
                ]
            ),
        ),
        ExpertAnalysis(
            transcript_id=uk.transcript_id,
            expert_name=uk.expert_name,
            role=uk.role,
            market=uk.market,
            answers=_answers(
                [
                    (
                        "Q1",
                        "Adoption is increasing but access varies by hospital.",
                        [("UK_002", "Describes UK adoption variation.")],
                    ),
                    (
                        "Q2",
                        "Funding and training capacity are both important barriers.",
                        [("UK_004", "Names funding and training barriers.")],
                    ),
                    (
                        "Q3",
                        "ROI matters but is balanced with clinical strategy.",
                        [("UK_006", "Describes balanced ROI discussion.")],
                    ),
                    (
                        "Q4",
                        "Training capacity is as important as funding.",
                        [("UK_004", "Highlights training capacity.")],
                    ),
                    (
                        "Q5",
                        "Could see above 15% procedure growth in some areas.",
                        [("UK_010", "Gives positive growth outlook.")],
                    ),
                    (
                        "Q6",
                        "Around six to nine months if funding is available.",
                        [("UK_012", "States shorter timeline when funded.")],
                    ),
                ]
            ),
        ),
    ]


def build_demo_comparison() -> CrossExpertComparison:
    """Deterministic comparison payload for offline demos (valid FR/DE/UK IDs)."""
    return CrossExpertComparison(
        themes=[
            ComparisonTheme(
                theme_id="T1",
                title="Uneven adoption concentrated in larger centres",
                summary=(
                    "All three experts describe growing but uneven adoption, "
                    "with larger or more advanced hospitals ahead of smaller ones."
                ),
                evidence=[
                    ComparisonEvidence(
                        transcript_id="FR",
                        segment_id="FR_002",
                        reason="France: concentrated in larger academic/private centres.",
                    ),
                    ComparisonEvidence(
                        transcript_id="DE",
                        segment_id="DE_002",
                        reason="Germany: university hospitals ahead of smaller hospitals.",
                    ),
                    ComparisonEvidence(
                        transcript_id="UK",
                        segment_id="UK_002",
                        reason="UK: access still varies significantly by hospital.",
                    ),
                ],
            ),
            ComparisonTheme(
                theme_id="T2",
                title="Training and utilisation are operationally critical",
                summary=(
                    "Experts across markets emphasise that limited trained surgeons "
                    "undermines utilisation and the business case."
                ),
                evidence=[
                    ComparisonEvidence(
                        transcript_id="FR",
                        segment_id="FR_008",
                        reason="France: training needed so utilisation is high enough.",
                    ),
                    ComparisonEvidence(
                        transcript_id="DE",
                        segment_id="DE_008",
                        reason="Germany: one trained surgeon weakens utilisation.",
                    ),
                    ComparisonEvidence(
                        transcript_id="UK",
                        segment_id="UK_004",
                        reason="UK: training capacity is as important as funding.",
                    ),
                ],
            ),
        ],
        differences=[
            ComparisonDifference(
                difference_id="D1",
                title="Purchase timeline length differs by market",
                summary=(
                    "Reported decision timelines differ: UK often 6–9 months when "
                    "funded, France 6–12 months, Germany commonly 9–18 months."
                ),
                evidence=[
                    ComparisonEvidence(
                        transcript_id="FR",
                        segment_id="FR_014",
                        reason="France: six to twelve months once serious.",
                    ),
                    ComparisonEvidence(
                        transcript_id="DE",
                        segment_id="DE_014",
                        reason="Germany: nine to eighteen months is common.",
                    ),
                    ComparisonEvidence(
                        transcript_id="UK",
                        segment_id="UK_012",
                        reason="UK: six to nine months if funding is available.",
                    ),
                ],
            ),
            ComparisonDifference(
                difference_id="D2",
                title="Relative weight of economics vs clinical strategy",
                summary=(
                    "France and Germany emphasise economic/procurement cases strongly; "
                    "the UK expert describes a more balanced economics and clinical "
                    "strategy discussion."
                ),
                evidence=[
                    ComparisonEvidence(
                        transcript_id="FR",
                        segment_id="FR_006",
                        reason="France: finance wants utilisation and payback.",
                    ),
                    ComparisonEvidence(
                        transcript_id="DE",
                        segment_id="DE_006",
                        reason="Germany: economic case decides approval.",
                    ),
                    ComparisonEvidence(
                        transcript_id="UK",
                        segment_id="UK_008",
                        reason="UK: economics and clinical strategy are balanced.",
                    ),
                ],
            ),
        ],
    )


def _print_comparison(
    comparison: CrossExpertComparison,
    resolved: list[ResolvedComparisonEvidence] | None = None,
) -> None:
    resolved = resolved or []
    by_item: dict[tuple[str, str], list[ResolvedComparisonEvidence]] = {}
    for item in resolved:
        by_item.setdefault((item.item_kind, item.item_id), []).append(item)

    print("COMMON THEMES")
    print("=" * 72)
    if not comparison.themes:
        print("(none)")
    for theme in comparison.themes:
        print(f"\n[{theme.theme_id}] {theme.title}")
        print(theme.summary)
        print("Supporting evidence IDs:")
        for ref in theme.evidence:
            print(f"  - {ref.transcript_id}/{ref.segment_id}: {ref.reason}")
        sources = by_item.get(("theme", theme.theme_id), [])
        if sources:
            print("Resolved source text:")
            for src in sources:
                print(
                    f"  [{src.segment_id}] {src.timestamp} | {src.speaker}: {src.text}"
                )

    print("\nAREAS OF DIFFERENCE")
    print("=" * 72)
    if not comparison.differences:
        print("(none)")
    for diff in comparison.differences:
        print(f"\n[{diff.difference_id}] {diff.title}")
        print(diff.summary)
        print("Supporting evidence IDs:")
        for ref in diff.evidence:
            print(f"  - {ref.transcript_id}/{ref.segment_id}: {ref.reason}")
        sources = by_item.get(("difference", diff.difference_id), [])
        if sources:
            print("Resolved source text:")
            for src in sources:
                print(
                    f"  [{src.segment_id}] {src.timestamp} | {src.speaker}: {src.text}"
                )


def _demo() -> None:
    """Demo comparison using mocked analyses by default; live LLM if configured."""
    import os

    from app.parser import parse_all_transcripts

    root = Path(__file__).resolve().parent.parent
    transcripts = parse_all_transcripts(root / "data")
    analyses = build_demo_analyses(transcripts)

    _load_dotenv_if_available()
    use_live = bool(os.getenv("GEMINI_API_KEY")) and os.getenv(
        "INSIGHTLENS_LIVE_COMPARISON", ""
    ).lower() in {"1", "true", "yes"}

    if use_live:
        print("Running LIVE LLM comparison (INSIGHTLENS_LIVE_COMPARISON enabled)...")
        try:
            client = create_llm_client_from_env()
            comparison = CrossExpertComparer(client).compare(analyses, transcripts)
        except (LLMClientError, ComparisonError) as exc:
            print(f"Live comparison failed: {exc}")
            print("Falling back to deterministic demo comparison.")
            comparison = build_demo_comparison()
            validate_comparison_evidence(comparison, transcripts)
    else:
        print(
            "Using deterministic mocked comparison "
            "(set GEMINI_API_KEY and INSIGHTLENS_LIVE_COMPARISON=1 for live LLM)."
        )

        class _FakeComparisonClient(LLMClient):
            def analyze_expert_transcript(self, transcript, questions):
                raise NotImplementedError

            def compare_expert_analyses(self, analyses_in):
                return build_demo_comparison()

            def answer_question(self, question, evidence):
                raise NotImplementedError

        comparison = CrossExpertComparer(_FakeComparisonClient()).compare(
            analyses, transcripts
        )

    resolved = resolve_comparison_evidence(comparison, transcripts)
    _print_comparison(comparison, resolved)


if __name__ == "__main__":
    _demo()
