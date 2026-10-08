# InsightLens

AI-powered expert interview intelligence. InsightLens analyses expert-call
transcripts against an interview guide, compares experts across markets, and
answers free-form questions, with every claim tied back to a timestamped
segment of the original transcript.

## Problem

Research teams run expert calls and then spend hours re-reading transcripts to
answer the same guide questions, spot where experts agree or disagree, and find
the exact quote behind a claim. Generic LLM summaries speed this up but cannot
be trusted. They paraphrase, invent quotes and fill gaps with outside knowledge.

InsightLens uses an LLM for reasoning and synthesis. It never trusts the LLM
for source material: every piece of evidence shown to the user is looked up
deterministically from the original transcript.

The case study contains three expert interviews on robotic surgery adoption
(France, Germany, UK) and a six-question interview guide.

## Features

- **Expert analysis**: each expert is answered against exactly six guide
  questions (Q1–Q6). Each answer has supporting evidence with timestamp,
  speaker and exact source text.
- **Evidence grounding**: the LLM returns segment IDs only. IDs are validated
  and resolved to the original transcript text in Python.
- **Cross-expert comparison**: common themes and meaningful differences across
  France, Germany and the UK. The comparison is descriptive and never ranks or
  scores experts or countries.
- **Grounded Q&A ("Ask InsightLens")**: free-form questions are answered from
  BM25-retrieved transcript segments only. Unsupported questions and false
  premises are handled explicitly.
- **Demo mode**: runs with no API key, using deterministic responses grounded
  in real segment IDs. **Live mode** uses Gemini.

## Architecture

```
data/Transcript_*.txt, data/Guide.txt
        │
        ▼
Deterministic parser ─────────── app/parser.py, app/analysis.py
  (segment IDs FR_001…, timestamp, speaker, exact text; Q1–Q6)
        │
        ▼
Expert analysis (LLM) ────────── app/analyzer.py
  LLM selects segment IDs → validated: Q1–Q6 exactly once, metadata matches,
  every ID exists in *this* transcript, no duplicates
        │
        ▼
Cross-expert comparison (LLM) ── app/comparison.py
  Input: structured analyses → themes / differences → every
  (transcript_id, segment_id) pair validated
        │
        ▼
BM25 retrieval ───────────────── app/retrieval.py
  expert answers indexed with the interviewer question they answer
        │
        ▼
Grounded Q&A (LLM) ───────────── app/qa.py
  retrieve first; no retrieval hits → no LLM call; cited IDs must be in the
  retrieved set
        │
        ▼
Evidence resolution ──────────── resolve_* functions
  ID → original TranscriptSegment → timestamp / speaker / exact text
        │
        ▼
Streamlit UI ─────────────────── app.py, ui/ (presentation only)
```

The LLM provider sits behind an abstract `LLMClient`
(`app/llm_client.py`). `GeminiLLMClient` is the live implementation, and the
tests and demo mode inject deterministic clients.

## Grounding Strategy

**The transcript is the source of truth. LLM-generated evidence IDs are
validated against deterministic transcript data, and displayed source text is
always retrieved from the original transcript.**

Concretely:

1. The parser assigns deterministic IDs (`FR_001`, `DE_004`, …) and keeps the
   original timestamp, speaker and text unchanged.
2. Structured output schemas (Pydantic) only allow the LLM to return
   `segment_id` plus a `reason`. There is no quote field, and any extra field
   the model adds is rejected or ignored, never displayed.
3. Every ID is validated. A nonexistent ID, an ID from another transcript, a
   wrong transcript/segment pairing, a duplicate, or (for Q&A) an ID that was
   not in the retrieved set makes the response fail with a clear error. No
   partial answer is shown.
4. Displayed evidence is built by looking the validated ID up in the parsed
   transcript, so it cannot contain an LLM-written quote.
5. Prompts forbid outside knowledge, rankings and acceptance of false
   premises. If retrieval finds nothing relevant, the LLM is not called.

## Tech Stack

- Python 3.10+ (developed and tested on 3.13)
- [Google GenAI Python SDK](https://pypi.org/project/google-genai/) (`google-genai`), with Gemini structured JSON output
- Pydantic v2 for schemas and validation
- A small built-in BM25 (Okapi) implementation with no extra dependency
- Streamlit for the UI
- pytest

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment

Copy the template and add your key (only needed for live mode):

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Live mode only | Google AI Studio API key |
| `GEMINI_MODEL` | No | Defaults to `gemini-2.5-flash` |

`.env` is git-ignored. Never commit it.

> **Model availability:** Google has stopped offering `gemini-2.5-flash` to
> newly created API keys. If live mode reports that the model is not available
> for your key, set `GEMINI_MODEL` to a model your key can access, for example
> `gemini-3.5-flash`. The free tier also has small per-model daily request
> limits. Analysing all experts plus the comparison uses 4 requests.

## Run

```bash
streamlit run app.py
```

The app opens in **demo mode** (deterministic responses, no API calls). If
`GEMINI_API_KEY` is set, turn on **Use live Gemini analysis** in the sidebar.
The first live load runs three expert analyses and one comparison, which takes
about 30 seconds.

Sections:

1. **Expert Analysis**: pick an expert and review Q1–Q6 answers with evidence.
2. **Compare Experts**: common themes and areas of difference, with evidence.
3. **Ask InsightLens**: ask any question across all transcripts.

Optional CLI demos:

```bash
python -m app.parser        # parse a transcript
python -m app.analysis      # load guide + empty analysis template
python -m app.analyzer      # live Gemini analysis of the France transcript
python -m app.comparison    # comparison (mocked; INSIGHTLENS_LIVE_COMPARISON=1 for live)
python -m app.qa            # Q&A (mocked; INSIGHTLENS_LIVE_QA=1 for live)
```

## Tests

```bash
pytest -q
```

The suite needs no network or API key. It covers parsing, schema and evidence
validation, adversarial LLM outputs (invented, duplicate, cross-transcript or
malformed IDs, missing or extra questions, empty responses, API failures),
retrieval quality and edge cases, Q&A grounding, the Gemini client (with a
fake SDK client) and Streamlit page smoke tests.

## Example Questions

- What are the main barriers to adoption?
- What role does training play?
- What challenges exist around procurement?
- What are the main cost concerns?
- What funding challenges were mentioned?
- What did the experts say about hospital procurement?
- How long does a purchase decision take in each market?

## Unsupported Questions

InsightLens answers only from the interviews. Questions outside the transcript
evidence (stock prices, population, salaries, market size, recent events) get
an answer saying the transcripts do not contain enough information, and no
sources are shown. If retrieval finds no relevant segments, the LLM is not
called.

Questions with a false or unsupported premise ("Why is Germany worse than
France?", "Which country has the highest adoption?", "Which expert is wrong?")
are not taken at face value. The answer states that the transcripts do not
establish the premise and then summarises what the experts actually said.

## Design Decisions

- **Deterministic parser**: IDs, timestamps and speakers must be stable and
  exact. Regex parsing is cheap, testable and cannot hallucinate.
- **Structured schemas**: Pydantic models define the contract with the LLM.
  Gemini is asked for JSON matching the schema, and the response is validated
  again locally. Schemas carry IDs, not quotes.
- **BM25**: for a corpus this small (3 transcripts, 21 expert answers),
  lexical retrieval is transparent, deterministic, free and needs no index
  infrastructure. Expert answers are indexed together with the interviewer
  question they answer, because experts rarely repeat the question's wording.
  For example, "What are the main barriers?" is answered with "The biggest
  issue is…".
- **LLM reasoning**: the LLM does what code cannot. It reads, synthesises,
  compares and decides which segments support a claim.
- **Evidence validation**: the boundary between the two. The LLM's
  references are validated before anything is shown.
- **Provider abstraction**: `LLMClient` keeps the provider swappable and lets
  every pipeline stage be tested without network access.

## Limitations

- **Lexical retrieval**: BM25 needs some word overlap. Paraphrased questions
  with no shared vocabulary can miss relevant answers. Broad meta-questions
  such as "What differences exist between markets?" are better served by the
  **Compare Experts** page than by Q&A.
- **Strict validation**: if the model cites one invalid segment ID, the whole
  response is rejected with an error rather than partially shown. Retrying
  usually succeeds.
- **Case-study specific**: market-to-ID mapping (FR/DE/UK), the three-expert
  comparison and the six-question guide are fixed for this case.
- **No caching across sessions**: live analyses are recomputed per browser
  session (about 4 requests), which matters on the Gemini free tier.
- **LLM quality**: synthesis quality depends on the model. Grounding
  guarantees that citations are real, not that every summary sentence is
  perfectly phrased.
- Demo mode returns fixed analyses and comparisons. Its Q&A answers are
  stitched from retrieved segments rather than reasoned.

## Future Scaling

For 30+ or hundreds of transcripts:

- Replace `LexicalBM25Retriever` with a hybrid retriever (BM25 + embeddings,
  optionally re-ranked) behind the same `Retriever` interface. Q&A, validation
  and resolution do not change.
- Persist parsed transcripts, retrieval indexes and per-transcript analyses so
  each transcript is analysed once, not once per session.
- Generate comparisons hierarchically (per market or segment, then across
  groups) instead of one prompt over all analyses.
- Add metadata filters (market, role, date) to retrieval and the UI.
