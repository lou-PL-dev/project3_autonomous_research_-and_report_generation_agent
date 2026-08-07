# V1: RAG (Pinecone) + LangGraph

Adds retrieval-augmented generation and a real spoiler-check feedback loop on top of the MVP.
Still single-source (Tavily) until TMDB/YouTube keys are added.

## Setup

```bash
cd main
pip install python-dotenv openai tavily-python pinecone langgraph flask flask-cors
```

Add to your `.env` (same file as MVP, plus one new key):
```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
PINECONE_API_KEY=...
```

Get a Pinecone API key at https://www.pinecone.io (free tier is enough for this).

## Run it

**CLI:**
```bash
python recap.py --episode 6
```

**UI:**
```bash
python app.py          # starts the API on localhost:5002
open recap_ui.html     # separate terminal, opens the styled page
```

## What's different from the MVP

- **Chunking + tagging**: each source is split into chunks, and one LLM call per source
  assigns `episode_number`/`phase` per chunk (null if not attributable to one episode,
  never guessed).
- **Retrieval**: instead of dumping all fetched content into one prompt, a semantic query
  against Pinecone pulls back only relevant chunks, filtered by `edition`, `season`, and
  `episode_number ≤ cutoff` (plus untagged general chunks for bios/background).
- **Spoiler-check node**: a second LLM pass audits the draft against the cutoff. On failure,
  the specific issue found feeds back into a bounded regeneration (max 1 retry), not a blind
  re-roll.
- **This directly fixes the MVP's context-length wall**: retrieval only pulls the chunks that
  matter, so source count no longer risks blowing past the model's context limit.

## Known unknowns going into this

- **Not yet tested end to end** — no Pinecone key was available when this was built.
  First run tomorrow should specifically check: does the index get created correctly, does
  metadata filtering behave as expected, does retrieval actually improve output quality or
  coverage over the MVP's raw-dump approach.
- **Chunk tagging is a new LLM call per source** — adds latency and cost per run versus the
  MVP. Worth watching if this becomes a bottleneck once more sources (TMDB, YouTube) are added.
- **ReAct isn't implemented yet** — with only one tool (Tavily) active, there's no real
  tool-selection decision to make. This becomes meaningful once multiple tools exist.
