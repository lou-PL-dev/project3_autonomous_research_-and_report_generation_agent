# V2: ReAct planner + source-first RAG

A rebuild, not a patch on V1. Core idea: rank sources by temporal fit to the user's
cutoff *before* chunking anything, using deterministic title-based episode extraction,
and use a real LangChain/LangGraph ReAct agent to decide search strategy instead of
four fixed queries every run.

## Why this exists

V1 kept losing precise, single-episode sources to broader ones in retrieval, purely
because the broader source had more semantic mass, not because it was more relevant.
The root cause: episode coverage was only known *after* chunking and LLM tagging,
by which point the wrong source had already won. V2 fixes this by extracting episode
coverage from titles with regex, for free, before any chunking, and using that as the
primary ranking signal.

## What changed from V1

- **`node_plan_and_search`**: a real ReAct agent (`langgraph.prebuilt.create_react_agent`)
  decides what to search for, given the four evidence needs (bios, season drama, this
  episode, fan reaction), rather than running the same four fixed queries every time.
- **`extract_episode_range_from_title()`**: regex-based, catches "Episode 6",
  "Episodes 6-9", "Eps. 6-8", "S1E6", etc. Runs before any LLM call.
- **`temporal_fit()`**: scores every discovered source against the user's cutoff.
  Sources entirely past the cutoff (`fit < 0`) are excluded outright, before they can
  contribute a single chunk.
- **`node_rank_and_select`**: source-first selection. Chunking, tagging, and embedding
  only happen for the top `MAX_SELECTED_SOURCES` (12) by temporal fit, not everything
  the search turned up.
- **Title-derived range wins over LLM-tagged range** when both exist (`node_index`),
  the LLM tagger is now a fallback only for sources the title regex couldn't resolve.

## What's unchanged (proven in V1, not broken)

- Pinecone RAG with range-based metadata filtering (`episode_start`/`episode_end`)
- Phase-based `main_drama` filtering (phases up to and including the resolved current
  phase), with the "unknown phase" chunks getting a safety-net inclusion only when
  their own episode range is independently spoiler-safe
- The four-section retrieve design (bios / season drama / this episode / reaction)
- The full grounding + tone prompt, and the false-positive-resistant spoiler-check
  rules from V1's later debugging
- Noise/general-domain/wrong-edition filtering (TikTok, Spotify, Wikipedia-other-season, etc.)

## Setup

```bash
cd v2
pip install python-dotenv openai tavily-python pinecone langgraph langchain-openai langchain-core flask flask-cors
```

`.env` needs the same three keys as V1: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`.

## Run it

```bash
python recap.py --episode 6
```

or the UI:
```bash
python app.py          # localhost:5003
open recap_ui.html
```

## Known unknowns going into this

- **Not yet run end to end against the API.** The regex extraction and temporal-fit
  scoring were unit-tested against real titles from V1's session tonight (all correct),
  but the ReAct agent's actual search behavior, tool-call count, query quality, is
  unverified until it's run for real.
- **`MAX_SELECTED_SOURCES = 12` is a guess**, not measured. May need tuning once you see
  how many genuinely distinct, well-covering sources the planner tends to find.
- **The ReAct agent could under- or over-search.** Nothing currently caps how many tool
  calls it makes, worth watching for runaway cost on the first real run.
- **Same single-source (Tavily) constraint as V1.** ReAct matters more once TMDB/YouTube
  give the agent an actual choice between tools, not just query phrasing.
