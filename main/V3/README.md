# V3: ReAct planner + source-first RAG + structured metadata + fan reaction

Autonomous recap agent for Love Is Blind. Given an edition, season, and the last
episode watched, it researches, retrieves, and generates a narrator-voiced,
spoiler-bounded recap, structured metadata and ground-truth data steer the
research and correct the output, a dedicated spoiler-check node audits the
result before it's shown.

## Architecture

```
                              USER
                               │
                    edition, season, last episode watched
                               │
                               ▼
                    ┌─────────────────────┐
                    │  FETCH SHOW METADATA │   ← OMDb + TMDB (structured, not RAG)
                    └──────────┬───────────┘
                               │  canonical episode titles (OMDb)
                               │  per-episode participants (TMDB)
                               ▼
                    ┌─────────────────────┐
                    │   RESEARCH AGENT      │   ← ReAct agent (LangGraph)
                    │   (plan_and_search)   │     decides its own search queries
                    └──────────┬───────────┘
                               │  web sources, each tagged with an
                               │  episode range (regex + title match)
                               ▼
                    ┌─────────────────────┐
              ┌────▶│  SEASON INDEX LOAD    │   ← hand-verified ground truth
              │     │ (episodes + cast CSV) │     (bypasses the research agent)
              │     └──────────┬───────────┘
              │                │
              │                ▼
              │     ┌─────────────────────┐
              └────▶│   RANK & SELECT       │   ← per-need budgets (bios/highlights/
                    │ (source-first RAG)    │     drama/reaction), not one shared race
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ FETCH YOUTUBE          │   ← only videos whose ENTIRE range
                    │ COMMENTS               │     is ≤ cutoff (confirmed necessary
                    └──────────┬───────────┘     via real spoiler-leak test)
                               │
                               ▼
                    ┌─────────────────────┐
                    │   SOURCE INDEX         │  ← chunk, tag (episode + phase), embed
                    │   (Pinecone)           │
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │      RETRIEVAL         │
                    └──────────┬───────────┘
          ┌────────┬───────────┼──────────┐
          ▼         ▼          ▼          ▼
        BIOS      DRAMA     EPISODE    REACTION
          │         │          │          │
          │         │          │          ▼
          │         │          │   ┌─────────────────┐
          │         │          │   │  FAN REACTION     │
          │         │          │   │  ANALYSIS         │
          │         │          │   └────────┬──────────┘
          └─────────┴──────────┴────────────┘
                               ▼
                    ┌─────────────────────┐
                    │      GENERATE          │
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │   SPOILER CHECK          │──┐
                    └──────────┬───────────┘  │ fail: retry with
                               │ pass          │ specific issue fed back
                               ▼               │
                      "Previously On..."◀──────┘
```

## Node-by-node logic

**FETCH SHOW METADATA** — Structured metadata, not a RAG source. OMDb supplies
canonical episode titles (used both to sharpen search queries and to match
sources that name an episode by title rather than number). TMDB supplies the
real per-episode cast list (hosts + contestants), used later to correct and
enrich the generated participant list. Both cover all 12 current editions via
`season_indexes/imdb_ids.csv` and `tmdb_ids.csv`, one entry per edition since
IMDb/TMDB use a single series ID across all seasons of an edition.

**RESEARCH AGENT** (`plan_and_search`) — A real LangChain/LangGraph ReAct agent,
not a fixed set of queries. It decides what to search for (cast bios, early-season
drama, this-episode-only content, fan reaction) and issues its own queries,
adapting based on what it finds. Every discovered source gets a deterministic
episode range: first a regex pass on the title (catches "Episode 6", "Episodes
6-9", "S1E6"), falling back to matching the source's title against OMDb's real
episode titles when no number is present in the text at all.

**SEASON INDEX LOAD** — Hand-verified ground truth (episode/phase milestones,
cast ages/professions), loaded independently of the research agent, so it's
never subject to search variance. Only rows up to the user's cutoff are ever
loaded into memory, a spoiler floor independent of anything downstream.

**RANK & SELECT** — The core fix underpinning V2/V3: bios, highlights, drama,
and reaction each get their own guaranteed selection budget, instead of one
shared ranking where episode-precise sources always beat general or
reaction-oriented ones for a shared pool of slots. Ground-truth sources are
included unconditionally, on top of this, never competing for a budget slot.

**FETCH YOUTUBE COMMENTS** — Only pulls comments from a video whose *entire*
tagged episode range is within the user's cutoff, not just "mostly" watched.
Confirmed necessary with real test data: a range video's comments can leak
later-episode content without ever naming an episode number (a wedding-dress
detail from a future episode surfaced in a "6-9" video's comment section, with
cutoff at 6).

**SOURCE INDEX** — Chunking (with a hard per-chunk size cap so no chunk can
exceed the embedding model's token limit), episode + phase tagging via batched
LLM calls (capped batch size for reliability), embedding into Pinecone. Each
edition/season gets its own namespace, cleared at the start of every run so
stale data from earlier runs or earlier bugs can't silently leak into results.

**RETRIEVAL** — Four independent, purpose-specific queries, not one blended
query serving every need. `main_drama` is filtered to strictly *before* the
current phase (the current episode's content belongs to `highlights`, not
`main_drama`). `highlights` requires the chunk's episode range to actually
contain the target episode. `bios` and `reaction` pull from general/background
content, with ground-truth chunks pulled by direct metadata filter rather than
competing on semantic similarity against much larger web transcripts.

**FAN REACTION ANALYSIS** — Synthesizes the pre-filtered, already spoiler-safe
comments into a structured summary: overall reception, what fans liked,
criticism, recurring themes, and a handful of short, paraphrased sample
reactions (never long verbatim quotes). Only runs when eligible comments exist.

**GENERATE** — The full grounding/tone prompt (conversational, hyped narrator
voice, strict "never state a claim not in the context" rule). After generation:
TMDB's participant data merges in and *corrects* (not just appends to) whatever
the model guessed, e.g. a vague "Filip" with a wrong profession becomes the
correctly disambiguated "Filip Lenz" with his real age and job. Structured fan
reaction data overrides the model's own weaker synthesis when available.
Non-clickable internal ground-truth "sources" are stripped from the visible
citation list before the user ever sees them.

**SPOILER CHECK** — Audits the complete draft after generation. On failure, the
specific issue found is fed back into one bounded retry of `generate`, not a
blind re-roll. Explicit rules distinguish dramatic-but-in-bounds content
(safe) from a specific fact about a later episode (a real spoiler), since
early testing showed the auditor was prone to false positives on the former.

## What's ground-truth vs. inferred, at a glance

| Data | Source | Reliability |
|---|---|---|
| Episode titles | OMDb | Structured, always available for the 12 covered editions |
| Per-episode cast | TMDB | Structured, confirmed accurate against real episode data |
| Episode/phase milestones | Hand-verified CSV | Only exists for Poland S1 currently |
| Cast ages/professions | Hand-verified CSV | Only exists for Poland S1 currently |
| Plot detail, drama, highlights | Web search (Tavily), ReAct-directed | Variable, depends on what the agent finds each run |
| Fan reaction | YouTube comments, range-gated | Variable, only as good as what's in eligible videos' comments |

## Setup

```bash
cd V3
pip install python-dotenv openai tavily-python pinecone langgraph langchain-openai langchain-core flask flask-cors requests
```

`.env` needs:
```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
PINECONE_API_KEY=...
OMDB_API_KEY=...          (or OMBD_API_KEY)
TMDB_API_KEY=...
YOUTUBE_API_KEY=...
```

## Run it

**CLI:**
```bash
python recap.py --episode 6
```

**UI:**
```bash
python app.py          # localhost:5004
open recap_ui.html      # separate terminal
```

## Known limitations

- **Only Poland Season 1 has hand-verified ground-truth files.** Every other
  edition will run, OMDb/TMDB metadata works for all 12, but `main_drama` and
  `highlights` quality for other editions depends entirely on what the ReAct
  agent finds via search, the same reliability profile this pipeline had before
  the Poland ground-truth files existed.
- **Search-result variance is real and unresolved.** The same query can surface
  different sources run to run, `highlights` in particular sometimes gets zero
  dedicated single-episode sources, though it degrades gracefully by falling
  back to range-tagged content rather than failing outright.
- **Fan reaction depends on eligible YouTube videos existing.** If no video's
  entire range is within the user's cutoff, `audience_reaction` falls back to
  a much weaker string generated from general web "reaction"-labeled content.