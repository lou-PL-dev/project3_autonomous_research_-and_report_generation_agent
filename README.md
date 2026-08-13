# Previously On: Love Is Blind

An autonomous, spoiler-bounded recap agent for the Love Is Blind franchise.
Tell it an edition, season, and the last episode you watched; it researches,
retrieves, and writes a narrator-voiced recap covering the season so far —
zero spoilers past your cutoff — with no human step in between beyond the
initial trigger.

Built for Project 3 (Autonomous Company/Industry Research & Report
Generation Agent), reinterpreted for the **Media, Journalism & Publishing**
industry: the "report" is a spoiler-bounded episode recap, the "buyer" is a
Love Is Blind viewer catching up week-to-week.

- **Stack decision**: [`stack_decision.md`](./stack_decision.md) — LangGraph, and why n8n is out of scope for this MVP
- **Future GTM sprints**: [`gtm_future_sprints.md`](./gtm_future_sprints.md)
- **Current project plan**: [`project_plan/planning.md`](./project_plan/planning.md)
- **Sample reports**: [`samples/`](./samples)
- **Full version history (V0–V4)**: [`versioning/`](./versioning)

---

## What it does

Given `edition`, `season`, `episode`, the agent:

1. Fetches canonical episode titles and cast metadata (OMDb + TMDB).
2. Runs a ReAct agent that plans and executes its own web searches (Tavily).
3. Cross-checks fetched sources against a hand-verified season index
   (ground truth for episode numbering and cast bios).
4. Ranks and selects sources per category (bios / highlights / drama /
   reaction) under a fixed budget.
5. Fetches YouTube comments, strictly range-gated so no comment about a
   future episode leaks in — running concurrently with steps 6-7 below,
   since neither depends on the other's output.
6. Chunks and indexes everything into Pinecone, tagged by
   `{edition, season, episode_number, phase}`.
7. Retrieves against that index, filtered to `episode_number ≤ cutoff` —
   this is what structurally enforces the spoiler boundary, not just a
   prompt instruction.
8. Synthesizes fan reaction from the retrieved comments (joins the two
   parallel branches above — the first step needing both YouTube comments
   and retrieved context).
9. Generates the structured recap.
10. Audits the draft in a dedicated spoiler-check node; on failure, routes
    back to regenerate (LangGraph conditional edge) instead of shipping a
    flawed draft.

Output: intro, season-wide main drama, ranked highlights from the last
episode, audience reaction (liked / criticized / themes / quotes),
participant list (name, age, occupation), sources, conclusion — written in
narrator-voice prose, not a flat data dump.

## Architecture

```
                          USER (edition, season, last episode watched)
                                        │
                                        ▼
                       ┌────────────────────────────┐
                       │  fetch_show_metadata         │  OMDb + TMDB: canonical
                       └──────────────┬───────────────┘  episode titles & cast
                                      │
                                      ▼
                       ┌────────────────────────────┐
                       │  plan_and_search              │  ReAct agent (LangGraph
                       └──────────────┬───────────────┘  prebuilt) + Tavily search
                                      │
                                      ▼
                       ┌────────────────────────────┐
                       │  load_season_index            │  hand-verified ground
                       └──────────────┬───────────────┘  truth (episodes + cast)
                                      │
                                      ▼
                       ┌────────────────────────────┐
                       │  rank_and_select               │  per-category budgets
                       └──────────────┬───────────────┘  (bios/highlights/drama/reaction)
                                      │
                      ┌───────────────┴────────────────┐
                      ▼  (parallel branches, fan-out)   ▼
       ┌────────────────────────────┐    ┌────────────────────────────┐
       │  fetch_youtube_comments        │    │  index                         │
       └──────────────┬───────────────┘    └──────────────┬───────────────┘
        range-gated to the user's cutoff     chunk + tag + embed → Pinecone
                      │                                    ▼
                      │                     ┌────────────────────────────┐
                      │                     │  retrieve                       │
                      │                     └──────────────┬───────────────┘
                      │                    4 targeted Pinecone queries, cutoff-filtered
                      └───────────────┬────────────────────┘
                                      ▼  (fan-in, both branches joined)
                       ┌────────────────────────────┐
                       │  analyze_fan_reaction           │  synthesize YouTube
                       └──────────────┬───────────────┘  comments → structured reaction
                                      │
                                      ▼
                       ┌────────────────────────────┐
                  ┌───▶│  generate                       │  writes the recap draft
                  │    └──────────────┬───────────────┘
                  │                   ▼
                  │    ┌────────────────────────────┐
                  │    │  spoiler_check                  │  audits draft against cutoff
                  │    └──────────────┬───────────────┘
                  │                   │
                  │        issues found?  │  clean?
                  └────────  retry ───────┘──────▶ END → structured recap returned
```

10 LangGraph nodes, typed state (`RecapState`) threaded through all of them,
one genuine conditional edge (`spoiler_check` → retry `generate` or `END`).
`fetch_youtube_comments` and `index`→`retrieve` fan out from
`rank_and_select` and run concurrently (they both depend only on
`selected_sources`, not on each other), joining at `analyze_fan_reaction`,
the first node that needs both `youtube_comments` and `context`. `index` is
the far longer chain (LLM tagging + embeddings + Pinecone), so this hides
the YouTube comment fetch's wall-clock almost entirely behind it.
See [`stack_decision.md`](./stack_decision.md) for why LangGraph over n8n.

`plan_and_search`'s ReAct/Tavily results are cached to disk (`.search_cache/`,
gitignored), keyed per `(edition, season, episode)` with a 1-week TTL — a
repeat request for the same episode skips the entire search loop rather than
re-running it. Deliberately keyed per-episode rather than per-season: the
node runs a mandatory query targeting the specific episode cutoff, reusing
another episode's cached results would silently miss that coverage.

`index`'s per-chunk LLM tagging and embedding calls are also cached to disk
(`.index_cache/`, gitignored), keyed by a hash of `(model, chunk text)` —
content-addressed, not run- or episode-scoped, so a source re-selected for a
different episode of the same season (e.g. a "meet the cast" article) still
hits the cache. A hit replays a real prior tag/embedding for that exact
text, it changes nothing about what gets written to Pinecone or how the
namespace is cleared and rebuilt each run — only the redundant OpenAI calls
before that write are skipped.

## Tools / APIs (4, exceeds the ≥3 requirement)

| Tool | Used for |
|---|---|
| Tavily | Live web search for recap/cast/reaction content |
| TMDB | Per-episode cast metadata |
| OMDb | Canonical episode titles |
| YouTube Data API | Comment fetch for fan-reaction analysis |

Plus **OpenAI** (generation + embeddings) and **Pinecone** (RAG vector
store) as core infrastructure, not "tools" in the agentic sense.

## Grounding / RAG

Pinecone, chunked and tagged by `{edition, season, episode_number, phase}`,
queried with a hard filter `episode_number ≤ cutoff`. This is the actual
mechanism enforcing the spoiler boundary — the `spoiler_check` node is a
second, independent audit layer on top of it, not the only line of
defense.

## Setup

Requires Python 3.10+ (the codebase uses `X | None` union-type syntax
throughout; on Python 3.9 this breaks at runtime when LangChain introspects
tool function signatures, not just at import time).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
PINECONE_API_KEY=...
TMDB_API_KEY=...
OMDB_API_KEY=...
YOUTUBE_API_KEY=...
DAILY_BUDGET=            # optional, USD; unset or 0 disables spend enforcement
```

No secrets are committed — `.env` is gitignored.

## Run it

**Option A: CLI**
```bash
python recap.py --edition Poland --season 1 --episode 6
```
Optional flags: `--edition` (default `Poland`), `--season` (default `1`), `--episode` (required).

**Option B: web UI**

Terminal 1:
```bash
python app.py
```
Terminal 2:
```bash
open recap_ui.html
```
Pick an edition, season, and last episode watched, then generate. A real
run takes roughly 30–90 seconds (multiple live API calls plus 2–3 LLM
calls) and costs a few cents (tracked per-run in `.spend_ledger.json`).

## Known limitations

- Participant age/occupation can come back missing for less-documented
  editions/seasons even when a source has the info — single-pass
  extraction over noisy web sources isn't perfectly reliable; the season
  index ground truth mitigates but doesn't eliminate this.
- Source coverage varies by edition popularity; mainstream editions (US,
  UK, Poland) have denser recap coverage on the open web than smaller
  editions.
- `n8n` is not implemented — LangGraph is the sole orchestration layer for
  this MVP; see [`stack_decision.md`](./stack_decision.md) for why.

## File map

```
README.md                  # this file
stack_decision.md            # LangGraph vs n8n justification
gtm_future_sprints.md         # 3 post-MVP go-to-market sprints
requirements.txt

app.py                     # Flask API for the UI
recap.py                    # CLI entrypoint
recap_ui.html                 # standalone frontend, calls app.py
graph.py                       # RecapState, LangGraph assembly, run_pipeline()
config.py                       # model/index/path constants
research.py                # ReAct planner agent + source ranking
metadata.py                 # OMDb + TMDB fetch
season_index.py               # hand-verified ground-truth CSV loaders
indexing.py                     # chunk/tag/embed → Pinecone upsert
retrieval.py                      # Pinecone query nodes
youtube.py                  # YouTube comment fetch + fan-reaction synthesis
generation.py                # recap draft generation
spoiler_check.py               # spoiler audit + retry routing
cost_tracker.py                  # per-run spend ledger + budget guard
logging_config.py                  # structured logging setup

season_indexes/             # shared ground-truth data (all editions/seasons)
  episodes/                  # episode number ↔ title per edition/season
  cast/                       # participant name/age/occupation per edition/season
  imdb_ids.csv, tmdb_ids.csv    # per-edition external ID lookups

samples/                    # ≥2 generated sample reports (required deliverable)
project_plan/
  planning.md                # current-state project plan (this version)

versioning/                 # full build history, preserved for reference
  MVP/                        # V0 — original single-source MVP + its original plan
  V1/, V2/, V3/                 # incremental builds (RAG, ReAct, OMDb/TMDB, fan reaction)
  V4/                             # archived copy of the code now at repo root
```
