# Previously On: Love Is Blind — V0 (MVP)

> **Archived version.** This is the original Day-1 MVP, preserved as-is for
> history. For the current, submitted version of the project, see the
> [repo root README](../../README.md). The full current-state plan is at
> [`project_plan/planning.md`](../../project_plan/planning.md); this
> version's original planning doc is the sibling
> [`planning.md`](./planning.md) in this same folder.

Autonomous, spoiler-bounded recap generator for Love Is Blind. Tell it your edition, season, and last episode watched, get back a narrator-voiced recap covering the season so far, without spoiling anything past your cutoff.

Full original plan: [`planning.md`](./planning.md)
This version's code: [`summary.py`](./summary.py) (CLI) or [`recap_ui.html`](./recap_ui.html) (UI)

## Current state: V0 (MVP)

Proves the core question: can a good, on-tone, spoiler-bounded recap be generated at all, before investing in RAG, LangGraph, or N8N.

**What it does:**
- Single source (Tavily web search)
- Cumulative recap: covers everything up to and including the requested episode, not just that episode alone
- Structured output: intro, season-wide main drama, ranked episode highlights, audience reaction, participant list (name/age/occupation), sources, conclusion
- Conversational, hyped narrator tone ("excited friend texting about the show"), grounded strictly in fetched sources (no invented names or claims)
- Runs as a CLI script or through a small local web UI

**What it doesn't do yet** (planned for later versions, see the plan):
- Multiple sources / RAG (Pinecone)
- LangGraph + ReAct + a dedicated spoiler-check node
- N8N orchestration
- Automated phase tagging, "After the show" module, multi-edition validation beyond Poland S1

## Setup

```bash
cd versioning/MVP
pip install python-dotenv openai tavily-python flask flask-cors
```

Create a `.env` file in the repo root with:
```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

## Run it

**Option A: command line**
```bash
python versioning/MVP/summary.py --episode 6
```
Optional flags: `--edition Poland` (default), `--season 1` (default).

**Option B: web UI**

Terminal 1, start the backend:
```bash
python versioning/MVP/app.py
```
Terminal 2, open the page:
```bash
open versioning/MVP/recap_ui.html
```
Pick an edition, season, and last episode watched, then click "SPILL THE TEA." A real generation takes 30 to 90 seconds (multiple live searches plus an LLM call).

## Known limitations (V0)

- Only validated against Love Is Blind Poland, Season 1. Other editions/seasons will run but source coverage is unconfirmed.
- Participant age/occupation can come back "unknown" even when a source has the info, since a single prompt pass over unchunked search results doesn't reliably extract every detail from long or noisy sources. This is the concrete case for RAG/chunking in the next version, not a bug to prompt-engineer around at this stage.
- Name spelling and specific claims can vary slightly between runs on live search, since results aren't cached or pinned.
- Debug logs (`debug_context_episode_N.txt`) are written to the working directory on each CLI run for manual grounding checks, not meant to be committed.

## Folder structure (this version)

```
versioning/MVP/
  summary.py        # core pipeline: fetch sources, generate recap (CLI entry point)
  app.py             # Flask API wrapping summary.py for the web UI
  recap_ui.html       # standalone frontend, calls app.py
  planning.md          # this version's original use case, tech stack, MVP scope, risks, timeline
  README.md             # this file
```