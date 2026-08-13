# Project Plan — Current State (V4)

> This is the live project plan, current as of last version in root folder. The original Day-1 plan
> (use case, initial stack reasoning, MVP scope, risk table as first drafted)
> is preserved unchanged at [`versioning/MVP/planning.md`](../versioning/MVP/planning.md) for
> history. This document reflects what was actually built and decided along
> the way; where a decision changed from the original plan, that's called
> out explicitly rather than silently rewritten.

## 1. Use Case (unchanged from original plan)

**Project**: "Previously On: Love Is Blind" — autonomous, spoiler-bounded
recap generator.

**Problem**: Love Is Blind viewers who watch week-by-week or catch up later
lose track of prior drama between viewings. Existing options (YouTube
recaps, articles) are too long, inconsistent in tone/coverage across
editions and seasons, and not bounded to the specific episode a viewer has
reached — creating either spoilers or an information gap.

**Target users**: Love Is Blind viewers generally, skewing female (73%) and
35–64 (60%) per Nielsen/Vault audience data, watching primarily for
relationship drama and emotional engagement rather than plot recall alone —
tone and entertainment value matter as much as accuracy.

**Industry alignment**: Media, Journalism & Publishing — content/coverage
synthesis and audience-reaction analysis for entertainment media.

**Competitive landscape**: Amazon Prime Video ships a native equivalent
(X-Ray text summaries, AI-narrated video recaps) but it's Prime-only,
limited to select scripted Prime Originals, and tonally neutral. This
product's edge: a comic narrator voice, fan reaction folded in alongside
plot facts, and coverage of a specific unscripted franchise Prime doesn't
serve this way.

Full original problem statement, user story, and acceptance criteria:
see [`versioning/MVP/planning.md`](../versioning/MVP/planning.md) §1 — unchanged.

## 2. Technology Stack — as built

**Primary stack: LangGraph.** Full justification: [`stack_decision.md`](../stack_decision.md).

**Stack in production (V4)**:
- **Core LLM**: OpenAI (`gpt-4o` for generation/fan-reaction synthesis,
  `gpt-4o-mini` for research planning, chunk tagging, spoiler audit — split
  by task cost-sensitivity, see [`config.py`](../config.py)).
- **Agent orchestration**: LangGraph `StateGraph` with 10 nodes and one
  conditional edge (spoiler-check retry loop). See
  [`graph.py`](../graph.py).
- **Tool-use pattern**: ReAct, via `langgraph.prebuilt.create_react_agent`,
  in the research-planning node — the agent decides its own search queries
  rather than following a fixed query template. See
  [`research.py`](../research.py).
- **RAG**: Pinecone. Chunks tagged `{edition, season, episode_number,
  phase}`; retrieval filtered by `edition = user's edition AND season =
  user's season AND episode_number ≤ cutoff`. This is the mechanism that
  structurally enforces the spoiler boundary, not just the spoiler-check
  node.
- **Tools/integrations** (4, exceeds the ≥3 requirement):
  - Tavily (web search)
  - TMDB (episode/cast metadata)
  - OMDb (canonical episode titles)
  - YouTube Data API (comment fetch for fan-reaction analysis, filtered to
    videos whose entire covered range is ≤ the user's cutoff)
- **Cost/budget guard**: a run-level spend ledger and pre-run budget check
  (`cost_tracker.py`) that refuses a run outright if it would exceed
  budget, rather than failing mid-run.
- **Orchestration/trigger surface**: Flask API (`app.py`) + static UI
  (`recap_ui.html`) for interactive use, plus a plain CLI (`recap.py`) for
  scripted/autonomous triggering. **n8n was not built.** See
  [`stack_decision.md`](../stack_decision.md) for why — this is a change
  from the original plan, which listed n8n as a requirement rather than an
  optional helper.

**Changes from the original plan**:
- n8n dropped entirely (see stack_decision.md) — the original plan
  mis-read the brief as requiring n8n alongside LangGraph; it's actually
  optional once LangGraph is the chosen primary stack.
- Metadata sourcing split across two services (TMDB + OMDb) instead of
  TMDB alone, once OMDb turned out to have more reliable canonical episode
  titles.
- Added a cost/budget guard node that wasn't in the original MVP scope,
  after early runs made API spend visibility a real operational concern.

**Alternatives considered and dropped**: unchanged from the original plan
(Reddit API, Rotten Tomatoes API, OMDb-as-redundant, direct scraping,
dedicated translation API) — see [`versioning/MVP/planning.md`](../versioning/MVP/planning.md) §2.

## 3. Scope — current state vs. original MVP scope

The original MVP scope (single source, one hardcoded edition/season,
terminal-only output) is superseded. Current scope:

- Multi-source: Tavily search + TMDB + OMDb + YouTube comments, merged via
  source-first RAG with per-category budgets (bios/highlights/drama/reaction).
- All editions and seasons supported (not hardcoded to Poland S1), backed
  by a hand-verified season-index CSV (episodes + cast) as ground truth
  that the research agent's fetched sources are checked against.
- Structured output: intro, season-wide main drama, ranked episode
  highlights, audience reaction (from YouTube comment analysis), full
  participant list (name/age/occupation), sources, conclusion — wrapped in
  narrator-voice prose, not a flat data dump.
- LangGraph state machine with a dedicated spoiler-check node and a
  conditional retry edge — the spoiler boundary is enforced structurally,
  not just by prompt instruction.
- Two interfaces: CLI (`recap.py`) and Flask API + web UI (`app.py` /
  `recap_ui.html`), the latter with live per-node progress reporting.
- Cost tracking and a hard budget cap per run.

**Still excluded / not yet built**: n8n orchestration (see
stack_decision.md — deliberately dropped, not deferred), scheduled/email
delivery, social sharing, interactive features (tagging, predictions/voting)
— these are scoped into the future GTM sprints, see
[`gtm_future_sprints.md`](../gtm_future_sprints.md), not the current MVP.

## 4. Risk Assessment

Unchanged from the original plan (first drafted at
[`versioning/MVP/planning.md`](../versioning/MVP/planning.md) §4) and still
the operative table — reproduced here in full so this document is readable
on its own:

| Category | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| Technical | LLM hallucinates plot details not in sources | Medium | High | Ground generation strictly in retrieved content, spoiler check and fact check pass in V3 |
| Technical | Structured output schema breaks | Medium | Medium | Validate output against schema, retry on failure |
| Technical | API rate limits hit mid run | Low to Medium | Medium | Cache results per episode, monitor quota usage |
| Data | Recap content sparse for less mainstream editions | Medium | Medium | Confirmed V1 scope is US only, revisit before expanding |
| Data | Sources conflict on details | Medium | Low to Medium | Generation prompt instructed to reconcile/flag, not silently pick one |
| Data | Phase tagging inference errors (wrong episode assigned to wrong phase) | Medium | Medium | Spoiler check step also validates phase boundaries before output |
| Legal/Copyright | Reproducing transcript or article text verbatim | Medium | Medium | Summarize/paraphrase only, never quote at length |
| Legal/Copyright | YouTube ToS on API use | Low | Medium | Official API endpoints only, respect quota and terms |
| Legal/Copyright | Netflix/Love Is Blind trademark and IP exposure (franchise name, branding, imagery) | Medium | Medium | Frame as non-commercial personal/fan project, avoid reproducing official Netflix imagery or logos, text-only branding references |
| Business/Scope | Scope creep (chatbot, extra sources) eating into core requirement time | High | High | Must-haves locked per version table, could-items only after musts pass |
| Business/Scope | Spoiler boundary failing breaks the core value proposition | Medium | High | Dedicated LangGraph spoiler-check node, not just a prompt instruction |
| Business | Recap not distinct/funny enough to be worth using vs. existing YouTube recaps | Medium | High | Early friend test on Day 3 validates this before further infra investment |

**Status update, current as of V4**: one risk has moved from theoretical to
live and is now tracked as a business decision rather than a pure
mitigation item — **Netflix/Love Is Blind IP exposure**. Current position:
treat a cease-and-desist as a validation signal rather than a pure
downside (see Prime Video's own X-Ray/recap feature as precedent that this
content category is viable), with acquisition or a hiring conversation
with Netflix as one possible exit path rather than a worst case to design
purely defensively around. Full detail in
[`gtm_future_sprints.md`](../gtm_future_sprints.md).

Also worth noting: the spoiler-boundary risk (row above, "spoiler boundary
failing breaks the core value proposition") is not fully closed by the
dedicated LangGraph spoiler-check node. Testing against Love Is Blind US
Season 1 Episode 1 found the node passing a recap that used a
participant's married surname from the show's finale — a real leak via the
name field, not the generated prose the node actually audits. Tracked as
an open follow-up, not yet mitigated.

## 5. Version history (delivered)

| Version | Goal | Delivered |
|---|---|---|
| V0 (MVP) | Prove summary is viable at all | Single source, hardcoded show/season, cumulative summary, structured output |
| V1 | Prove spoiler boundary holds structurally | LangGraph state machine, ReAct pattern, spoiler-check conditional edge |
| V2 | Prove more sources help | TMDB + YouTube comments added, Pinecone RAG with phase tagging |
| V3 | Prove autonomous end-to-end run | Modular node structure, OMDb added, fan-reaction analysis, full README + architecture diagram |
| V4 | Prove system is demo-ready and scalable | All editions/seasons (season-index ground truth replacing hardcoded show), cost tracker + budget guard, UI progress reporting, parallelized I/O nodes, logging |

## 6. Success Metrics — status

Delivery/technical (from original plan, §6):
- ✅ ReAct, LangGraph, RAG (Pinecone), 4 tool integrations, all present and functional.
- ⚠️ n8n: deliberately not built — see stack_decision.md.
- ✅ Cost tracking with hard budget cap per run.
- ◻ 2–3 generated report examples across different inputs — see
  [`samples/`](../samples), populated on an ongoing basis.
- ◻ README reflecting current (V4) state — in progress.

As-a-user metrics (spoiler leakage, tone, source list presence): validated
manually during development per the original plan's method; no changes to
the acceptance criteria.
