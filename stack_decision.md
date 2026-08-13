# Stack Decision: LangGraph (primary)

## Decision

**Primary orchestration: LangGraph.** n8n was evaluated and is not used —
not even as a thin optional trigger — for reasons specific to this
workflow's shape.

## Why LangGraph fits this problem

1. **A genuine conditional edge, not just tool-calling.** The core
   correctness requirement of this product is the spoiler boundary: a
   recap must never reference anything past the user's stated cutoff
   episode. That's enforced with a dedicated `spoiler_check` node that
   audits the drafted output and, on failure, routes back to `generate`
   for another attempt rather than shipping a flawed draft
   ([`graph.py`](graph.py)). This is exactly the case LangGraph is
   built for — real branching logic driven by the content of the state,
   not a fixed pipeline with optional retries bolted on. n8n's branching
   (IF/Switch nodes) is aimed at routing between different external
   systems or event types, not at looping an LLM output through a
   quality gate with shared typed state.

2. **Rich, evolving typed state threaded through 10 steps.** `RecapState`
   carries edition/season/episode, fetched metadata, ranked sources,
   YouTube comments, RAG chunks, the draft, and spoiler-check results
   across the whole run. LangGraph's state object is a first-class,
   type-checked citizen of the code; representing the same thing in n8n
   would mean passing a growing JSON blob between nodes with no schema
   enforcement, which gets fragile fast at this level of nesting.

3. **A real ReAct loop, not scripted API calls.** The research-planning
   node uses `langgraph.prebuilt.create_react_agent`
   ([`research.py`](research.py)) — the agent decides its own
   search queries based on what it's already found, rather than following
   a fixed call sequence. This reason-act loop is native to
   LangChain/LangGraph's tool-calling model; replicating it in n8n would
   mean either faking it with static branches (defeats the purpose) or
   wrapping the same LangGraph agent in an n8n Code node anyway, which
   just makes n8n a redundant wrapper.

4. **Cost and latency profile favors code over visual orchestration.**
   With 10 nodes, several of them LLM calls and several parallelizable
   I/O calls (TMDB/OMDb/YouTube/Tavily), fine-grained control over
   concurrency, retries, and per-run budget enforcement
   (`cost_tracker.py`) is easier to write, test, and reason about as
   Python than as a canvas of nodes and expressions.

5. **Development speed for a single developer.** This is a solo,
   code-first build. Writing and debugging Python with a debugger and
   normal version control is faster for one person than building and
   iterating on a visual workflow, especially once the graph has
   conditional routing and typed state.

## Why n8n is out of scope for this MVP (not just "secondary")

The brief allows n8n as a thin optional trigger/webhook layer in front of
a LangGraph brain. This project's original plan (see
[`versioning/MVP/planning.md`](versioning/MVP/planning.md) §2) assumed n8n would be used this
way — a webhook triggering an Execute Command node that runs the Python
pipeline, mainly for visual run history and a free HTTP endpoint. In
practice this was dropped for V1 onward, for three reasons:

- **No second trigger surface is actually needed.** The product already
  has two real interfaces — a CLI for scripted/autonomous runs and a
  Flask API + web UI for interactive use. A webhook-only n8n layer in
  front of the same Python entrypoint would duplicate what Flask already
  does, for a cost tracker/run-history benefit that's already covered by
  `cost_tracker.py` and structured logging (`logging_config.py`).
- **The "business system integration" case n8n is good at doesn't exist
  here yet.** n8n earns its place when a workflow needs to talk to many
  external SaaS systems with pre-built connectors (Slack, email, CRMs,
  spreadsheets) around an AI core. This MVP's external calls are all
  direct API integrations already coded against directly (Tavily, TMDB,
  OMDb, YouTube, Pinecone, OpenAI) — none of them benefit from an n8n
  connector node over a direct SDK/HTTP call. That need shows up in the
  future GTM sprints (email delivery, social sharing — see
  [`gtm_future_sprints.md`](gtm_future_sprints.md)), which is exactly
  where n8n gets reconsidered as a lightweight glue layer, not before.
- **Avoiding a split brain.** Splitting orchestration across two systems
  (n8n for triggering/retries, LangGraph for the actual agent logic) adds
  an operational seam — two places to look when something fails — for a
  single-developer MVP where that seam buys nothing yet.

**Conclusion**: LangGraph is the primary and only orchestration layer for
the MVP. n8n is a reasonable future addition once the product needs to
integrate with external delivery/distribution systems (see GTM sprints),
not before.
