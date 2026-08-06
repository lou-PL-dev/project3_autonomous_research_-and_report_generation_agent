# Autonomous Agent Project Plan

## 1. Use Case

**Project**: "Previously On: Love Is Blind" — autonomous, spoiler-bounded summary generator.

**Problem statement**: As a Love Is Blind viewer who watches week by week (or catches up later), I forget prior drama between viewings, and I want a quick, readable summary of what happened in the season I'm watching, right before an episode or between watch sessions. 

**Existing solutions**: Existing summary content (YouTube videos, articles) is too long, inconsistent in style and coverage across seasons and editions, and doesn't reliably match a specific viewer's progress, creating either spoilers or an information gap.

**Target users**: Love Is Blind viewers generally. Audience data (Nielsen, Vault) shows the show skews heavily female (73%) and 30+ (60% aged 35 to 64), watching primarily for relationship drama and emotional engagement, an audience where entertainment value matters as much as recall.

**Verify**:
- MVP level: summary is tonally consistent and recognizably comic/dramatic across repeated runs, correctly bounded to the user's cutoff (zero spoilers past it), returns structured output.
- Final Product level: the user and a friend actually use it unprompted the following week, in preference to a 20 minute YouTube commenter video or a web search.

**Current existing solution(s)**: Informally asking other viewers, occasionally checking a YouTube video. Coverage and tone vary by creator, season, and edition, and there is no reliable way to get a summary bounded to the user's own progress.

**Industry alignment**: Positioned under Media, Journalism & Publishing (content trend and audience reaction analysis for entertainment media).

**Competitors and competitive advantage**: Amazon Prime Video ships a native equivalent (X-Ray summarys, spoiler-free text summaries; Video summarys, AI-narrated video summarys), but it's Prime-only,limited to select scripted Prime Originals, and tonally neutral. My product edge is a comic narrator voice, inclusion of fan reaction alongside plot facts specifically for the show Love is Blind.


**User story**:
As a Love Is Blind viewer, I want to tell my edition, season, and the last episode I watched, so that I get a summary tailored exactly to where I've stopped, without spoilers or hunting through inconsistent YouTube videos and articles.

**Acceptance criteria**:
- Given I provide my edition, season, and last episode watched, then I receive a summary covering everything up to and including that episode, not just that episode alone, with a tone that fits with the love is Blind Reality Show drama.
- Given the summary, when I read it, then it includes: participant bios and behavior so far, a highlight of my last episode, the main drama of the season so far, and the sources this information came from.
- Given the show naturally unfolds in phases (Pods, Reveal, Honeymoon, Moving in together, Wedding, Reunion, After the show), when I get my summary, then its focus matches the phase my last episode falls into.
- Given I ask for "After the show", when I get my summary, then I see where the couples stand today.
---

## 2. Technology Stack

**Framework answers**:
- Needs external knowledge: Yes → RAG.
- Interacts with external systems: Yes → tool integrations.
- Needs multi step reasoning: Yes → LangGraph plus ReAct.
- Integrates with business systems: Yes → n8n (project requirement, also aids debugging/orchestration).
- Autonomous: Yes → needs error handling, retry logic, validation.

**Stack**:
- Core LLM: OpenAI.
- RAG: Pinecone, chunks tagged by `{edition, season, episode_number, phase}`, retrieval filtered by `edition = user's edition AND season = user's season AND episode_number ≤ cutoff`.
- Agent framework: LangChain (ReAct tool orchestration) + LangGraph (state management, conditional routing for the spoiler check loop).
- Orchestration: n8n, webhook trigger → Execute Command node → Python script.
- Tools/integrations (MCP): TMDB (metadata), YouTube Data API (video discovery + comments), youtube-transcript-api (transcripts), web search API (articles).

**Justification**:
- LangGraph over a simple agent loop: the workflow has a genuine conditional edge (regenerate on spoiler check failure), not just tool calling.
- Pinecone/RAG: justified by reuse (index once per episode, query repeatedly) and distillation (multiple sources per episode need semantic filtering).
- YouTube API + youtube-transcript-api over official captions: official caption download requires OAuth and video ownership, which fails for third party videos.
- n8n Execute Command: required by the project rubric; technically it saves writing a small web server, since it gives a webhook endpoint and visual run history for free around the existing Python script.

**Alternatives considered and dropped**:
- Reddit API: official access requires manual approval (2 to 4 week wait), and non commercial self serve registration is closed as of 2025, too fragile for a 5 day build. YouTube comments serve as a legally accessible substitute for fan reaction data.
- Rotten Tomatoes API: effectively closed to individual developers ($60k/year, 60 day approval).
- OMDb API: considered for ratings enrichment, dropped as redundant given the other sources already cover the need.
- Direct web scraping (Reddit, pricing pages, etc.): rejected in favor of official APIs and general web search, since scraping is fragile against ToS enforcement and site structure changes.
- Dedicated translation API (e.g. DeepL): considered for multilingual editions, deferred. The LLM can read non English source text natively during generation without a separate translation step for MVP.

---

## 3. MVP Scope

**MVP features (must have)**:
- Single source: web search only.
- Love Is Blind Poland Season 1 Episode 6, hardcoded.
- Cumulative summary: covers everything up to and including the user's requested episode, not the episode in isolation.
- Structured output, must have categories: `participants`, `highlights`, `main_drama`, `sources`, wrapped in storytelling, tone specific sentences rather than a flat data dump.
- Terminal output only.

**Excluded from MVP**: multiple sources, RAG/Pinecone, LangGraph, spoiler check node, N8N, automated phase tagging, after show module, mini chatbot, PDF export, automated edition/season selection (a dropdown UI letting the user pick edition and season, auto populated via TMDB rather than hardcoded, planned for a later version).

**Justification**: proves the core value question, whether a good, on tone, correctly bounded summary can be generated at all, before investing in infrastructure (RAG, LangGraph, N8N). If summary quality or the spoiler boundary fails at this simple level, no amount of architecture fixes that. The most risky part is the retrieval of multiple sources with spoiler cut off though, but I am deciding to first to a small product POC before digging in the hardest part.

**MVP specific success metrics**:
- Tone recognizability and consistency across repeated runs for the same episode input.
- Zero spoiler leakage past the requested episode.
- All four required fields correctly populated, wrapped in narrator voice prose rather than a flat list.
- Source coverage accuracy (no missing major beats, no hallucinated URLs).

---

## 4. Risk Assessment

| Category | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| Technical | LLM hallucinates plot details not in sources | Medium | High | Ground generation strictly in retrieved content, spoiler check and fact check pass in V3 |
| Technical | Structured output schema breaks | Medium | Medium | Validate output against schema, retry on failure |
| Technical | API rate limits hit mid run | Low to Medium | Medium | Cache results per episode, monitor quota usage |
| Data | summary content sparse for less mainstream editions | Medium | Medium | Confirmed V1 scope is US only, revisit before expanding |
| Data | Sources conflict on details | Medium | Low to Medium | Generation prompt instructed to reconcile/flag, not silently pick one |
| Data | Phase tagging inference errors (wrong episode assigned to wrong phase) | Medium | Medium | Spoiler check step also validates phase boundaries before output |
| Legal/Copyright | Reproducing transcript or article text verbatim | Medium | Medium | Summarize/paraphrase only, never quote at length |
| Legal/Copyright | YouTube ToS on API use | Low | Medium | Official API endpoints only, respect quota and terms |
| Legal/Copyright | Netflix/Love Is Blind trademark and IP exposure (franchise name, branding, imagery) | Medium | Medium | Frame as non commercial personal/fan project, avoid reproducing official Netflix imagery or logos, text only branding references |
| Business/Scope | Scope creep (chatbot, extra sources) eating into core requirement time | High | High | Must haves locked per version table, Could items only after Musts pass |
| Business/Scope | Spoiler boundary failing breaks the core value proposition | Medium | High | Dedicated LangGraph spoiler check node, not just a prompt instruction |
| Business | summary not distinct/funny enough to be worth using vs. existing YouTube summarys | Medium | High | Early friend test on Day 3 validates this before further infra investment |

---

## 5. Implementation Plan

## 5. Implementation Plan

**Risk first order**: rather than following the brief's suggested RAG then LangGraph sequence, LangGraph and the spoiler check node are built right after the MVP, ahead of multi source RAG. The core project risk is not infrastructure, it is whether the spoiler boundary can be reliably enforced and whether the workflow structure holds up at all. Validating that early, on top of the MVP's simple single source content, surfaces a fundamental failure after one phase instead of after three.

| Version | Main version goal | Hypothesis tested | Proof of hypothesis | Scope |
|---|---|---|---|---|
| V0 (MVP) | Prove summary is viable | A good, on-tone, spoiler-bounded summary can be generated at all | Manual review against MVP success metrics: tone, zero leakage, structured fields, source accuracy | Single source (web search), hardcoded show/season, cumulative summary, structured output |
| V1 | Prove spoiler boundary holds | The spoiler boundary can be enforced structurally, not just by prompt instruction | Spoiler check node catches injected/known violations, zero leakage across test cutoffs | LangGraph state machine, ReAct pattern, spoiler check conditional edge |
| V2 | Prove more sources help | Adding sources and RAG measurably improves summary quality over single source | A/B comparison of V0/V1 output vs. V2 output, source coverage accuracy metric | Add TMDB and YouTube transcripts (3 MCP tools total), Pinecone/RAG with phase tagging, external friend test |
| V3 | Prove autonomous deployment works | The agent can run end to end via a real trigger with no manual intervention | Webhook triggers a full run with no manual step, forced failure recovers via retry | N8N webhook, Execute Command node, error handling and retry logic, refine from friend feedback |
| V4 | Prove system is demo ready | The system is understandable and usable by someone other than the builder | Final autonomous end to end run succeeds, demo clearly explains trigger to report flow | Polish, README, architecture diagrams, demo prep (slides and demo) |

**Timeline**:
- **Today**: planning, research, API key setup. MVP.
- **Day 1**: V0.
- **Day 2**: V1.
- **Day 3**: V2, including the external friend test.
- **Day 4**: V3.
- **Day 5**: V4, including final autonomous end to end test and demo preparation (slides and demo).

**Dependencies**: V1's spoiler check depends on V0's content fetching pipeline; V2's RAG and phase tagging depend on V1's LangGraph state structure being stable; V3's N8N depends on V2's Python agent being stable; demo depends on V3 completing successfully.

---

## 6. Success Metrics

**Delivery / technical**:
- All required components integrated and functional: ReAct, LangGraph, RAG with Pinecone, 3+ MCP tools, N8N via Execute Command node.
- 100% of test runs complete end to end with zero manual intervention.
- Tool call error rate below an agreed threshold, with retries recovering the rest.
- 2 to 3 generated report examples produced, across different inputs.
- README, architecture diagrams, and workflow documentation complete.

**As a user**:
- 0 spoilers past the stated cutoff, across all tested episode inputs.
- 100% of generated summarys include a valid source list.
- Friend tester rates the summary's tone as "clearly comic/dramatic" (not neutral) on a majority of test reads.
- Friend tester says yes when asked "would you use this again next week?" during the Day 3 test.
