"""
V1: RAG (Pinecone) + LangGraph pipeline for the Love Is Blind recap generator.

Still single-source (Tavily web search) until TMDB/YouTube keys are added.

Usage (CLI):
    python main/recap.py --episode 6
"""

import json
import os
from collections import Counter
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

DEFAULT_EDITION = "Poland"
DEFAULT_SEASON = 1

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
PINECONE_INDEX_NAME = "love-is-blind-recaps"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# Raised from 6000 based on real measured transcript lengths (20k-70k chars for
# typical YouTube recap videos), the old cap was cutting the best sources down
# to 9-30% of their actual content.
MAX_CHARS_PER_SOURCE = 75000
MAX_SPOILER_RETRIES = 1

PHASES = ["Pods", "Honeymoon", "Moving In Together", "Wedding", "Reunion"]

# Structurally general sources: cast/show reference pages, never episode-specific.
GENERAL_DOMAINS = ["wikipedia.org", "themoviedb.org", "rottentomatoes.com", "imdb.com", "netflix.com"]

# Domains confirmed, across multiple runs, to return unusable content (SPA
# navigation chrome, no actual captions/article text). Excluded entirely at
# fetch time, since they were observed winning retrieval slots purely on
# repetitive boilerplate text.
NOISE_DOMAINS = ["tiktok.com", "spotify.com"]


def is_noise_domain(url: str) -> bool:
    return any(domain in url for domain in NOISE_DOMAINS)

# Extra name variants per edition, beyond the edition string itself (e.g.
# Poland's Polish-language title uses "Polska").
EDITION_ALIASES = {
    "poland": ["polska"],
}


def is_general_domain(url: str) -> bool:
    return any(domain in url for domain in GENERAL_DOMAINS)


def matches_edition(edition: str, title: str, content: str) -> bool:
    """Cheap wrong-show filter: does this source actually appear to be about the
    requested edition? A generic 'Love Is Blind' query pulls in other editions'
    recap content that looks on-topic to a keyword search but is entirely
    irrelevant, or actively misleading, for this edition/season.
    """
    terms = [edition.lower()] + EDITION_ALIASES.get(edition.lower(), [])
    haystack = (title + " " + content[:1500]).lower()
    return any(term in haystack for term in terms)


class RecapState(TypedDict):
    edition: str
    season: int
    episode: int
    phase: str  # set by node_retrieve from tagged chunk content, "unknown" until then
    sources: list[dict]
    chunks: list[dict]
    context: str
    draft: dict
    spoiler_issues: list[dict]  # [{"field": str, "moment_index": int|None, "issue": str}]
    spoiler_passed: bool
    attempts: int


# ---------------------------------------------------------------------------
# Node 1: fetch
# ---------------------------------------------------------------------------

def node_fetch(state: RecapState) -> dict:
    edition, season, episode = state["edition"], state["season"], state["episode"]
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    queries = [
        (f"Love is Blind {edition} season {season} cast participants ages professions", 8),
        (f"Love is Blind {edition} season {season} episode {episode} recap", 5),
        (f"Love is Blind {edition} season {season} episodes 1-{episode} recap", 8),
        (f"Love is Blind {edition} season {season} episode 1 through {episode} what happened so far", 5),
        (f"Love is Blind {edition} season {season} episode {episode} fan reaction discussion", 5),
    ]

    results = []
    seen_urls = set()
    skipped_wrong_edition = []
    skipped_noise = []
    for query, max_results in queries:
        response = tavily.search(query=query, search_depth="advanced", max_results=max_results, include_raw_content=True)
        for item in response.get("results", []):
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                title = item.get("title", "")
                if is_noise_domain(url):
                    skipped_noise.append((title, url))
                    continue
                raw = item.get("raw_content") or item.get("content", "")
                if not matches_edition(edition, title, raw):
                    skipped_wrong_edition.append((title, url))
                    continue
                results.append({
                    "title": title,
                    "url": url,
                    "content": raw[:MAX_CHARS_PER_SOURCE],
                    "_raw_length": len(raw),
                })

    if skipped_noise:
        print(f"[fetch] skipped {len(skipped_noise)} noise-domain sources (no usable content):")
        for title, url in skipped_noise:
            print(f"    - {title}: {url}")

    if skipped_wrong_edition:
        print(f"[fetch] skipped {len(skipped_wrong_edition)} sources not matching edition '{edition}':")
        for title, url in skipped_wrong_edition:
            print(f"    - {title}: {url}")

    print(f"[fetch] {len(results)} sources")
    for r in results:
        truncated_flag = " TRUNCATED" if r["_raw_length"] > MAX_CHARS_PER_SOURCE else ""
        print(f"  - raw={r['_raw_length']} used={len(r['content'])}{truncated_flag} | {r['title']}: {r['url']}")
    return {"sources": results}


# ---------------------------------------------------------------------------
# Node 2: index — chunk, tag (episode/phase via LLM, content-based), embed, upsert
# ---------------------------------------------------------------------------

def split_into_chunks(text: str) -> list[str]:
    """Fixed-size chunking with overlap, split on paragraph boundaries where possible.

    A single paragraph can itself be larger than CHUNK_SIZE (e.g. a transcript
    pasted as one dense block with no line breaks). Those get hard-sliced directly
    instead of ever becoming one oversized chunk, which previously could exceed
    the embedding model's token limit regardless of CHUNK_SIZE.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buffer = ""
    for para in paragraphs:
        if len(para) > CHUNK_SIZE:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            start = 0
            while start < len(para):
                chunks.append(para[start:start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
            continue
        if len(buffer) + len(para) <= CHUNK_SIZE:
            buffer += (" " if buffer else "") + para
        else:
            if buffer:
                chunks.append(buffer)
            overlap = buffer[-CHUNK_OVERLAP:] if buffer else ""
            buffer = (overlap + " " + para).strip()
    if buffer:
        chunks.append(buffer)
    return chunks or [text[:CHUNK_SIZE]]


def tag_chunks_batch(client: OpenAI, items: list[dict]) -> dict[tuple[int, int], dict]:
    """One LLM call across ALL sources that need tagging (skips known-general domains).

    items: [{"source_index": int, "chunk_index": int, "source_title": str, "text": str}]
    Returns a dict keyed by (source_index, chunk_index) -> {"episode_start": ..., "episode_end": ..., "phase": ...}.
    Matching by explicit IDs rather than array position, so a misaligned or missing
    response item just defaults that one chunk to null instead of shifting everything after it.
    """
    if not items:
        return {}

    numbered = "\n\n".join(
        f"[src={item['source_index']} chunk={item['chunk_index']}] (from: {item['source_title']})\n{item['text']}"
        for item in items
    )
    prompt = f"""Below are numbered text chunks from several sources about a Love Is Blind season.
For each chunk, determine:

- episode_start and episode_end: the range of episode numbers this chunk narrates events from.
  Many recap sources cover multiple episodes in one continuous narration (e.g. "Episodes 1-5
  Recap"), so a single episode number is often wrong for real content. If the chunk covers only
  one episode, set episode_start and episode_end to that same number. If it narrates a range
  (e.g. discusses events across episodes 2 through 4), set episode_start to the lowest and
  episode_end to the highest episode actually narrated. If not determinable, set both to null.
  Be conservative about WHICH episodes, a short caption or generic mention with a number isn't
  enough on its own, only include a range the chunk actually narrates. When in doubt, use null
  rather than guessing a range.

- phase: one of {", ".join(PHASES)}, or null if not determinable. Judge by observable content,
  not general knowledge of the show. Use these signals:
  - Pods: conversations happening through a wall/screen (contestants haven't seen each other),
    or the reveal moment itself (first time they see each other face to face).
  - Honeymoon: an engaged couple traveling together, typically abroad.
  - Moving In Together: couples living in a shared apartment, meeting each other's family or friends.
  - Wedding: dress or suit fittings, wedding preparation, vows, the ceremony itself.
  - Reunion: a separate post-finale special, cast answering questions about where they are now.
  If the chunk doesn't clearly match one of these signals, use null rather than guessing.
  IMPORTANT: a chunk covering a later episode range may still be narrating a FLASHBACK or
  retrospective mention of an earlier moment. In that case, tag the phase for the moment
  actually being described, not wherever the episode range would normally sit.

Chunks:
{numbered}

Return ONLY a JSON array, no markdown fences, one object per chunk:
[{{"source_index": int, "chunk_index": int, "episode_start": int_or_null, "episode_end": int_or_null, "phase": "string_or_null"}}]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    try:
        tags = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    return {(t["source_index"], t["chunk_index"]): t for t in tags if "source_index" in t and "chunk_index" in t}


def namespace_for(edition: str, season: int) -> str:
    return f"{edition.lower().replace(' ', '_')}_s{season}"


def node_index(state: RecapState) -> dict:
    client = OpenAI()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    namespace = namespace_for(state["edition"], state["season"])

    # Wipe this edition/season's namespace before indexing fresh. Without this,
    # every prior run's vectors stay in the index forever and keep competing in
    # retrieval alongside new, better data.
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"[index] cleared namespace '{namespace}' before indexing")
    except Exception as e:
        print(f"[index] namespace '{namespace}' clear skipped (likely didn't exist yet): {e}")

    source_chunks: list[list[str]] = []
    tag_queue = []
    for source_idx, source in enumerate(state["sources"]):
        chunks = split_into_chunks(source["content"])
        source_chunks.append(chunks)
        if not is_general_domain(source["url"]):
            for chunk_idx, chunk_text in enumerate(chunks):
                tag_queue.append({
                    "source_index": source_idx,
                    "chunk_index": chunk_idx,
                    "source_title": source["title"],
                    "text": chunk_text,
                })

    print(f"[index] {len(tag_queue)} chunks need tagging, "
          f"{sum(len(c) for c in source_chunks) - len(tag_queue)} skipped as general-domain")

    # Batching in one call for everything proved unreliable at volume: a single
    # 135-chunk call returned tags for only 80 of them, silently dropping the rest
    # to "general" regardless of what they actually contained. Capping batch size
    # trades a few more calls for actually getting full coverage.
    TAG_BATCH_SIZE = 30
    tags_by_id: dict[tuple[int, int], dict] = {}
    for batch_start in range(0, len(tag_queue), TAG_BATCH_SIZE):
        batch = tag_queue[batch_start:batch_start + TAG_BATCH_SIZE]
        batch_tags = tag_chunks_batch(client, batch)
        tags_by_id.update(batch_tags)
        print(f"[index]   batch {batch_start // TAG_BATCH_SIZE + 1}: "
              f"{len(batch_tags)}/{len(batch)} tagged")
    print(f"[index] tagging complete: {len(tags_by_id)}/{len(tag_queue)} chunks tagged overall")

    all_chunks = []
    vectors_to_upsert = []
    for source_idx, source in enumerate(state["sources"]):
        chunks = source_chunks[source_idx]
        if not chunks:
            continue
        embeddings = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)

        for chunk_idx, chunk_text in enumerate(chunks):
            tag = tags_by_id.get((source_idx, chunk_idx), {"episode_start": None, "episode_end": None, "phase": None})
            ep_start = tag.get("episode_start")
            ep_end = tag.get("episode_end")
            # Tagging LLM occasionally emits the string "null" instead of JSON null.
            # Normalize any such stray string to a real None before it gets stored.
            raw_phase = tag.get("phase")
            if isinstance(raw_phase, str) and raw_phase.strip().lower() in ("null", "none", ""):
                raw_phase = None
            chunk_id = f"{state['edition']}-{state['season']}-{source['url']}-{chunk_idx}".replace(" ", "_")[:512]
            metadata = {
                "edition": state["edition"],
                "season": state["season"],
                "episode_start": ep_start if ep_start is not None else -1,
                "episode_end": ep_end if ep_end is not None else -1,
                "phase": raw_phase or "unknown",
                "source_url": source["url"],
                "source_title": source["title"],
                "text": chunk_text,
            }
            all_chunks.append(metadata)
            vectors_to_upsert.append({
                "id": chunk_id,
                "values": embeddings.data[chunk_idx].embedding,
                "metadata": metadata,
            })

    # Upsert in batches: one request with all vectors can exceed Pinecone's 4MB
    # payload limit once volume is high. 200/batch stays safely under that with margin.
    UPSERT_BATCH_SIZE = 200
    for batch_start in range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE):
        batch = vectors_to_upsert[batch_start:batch_start + UPSERT_BATCH_SIZE]
        if batch:
            index.upsert(vectors=batch, namespace=namespace)

    by_range: dict[tuple[int, int], list[str]] = {}
    for c in all_chunks:
        by_range.setdefault((c["episode_start"], c["episode_end"]), []).append(c["phase"])
    print("[index] episode range -> phase tag breakdown:")
    for rng in sorted(by_range):
        phases = by_range[rng]
        known = sum(1 for p in phases if p != "unknown")
        label = "general" if rng == (-1, -1) else (f"ep {rng[0]}" if rng[0] == rng[1] else f"ep {rng[0]}-{rng[1]}")
        print(f"    {label}: {len(phases)} chunks, {known} with a known phase")

    print(f"[index] {len(all_chunks)} chunks tagged and upserted")
    return {"chunks": all_chunks}


# ---------------------------------------------------------------------------
# Phase resolution — derived from tagged chunk content, not a hardcoded formula
# ---------------------------------------------------------------------------

def resolve_phase(chunks: list[dict], episode: int) -> str:
    """Majority-vote the phase among chunks whose range contains this exact episode.
    If none contain it yet, fall back to the nearest range ending at or before it.
    Returns "unknown" if nothing usable is found, callers should treat "unknown"
    with the same caution as "Pods" (fail toward more spoiler protection, not less).

    Whitelists against PHASES rather than just excluding "unknown", since the tagging
    LLM has been observed emitting stray values like the literal string "null" instead
    of JSON null, which would otherwise pass through as a bogus "resolved" phase and
    crash PHASES.index() downstream.
    """
    candidates = [
        c for c in chunks
        if c["episode_start"] >= 0 and c["episode_start"] <= episode <= c["episode_end"] and c["phase"] in PHASES
    ]

    if not candidates:
        lower = [c for c in chunks if 0 <= c["episode_end"] <= episode and c["phase"] in PHASES]
        if lower:
            max_end = max(c["episode_end"] for c in lower)
            candidates = [c for c in lower if c["episode_end"] == max_end]

    if not candidates:
        return "unknown"

    return Counter(c["phase"] for c in candidates).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Node 3: retrieve — four targeted queries instead of one blended query
# ---------------------------------------------------------------------------

def pinecone_query(client: OpenAI, index, query_text: str, filter_dict: dict, top_k: int, namespace: str) -> list[dict]:
    embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text]).data[0].embedding
    result = index.query(vector=embedding, top_k=top_k, filter=filter_dict, include_metadata=True, namespace=namespace)
    return [match["metadata"] for match in result["matches"]]


def cap_per_source(metas: list[dict], max_per_source: int = 3) -> list[dict]:
    """Limit how many chunks from any single source can occupy a bucket's slots.
    Without this, one heavily-covered storyline, or one repetitive source, can
    crowd out every other storyline entirely. Preserves original ranking order,
    just skips over-quota items.
    """
    counts: dict[str, int] = {}
    capped = []
    for meta in metas:
        url = meta["source_url"]
        if counts.get(url, 0) >= max_per_source:
            continue
        counts[url] = counts.get(url, 0) + 1
        capped.append(meta)
    return capped


def node_retrieve(state: RecapState) -> dict:
    client = OpenAI()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX_NAME)

    edition, season, episode = state["edition"], state["season"], state["episode"]
    namespace = namespace_for(edition, season)
    base_filter = {"edition": {"$eq": edition}, "season": {"$eq": season}}
    general_filter = {**base_filter, "episode_start": {"$eq": -1}}
    up_to_cutoff_filter = {**base_filter, "episode_start": {"$gte": 0}, "episode_end": {"$lte": episode}}
    exact_episode_filter = {**base_filter, "episode_start": {"$lte": episode}, "episode_end": {"$gte": episode}}

    # Phase resolved first: drama_episodes' filter below needs to know which phases
    # are "allowed" (strictly before the current one) before it can query. Current
    # phase is excluded from "so far" content on purpose, that belongs in highlights.
    phase = resolve_phase(state["chunks"], episode)
    if phase == "unknown":
        print("Oooh noooo! I couldn't find the episode phase!")
        allowed_phases = []
    else:
        allowed_phases = PHASES[:PHASES.index(phase)]

    # 1. Bios: general/background content only, dedicated lane so they stop
    #    competing with drama/franchise content for the same slots.
    bios = pinecone_query(client, index, f"Love is Blind {edition} season {season} cast member names ages professions occupations", general_filter, top_k=20, namespace=namespace)

    # 2. Season-wide drama: filtered by PHASE rather than episode range. Phase signals
    #    (a wall/screen conversation, a shared apartment, a dress fitting) are observable
    #    content markers that don't depend on a source ever stating an episode number,
    #    which real recap narration often doesn't do explicitly.
    known_phase_filter = {**base_filter, "phase": {"$in": allowed_phases}}
    drama_general = pinecone_query(client, index, f"Love is Blind {edition} season {season} main storylines couples conflicts", general_filter, top_k=10, namespace=namespace)
    # Pulls ONLY from chunks with a confidently-resolved phase strictly before the
    # current one. No fallback readmitting "unknown phase" chunks by episode range,
    # that previously let current-episode content leak back into "so far" whenever
    # phase tagging failed on it (which was common for multi-episode-spanning sources).
    drama_episodes = pinecone_query(client, index, f"Love is Blind {edition} season {season} drama and relationships so far", known_phase_filter, top_k=25, namespace=namespace)
    drama_episodes = cap_per_source(drama_episodes, max_per_source=3)

    # 3. This episode specifically: exact match first (for highlights), falls back to
    #    up-to-cutoff if the exact episode has little tagged content yet. Kept
    #    episode-range based (not phase), highlights needs single-episode precision.
    this_episode = pinecone_query(client, index, f"Love is Blind {edition} season {season} episode {episode} events", exact_episode_filter, top_k=20, namespace=namespace)
    if len(cap_per_source(this_episode, max_per_source=3)) < 5:
        this_episode += pinecone_query(client, index, f"Love is Blind {edition} season {season} episode {episode} events", up_to_cutoff_filter, top_k=10, namespace=namespace)
    this_episode = cap_per_source(this_episode, max_per_source=3)

    # 4. Audience reaction: its own query, doesn't share slots with plot content.
    reaction = pinecone_query(client, index, f"Love is Blind {edition} season {season} episode {episode} fan reaction audience opinion", up_to_cutoff_filter, top_k=6, namespace=namespace)
    reaction += pinecone_query(client, index, f"Love is Blind {edition} season {season} fan reaction audience opinion", general_filter, top_k=3, namespace=namespace)

    seen_texts = set()

    def format_block(meta):
        if meta["text"] in seen_texts:
            return None
        seen_texts.add(meta["text"])
        if meta["episode_start"] < 0:
            ep_label = "general"
        elif meta["episode_start"] == meta["episode_end"]:
            ep_label = f"episode {meta['episode_start']}"
        else:
            ep_label = f"episodes {meta['episode_start']}-{meta['episode_end']}"
        return f"SOURCE: {meta['source_url']}\nTITLE: {meta['source_title']}\nEPISODE: {ep_label}\n{meta['text']}\n"

    def format_section(name, metas):
        blocks = [b for b in (format_block(m) for m in metas) if b]
        return f"=== {name} ===\n" + ("\n---\n".join(blocks) if blocks else "(nothing retrieved for this section)")

    context = "\n\n".join([
        format_section("PARTICIPANT BIOS", bios),
        format_section("PRIOR EPISODES: SPECIFIC EVENTS", drama_general + drama_episodes),
        format_section(f"EPISODE {episode} SPECIFIC EVENTS", this_episode),
        format_section("AUDIENCE REACTION", reaction),
    ])

    print(f"[retrieve] bios={len(bios)} drama={len(drama_general) + len(drama_episodes)} "
          f"episode={len(this_episode)} reaction={len(reaction)} | phase resolved: {phase}")

    def print_sources(name, metas):
        print(f"  [{name}]")
        for m in metas:
            rng = "general" if m["episode_start"] < 0 else f"{m['episode_start']}-{m['episode_end']}"
            print(f"    - {m['source_title']} (ep {rng})")

    print_sources("bios", bios)
    print_sources("drama", drama_general + drama_episodes)
    print_sources("this_episode", this_episode)
    print_sources("reaction", reaction)

    print("[debug] actual text content of ep 1-5 chunks (checking for thin/noise content):")
    seen_debug = set()
    for m in drama_general + drama_episodes:
        if m["episode_start"] == 1 and m["episode_end"] == 5 and m["source_url"] not in seen_debug:
            seen_debug.add(m["source_url"])
            print(f"  --- {m['source_title']} ({m['source_url']}) ---")
            print(f"  {m['text'][:600]}")
            print()

    return {"context": context, "phase": phase}


# ---------------------------------------------------------------------------
# Node 4: generate
# ---------------------------------------------------------------------------

def node_generate(state: RecapState) -> dict:
    client = OpenAI()
    edition, season, episode, phase = state["edition"], state["season"], state["episode"], state["phase"]

    pods_phase_rule = ""
    if phase in ("Pods", "unknown"):
        pods_phase_rule = f"""

CRITICAL PODS-PHASE RULE: episode {episode}'s phase is "{phase}", which means it is either
confirmed to be the Pods phase (before couples are living together) or could not be confirmed
as being past it. Treat this conservatively as Pods. Do not reveal which people end up
together, engaged, or paired as couples, even if your sources describe final pairings.
Describe only individual contestants and dynamics visible to a viewer who has watched up to
episode {episode} and no further.
For "participants": return exactly one entry:
{{"name": "Wait for it!", "age": null, "profession": "Full participant profiles arrive after the Reveal, no spoilers here!"}}
"""

    retry_note = ""
    if state.get("spoiler_issues"):
        issues = "; ".join(state["spoiler_issues"])
        retry_note = f"""

PREVIOUS ATTEMPT FAILED THE SPOILER CHECK. Specific issues found: {issues}
Regenerate the recap excluding these, using only the context provided below."""

    system_prompt = f"""You are a comic, dramatic soap-opera narrator writing a "previously on"
recap for the reality show Love Is Blind {edition}, Season {season}.

STRICT RULE: only use information about events up to and including episode {episode}.
Never mention or hint at anything that happens after episode {episode}, even if it
appears in the provided context. If the context discusses later episodes, ignore that part.
{pods_phase_rule}{retry_note}

GROUNDING RULES, follow these exactly:
- Every person's name must be copied character-for-character as it is spelled in the
  provided context. If a name is spelled differently across sources, use the spelling
  that appears most often. Never guess a spelling or invent a variant.
- Do not state any specific claim (a name, an event, a relationship detail) unless it
  appears explicitly in the provided context. If the context doesn't cover something,
  leave it out rather than filling the gap.
- If the context is thin or vague on a topic, keep that part of the recap general
  rather than inventing specifics to sound more dramatic.

Write like an excited friend texting another friend about the show, or a YouTube commenter
hyped about the drama, not like a formal narrator. Conversational, casual, genuinely excited.
Use occasional ALL CAPS for emphasis and exclamation points where it fits naturally. No emojis.

Avoid flowery or overwritten vocabulary. Do not use words like: whirlwind, swirling, tangled
web(s), rollercoaster, tapestry, saga, riveting, utterly, ablaze, or similar. Write like someone
would actually talk, not like a dramatic voiceover script.

The context below is organized into four labeled sections: PARTICIPANT BIOS, PRIOR EPISODES:
SPECIFIC EVENTS, EPISODE {episode} SPECIFIC EVENTS, and AUDIENCE REACTION. Use each section for
its matching field, described below.

Return ONLY valid JSON, no markdown fences, no preamble, with this exact shape:
{{
  "intro": "string, one short hype sentence that opens the whole recap, conversational",
  "main_drama": "string, the SPECIFIC events and facts from episodes before {episode}'s phase,
    naming names and exactly what they did. Not a vague season-level summary, precise, concrete
    beats from earlier episodes, told the same way highlights are: what happened, who, why it
    matters. Current-episode events belong in highlights only, never here.",
  "highlights": {{
    "episode_number": {episode},
    "episode_title": "string or null if not known from context",
    "moments": [
      {{"text": "string, one specific dramatic moment from THIS episode only, 1 to 2 sentences with real detail, conversational tone", "drama_rank": 1}}
    ]
  }},
  "audience_reaction": "string summarizing how fans reacted, or null if the AUDIENCE REACTION section has nothing usable",
  "participants": [
    {{
      "name": "string, exact spelling from sources",
      "age": "integer or null if not stated in sources",
      "profession": "string or null if not stated in sources"
    }}
  ],
  "sources": [
    {{"title": "string", "url": "string"}}
  ],
  "conclusion": "string, one short closing sentence that wraps up the recap, conversational, teases what's next without spoiling"
}}

For "main_drama", use ONLY the PRIOR EPISODES: SPECIFIC EVENTS section. Never use EPISODE
{episode} SPECIFIC EVENTS here, even if it seems relevant, that section is reserved for
highlights and may contain content from episodes after {episode}. This section already only
contains content confirmed to be safely before the current episode, so treat it as a source of
FACTS to report precisely, not a prompt to summarize loosely. Name specific people and specific
things they did across multiple earlier episodes if the context covers several, don't collapse
it into a generic "this season has been wild" summary.

For "highlights.moments", use ONLY the EPISODE {episode} SPECIFIC EVENTS section. Aim for 3 to 4
distinct dramatic moments, ranked by how dramatic or discussed they are, drama_rank 1 being the
most dramatic. Each moment should have enough detail to actually land (who, what, why it matters),
not a single clipped line. Do not blend them into one paragraph. A moment must describe what a
character did or what happened in the show itself, NOT how fans or viewers reacted to it, that
belongs in "audience_reaction" only, never in highlights.

For "audience_reaction", use the AUDIENCE REACTION section specifically.

For "participants", first identify which names actually appear in the PRIOR EPISODES: SPECIFIC EVENTS or
EPISODE {episode} SPECIFIC EVENTS sections, those are the real storyline participants for this
recap. Only include people from that set. Use the PARTICIPANT BIOS section only to fill in age
and profession for those specific names, never to introduce a name that doesn't otherwise appear
in the drama or episode content, even if that person has a bio available. Only include age or
profession if explicitly stated, use null otherwise, never guess. Do not add a personality
summary per participant, name/age/profession only.

Only include a source in "sources" if you actually drew a specific claim from it. A source
that appears in the context but wasn't used for any claim in this recap must not be listed.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n\n{state['context']}"},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    draft = json.loads(raw)

    print(f"[generate] attempt {state.get('attempts', 0) + 1}")
    return {"draft": draft, "attempts": state.get("attempts", 0) + 1}


# ---------------------------------------------------------------------------
# Node 5: spoiler_check
# ---------------------------------------------------------------------------

def node_spoiler_check(state: RecapState) -> dict:
    client = OpenAI()
    episode = state["episode"]
    draft_text = json.dumps(state["draft"])

    prompt = f"""You are a spoiler-check auditor. The user has only watched up to and including
episode {episode} of this Love Is Blind season. Review this draft recap (JSON) and check
whether it references anything that would only be known from episode {episode + 1} onward.

CRITICAL RULE: content that is explicitly part of episode {episode} itself is NEVER a spoiler,
no matter how dramatic, shocking, or consequential it sounds. A confession, a cheating reveal,
a breakup, a confrontation, any of these are completely fine to include if they are the actual
events of episode {episode}. Do not flag something just because it feels dramatic or because it
implies future consequences in a general sense, someone being upset or a relationship being in
trouble is not a spoiler on its own.

Only flag a claim if it states or clearly implies a SPECIFIC fact that is confirmed to happen
in episode {episode + 1} or later, for example: naming a couple's final wedding decision before
the Wedding phase has aired, revealing pod pairings before the reveal, or stating whether a
couple is still together after the reunion.

NOT A SPOILER: generic forward-looking closers that don't name a specific person, pairing, or
outcome, like "can't wait to see what happens next!" or "this drama is far from over!". These
are just narrator voice, not a claim about the future. Only flag language that ties to a
concrete, future-confirmed fact, never flag tone or excitement alone.

NOT A SPOILER, structural rule: a confession, reveal, admission, or confrontation described as
something that ALREADY HAPPENED ("came out that...", "admitted to...", "was revealed to...",
"it turns out...") is, by definition, a past or current event, not a future one. This is true
no matter how dramatic the reveal is or how much unresolved tension it implies. Do not flag a
reveal just because it's dramatic or because a reasonable viewer would wonder what happens next,
that wondering is the intended effect of a "previously on" recap, not evidence of a spoiler.
Only flag what comes AFTER such a reveal: whether the relationship survives it, how it gets
resolved, or any other consequence that would only be confirmed in a LATER episode than the one
the reveal itself belongs to.

Example of what is NOT a spoiler: "it came out that one contestant had been secretly seeing an
ex" — this describes a reveal itself, a normal recap-worthy plot event, not a spoiler.
Example of what IS a spoiler: "the couple ultimately splits up over it" or "she chooses to stay
with him at the altar" — these state outcomes or resolutions from phases/episodes that haven't
aired yet, not the reveal itself.

GROUNDING: below is the same sourced context that was used to write this draft. Each block is
labeled with the episode number or range it actually covers (or "general" if not episode-specific).
Use this to check your judgment on any claim you're unsure about: if a claim is corroborated by
at least one block labeled with an episode number at or before {episode}, treat it as confirmed
episode-{episode}-or-earlier content and do NOT flag it, even if that same claim also appears in
a block labeled with a wider range that extends past {episode} (recap sources often bundle
several episodes into one video/article, that doesn't make the underlying event itself a
spoiler). Only flag a claim if you cannot find it grounded in any block labeled at or before
{episode}, and it matches the future-consequence pattern described above.

SOURCE CONTEXT:
{state['context']}

Draft recap:
{draft_text}

For each issue found, identify exactly which field it's in, so a downstream repair step can
target just that field:
- "intro": the intro field
- "main_drama": the main_drama field
- "highlights_moment": one entry in highlights.moments (include its index, 0-based)
- "audience_reaction": the audience_reaction field
- "conclusion": the conclusion field
Do not flag "participants" or "sources", a name or a URL is never a spoiler on its own.

Return ONLY JSON:
{{"passed": true_or_false, "issues": [
  {{"field": "conclusion", "moment_index": null, "issue": "specific issue description"}}
]}}
If passed is true, issues should be an empty array. moment_index is only used when
field is "highlights_moment", null otherwise."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"passed": True, "issues": []}

    print(f"[spoiler_check] passed={result['passed']} issues={result.get('issues')}")
    return {"spoiler_passed": result["passed"], "spoiler_issues": result.get("issues", [])}

# ---------------------------------------------------------------------------
# Node: repair — targeted rewrite of only the flagged fields, not a full regenerate
# ---------------------------------------------------------------------------

# The checker is prompted to return exactly "highlights_moment", but LLM output on an
# exact-string field isn't guaranteed, "highlights.moments", "highlights", "moments" have
# all been observed. Normalize known variants rather than crashing on a KeyError.
_HIGHLIGHTS_FIELD_ALIASES = {"highlights_moment", "highlights.moments", "highlights", "moments"}
_VALID_TOP_LEVEL_FIELDS = {"intro", "main_drama", "audience_reaction", "conclusion"}


def _normalize_field(raw_field: str) -> str | None:
    if raw_field in _HIGHLIGHTS_FIELD_ALIASES:
        return "highlights_moment"
    if raw_field in _VALID_TOP_LEVEL_FIELDS:
        return raw_field
    return None


def _get_field_text(draft: dict, issue: dict) -> str:
    if issue["field"] == "highlights_moment":
        return draft["highlights"]["moments"][issue["moment_index"]]["text"]
    return draft[issue["field"]]


def _set_field_text(draft: dict, issue: dict, new_text: str) -> None:
    if issue["field"] == "highlights_moment":
        draft["highlights"]["moments"][issue["moment_index"]]["text"] = new_text
    else:
        draft[issue["field"]] = new_text


def node_repair(state: RecapState) -> dict:
    client = OpenAI()
    episode = state["episode"]
    draft = json.loads(json.dumps(state["draft"]))

    for issue in state["spoiler_issues"]:
        normalized = _normalize_field(issue.get("field", ""))
        if normalized is None:
            print(f"[repair] skipping unrecognized field '{issue.get('field')}', leaving as-is")
            continue
        issue = {**issue, "field": normalized}

        if normalized == "highlights_moment" and issue.get("moment_index") is None:
            print("[repair] skipping highlights_moment issue with no moment_index")
            continue

        current_text = _get_field_text(draft, issue)
        prompt = f"""You are revising one small piece of a "previously on" recap for Love Is Blind,
to remove a spoiler flagged by an editor. The user has only watched up to episode {episode}.

CURRENT TEXT:
{current_text}

FLAGGED ISSUE:
{issue['issue']}

Rewrite ONLY this text to remove the spoiler, keeping the same casual, excited, conversational
narrator voice as the original. Keep it roughly the same length. Do not introduce new claims,
if removing the spoiler leaves the text thin, keep it general rather than inventing detail.

Return ONLY the rewritten text, no quotes, no JSON, no preamble."""

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        new_text = response.choices[0].message.content.strip()
        _set_field_text(draft, issue, new_text)
        print(f"[repair] rewrote field '{issue['field']}'"
              + (f" (moment {issue['moment_index']})" if issue["field"] == "highlights_moment" else ""))

    return {"draft": draft, "attempts": state.get("attempts", 0) + 1}


# ---------------------------------------------------------------------------
# Node: mark_unverified — stamps the draft when max retries are hit without passing
# ---------------------------------------------------------------------------

def node_mark_unverified(state: RecapState) -> dict:
    draft = json.loads(json.dumps(state["draft"]))
    draft["_spoiler_unverified"] = True
    return {"draft": draft}


def route_after_spoiler_check(state: RecapState) -> str:
    if state["spoiler_passed"]:
        return "end_passed"
    if state["attempts"] > MAX_SPOILER_RETRIES:
        print("[spoiler_check] max retries reached, returning draft as-is (flagged unverified)")
        return "end_failed"
    return "retry"

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(RecapState)
    graph.add_node("fetch", node_fetch)
    graph.add_node("index", node_index)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("spoiler_check", node_spoiler_check)
    graph.add_node("repair", node_repair)
    graph.add_node("mark_unverified", node_mark_unverified)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "index")
    graph.add_edge("index", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "spoiler_check")
    graph.add_conditional_edges(
        "spoiler_check",
        route_after_spoiler_check,
        {"end_passed": END, "end_failed": "mark_unverified", "retry": "repair"},
    )
    graph.add_edge("repair", "spoiler_check")
    graph.add_edge("mark_unverified", END)

    return graph.compile()


def run_pipeline(edition: str, season: int, episode: int) -> dict:
    load_dotenv()
    app = build_graph()
    initial_state = {
        "edition": edition,
        "season": season,
        "episode": episode,
        "phase": "unknown",
        "sources": [],
        "chunks": [],
        "context": "",
        "draft": {},
        "spoiler_issues": [],
        "spoiler_passed": False,
        "attempts": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state["draft"]