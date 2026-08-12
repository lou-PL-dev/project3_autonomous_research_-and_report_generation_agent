"""Node 4: retrieve — four targeted Pinecone queries (bios/drama/this_episode/
reaction), phase-based drama filter (V1's proven design).
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pinecone import Pinecone

from config import EMBEDDING_MODEL, PHASES, PINECONE_INDEX_NAME
from indexing import resolve_phase
from research import namespace_for


def pinecone_query(client: OpenAI, index, query_text: str, filter_dict: dict, top_k: int, namespace: str) -> list[dict]:
    embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text]).data[0].embedding
    result = index.query(vector=embedding, top_k=top_k, filter=filter_dict, include_metadata=True, namespace=namespace)
    return [m["metadata"] for m in result["matches"]]


def cap_per_source(metas: list[dict], max_per_source: int = 3) -> list[dict]:
    counts: dict[str, int] = {}
    capped = []
    for meta in metas:
        url = meta["source_url"]
        if counts.get(url, 0) >= max_per_source:
            continue
        counts[url] = counts.get(url, 0) + 1
        capped.append(meta)
    return capped


def node_retrieve(state: dict) -> dict:
    client = OpenAI()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX_NAME)

    edition, season, episode = state["edition"], state["season"], state["episode"]
    namespace = namespace_for(edition, season)
    base_filter = {"edition": {"$eq": edition}, "season": {"$eq": season}}
    general_filter = {**base_filter, "episode_start": {"$eq": -1}}
    up_to_cutoff_filter = {**base_filter, "episode_start": {"$gte": 0}, "episode_end": {"$lte": episode}}
    exact_episode_filter = {**base_filter, "episode_start": {"$lte": episode}, "episode_end": {"$gte": episode}}

    phase = resolve_phase(state["chunks"], episode)
    # main_drama covers what happened BEFORE the current phase only, not including
    # it, highlights owns the current episode/phase specifically. If the current
    # phase is the season's first (Pods) or unresolved, there's no "before" yet.
    phase_idx = PHASES.index(phase) if phase in PHASES else 0
    strictly_before_phases = PHASES[:phase_idx]

    # Cast ground truth (season_indexes/*_cast.csv) guaranteed inclusion, not left
    # to compete on semantic similarity against Rotten Tomatoes/IMDb pages, that
    # competition is exactly why bios kept coming back empty all night.
    ground_truth_bios_filter = {**base_filter, "is_ground_truth": {"$eq": True}, "episode_start": {"$eq": -1}}
    unknown_phase_safe_filter = {**base_filter, "phase": {"$eq": "unknown"}, "episode_start": {"$gte": 0}, "episode_end": {"$lte": episode}}

    # All of the following queries are independent (different filters/text),
    # fire them concurrently instead of one round trip at a time, each is its
    # own embedding call + Pinecone query over the network.
    tasks = {
        "bios_semantic": (f"Love is Blind {edition} season {season} cast member names ages professions occupations", general_filter, 20),
        "ground_truth_bios": (f"Love is Blind {edition} season {season} cast", ground_truth_bios_filter, 20),
        "drama_general_raw": (f"Love is Blind {edition} season {season} main storylines couples conflicts", general_filter, 10),
        "drama_unknown_phase": (f"Love is Blind {edition} season {season} drama and relationships so far", unknown_phase_safe_filter, 10),
        "this_episode_exact": (f"Love is Blind {edition} season {season} episode {episode} events", exact_episode_filter, 20),
        "reaction_cutoff": (f"Love is Blind {edition} season {season} episode {episode} fan reaction audience opinion", up_to_cutoff_filter, 6),
        "reaction_general": (f"Love is Blind {edition} season {season} fan reaction audience opinion", general_filter, 3),
    }
    if strictly_before_phases:
        known_phase_filter = {**base_filter, "phase": {"$in": strictly_before_phases}}
        ground_truth_filter = {**base_filter, "is_ground_truth": {"$eq": True}, "phase": {"$in": strictly_before_phases}}
        tasks["drama_known_phase"] = (f"Love is Blind {edition} season {season} drama and relationships so far", known_phase_filter, 15)
        tasks["ground_truth_drama"] = (f"Love is Blind {edition} season {season}", ground_truth_filter, 20)

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            name: pool.submit(pinecone_query, client, index, query_text, filt, top_k, namespace)
            for name, (query_text, filt, top_k) in tasks.items()
        }
        results = {name: f.result() for name, f in futures.items()}

    bios = results["bios_semantic"] + results["ground_truth_bios"]

    # Deterministic backstop: a chunk tagged "general" (episode_start=-1) can still
    # explicitly name the current episode if the LLM tagger was too conservative
    # and defaulted to null instead of catching it. Reject those here rather than
    # trust the tag blindly, this is what let a Rotten Tomatoes season page leak
    # episode-6-specific content into main_drama despite phase filtering.
    current_episode_mention = re.compile(rf"\bepisode\s+{episode}\b", re.IGNORECASE)
    drama_general_raw = results["drama_general_raw"]
    drama_general = [m for m in drama_general_raw if not current_episode_mention.search(m["text"])]
    rejected_count = len(drama_general_raw) - len(drama_general)
    if rejected_count:
        print(f"[retrieve] rejected {rejected_count} 'general' chunks from main_drama, "
              f"explicitly mention episode {episode} despite being untagged")

    drama_episodes = results.get("drama_known_phase", []) + results["drama_unknown_phase"]
    ground_truth_drama = results.get("ground_truth_drama", [])
    drama_episodes = cap_per_source(drama_episodes, max_per_source=3) + ground_truth_drama

    this_episode = results["this_episode_exact"]
    if len(cap_per_source(this_episode, max_per_source=3)) < 5:
        this_episode += pinecone_query(client, index, f"Love is Blind {edition} season {season} episode {episode} events", up_to_cutoff_filter, top_k=10, namespace=namespace)
    this_episode = cap_per_source(this_episode, max_per_source=3)

    reaction = results["reaction_cutoff"] + results["reaction_general"]

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
        format_section("SEASON-WIDE DRAMA", drama_general + drama_episodes),
        format_section(f"EPISODE {episode} SPECIFIC EVENTS", this_episode),
        format_section("AUDIENCE REACTION", reaction),
    ])

    print(f"[retrieve] bios={len(bios)} drama={len(drama_general) + len(drama_episodes)} "
          f"episode={len(this_episode)} reaction={len(reaction)} | phase resolved: {phase}")

    all_context_metas = bios + drama_general + drama_episodes + this_episode + reaction
    unique_sources_in_context = {}
    for m in all_context_metas:
        unique_sources_in_context.setdefault(m["source_url"], m["source_title"])
    print(f"[retrieve] {len(unique_sources_in_context)} unique sources actually present in the context "
          f"(compare this to the final SOURCES list, the gap is what the model chose NOT to cite):")
    for url, title in unique_sources_in_context.items():
        print(f"    - {title}: {url}")

    return {"context": context, "phase": phase}
