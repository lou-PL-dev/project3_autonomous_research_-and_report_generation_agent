"""Chunking, LLM tagging (episode range + phase), Pinecone embedding/upsert,
and phase resolution from indexed chunks.
"""

import json
import os
from collections import Counter

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    PHASES,
    PINECONE_INDEX_NAME,
    TAG_BATCH_SIZE,
    UPSERT_BATCH_SIZE,
)
from research import namespace_for


def split_into_chunks(text: str) -> list[str]:
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
    """Batched LLM tagging, same design proven in V1: explicit source/chunk IDs
    (not array position) so a malformed or missing response item only defaults
    that one chunk rather than corrupting the batch.
    """
    if not items:
        return {}

    numbered = "\n\n".join(
        f"[src={item['source_index']} chunk={item['chunk_index']}] (from: {item['source_title']})\n{item['text']}"
        for item in items
    )
    prompt = f"""Below are numbered text chunks from several sources about a Love Is Blind season.
For each chunk, determine:

- episode_start and episode_end: the range of episodes this chunk narrates events from.
  If it covers one episode, both are the same number. If it spans a range, set start to the
  lowest and end to the highest episode actually narrated. If not determinable, both null.
  Be conservative, a short caption or generic mention isn't enough, only tag what the chunk
  actually narrates. When in doubt, use null.

- phase: one of {", ".join(PHASES)}, or null. Judge by observable content:
  - Pods: conversations through a wall/screen, or the reveal moment itself.
  - Honeymoon: an engaged couple traveling together, typically abroad.
  - Moving In Together: shared apartment, meeting family or friends.
  - Wedding: dress/suit fittings, vows, the ceremony itself.
  - Reunion: a separate post-finale special, cast answering questions about now.
  A chunk covering a later episode range may still narrate a FLASHBACK to an earlier moment,
  tag the phase for what's actually described, not wherever the episode range would sit.
  If unclear, use null.

Chunks:
{numbered}

Return ONLY a JSON array, no markdown fences:
[{{"source_index": int, "chunk_index": int, "episode_start": int_or_null, "episode_end": int_or_null, "phase": "string_or_null"}}]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
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


def node_index(state: dict) -> dict:
    client = OpenAI()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    namespace = namespace_for(state["edition"], state["season"])

    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"[index] cleared namespace '{namespace}' before indexing")
    except Exception as e:
        print(f"[index] namespace clear skipped (likely didn't exist yet): {e}")

    source_chunks: list[list[str]] = []
    tag_queue = []
    for source_idx, source in enumerate(state["selected_sources"]):
        chunks = split_into_chunks(source["content"])
        source_chunks.append(chunks)
        if source.get("is_ground_truth"):
            continue  # episode range and phase are already known exactly, no LLM needed
        # Only queue for LLM tagging what the title regex couldn't already resolve
        # for episode range, but phase always needs real LLM judgment either way.
        for chunk_idx in range(len(chunks)):
            tag_queue.append({
                "source_index": source_idx, "chunk_index": chunk_idx,
                "source_title": source["title"], "text": chunks[chunk_idx],
            })

    tags_by_id: dict[tuple[int, int], dict] = {}
    for batch_start in range(0, len(tag_queue), TAG_BATCH_SIZE):
        batch = tag_queue[batch_start:batch_start + TAG_BATCH_SIZE]
        batch_tags = tag_chunks_batch(client, batch)
        tags_by_id.update(batch_tags)
    print(f"[index] tagged {len(tags_by_id)}/{len(tag_queue)} chunks across "
          f"{len(state['selected_sources'])} selected sources (source-first: not the full fetch pool)")

    all_chunks = []
    vectors_to_upsert = []
    for source_idx, source in enumerate(state["selected_sources"]):
        chunks = source_chunks[source_idx]
        if not chunks:
            continue
        embeddings = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
        title_start, title_end = source["episode_start"], source["episode_end"]

        for chunk_idx, chunk_text in enumerate(chunks):
            # The model occasionally emits the STRING "null" instead of JSON's null
            # keyword. A quoted "null" is truthy in Python, so `x or default` doesn't
            # catch it, normalize it to real None here, at the source, rather than
            # patching every downstream consumer separately.
            def clean(v):
                return None if (v is None or (isinstance(v, str) and v.strip().lower() == "null")) else v

            if source.get("is_ground_truth"):
                # Hand-verified: episode and phase are exact, no tagging needed.
                ep_start, ep_end = title_start, title_end
                phase_value = source["ground_truth_phase"]
            else:
                tag = tags_by_id.get((source_idx, chunk_idx), {})
                # Title-derived range wins when available (deterministic, free); LLM
                # tag is the fallback only for sources the title regex couldn't resolve.
                ep_start = title_start if title_start is not None else clean(tag.get("episode_start"))
                ep_end = title_end if title_end is not None else clean(tag.get("episode_end"))
                phase_value = clean(tag.get("phase")) or "unknown"

            chunk_id = f"{state['edition']}-{state['season']}-{source['url']}-{chunk_idx}".replace(" ", "_")[:512]
            metadata = {
                "edition": state["edition"], "season": state["season"],
                "episode_start": ep_start if ep_start is not None else -1,
                "episode_end": ep_end if ep_end is not None else -1,
                "phase": phase_value,
                "is_ground_truth": bool(source.get("is_ground_truth")),
                "source_url": source["url"], "source_title": source["title"], "text": chunk_text,
            }
            all_chunks.append(metadata)
            vectors_to_upsert.append({"id": chunk_id, "values": embeddings.data[chunk_idx].embedding, "metadata": metadata})

    for i in range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE):
        batch = vectors_to_upsert[i:i + UPSERT_BATCH_SIZE]
        if batch:
            index.upsert(vectors=batch, namespace=namespace)

    print(f"[index] {len(all_chunks)} chunks tagged and upserted")
    return {"chunks": all_chunks}


def resolve_phase(chunks: list[dict], episode: int) -> str:
    exact = [c for c in chunks if c["episode_start"] >= 0 and c["episode_start"] <= episode <= c["episode_end"] and c["phase"] in PHASES]

    # Ground-truth (season index) always wins over LLM-inferred votes when present,
    # it's hand-verified, no reason to let a web-tagged guess compete with it.
    ground_truth = [c for c in exact if c.get("is_ground_truth")]
    if ground_truth:
        return Counter(c["phase"] for c in ground_truth).most_common(1)[0][0]

    candidates = exact
    if not candidates:
        lower = [c for c in chunks if 0 <= c["episode_end"] <= episode and c["phase"] in PHASES]
        if lower:
            max_end = max(c["episode_end"] for c in lower)
            candidates = [c for c in lower if c["episode_end"] == max_end]
    if not candidates:
        return "unknown"
    return Counter(c["phase"] for c in candidates).most_common(1)[0][0]
