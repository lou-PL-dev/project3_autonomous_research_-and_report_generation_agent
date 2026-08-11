"""
Fast-iteration test harness for the PARTICIPANTS section only.

Reuses whatever is already indexed in Pinecone from your last full `recap.py` run,
skips plan_and_search, rank_and_select, and index entirely. Only re-runs bios
retrieval + a participants-only generation call, a few seconds instead of minutes.

WARNING: this only works if you have NOT run the full pipeline (recap.py or app.py)
since the run whose participants you want to test, node_index clears the Pinecone
namespace at the start of every full run. If you rerun the full pipeline, you need
to re-run this script fresh afterward too.

Usage:
    python test_participants.py --episode 6
"""

import argparse
import json

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from pipeline import (
    DEFAULT_EDITION, DEFAULT_SEASON, PINECONE_INDEX_NAME,
    namespace_for, pinecone_query,
)


def generate_participants_only(client: OpenAI, edition: str, season: int, context: str) -> dict:
    """Isolated participants-only prompt, edit this freely to test changes fast."""
    system_prompt = f"""You are extracting cast participant information for a Love Is Blind
{edition} Season {season} recap.

Below is retrieved bio/cast content. Extract every participant the content gives enough
information about.

GROUNDING RULES:
- Names must be copied character-for-character as spelled in the content. If a name is
  spelled differently across sources, use whichever spelling appears most often.
- Only include age or profession if explicitly stated in the content, use null otherwise.
  Never guess or infer.
- Name/age/profession only, no personality summary.

Return ONLY valid JSON, no markdown fences:
{{"participants": [{{"name": "string", "age": "int or null", "profession": "string or null"}}]}}
"""
    response = client.chat.completions.create(
        model="gpt-4o", temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n\n{context}"},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    return json.loads(raw)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--edition", type=str, default=DEFAULT_EDITION)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()

    client = OpenAI()
    pc = Pinecone()
    index = pc.Index(PINECONE_INDEX_NAME)
    namespace = namespace_for(args.edition, args.season)

    general_filter = {"edition": {"$eq": args.edition}, "season": {"$eq": args.season}, "episode_start": {"$eq": -1}}
    bios = pinecone_query(client, index, f"Love is Blind {args.edition} season {args.season} cast member names ages professions occupations", general_filter, top_k=20, namespace=namespace)

    print(f"[test] {len(bios)} bio chunks retrieved from existing namespace '{namespace}':")
    seen_urls = set()
    for m in bios:
        if m["source_url"] not in seen_urls:
            seen_urls.add(m["source_url"])
            print(f"    - {m['source_title']}: {m['source_url']}")

    context = "\n---\n".join(f"SOURCE: {m['source_url']}\n{m['text']}" for m in bios)
    result = generate_participants_only(client, args.edition, args.season, context)

    print("\n=== PARTICIPANTS ===")
    for p in result["participants"]:
        age = p["age"] if p["age"] else "age unknown"
        profession = p["profession"] if p["profession"] else "profession unknown"
        print(f"{p['name']} ({age}, {profession})")


if __name__ == "__main__":
    main()
