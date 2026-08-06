"""
MVP: generate a spoiler-bounded, narrator-voiced recap for
Love Is Blind Poland, Season 1, up to a given episode.

Usage:
    python MVP/summary.py --episode 6
"""

import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

# Hardcoded for the MVP. Edition/season selection is a later version's job.
EDITION = "Poland"
SEASON = 1


def fetch_sources(episode: int) -> list[dict]:
    """Run a few targeted Tavily searches and return raw results (title, url, content)."""
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    queries = [
        (f"Love is Blind Poland season {SEASON} cast participants ages professions", 8),
        (f"Love is Blind Poland season {SEASON} episode {episode} recap", 5),
        (f"Love is Blind Poland season {SEASON} drama so far", 5),
        (f"Love is Blind Poland season {SEASON} episode {episode} fan reaction discussion", 5),
    ]

    results = []
    seen_urls = set()

    for query, max_results in queries:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=True,
        )
        for item in response.get("results", []):
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": url,
                        "content": item.get("raw_content") or item.get("content", ""),
                    }
                )

    return results


def build_context(sources: list[dict]) -> str:
    """Flatten fetched sources into a single text block for the prompt."""
    blocks = []
    for source in sources:
        blocks.append(f"SOURCE: {source['url']}\nTITLE: {source['title']}\n{source['content']}\n")
    return "\n---\n".join(blocks)


def log_sources(sources: list[dict], context: str, episode: int) -> None:
    """Print a short preview of each fetched source, and save the full context to a log file.

    This lets you check, after a run, whether a claim in the output (a name, a plot
    beat) actually appears in what was fetched, or whether the model invented it.
    """
    print(f"--- Fetched {len(sources)} sources ---")
    for source in sources:
        preview = source["content"][:200].replace("\n", " ")
        print(f"[{source['url']}]\n{source['title']}\n{preview}...\n")

    log_path = f"debug_context_episode_{episode}.txt"
    with open(log_path, "w") as f:
        f.write(context)
    print(f"Full fetched content saved to {log_path}\n")


def generate_recap(episode: int, context: str) -> dict:
    """Single LLM call: generate the structured, spoiler-bounded recap."""
    client = OpenAI()

    system_prompt = f"""You are a comic, dramatic soap-opera narrator writing a "previously on"
recap for the reality show Love Is Blind {EDITION}, Season {SEASON}.

STRICT RULE: only use information about events up to and including episode {episode}.
Never mention or hint at anything that happens after episode {episode}, even if it
appears in the provided sources. If a source discusses later episodes, ignore that part.

GROUNDING RULES, follow these exactly:
- Every person's name must be copied character-for-character as it is spelled in the
  provided sources. If a name is spelled differently across sources, use the spelling
  that appears most often. Never guess a spelling or invent a variant.
- Do not state any specific claim (a name, an event, a relationship detail) unless it
  appears explicitly in the provided sources. If the sources don't cover something,
  leave it out rather than filling the gap.
- If the sources are thin or vague on a topic, keep that part of the recap general
  rather than inventing specifics to sound more dramatic.

Write like an excited friend texting another friend about the show, or a YouTube commenter
hyped about the drama, not like a formal narrator. Conversational, casual, genuinely excited.
Use occasional ALL CAPS for emphasis and exclamation points where it fits naturally. No emojis.

Avoid flowery or overwritten vocabulary. Do not use words like: whirlwind, swirling, tangled
web(s), rollercoaster, tapestry, saga, riveting, utterly, ablaze, or similar. Write like someone
would actually talk, not like a dramatic voiceover script.

Return ONLY valid JSON, no markdown fences, no preamble, with this exact shape:
{{
  "intro": "string, one short hype sentence that opens the whole recap, conversational",
  "main_drama": "string, everything that's happened THIS SEASON SO FAR, across all episodes up
    to {episode}, naming names and what they did. This is the season-wide picture, not a
    retelling of the most recent episode, that belongs in highlights instead.",
  "highlights": {{
    "episode_number": {episode},
    "episode_title": "string or null if not known from sources",
    "moments": [
      {{"text": "string, one specific dramatic moment from THIS episode only, 1 to 2 sentences with real detail, conversational tone", "drama_rank": 1}}
    ]
  }},
  "audience_reaction": "string summarizing how fans reacted, or null if sources have nothing on this",
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

For "main_drama", pull together the season-wide arc, multiple couples and storylines across
multiple episodes if the sources cover that, not just whatever happened most recently.

For "highlights.moments", aim for 3 to 4 distinct dramatic moments from this specific episode,
ranked by how dramatic or discussed they are, drama_rank 1 being the most dramatic. Each moment
should have enough detail to actually land (who, what, why it matters), not a single clipped
line. Do not blend them into one paragraph.

For "participants", include every participant the sources give enough information about.
Only include age or profession if a source explicitly states it, use null otherwise, never guess.
Do not add a personality summary per participant, name/age/profession only.

Only include a source in "sources" if you actually drew a specific claim from it. A source
that was fetched but not used for any claim in this recap must not be listed.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"SOURCES:\n\n{context}"},
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Defensive: strip accidental markdown fences if the model adds them anyway.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1)

    return json.loads(raw)


def print_recap(recap: dict) -> None:
    """Print the recap to terminal in the agreed layout."""
    print(f"=== Previously On: Love Is Blind {EDITION}, Season {SEASON} ===\n")

    print(recap["intro"] + "\n")

    print("MAIN DRAMA SO FAR")
    print(recap["main_drama"] + "\n")

    ep_num = recap["highlights"]["episode_number"]
    ep_title = recap["highlights"]["episode_title"]
    header = f"HIGHLIGHTS OF THE LAST EPISODE (Episode {ep_num}"
    header += f": {ep_title})" if ep_title else ")"
    print(header)
    moments = sorted(recap["highlights"]["moments"], key=lambda m: m["drama_rank"])
    for moment in moments:
        print(f"- {moment['text']}")
    print()

    print("AUDIENCE REACTION")
    reaction = recap["audience_reaction"]
    print(reaction if reaction else "No fan reaction found in sources for this run.")
    print()

    print("PARTICIPANTS")
    for person in recap["participants"]:
        age = person["age"] if person["age"] else "age unknown"
        profession = person["profession"] if person["profession"] else "profession unknown"
        print(f"{person['name']} ({age}, {profession})")
    print()

    print(recap["conclusion"])
    print()

    print("SOURCES")
    for source in recap["sources"]:
        print(f"- {source['title']}: {source['url']}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate a Love Is Blind Poland S1 recap.")
    parser.add_argument("--episode", type=int, required=True, help="Last episode watched.")
    args = parser.parse_args()

    sources = fetch_sources(args.episode)
    context = build_context(sources)
    log_sources(sources, context, args.episode)
    recap = generate_recap(args.episode, context)
    print_recap(recap)


if __name__ == "__main__":
    main()