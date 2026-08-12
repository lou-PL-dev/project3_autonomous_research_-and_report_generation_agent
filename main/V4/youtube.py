"""Real fan reaction: YouTube comment fetch (strictly range-gated to avoid
comment-implied spoilers) and LLM synthesis into structured reaction data.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from openai import OpenAI

from config import OPENAI_CHAT_MODEL

logger = logging.getLogger(__name__)


def extract_youtube_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def fetch_video_comments(video_id: str, api_key: str, max_results: int = 20) -> list[dict]:
    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/commentThreads",
            params={"part": "snippet", "videoId": video_id, "maxResults": max_results, "order": "relevance", "key": api_key},
            timeout=10,
        )
        data = response.json()
    except requests.RequestException:
        return []
    if response.status_code != 200:
        return []
    comments = []
    for item in data.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append({"text": snippet.get("textDisplay", ""), "likes": snippet.get("likeCount", 0)})
    return comments


# ---------------------------------------------------------------------------
# Node: fetch_youtube_comments — real fan reaction data, strictly range-gated.
# Confirmed via manual testing: comments rarely name a specific episode number,
# but a range video's comments can still reference later-episode content
# implicitly (e.g. wedding dress shopping details from episode 8 showing up in
# an "Episodes 6-9" video's comments with cutoff=6). Only videos whose ENTIRE
# tagged range is <= cutoff are safe, a partial-range video is not, even
# though "most" of it was already watched.
# ---------------------------------------------------------------------------

def node_fetch_youtube_comments(state: dict) -> dict:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.warning("no YOUTUBE_API_KEY found, skipping comment fetch")
        return {"youtube_comments": []}

    episode = state["episode"]

    eligible = []
    for source in state["selected_sources"]:
        url = source.get("url", "")
        if "youtube.com" not in url and "youtu.be" not in url:
            continue
        ep_start, ep_end = source.get("episode_start"), source.get("episode_end")
        # Entire range must be known and <= cutoff, a general (-1) or
        # partial-range-past-cutoff video is not safe, per confirmed test data.
        if ep_start is None or ep_end is None or ep_start < 0 or ep_end > episode:
            continue
        video_id = extract_youtube_video_id(url)
        if not video_id:
            continue
        eligible.append((source, video_id))

    def fetch_for(pair):
        source, video_id = pair
        comments = fetch_video_comments(video_id, api_key)
        for c in comments:
            c["source_title"] = source["title"]
            c["source_url"] = source["url"]
        logger.debug("fetched %d comments from ep %s-%s: %s",
                     len(comments), source.get("episode_start"), source.get("episode_end"), source["title"])
        return comments

    all_comments = []
    if eligible:
        # Each video's comment fetch is an independent HTTP request, run them
        # concurrently instead of one-by-one.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for comments in pool.map(fetch_for, eligible):
                all_comments.extend(comments)

    if not all_comments:
        logger.warning("no eligible (fully <= cutoff) YouTube sources with comments found")
    return {"youtube_comments": all_comments}


# ---------------------------------------------------------------------------
# Node: analyze_fan_reaction — synthesizes raw comments into structured form.
# Only runs on comments already gated to fully-within-cutoff videos, so this
# node's job is synthesis quality, not spoiler filtering, that's handled
# upstream. spoiler_check still audits the final result as a second layer.
# ---------------------------------------------------------------------------

def node_analyze_fan_reaction(state: dict) -> dict:
    comments = state.get("youtube_comments", [])
    if not comments:
        logger.info("no comments available, skipping analysis")
        return {"fan_reaction_analysis": None}

    client = OpenAI()
    edition, season, episode = state["edition"], state["season"], state["episode"]
    comments_text = "\n".join(f"({c['likes']} likes) {c['text'][:400]}" for c in comments)

    # Real, on-screen names for THIS watched range (episodes 1..cutoff), used to
    # stop the synthesis from naming someone a YouTube commenter mentioned who
    # isn't actually a cast member of this edition/season, comments are
    # unmoderated and can reference a different show, a different edition, or
    # an unrelated tangent while still being "explicitly tied" to that sentence.
    known_names = sorted({
        person["name"]
        for people in state.get("tmdb_participants", {}).values()
        for person in people
        if person.get("name")
    })
    known_names_block = ", ".join(known_names) if known_names else "(no known cast names available)"

    # Pull just the episode-specific section from retrieval so the synthesis
    # model knows what actually happened, without this it has no way to tell
    # a plot-relevant reaction from an off-topic tangent in the comments.
    context = state.get("context", "")
    episode_section_marker = f"EPISODE {episode} SPECIFIC EVENTS"
    episode_context = ""
    if episode_section_marker in context:
        start = context.index(episode_section_marker)
        line_end = context.index("\n", start)
        end = context.find("===", line_end)
        episode_context = context[line_end:end if end != -1 else len(context)][:3000]

    prompt = f"""You are the SAME comic, hyped soap-opera narrator writing the rest of this
"previously on" recap for Love Is Blind {edition} Season {season}, episode {episode}. This
section covers what fans are saying, but it should sound like you, not like a neutral
analyst writing a report. Write like an excited friend gossiping about what the comment
section is saying, conversational and casual, not clinical.

Here is what actually happened in this episode, use this to judge whether a comment is
actually about the show or just tangential chatter:
{episode_context or "(no episode-specific context available)"}

The real, confirmed cast/hosts for this edition and season, up through episode {episode}:
{known_names_block}

Comments:
{comments_text}

Return ONLY valid JSON, no markdown fences:
{{
  "overall_reception": "string, one or two hyped, conversational sentences on overall sentiment, tied to specific events when possible, written like you're catching a friend up, not summarizing a survey",
  "liked": ["string, phrased like you're excitedly telling a friend what people are loving"],
  "criticism": ["string, phrased the same conversational way, not a formal complaint list"],
  "themes": ["string, a recurring theme or debate, told with personality, not a bland label"],
  "sample_quotes": [{{"text": "string, a SHORT paraphrase or fragment (under 15 words), never a full comment reproduced verbatim", "context": "string, one short phrase on why this reaction stood out"}}]
}}

CRITICAL RULES:
- NEVER attribute an action or reaction to a named person unless that specific name is
  directly, explicitly tied to that specific action in the source material (the episode
  context or the comment itself). If a name appears in the comments in one context, do not
  reuse it for a different event or person, e.g. do not call Julia's mom "Kinga" just because
  "Kinga" appears somewhere else in the comments, they are different, unrelated people. When
  unsure exactly who a comment is referring to, describe the event without guessing a name
  rather than risk attaching the wrong one.
- NEVER name a specific person unless their name (or an obvious short form/nickname of it,
  e.g. "Ash" for "Ashley") appears in the CAST/HOSTS list above. YouTube comments are
  unmoderated and sometimes reference a different show, a different edition or season, or an
  unrelated tangent while still reading as "about" this video. If a comment names someone who
  isn't in the cast list, describe what the comment says without using that name, or exclude
  the comment entirely if it can't be rephrased without a name.
- Match the tone: excited, a little dramatic, conversational, ALL CAPS or an exclamation point
  here and there where it actually fits, the same voice as the rest of this recap. Not a
  formal report, not "viewers expressed mixed sentiments regarding..."
- Avoid flowery vocabulary (whirlwind, swirling, tangled web(s), rollercoaster, tapestry, saga,
  riveting, utterly, ablaze).
- Every entry in "liked", "criticism", "themes", and "sample_quotes" must reference a SPECIFIC
  named person, event, or moment, either from the episode context above or clearly stated in
  the comment itself. Do not write vague generic reactions like "some viewers enjoyed it while
  others found it awkward", that tells the reader nothing.
- If a comment goes off on a tangent unrelated to the show itself (e.g. general cultural trivia,
  a side conversation about geography or demographics not tied to anything that happened on
  screen), EXCLUDE it entirely. Do not surface a bare topic label like "Chicago" without
  explaining, in the same entry, exactly how it connects to something in the episode, if it
  can't be clearly connected, leave it out rather than mention it vaguely.
- Ground everything in what the comments actually say, do not invent reactions.
- 3 to 5 entries each for "liked", "criticism" (if present), "themes", and "sample_quotes". If
  fewer than 3 genuinely specific, well-grounded entries exist for a category, return fewer
  rather than padding with vague ones.
- If there's genuinely no criticism in the comments, return an empty list for "criticism".
- Do not reveal anything from after episode {episode}, only synthesize what's actually in
  the comments provided.
"""

    response = client.chat.completions.create(model=OPENAI_CHAT_MODEL, temperature=0.3, messages=[{"role": "user", "content": prompt}])
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("fan reaction analysis parse failed, skipping")
        return {"fan_reaction_analysis": None}

    logger.info("synthesized from %d comments: %d liked, %d criticism, %d themes, %d quotes",
                len(comments), len(analysis.get("liked", [])), len(analysis.get("criticism", [])),
                len(analysis.get("themes", [])), len(analysis.get("sample_quotes", [])))
    return {"fan_reaction_analysis": analysis}
