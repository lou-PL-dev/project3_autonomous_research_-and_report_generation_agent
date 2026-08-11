"""
V3: source-first RAG with a real ReAct planner agent, multi-edition structured
metadata (OMDb + TMDB), and TMDB-supplemented participants.

Core architectural change from V1: instead of running 4 fixed queries and chunking
everything fetched, a ReAct agent decides what searches are needed, episode ranges
are extracted deterministically from titles (regex, free, before any LLM call), and
only a temporally-relevant, hand-selected set of sources gets chunked/tagged/embedded.
This directly targets V1's confirmed failure mode: broad, high-content sources winning
retrieval over precise ones purely on volume, because temporal fit was never a
first-class signal.

Kept from V1 (proven, not broken): Pinecone RAG with range-based metadata filtering,
the four-section retrieve design (bios/drama/this_episode/reaction), the full
grounding/tone prompt, and the LLM spoiler-check with its false-positive-resistant
rules from tonight's fixes.

New in V3: OMDb supplies canonical episode titles (used both for source discovery and
as a title-matching fallback), and TMDB supplies real per-episode participant lists
(hosts + contestants), used to supplement the model's own participant extraction and
cross-referenced with the hand-verified cast CSV for age/profession. Both cover all 12
current Love Is Blind editions via season_indexes/imdb_ids.csv and tmdb_ids.csv.

Sources are still web search (Tavily) only, YouTube comment data (for real audience
reaction, not just whatever a recap video happens to mention) is planned next, not yet
integrated.

Usage (CLI):
    python recap.py --episode 6
"""

import csv
import json
import os
import re
import requests
from collections import Counter
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

DEFAULT_EDITION = "Poland"
DEFAULT_SEASON = 1

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
PINECONE_INDEX_NAME = "love-is-blind-recaps-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHARS_PER_SOURCE = 75000
MAX_SPOILER_RETRIES = 1
CATEGORY_BUDGETS = {"bios": 3, "highlights": 4, "drama": 4, "reaction": 3}

PHASES = ["Pods", "Honeymoon", "Moving In Together", "Wedding", "Reunion"]

GENERAL_DOMAINS = ["wikipedia.org", "themoviedb.org", "rottentomatoes.com", "imdb.com", "netflix.com"]
NOISE_DOMAINS = ["tiktok.com", "spotify.com"]

EDITION_ALIASES = {"poland": ["polska"]}


def is_general_domain(url: str) -> bool:
    return any(d in url for d in GENERAL_DOMAINS)


def is_noise_domain(url: str) -> bool:
    return any(d in url for d in NOISE_DOMAINS)


FRANCHISE_TERMS = ["love is blind", "casamento às cegas", "casamento as cegas"]  # franchise name across known localized titles


def matches_edition(edition: str, title: str, content: str) -> bool:
    """Requires BOTH the franchise name and the edition/country name. Edition names
    like "Poland", "France", "Germany" are also just country names, matching on
    that alone lets an entirely unrelated show slip through if it happens to be
    from the same country (confirmed: an HBO Max show called "True Love Extra
    (Poland)" passed the old edition-only check and landed in highlights with a
    perfect temporal-fit score).
    """
    terms = [edition.lower()] + EDITION_ALIASES.get(edition.lower(), [])
    haystack = (title + " " + content[:1500]).lower()
    has_edition = any(term in haystack for term in terms)
    has_franchise = any(term in haystack for term in FRANCHISE_TERMS)
    return has_edition and has_franchise


def namespace_for(edition: str, season: int) -> str:
    return f"v2_{edition.lower().replace(' ', '_')}_s{season}"


# ---------------------------------------------------------------------------
# Deterministic episode-range extraction from titles (no LLM, free, run first)
# ---------------------------------------------------------------------------

MAX_PLAUSIBLE_EPISODE = 20  # reality show seasons don't run this long; guards against
                             # matching a year (e.g. IMDb's "TV Episode 2026") as an episode number


def extract_episode_range_from_title(title: str) -> tuple[int | None, int | None]:
    """Pull an episode range straight out of a title/URL string via regex.
    This is the core new idea: real recap titles almost always state their
    coverage explicitly ("Episode 6", "Episodes 6-9", "S1E6"), so this is free,
    reliable signal available before any chunking or LLM tagging happens.
    Returns (None, None) if nothing matches, or if a match is implausible (e.g. a
    year mistaken for an episode number), callers must not guess further.
    """
    t = title.lower()

    range_patterns = [
        r"episodes?\s+(\d+)\s*[-–&]\s*(\d+)",
        r"eps?\.?\s+(\d+)\s*[-–]\s*(\d+)",
    ]
    for pat in range_patterns:
        m = re.search(pat, t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > MAX_PLAUSIBLE_EPISODE or b > MAX_PLAUSIBLE_EPISODE:
                continue
            return (min(a, b), max(a, b))

    single_patterns = [
        r"\bs\d+e(\d+)\b",
        r"episodes?\s+(\d+)\b",
        r"\bep\.?\s+(\d+)\b",
    ]
    for pat in single_patterns:
        m = re.search(pat, t)
        if m:
            n = int(m.group(1))
            if n > MAX_PLAUSIBLE_EPISODE:
                continue
            return (n, n)

    return (None, None)


def match_by_episode_title(title: str, episode_titles: dict[int, str]) -> tuple[int | None, int | None]:
    """Fallback for sources that name an episode by its actual title rather than
    its number (common: "You Can Never Count on Men Recap" never says "episode 6").
    Only usable once OMDb metadata has been fetched. Requires the canonical title
    to be a real substring match, not a guess.
    """
    if not episode_titles:
        return (None, None)
    t = title.lower()
    for ep_num, ep_title in episode_titles.items():
        if ep_title and len(ep_title) > 4 and ep_title.lower() in t:
            return (ep_num, ep_num)
    return (None, None)


def temporal_fit(ep_start: int | None, ep_end: int | None, cutoff: int) -> float:
    """Score how well a source's episode coverage fits the user's cutoff.
    Returns -1.0 for sources that must be excluded entirely (they only cover
    episodes after the cutoff, pure spoiler risk with no safe content).
    Returns 0.3 as a baseline for sources with no extractable range (general/bio
    content), still useful, just not temporally precise.
    """
    if ep_start is None:
        return 0.3
    if ep_start > cutoff:
        return -1.0  # entirely in the future, exclude
    if ep_start <= cutoff <= ep_end:
        span = ep_end - ep_start + 1
        return 1.0 if span == 1 else max(0.5, 1.0 - 0.05 * (span - 1))
    # entirely before cutoff: still useful "so far" context, closer gap scores higher
    gap = cutoff - ep_end
    return max(0.3, 0.9 - 0.05 * gap)


SEASON_INDEX_DIR = os.path.join(os.path.dirname(__file__), "season_indexes")

_imdb_ids_cache: dict[str, str] | None = None


def load_imdb_id(edition: str) -> str | None:
    """Look up the hardcoded, verified IMDb series ID for an edition from
    season_indexes/imdb_ids.csv (one ID per edition, IMDb/OMDb use a single
    series ID covering ALL seasons of an edition, selected via a separate
    Season parameter). Returns None if the edition isn't in the file yet.
    """
    global _imdb_ids_cache
    if _imdb_ids_cache is None:
        _imdb_ids_cache = {}
        path = os.path.join(SEASON_INDEX_DIR, "imdb_ids.csv")
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    _imdb_ids_cache[row["edition"].strip().lower()] = row["imdb_id"].strip()
    return _imdb_ids_cache.get(edition.lower())


_tmdb_ids_cache: dict[str, str] | None = None


def load_tmdb_id(edition: str) -> str | None:
    """Look up the hardcoded, verified TMDB series ID for an edition from
    season_indexes/tmdb_ids.csv. Returns None if not added yet.
    """
    global _tmdb_ids_cache
    if _tmdb_ids_cache is None:
        _tmdb_ids_cache = {}
        path = os.path.join(SEASON_INDEX_DIR, "tmdb_ids.csv")
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    _tmdb_ids_cache[row["edition"].strip().lower()] = row["tmdb_id"].strip()
    return _tmdb_ids_cache.get(edition.lower())


def load_cast_lookup(edition: str, season: int) -> dict[str, dict]:
    """Parse season_indexes/*_cast.csv into a name -> {age, profession} lookup,
    used to enrich TMDB-supplied names (which have no age/profession of their
    own) without ever inventing a value not present in the hand-verified CSV.
    """
    filename = f"{edition.lower().replace(' ', '_')}_s{season}_cast.csv"
    path = os.path.join(SEASON_INDEX_DIR, filename)
    if not os.path.exists(path):
        return {}
    lookup = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["name"].strip().lower()] = {
                "age": row["age"].strip() or None,
                "profession": row["profession"].strip() or None,
            }
    return lookup


def fetch_tmdb_episode_participants(tmdb_id: str, season: int, episode: int, api_key: str) -> list[dict]:
    """One episode's real on-screen participants (cast = hosts, guest_stars =
    contestants) from TMDB's episode-credits endpoint. Confirmed via manual
    testing to carry real, actively-maintained data for this show.
    """
    url = f"https://api.themoviedb.org/3/tv/{tmdb_id}/season/{season}/episode/{episode}/credits"
    try:
        response = requests.get(url, params={"api_key": api_key}, timeout=10)
        data = response.json()
    except requests.RequestException:
        return []
    if response.status_code != 200:
        return []
    people = data.get("cast", []) + data.get("guest_stars", [])
    return [{"name": p.get("name", ""), "role": p.get("character", "")} for p in people if p.get("name")]


def load_season_index(edition: str, season: int, cutoff_episode: int) -> list[dict]:
    """Load a hand-verified episode/phase index if one exists for this edition/season.
    Returns synthetic "source" dicts, same shape as web-fetched ones, marked
    is_ground_truth=True. Only rows up to the cutoff are ever loaded, a row for
    episode 9 must not exist in memory at all when the user's cutoff is 6, that's
    a spoiler-safety floor independent of anything downstream.
    Returns an empty list if no index file exists for this edition/season yet.
    """
    filename = f"{edition.lower().replace(' ', '_')}_s{season}.csv"
    path = os.path.join(SEASON_INDEX_DIR, filename)
    if not os.path.exists(path):
        return []

    sources = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = int(row["episode"])
            if ep > cutoff_episode:
                continue  # spoiler floor: never load rows past the user's cutoff
            content = f"{row['title_en']} ({row['title_pl']}). Phase: {row['phase']}. {row['milestones']}"
            sources.append({
                "title": f"Season Index: Episode {ep} - {row['title_en']}",
                "url": f"internal://season-index/{edition.lower()}-s{season}/ep{ep}",
                "content": content,
                "episode_start": ep,
                "episode_end": ep,
                "ground_truth_phase": row["phase"],
                "is_ground_truth": True,
                "_raw_title_en": row["title_en"],
            })
    return sources


def load_cast_index(edition: str, season: int) -> list[dict]:
    """Load a hand-verified cast/host list (name, age, profession) if one exists.
    No episode cutoff applies, bios aren't spoiler-relevant. Returns a single
    synthetic source containing every entry, so it either fully appears or not,
    never gets split apart across the top-k competition retrieval used elsewhere.
    """
    filename = f"{edition.lower().replace(' ', '_')}_s{season}_cast.csv"
    path = os.path.join(SEASON_INDEX_DIR, filename)
    if not os.path.exists(path):
        return []

    lines = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            age = row["age"] if row["age"] else "age unknown"
            lines.append(f"{row['name']}: {age}, {row['profession']}")

    return [{
        "title": "Season Index: Cast & Hosts",
        "url": f"internal://season-index/{edition.lower()}-s{season}/cast",
        "content": "\n".join(lines),
        "episode_start": -1,
        "episode_end": -1,
        "ground_truth_phase": "unknown",
        "is_ground_truth": True,
    }]


class RecapState(TypedDict):
    edition: str
    season: int
    episode: int
    phase: str
    episode_titles: dict[int, str]
    tmdb_participants: dict[int, list[dict]]
    raw_sources: list[dict]
    ground_truth_sources: list[dict]
    selected_sources: list[dict]
    youtube_comments: list[dict]
    fan_reaction_analysis: dict | None
    chunks: list[dict]
    context: str
    draft: dict
    spoiler_issues: list[str]
    spoiler_passed: bool
    attempts: int


# ---------------------------------------------------------------------------
# Node 0: fetch_show_metadata — structured metadata / entity resolution, not a
# RAG source. Establishes the canonical episode structure (title <-> number)
# BEFORE any messy web source gets asked "which episode is this?". Messy
# sources only need to answer "what happened?" from here on.
# ---------------------------------------------------------------------------

def node_fetch_show_metadata(state: RecapState) -> dict:
    edition, season, episode = state["edition"], state["season"], state["episode"]

    # --- OMDb: canonical episode titles ---
    episode_titles = {}
    imdb_id = load_imdb_id(edition)
    omdb_key = os.getenv("OMDB_API_KEY") or os.getenv("OMBD_API_KEY")
    if not imdb_id:
        print(f"[omdb] no hardcoded IMDb ID for {edition} season {season}, skipping")
    elif not omdb_key:
        print("[omdb] no OMDB_API_KEY/OMBD_API_KEY found, skipping")
    else:
        try:
            response = requests.get("http://www.omdbapi.com/", params={"apikey": omdb_key, "i": imdb_id, "Season": season}, timeout=10)
            data = response.json()
            if data.get("Response") != "True":
                print(f"[omdb] lookup failed: {data.get('Error')}, continuing without it")
            else:
                for ep in data.get("Episodes", []):
                    try:
                        episode_titles[int(ep["Episode"])] = ep["Title"]
                    except (KeyError, ValueError):
                        continue
                print(f"[omdb] fetched {len(episode_titles)} canonical episode titles for {edition} season {season}")
                for ep_num, title in sorted(episode_titles.items()):
                    print(f"    - ep {ep_num}: {title}")
        except requests.RequestException as e:
            print(f"[omdb] request failed ({e}), continuing without it")

    # --- TMDB: real per-episode participants (hosts + contestants), used as a
    # SUPPLEMENT to whatever generation finds. Kept PER-EPISODE (not flattened
    # across the whole range), episode N's guest_stars are exactly the people
    # still active in the story at that point, confirmed directly: episode 6's
    # list is precisely the 5 couples still in play, nobody who dropped out
    # earlier. Flattening 1-N together would reintroduce everyone who was ever
    # on screen, which is not what a recap's current participant list should be.
    tmdb_participants: dict[int, list[dict]] = {}
    tmdb_id = load_tmdb_id(edition)
    tmdb_key = os.getenv("TMDB_API_KEY")
    if not tmdb_id:
        print(f"[tmdb] no hardcoded TMDB ID for {edition} season {season}, skipping")
    elif not tmdb_key:
        print("[tmdb] no TMDB_API_KEY found, skipping")
    else:
        for ep in range(1, episode + 1):
            tmdb_participants[ep] = fetch_tmdb_episode_participants(tmdb_id, season, ep, tmdb_key)
        current_ep_people = tmdb_participants.get(episode, [])
        print(f"[tmdb] fetched participants for episodes 1-{episode}, "
              f"episode {episode} itself has {len(current_ep_people)}:")
        for person in current_ep_people:
            print(f"    - {person['name']} ({person['role']})")

    return {"episode_titles": episode_titles, "tmdb_participants": tmdb_participants}


def check_season_index_mismatch(episode_titles: dict[int, str], season_index_rows: list[dict]) -> None:
    """Log (not fail on) any title mismatch between OMDb and the hand-verified
    season index, both should agree, but this is a sanity check, not a gate.
    """
    for row in season_index_rows:
        ep_num = row.get("episode_start")
        omdb_title = episode_titles.get(ep_num)
        csv_title = row.get("_raw_title_en")
        if omdb_title and csv_title and omdb_title.strip().lower() != csv_title.strip().lower():
            print(f"[omdb] MISMATCH (not fatal) for episode {ep_num}: "
                  f"OMDb says \"{omdb_title}\", season index says \"{csv_title}\"")


# ---------------------------------------------------------------------------
# Node 1: plan_and_search — a real ReAct agent decides what to search for
# ---------------------------------------------------------------------------

def node_plan_and_search(state: RecapState) -> dict:
    edition, season, episode = state["edition"], state["season"], state["episode"]
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    collected: list[dict] = []
    seen_urls = set()

    @tool
    def search_show_content(query: str) -> str:
        """Search the web for Love Is Blind recap, cast bio, or fan reaction content.
        Use a query that names the exact episode number when you need precise
        episode-specific content. Use a broader query for season-wide drama or
        cast/bio content. Returns a short summary of what was found, including each
        result's extracted episode coverage so you can judge whether you need to
        search again with a different or more specific query."""
        print(f"[plan] agent query: \"{query}\"")
        response = tavily.search(query=query, search_depth="advanced", max_results=10, include_raw_content=True)
        summaries = []
        for item in response.get("results", []):
            url = item.get("url")
            title = item.get("title", "")
            if not url:
                continue
            if url in seen_urls:
                print(f"    - SKIP (already seen): {title}")
                continue
            raw = item.get("raw_content") or item.get("content", "")
            if is_noise_domain(url):
                print(f"    - SKIP (noise domain): {title}")
                continue
            if not matches_edition(edition, title, raw):
                print(f"    - SKIP (wrong edition): {title}")
                continue
            seen_urls.add(url)
            ep_start, ep_end = extract_episode_range_from_title(title)
            if ep_start is None:
                ep_start, ep_end = match_by_episode_title(title, state["episode_titles"])
            ep_label = "general" if ep_start is None else (f"ep {ep_start}" if ep_start == ep_end else f"ep {ep_start}-{ep_end}")
            print(f"    - KEPT ({ep_label}): {title}")
            collected.append({
                "title": title,
                "url": url,
                "content": raw[:MAX_CHARS_PER_SOURCE],
                "episode_start": ep_start,
                "episode_end": ep_end,
                "is_ground_truth": False,
            })
            ep_label = "general/unknown" if ep_start is None else (
                f"episode {ep_start}" if ep_start == ep_end else f"episodes {ep_start}-{ep_end}"
            )
            summaries.append(f"- {title} ({ep_label}, {len(raw)} chars)")
        return "\n".join(summaries) if summaries else "No new results for this query."

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(model, [search_show_content])

    episode_title_hint = ""
    known_title = state["episode_titles"].get(episode)
    if known_title:
        episode_title_hint = f'\nEpisode {episode}\'s real title is "{known_title}". Try at least one search using this exact title (not just the episode number), some sources name episodes by title without ever stating the number.'

    planning_prompt = f"""You are a research planner for a Love Is Blind recap tool.
The user has watched Love Is Blind {edition} Season {season} up through episode {episode}.
{episode_title_hint}

You need to gather enough evidence to write a recap covering four things:
1. Cast/participant bios (names, ages, professions). Prefer a query like "meet the cast" or
   "meet the Love is Blind {edition} cast", entertainment-press articles with that phrasing
   typically list real ages and professions, reference pages (Rotten Tomatoes, IMDb, TMDB)
   usually only have names. Run at least one search specifically using "meet the cast" phrasing,
   don't rely only on a generic "cast bios" query.
2. The season's drama EARLY ON: specifically the Pods phase and early episodes (episode 1
   through roughly episode 4 or 5), before the couples left the pods. This is NOT the same
   as "so far", it means the START of the season specifically. You MUST run at least one
   search targeting this explicitly, for example "Love is Blind {edition} season {season}
   episode 1 pods" or "Love is Blind {edition} season {season} episodes 1-3 recap". Do not
   skip this even if other searches already turned up episode {episode}-adjacent content,
   early-episode content will not show up unless you search for it specifically.
3. Specific events from episode {episode} itself, and ONLY episode {episode}, not a range.
   You MUST run at least one search using a query naming ONLY episode {episode} with no range
   language at all (do not say "episodes X-Y"), for example "Love is Blind {edition} season
   {season} episode {episode} only" or "Love is Blind {edition} season {season} episode
   {episode} recap review". The word "only" helps exclude range-covering videos from the
   results. This is required even if other searches already
   found range-covering videos like "Episodes {episode}-9", those do not substitute for a
   single-episode-only source, which is significantly more useful for precise, spoiler-safe
   detail about this exact episode.
4. Audience/fan reaction to episode {episode}

Call the search tool with different, specific queries to cover each of these needs, including
the mandatory early-episode search in point 2 and the mandatory single-episode-only search in
point 3. Call the tool multiple times (aim for at least 6-8 calls) with varied phrasing until
you believe you have reasonable coverage of all needs above, INCLUDING early-episode content
and a true single-episode source, not just range-covering videos near episode {episode}. Do
not fabricate information, only rely on what the tool actually returns. When you believe you
have enough coverage, respond with a short confirmation summarizing what you found, and
explicitly confirm whether you found (a) early-episode/Pods-phase content and (b) a single-
episode-only source for episode {episode} specifically."""

    result = agent.invoke({"messages": [("user", planning_prompt)]})
    tool_calls = sum(1 for m in result["messages"] if getattr(m, "tool_calls", None))
    print(f"[plan] agent made {len(collected)} unique source discoveries across its search calls")
    final_message = result["messages"][-1].content if result["messages"] else ""
    print(f"[plan] agent's final summary: {final_message[:300]}")

    return {"raw_sources": collected}


# ---------------------------------------------------------------------------
# Node 2: rank_and_select — deterministic temporal-fit ranking, source-first
# ---------------------------------------------------------------------------

def node_load_season_index(state: RecapState) -> dict:
    episode_ground_truth = load_season_index(state["edition"], state["season"], state["episode"])
    if episode_ground_truth:
        print(f"[season_index] loaded {len(episode_ground_truth)} hand-verified episode entries (up to episode {state['episode']})")
        check_season_index_mismatch(state["episode_titles"], episode_ground_truth)
    else:
        print(f"[season_index] no episode index file found for {state['edition']} season {state['season']}, skipping")

    cast_ground_truth = load_cast_index(state["edition"], state["season"])
    if cast_ground_truth:
        print(f"[season_index] loaded hand-verified cast list")
    else:
        print(f"[season_index] no cast index file found for {state['edition']} season {state['season']}, skipping")

    return {"ground_truth_sources": episode_ground_truth + cast_ground_truth}


# ---------------------------------------------------------------------------
# Node 2: rank_and_select — per-need budgeted selection, not one shared
# competition. Different sections need fundamentally different sources: bios
# needs general/reference content (irrelevant to episode fit), highlights needs
# single-episode precision, drama needs broad multi-episode coverage, reaction
# needs fan/audience content specifically. A single ranking by temporal fit
# guaranteed general/reaction content would always lose to episode-matched
# recap content, that's what caused bios=0 across multiple runs. Each category
# now gets its own budget, filled independently, never competing with another.
# Ground-truth (season index) sources bypass all of this, unconditional.
# ---------------------------------------------------------------------------

def categorize_source(source: dict, episode: int) -> str:
    title_lower = source["title"].lower()
    if any(kw in title_lower for kw in ["reaction", "react", "audience", "fan"]):
        return "reaction"
    if is_general_domain(source["url"]):
        return "bios"
    if source["episode_start"] is not None and source["episode_start"] == episode and source["episode_end"] == episode:
        return "highlights"
    return "drama"


def node_rank_and_select(state: RecapState) -> dict:
    episode = state["episode"]
    excluded_future = []
    by_category: dict[str, list[tuple[float, dict]]] = {"bios": [], "highlights": [], "drama": [], "reaction": []}

    for s in state["raw_sources"]:
        fit = temporal_fit(s["episode_start"], s["episode_end"], episode)
        if fit < 0:
            excluded_future.append(s)
            continue
        category = categorize_source(s, episode)
        by_category[category].append((fit, s))

    # Bios' fit score is always a flat 0.3 (episode timing is irrelevant to it),
    # so rank by content length instead, a proxy for a richer bio article versus
    # a thin stub page.
    by_category["bios"].sort(key=lambda pair: -len(pair[1]["content"]))
    for cat in ("highlights", "reaction"):
        by_category[cat].sort(key=lambda pair: -pair[0])
    # Drama needs BREADTH of coverage, not proximity to the cutoff episode, a
    # source titled "Episodes 1-5" is more valuable here than "Episodes 6-7",
    # even though the latter scores higher on temporal_fit. Rank by span instead.
    by_category["drama"].sort(key=lambda pair: -(
        (pair[1]["episode_end"] - pair[1]["episode_start"]) if pair[1]["episode_start"] is not None else 0
    ))

    web_selected = []
    print(f"[rank] {len(excluded_future)} sources excluded, entirely past the cutoff")
    print(f"[rank] {len(state['ground_truth_sources'])} ground-truth sources included unconditionally")
    for category, budget in CATEGORY_BUDGETS.items():
        chosen = by_category[category][:budget]
        print(f"[rank] {category}: selected {len(chosen)}/{len(by_category[category])} candidates (budget {budget})")
        for fit, s in chosen:
            print(f"    - fit={fit:.2f} {s['title']}")
        web_selected.extend(s for _, s in chosen)

    seen_urls = set()
    deduped = []
    for s in web_selected:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            deduped.append(s)

    selected = state["ground_truth_sources"] + deduped
    print(f"[rank] total selected: {len(selected)} ({len(deduped)} web across 4 categories + {len(state['ground_truth_sources'])} ground truth)")

    return {"selected_sources": selected}


# ---------------------------------------------------------------------------
# Node 3: index — chunk, tag phase (+ episode range only when title didn't give it), embed
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Node: fetch_youtube_comments — real fan reaction data, strictly range-gated.
# Confirmed via manual testing: comments rarely name a specific episode number,
# but a range video's comments can still reference later-episode content
# implicitly (e.g. wedding dress shopping details from episode 8 showing up in
# an "Episodes 6-9" video's comments with cutoff=6). Only videos whose ENTIRE
# tagged range is <= cutoff are safe, a partial-range video is not, even
# though "most" of it was already watched.
# ---------------------------------------------------------------------------

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


def node_fetch_youtube_comments(state: RecapState) -> dict:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("[youtube] no YOUTUBE_API_KEY found, skipping comment fetch")
        return {"youtube_comments": []}

    episode = state["episode"]
    all_comments = []
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
        comments = fetch_video_comments(video_id, api_key)
        for c in comments:
            c["source_title"] = source["title"]
            c["source_url"] = url
        all_comments.extend(comments)
        print(f"[youtube] fetched {len(comments)} comments from ep {ep_start}-{ep_end}: {source['title']}")

    if not all_comments:
        print("[youtube] no eligible (fully <= cutoff) YouTube sources with comments found")
    return {"youtube_comments": all_comments}


def node_index(state: RecapState) -> dict:
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

    TAG_BATCH_SIZE = 30
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

    UPSERT_BATCH_SIZE = 200
    for i in range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE):
        batch = vectors_to_upsert[i:i + UPSERT_BATCH_SIZE]
        if batch:
            index.upsert(vectors=batch, namespace=namespace)

    print(f"[index] {len(all_chunks)} chunks tagged and upserted")
    return {"chunks": all_chunks}


# ---------------------------------------------------------------------------
# Node: analyze_fan_reaction — synthesizes raw comments into structured form.
# Only runs on comments already gated to fully-within-cutoff videos, so this
# node's job is synthesis quality, not spoiler filtering, that's handled
# upstream. spoiler_check still audits the final result as a second layer.
# ---------------------------------------------------------------------------

def node_analyze_fan_reaction(state: RecapState) -> dict:
    comments = state.get("youtube_comments", [])
    if not comments:
        print("[fan_reaction] no comments available, skipping analysis")
        return {"fan_reaction_analysis": None}

    client = OpenAI()
    edition, season, episode = state["edition"], state["season"], state["episode"]
    comments_text = "\n".join(f"({c['likes']} likes) {c['text'][:400]}" for c in comments)

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

    response = client.chat.completions.create(model="gpt-4o", temperature=0.3, messages=[{"role": "user", "content": prompt}])
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        print("[fan_reaction] analysis parse failed, skipping")
        return {"fan_reaction_analysis": None}

    print(f"[fan_reaction] synthesized from {len(comments)} comments: "
          f"{len(analysis.get('liked', []))} liked, {len(analysis.get('criticism', []))} criticism, "
          f"{len(analysis.get('themes', []))} themes, {len(analysis.get('sample_quotes', []))} quotes")
    return {"fan_reaction_analysis": analysis}


# ---------------------------------------------------------------------------
# Phase resolution
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Node 4: retrieve — four targeted queries, phase-based drama filter (V1's proven design)
# ---------------------------------------------------------------------------

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

    phase = resolve_phase(state["chunks"], episode)
    # main_drama covers what happened BEFORE the current phase only, not including
    # it, highlights owns the current episode/phase specifically. If the current
    # phase is the season's first (Pods) or unresolved, there's no "before" yet.
    phase_idx = PHASES.index(phase) if phase in PHASES else 0
    strictly_before_phases = PHASES[:phase_idx]

    bios_semantic = pinecone_query(client, index, f"Love is Blind {edition} season {season} cast member names ages professions occupations", general_filter, top_k=20, namespace=namespace)
    # Cast ground truth (season_indexes/*_cast.csv) guaranteed inclusion, not left
    # to compete on semantic similarity against Rotten Tomatoes/IMDb pages, that
    # competition is exactly why bios kept coming back empty all night.
    ground_truth_bios_filter = {**base_filter, "is_ground_truth": {"$eq": True}, "episode_start": {"$eq": -1}}
    ground_truth_bios = pinecone_query(client, index, f"Love is Blind {edition} season {season} cast", ground_truth_bios_filter, top_k=20, namespace=namespace)
    bios = bios_semantic + ground_truth_bios

    drama_general_raw = pinecone_query(client, index, f"Love is Blind {edition} season {season} main storylines couples conflicts", general_filter, top_k=10, namespace=namespace)
    # Deterministic backstop: a chunk tagged "general" (episode_start=-1) can still
    # explicitly name the current episode if the LLM tagger was too conservative
    # and defaulted to null instead of catching it. Reject those here rather than
    # trust the tag blindly, this is what let a Rotten Tomatoes season page leak
    # episode-6-specific content into main_drama despite phase filtering.
    current_episode_mention = re.compile(rf"\bepisode\s+{episode}\b", re.IGNORECASE)
    drama_general = [m for m in drama_general_raw if not current_episode_mention.search(m["text"])]
    rejected_count = len(drama_general_raw) - len(drama_general)
    if rejected_count:
        print(f"[retrieve] rejected {rejected_count} 'general' chunks from main_drama, "
              f"explicitly mention episode {episode} despite being untagged")

    drama_episodes = []
    if strictly_before_phases:
        known_phase_filter = {**base_filter, "phase": {"$in": strictly_before_phases}}
        drama_episodes = pinecone_query(client, index, f"Love is Blind {edition} season {season} drama and relationships so far", known_phase_filter, top_k=15, namespace=namespace)
    # Unknown-phase chunks are a safety-net fallback only, phase can't tell us
    # anything about them, so episode range is the only signal available, kept
    # strictly under the cutoff (not just "before current phase") since that's
    # the only guarantee we actually have for them.
    unknown_phase_safe_filter = {**base_filter, "phase": {"$eq": "unknown"}, "episode_start": {"$gte": 0}, "episode_end": {"$lte": episode}}
    drama_episodes += pinecone_query(client, index, f"Love is Blind {edition} season {season} drama and relationships so far", unknown_phase_safe_filter, top_k=10, namespace=namespace)
    # Ground-truth chunks are guaranteed inclusion, not left to compete on semantic
    # similarity against much larger web transcripts. Uses the same strict
    # phase-before-current filter as web content, since ground truth has real,
    # accurate phase data and shouldn't fall back to the weaker episode-number proxy.
    ground_truth_drama = []
    if strictly_before_phases:
        ground_truth_filter = {**base_filter, "is_ground_truth": {"$eq": True}, "phase": {"$in": strictly_before_phases}}
        ground_truth_drama = pinecone_query(client, index, f"Love is Blind {edition} season {season}", ground_truth_filter, top_k=20, namespace=namespace)
    drama_episodes = cap_per_source(drama_episodes, max_per_source=3) + ground_truth_drama

    this_episode = pinecone_query(client, index, f"Love is Blind {edition} season {season} episode {episode} events", exact_episode_filter, top_k=20, namespace=namespace)
    if len(cap_per_source(this_episode, max_per_source=3)) < 5:
        this_episode += pinecone_query(client, index, f"Love is Blind {edition} season {season} episode {episode} events", up_to_cutoff_filter, top_k=10, namespace=namespace)
    this_episode = cap_per_source(this_episode, max_per_source=3)

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


# ---------------------------------------------------------------------------
# Node 5: generate
# ---------------------------------------------------------------------------

def node_generate(state: RecapState) -> dict:
    client = OpenAI()
    edition, season, episode, phase = state["edition"], state["season"], state["episode"], state["phase"]

    pods_phase_rule = ""
    if phase in ("Pods", "unknown"):
        pods_phase_rule = f"""

CRITICAL PODS-PHASE RULE: episode {episode}'s phase is "{phase}". Treat this conservatively
as Pods. Do not reveal which people end up together, engaged, or paired as couples, even if
sources describe final pairings. For "participants": return exactly one entry:
{{"name": "Wait for it!", "age": null, "profession": "Full participant profiles arrive after the Reveal, no spoilers here!"}}
"""

    retry_note = ""
    if state.get("spoiler_issues"):
        issues = "; ".join(state["spoiler_issues"])
        retry_note = f"""

PREVIOUS ATTEMPT FAILED THE SPOILER CHECK. Specific issues found: {issues}
Regenerate excluding these, using only the context provided below."""

    system_prompt = f"""You are a comic, dramatic soap-opera narrator writing a "previously on"
recap for the reality show Love Is Blind {edition}, Season {season}.

STRICT RULE: only use information about events up to and including episode {episode}.
Never mention or hint at anything past episode {episode}, even if it appears in the context.
{pods_phase_rule}{retry_note}

GROUNDING RULES, follow these exactly:
- Every name must be copied character-for-character as spelled in the context. If spelled
  differently across sources, use whichever spelling appears most often. Never guess.
- Do not state any specific claim unless it appears explicitly in the context.
- If the context is thin on a topic, keep that part general rather than inventing specifics.

Write like an excited friend texting another friend about the show, or a YouTube commenter
hyped about the drama, not a formal narrator. Conversational, casual, genuinely excited.
Occasional ALL CAPS and exclamation points where natural. No emojis. Avoid flowery vocabulary
(whirlwind, swirling, tangled web(s), rollercoaster, tapestry, saga, riveting, utterly, ablaze).

The context is organized into four labeled sections: PARTICIPANT BIOS, SEASON-WIDE DRAMA,
EPISODE {episode} SPECIFIC EVENTS, and AUDIENCE REACTION. Use each for its matching field.

Return ONLY valid JSON, no markdown fences, no preamble, with this exact shape:
{{
  "intro": "string, one short hype sentence",
  "main_drama": "string, everything that's happened THIS SEASON SO FAR up to {episode}, naming names",
  "highlights": {{
    "episode_number": {episode},
    "episode_title": "string or null",
    "moments": [{{"text": "string, 1-2 sentences with real detail", "drama_rank": 1}}]
  }},
  "audience_reaction": "string or null if nothing usable",
  "participants": [{{"name": "string", "age": "int or null", "profession": "string or null"}}],
  "sources": [{{"title": "string", "url": "string"}}],
  "conclusion": "string, one short closing sentence, teases what's next without spoiling"
}}

For "main_drama", use ONLY the SEASON-WIDE DRAMA section. Do NOT pull from EPISODE {episode}
SPECIFIC EVENTS, that section is reserved for "highlights" only. main_drama covers what
happened BEFORE episode {episode}, not episode {episode} itself.
For "highlights.moments", use ONLY the EPISODE {episode} SPECIFIC EVENTS section, 3-4 moments,
ranked by drama_rank (1 = most dramatic). A moment describes what happened, not how fans reacted.
For "audience_reaction", use the AUDIENCE REACTION section specifically.
For "participants", only include people who appear in the drama/episode content, use the
PARTICIPANT BIOS section only to fill in age/profession for those specific names, never to
introduce a name that doesn't otherwise appear. Name/age/profession only, no personality summary.

For "sources": this is not optional and one citation is almost never enough. After writing the
recap, go back through EACH of the four context sections (PARTICIPANT BIOS, SEASON-WIDE DRAMA,
EPISODE {episode} SPECIFIC EVENTS, AUDIENCE REACTION) one at a time and check whether anything
from that section ended up in your recap, if it did, that source belongs in the list. A recap
drawing on multiple sections should almost always cite multiple sources, one per section is a
reasonable floor, not a ceiling. Only exclude a source if you reviewed it and genuinely used
nothing from it at all.
"""

    response = client.chat.completions.create(
        model="gpt-4o", temperature=0.4,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"CONTEXT:\n\n{state['context']}"}],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    draft = json.loads(raw)
    # Ground-truth (season index) sources are real content the model may draw on,
    # but internal:// isn't a real, clickable URL, never show it to the user even
    # if the model cited it. Deterministic, not left to prompt compliance.
    draft["sources"] = [s for s in draft.get("sources", []) if not s.get("url", "").startswith("internal://")]

    # Structured, real YouTube-comment-based fan reaction overrides generate's own
    # simple string synthesis when available. The simple string stays as the
    # fallback for when no eligible (fully <= cutoff) YouTube video was found.
    if state.get("fan_reaction_analysis"):
        draft["audience_reaction"] = state["fan_reaction_analysis"]

    # Supplement participants with real TMDB names for THIS episode specifically
    # (not the whole 1-N range, confirmed that flattens into every contestant
    # who ever appeared, not who's actually still in the story). A match
    # ENRICHES the existing entry (fuller name, cast-CSV age/profession) rather
    # than just being skipped, otherwise a vague/wrong model-generated entry
    # (e.g. bare "Filip" with the wrong profession) survives untouched even
    # when TMDB had the correct disambiguated name sitting right there.
    # Age/profession only ever come from the cast CSV, never TMDB's role field,
    # "Self - Contestant" is not real profession data.
    # Skipped entirely during the Pods-phase placeholder, adding/editing real
    # names there would defeat the spoiler protection that placeholder exists for.
    if phase not in ("Pods", "unknown"):
        cast_lookup = load_cast_lookup(edition, season)
        current_ep_tmdb = state.get("tmdb_participants", {}).get(episode, [])
        participants = draft.setdefault("participants", [])

        def find_match_index(tmdb_name: str) -> int | None:
            tmdb_lower = tmdb_name.lower()
            tmdb_tokens = set(tmdb_lower.split())
            for i, p in enumerate(participants):
                existing = p["name"].lower()
                if tmdb_lower in existing or existing in tmdb_lower:
                    return i
                # First-name-token overlap catches nickname cases (e.g. "Kamil
                # Uno" vs TMDB's "Kamil Michał Osiak"), approximate but safer
                # than missing an obvious same-person match entirely.
                if tmdb_tokens & set(existing.split()):
                    return i
            return None

        for person in current_ep_tmdb:
            tmdb_name = person["name"]
            info = cast_lookup.get(tmdb_name.lower(), {})
            match_idx = find_match_index(tmdb_name)

            if match_idx is not None:
                existing = participants[match_idx]
                name_upgraded = len(tmdb_name.split()) > len(existing["name"].split())
                if name_upgraded:
                    existing["name"] = tmdb_name
                # Once the identity is confirmed/disambiguated via TMDB, cast CSV
                # data is authoritative for THAT specific person, it overrides
                # rather than just fills gaps: the model's old age/profession may
                # have been attached to the wrong, ambiguous identity entirely
                # (e.g. "Filip" guessed as an Engineer, when the real Filip in
                # this episode, Filip Lenz, is a Flight Attendant per the CSV).
                # If the CSV has no entry for this exact person, leave whatever
                # the model already had rather than erasing it.
                if info.get("age"):
                    existing["age"] = info["age"]
                if info.get("profession"):
                    existing["profession"] = info["profession"]
            else:
                participants.append({
                    "name": tmdb_name,
                    "age": info.get("age"),
                    "profession": info.get("profession"),
                })

    print(f"[generate] attempt {state.get('attempts', 0) + 1}")
    return {"draft": draft, "attempts": state.get("attempts", 0) + 1}


# ---------------------------------------------------------------------------
# Node 6: spoiler_check
# ---------------------------------------------------------------------------

def node_spoiler_check(state: RecapState) -> dict:
    client = OpenAI()
    episode = state["episode"]
    draft_text = json.dumps(state["draft"])

    prompt = f"""You are a spoiler-check auditor. The user has only watched up to and including
episode {episode}. Review this draft recap (JSON) and check whether it references anything
that would only be known from episode {episode + 1} onward.

CRITICAL RULE: content explicitly part of episode {episode} itself is NEVER a spoiler, no
matter how dramatic. A confession, a cheating reveal, a breakup, a confrontation are all fine
if they are the actual events of episode {episode}. Do not flag something just because it
sounds dramatic or implies future consequences in a general sense.

Only flag a claim that states or clearly implies a SPECIFIC fact confirmed to happen in
episode {episode + 1} or later, e.g. naming a wedding outcome before the Wedding phase, or
revealing pod pairings before the reveal.

CONCLUSION FIELD RULE: the "conclusion" field is deliberately written as a vague, generic
teaser ("can't wait to see what happens next", "as they prepare for what's ahead"). This is
intentional and REQUIRED by design, not a leak. Only flag the conclusion if it states a
SPECIFIC fact about a future episode (a name, an outcome, an event), not for containing
forward-looking phrasing in general.

Example of what is NOT a spoiler: "X admitted to cheating on Y with Z in episode {episode}."
Example of what is NOT a spoiler: "Can't wait to see how these relationships unfold next!"
Example of what IS a spoiler: "X and Y ultimately divorce" or "at the wedding, X says no."

Draft recap:
{draft_text}

Return ONLY JSON: {{"passed": true_or_false, "issues": ["specific issue 1", ...]}}
If passed is true, issues should be an empty array."""

    response = client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[{"role": "user", "content": prompt}])
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"passed": True, "issues": []}

    print(f"[spoiler_check] passed={result['passed']} issues={result.get('issues')}")
    return {"spoiler_passed": result["passed"], "spoiler_issues": result.get("issues", [])}


def route_after_spoiler_check(state: RecapState) -> str:
    if state["spoiler_passed"]:
        return "end"
    if state["attempts"] > MAX_SPOILER_RETRIES:
        print("[spoiler_check] max retries reached, returning draft as-is")
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(RecapState)
    graph.add_node("fetch_show_metadata", node_fetch_show_metadata)
    graph.add_node("plan_and_search", node_plan_and_search)
    graph.add_node("load_season_index", node_load_season_index)
    graph.add_node("rank_and_select", node_rank_and_select)
    graph.add_node("fetch_youtube_comments", node_fetch_youtube_comments)
    graph.add_node("index", node_index)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("analyze_fan_reaction", node_analyze_fan_reaction)
    graph.add_node("generate", node_generate)
    graph.add_node("spoiler_check", node_spoiler_check)

    graph.set_entry_point("fetch_show_metadata")
    graph.add_edge("fetch_show_metadata", "plan_and_search")
    graph.add_edge("plan_and_search", "load_season_index")
    graph.add_edge("load_season_index", "rank_and_select")
    graph.add_edge("rank_and_select", "fetch_youtube_comments")
    graph.add_edge("fetch_youtube_comments", "index")
    graph.add_edge("index", "retrieve")
    graph.add_edge("retrieve", "analyze_fan_reaction")
    graph.add_edge("analyze_fan_reaction", "generate")
    graph.add_edge("generate", "spoiler_check")
    graph.add_conditional_edges("spoiler_check", route_after_spoiler_check, {"end": END, "retry": "generate"})

    return graph.compile()


def run_pipeline(edition: str, season: int, episode: int) -> dict:
    load_dotenv()
    app = build_graph()
    initial_state = {
        "edition": edition, "season": season, "episode": episode, "phase": "unknown",
        "episode_titles": {}, "tmdb_participants": {},
        "raw_sources": [], "ground_truth_sources": [], "selected_sources": [],
        "youtube_comments": [], "fan_reaction_analysis": None,
        "chunks": [], "context": "",
        "draft": {}, "spoiler_issues": [], "spoiler_passed": False, "attempts": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state["draft"]