"""ReAct planner agent that decides what to search for, deterministic
episode-range extraction from titles, temporal-fit scoring, and the
per-category budgeted source ranking/selection.
"""

import logging
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

from config import (
    CATEGORY_BUDGETS,
    EDITION_ALIASES,
    FRANCHISE_TERMS,
    GENERAL_DOMAINS,
    MAX_CHARS_PER_SOURCE,
    MAX_PLAUSIBLE_EPISODE,
    NOISE_DOMAINS,
    OPENAI_MINI_MODEL,
)

logger = logging.getLogger(__name__)


def is_general_domain(url: str) -> bool:
    return any(d in url for d in GENERAL_DOMAINS)


def is_noise_domain(url: str) -> bool:
    return any(d in url for d in NOISE_DOMAINS)


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


# ---------------------------------------------------------------------------
# Node 1: plan_and_search — a real ReAct agent decides what to search for
# ---------------------------------------------------------------------------

def node_plan_and_search(state: dict) -> dict:
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
        logger.info('agent query: "%s"', query)
        response = tavily.search(query=query, search_depth="advanced", max_results=10, include_raw_content=True)
        summaries = []
        for item in response.get("results", []):
            url = item.get("url")
            title = item.get("title", "")
            if not url:
                continue
            if url in seen_urls:
                logger.debug("SKIP (already seen): %s", title)
                continue
            raw = item.get("raw_content") or item.get("content", "")
            if is_noise_domain(url):
                logger.debug("SKIP (noise domain): %s", title)
                continue
            if not matches_edition(edition, title, raw):
                logger.debug("SKIP (wrong edition): %s", title)
                continue
            seen_urls.add(url)
            ep_start, ep_end = extract_episode_range_from_title(title)
            if ep_start is None:
                ep_start, ep_end = match_by_episode_title(title, state["episode_titles"])
            ep_label = "general" if ep_start is None else (f"ep {ep_start}" if ep_start == ep_end else f"ep {ep_start}-{ep_end}")
            logger.debug("KEPT (%s): %s", ep_label, title)
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

    model = ChatOpenAI(model=OPENAI_MINI_MODEL, temperature=0)
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
    logger.info("agent made %d unique source discoveries across its search calls", len(collected))
    final_message = result["messages"][-1].content if result["messages"] else ""
    logger.debug("agent's final summary: %s", final_message[:300])

    return {"raw_sources": collected}


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


def node_rank_and_select(state: dict) -> dict:
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
    logger.info("%d sources excluded, entirely past the cutoff", len(excluded_future))
    logger.info("%d ground-truth sources included unconditionally", len(state["ground_truth_sources"]))
    for category, budget in CATEGORY_BUDGETS.items():
        chosen = by_category[category][:budget]
        logger.info("%s: selected %d/%d candidates (budget %d)", category, len(chosen), len(by_category[category]), budget)
        for fit, s in chosen:
            logger.debug("fit=%.2f %s", fit, s["title"])
        web_selected.extend(s for _, s in chosen)

    seen_urls = set()
    deduped = []
    for s in web_selected:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            deduped.append(s)

    selected = state["ground_truth_sources"] + deduped
    logger.info("total selected: %d (%d web across 4 categories + %d ground truth)",
                len(selected), len(deduped), len(state["ground_truth_sources"]))

    return {"selected_sources": selected}
