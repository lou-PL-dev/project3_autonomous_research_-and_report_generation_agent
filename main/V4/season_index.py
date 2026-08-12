"""Hand-verified season index / cast CSV loaders (season_indexes/episodes/*.csv
and season_indexes/cast/*.csv), used as ground-truth sources that bypass
web-search ranking entirely, plus a sanity check against OMDb's canonical titles.
"""

import csv
import os

from config import CAST_INDEX_DIR, EPISODE_INDEX_DIR


def load_cast_lookup(edition: str, season: int) -> dict[str, dict]:
    """Parse season_indexes/cast/*_cast.csv into a name -> {age, profession}
    lookup, used to enrich TMDB-supplied names (which have no age/profession
    of their own) without ever inventing a value not present in the
    hand-verified CSV.
    """
    filename = f"{edition.lower().replace(' ', '_')}_s{season}_cast.csv"
    path = os.path.join(CAST_INDEX_DIR, filename)
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


def load_season_index(edition: str, season: int, cutoff_episode: int) -> list[dict]:
    """Load a hand-verified episode/phase index if one exists for this edition/season.
    Returns synthetic "source" dicts, same shape as web-fetched ones, marked
    is_ground_truth=True. Only rows up to the cutoff are ever loaded, a row for
    episode 9 must not exist in memory at all when the user's cutoff is 6, that's
    a spoiler-safety floor independent of anything downstream.
    Returns an empty list if no index file exists for this edition/season yet.
    """
    filename = f"{edition.lower().replace(' ', '_')}_s{season}.csv"
    path = os.path.join(EPISODE_INDEX_DIR, filename)
    if not os.path.exists(path):
        return []

    sources = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = int(row["episode"])
            if ep > cutoff_episode:
                continue  # spoiler floor: never load rows past the user's cutoff
            content = f"{row['title_en']} ({row['title_original']}). Phase: {row['phase']}. {row['milestones']}"
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
    path = os.path.join(CAST_INDEX_DIR, filename)
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


def node_load_season_index(state: dict) -> dict:
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
