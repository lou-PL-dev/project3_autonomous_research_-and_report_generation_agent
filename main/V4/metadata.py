"""OMDb + TMDB fetch: canonical episode titles and real per-episode participants.
Establishes the canonical episode structure (title <-> number) BEFORE any messy
web source gets asked "which episode is this?".
"""

import csv
import os

import requests

from config import SEASON_INDEX_DIR

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


def node_fetch_show_metadata(state: dict) -> dict:
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
